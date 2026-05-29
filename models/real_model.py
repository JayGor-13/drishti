"""Pretrained vision encoders and monkey-patched quantized LLM for actual VQA."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.cache import EventTokenCache
from models.router import TemporallyAwareRouter, RouterOutput
from models.moe_layer import MoEForwardStats, MoEForwardOutput

# Load HF transformers
try:
    from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    CLIPVisionModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class RealLoRALinear(nn.Module):
    """Wraps an existing Linear (or quantized Linear4bit) module with a LoRA adapter."""

    def __init__(self, base_linear: nn.Module, rank: int = 64, alpha: float = 128.0) -> None:
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha

        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad_(False)

        if rank > 0:
            # We determine in/out features dynamically from base module weight shape
            if hasattr(base_linear, "weight"):
                # bnb layers might have quantized weights, check shape or default attributes
                in_features = getattr(base_linear, "in_features", base_linear.weight.shape[-1])
                out_features = getattr(base_linear, "out_features", base_linear.weight.shape[0])
            else:
                in_features = base_linear.in_features
                out_features = base_linear.out_features

            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank if self.rank > 0 else 0.0

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if self.rank > 0:
            adapter = F.linear(F.linear(x, self.lora_a), self.lora_b)
            y = y + adapter * self.scaling
        return y


class RealSwiGLUExpert(nn.Module):
    """FFN expert that wraps the base projection modules with their own LoRA adapters."""

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        lora_rank: int = 64,
        lora_alpha: float = 128.0,
    ) -> None:
        super().__init__()
        self.gate_proj = RealLoRALinear(gate_proj, rank=lora_rank, alpha=lora_alpha)
        self.up_proj = RealLoRALinear(up_proj, rank=lora_rank, alpha=lora_alpha)
        self.down_proj = RealLoRALinear(down_proj, rank=lora_rank, alpha=lora_alpha)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def lora_b_vector(self) -> Tensor:
        matrices = [
            module.lora_b.flatten()
            for module in (self.gate_proj, self.up_proj, self.down_proj)
            if module.lora_b is not None
        ]
        if matrices:
            return torch.cat(matrices)
        return torch.zeros(1, device=self.up_proj.lora_a.device)


class RealMicroMoELayer(nn.Module):
    """MoE Layer that patches an existing MLP module with 8 LoRA-adapted SwiGLU experts."""

    def __init__(
        self,
        original_mlp: nn.Module,
        num_experts: int = 8,
        top_k: int = 2,
        history_window: int = 2,
        lora_rank: int = 64,
        lora_alpha: float = 128.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        # Determine LLM hidden size dynamically
        if hasattr(original_mlp.gate_proj, "weight"):
            hidden_dim = getattr(original_mlp.gate_proj, "in_features", original_mlp.gate_proj.weight.shape[-1])
        else:
            hidden_dim = original_mlp.gate_proj.in_features

        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.num_experts = num_experts

        self.router = TemporallyAwareRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            history_window=history_window,
            temperature=temperature,
        )

        # Experts share the original base weights but have independent LoRA parameters
        self.original_mlp = original_mlp
        self.experts = nn.ModuleList([
            RealSwiGLUExpert(
                original_mlp.gate_proj,
                original_mlp.up_proj,
                original_mlp.down_proj,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            for _ in range(num_experts)
        ])

    def _run_experts(self, flat_tokens: Tensor, flat_router: RouterOutput) -> Tensor:
        flat_topk_idx = flat_router.topk_indices.reshape(-1, self.top_k)
        flat_topk_scores = flat_router.topk_scores.reshape(-1, self.top_k)
        output = torch.zeros_like(flat_tokens)

        for expert_idx, expert in enumerate(self.experts):
            selected = flat_topk_idx == expert_idx
            if not selected.any():
                continue
            token_rows = selected.any(dim=-1)
            expert_input = flat_tokens[token_rows]
            expert_output = expert(expert_input)
            weights = (selected[token_rows].float() * flat_topk_scores[token_rows]).sum(dim=-1)
            output[token_rows] = output[token_rows] + expert_output * weights.unsqueeze(-1)

        return output

    def _slice_router(self, router: RouterOutput, frame_index: int) -> RouterOutput:
        return RouterOutput(
            logits=router.logits[:, frame_index : frame_index + 1],
            probs=router.probs[:, frame_index : frame_index + 1],
            topk_indices=router.topk_indices[:, frame_index : frame_index + 1],
            topk_scores=router.topk_scores[:, frame_index : frame_index + 1],
            context=router.context[:, frame_index : frame_index + 1],
            entropy=router.entropy[:, frame_index : frame_index + 1],
        )

    def forward(
        self,
        tokens: Tensor,
        motion_embeddings: Tensor | None = None,
        motion_confidence: Tensor | None = None,
        cache: EventTokenCache | None = None,
    ) -> MoEForwardOutput:
        # tokens: [B, T, S, H]
        batch, time, slots, hidden = tokens.shape
        router = self.router(tokens, motion_embeddings=motion_embeddings)
        ffn_outputs = torch.zeros_like(tokens)
        executed_tokens = 0
        cached_tokens = 0

        if cache is None or motion_confidence is None:
            flat_tokens = tokens.reshape(batch * time * slots, hidden)
            ffn_outputs = self._run_experts(flat_tokens, router).view_as(tokens)
            executed_tokens = batch * time * slots
            return MoEForwardOutput(
                hidden_states=tokens + ffn_outputs,
                router=router,
                stats=MoEForwardStats(
                    executed_tokens=executed_tokens,
                    cached_tokens=cached_tokens,
                    total_tokens=batch * time * slots,
                ),
            )

        cache.ensure_shape(batch, slots, hidden, tokens.device, tokens.dtype)
        for frame_idx in range(time):
            frame_tokens = tokens[:, frame_idx]
            confidence = motion_confidence[:, frame_idx]
            cached_mask = cache.readable_mask(confidence)
            compute_mask = ~cached_mask

            if cached_mask.any():
                cached_branch = cache.read(cached_mask)
                ffn_outputs[:, frame_idx] = torch.where(
                    cached_mask.unsqueeze(-1), cached_branch, ffn_outputs[:, frame_idx]
                )
                cached_tokens += int(cached_mask.sum().item())

            if compute_mask.any():
                frame_router = self._slice_router(router, frame_idx)
                flat_compute_mask = compute_mask.view(-1)
                active_tokens = frame_tokens.reshape(batch * slots, hidden)[flat_compute_mask]
                
                active_router = RouterOutput(
                    logits=frame_router.logits.reshape(batch * slots, -1)[flat_compute_mask],
                    probs=frame_router.probs.reshape(batch * slots, -1)[flat_compute_mask],
                    topk_indices=frame_router.topk_indices.reshape(batch * slots, -1)[flat_compute_mask],
                    topk_scores=frame_router.topk_scores.reshape(batch * slots, -1)[flat_compute_mask],
                    context=frame_router.context.reshape(batch * slots, -1)[flat_compute_mask],
                    entropy=frame_router.entropy.reshape(batch * slots)[flat_compute_mask],
                )
                
                active_output = self._run_experts(active_tokens, active_router)
                flat_branch = torch.zeros(batch * slots, hidden, device=tokens.device, dtype=tokens.dtype)
                flat_branch[flat_compute_mask] = active_output
                flat_branch = flat_branch.reshape(batch, slots, hidden)

                ffn_outputs[:, frame_idx] = torch.where(
                    compute_mask.unsqueeze(-1),
                    flat_branch,
                    ffn_outputs[:, frame_idx],
                )
                cache.write(flat_branch, compute_mask)
                executed_tokens += int(compute_mask.sum().item())

        return MoEForwardOutput(
            hidden_states=tokens + ffn_outputs,
            router=router,
            stats=MoEForwardStats(
                executed_tokens=executed_tokens,
                cached_tokens=cached_tokens,
                total_tokens=batch * time * slots,
            ),
        )


class RealTMoELLaVA(nn.Module):
    """End-to-End Actual model wrapping CLIP-Large, X3D-Tiny, and monkey-patched quantized LLM."""

    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        use_4bit: bool = True,
        moe_layers_to_patch: list[int] | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        lora_rank: int = 64,
        lora_alpha: float = 128.0,
        cache_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.cache_threshold = cache_threshold

        # 1. Load CLIP-Large
        print(f"Loading visual encoder: {clip_model_name}")
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name) if CLIPVisionModel else None
        self.clip_dim = 1024  # openai/clip-vit-large-patch14 hidden size

        # 2. Load Motion Encoder (X3D)
        print("Loading motion encoder: torchvision X3D-Tiny")
        try:
            import torchvision
            # torchvision.models.video.x3d_xs is the tiny variant of X3D
            self.motion_encoder = torchvision.models.video.x3d_xs(pretrained=True)
            # Remove output projection heads, only keep feature blocks
            self.motion_encoder.blocks[5] = nn.Identity()
            self.motion_dim = 2048  # Output channels of block 4 in X3D-XS
        except Exception as e:
            print(f"WARNING: torchvision X3D-Tiny load failed ({e}). Using proxy motion stem.")
            self.motion_encoder = None
            self.motion_dim = 256

        # 3. Load Base LLM
        print(f"Loading base LLM: {llm_model_name}")
        quantization_config = None
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("WARNING: bitsandbytes not installed. Defaulting to float16 LLM load.")

        if AutoModelForCausalLM:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto" if use_4bit else None,
            )
        else:
            self.llm = None

        self.llm_dim = getattr(self.llm.config, "hidden_size", 2048) if self.llm else 1024

        # 4. Projections to map CLIP and X3D feature space to LLM hidden dimension
        self.visual_proj = nn.Linear(self.clip_dim, self.llm_dim, bias=False)
        self.motion_proj = nn.Linear(self.motion_dim, self.llm_dim, bias=False)
        self.motion_conf_head = nn.Linear(self.llm_dim, 1)

        # 5. Monkey-patch chosen layers
        self.patched_layers = []
        if self.llm:
            total_layers = len(self.llm.model.layers)
            if moe_layers_to_patch is None:
                # Patch alternate layers in top half
                moe_layers_to_patch = list(range(total_layers // 2, total_layers, 2))

            print(f"Monkey-patching MoE layers into indices: {moe_layers_to_patch}")
            self.caches = nn.ModuleList()
            for idx in moe_layers_to_patch:
                original_mlp = self.llm.model.layers[idx].mlp
                patched_mlp = RealMicroMoELayer(
                    original_mlp,
                    num_experts=num_experts,
                    top_k=top_k,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
                self.llm.model.layers[idx].mlp = patched_mlp
                
                # Create a cache buffer for this layer
                cache = EventTokenCache(threshold=cache_threshold)
                self.caches.append(cache)
                self.patched_layers.append((idx, patched_mlp, cache))

            # Enable gradient checkpointing to save memory
            self.llm.gradient_checkpointing_enable()

    def reset_caches(self) -> None:
        for _, _, cache in self.patched_layers:
            cache.reset()

    def forward(
        self,
        frames: Tensor,
        input_ids: Tensor,
        labels: Tensor | None = None,
        reset_cache: bool = True,
    ) -> dict[str, Tensor | list]:
        if reset_cache:
            self.reset_caches()

        batch, time, channels, height, width = frames.shape

        # 1. Visual tokens from CLIP
        # Reshape to run CLIP flat over all frames
        flat_frames = frames.reshape(batch * time, channels, height, width)
        if self.clip:
            # CLIP ViT outputs patch sequence in output[0]
            clip_outputs = self.clip(flat_frames)[0]  # [B*T, seq_len, clip_dim]
            visual_tokens = self.visual_proj(clip_outputs)  # [B*T, seq_len, llm_dim]
            seq_len = visual_tokens.shape[1]
            visual_tokens = visual_tokens.reshape(batch, time, seq_len, self.llm_dim)
        else:
            visual_tokens = torch.zeros(batch, time, 16, self.llm_dim, device=frames.device)
            seq_len = 16

        # 2. Motion embeddings & confidence from X3D
        if self.motion_encoder:
            # X3D consumes video clips [B, C, T, H, W]
            x3d_input = frames.permute(0, 2, 1, 3, 4)
            # Intermediate features from conv blocks
            motion_features = self.motion_encoder(x3d_input)  # [B, motion_dim, T, H', W']
            # Pool to match visual patch sequence length
            pooled = F.adaptive_avg_pool3d(motion_features, (time, int(seq_len**0.5), int(seq_len**0.5)))
            pooled = pooled.flatten(3).transpose(2, 3)  # [B, motion_dim, patches, time] -> swap dims
            pooled = pooled.permute(0, 3, 2, 1)  # [B, time, patches, motion_dim]
            motion_embeddings = self.motion_proj(pooled)  # [B, time, patches, llm_dim]
        else:
            motion_embeddings = torch.zeros(batch, time, seq_len, self.llm_dim, device=frames.device)

        motion_confidence = torch.sigmoid(self.motion_conf_head(motion_embeddings)).squeeze(-1)

        # Store dynamic visual sequence attributes for pre-hooks access
        self.current_time = time
        self.current_seq_len = seq_len

        # 3. Inject visual & motion tokens into LLM context (Inputs embeds concatenation)
        if self.llm:
            text_embeds = self.llm.model.embed_tokens(input_ids)  # [B, S, H]
            visual_flat = visual_tokens.flatten(1, 2)  # [B, T*seq_len, H]
            inputs_embeds = torch.cat([visual_flat, text_embeds], dim=1)  # [B, T*seq_len + S, H]
            
            if labels is not None:
                v_labels = torch.full((batch, time * seq_len), -100, dtype=torch.long, device=labels.device)
                labels = torch.cat([v_labels, labels], dim=1)
        else:
            inputs_embeds = torch.zeros(batch, time * seq_len + input_ids.shape[1], self.llm_dim, device=frames.device)

        # We hook into the forward passes of the patched layers dynamically to supply motion conditioning
        hooks = []
        for idx, patched_mlp, cache in self.patched_layers:
            # Set the motion parameters dynamically on the MLP layer so they are accessible during the LLM loop
            patched_mlp.current_motion_embeddings = motion_embeddings
            patched_mlp.current_motion_confidence = motion_confidence
            patched_mlp.current_cache = cache

            # Re-intercept the layer's forward pass to reshape attention sequence to [B, T, S, H] for MoE routing
            def make_pre_hook(mlp_layer, c_cache, parent_model):
                def pre_hook(module, inputs):
                    x = inputs[0]  # Shape: [B, seq_len_total, H]
                    b, s, h = x.shape
                    
                    time_dim = parent_model.current_time
                    seq_len_dim = parent_model.current_seq_len
                    v_len = time_dim * seq_len_dim
                    
                    if s >= v_len:
                        x_visual = x[:, :v_len]
                        x_text = x[:, v_len:]
                        
                        x_visual_reshaped = x_visual.reshape(b, time_dim, seq_len_dim, h)
                        output_visual = module(
                            x_visual_reshaped,
                            motion_embeddings=module.current_motion_embeddings,
                            motion_confidence=module.current_motion_confidence,
                            cache=module.current_cache,
                        )
                        flat_visual = output_visual.hidden_states.reshape(b, v_len, h)
                        
                        if s > v_len:
                            output_text = module.original_mlp(x_text)
                            output = torch.cat([flat_visual, output_text], dim=1)
                        else:
                            output = flat_visual
                        return output
                    else:
                        # Fallback for next token autoregressive text-only generations
                        output = module.original_mlp(x)
                        return output
                return pre_hook
            
            # Register forward override hook
            hook = patched_mlp.register_forward_pre_hook(make_pre_hook(patched_mlp, cache, self))
            hooks.append(hook)

        try:
            # Run the Hugging Face LLM forward pass using inputs_embeds
            if self.llm:
                outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                loss = torch.tensor(0.0, device=frames.device)
                logits = torch.zeros(batch, time * seq_len + input_ids.shape[1], self.llm_dim, device=frames.device)
        finally:
            # Clean up hooks to prevent memory leaks or state overlap in future forward passes
            for hook in hooks:
                hook.remove()

        # Collect routing probabilities from the first patched block for diagnostics
        router_outputs = [layer.router for _, layer, _ in self.patched_layers]

        return {
            "loss": loss,
            "logits": logits,
            "motion_confidence": motion_confidence,
            "router_outputs": router_outputs,
        }
