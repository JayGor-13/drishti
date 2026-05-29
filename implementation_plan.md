# Engineering Design Document & Changes Plan: Micro-MoE SOTA Optimization

**Author:** Antigravity AI Coding Assistant  
**Status:** PROPOSED  
**Target Architecture:** Micro-MoE (T-MoE-LLaVA 2.0)  

---

## 1. Executive Summary & Design Goals
The goal of the **Micro-MoE** (T-MoE-LLaVA 2.0) project is to deliver extreme efficiency for video-language understanding on edge devices (low VRAM/compute environments). Our baseline implementation has passed smoke testing, but code review has revealed two critical architectural bottlenecks and the absence of an ablation pipeline. 

To challenge state-of-the-art (SOTA) models from major research organizations (Google, NVIDIA, Apple), this document outlines a rigorous changes plan. We address:
1. **Dynamic Expert Bypassing Correctness:** Transitioning from "execute-and-mask" to true token-level $O(1)$ expert execution.
2. **Multi-layer Regularization Integrity:** Ensuring that routing balance, temporal consistency (CFCR), and expert orthogonalization scale across all model layers.
3. **Rigorous Experimental Validation:** Establishing an automated, scientific ablation script to verify all claims on synthetic video-language streams.

---

## 2. Technical Context & High-Level Block Diagram

The following Mermaid diagram shows the target flow of multimodal token sequences through our layers, illustrating the event-based token bypass mechanism:

```mermaid
graph TD
    subgraph Input Processing
        VF[Video Frames f_1..f_T] --> VE[Visual Encoder]
        VF --> ME[Motion Encoder]
        VE --> VT[Visual Tokens]
        ME --> MC[Motion Confidence m_i^t]
        ME --> ME_emb[Motion Embeddings]
    end

    subgraph Block Execution (Layer l)
        VT --> Attn[Multi-Head Self-Attention]
        Attn --> H_attn[Attention Hidden States]
        
        MC --> CacheBranch{m_i^t < epsilon ?}
        
        CacheBranch -- Yes (Static) --> ReadCache[Retrieve from EventTokenCache]
        CacheBranch -- No (Dynamic) --> FilterTokens[Filter: Active Tokens Only]
        
        FilterTokens --> TA_Router[Temporally-Aware Router]
        TA_Router --> SelExperts[Top-2 SwiGLU Experts]
        SelExperts --> ScatterTokens[Scatter Back to Full Frame Grid]
        
        ReadCache --> ConcatOutput[Combine Static & Dynamic Outputs]
        ScatterTokens --> ConcatOutput
        
        ConcatOutput --> WriteCache[Update EventTokenCache]
        H_attn --> AddResidual[Residual Connection]
        ConcatOutput --> AddResidual
    end
    
    AddResidual --> NextBlock[Next Layer / Output Head]
```

---

## 3. Discrepancy Breakdown & Code Specifications

### 3.1. Issue 1: Inefficient Token Cache Bypass
#### Why the change is required:
In the current implementation of `MicroMoELayer.forward`, when any token within a frame requires computation (`compute_mask.any() == True`), the FFN experts are executed over all tokens in the frame:
```python
flat_branch = self._run_experts(
    frame_tokens.reshape(batch * slots, hidden),
    frame_router,
).reshape(batch, slots, hidden)
```
This performs expert computations (matrix multiplications) on static tokens whose outputs are already available in the cache, wasting substantial FLOPs. It fails the core thesis of the paper: achieving $O(1)$ token bypass.

#### How to implement the change:
We will filter both the tokens and the router output metrics before calling `_run_experts` so that only the active tokens are processed. We then scatter the active outputs back to their respective slots.

**Implementation details (Target: [models/moe_layer.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/Micro-MoE/models/moe_layer.py)):**

```python
            if compute_mask.any():
                frame_router = self._slice_router(router, frame_idx)
                flat_compute_mask = compute_mask.view(-1)  # [B * S]
                
                # Filter down to active tokens
                flat_frame_tokens = frame_tokens.reshape(batch * slots, hidden)
                active_tokens = flat_frame_tokens[flat_compute_mask]  # [N_active, hidden]
                
                # Slice router attributes for active tokens
                active_router = RouterOutput(
                    logits=frame_router.logits.reshape(batch * slots, -1)[flat_compute_mask],
                    probs=frame_router.probs.reshape(batch * slots, -1)[flat_compute_mask],
                    topk_indices=frame_router.topk_indices.reshape(batch * slots, -1)[flat_compute_mask],
                    topk_scores=frame_router.topk_scores.reshape(batch * slots, -1)[flat_compute_mask],
                    context=frame_router.context.reshape(batch * slots, -1)[flat_compute_mask],
                    entropy=frame_router.entropy.reshape(batch * slots)[flat_compute_mask],
                )
                
                # Execute experts only on active tokens
                active_output = self._run_experts(active_tokens, active_router)  # [N_active, hidden]
                
                # Scatter back to the full frame sequence
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
```

---

### 3.2. Issue 2: Single-Layer Loss Bottleneck in Training
#### Why the change is required:
In [train/trainer.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/Micro-MoE/train/trainer.py#L48-L57), loss regularization (auxiliary load balancing, CFCR, and expert orthogonalization) is only computed for the first block:
```python
losses = total_tmoe_loss(
    logits=output.logits,
    labels=labels,
    router_probs=first_router.probs,
    motion_confidence=output.motion_confidence,
    experts=first_block.moe.experts,
    weights=self.config.loss_weights,
)
```
Consequently, all subsequent layer blocks (`blocks[1:]`) are completely unregularized, causing:
1. Expert collapse in upper layers.
2. Divergent routing distribution shifts.
3. Severe degradation of cross-frame consistent routing.

#### How to implement the change:
We will compute these routing and parameter-level losses across all layers, average them, and combine them with the standard autoregressive language modeling loss.

**Implementation details (Target: [train/trainer.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/Micro-MoE/train/trainer.py)):**

```python
    def train_step(
        self,
        frames: Tensor,
        input_ids: Tensor,
        labels: Tensor,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(frames, input_ids, reset_cache=True)
        
        # 1. Autoregressive Language Loss
        from .loss import autoregressive_loss, load_balancing_loss, cfcr_loss, orthogonalization_loss
        ar_loss = autoregressive_loss(output.logits, labels)
        
        # 2. Compute Routing & Expert Divergence across ALL layers
        total_aux = 0.0
        total_cfcr = 0.0
        total_ortho = 0.0
        num_layers = len(self.model.blocks)
        
        for block, router_out in zip(self.model.blocks, output.router_outputs):
            total_aux += load_balancing_loss(router_out.probs)
            total_cfcr += cfcr_loss(router_out.probs, output.motion_confidence)
            total_ortho += orthogonalization_loss(block.moe.experts)
            
        aux_loss = total_aux / num_layers
        cfcr_loss_val = total_cfcr / num_layers
        ortho_loss = total_ortho / num_layers
        
        # 3. Aggregate total weighted loss
        w = self.config.loss_weights
        total_loss = ar_loss + (w.alpha_aux * aux_loss) + (w.beta_cfcr * cfcr_loss_val) + (w.gamma_ortho * ortho_loss)
        
        total_loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        
        return {
            "loss": total_loss.detach().item(),
            "ar": ar_loss.detach().item(),
            "aux": aux_loss.detach().item(),
            "cfcr": cfcr_loss_val.detach().item(),
            "ortho": ortho_loss.detach().item(),
        }
```

---

### 3.3. Feature 1: Comprehensive Ablation Study System
#### Goal:
Provide an automated entrypoint to validate the scientific claims of the Micro-MoE architecture. The script must compare various configurations against the control model (Full configuration).

#### Configuration Design (Target: [NEW] [run_ablations.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/Micro-MoE/run_ablations.py)):
We define a series of runs using synthetic dataset iterations:
1. **Full (Control)**: Standard model with all loss weights active ($\alpha=0.01, \beta=0.1, \gamma=0.01$) and temporal features enabled.
2. **No Orthogonalization**: $\gamma = 0$ (probes expert collapse).
3. **No CFCR**: $\beta = 0$ (probes temporal consistency).
4. **No Router History**: history_window = 0 (probes causal Conv1D context).
5. **No Motion Routing**: `use_motion_conditioning = False` (probes routing awareness).
6. **No Cache**: `cache_threshold = 0.0` (forces full compute).
7. **Aggressive Cache**: `cache_threshold = 0.1` (probes accuracy-efficiency trade-off).

**Expected Output Table:**
At the end of execution, the script will output a clean Markdown table summarizing the metrics:
```text
| Configuration | Total Loss | AR Loss | Aux Loss | CFCR Loss | Ortho Loss | Caching Efficiency (%) | Routing Entropy | Cosine Sim |
```

---

## 4. Verification Plan

### 4.1. Unit Test Additions (Target: [tests/test_modules.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/Micro-MoE/tests/test_modules.py))
To guarantee the token-level bypass is actually bypassing computation at the mathematical level, we will implement `test_token_level_cache_mixed_motion()`:
- Input a batch where frame 0 initializes the cache.
- In frame 1, set `motion_confidence` such that 2 tokens are static ($0.0$) and 2 tokens are active ($1.0$).
- Record the output and verify that `executed_tokens` is exactly $2$, and `cached_tokens` is exactly $2$.
- Verify that the outputs match the expected results.

### 4.2. Execution Verification Commands
1. Run standard unit tests:
   ```bash
   pytest -v
   ```
2. Execute the smoke test pipeline to ensure output shapes and caching stats print correctly:
   ```bash
   python run_pipeline.py
   ```
3. Run the ablation studies suite:
   ```bash
   python run_ablations.py
   ```
