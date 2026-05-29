"""Actual VQA model evaluation script for Micro-MoE models."""

from __future__ import annotations

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from train.dataset import VideoVQADataset, VideoQACollator
from models.real_model import RealTMoELLaVA, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", type=str, required=True, help="Path to Video VQA test JSON annotations")
    parser.add_argument("--video-dir", type=str, required=True, help="Path to MP4 test video files directory")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained checkpoint weight file (.pt)")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Hugging Face LLM repository name")
    parser.add_argument("--output-json", type=str, default="evaluation_results.json", help="Path to save output generated answers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of video frames to sample")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts in MoE layers")
    parser.add_argument("--top-k", type=int, default=2, help="Number of active experts per token")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA adapter rank")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max text tokens to generate for response")
    
    return parser.parse_args()


def load_trained_checkpoint(model: RealTMoELLaVA, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Loading trained weights from: {checkpoint_path}")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    # Load the parameters (strict=False because checkpoint only contains trained projection + adapters)
    missing, unexpected = model.load_state_dict(checkpoint_dict, strict=False)
    print(f"Loaded checkpoint. Missing: {len(missing)} parameters (frozen base), Unexpected: {len(unexpected)}")


def main() -> None:
    args = parse_args()
    print("=== Launching Actual Micro-MoE Evaluation ===")

    # 1. Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup dataset
    print("Loading VQA evaluation dataset...")
    dataset = VideoVQADataset(
        json_path=args.json_path,
        video_dir=args.video_dir,
        tokenizer=tokenizer,
        num_frames=args.num_frames,
    )

    # 3. Initialize Model and load weights
    print("Loading patched VLM architecture...")
    device = torch.device(args.device)
    model = RealTMoELLaVA(
        llm_model_name=args.llm_model,
        use_4bit=(device.type == "cuda"),
        num_experts=args.num_experts,
        top_k=args.top_k,
        lora_rank=args.lora_rank,
    )
    load_trained_checkpoint(model, args.checkpoint_path)
    
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    results = []
    correct_matches = 0

    print("Running autoregressive inference...")
    for idx in tqdm(range(len(dataset))):
        item = dataset.data[idx]
        sample = dataset[idx]

        # Extract features on device
        frames = sample["frames"].unsqueeze(0).to(device)  # [1, T, C, H, W]
        # For prompt input, we tokenise only the question/prompt (without assistant answer)
        question = item.get("question") or item["QA"][0]["q"]
        ground_truth = item.get("answer") or item["QA"][0]["a"]

        prompt = f"<|im_start|>user\n<video>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Retrieve visual and motion tokens
        with torch.no_grad():
            # Get video features from encoders
            batch, time, channels, height, width = frames.shape
            flat_frames = frames.reshape(batch * time, channels, height, width)
            
            if model.clip:
                clip_outputs = model.clip(flat_frames)[0]
                visual_tokens = model.visual_proj(clip_outputs)
                seq_len = visual_tokens.shape[1]
                visual_tokens = visual_tokens.reshape(batch, time, seq_len, model.llm_dim)
            else:
                visual_tokens = torch.zeros(batch, time, 16, model.llm_dim, device=device)
                seq_len = 16

            if model.motion_encoder:
                x3d_input = frames.permute(0, 2, 1, 3, 4)
                motion_features = model.motion_encoder(x3d_input)
                pooled = F.adaptive_avg_pool3d(motion_features, (time, int(seq_len**0.5), int(seq_len**0.5)))
                pooled = pooled.flatten(3).transpose(2, 3).permute(0, 3, 2, 1)
                motion_embeddings = model.motion_proj(pooled)
            else:
                motion_embeddings = torch.zeros(batch, time, seq_len, model.llm_dim, device=device)

            motion_confidence = torch.sigmoid(model.motion_conf_head(motion_embeddings)).squeeze(-1)

            # Build inputs_embeds
            text_embeds = model.llm.model.embed_tokens(input_ids)
            visual_flat = visual_tokens.flatten(1, 2)
            inputs_embeds = torch.cat([visual_flat, text_embeds], dim=1)

            # Hook visual properties to patched MLPs dynamically for generation pass
            model.current_time = time
            model.current_seq_len = seq_len
            hooks = []
            for _, block, cache in model.patched_layers:
                block.current_motion_embeddings = motion_embeddings
                block.current_motion_confidence = motion_confidence
                block.current_cache = cache

                def make_pre_hook(mlp_layer, c_cache, parent_model):
                    def pre_hook(module, inputs):
                        x = inputs[0]
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
                            output = module.original_mlp(x)
                            return output
                    return pre_hook

                hook = block.register_forward_pre_hook(make_pre_hook(block, cache, model))
                hooks.append(hook)

            try:
                # Generate answer using Hugging Face native generation loop
                gen_outputs = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                )
                generated_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            finally:
                for hook in hooks:
                    hook.remove()

        # Print outputs
        print(f"\nQuestion: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction: {generated_text}")
        
        # Calculate simple exact match metric (lowercased, stripped)
        is_correct = ground_truth.strip().lower() in generated_text.strip().lower()
        if is_correct:
            correct_matches += 1

        results.append({
            "video": item.get("video"),
            "question": question,
            "ground_truth": ground_truth,
            "prediction": generated_text,
            "is_correct": is_correct,
        })

    # Summary
    accuracy = 100.0 * correct_matches / len(dataset)
    print(f"\nEvaluation Complete | Total Samples: {len(dataset)} | Text Match Accuracy: {accuracy:.2f}%")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Detailed answers saved to: {args.output_json}")


if __name__ == "__main__":
    main()
