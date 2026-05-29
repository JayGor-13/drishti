"""Actual VQA training runner script for Micro-MoE models."""

from __future__ import annotations

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from train.dataset import VideoVQADataset, VideoQACollator
from models.real_model import RealTMoELLaVA, AutoTokenizer
from train.loss import TMoELossWeights, cfcr_loss, load_balancing_loss, orthogonalization_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", type=str, required=True, help="Path to Video VQA json dataset annotations")
    parser.add_argument("--video-dir", type=str, required=True, help="Path to MP4 video files directory")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="Hugging Face LLM repository name")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model weights checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run training on")
    
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per training step")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Number of accumulation steps to simulate larger batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=-1, help="Max steps to run per epoch (for smoke testing)")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of video frames to sample per clip")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA adapter rank")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts in MoE layers")
    parser.add_argument("--top-k", type=int, default=2, help="Number of active experts per token")
    parser.add_argument("--cache-threshold", type=float, default=0.05, help="Caching threshold for static bypass")
    
    # Loss Weights
    parser.add_argument("--alpha-aux", type=float, default=0.01, help="Auxiliary expert load balancing loss coefficient")
    parser.add_argument("--beta-cfcr", type=float, default=0.1, help="Cross-Frame routing consistency loss coefficient")
    parser.add_argument("--gamma-ortho", type=float, default=0.01, help="Expert parameter orthogonalization loss coefficient")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== Launching Actual Micro-MoE Training ===")
    
    # 1. Setup tokenizer
    print(f"Loading tokenizer for: {args.llm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Setup dataset & dataloader
    print("Initializing Video VQA Dataloader...")
    dataset = VideoVQADataset(
        json_path=args.json_path,
        video_dir=args.video_dir,
        tokenizer=tokenizer,
        num_frames=args.num_frames,
    )
    collator = VideoQACollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # 3. Initialize Model
    print("Constructing RealTMoELLaVA model...")
    device = torch.device(args.device)
    model = RealTMoELLaVA(
        llm_model_name=args.llm_model,
        use_4bit=(device.type == "cuda"),
        num_experts=args.num_experts,
        top_k=args.top_k,
        lora_rank=args.lora_rank,
        cache_threshold=args.cache_threshold,
    )
    # If not loaded via bitsandbytes device map, manually place the non-frozen parameters
    if device.type != "cuda":
        model = model.to(device)

    # 4. Filter parameters to optimize (only train projection layers, routers, and LoRA parameters)
    trainable_params = []
    print("Trainable parameters initialized:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"  - {name}: shape={tuple(param.shape)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    
    loss_weights = TMoELossWeights(
        alpha_aux=args.alpha_aux,
        beta_cfcr=args.beta_cfcr,
        gamma_ortho=args.gamma_ortho,
    )

    # 5. Training Loop
    model.train()
    for epoch in range(args.epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{args.epochs} ---")
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        
        for step_idx, batch in enumerate(dataloader):
            if args.steps_per_epoch > 0 and step_idx >= args.steps_per_epoch:
                break
                
            # Place base frames & ids on device
            frames = batch["frames"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass under AMP mixed precision
            with autocast("cuda" if device.type == "cuda" else "cpu"):
                outputs = model(frames, input_ids, labels=labels, reset_cache=True)
                ar_loss = outputs["loss"]
                
                # Retrieve routing diagnostics to compute regularization losses across all layers
                aux_losses = []
                cfcr_losses = []
                ortho_losses = []
                for _, block, _ in model.patched_layers:
                    router = block.router
                    aux_losses.append(load_balancing_loss(router.probs))
                    cfcr_losses.append(cfcr_loss(router.probs, outputs["motion_confidence"]))
                    ortho_losses.append(orthogonalization_loss(block.experts))
                    
                aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)
                cfcr = torch.stack(cfcr_losses).mean() if cfcr_losses else torch.tensor(0.0, device=device)
                ortho = torch.stack(ortho_losses).mean() if ortho_losses else torch.tensor(0.0, device=device)
                
                total_loss = ar_loss + loss_weights.alpha_aux * aux
                total_loss = total_loss + loss_weights.beta_cfcr * cfcr + loss_weights.gamma_ortho * ortho
                # Divide by gradient accumulation steps
                total_loss = total_loss / args.grad_accum_steps

            # Backward pass using scale/optimizer
            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            epoch_loss += total_loss.item() * args.grad_accum_steps
            
            # Optimizer Step (Gradient accumulation boundary)
            if (step_idx + 1) % args.grad_accum_steps == 0 or (step_idx + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                
                if step_count % 10 == 0 or args.steps_per_epoch > 0:
                    print(
                        f"Step {step_count} | Loss: {total_loss.item() * args.grad_accum_steps:.4f} "
                        f"| AR: {ar_loss.item():.4f} | Aux: {aux.item():.4f} | CFCR: {cfcr.item():.4f}"
                    )

        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save Checkpoint at end of epoch
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        # Save only the trained projection & adapter weights to reduce disk footprint
        checkpoint_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                checkpoint_dict[name] = param.data.cpu()
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Saved trainable checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
