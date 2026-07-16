from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
# Prevent "Too many open files" error on Linux when using many dataloader workers
torch.multiprocessing.set_sharing_strategy('file_system')

from drishti_v2.experiments.common import build_loader, build_model, load_config, resolve_device, seed_everything
from drishti_v2.training import DRISHTITrainer, StageLossFactory

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--start-stage", type=int, default=1, help="Which stage to start from (1 to 4)")
    parser.add_argument("--frames-root", type=str, default="/home/newuser1/dataset_frames", help="Path to dataset frames")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config("./configs/default.yaml")
    if args.resume:
        config.checkpoint = args.resume

    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config)
    output_root = Path("results/full_training")

    print(f"Building model and loading frames from {args.frames_root}...")
    model = build_model(config, checkpoint=config.checkpoint, device=device)

    # 2. Build DataLoaders
    train_loader = build_loader(
        config,
        data_root=None,
        frames_root=args.frames_root,
        split="train",
        batch_size=config.train_batch_size,
        shuffle=True,
    )
    
    val_loader = build_loader(
        config,
        data_root=None,
        frames_root=args.frames_root,
        split="val",
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    # Define optimal epochs for each stage as requested
    stage_epochs = {
        "stage1": 80,
        "stage2": 40,
        "stage3": 30,
        "stage4": 30
    }

    stages = [f"stage{i}" for i in range(args.start_stage, 5)]
    for stage in stages:
        current_epochs = stage_epochs[stage]
        
        print(f"\n======================================")
        print(f"Starting training for {stage} ({current_epochs} epochs)...")
        print(f"======================================")
        
        loss_fn = StageLossFactory.make_loss(stage, config=config)
        
        trainer = DRISHTITrainer(
            model, 
            train_loader, 
            val_loader, 
            loss_fn, 
            output_dir=output_root / stage, 
            device=device
        )

        is_first_stage_in_loop = (stage == stages[0])
        resume_path = args.resume if (args.resume and is_first_stage_in_loop) else None
        
        trainer.fit(
            stage=stage, 
            epochs=current_epochs, 
            lr=config.smoke_lr,
            resume_checkpoint=resume_path
        )

    print("\nAll training stages complete!")

if __name__ == "__main__":
    main()
