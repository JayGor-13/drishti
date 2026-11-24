from __future__ import annotations

from pathlib import Path

from drishti_v2.experiments.common import build_loader, build_model, load_config, resolve_device, seed_everything
from drishti_v2.training import DRISHTITrainer, StageLossFactory


import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4], help="Which stage to start/resume from (1-4)")
    args = parser.parse_args()

    # 1. Load and customize configuration
    config = load_config(r"./configs/default.yaml")
    if args.resume:
        config.checkpoint = args.resume

    
    # Set model/data properties for small/fast test run
    config.image_height = 128
    config.image_width = 128
    config.crop_size = 32
    config.temporal_window = 3
    config.train_batch_size = 1
    config.eval_batch_size = 1
    
    # Point directly to your dataset root 
    config.data_root = "my_video_dataset"
    config.validate()

    seed_everything(config.seed)
    device = resolve_device(config)
    output_root = Path("results/full_training")

    print(f"Building model and loading data from {config.data_root}...")
    model = build_model(config, checkpoint=config.checkpoint, device=device)

    # 2. Build DataLoaders for full epochs
    train_loader = build_loader(
        config,
        data_root=config.data_root,
        split="train",
        batch_size=config.train_batch_size,
        shuffle=True,
    )
    
    # We use the train loader for validation too since you only have 1 video right now
    val_loader = build_loader(
        config,
        data_root=config.data_root,
        split="train", 
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    stages = [f"stage{i}" for i in range(args.start_stage, 5)]
    for stage in stages:
        print(f"\n======================================")
        print(f"Starting training for {stage} (5 epochs)...")
        print(f"======================================")
        
        # Initialize loss function for the current stage
        loss_fn = StageLossFactory.make_loss(stage, config=config)
        
        # Create a new trainer pointing to a stage-specific output directory
        trainer = DRISHTITrainer(
            model, 
            train_loader, 
            val_loader, 
            loss_fn, 
            output_dir=output_root / stage, 
            device=device
        )

        # Run the training loop for 5 epochs
        trainer.fit(stage=stage, epochs=5, lr=config.smoke_lr)
    
    print(f"\nAll training stages complete! Models and metrics are saved in '{output_root}'")


if __name__ == "__main__":
    main()
