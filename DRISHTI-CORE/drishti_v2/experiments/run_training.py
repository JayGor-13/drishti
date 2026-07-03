from __future__ import annotations

import argparse
from pathlib import Path

from drishti_v2.experiments.common import add_common_args, build_loader, build_model, load_config, resolve_device, seed_everything
from drishti_v2.training import DRISHTILoss, DRISHTITrainer


STAGE_DEFAULTS = {
    "stage1": {"epochs": 80, "lr": 1e-4},
    "stage2": {"epochs": 30, "lr": 5e-5},
    "stage3": {"epochs": 20, "lr": 1e-5},
    "finetune": {"epochs": 10, "lr": 2e-6},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DRISHTI-CORE v2.")
    add_common_args(parser)
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), default="stage1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default="results/train")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume optimizer state and epoch from checkpoint.")
    parser.add_argument("--resume-from", default=None, help="Checkpoint to resume from. Defaults to output-dir/checkpoints/latest.pt.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(config, args.device)
    output_dir = Path(args.output_dir)
    resume_from = None
    if args.resume:
        candidates = [
            Path(args.resume_from) if args.resume_from else None,
            Path(args.checkpoint) if args.checkpoint else None,
            output_dir / "checkpoints" / "latest.pt",
            output_dir / f"{args.stage}_last.pt",
        ]
        resume_from = next((path for path in candidates if path is not None and path.exists()), None)
        if resume_from is None:
            print(f"--resume was set, but no checkpoint was found under {output_dir}. Starting from scratch.")

    model_checkpoint = None if resume_from is not None else args.checkpoint
    model = build_model(config, checkpoint=model_checkpoint, device=device)
    train_loader = build_loader(
        config,
        args.data_root,
        "train",
        config.train_batch_size,
        synthetic=args.synthetic,
        shuffle=True,
        frames_root=args.frames_root,
        modality=args.modality,
        clip_stride=args.clip_stride,
        frame_stride=args.frame_stride,
        box_format=args.box_format,
    )
    val_loader = build_loader(
        config,
        args.data_root,
        "val",
        config.eval_batch_size,
        synthetic=args.synthetic,
        shuffle=False,
        frames_root=args.frames_root,
        modality=args.modality,
        clip_stride=args.clip_stride,
        frame_stride=args.frame_stride,
        box_format=args.box_format,
    )
    loss_fn = DRISHTILoss(w_balance=config.moe_balance_weight)
    trainer = DRISHTITrainer(model, train_loader, val_loader, loss_fn, output_dir=args.output_dir, device=device)
    defaults = STAGE_DEFAULTS[args.stage]
    history = trainer.fit(
        stage=args.stage,
        epochs=args.epochs or defaults["epochs"],
        lr=args.lr or defaults["lr"],
        checkpoint_name="best_model.pt",
        resume_from=resume_from,
    )
    print(f"Training complete. Results written to {Path(args.output_dir).resolve()}")
    if history:
        print(f"Final epoch: {int(history[-1]['epoch'])}")


if __name__ == "__main__":
    main()
