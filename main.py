from __future__ import annotations

import argparse
import json
from pathlib import Path

from drishti_v2.evaluation import DRISHTIEvaluator
from drishti_v2.experiments.common import build_loader, build_model, load_config, resolve_device, seed_everything
from drishti_v2.experiments.smoke import run_smoke
from drishti_v2.training import DRISHTITrainer, StageLossFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRISHTI-CORE v2 runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "smoke"], default="smoke")
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2", "stage3", "stage4", "finetune"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sequence-dir", default=None, help="Override config.smoke_sequence_dir for smoke mode.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for train/eval smoke checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(config, args.device)
    output_root = Path(config.output_dir)

    if args.mode == "smoke":
        sequence_dir = args.sequence_dir or config.smoke_sequence_dir
        if not sequence_dir:
            raise ValueError("Set smoke_sequence_dir in config or pass --sequence-dir.")
        summary = run_smoke(config, sequence_dir, output_dir=Path(config.smoke_output_video).parent, device=device)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    model = build_model(config, checkpoint=config.checkpoint, device=device)

    if args.mode == "eval":
        loader = build_loader(
            config,
            data_root=None,
            split=config.eval_split,
            batch_size=config.eval_batch_size,
            synthetic=args.synthetic,
            shuffle=False,
            frames_root=None,
        )
        metrics = DRISHTIEvaluator(model, loader, device=device, threshold=config.objectness_threshold).evaluate(
            print_results=True,
            output_path=output_root / "eval" / "metrics.json",
            save_visualizations=True,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    train_loader = build_loader(
        config,
        data_root=None,
        split="train",
        batch_size=config.train_batch_size,
        synthetic=args.synthetic,
        shuffle=True,
        frames_root=None,
    )
    val_loader = build_loader(
        config,
        data_root=None,
        split=config.eval_split,
        batch_size=config.eval_batch_size,
        synthetic=args.synthetic,
        shuffle=False,
        frames_root=None,
    )
    loss_fn = StageLossFactory.make_loss(args.stage, config=config)
    trainer = DRISHTITrainer(model, train_loader, val_loader, loss_fn, output_dir=output_root / "train", device=device)
    trainer.fit(stage=args.stage, epochs=args.epochs, lr=args.lr or config.smoke_lr)


if __name__ == "__main__":
    main()
