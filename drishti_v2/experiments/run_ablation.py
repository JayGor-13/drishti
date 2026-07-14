from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from drishti_v2.evaluation import DRISHTIEvaluator
from drishti_v2.experiments.common import add_common_args, build_loader, load_config, resolve_device, seed_everything
from drishti_v2.models import DRISHTIPipeline


def variants(config):
    return {
        "full": config,
        "no_ldmi": replace(config, use_ldmi=False),
        "no_edge_crops": replace(config, use_edge_crops=False),
        "dense_moe": replace(config, dense_moe=True),
        "scan_period_2": replace(config, scan_period=2),
        "scan_period_8": replace(config, scan_period=8),
        "num_crops_4": replace(config, num_crops=4),
        "num_crops_16": replace(config, num_crops=16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick DRISHTI ablation evaluation.")
    add_common_args(parser)
    parser.add_argument("--output", default="results/ablation/summary.csv")
    args = parser.parse_args()

    base_config = load_config(args.config)
    seed_everything(base_config.seed)
    device = resolve_device(base_config, args.device)
    rows = []
    for name, config in variants(base_config).items():
        model = DRISHTIPipeline(config).to(device)
        loader = build_loader(
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
        metrics = DRISHTIEvaluator(model, loader, device=device, threshold=config.objectness_threshold).evaluate(print_results=False)
        row = {"variant": name, **metrics}
        rows.append(row)
        print(row)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ablation summary written to {path.resolve()}")


if __name__ == "__main__":
    main()
