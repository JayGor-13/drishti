from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

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


def load_compatible_weights(model, checkpoint: str | None, device) -> tuple[int, int]:
    """Load every tensor whose name and shape match; return (loaded, skipped) counts.

    Variants that change tensor shapes (no LDMI changes the MotionCNN input channels) keep random
    weights for the mismatched tensors. Such rows are flagged in the summary because they need a
    retrain to be a valid ablation.
    """

    if not checkpoint:
        return 0, sum(1 for _ in model.state_dict())
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    own = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in own and own[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(own) - len(compatible)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference-time DRISHTI ablations on a trained checkpoint. Variants that change the "
            "architecture are flagged 'retrain_required' and their numbers must not be reported."
        )
    )
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint (.pt) to evaluate the variants with.")
    parser.add_argument("--output", default="results/ablation/summary.csv")
    args = parser.parse_args()
    if not args.checkpoint:
        print("WARNING: no --checkpoint given; every variant runs with random weights and the numbers are meaningless.")

    base_config = load_config(args.config)
    seed_everything(base_config.seed)
    device = resolve_device(base_config, args.device)
    rows = []
    for name, config in variants(base_config).items():
        model = DRISHTIPipeline(config).to(device)
        loaded, skipped = load_compatible_weights(model, args.checkpoint, device)
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
        row = {
            "variant": name,
            "weights": "trained" if skipped == 0 and loaded > 0 else "retrain_required",
            "tensors_loaded": loaded,
            "tensors_skipped": skipped,
            **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        }
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
