from __future__ import annotations

import argparse

from drishti_v2.evaluation import DRISHTIEvaluator
from drishti_v2.experiments.common import add_common_args, build_loader, build_model, load_config, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DRISHTI-CORE v2.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="results/eval/metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(config, args.device)
    model = build_model(config, checkpoint=args.checkpoint, device=device)
    loader = build_loader(
        config,
        args.data_root,
        args.split,
        config.eval_batch_size,
        synthetic=args.synthetic,
        shuffle=False,
        frames_root=args.frames_root,
        modality=args.modality,
        clip_stride=args.clip_stride,
        frame_stride=args.frame_stride,
        box_format=args.box_format,
    )
    evaluator = DRISHTIEvaluator(model, loader, device=device, threshold=config.objectness_threshold)
    evaluator.evaluate(print_results=True, output_path=args.output)


if __name__ == "__main__":
    main()
