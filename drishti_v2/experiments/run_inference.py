from __future__ import annotations

import argparse
import time

import torch

from drishti_v2.experiments.common import add_common_args, build_model, load_config, resolve_device, seed_everything
from drishti_v2.tracker import SimpleTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream synthetic DRISHTI inference and print detections/tracks.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(config, args.device)
    model = build_model(config, checkpoint=args.checkpoint, device=device)
    tracker = SimpleTracker(config.tracker_dist_threshold, config.tracker_max_coast, config.tracker_birth_threshold)
    model.reset_stream()

    start = time.perf_counter()
    for frame_idx in range(args.frames):
        tracker.predict()
        guided = tracker.get_guided_centers()
        if guided is not None:
            guided = guided.to(device)
        frame = torch.rand(1, config.image_channels, config.image_height, config.image_width, device=device) * 0.05
        output = model.forward_stream(frame, frame_idx, guided)
        tracker.update(output.boxes[0], output.objectness_logits[0])
        scores = torch.sigmoid(output.objectness_logits[0, :, 0])
        keep = scores > config.objectness_threshold
        print(
            {
                "frame": frame_idx,
                "detections": int(keep.sum().item()),
                "max_score": round(float(scores.max().item()), 4),
                "tracks": len(tracker.tracks),
            }
        )
    elapsed = time.perf_counter() - start
    print({"frames": args.frames, "elapsed_sec": round(elapsed, 4), "fps": round(args.frames / max(elapsed, 1e-8), 2)})


if __name__ == "__main__":
    main()
