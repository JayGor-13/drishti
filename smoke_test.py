from __future__ import annotations

from pathlib import Path

from drishti_v2.experiments.common import load_config, resolve_device, seed_everything
from drishti_v2.experiments.smoke import run_smoke


def main() -> None:
    config = load_config(r"./configs/default.yaml")
    config.image_height = 128
    config.image_width = 128
    config.crop_size = 32
    config.temporal_window = 3
    config.train_batch_size = 1
    config.eval_batch_size = 1
    config.smoke_max_frames = 12
    # Set smoke_train_steps to 30 to test for 30 epochs
    config.smoke_train_steps = 30 
    config.smoke_sequence_dir = "my_video_dataset/train/my_first_video"
    config.smoke_output_video = "results/smokerun/bounding_boxes.mp4"
    config.validate()

    seed_everything(config.seed)
    device = resolve_device(config)
    summary = run_smoke(
        config,
        sequence_root=Path(config.smoke_sequence_dir),
        output_dir=Path(config.smoke_output_video).parent,
        device=device,
    )
    print(summary)


if __name__ == "__main__":
    main()
