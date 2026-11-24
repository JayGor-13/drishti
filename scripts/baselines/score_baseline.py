"""Score ANY baseline's predictions with exactly the DRISHTI protocol, in one command.

Give it the baseline's predictions for every frame of every VAL and TEST sequence (one JSON per
sequence, format below). It writes:
  score_test_raw.json      test score of the raw output (all boxes, threshold 0.3)
  postprocess_sweep.json   duplicate merging / one box per frame / threshold / smoothing chosen on VAL,
                           then applied once to TEST  <- the number that goes in the paper table

Prediction format, one file per sequence, named <sequence>.json:
    {"frames": {"0": {"boxes": [[...4 numbers...], ...], "scores": [0.93, ...]},
                "1": {"boxes": [], "scores": []}, ...}}
  * frame keys are 0-based frame indices in sorted file order; a missing frame = no detection
  * keep ALL boxes with their scores (no threshold), at most ~10 per frame is plenty
  * --box-format normalized (default): cx, cy, w, h in [0, 1] (YOLO's xywhn)
    --box-format pixel: x, y, w, h in original image pixels, top-left corner (COCO / Anti-UAV)

Usage
-----
    python scripts/baselines/score_baseline.py --frames-root /workspace/data/antiuav410_frames \
        --val-pred <dir with val jsons> --test-pred <dir with test jsons> --box-format pixel \
        --name DEIM-D-FINE-S --out /workspace/runs/baseline_deim_s
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from drishti_v2.data.utils import list_image_files  # noqa: E402


def to_normalized(pred_dir: Path, frames_root: Path, split: str, modality: str, out_dir: Path) -> Path:
    """Convert pixel x,y,w,h predictions to normalized cx,cy,w,h (what the sweep reads)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pred_dir.glob("*.json")):
        frames = list_image_files(frames_root / split / path.stem / modality)
        if not frames:
            print(f"WARNING: no frames for {path.stem} in {split}; skipped")
            continue
        with Image.open(frames[0]) as im:
            width, height = im.size
        data = json.loads(path.read_text(encoding="utf-8"))["frames"]
        converted = {}
        for key, item in data.items():
            boxes = [[(x + w / 2) / width, (y + h / 2) / height, w / width, h / height] for x, y, w, h in item.get("boxes", [])]
            converted[key] = {"boxes": boxes, "scores": item.get("scores", [])}
        (out_dir / path.name).write_text(json.dumps({"frames": converted}), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a baseline with the DRISHTI protocol (val-chosen post-processing).")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--val-pred", required=True)
    parser.add_argument("--test-pred", required=True)
    parser.add_argument("--box-format", default="normalized", choices=["normalized", "pixel"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root)
    val_dir, test_dir = Path(args.val_pred), Path(args.test_pred)
    if args.box_format == "pixel":
        val_dir = to_normalized(val_dir, frames_root, "val", args.modality, out / "normalized_val")
        test_dir = to_normalized(test_dir, frames_root, "test", args.modality, out / "normalized_test")
    py = sys.executable
    subprocess.run([py, str(CODE_ROOT / "scripts" / "score_predictions.py"), "--frames-root", args.frames_root, "--split", "test",
                    "--modality", args.modality, "--pred-dir", str(test_dir), "--name", args.name, "--normalized",
                    "--out", str(out / "score_test_raw.json")], check=True)
    subprocess.run([py, str(CODE_ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
                    "--modality", args.modality, "--val-raw", str(val_dir), "--test-raw", str(test_dir),
                    "--out", str(out / "postprocess_sweep.json")], check=True)
    print(f"done: {out / 'postprocess_sweep.json'} (field chosen_on_val.test is the paper number)")


if __name__ == "__main__":
    main()
