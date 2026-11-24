"""Export the extracted Anti-UAV frames to COCO json, one file per split, for training baselines.

Images are NOT copied: ``file_name`` is the path relative to ``--frames-root``, so point each
baseline's image root at the same frames folder. Frames where the target is absent get no
annotation (they stay in ``images`` as negatives). One category: ``uav`` (id 1).

Usage
-----
    python scripts/export_coco.py --frames-root ~/dataset_frames --modality infrared --out ~/coco_antiuav
    # optional: every Nth frame only (faster baseline training), test split always full
    python scripts/export_coco.py ... --train-stride 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.dataset import _box_exists, _candidate_boxes, _read_antiuav_json  # noqa: E402
from drishti_v2.data.utils import list_image_files  # noqa: E402


def export_split(frames_root: Path, split: str, modality: str, stride: int) -> dict:
    images, annotations = [], []
    image_id = ann_id = 1
    for seq_dir in sorted(p for p in (frames_root / split).iterdir() if p.is_dir()):
        frame_dir, ann_path = seq_dir / modality, seq_dir / f"{modality}.json"
        if not frame_dir.exists() or not ann_path.exists():
            continue
        frames = list_image_files(frame_dir)
        if not frames:
            continue
        boxes, exists = _read_antiuav_json(ann_path)
        with Image.open(frames[0]) as im:
            width, height = im.size
        for idx in range(0, len(frames), max(stride, 1)):
            images.append({
                "id": image_id, "file_name": str(frames[idx].relative_to(frames_root)).replace("\\", "/"),
                "width": width, "height": height, "sequence": seq_dir.name, "frame_index": idx,
            })
            raw = boxes[idx] if idx < len(boxes) else []
            present = exists[idx] if idx < len(exists) else _box_exists(raw)
            if present:
                for x, y, w, h in _candidate_boxes(raw):
                    if w > 0 and h > 0:
                        annotations.append({"id": ann_id, "image_id": image_id, "category_id": 1,
                                            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                        ann_id += 1
            image_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "uav"}]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Anti-UAV extracted frames to COCO json.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared", choices=["infrared", "visible"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--train-stride", type=int, default=1, help="Keep every Nth frame in train/val (test is always full).")
    args = parser.parse_args()
    root, out = Path(args.frames_root).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        coco = export_split(root, split, args.modality, 1 if split == "test" else args.train_stride)
        path = out / f"{split}_{args.modality}.json"
        path.write_text(json.dumps(coco), encoding="utf-8")
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} boxes -> {path}")


if __name__ == "__main__":
    main()
