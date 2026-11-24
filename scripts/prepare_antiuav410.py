"""Unpack the official Anti-UAV410 zip into the frame layout our code expects.

Official layout (inside Anti-UAV410.zip):   <split>/<seq>/*.jpg  +  <split>/<seq>/IR_label.json
Our layout (what every script reads):      <out>/<split>/<seq>/infrared/*.jpg + <out>/<split>/<seq>/infrared.json

The label file keeps every key (exist, gt_rect and the attribute flags LR, OC, FM, SV, ...), so
per-attribute results can be computed later. Reads straight from the zip; no temporary copy.

Usage
-----
    python scripts/prepare_antiuav410.py --zip Anti-UAV410.zip --out antiuav410_frames
"""
from __future__ import annotations

import argparse
import collections
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack Anti-UAV410 into the DRISHTI frame layout.")
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    counts: collections.Counter = collections.Counter()
    with zipfile.ZipFile(Path(args.zip).expanduser()) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 3 or parts[0] not in {"train", "val", "test"}:
                continue
            split, seq, name = parts
            if name == "IR_label.json":
                target = out / split / seq / "infrared.json"
                counts[f"{split}_sequences"] += 1
            elif name.lower().endswith(".jpg"):
                target = out / split / seq / "infrared" / name
                counts[f"{split}_frames"] += 1
            else:
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
