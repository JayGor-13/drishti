"""Convert the official Anti-UAV410 tracker outputs (one x,y,w,h line per frame) into the
prediction JSON that scripts/score_predictions.py reads, for every tracker in the archive.

Trackers output exactly one box per frame and no confidence, so each box gets score 1.0.
A frame is "no target" when the line is empty, NaN, or has non-positive width or height.
Note: these trackers are given the first-frame box; report them in a separate table block.

Usage
-----
    python scripts/import_tracker_results.py --results-zip tracker_results.zip \
        --group Trained_with_antiuav410 --out tracker_preds
    # then, per tracker:
    python scripts/score_predictions.py --frames-root antiuav410_frames --split test \
        --pred-dir tracker_preds/SiamDT --name SiamDT --out scores/SiamDT_test.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import zipfile
from pathlib import Path


# Trackers whose files hold inclusive corners x1,y1,x2,y2 instead of x,y,w,h. Verified on 29 Sep 2026 by
# matching the first output line (the init box) to the ground-truth frame-0 box [206, 314, 8, 8] of
# 02_6319_1500-2999: GlobalTrack writes 206,314,213,321 and SwinTrack writes 206.5,314.5,213.5,321.5.
CORNER_TRACKERS = {"GlobalTrack", "SwinTrack-Base", "SwinTrack-Tiny"}


def corners_to_xywh(box: list[float] | None) -> list[float] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    return [x1, y1, w, h] if w > 0 and h > 0 else None


def read_boxes(text: str) -> list[list[float] | None]:
    """One box (or None) per frame; handles line-per-frame files and SiamDT's {"res": [...]} JSON."""

    stripped = text.strip()
    if stripped.startswith("{"):
        rows = json.loads(stripped).get("res", [])
        return [parse_line(",".join(str(v) for v in row)) if isinstance(row, list) else None for row in rows]
    return [parse_line(line) for line in text.splitlines()]


def parse_line(line: str) -> list[float] | None:
    values = [v for v in line.replace("\t", ",").replace(" ", ",").split(",") if v]
    if len(values) < 4:
        return None
    try:
        x, y, w, h = (float(v) for v in values[:4])
    except ValueError:
        return None
    if any(math.isnan(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Anti-UAV410 tracker result files.")
    parser.add_argument("--results-zip", required=True)
    parser.add_argument("--group", default="Trained_with_antiuav410", choices=["Trained_with_antiuav410", "Defaults"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    written: collections.Counter = collections.Counter()
    seen: set[tuple[str, str]] = set()
    variant_of: dict[str, str] = {}
    with zipfile.ZipFile(Path(args.results_zip).expanduser()) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            # Layouts in the archive (keep the first variant per tracker, in sorted order):
            #   <group>/<tracker>/<seq>.txt                       (SiamDT, Stark, ...)
            #   <group>/<tracker>/<variant>[/<sub>]/<seq>.txt     (DiMP/dimp50, DropTrack/ostrack/...)
            #   <group>/<tracker>/.../<seq>/bounding_box.txt      (SwinTrack)
            # Timing files (<seq>_time.txt, time.txt, a 'times' folder) are skipped.
            if len(parts) < 4 or parts[1] != args.group or not name.endswith(".txt"):
                continue
            stem = Path(parts[-1]).stem
            if stem == "time" or stem.endswith("_time") or "times" in parts[3:-1]:
                continue
            tracker = parts[2]
            sequence = parts[-2] if stem == "bounding_box" else stem
            if tracker in variant_of and variant_of[tracker] != "/".join(parts[3:-1] if stem != "bounding_box" else parts[3:-2]):
                continue
            variant_of.setdefault(tracker, "/".join(parts[3:-1] if stem != "bounding_box" else parts[3:-2]))
            if (tracker, sequence) in seen:
                continue
            seen.add((tracker, sequence))
            frames = {}
            for idx, box in enumerate(read_boxes(archive.read(name).decode("utf-8", "ignore"))):
                if tracker in CORNER_TRACKERS:
                    box = corners_to_xywh(box)
                frames[str(idx)] = {"boxes": [box] if box else [], "scores": [1.0] if box else []}
            target = out / tracker / f"{sequence}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            written[tracker] += 1
    for tracker, count in sorted(written.items()):
        print(f"{tracker}: {count} sequences")


if __name__ == "__main__":
    main()
