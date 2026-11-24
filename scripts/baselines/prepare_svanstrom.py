"""Prepare the Svanström v1.0.0 thermal videos for drone detection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
from mcos_decoder import load_groundtruth

from small_object_data import sha256, write_json

RELEASE = "https://doi.org/10.5281/zenodo.5500576"
CLIP_COUNTS = {"DRONE": 157, "AIRPLANE": 74, "BIRD": 79, "HELICOPTER": 55}
CLIP_NAME = re.compile(r"IR_(DRONE|AIRPLANE|BIRD|HELICOPTER)_\d+\.mp4")
# The pinned decoder returns a different number of label slots than video frames.
# Exclude these fixed clips rather than assuming which frames the labels belong to.
UNALIGNED_CLIPS = ("IR_DRONE_067", "IR_DRONE_096", "IR_DRONE_101", "IR_DRONE_104")


def drone_box(box, width, height):
    x, y, w, h = box
    if not all(map(math.isfinite, box)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid drone box {box}")
    # The first pixel center is (1, 1) in MATLAB and (0.5, 0.5) from the COCO image edge.
    left, top = max(0.0, x - 0.5), max(0.0, y - 0.5)
    right, bottom = min(width, x - 0.5 + w), min(height, y - 0.5 + h)
    if right <= left or bottom <= top:
        raise ValueError(f"Drone box {box} lies outside the {width} x {height} frame")
    return [left, top, right - left, bottom - top]


def convert_clip(video, labels, category, output, data):
    clip = video.stem
    positive = category == "DRONE"
    relative = Path("images") / ("drone" if positive else "other") / clip
    (output / relative).mkdir(parents=True)
    boxes = load_groundtruth(labels) if positive else None
    if positive and not boxes:
        raise ValueError(f"{labels.name}: no decoded label slots")

    capture = cv2.VideoCapture(str(video))
    video_id = len(data["videos"]) + 1
    frame_id = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {video.name}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        data["videos"].append({"id": video_id, "name": clip, "fps": fps, "source_class": category})
        expected = len(boxes) if positive else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id >= expected:
                raise ValueError(f"{video.name}: decoded more than {expected} expected frames")
            height, width = frame.shape[:2]
            name = relative / f"{frame_id:06d}.png"
            if not cv2.imwrite(str(output / name), frame):
                raise OSError(f"Could not write {output / name}")
            image_id = len(data["images"]) + 1
            data["images"].append({
                "id": image_id, "file_name": name.as_posix(), "video_id": video_id,
                "frame_id": frame_id, "width": width, "height": height,
            })
            if positive and boxes[frame_id] is not None:
                try:
                    bbox = drone_box(boxes[frame_id], width, height)
                except ValueError as error:
                    raise ValueError(f"{labels.name} frame {frame_id}: {error}") from error
                data["annotations"].append({
                    "id": len(data["annotations"]) + 1, "image_id": image_id, "category_id": 1,
                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                })
            frame_id += 1
    finally:
        capture.release()
    if not frame_id or frame_id != expected:
        raise ValueError(f"{video.name}: decoded {frame_id} frames, expected {expected}")


def release_clips(members, source):
    clips = {}
    for entry in members:
        match = CLIP_NAME.fullmatch(entry.name)
        if entry.parts[-3:-1] == ("Data", "Video_IR") and match:
            if entry.stem in clips:
                raise ValueError(f"Duplicate video {entry.name} in {source}")
            clips[entry.stem] = (entry, match[1])
    counts = Counter(category for _, category in clips.values())
    if counts != CLIP_COUNTS:
        raise ValueError(f"Expected the complete thermal release {CLIP_COUNTS}, found {dict(counts)}")

    available = set(members)
    for clip, (video, category) in sorted(clips.items()):
        if clip in UNALIGNED_CLIPS:
            continue
        label = video.with_name(clip + "_LABELS.mat")
        if label not in available:
            raise ValueError(f"Missing {label} in {source}")
        yield video, label, category


def folder_digest(source, entries):
    digest = hashlib.sha256()
    for video, label, _ in entries:
        for entry in (video, label):
            digest.update(f"{entry.as_posix()} {sha256(source / entry)}\n".encode())
    return digest.hexdigest()


def write_dataset(source, stream, entries, provenance, output):
    annotation = output / "annotations/test.json"
    if output.exists():
        if annotation.is_file():
            info = json.loads(annotation.read_text())["info"]
            if all(info.get(key) == value for key, value in provenance.items()):
                print(f"Reusing {annotation}", flush=True)
                return
        raise FileExistsError(f"{output} already exists. Choose a new --out directory.")

    data = {"info": {"source": RELEASE, **provenance, "excluded_clips": list(UNALIGNED_CLIPS),
                     "description": "Aligned IR videos, with UAV as the only scored class"},
            "categories": [{"id": 1, "name": "uav"}], "images": [], "annotations": [], "videos": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="svanstrom-") as temp:
        staging = Path(temp) / "dataset"
        labels_dir = staging / "annotations"
        labels_dir.mkdir(parents=True)
        for number, (video_entry, label_entry, category) in enumerate(entries, 1):
            if stream is None:
                video, label = source / video_entry, source / label_entry
            else:
                video, label = Path(temp) / video_entry.name, Path(temp) / label_entry.name
                with stream.open(str(video_entry)) as member, video.open("wb") as destination:
                    shutil.copyfileobj(member, destination)
                if category == "DRONE":
                    label.write_bytes(stream.read(str(label_entry)))
            convert_clip(video, label, category, staging, data)
            if stream is not None:
                video.unlink()
                label.unlink(missing_ok=True)
            if number % 25 == 0:
                print(f"Prepared {number}/{len(entries)} clips", flush=True)
        write_json(labels_dir / "test.json", data)
        staging.rename(output)
    print(f"Prepared {len(entries)} clips, {len(data['images'])} frames, "
          f"and {len(data['annotations'])} drone boxes in {output}", flush=True)


def prepare(source: Path, output: Path) -> None:
    folder = source.is_dir()
    with (nullcontext() if folder else ZipFile(source)) as stream:
        if folder:
            members = [PurePosixPath(path.relative_to(source).as_posix())
                       for path in (source / "Data" / "Video_IR").iterdir() if path.is_file()]
        else:
            members = [PurePosixPath(name) for name in stream.namelist()]
        entries = list(release_clips(members, source))
        provenance = ({"thermal_files_sha256": folder_digest(source, entries)} if folder
                      else {"archive_sha256": sha256(source)})
        write_dataset(source, stream, entries, provenance, output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="The original Zenodo ZIP or the extracted dataset folder")
    parser.add_argument("--out", type=Path, required=True, help="Destination for the prepared dataset")
    args = parser.parse_args()
    prepare(args.source, args.out)
