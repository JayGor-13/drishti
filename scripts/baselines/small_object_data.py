"""Input adapters for the video and still-image small-object benchmarks."""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

YOLOFT = Path("/opt/YOLOFT")
VISDRONE_FILES = {
    "train": "1NSNapZQHar22OYzQYuXCugA3QlMndzvw",
    "val": "1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B",
    "test-dev": "1-BEq--FcjshTF1UwUabby_LHhYj41os5",
}
VISDRONE_NAMES = (
    "pedestrian", "person", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Split:
    root: Path
    coco: dict
    clips: dict[str, list[dict]]

    def image_path(self, image: dict) -> Path:
        return self.root / image["_relative_path"]


def write_json(path: Path, data) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(data, f, allow_nan=False)
        except BaseException:
            temp.unlink()
            raise
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_xsvid(root: Path, hub: str, revision: str | None, offline: bool) -> None:
    required = ("annotations/train_dev.json", "annotations/val_dev.json",
                "annotations/protocols/paper_detection_test.json", "images")
    if all((root / item).exists() for item in required) and (
        (root / ".drishti-download-complete").is_file() or not (root / ".drishti-downloading").exists()
    ):
        return
    if offline:
        raise FileNotFoundError(f"Missing XS-VID release files under {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".drishti-downloading"
    marker.touch()
    command = [sys.executable, str(YOLOFT / "tools/xsvid/download_release.py"),
               "--hub", hub, "--component", "dataset", "--destination", str(root.parent)]
    revision = revision or ("d6cbae5448f87b205794479ad68f5e5f38fa89e9" if hub == "hf" else None)
    if revision:
        command += ["--revision", revision]
    subprocess.run(command, check=True)
    if not all((root / item).exists() for item in required):
        raise FileNotFoundError(f"Downloader did not produce the expected XS-VID layout under {root}")
    marker.replace(root / ".drishti-download-complete")


def ensure_visdrone(root: Path, splits: list[str], offline: bool) -> None:
    for split in splits:
        name = f"VisDrone2019-VID-{split}"
        destination = root / name
        if (destination / "annotations").is_dir() and (destination / "sequences").is_dir():
            continue
        if offline:
            raise FileNotFoundError(f"Missing VisDrone split {destination}")
        root.mkdir(parents=True, exist_ok=True)
        archive = root / f"{name}.zip"
        if not archive.is_file():
            partial = root / f"{name}.zip.partial"
            subprocess.run(["gdown", VISDRONE_FILES[split], "-O", str(partial), "--continue"], check=True)
            if not zipfile.is_zipfile(partial):
                raise ValueError(f"Download is not a ZIP archive: {partial}")
            partial.replace(archive)
        with tempfile.TemporaryDirectory(dir=root, prefix="extract-") as temporary:
            with zipfile.ZipFile(archive) as stream:
                stream.extractall(temporary)
            extracted = Path(temporary) / name
            if not (extracted / "sequences").is_dir() or not (extracted / "annotations").is_dir():
                raise ValueError(f"Unexpected layout in {archive}. Expected {name}/sequences and annotations.")
            extracted.rename(destination)


def media_index(root: Path) -> dict[str, Path | None]:
    """Fallback for releases whose JSON gives basenames rather than relative paths."""
    index: dict[str, Path | None] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            index[path.name] = None if path.name in index else path
    return index


def coco_split(root: Path, annotation: Path, video: bool, limit: int | None = None) -> Split:
    data = json.loads(annotation.read_text())
    if limit is not None:
        def clip_key(row):
            if not video:
                return str(row["id"])
            parent = Path(row["file_name"]).parent
            if parent not in (Path("."), Path("images")):
                return parent.as_posix()
            identifier = row.get("video_id", row.get("sequence_id"))
            return str(identifier) if identifier is not None else None

        keys = [clip_key(row) for row in data["images"]]
        if None not in keys:
            chosen = set(sorted(set(keys))[:limit])
            data["images"] = [row for row, key in zip(data["images"], keys) if key in chosen]
            ids = {row["id"] for row in data["images"]}
            data["annotations"] = [row for row in data["annotations"] if row["image_id"] in ids]
    clips = defaultdict(list)
    index = None
    seen = set()
    if len({row["id"] for row in data["images"]}) != len(data["images"]):
        raise ValueError(f"Duplicate image IDs in {annotation}")
    for row in data["images"]:
        name = Path(row["file_name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Expected a dataset-relative file name in {annotation}: {name}")
        path = root / name
        if not path.is_file() and name.parts[0] != "images":
            path = root / "images" / name
        if not path.is_file():
            if index is None:
                index = media_index(root)
            path = index.get(name.name)
        if path is None:
            raise FileNotFoundError(f"{annotation}: missing or ambiguous image {name} under {root}")
        if path in seen:
            raise ValueError(f"{annotation}: duplicate image path {path}")
        seen.add(path)
        relative = path.relative_to(root)
        if video:
            clip = relative.parent.as_posix()
            if clip in (".", "images"):
                clip = row.get("video_id", row.get("sequence_id"))
                if clip is None:
                    raise ValueError(f"No video ID or sequence directory for {relative}")
            number = row.get("frame_id", row.get("frame_index"))
            if number is None:
                match = re.search(r"(\d+)$", name.stem)
                if not match:
                    raise ValueError(f"Cannot determine frame order for {relative}")
                number = match.group(1)
        else:
            clip, number = row["id"], 0
        clips[str(clip)].append({**row, "_relative_path": relative.as_posix(), "_frame": int(number)})
    for clip, frames in clips.items():
        frames.sort(key=lambda row: row["_frame"])
        if any(a["_frame"] == b["_frame"] for a, b in zip(frames, frames[1:])):
            raise ValueError(f"Duplicate frame index in {annotation}, clip {clip}")
    if not clips:
        raise ValueError(f"No images in {annotation}")
    return Split(root, data, dict(clips))


def visdrone_split(root: Path, limit: int | None = None) -> Split:
    from PIL import Image

    images, annotations, clips = [], [], {}
    categories = [{"id": number, "name": name} for number, name in enumerate(VISDRONE_NAMES, 1)]
    for label in sorted((root / "annotations").glob("*.txt"))[:limit]:
        clip = label.stem
        folder = root / "sequences" / clip
        paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                       key=lambda p: int(p.stem))
        if not paths:
            raise ValueError(f"No frames in {folder}")
        with Image.open(paths[0]) as image:
            width, height = image.size
        by_frame = {}
        for path in paths:
            number = int(path.stem)
            if number in by_frame:
                raise ValueError(f"Duplicate VisDrone frame {number} in {folder}")
            row = {"id": len(images) + 1, "file_name": path.relative_to(root).as_posix(),
                   "width": width, "height": height}
            images.append(row)
            by_frame[number] = {**row, "_relative_path": row["file_name"], "_frame": number}
        clips[clip] = list(by_frame.values())
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            frame, _, x, y, w, h, score, category, *_ = map(float, line.rstrip(",").split(","))
            category = int(category)
            if w <= 0 or h <= 0 or category not in range(11):
                continue
            if frame not in by_frame:
                raise ValueError(f"{label}: missing media for frame {frame:g}")
            annotations.append({"id": len(annotations) + 1, "image_id": by_frame[frame]["id"],
                                "category_id": category, "bbox": [x, y, w, h], "area": w * h,
                                "iscrowd": int(category == 0 or not score)})
    if not clips:
        raise ValueError(f"No VisDrone annotations under {root}")
    return Split(root, {"images": images, "annotations": annotations, "categories": categories}, clips)


def subset(split: Split, keys: list[str]) -> Split:
    clips = {key: split.clips[key] for key in keys}
    ids = {frame["id"] for frames in clips.values() for frame in frames}
    return Split(split.root, {
        **split.coco,
        "images": [row for row in split.coco["images"] if row["id"] in ids],
        "annotations": [row for row in split.coco["annotations"] if row["image_id"] in ids],
    }, clips)


def holdout(split: Split) -> tuple[Split, Split]:
    keys = sorted(split.clips)
    if len(keys) < 2:
        raise ValueError("At least two independent clips/images are required for a development split")
    random.Random(42).shuffle(keys)
    count = max(1, round(len(keys) * 0.15))
    return subset(split, keys[count:]), subset(split, keys[:count])


def svanstrom_splits(release: Split) -> tuple[Split, Split, Split]:
    """Hold out ordered blocks within each source class and skip the boundary clips."""
    groups = defaultdict(list)
    source_classes = {row["id"]: row["source_class"] for row in release.coco["videos"]}
    for clip, frames in release.clips.items():
        groups[source_classes[frames[0]["video_id"]]].append(clip)
    train, val, test = [], [], []
    for clips in groups.values():
        clips.sort(key=lambda clip: int(Path(clip).name.rsplit("_", 1)[1]))
        if len(clips) < 5:
            raise ValueError("Svanström needs at least five clips per source class")
        test_start = len(clips) - max(1, round(len(clips) * 0.20))
        val_start = test_start - 1 - max(1, round(len(clips) * 0.15))
        train.extend(clips[:val_start - 1])
        val.extend(clips[val_start:test_start - 1])
        test.extend(clips[test_start:])
    return subset(release, train), subset(release, val), subset(release, test)


def itsdt_development(release: Split, train_list: Path) -> Split:
    """Restrict the archive's larger COCO training export to the published train list."""
    names = {Path(*Path(line.split()[0]).parts[-3:]).as_posix()
             for line in train_list.read_text().splitlines() if line.strip()}
    clips = sorted({Path(name).parent.as_posix() for name in names})
    if not set(clips) <= release.clips.keys():
        raise ValueError(f"{train_list} names clips missing from the COCO training export")
    selected = subset(release, clips)
    if names != {row["file_name"] for row in selected.coco["images"]}:
        raise ValueError(f"{train_list} and the COCO training export disagree on the selected frames")
    return selected


def check_disjoint(train: Split, val: Split, test: Split, video: bool) -> None:
    seen_paths = set()
    seen_clips = set()
    for split in (train, val, test):
        paths = {split.image_path(frame).resolve() for frames in split.clips.values() for frame in frames}
        if seen_paths & paths:
            raise ValueError("An image appears in more than one split")
        if video and seen_clips & split.clips.keys():
            raise ValueError("A clip appears in more than one split")
        seen_paths.update(paths)
        seen_clips.update(split.clips)


def clip_boxes(rows: list[dict], category_map: dict, width: int, height: int) -> tuple[list, list]:
    targets, ignored = [], []
    for row in rows:
        x, y, w, h = row["bbox"]
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + w), min(height, y + h)
        if right <= left or bottom <= top:
            continue
        category = row["category_id"]
        if row.get("ignore") or row.get("iscrowd") or category not in category_map:
            ignored.append((left, top, right, bottom))
        else:
            targets.append((category_map[category], left, top, right, bottom))
    return targets, ignored
