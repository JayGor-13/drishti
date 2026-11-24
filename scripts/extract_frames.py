from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _find_annotation(sequence_dir: Path, data_root: Path, split: str, modality: str, label_dir_name: str = "label_new") -> Path | None:
    name = sequence_dir.name
    label_root = data_root / label_dir_name
    candidates = [
        sequence_dir / f"{modality}.json",
        label_root / split / name / f"{modality}.json",
        label_root / name / f"{modality}.json",
        label_root / split / f"{name}_{modality}.json",
        label_root / split / f"{modality}_{name}.json",
        label_root / split / f"{name}.json",
        label_root / f"{name}_{modality}.json",
        label_root / f"{modality}_{name}.json",
        label_root / f"{name}.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def extract_task(
    video_path: Path,
    annotation_path: Path,
    output_frame_dir: Path,
    output_annotation_path: Path,
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
) -> dict:
    import cv2

    existing = list(output_frame_dir.glob(f"*{image_ext}"))
    if existing and not overwrite:
        return {"video": str(video_path), "status": "skipped", "frames": len(existing)}

    output_frame_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_frame_dir.glob(f"*{image_ext}"):
            path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"video": str(video_path), "status": "failed_open", "frames": 0}

    params = []
    if image_ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output_path = output_frame_dir / f"{count:06d}{image_ext}"
        cv2.imwrite(str(output_path), frame, params)
        count += 1
    capture.release()
    output_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotation_path, output_annotation_path)
    return {"video": str(video_path), "status": "extracted", "frames": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Anti-UAV videos into frame directories.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--modalities", nargs="+", default=["visible"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-ext", default=".jpg", choices=[".jpg", ".jpeg", ".png"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    tasks = []
    for split in args.splits:
        split_root = data_root / split
        if not split_root.exists():
            continue
        for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for modality in args.modalities:
                video_path = sequence_dir / f"{modality}.mp4"
                annotation_path = _find_annotation(sequence_dir, data_root, split, modality)
                if video_path.exists() and annotation_path is not None:
                    tasks.append(
                        (
                            video_path,
                            annotation_path,
                            output_root / split / sequence_dir.name / modality,
                            output_root / split / sequence_dir.name / f"{modality}.json",
                            args.image_ext,
                            args.jpeg_quality,
                            args.overwrite,
                        )
                    )

    print({"tasks": len(tasks), "output_root": str(output_root.resolve())})
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(extract_task, *task) for task in tasks]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
