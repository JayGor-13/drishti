"""Prepare, train, predict, and score small-object benchmarks."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import islice
from pathlib import Path
from statistics import mean, stdev
from unittest.mock import patch

from small_object_data import (
    IMAGE_SUFFIXES, Split, check_disjoint, clip_boxes, coco_split, ensure_visdrone,
    ensure_xsvid, holdout, itsdt_development, sha256, subset, svanstrom_splits,
    visdrone_split, write_json,
)
from small_object_eval import AITOD_EVAL, XS_EVAL, score_split

VIDEO_VARIANTS = {
    "rgb": ("rgb", False, "iou"),
    "gray": ("gray", False, "iou"),
    "raw-short": ("short", False, "iou"),
    "raw-long": ("long", False, "iou"),
    "aligned-short": ("short", True, "iou"),
    "aligned-long": ("long", True, "iou"),
    "aligned-long-nwd": ("long", True, "relative"),
}
IMAGE_VARIANTS = {
    "rgb": VIDEO_VARIANTS["rgb"],
    "fixed-nwd-assign": ("rgb", False, "fixed-assign"),
    "relative-nwd-assign": ("rgb", False, "relative-assign"),
    "relative-nwd": ("rgb", False, "relative"),
}
MAX_DET = {"xsvid": 700, "visdrone": 5000, "irdst-r": 100, "aitod-v2": 12000,
           "svanstrom": 100, "itsdt-15k": 100}
NUMBERED_HISTORY_DATASETS = {"irdst-r", "itsdt-15k"}
TRANSFER_DATASETS = {"svanstrom", "itsdt-15k"}
COCO_METRICS = ("AP", "AP50", "AP75", "AP_small")
XS_REFERENCES = (
    ("YOLOv8-S", 22.7, 9.0), ("CFINet", 19.8, 5.4),
    ("YOLOFT-S", 25.8, 9.8), ("YOLOFT-L", 29.3, 11.5),
)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def load_splits(args):
    dataset, root = args.dataset, args.data_root
    if dataset == "xsvid":
        release = root / "XS-VID-v2"
        with exclusive_lock(root / ".xsvid.lock"):
            ensure_xsvid(release, args.xs_hub, args.xs_revision, args.offline)
        train_json = release / "annotations/train_dev.json"
        val_json = release / "annotations/val_dev.json"
        test_json = release / "annotations/protocols/paper_detection_test.json"
        train = coco_split(release, train_json, video=True)
        val = coco_split(release, val_json, video=True)
        test = coco_split(release, test_json, video=True)
        provenance = {"train_json": sha256(train_json), "val_json": sha256(val_json),
                      "test_json": sha256(test_json), "xs_evaluator": sha256(XS_EVAL)}
    elif dataset == "visdrone":
        names = ["train", "val"] + (["test-dev"] if args.vis_eval == "test-dev" else [])
        with exclusive_lock(root / ".visdrone.lock"):
            ensure_visdrone(root, names, args.offline)
        splits = {name: visdrone_split(root / f"VisDrone2019-VID-{name}") for name in names}
        if args.vis_eval == "val":
            train, val = holdout(splits["train"])
        else:
            train, val = splits["train"], splits["val"]
        test = splits[args.vis_eval]
        provenance = {name: {p.name: sha256(p) for p in sorted((split.root / "annotations").glob("*.txt"))}
                      for name, split in splits.items()}
        provenance["vis_evaluator"] = Path("/opt/VisDrone-VID/REVISION").read_text().strip()
    elif dataset == "svanstrom":
        source = root / "annotations/test.json"
        train, val, test = svanstrom_splits(coco_split(root, source, video=True))
        provenance = {"source_json": sha256(source), "split": "ordered-class-blocks-v1"}
    elif dataset == "itsdt-15k":
        train_json = args.train_json or root / "annotations/instances_train2017.json"
        test_json = args.test_json or root / "annotations/instances_test2017.json"
        development = coco_split(root, train_json, video=True)
        provenance = {"train_json": sha256(train_json), "test_json": sha256(test_json)}
        if args.train_json is None:
            train_list = root / "coco_train_ITSDT.txt"
            development = itsdt_development(development, train_list)
            provenance["train_list"] = sha256(train_list)
        train, val = holdout(development)
        test = coco_split(root, test_json, video=True)
    else:
        video = dataset == "irdst-r"
        train_json = args.train_json or root / ("instances_train2017.json" if video else "annotations/trainval.json")
        test_json = args.test_json or root / ("instances_test2017.json" if video else "annotations/test.json")
        train, val = holdout(coco_split(root, train_json, video=video))
        test = coco_split(root, test_json, video=video)
        provenance = {"train_json": sha256(train_json), "test_json": sha256(test_json)}
        if not video:
            provenance["aitod_evaluator"] = sha256(AITOD_EVAL)
    check_disjoint(train, val, test, video=dataset != "aitod-v2")
    if args.limit:
        train, val, test = (subset(split, sorted(split.clips)[:args.limit]) for split in (train, val, test))
    return {"train": train, "val": val, "test": test}, provenance


def load_test_split(args):
    """Open only the evaluation split for checkpoint inference or external scoring."""
    dataset, root = args.dataset, args.data_root
    if dataset == "visdrone":
        name = args.vis_eval
        with exclusive_lock(root / ".visdrone.lock"):
            ensure_visdrone(root, [name], args.offline)
        test = visdrone_split(root / f"VisDrone2019-VID-{name}", limit=args.limit)
        provenance = {"test_annotations": {p.name: sha256(p)
                      for p in sorted((test.root / "annotations").glob("*.txt"))}}
    else:
        if dataset == "xsvid":
            root = root / "XS-VID-v2"
            default = root / "annotations/protocols/paper_detection_test.json"
        elif dataset == "irdst-r":
            default = root / "instances_test2017.json"
        elif dataset == "itsdt-15k":
            default = root / "annotations/instances_test2017.json"
        else:
            default = root / "annotations/test.json"
        annotation = args.test_json or default
        if args.stage == "predict":
            test = coco_split(root, annotation, video=dataset != "aitod-v2", limit=args.limit)
        else:
            # COCO scorers need annotations and predictions, so scoring can run without images.
            test = Split(root, json.loads(annotation.read_text()), {})
        provenance = {"test_json": str(annotation.resolve()), "test_sha256": sha256(annotation)}
    if args.limit:
        test = subset(test, sorted(test.clips)[:args.limit])
    return test, provenance


def timeline(split, frames, include_unlabelled):
    paths = [split.image_path(frame) for frame in frames]
    if not include_unlabelled:
        return paths, list(enumerate(frames))
    # Keep earlier numbered frames even when only selected frames are scored.
    folder = paths[0].parent
    if any(path.parent != folder for path in paths):
        raise ValueError(f"Numbered clip spans multiple folders: {folder}")
    all_paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.stem.isdecimal()),
                       key=lambda p: int(p.stem))
    positions = {path: index for index, path in enumerate(all_paths)}
    return all_paths, [(positions[path], frame) for path, frame in zip(paths, frames)]


def make_encoder(mode, compensate, args):
    if mode in ("rgb", "gray"):
        return None
    from yolo_antiuav import MotionEncoder

    gap = args.short_gap if mode == "short" else args.long_gap
    return MotionEncoder("diff", gap=gap, compensate=compensate)


def encoded_image(paths, index, mode, encoder):
    from PIL import Image

    if encoder is not None:
        return Image.fromarray(encoder(paths, index))
    with Image.open(paths[index]) as image:
        return (image.convert("L") if mode == "gray" else image).convert("RGB")


def mask_ignored(image, targets, ignored):
    from PIL import ImageDraw

    if not ignored:
        return image
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for left, top, right, bottom in ignored:
        draw.rectangle((math.floor(left), math.floor(top), math.ceil(right) - 1, math.ceil(bottom) - 1),
                       fill=(114, 114, 114))
    for _, left, top, right, bottom in targets:
        box = math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
        output.paste(image.crop(box), box[:2])
    return output


def prepare_view(root, splits, classes, mode, compensate, args):
    import yaml

    config = root / "dataset.yaml"
    if config.is_file():
        return config
    class_map = {category["id"]: number for number, category in enumerate(classes)}
    for name in ("train", "val"):
        split = splits[name]
        image_dir, label_dir = root / "images" / name, root / "labels" / name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        labels = defaultdict(list)
        for row in split.coco["annotations"]:
            labels[row["image_id"]].append(row)
        stride = args.train_stride if name == "train" else args.val_stride
        for frames in split.clips.values():
            paths, ordered = timeline(split, frames, args.dataset in NUMBERED_HISTORY_DATASETS)
            encoder = make_encoder(mode, compensate, args)
            for index, frame in ordered[::stride]:
                source = paths[index]
                width, height = frame["width"], frame["height"]
                targets, ignored = clip_boxes(labels[frame["id"]], class_map, width, height)
                stem = str(frame["id"])
                label = label_dir / f"{stem}.txt"
                # Apply the same training mask across all input ablations.
                ignored = ignored if name == "train" else []
                if mode == "rgb" and not ignored:
                    image = image_dir / (stem + source.suffix.lower())
                    if not image.is_symlink():
                        image.symlink_to(source)
                else:
                    image = image_dir / f"{stem}.jpg"
                    if not image.is_file():
                        pixels = mask_ignored(encoded_image(paths, index, mode, encoder), targets, ignored)
                        temporary = image.with_suffix(".tmp")
                        pixels.save(temporary, format="JPEG", quality=95, subsampling=0)
                        temporary.replace(image)
                lines = []
                for category, left, top, right, bottom in targets:
                    cx, cy = (left + right) / (2 * width), (top + bottom) / (2 * height)
                    lines.append(f"{category} {cx:.9f} {cy:.9f} {(right-left)/width:.9f} {(bottom-top)/height:.9f}\n")
                label.write_text("".join(lines))
        print(f"Prepared {name} for {mode}, aligned={compensate}", flush=True)
    temporary = config.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                         "names": [row["name"] for row in classes]}))
    temporary.replace(config)
    return config


@contextmanager
def box_loss(kind):
    from ultralytics.utils import loss, tal
    from yolo_antiuav import install_nwd

    original_loss, original_assign = loss.bbox_iou, tal.bbox_iou
    try:
        if kind == "relative":
            install_nwd(1.0)
        elif kind.endswith("-assign"):
            install_nwd(12.8 if kind == "fixed-assign" else 1.0,
                        assign_only=True, fixed_scale=kind == "fixed-assign")
        yield
    finally:
        loss.bbox_iou, tal.bbox_iou = original_loss, original_assign


def complete_nms():
    from ultralytics.utils import nms

    # Ultralytics can leave later frames in a batch empty when its NMS timer expires.
    return patch.object(nms, "non_max_suppression", partial(nms.non_max_suppression, max_time_img=math.inf))


def train(run_dir, config, kind, seed, args):
    checkpoint_record = run_dir / "checkpoint.json"
    if checkpoint_record.is_file():
        checkpoint = run_dir / json.loads(checkpoint_record.read_text())["weights"]
        if checkpoint.is_file():
            return checkpoint
        raise FileNotFoundError(checkpoint)

    from ultralytics import YOLO

    attempt = Path(tempfile.mkdtemp(prefix="train-", dir=run_dir))
    options = {}
    if args.finetune_lr is not None:
        options.update(optimizer="AdamW", lr0=args.finetune_lr, warmup_epochs=0.5,
                       warmup_bias_lr=0.0, nbs=16, mosaic=0.0)
    with box_loss(kind), complete_nms():
        model = YOLO(args.model)
        model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=args.workers, seed=seed, project=str(attempt), name="fit",
                    patience=0, exist_ok=True, hsv_h=0, hsv_s=0, hsv_v=0, amp=False, plots=False,
                    **options)
        checkpoint = Path(model.trainer.best)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training finished without a best checkpoint: {checkpoint}")
    write_json(checkpoint_record, {"weights": str(checkpoint.relative_to(run_dir))})
    return checkpoint


def predict_batch(model, sources, args):
    return model.predict(sources, imgsz=args.imgsz, device=args.device, conf=0.001, iou=0.6,
                         max_det=MAX_DET[args.dataset], rect=False, verbose=False)


def result_rows(image_id, result, category_ids):
    for box, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                    result.boxes.cls.int().cpu().tolist()):
        x, y, right, bottom = box
        category_id = category_ids[category]
        if category_id is not None and right > x and bottom > y:
            yield {"image_id": image_id, "category_id": category_id,
                   "bbox": [x, y, right - x, bottom - y], "score": score}


def prediction_rows(split, model, category_ids, mode, compensate, args, *, report_progress=False):
    total = len(split.clips)
    for number, frames in enumerate(split.clips.values(), 1):
        paths, ordered = timeline(split, frames, args.dataset in NUMBERED_HISTORY_DATASETS)
        encoder = make_encoder(mode, compensate, args)
        for start in range(0, len(ordered), args.predict_batch):
            batch = ordered[start:start + args.predict_batch]
            # PIL inputs are read as RGB by Ultralytics, like the cached training composites.
            sources = [str(paths[index]) if mode == "rgb" else encoded_image(paths, index, mode, encoder)
                       for index, _ in batch]
            results = predict_batch(model, sources, args)
            if len(results) != len(batch):
                raise RuntimeError("Detector returned the wrong number of frame results")
            for (_, frame), result in zip(batch, results):
                yield from result_rows(frame["id"], result, category_ids)
        if report_progress and (number % 10 == 0 or number == total):
            print(f"Predicted {number}/{total} clips", flush=True)


def predict(split, model, classes, mode, compensate, args):
    split_categories = {row["name"]: row["id"] for row in split.coco["categories"]}
    category_ids = [split_categories[row["name"]] for row in classes]
    return list(prediction_rows(split, model, category_ids, mode, compensate, args,
                                report_progress=args.dataset in TRANSFER_DATASETS))


def latency(split, model, mode, compensate, args):
    import torch

    frames = max(split.clips.values(), key=len)
    paths, ordered = timeline(split, frames, args.dataset == "irdst-r")
    encoder = make_encoder(mode, compensate, args)
    samples = []
    warmup = min(5, len(ordered) - 1)
    for number, (index, _) in enumerate(islice(ordered, warmup + args.timing_frames)):
        begin = time.perf_counter()
        source = str(paths[index]) if mode == "rgb" else encoded_image(paths, index, mode, encoder)
        result = predict_batch(model, [source], args)[0]
        result.boxes.data.cpu()
        if number >= warmup:
            samples.append(1000 * (time.perf_counter() - begin))
    device = model.predictor.device
    return {"batch1_ms": mean(samples), "timed_frames": len(samples),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)}


def run_model(run_dir, config, splits, classes, variant, seed, args):
    from ultralytics import YOLO

    mode, compensate, kind = variant
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = train(run_dir, config, kind, seed, args)
    model = None
    for name in ("val", "test"):
        directory = run_dir / name
        directory.mkdir(exist_ok=True)
        prediction_path, metrics_path = directory / "predictions.json", directory / "metrics.json"
        if not metrics_path.is_file():
            if prediction_path.is_file():
                predictions = json.loads(prediction_path.read_text())
            else:
                model = model or YOLO(str(checkpoint))
                with complete_nms():
                    predictions = predict(splits[name], model, classes, mode, compensate, args)
                write_json(prediction_path, predictions)
            write_json(metrics_path, score_split(splits[name], predictions, args.dataset, directory))
    cost_path = run_dir / "cost.json"
    if not cost_path.is_file():
        model = model or YOLO(str(checkpoint))
        write_json(cost_path, latency(splits["test"], model, mode, compensate, args))


def checkpoint_category_ids(names, categories, class_agnostic):
    ignored = {"ignore", "ignored", "others"}
    scored = {row["name"].strip().casefold(): row["id"] for row in categories
              if row["name"].strip().casefold() not in ignored}
    names = {int(index): name.strip().casefold() for index, name in names.items()}
    if not scored:
        raise ValueError("No scored object classes in the benchmark annotations")
    if class_agnostic:
        if len(names) != 1 or len(scored) != 1:
            raise ValueError("--class-agnostic requires one checkpoint class and one scored benchmark class")
        return {next(iter(names)): next(iter(scored.values()))}
    missing = scored.keys() - names.values()
    if missing:
        raise ValueError(f"Checkpoint is missing benchmark classes: {', '.join(sorted(missing))}. "
                         f"Checkpoint classes: {', '.join(names.values())}")
    return {index: scored.get(name) for index, name in names.items()}


def run_inference(args, variant):
    from ultralytics import YOLO

    mode, compensate, _ = variant
    weights = args.weights.resolve()
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError(f"Expected a YOLO detection checkpoint, found task {model.task}")
    split, provenance = load_test_split(args)
    category_ids = checkpoint_category_ids(model.names, split.coco["categories"], args.class_agnostic)
    output = args.out / "predictions.json"
    manifest = args.out / "inference.json"
    config = {"dataset": args.dataset, "vis_eval": args.vis_eval, "test": provenance,
              "weights": str(weights), "weights_sha256": sha256(weights),
              "data_root": str(split.root), "classes": model.names, "category_ids": category_ids,
              "class_agnostic": args.class_agnostic, "variant": args.variants[0], "device": args.device,
              "imgsz": args.imgsz, "predict_batch": args.predict_batch, "short_gap": args.short_gap,
              "long_gap": args.long_gap, "pilot_limit": args.limit, "clips": len(split.clips),
              "frames": len(split.coco["images"])}
    # Round trip once so integer dictionary keys compare equal to the cached JSON.
    config = json.loads(json.dumps(config))
    with exclusive_lock(args.out / ".predict.lock"):
        if manifest.is_file():
            prior = json.loads(manifest.read_text())
            if prior["configuration"] != config:
                raise ValueError(f"{args.out} already contains another inference run; use a fresh --out")
            if output.is_file():
                print(f"Reusing {output}", flush=True)
                return
        elif output.exists():
            raise FileExistsError(f"{output} already exists; use a fresh --out")
        write_json(manifest, {"configuration": config})
        print(f"Predicting {config['frames']} test frames with {args.device}", flush=True)
        start = time.perf_counter()
        count = 0
        with (
            complete_nms(),
            tempfile.NamedTemporaryFile(mode="w", dir=args.out, encoding="utf-8", delete=False) as f,
        ):
            temp = Path(f.name)
            try:
                f.write("[")
                for row in prediction_rows(split, model, category_ids, mode, compensate, args, report_progress=True):
                    if count:
                        f.write(",\n")
                    json.dump(row, f, allow_nan=False)
                    count += 1
                f.write("]\n")
            except BaseException:
                temp.unlink()
                raise
        temp.replace(output)
        write_json(manifest, {"configuration": config, "boxes": count,
                              "elapsed_seconds": time.perf_counter() - start})
        print(f"Wrote {count} boxes to {output}", flush=True)


def run_external_score(args):
    split, provenance = load_test_split(args)
    prediction_manifest = args.predictions.with_name("inference.json")
    if args.predictions.name == "predictions.json" and prediction_manifest.is_file():
        config = json.loads(prediction_manifest.read_text())["configuration"]
        if config["pilot_limit"] is not None:
            raise ValueError("Pilot predictions cover a partial split. Rerun without --limit before scoring.")
        same_test = (config["test"].get("test_sha256") == provenance.get("test_sha256")
                     if "test_sha256" in provenance else config["test"] == provenance)
        if (config["dataset"] != args.dataset or not same_test
                or args.dataset == "visdrone" and config["vis_eval"] != args.vis_eval):
            raise ValueError("Predictions were generated for a different evaluation split.")
    else:
        config = None

    predictions = json.loads(args.predictions.read_text())
    evaluator_path = args.coco_evaluator
    if evaluator_path is None:
        evaluator_path = {"xsvid": XS_EVAL, "visdrone": XS_EVAL, "aitod-v2": AITOD_EVAL}.get(args.dataset)
    if evaluator_path is not None:
        if not evaluator_path.is_file():
            raise FileNotFoundError(f"Evaluator not found at {evaluator_path}; pass --coco-evaluator")
        provenance["coco_evaluator_sha256"] = sha256(evaluator_path)
    key = hashlib.sha256(json.dumps([sha256(args.predictions), provenance, args.vis_eval],
                                     sort_keys=True).encode()).hexdigest()[:12]
    directory = args.out / args.dataset / "external" / key
    directory.mkdir(parents=True, exist_ok=True)
    metrics = score_split(split, predictions, args.dataset, directory, coco_evaluator=args.coco_evaluator,
                          vis_toolkit=args.vis_toolkit)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "manifest.json", {"dataset": args.dataset, "provenance": provenance,
               "vis_eval": args.vis_eval, "predictions": str(args.predictions.resolve()),
               "prediction_configuration": config})
    print(f"Metrics: {directory / 'metrics.json'}", flush=True)
    print(metrics)


def display_value(value):
    return f"{value:.2f}" if value is not None else "—"


def summarize(suite, dataset, limit):
    metrics = {
        "xsvid": ("AP", "AP50", "AP75", "AP_es", "AP_rs", "AP_gs"),
        "visdrone": ("AP", "AP50", "AP75", "diagnostic_AP_es", "diagnostic_AP_rs", "diagnostic_AP_gs"),
        "irdst-r": ("AP", "AP50", "AP75", "AP_small", "F1_CHAL"),
        "aitod-v2": ("AP", "AP50", "AP75", "AP_vt", "AP_t", "AP_small"),
        "svanstrom": COCO_METRICS, "itsdt-15k": COCO_METRICS,
    }[dataset]
    primary = {"xsvid": "AP_es", "visdrone": "diagnostic_AP_es",
               "irdst-r": "AP50", "aitod-v2": "AP_vt",
               "svanstrom": "AP50", "itsdt-15k": "AP50"}[dataset]
    columns = ("Variant", "Seeds", *metrics, f"Δ{primary} vs RGB", "ms/frame")
    lines = [f"# {dataset} test results" + (" (pilot subset)" if limit else ""),
             "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    results = defaultdict(dict)
    for path in sorted(suite.glob("*/seed-*")):
        score_file, cost_file = path / "test/metrics.json", path / "cost.json"
        if score_file.is_file() and cost_file.is_file():
            score = json.loads(score_file.read_text())
            score["ms/frame"] = json.loads(cost_file.read_text())["batch1_ms"]
            results[path.parent.name][path.name] = score
    for variant, rows in results.items():
        cells = [variant, str(len(rows))]
        for metric in metrics:
            values = [row[metric] for row in rows.values() if row[metric] is not None]
            cell = display_value(mean(values) if values else None)
            if len(values) > 1:
                cell += f" ± {stdev(values):.2f}"
            cells.append(cell)
        rgb = results.get("rgb", {})
        differences = [row[primary] - rgb[seed][primary] for seed, row in rows.items()
                       if seed in rgb and row[primary] is not None and rgb[seed][primary] is not None]
        cells.append(f"{mean(differences):+.2f} ({len(differences)})" if differences else "—")
        cells.append(display_value(mean(row["ms/frame"] for row in rows.values())))
        lines.append("| " + " | ".join(cells) + " |")
    if dataset == "xsvid" and not limit:
        lines += ["", "Published reference scores on the XS-VID paper Detection protocol.",
                  "", "| Method | AP | AP_es |", "| --- | --- | --- |"]
        for name, ap, extra_small in XS_REFERENCES:
            lines.append(f"| {name} | {ap:.1f} | {extra_small:.1f} |")
        lines += ["", "[XS-VID paper, Table III](https://arxiv.org/html/2407.18137)"]
    summary = "\n".join(lines) + "\n"
    (suite / "SUMMARY.md").write_text(summary)
    print(summary, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MAX_DET), default="xsvid")
    parser.add_argument("--stage", choices=("data", "prepare", "run", "predict", "score"), default="run")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/runs/small_objects"))
    parser.add_argument("--train-json", type=Path, help="COCO training labels for IRDST-R or AI-TOD-v2")
    parser.add_argument("--test-json", type=Path, help="Override the COCO test annotations")
    parser.add_argument("--xs-hub", choices=("hf", "ms"), default="hf")
    parser.add_argument("--xs-revision", help="Hub revision for a fresh XS-VID download")
    parser.add_argument("--vis-eval", choices=("val", "test-dev"), default="test-dev")
    parser.add_argument("--offline", action="store_true", help="Reuse staged dataset files without downloading data")
    parser.add_argument("--predictions", type=Path, help="COCO prediction JSON for --stage score")
    parser.add_argument("--variants", nargs="+", help="Run only the named input/loss variants")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO checkpoint or model YAML")
    parser.add_argument("--device", help="CPU, Apple MPS, or one CUDA index")
    parser.add_argument("--weights", type=Path, help="Local YOLO detection .pt checkpoint for --stage predict")
    parser.add_argument("--class-agnostic", action="store_true",
                        help="Zero-shot prediction when checkpoint and benchmark each have one class")
    parser.add_argument("--coco-evaluator", type=Path, help="Upstream COCO evaluator source for native scoring")
    parser.add_argument("--vis-toolkit", type=Path, help="Upstream VisDrone VID toolkit folder for native scoring")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--finetune-lr", type=float, help="AdamW learning rate for a local starting checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--val-stride", type=int, default=4)
    parser.add_argument("--predict-batch", type=int)
    parser.add_argument("--short-gap", type=int, default=2)
    parser.add_argument("--long-gap", type=int, default=12)
    parser.add_argument("--timing-frames", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Pilot only: keep this many clips/images per split")
    args = parser.parse_args()
    if args.dataset == "svanstrom" and args.stage in ("data", "prepare", "run") and (args.train_json or args.test_json):
        parser.error("Svanström training uses the prepared release and its built-in clip split")
    if args.finetune_lr is not None and (
        not math.isfinite(args.finetune_lr) or args.finetune_lr <= 0 or not Path(args.model).is_file()
        or Path(args.model).suffix.lower() != ".pt"
    ):
        parser.error("--finetune-lr requires a positive rate and a local --model .pt checkpoint")
    if args.device is None:
        args.device = "cpu" if args.stage == "predict" else "0"
    if args.predict_batch is None:
        args.predict_batch = 1 if args.stage == "predict" else 16
    for name in ("epochs", "imgsz", "batch", "train_stride", "val_stride", "predict_batch", "timing_frames"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.limit is not None and args.limit < 1 or not 1 < args.short_gap < args.long_gap:
        parser.error("Require workers >= 0, positive limit, and 1 < short-gap < long-gap")
    if args.device not in ("cpu", "mps") and not args.device.isdecimal():
        parser.error("--device must be cpu, mps, or one CUDA index")
    if args.stage == "predict":
        if args.weights is None or args.weights.suffix.lower() != ".pt" or not args.weights.is_file():
            parser.error("--stage predict requires --weights pointing to a local YOLO .pt checkpoint")
        if args.variants is None or len(args.variants) != 1:
            parser.error("--stage predict requires exactly one --variants value matching the checkpoint input")
    elif args.weights is not None or args.class_agnostic:
        parser.error("--weights and --class-agnostic are only available with --stage predict")
    if args.stage == "score" and (
        args.predictions is None or not args.predictions.is_file() or args.limit is not None
    ):
        parser.error("--stage score requires an existing --predictions JSON and the full test split")
    args.data_root, args.out = args.data_root.resolve(), args.out.resolve()
    return args


def main():
    args = parse_args()
    dataset = args.dataset
    variants = IMAGE_VARIANTS if dataset == "aitod-v2" else VIDEO_VARIANTS
    default = {
        "xsvid": tuple(VIDEO_VARIANTS)[:-1], "visdrone": ("rgb", "aligned-long"),
        "irdst-r": ("rgb", "aligned-long"), "aitod-v2": tuple(IMAGE_VARIANTS),
        "svanstrom": ("aligned-long-nwd",), "itsdt-15k": ("aligned-long-nwd",),
    }[dataset]
    chosen = args.variants or default
    if any(name not in variants for name in chosen):
        raise SystemExit(f"Available variants for {dataset}: {', '.join(variants)}")
    if args.stage == "predict":
        run_inference(args, variants[chosen[0]])
        return
    if dataset == "aitod-v2":
        args.train_stride = args.val_stride = 1  # independent still images
    with exclusive_lock(args.out / f".{dataset}.lock"):
        if args.stage == "score":
            run_external_score(args)
            return
        splits, provenance = load_splits(args)
        if args.stage == "data":
            for name, split in splits.items():
                print(f"{name}: {len(split.clips)} clips/images, {len(split.coco['images'])} frames")
            return

        classes = sorted((row for row in splits["train"].coco["categories"]
                          if row["name"].lower() not in ("ignore", "ignored", "others")), key=lambda row: row["id"])
        if not classes:
            raise ValueError("No trainable object categories")
        settings = {name: getattr(args, name) for name in (
            "model", "epochs", "imgsz", "batch", "device", "train_stride", "val_stride", "short_gap",
            "long_gap", "limit", "vis_eval", "predict_batch", "timing_frames", "workers",
        )}
        if args.finetune_lr is not None:
            settings["finetune_lr"] = args.finetune_lr
        settings.update({"dataset": dataset, "provenance": provenance, "data_root": str(args.data_root),
                         "fps": {name: sorted({video["fps"] for video in split.coco.get("videos", [])
                                               if isinstance(video.get("fps"), (int, float))})
                                 for name, split in splits.items()},
                         "code": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("small_object_*.py"))},
                         "motion_encoder": sha256(Path(__file__).with_name("yolo_antiuav.py")),
                         "splits": {name: sorted(split.clips) for name, split in splits.items()},
                         "classes": classes})
        weights = Path(args.model)
        if weights.parent != Path(".") and weights.is_file():
            settings["model_sha256"] = sha256(weights)
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        suite = args.out / dataset / key
        suite.mkdir(parents=True, exist_ok=True)
        write_json(suite / "manifest.json", settings)
        if dataset in TRANSFER_DATASETS:
            write_json(suite / "test_annotations.json", splits["test"].coco)
        print(f"Output: {suite}", flush=True)
        for name in chosen:
            mode, compensate, _ = variants[name]
            view = suite / "views" / f"{mode}-{'aligned' if compensate else 'raw'}"
            config = prepare_view(view, splits, classes, mode, compensate, args)
            if args.stage == "run":
                for seed in args.seeds:
                    run_model(suite / name / f"seed-{seed}", config, splits, classes, variants[name], seed, args)
                summarize(suite, dataset, args.limit)


if __name__ == "__main__":
    main()
