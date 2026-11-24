"""Benchmark-native detection scores and reference rows."""
from __future__ import annotations

import importlib.util
import subprocess
from collections import defaultdict
from pathlib import Path

from small_object_data import Split, YOLOFT

XS_EVAL = YOLOFT / "ultralytics/data/cocoeval_xs_vid.py"
AITOD_EVAL = Path("/opt/aitod/cocoeval.py")
VIS_EVAL = Path(__file__).resolve().parents[2] / "dgx/score_visdrone_vid.m"
VIS_TOOLKIT = Path("/opt/VisDrone-VID")
XS_BINS = {"AP_es": "0-12", "AP_rs": "12-20", "AP_gs": "20-32"}


def evaluator(source: Path):
    if not source.is_file():
        raise FileNotFoundError(f"Evaluator not found at {source}; pass --coco-evaluator with its source file")
    # Both upstream evaluators use a relative import for the COCO mask API.
    spec = importlib.util.spec_from_file_location(f"pycocotools._drishti_{source.stem}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COCOeval


def score_coco(ground_truth: dict, predictions: list[dict], protocol: str,
               coco_evaluator: Path | None = None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = {"info": {}, **ground_truth}
    gt.createIndex()
    if predictions:
        det = gt.loadRes(predictions)
    else:
        det = COCO()
        det.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"],
                       "annotations": []}
        det.createIndex()

    if protocol == "xsvid":
        COCOeval = evaluator(coco_evaluator or XS_EVAL)
    elif protocol == "aitod-v2":
        COCOeval = evaluator(coco_evaluator or AITOD_EVAL)
    result = COCOeval(gt, det, "bbox")
    result.evaluate()
    result.accumulate()

    precision = result.eval["precision"]
    bins = {**XS_BINS, "AP_small": "small"} if protocol == "xsvid" else {"AP_small": "small"}
    if protocol == "aitod-v2":
        bins = {"AP_vt": "verytiny", "AP_t": "tiny", "AP_small": "small", "AP_medium": "medium"}

    def ap(area="all", iou=None):
        area_index = result.params.areaRngLbl.index(area)
        iou_index = slice(None) if iou is None else min(
            range(len(result.params.iouThrs)), key=lambda i: abs(result.params.iouThrs[i] - iou))
        values = precision[iou_index, :, :, area_index, -1]
        valid = values[values >= 0]
        return float(valid.mean() * 100) if valid.size else None

    metrics = {"AP": ap(), "AP50": ap(iou=0.5), "AP75": ap(iou=0.75),
               **{metric: ap(area) for metric, area in bins.items()}}
    if protocol == "irdst-r":
        # Match CHAL's reported statistic over the reachable part of the AP50 curve.
        recall = max(0.0, float(result.eval["recall"][0, 0, 0, -1]))
        reachable = precision[0, :int(recall * 100), 0, 0, -1]
        average_precision = float(reachable.mean()) if reachable.size else 0.0
        f1 = 2 * average_precision * recall / (average_precision + recall) if average_precision + recall else 0.0
        metrics.update(precision_CHAL=100 * average_precision, recall_CHAL=100 * recall, F1_CHAL=100 * f1)
    return metrics


def visdrone_predictions(split: Split, predictions: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for row in predictions:
        by_image[row["image_id"]].append(row)
    for clip, frames in split.clips.items():
        with (directory / f"{clip}.txt").open("w") as f:
            for image in frames:
                for row in sorted(by_image[image["id"]], key=lambda item: -item["score"]):
                    x, y, w, h = row["bbox"]
                    f.write(f'{image["_frame"]},-1,{x:.6f},{y:.6f},{w:.6f},{h:.6f},'
                            f'{row["score"]:.8f},{row["category_id"]},-1,-1\n')


def score_visdrone(split: Split, predictions: list[dict], directory: Path, toolkit: Path) -> dict:
    if not (toolkit / "calcAccuracy.m").is_file():
        raise FileNotFoundError(f"VisDrone VID toolkit not found at {toolkit}; pass --vis-toolkit")
    submission = directory / "visdrone_submission"
    visdrone_predictions(split, predictions, submission)
    clips = directory / "visdrone_clips.txt"
    clips.write_text("".join(f"{clip}\n" for clip in split.clips))
    output = directory / "visdrone_scores.txt"
    subprocess.run(["octave-cli", "--quiet", str(VIS_EVAL), str(split.root), str(submission),
                    str(clips), str(output), str(toolkit.resolve())], check=True)
    ap, ap50, ap75, ar1, ar10, ar100, ar500 = map(float, output.read_text().split())
    return {"AP": ap, "AP50": ap50, "AP75": ap75,
            "AR1": ar1, "AR10": ar10, "AR100": ar100, "AR500": ar500}


def visdrone_size_diagnostic(ground_truth: dict, predictions: list[dict],
                             coco_evaluator: Path | None = None) -> dict:
    """COCO-style size AP with class-agnostic ignore regions mapped to every class."""
    from copy import deepcopy

    data = deepcopy(ground_truth)
    rows = data["annotations"]
    annotations = [row for row in rows if row["category_id"] != 0]
    next_id = max((row["id"] for row in rows), default=0) + 1
    for row in rows:
        if row["category_id"] == 0:
            for category in data["categories"]:
                annotations.append({**row, "id": next_id, "category_id": category["id"]})
                next_id += 1
    data["annotations"] = annotations
    scores = score_coco(data, predictions, "xsvid", coco_evaluator)
    return {f"diagnostic_{name}": scores[name] for name in (*XS_BINS, "AP_small")}


def score_split(split: Split, predictions: list[dict], dataset: str, directory: Path, *,
                coco_evaluator: Path | None = None, vis_toolkit: Path | None = None) -> dict:
    if dataset == "visdrone":
        return {**score_visdrone(split, predictions, directory, vis_toolkit or VIS_TOOLKIT),
                **visdrone_size_diagnostic(split.coco, predictions, coco_evaluator)}
    return score_coco(split.coco, predictions, dataset, coco_evaluator)
