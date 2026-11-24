"""MMDetection baselines on Anti-UAV410 (DINO-4scale R50, Deformable-DETR R50), scored with the DRISHTI protocol.

Same role as yolo_antiuav.py, for the DETR rows of Gaze-DETR's comparison table:
  export   COCO json of train (every --train-stride-th frame) and val (every --val-stride-th) via export_coco.py
  train    fine-tune from the COCO-pretrained checkpoint of the MMDetection model zoo (URL read from the
           installed package's metafile), 1 class, input resized to --imgsz on the long side
  predict  every frame of every val and test sequence -> <out>/raw_{split}/<seq>.json (normalized cx,cy,w,h,
           top --max-det boxes at score >= 0.001), the format all our scoring scripts read
  score    val-chosen post-processing applied to test (scripts/sweep_postprocess.py) + raw test score
  cost     parameters and batch-1 latency

Usage (inside the drishti-mmdet image, from the code folder):
  python scripts/baselines/mmdet_antiuav.py --stage all --model dino --frames-root /workspace/data/antiuav410_frames \
      --coco-root /workspace/data/coco_a410 --out /workspace/runs/baseline_dino_r50_512
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# the MMDetection image mixes Intel MKL and GNU OpenMP; without this the scoring subprocess aborts (20 Oct)
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "baselines"))

MODELS = {
    "dino": ("dino", "dino-4scale_r50_8xb2-12e_coco"),
    "deformable_detr": ("deformable_detr", "deformable-detr_r50_16xb2-50e_coco"),
}


def mmdet_config_dir() -> Path:
    import mmdet

    root = Path(mmdet.__file__).parent / ".mim" / "configs"
    if not root.exists():
        raise SystemExit(f"MMDetection configs not found at {root}")
    return root


def checkpoint_url(model: str) -> str:
    import yaml

    folder, name = MODELS[model]
    meta = yaml.safe_load((mmdet_config_dir() / folder / "metafile.yml").read_text(encoding="utf-8"))
    for entry in meta.get("Models", []):
        if entry.get("Name") == name:
            return entry["Weights"]
    raise SystemExit(f"no weights for {name} in the metafile")


def export(args) -> None:
    coco = Path(args.coco_root)
    if (coco / f"train_{args.modality}.json").exists() and (coco / f"val_{args.modality}.json").exists():
        print("COCO export exists; skipping")
        return
    coco.mkdir(parents=True, exist_ok=True)
    for split, stride in (("train", args.train_stride), ("val", args.val_stride)):
        subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "export_coco.py"), "--frames-root", args.frames_root,
                        "--modality", args.modality, "--out", str(coco), "--splits", split, "--train-stride", str(stride)], check=True)
    print(f"exported -> {coco}: {sorted(p.name for p in coco.glob('*.json'))}")


def build_config(args):
    from mmengine.config import Config

    folder, name = MODELS[args.model]
    cfg = Config.fromfile(str(mmdet_config_dir() / folder / f"{name}.py"))
    out = Path(args.out)
    w, h = args.imgsz, int(round(args.imgsz * 0.8))  # Anti-UAV410 frames are 640x512
    train_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="Resize", scale=(w, h), keep_ratio=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PackDetInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="Resize", scale=(w, h), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
    ]
    meta = dict(classes=("uav",), palette=[(220, 20, 60)])
    coco = Path(args.coco_root)
    for key, ann, pipe, shuffle in (("train_dataloader", f"train_{args.modality}.json", train_pipeline, True),
                                    ("val_dataloader", f"val_{args.modality}.json", test_pipeline, False)):
        dl = cfg[key]
        dl.batch_size = args.batch if key == "train_dataloader" else 8
        dl.num_workers = args.workers
        ds = dl.dataset
        ds.data_root = args.frames_root.rstrip("/") + "/"
        ds.ann_file = str(coco / ann)
        ds.data_prefix = dict(img="")
        ds.metainfo = meta
        ds.pipeline = pipe
        if key == "train_dataloader":
            ds.filter_cfg = dict(filter_empty_gt=False, min_size=1)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(type="CocoMetric", ann_file=str(coco / f"val_{args.modality}.json"), metric="bbox", format_only=False)
    cfg.test_evaluator = cfg.val_evaluator
    cfg.model.bbox_head.num_classes = 1
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = max(args.epochs // 3, 1)
    if "param_scheduler" in cfg and isinstance(cfg.param_scheduler, list):
        for s in cfg.param_scheduler:
            if s.get("type") == "MultiStepLR":
                s.end = args.epochs
                s.milestones = [max(int(args.epochs * 0.75), 1)]
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
    cfg.load_from = checkpoint_url(args.model)
    cfg.work_dir = str(out / "train")
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="coco/bbox_mAP_50", rule="greater")
    cfg.default_hooks.logger = dict(type="LoggerHook", interval=200)
    cfg.randomness = dict(seed=args.seed)
    cfg.visualizer.vis_backends = [dict(type="LocalVisBackend")]
    return cfg


def best_checkpoint(out: Path) -> Path:
    found = sorted((out / "train").glob("best_coco_bbox_mAP_50_epoch_*.pth"))
    if not found:
        found = sorted((out / "train").glob("epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if not found:
        raise SystemExit("no checkpoint found; train first")
    return found[-1]


def train(args) -> None:
    from mmengine.runner import Runner

    cfg = build_config(args)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    cfg.dump(str(Path(args.out) / "config.py"))
    runner = Runner.from_cfg(cfg)
    runner.train()
    print(f"best checkpoint: {best_checkpoint(Path(args.out))}")


def load_model(args):
    from mmdet.apis import init_detector

    cfg_path = Path(args.out) / "config.py"
    if not cfg_path.exists():
        build_config(args).dump(str(cfg_path))
    return init_detector(str(cfg_path), str(best_checkpoint(Path(args.out))), device="cuda:0")


def predict(args) -> None:
    import torch
    from mmdet.apis import inference_detector
    from PIL import Image
    from yolo_antiuav import sequences

    model = load_model(args)
    out = Path(args.out)
    for split in [s.strip() for s in args.predict_splits.split(",") if s.strip()]:
        raw_dir = out / f"raw_{split}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for k, (name, frames, _) in enumerate(sequences(Path(args.frames_root), split, args.modality)):
            if args.max_seqs is not None and k >= args.max_seqs:
                break
            target = raw_dir / f"{name}.json"
            if target.exists():
                continue
            with Image.open(frames[0]) as im:
                width, height = im.size
            result = {}
            for start in range(0, len(frames), args.pred_batch):
                batch = [str(f) for f in frames[start : start + args.pred_batch]]
                with torch.no_grad():
                    outs = inference_detector(model, batch)
                for j, o in enumerate(outs):
                    inst = o.pred_instances
                    keep = inst.scores >= 0.001
                    scores, boxes = inst.scores[keep], inst.bboxes[keep]
                    order = scores.argsort(descending=True)[: args.max_det]
                    scores, boxes = scores[order].cpu(), boxes[order].cpu()
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
                    bw = (boxes[:, 2] - boxes[:, 0]) / width
                    bh = (boxes[:, 3] - boxes[:, 1]) / height
                    result[str(start + j)] = {"boxes": torch.stack([cx, cy, bw, bh], 1).tolist(), "scores": scores.tolist()}
            target.write_text(json.dumps({"frames": result}), encoding="utf-8")
        print(f"predictions for {split} -> {raw_dir}", flush=True)


def score(args) -> None:
    out = Path(args.out)
    py = sys.executable
    subprocess.run([py, str(CODE_ROOT / "scripts" / "score_predictions.py"), "--frames-root", args.frames_root, "--split", "test",
                    "--modality", args.modality, "--pred-dir", str(out / "raw_test"), "--name", args.model,
                    "--normalized", "--out", str(out / "score_test_raw.json")], check=True)
    subprocess.run([py, str(CODE_ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
                    "--modality", args.modality, "--val-raw", str(out / "raw_val"), "--test-raw", str(out / "raw_test"),
                    "--out", str(out / "postprocess_sweep.json")], check=True)


def cost(args) -> None:
    import torch
    from mmdet.apis import inference_detector
    from yolo_antiuav import sequences

    model = load_model(args)
    params = sum(p.numel() for p in model.parameters())
    frames = next(sequences(Path(args.frames_root), "test", args.modality))[1][:120]
    for f in frames[:20]:
        inference_detector(model, str(f))
    torch.cuda.synchronize()
    started = time.perf_counter()
    for f in frames[20:]:
        inference_detector(model, str(f))
    torch.cuda.synchronize()
    ms = (time.perf_counter() - started) / max(len(frames) - 20, 1) * 1000
    gflops = None
    try:
        from mmengine.analysis import get_model_complexity_info

        w, h = args.imgsz, int(round(args.imgsz * 0.8))
        gflops = get_model_complexity_info(model, input_shape=(3, h, w))["flops"] / 1e9
    except Exception as exc:  # FLOP counting of deformable attention is not always supported
        print(f"GFLOPs not computed: {exc}")
    info = {"model": args.model, "imgsz": args.imgsz, "params": params, "gflops": gflops, "latency_ms_batch1": ms, "fps": 1000 / ms}
    (Path(args.out) / "cost.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


def main() -> None:
    parser = argparse.ArgumentParser(description="MMDetection baselines on Anti-UAV410, scored with the DRISHTI protocol.")
    parser.add_argument("--stage", default="all", choices=["export", "train", "predict", "score", "cost", "all"])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--coco-root", required=True, help="COCO export folder (shared by the MMDetection runs)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640, help="long side (512 x 640 frames are kept at native size)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--val-stride", type=int, default=10)
    parser.add_argument("--predict-splits", default="val,test")
    parser.add_argument("--pred-batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--max-seqs", type=int, default=None, help="smoke test: first N sequences per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stages = ["export", "train", "predict", "score", "cost"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if args.stage == "all" and stage == "train":
            try:
                best_checkpoint(Path(args.out))
                print("trained checkpoint exists; skipping training")
                continue
            except SystemExit:
                pass
        print(f"===== {stage}", flush=True)
        globals()[stage](args)
    print("ALL DONE")


if __name__ == "__main__":
    main()
