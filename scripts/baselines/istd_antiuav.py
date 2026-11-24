"""CHAL (CVPR'26) and SSTNet (TGRS'24) on Anti-UAV410, scored with the DRISHTI protocol.

Both are 5-frame moving infrared small-target detectors from UESTC-nnLab, written in the same code style.
Their pretrained weights are only on Baidu Pan, so both are trained from scratch on Anti-UAV410 train.
  prepare  a frame view in their naming convention: <view>/<split>/<seq>/<i>.bmp (symlinks to our JPEGs,
           0-based, unpadded; PIL reads by content, so the .bmp name is harmless) and annotation files
           "<path> x1,y1,x2,y2,0" for every --train-stride-th train frame and --val-stride-th val frame
  train    their own training script, run from a private copy with these settings replaced by name:
           pretrained weights none, epochs, batch size, workers, 512x512 input, in-training evaluation off
  predict  our wrapper around their model and their decode/NMS code: every frame of every val and test
           sequence (5 frames t-4..t, clamped at the first frame, as in their loader) ->
           <out>/raw_{split}/<seq>.json in normalized cx,cy,w,h (top --max-det boxes at score >= 0.001)
  score    val-chosen post-processing applied to test (scripts/sweep_postprocess.py), same as every other row
  cost     parameters and batch-1 latency

Usage (inside the drishti-istd image, from the code folder):
  python scripts/baselines/istd_antiuav.py --stage all --repo chal --frames-root /workspace/data/antiuav410_frames \
      --view /workspace/data/istd_a410 --out /workspace/runs/baseline_chal
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "baselines"))

REPOS = {
    # repo: (source folder, training script, model module, model class)
    "chal": ("/opt/CHAL", "train.py", "nets.slowfastnet", "slowfastnet"),
    "sstnet": ("/opt/SSTNet", "train_IRDST.py", "nets.Network", "Network"),
}


def prepare(args) -> None:
    from drishti_v2.data.dataset import _box_exists, _candidate_boxes, _read_antiuav_json
    from PIL import Image
    from yolo_antiuav import sequences

    view = Path(args.view)
    if (view / "train.txt").exists() and (view / "val.txt").exists():
        print("view exists; skipping")
        return
    for split, stride in (("train", args.train_stride), ("val", args.val_stride)):
        lines = []
        for name, frames, label in sequences(Path(args.frames_root), split, args.modality):
            seq_dir = view / split / name
            seq_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                link = seq_dir / f"{i}.bmp"
                if not link.exists():
                    os.symlink(f.resolve(), link)
            if stride <= 0:
                continue
            boxes, exists = _read_antiuav_json(label)
            for i in range(0, len(frames), stride):
                raw = boxes[i] if i < len(boxes) else []
                present = exists[i] if i < len(exists) else _box_exists(raw)
                tokens = [str(seq_dir / f"{i}.bmp")]
                if present:
                    for x, y, w, h in _candidate_boxes(raw):
                        if w > 1 and h > 1:
                            tokens.append(f"{int(round(x))},{int(round(y))},{int(round(x + w))},{int(round(y + h))},0")
                lines.append(" ".join(tokens))
        if stride > 0:
            (view / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{split}: {len(lines)} samples -> {view / f'{split}.txt'}")


def patched_copy(args) -> Path:
    """A private copy of the repository with the training settings replaced (the original stays untouched)."""
    src, script, _, _ = REPOS[args.repo]
    work = Path(args.out) / "repo"
    if not work.exists():
        shutil.copytree(src, work, symlinks=True)
    text = (Path(src) / script).read_text(encoding="utf-8")
    main_at = text.find("if __name__")
    head, body = text[:main_at], text[main_at:]
    view = Path(args.view)
    settings = {
        "model_path": "''",
        "input_shape": "[512, 512]",
        "Init_Epoch": "0",
        "Freeze_Epoch": str(args.epochs),
        "UnFreeze_Epoch": str(args.epochs),
        "Freeze_batch_size": str(args.batch),
        "Unfreeze_batch_size": str(args.batch),
        "Freeze_Train": "False",
        "num_workers": str(args.workers),
        "eval_flag": "False",
        "save_period": str(max(args.epochs // 3, 1)),
        "save_dir": repr(str(Path(args.out) / "logs")),
        "train_annotation_path": repr(str(view / "train.txt")),
        "val_annotation_path": repr(str(view / "val.txt")),
        "DATA_PATH": repr(str(view) + "/"),
    }
    for name, value in settings.items():
        body, n = re.subn(rf"^(\s*){name}\s*=.*$", rf"\g<1>{name} = {value}", body, count=1, flags=re.M)
        print(f"  {name:22s} {'set to ' + value if n else '(not in script)'}")
    (work / script).write_text(head + body, encoding="utf-8")
    return work


def train(args) -> None:
    work = patched_copy(args)
    (Path(args.out) / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CHAL_TRAIN_ANNOTATION=str(Path(args.view) / "train.txt"), CHAL_VAL_ANNOTATION=str(Path(args.view) / "val.txt"))
    subprocess.run([sys.executable, REPOS[args.repo][1]], cwd=str(work), env=env, check=True)
    print(f"weights: {weights(args)}")


def weights(args) -> Path:
    logs = Path(args.out) / "logs"
    for name in ("best_epoch_weights.pth", "last_epoch_weights.pth"):  # they may sit in a dated subfolder
        found = sorted(logs.rglob(name), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    found = sorted(logs.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no weights in {logs}; train first")
    return found[-1]


def load_model(args):
    import importlib

    import torch

    work = Path(args.out) / "repo"
    if not work.exists():
        work = Path(REPOS[args.repo][0])
    sys.path.insert(0, str(work))
    os.chdir(work)
    _, _, module, cls = REPOS[args.repo]
    net = getattr(importlib.import_module(module), cls)(1, num_frame=5)
    net.load_state_dict(torch.load(str(weights(args)), map_location="cpu"))
    bbox = importlib.import_module("utils.utils_bbox")
    utils = importlib.import_module("utils.utils")
    return net.cuda().eval(), bbox, utils


def predict(args) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from yolo_antiuav import sequences

    frames_root = Path(args.frames_root).resolve()
    out = Path(args.out).resolve()
    net, bbox, utils = load_model(args)
    size = (512, 512)
    for split in [s.strip() for s in args.predict_splits.split(",") if s.strip()]:
        raw_dir = out / f"raw_{split}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for k, (name, frames, _) in enumerate(sequences(frames_root, split, args.modality)):
            if args.max_seqs is not None and k >= args.max_seqs:
                break
            target = raw_dir / f"{name}.json"
            if target.exists():
                continue
            with Image.open(frames[0]) as im:
                width, height = im.size
            cache: dict[int, np.ndarray] = {}

            def prepped(i: int) -> np.ndarray:
                i = max(i, 0)
                if i not in cache:
                    with Image.open(frames[i]) as im:
                        img = utils.resize_image(utils.cvtColor(im), size, args.letterbox)
                    cache[i] = np.transpose(utils.preprocess_input(np.array(img, dtype="float32")), (2, 0, 1))
                    for old in [j for j in cache if j < i - 8]:
                        del cache[old]
                return cache[i]

            result = {}
            for start in range(0, len(frames), args.pred_batch):
                idx = list(range(start, min(start + args.pred_batch, len(frames))))
                batch = np.stack([np.stack([prepped(t - d) for d in range(4, -1, -1)], axis=1) for t in idx])  # B,3,5,H,W
                with torch.no_grad():
                    outputs = net(torch.from_numpy(batch).cuda())
                    outputs = bbox.decode_outputs(outputs, list(size))
                    outputs = bbox.non_max_suppression(outputs, 1, list(size), np.array([height, width]), args.letterbox,
                                                       conf_thres=0.001, nms_thres=0.65)
                for t, o in zip(idx, outputs):
                    if o is None or len(o) == 0:
                        result[str(t)] = {"boxes": [], "scores": []}
                        continue
                    o = np.asarray(o.cpu() if hasattr(o, "cpu") else o, dtype=np.float32)
                    scores = o[:, 4] * o[:, 5]
                    order = np.argsort(-scores)[: args.max_det]
                    y1, x1, y2, x2 = (o[order, c] for c in range(4))
                    x1, x2 = np.clip(x1, 0, width), np.clip(x2, 0, width)
                    y1, y2 = np.clip(y1, 0, height), np.clip(y2, 0, height)
                    b = np.stack([(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height], 1)
                    result[str(t)] = {"boxes": b.tolist(), "scores": scores[order].tolist()}
            target.write_text(json.dumps({"frames": result}), encoding="utf-8")
        print(f"predictions for {split} -> {raw_dir}", flush=True)


def score(args) -> None:
    out = Path(args.out).resolve()
    py = sys.executable
    subprocess.run([py, str(CODE_ROOT / "scripts" / "score_predictions.py"), "--frames-root", args.frames_root, "--split", "test",
                    "--modality", args.modality, "--pred-dir", str(out / "raw_test"), "--name", args.repo,
                    "--normalized", "--out", str(out / "score_test_raw.json")], check=True, cwd=str(CODE_ROOT))
    subprocess.run([py, str(CODE_ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
                    "--modality", args.modality, "--val-raw", str(out / "raw_val"), "--test-raw", str(out / "raw_test"),
                    "--out", str(out / "postprocess_sweep.json")], check=True, cwd=str(CODE_ROOT))


def cost(args) -> None:
    import torch

    net, _, _ = load_model(args)
    params = sum(p.numel() for p in net.parameters())
    x = torch.randn(1, 3, 5, 512, 512).cuda()
    with torch.no_grad():
        for _ in range(10):
            net(x)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(50):
            net(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - started) / 50 * 1000
    gflops = None
    try:
        from thop import profile

        gflops = profile(net, inputs=(x,), verbose=False)[0] * 2 / 1e9
    except Exception as exc:
        print(f"GFLOPs not computed: {exc}")
    info = {"model": args.repo, "params": params, "gflops": gflops, "latency_ms_batch1": ms, "fps": 1000 / ms}
    (Path(args.out) / "cost.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


def main() -> None:
    parser = argparse.ArgumentParser(description="CHAL / SSTNet on Anti-UAV410, scored with the DRISHTI protocol.")
    parser.add_argument("--stage", default="all", choices=["prepare", "train", "predict", "score", "cost", "all"])
    parser.add_argument("--repo", required=True, choices=sorted(REPOS))
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--view", required=True, help="frame view + annotation files (shared by CHAL and SSTNet)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=10, help="every Nth train frame (5-frame models are slow)")
    parser.add_argument("--val-stride", type=int, default=20)
    parser.add_argument("--letterbox", type=int, default=0, help="their test default for CHAL is 0; SSTNet uses 1")
    parser.add_argument("--predict-splits", default="val,test")
    parser.add_argument("--pred-batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--max-seqs", type=int, default=None, help="smoke test: first N sequences per split")
    args = parser.parse_args()
    args.letterbox = bool(args.letterbox)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = ["prepare", "train", "predict", "score", "cost"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if args.stage == "all" and stage == "train":
            try:
                weights(args)
                print("trained weights exist; skipping training")
                continue
            except SystemExit:
                pass
        print(f"===== {stage}", flush=True)
        globals()[stage](args)
        os.chdir(str(CODE_ROOT))
    print("ALL DONE")


if __name__ == "__main__":
    main()
