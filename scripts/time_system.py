"""Cost of the full system, per frame, batch 1, streaming (one frame in, one decision out).

Measures on real test frames, in order, on one GPU:
  input    building the gap-k frame-difference image (CPU)
  yolo     the appearance eye (Ultralytics, half precision)
  ours     our motion network, forward_stream with the detector's top boxes + memory as guided crops
  decoder  the online sequence decoder (lag 0), per frame, on a sequence of the same size
and adds the GFLOPs of the two networks (YOLO from its cost.json, ours from scripts/count_flops.py).
Run it inside the drishti-yolo image (it has Ultralytics and everything our code needs).

Usage:
  python scripts/time_system.py --frames-root /workspace/data/antiuav410_frames \
      --yolo-run /workspace/runs/baseline_yolo11n_640_dyn_k8 --dyn-gap 8 \
      --config /workspace/runs/a410_focus_fixed_s42/config.yaml \
      --checkpoint /workspace/runs/a410_focus_fixed_s42/joint/checkpoints/joint_best.pt \
      --out /workspace/runs/system_cost.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "baselines"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-frame cost of the full system.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--yolo-run", required=True)
    parser.add_argument("--dyn-gap", type=int, default=8)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    import decode_candidates as dc
    from drishti_v2.data.dataset import _pil_to_tensor, _resize_chw
    from drishti_v2.experiments.common import build_model, load_config
    from drishti_v2.tracker import TargetMemory
    from yolo_antiuav import MotionEncoder, sequences

    device = torch.device("cuda")
    yolo = YOLO(str(Path(args.yolo_run) / "train" / "weights" / "best.pt"))
    config = load_config(args.config)
    config.num_crops = 12
    ours = build_model(config, checkpoint=args.checkpoint, device=device).eval()
    encoder = MotionEncoder("diff", gap=args.dyn_gap)
    name, frames, _ = next(sequences(Path(args.frames_root), "test", "infrared"))
    frames = frames[: args.frames]
    times = {"input": [], "yolo": [], "ours": []}
    memory = TargetMemory(threshold=config.objectness_threshold, max_age=30)
    ours.reset_stream()

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    with torch.no_grad():
        for i in range(len(frames)):
            t0 = time.perf_counter()
            img = encoder(frames, i)[..., ::-1].copy()
            t1 = sync()
            r = yolo.predict(source=img, imgsz=640, conf=0.001, iou=0.6, max_det=10, device=0, verbose=False, half=True)[0]
            t2 = sync()
            eb, es = r.boxes.xywhn.cpu(), r.boxes.conf.cpu()
            keep = es >= 0.05
            eb, es = eb[keep][:3], es[keep][:3]
            frame, _ = _pil_to_tensor(frames[i], config.image_channels)
            frame = _resize_chw(frame, config.image_height, config.image_width).unsqueeze(0).to(device)
            guided = eb[:, :2].clamp(0, 1).unsqueeze(0) if eb.numel() else None
            mem = memory.guided_centers()
            if mem is not None:
                guided = mem if guided is None else torch.cat([guided, mem], dim=1)
            t3 = sync()
            out = ours.forward_stream(frame, i, guided.to(device) if guided is not None else None)
            t4 = sync()
            scores = torch.sigmoid(out.objectness_logits[0, :, 0]).cpu()
            memory.update(torch.cat([out.boxes[0].cpu(), eb]), torch.cat([scores, es]))
            if i >= 20:  # warm-up excluded
                times["input"].append((t1 - t0) * 1000)
                times["yolo"].append((t2 - t1) * 1000)
                times["ours"].append((t4 - t3) * 1000)

    # decoder: a synthetic sequence of the same size (22 candidates per frame), per-frame cost of online decoding
    T, K = 1000, 22
    rng = np.random.default_rng(0)
    seq = {"boxes": np.concatenate([rng.uniform(0.1, 0.9, (T, K, 2)), rng.uniform(0.01, 0.05, (T, K, 2))], -1).astype(np.float32),
           "mask": np.ones((T, K), dtype=bool), "score": rng.uniform(0, 1, (T, K)).astype(np.float32),
           "ours": np.arange(K)[None, :].repeat(T, 0) < 12, "gt": np.zeros((T, 4), np.float32), "visible": np.zeros(T, bool)}
    p = {"alpha": 0.4, "absent": -4.0, "switch": 4.0, "w_move": 0.25, "jump_cap": 5.0, "w_size": 0.5, "merge": 0, "lag": 0}
    started = time.perf_counter()
    dc.decode(seq, p)
    decoder_ms = (time.perf_counter() - started) / T * 1000

    flops_ours = None
    try:
        res = subprocess.run([sys.executable, str(CODE_ROOT / "scripts" / "count_flops.py"), "--config", args.config,
                              "--checkpoint", args.checkpoint, "--device", "cuda"], capture_output=True, text=True, check=True)
        text = res.stdout
        flops_ours = json.loads(text[text.index("{"):])["gflops_per_frame_stream"]
    except Exception as exc:
        print(f"our GFLOPs not computed: {exc}")
    yolo_cost = json.loads((Path(args.yolo_run) / "cost.json").read_text(encoding="utf-8"))
    ms = {k: float(np.mean(v)) for k, v in times.items()}
    ms["decoder"] = decoder_ms
    total = sum(ms.values())
    report = {"sequence": name, "frames_timed": len(times["yolo"]), "ms_per_frame": {k: round(v, 2) for k, v in ms.items()},
              "total_ms_per_frame": round(total, 2), "fps_sequential": round(1000 / total, 1),
              "gflops": {"yolo": yolo_cost.get("gflops"), "ours": flops_ours,
                         "total": (yolo_cost.get("gflops") or 0) + (flops_ours or 0)},
              "params_millions": {"yolo": round(yolo_cost.get("params", 0) / 1e6, 2),
                                  "ours": round(sum(p.numel() for p in ours.parameters()) / 1e6, 2)},
              "gpu": torch.cuda.get_device_name(0)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("ALL DONE")


if __name__ == "__main__":
    main()
