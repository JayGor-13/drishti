# DRISHTI: baselines to run (updated 17 Oct 2026)

Deadline: CVPR 2027, paper registration 10 Nov, submission 16 Nov. Baseline numbers are the most
important missing piece of the paper. Read this page fully once; it is short on purpose.

## What changed since the last task sheet

- **Main benchmark is Anti-UAV410 (test split, 120 videos).** Anti-UAV300 is only a development set:
  most of its videos are inside Anti-UAV410.
- **Our model now has a trained, ImageNet-initialised crop encoder and Chain-of-Focus refinement.**
  Current Anti-UAV410 test result (tracker off, raw): mAP@50 76.2, mAP@75 35.1. Details and reasons for
  every change: `CHANGES_2026-09-28.md` (sections 8 to 12).
- **We are a Track-2 method**: we never get the drone's box in the first frame and must also decide
  whether a drone is present at all. The 17 published trackers are Track-1 (they get the first-frame
  box), so they are only a "for reference" table; their scores are already computed
  (`BASELINES_AntiUAV410_trackers.md`). **The main table needs detectors that, like us, get no
  first-frame box. That is your job.**
- Training now runs on the DGX in Docker (no more throttled laptop). See `dgx/README.md`.

## 0. Setup on the DGX

Folder layout (`ROOT=/raid/scratch/<your user>/drishti`):
```
ROOT/code        this repository (unzip this package here)
ROOT/data        antiuav410_frames/ (converted Anti-UAV410), dataset_frames/ (Anti-UAV300)
ROOT/pretrained  resnet18-f37072fd.pth
ROOT/runs        everything you produce (runs/logs/<job>.log)
```
1. Unzip: `cd ROOT && python3 -m zipfile -e drishti_cvpr_handoff_2026-10-15.zip .` (creates `ROOT/code`).
2. Data: follow **`DATASET.md`** exactly (download the official zip, check its checksum, convert it,
   and run `scripts/verify_antiuav410.py`; it must report the paper fingerprint). Do this after step 3
   (the conversion runs in the Docker image). ImageNet weights for our encoder (only needed to train
   our model, not for baselines): `mkdir -p ROOT/pretrained && wget -P ROOT/pretrained https://download.pytorch.org/models/resnet18-f37072fd.pth`
3. Images: `docker build -t drishti:latest -f code/dgx/Dockerfile code/dgx` and, for YOLO,
   `docker build -t drishti-yolo:latest -f code/dgx/Dockerfile.yolo code/dgx`.
4. Start every job with `bash code/dgx/run_job.sh <gpu> <job-name> <script and arguments>`: it runs the
   job in its own container, in the background, on one GPU, capped at 16 CPU cores, with the log in
   `runs/logs/<job-name>.log`. **Use only the GPU you were given** (check `nvidia-smi` first; other
   people's jobs run on this machine). Stop a job with `docker stop <job-name>`.

## 1. The protocol (identical for every baseline)

- Train on Anti-UAV410 **train**, choose the checkpoint on **val**, never look at test while tuning.
- Predict on **every frame of every val and test video** (not a subsample), keep all boxes with scores.
- Score with **our scripts only**, so the presence threshold, one-box-per-frame, duplicate merging and
  smoothing are chosen on val exactly as for our model:
  `python scripts/baselines/score_baseline.py --frames-root ROOT/data/antiuav410_frames --val-pred <val jsons> --test-pred <test jsons> --box-format <normalized|pixel> --name <method> --out runs/baseline_<method>`
  The prediction file format is at the top of `scripts/baselines/score_baseline.py`.
  The number for the paper is `chosen_on_val.test` in `postprocess_sweep.json`.
- Cost, measured on the same DGX GPU: parameters, GFLOPs at the input size you trained with, and
  latency at batch size 1 (warm up 20 frames, time 200). Write it to `runs/baseline_<method>/cost.json`.
- For training data in COCO format: `python scripts/export_coco.py --frames-root ROOT/data/antiuav410_frames --modality infrared --out ROOT/data/coco_a410 --train-stride 4`
  (train/val every 4th frame; test every frame).

## 2. Baselines, in priority order (revised 17 Oct)

We follow the comparison set of Gaze-DETR's Table III (arXiv 2607.19040, the most complete recent list on
Anti-UAV410), minus arXiv-only methods. **The Ultralytics models (YOLOv8n, v9t, v10n, YOLO11n, YOLO12n,
YOLO26n, RT-DETR-l) are run by us on the DGX; you do not need to run them.** Your list:

| # | Method | Venue | Code | What to run |
| --- | --- | --- | --- | --- |
| 1 | 2025 Anti-UAV challenge winner ("A Simple Detector with Frame Dynamics is a Strong Tracker") | CVPR'25 workshop | https://github.com/facias914/A-Simple-Detector-is-a-Strong-Tracker | Their recipe (frame-difference input + trajectory filtering). Their data layout (`IR_label.json` per video) matches the official Anti-UAV410 zip |
| 2 | CHAL | CVPR'26 | https://github.com/UESTC-nnLab/CHAL | Their default recipe |
| 3 | SSTNet (sliced spatio-temporal network, cross-slice ConvLSTM) | TGRS'24 | code link in the paper (same lab as CHAL) | Their default recipe |
| 4 | DINO-DETR (R50, 4 scales) | ICLR'23 | mmdetection or detrex | 12 epochs, 512 input, on `coco_a410` |
| 5 | PConv (pinwheel-shaped convolution) | AAAI'25 | code link on arXiv 2412.16986 | Their YOLO-based setting, 512 input |
| 6 | L2SKNet ("Saliency at the helm", learnable kernels) | TGRS'25 | code link in the paper | Their default recipe |
| 7 | Deformable-DETR (R50) | ICLR'21 | mmdetection or detrex | Only if time is left |

Input size 512 for all (as in Gaze-DETR's table) unless the method needs another size; note it if so.
Every method predicts **every frame of val and test** and is scored with `scripts/baselines/score_baseline.py`.
Dropped from the earlier sheet: Gaze-DETR (arXiv only), DEIM, YOLOV++, MOCID.

Already done, nothing to run: the 17 retrained Anti-UAV410 trackers (Track-1 reference table).

**Time box:** if a repository does not train and predict within 2 days, stop and tell me; do not sink a
week into one baseline. Priorities 1 to 4 are the must-haves for the paper.

### Baseline 1: YOLO11, step by step

Everything (convert data, train, predict every frame, score, measure cost) is one script:
`scripts/baselines/yolo_antiuav.py`.

1. Smoke test, about 10 minutes (checks the whole chain on a tiny subset):
```
DRISHTI_IMAGE=drishti-yolo:latest bash code/dgx/run_job.sh <gpu> yolo_smoke scripts/baselines/yolo_antiuav.py --stage all --model yolo11n.pt --imgsz 640 --epochs 1 --train-stride 50 --val-stride 50 --max-seqs 2 --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410_smoke --out /workspace/runs/baseline_yolo_smoke
```
   Wait for `ALL DONE` in `runs/logs/yolo_smoke.log`. If it fails, send me the last 30 lines of the log.
2. Real runs (about 3 to 6 hours each; start the second a few minutes after the first):
```
DRISHTI_IMAGE=drishti-yolo:latest bash code/dgx/run_job.sh <gpu> baseline_yolo11s_640 scripts/baselines/yolo_antiuav.py --stage all --model yolo11s.pt --imgsz 640 --epochs 40 --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410 --out /workspace/runs/baseline_yolo11s_640

DRISHTI_IMAGE=drishti-yolo:latest bash code/dgx/run_job.sh <gpu> baseline_yolo11n_640 scripts/baselines/yolo_antiuav.py --stage all --model yolo11n.pt --imgsz 640 --epochs 40 --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410 --out /workspace/runs/baseline_yolo11n_640
```
3. Results land in `runs/baseline_yolo11{s,n}_640/`: `score_test_raw.json`, `postprocess_sweep.json`,
   `cost.json`. Then, if time allows, the same at `--imgsz 448` (points for the accuracy-vs-compute
   figure).

## 3. Metrics (same names in every result file)

- **Detection:** mAP@50, mAP@75, precision, recall, F1, false positives per frame.
- **Anti-UAV tracking accuracy** (the challenge metric): per video, then averaged over videos.
- **Temporal:** detection flicker rate, temporal IoU between consecutive boxes.
- **Cost:** parameters, GFLOPs, latency and FPS at batch 1 on the DGX GPU.

## 4. Hand-back

- Keep `RUNS.md` in your run folder: one line per run (date, job name, exact command, GPU, output folder).
- Send me each `runs/baseline_<method>/` folder's JSON files (not the predictions or checkpoints).
- If a number looks surprising, tell me; do not rerun until it looks good. Every number must trace to a file.
