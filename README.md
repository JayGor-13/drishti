# DRISHTI: budgeted, causal detection of tiny drones in infrared video

A detector for tiny UAVs that never gets the first-frame box (Anti-UAV Track-2 setting). Per frame it
computes a motion heatmap, cuts a small budget of crops where the drone is likely to be, encodes them
with an ImageNet-initialised ResNet-18 stem, fuses the last 5 frames causally, and refines the best
candidates with a closer look (Chain-of-Focus). About 4.6 M parameters and 3.3 GFLOPs per frame.

## Where to start

| File | What it is |
| --- | --- |
| `STUDENT_TASKS.md` | What to run now: baselines in priority order (YOLO11 first), protocol, hand-back |
| `DATASET.md` | Where to download Anti-UAV410, checksums, conversion, and proof that your frames match ours |
| `CHANGES_2026-09-28.md` | Every code change since the September review, with the reason, in plain words |
| `PLAN_CVPR2027.md` | The paper plan and decisions (benchmark, framing, schedule) |
| `dgx/README.md` | Exact DGX commands: Docker image, jobs on one GPU, monitoring, copying results |
| `BASELINES_AntiUAV410_trackers.md` | The 17 Track-1 trackers already scored with our code |

## Main scripts

| Script | Purpose |
| --- | --- |
| `scripts/run_all_stages.py` | Train (`--schedule joint`) and evaluate one model end to end; `--eval-only` re-evaluates |
| `drishti_v2/experiments/run_sequence_eval.py` | Streaming, every-frame evaluation (tracker, memory crops, smoothing options) |
| `scripts/sweep_postprocess.py` | Choose threshold / one box per frame / merging / smoothing on val, apply once to test |
| `scripts/baselines/yolo_antiuav.py` | YOLO11 baseline, end to end |
| `scripts/baselines/score_baseline.py` | Score any baseline's predictions with the same protocol |
| `scripts/prepare_antiuav410.py`, `scripts/export_coco.py`, `scripts/verify_antiuav410.py` | Data conversion (our layout, COCO) and verification |
| `scripts/count_flops.py`, `scripts/probe_layers.py` | Cost accounting; per-layer location probe |

## Main configs (Anti-UAV410, Docker paths)

| Config | Model |
| --- | --- |
| `configs/dgx_antiuav410_enc_resnet18.yaml` | Reference: trained ResNet-18 encoder, no focus |
| `configs/dgx_antiuav410_focus_fixed.yaml` | **Current best**: + Chain-of-Focus, one closer look per candidate |
| `configs/dgx_antiuav410_focus_adaptive.yaml` | Chain-of-Focus with a learned stop (ablation) |
| `configs/dgx_antiuav410_v4.yaml` | + recurrent heatmap + two focus steps (in progress) |

Train with `--schedule joint --epochs joint=40`. Tests: `python -m pytest -q` (one visualization test
needs OpenCV and may fail; it is not used in training).

For the XS-VID v2, VisDrone-VID, IRDST-R, and AI-TOD-v2 Docker experiments, see [Small-object benchmarks](dgx/SMALL_OBJECT_BENCHMARKS.md).
