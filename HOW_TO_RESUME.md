# How to resume the experiments: data, code, commands

For a co-author who needs to re-run or extend any experiment. Everything runs in Docker on a Linux GPU
machine; nothing needs the original laptop. Plain words; copy the commands as they are.

## 1. The data: Anti-UAV410 (the only dataset used)

1. Download the official `Anti-UAV410.zip` (9.36 GB) from the Anti-UAV410 authors' Google Drive, file id
   `1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR` (link on https://github.com/HwangBo94/Anti-UAV410). Check it:
   `sha256sum Anti-UAV410.zip` must print `339e0e56e84d620acfd2e315df39ba42a6fd5bdc958fcc0c16e2130c6b41e055`.
2. Extract frames into our layout (one folder per sequence, `infrared/000001.jpg ...` plus `infrared.json`):
   ```
   python scripts/prepare_antiuav410.py --zip /workspace/data/Anti-UAV410.zip --out /workspace/data/antiuav410_frames
   ```
   Expected: train 200 / val 90 / test 120 sequences; 213,995 / 94,711 / 129,691 frames (11 GB).
   `scripts/verify_antiuav410.py` checks the result against the fingerprint in `DATASET.md`.
3. Tracker results (only for the reference table, already scored in `results/antiuav410_scores/`): the
   authors' `tracker_results.zip`, Google Drive id `1AlLpoMorj-7bKA1zqo1DkuEZ9h0jQs_-`; import with
   `scripts/import_tracker_results.py`, score with `scripts/baselines/score_baseline.py`.

No other data is used. Anti-UAV300 was dropped (it is a near-subset of Anti-UAV410).

## 2. The code and the machine layout

```
<ROOT>/code       this package's code/ folder (the repository)
<ROOT>/data       antiuav410_frames/ and the YOLO "views" the scripts build themselves (~5 GB each)
<ROOT>/runs       one folder per experiment (results JSONs, weights, raw predictions), runs/logs/
```
`<ROOT>` is mounted as `/workspace` inside the containers; all paths in commands are `/workspace/...`.

Docker images (build once, from `code/dgx/`): `Dockerfile` (our evaluation code), `Dockerfile.yolo`
(Ultralytics YOLO, used for every detector result in the paper), `Dockerfile.yolo26` (YOLO26 only),
`Dockerfile.mmdet` (DINO, Deformable-DETR; compiles mmcv for H100/H200), `Dockerfile.istd` (CHAL, SSTNet).
```
docker build -t drishti:latest       -f code/dgx/Dockerfile       code/dgx
docker build -t drishti-yolo:latest  -f code/dgx/Dockerfile.yolo  code/dgx
```
Every job is started with `code/dgx/run_job.sh <gpu> <job-name> <python args...>`; it runs in the
background, logs to `runs/logs/<job-name>.log`, and `docker ps` shows it. Set `DRISHTI_CODE=code` (the
scripts default to `code_v4`, the folder name used on our machine): `export DRISHTI_CODE=code`.

## 3. The one script behind every detector number

`scripts/baselines/yolo_antiuav.py` trains, predicts every val/test frame, and scores with the paper's
protocol. The final system is:
```
DRISHTI_IMAGE=drishti-yolo:latest bash code/dgx/run_job.sh 0 final_s42 scripts/baselines/yolo_antiuav.py --stage all \
  --model yolo11n.pt --epochs 40 --patience 0 --pick-weights val --imgsz 640 --seed 42 \
  --motion-encoding diff --dyn-gap 12 --compensate --nwd 1.0 \
  --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410_dyn_k12_comp \
  --out /workspace/runs/baseline_yolo11n_640_comp_k12_nwd
```
About 1 h to build the input view (once), 2.5 h to train, 1.5 h to predict and score on one H200.
Results land in `runs/<out>/postprocess_sweep.json` (`chosen_on_val.test` = the table numbers), `cost.json`.
Flags that define the ablations: drop `--compensate`, drop `--nwd`, drop `--motion-encoding ... --dyn-gap`
(plain frames), `--dyn-gap 8`, `--model yolov8n-p2.yaml --pretrained yolov8n.pt` (P2 head), `--seed`.
`--help` lists everything; the docstring at the top of the file explains each option.

## 3b. Trained weights of the final system (in `weights/` of the handover package)

| File | Model | Test tracking accuracy |
| --- | --- | --- |
| `weights/final_yolo11n_comp_k12_nwd_s42.pt` | final system, seed 42 | 70.39 |
| `weights/final_yolo11n_comp_k12_nwd_s1.pt` | seed 1 | 70.30 |
| `weights/final_yolo11n_comp_k12_nwd_s2.pt` | seed 2 | 70.21 |
| `weights/final_yolov8n_p2_comp_k12_nwd.pt` | P2-head variant (`--model yolov8n-p2.yaml`) | 70.29 |

They are plain Ultralytics checkpoints (`best.pt`, chosen on val). To reproduce a test number without
training, predict and score with the same input settings the model was trained with:
```
DRISHTI_IMAGE=drishti-yolo:latest bash code/dgx/run_job.sh 0 final_s42_eval scripts/baselines/yolo_antiuav.py --stage predict \
  --model yolo11n.pt --weights /workspace/weights/final_yolo11n_comp_k12_nwd_s42.pt --imgsz 640 \
  --motion-encoding diff --dyn-gap 12 --compensate \
  --frames-root /workspace/data/antiuav410_frames --out /workspace/runs/final_s42_eval
# then:  ... --stage score --model yolo11n.pt --imgsz 640 --frames-root /workspace/data/antiuav410_frames --out /workspace/runs/final_s42_eval
```
(`--pick-weights` is not needed with `--weights`.) For the P2 file use `--model yolov8n-p2.yaml`.
The input must be built exactly as in training: `--motion-encoding diff --dyn-gap 12 --compensate`; the
weights are useless on plain frames.

## 4. The launchers (what we actually ran, in order; each prints a table at the end)

| Launcher (`code/dgx/`) | What it ran |
| --- | --- |
| `run_detector_baselines.sh` | the 7 Ultralytics detectors at 512 (Gaze-DETR's comparison set) |
| `run_motion_inputs.sh`, `run_extras.sh` | motion-input ablation, gap sweep, seeds, other detectors |
| `run_rescore.sh` | fair re-scoring (thresholds down to 0.001) + decoder row for every run (CPU) |
| `run_levers.sh`, `run_ideas.sh` | the levers and ideas that did not help (kept for the negative-results paragraph) |
| `run_framework.sh`, `run_combo.sh`, `run_night.sh` | compensation, NWD, P2, the final combination with seeds |
| `run_ablation_final.sh`, `run_final_night.sh` | ablation of the final system + decoder rows (the last thing run) |
| `run_remaining.sh` | DINO, Deformable-DETR (`scripts/baselines/mmdet_antiuav.py`), CHAL, SSTNet (`istd_antiuav.py`) |

Other scripts used in the paper: `scripts/sweep_postprocess.py` (val-chosen post-processing),
`scripts/decode_candidates.py` (the sequence decoder; `--raw-val/--raw-test` works on any run),
`scripts/analyze_by_attribute.py` (per-size / per-attribute breakdown), `scripts/check_annotation_noise.py`,
`scripts/make_tables.py` (all tables from the result files), `scripts/time_system.py` (speed).

## 5. If something is still missing when you read this

`results/TABLES.md` shows "-" for any missing cell. The last queued job was `run_final_night.sh`; re-running
it on the machine fills the ablation and decoder rows (finished pieces are skipped). After copying the new
JSONs into `results/runs/`, run `python scripts/make_tables.py --runs results/runs --trackers results/antiuav410_scores --out results/TABLES.md`.
