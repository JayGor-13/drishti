# Running DRISHTI on the DGX with Docker (GPU 0, background)

DGX: `akshita_sit@100.89.45.1`, work folder `/raid/scratch/akshita_sit/drishti` (called ROOT below).
Laptop folder: `C:\Users\akshita_RF\Documents\Akshita_Projects`.

| File to send | Size | Needed for |
| --- | --- | --- |
| `dgx_transfer\drishti_code.zip` | 0.2 MB | everything |
| `dgx_transfer\resnet18-f37072fd.pth` | 47 MB | ImageNet encoder |
| `antiuav410\Anti-UAV410.zip` (official) | 9.4 GB | main benchmark |
| `dgx_transfer\dataset_frames.tar` (Anti-UAV300) | 46 GB | ablations only (optional) |

## 1. Laptop (PowerShell): optional, pack Anti-UAV300 into one file

Only if the ablations will also run on the DGX. Copying one big file is far faster than ~600k small ones.
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
tar -cf dgx_transfer\dataset_frames.tar dataset_frames
```

## 2. Laptop: create the folders on the DGX
```
ssh akshita_sit@100.89.45.1 "mkdir -p /raid/scratch/akshita_sit/drishti/data /raid/scratch/akshita_sit/drishti/pretrained /raid/scratch/akshita_sit/drishti/runs/logs"
```

## 3. Laptop: copy with scp
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/
scp dgx_transfer\resnet18-f37072fd.pth akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/pretrained/
scp antiuav410\Anti-UAV410.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/data/
scp dgx_transfer\dataset_frames.tar akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/data/
```
(the last line only if you made the tar in step 1)

## 4. DGX: unpack the code and build the Docker image
```
ssh akshita_sit@100.89.45.1
cd /raid/scratch/akshita_sit/drishti
python3 -m zipfile -e drishti_code.zip .
docker build -t drishti:latest -f code/dgx/Dockerfile code/dgx
```
`python3 -m zipfile -e` creates `ROOT/code`; it works even if `unzip` is not installed.
If `docker build` cannot reach Docker Hub, change the `FROM` line in `code/dgx/Dockerfile` to
`nvcr.io/nvidia/pytorch:24.10-py3` and build again.

## 5. DGX: prepare the data (no GPU needed)
```
cd /raid/scratch/akshita_sit/drishti
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python scripts/prepare_antiuav410.py --zip /workspace/data/Anti-UAV410.zip --out /workspace/data/antiuav410_frames
```
Expected output: train 200 / val 90 / test 120 sequences; 213995 / 94711 / 129691 frames.

Anti-UAV300, only if you copied it:
```
tar -xf data/dataset_frames.tar -C data/
ls data/dataset_frames/train | wc -l      # expect 160
```

## 6. DGX: check GPU 0 and run the tests
```
docker run --rm --gpus '"device=0"' -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  bash -c "python -c 'import torch; print(torch.cuda.get_device_name(0))' && python -m pytest -q"
```
Expect one failure, `test_grayscale_visualization_accepts_single_channel` (needs OpenCV; not used in training).

## 7. DGX: start training on GPU 0 in the background

Both encoder variants on Anti-UAV410, side by side on GPU 0 (about 5 GB and 7 GB of GPU memory):
```
cd /raid/scratch/akshita_sit/drishti

bash code/dgx/run_job.sh 0 a410_resnet18_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_enc_resnet18.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_resnet18_s42 --seed 42 \
  --epochs stage1=30 stage2=15 stage3=10 finetune=5 --eval-split test

bash code/dgx/run_job.sh 0 a410_simple_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_enc_simple.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_simple_s42 --seed 42 \
  --epochs stage1=30 stage2=15 stage3=10 finetune=5 --eval-split test
```
Each job: stage 1 to finetune on Anti-UAV410 train, picks checkpoints on val, then evaluates on the
test set (clip-level, and sequence-level with tracker on and off). You can log out; the jobs keep running.

## 8. DGX: monitor
```
docker ps                                                     # running jobs
nvidia-smi -i 0                                               # GPU 0 load and memory
tail -c 3000 runs/logs/a410_resnet18_s42.log | tr '\r' '\n' | tail -3     # live progress
python3 - <<'EOF'                                             # per-epoch validation numbers
import csv, glob
for f in sorted(glob.glob("runs/a410_*/*/history.csv")):
    for r in csv.DictReader(open(f)):
        print(f.split("/")[1], f.split("/")[2], int(float(r["epoch"])),
              "mAP50 %.1f" % (100*float(r["val_map50"])), "recall %.1f" % (100*float(r["val_recall"])),
              "prec %.1f" % (100*float(r["val_precision"])), "track %.1f" % (100*float(r["val_antiuav_tracking_accuracy"])))
EOF
```
A job is finished when `runs/<job>/ALL_STAGES_COMPLETE.txt` exists.

## 9. Stop, resume
```
docker stop a410_resnet18_s42          # stop (keeps everything saved up to the last epoch)
```
To resume, run exactly the same `bash code/dgx/run_job.sh ...` command again; it continues from the
last saved epoch of the current stage.

## 10. Laptop: copy the results back (PowerShell)
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
mkdir a410_resnet18_s42
scp -r "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_resnet18_s42/sequence_eval" a410_resnet18_s42\
scp -r "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_resnet18_s42/final_eval" a410_resnet18_s42\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/logs/a410_*.log" logs\
```
Checkpoints stay on the DGX; copy `runs/<job>/finetune/checkpoints/finetune_best.pt` only if needed.

---

# Round 2 (13 Oct): post-processing fixes and two new training runs

Copy the new `drishti_code.zip` over the old one and unpack it again (step 3 and step 4's
`python3 -m zipfile -e` line; your data and runs are not touched). No need to rebuild the image.

All three jobs below fit on GPU 0 together (about 2, 5 and 8 GB). Start them from
`/raid/scratch/akshita_sit/drishti`.

**Job 1, evaluation only (about 3 hours, no training)** on the model you already trained.
Picks duplicate merging (NMS), top-1 and the presence threshold on VAL, applies them to TEST,
and re-tests tracker feedback with duplicates merged:
```
bash code/dgx/run_job.sh 0 a410_resnet18_s42_eval2 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_enc_resnet18.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_resnet18_s42 --eval-only --eval-split test
```

**Job 2, joint training at 448x448**: one stage, every module trains together from the start
(tests whether the temporal transformer and MoE help when they learn alongside the encoder):
```
bash code/dgx/run_job.sh 0 a410_joint448_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_enc_resnet18.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_joint448_s42 --seed 42 --schedule joint --epochs joint=40 --eval-split test
```

**Job 3, joint training at native 512x640** (frames are no longer shrunk; targets for tighter boxes):
```
bash code/dgx/run_job.sh 0 a410_joint640_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_enc_resnet18_640.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_joint640_s42 --seed 42 --schedule joint --epochs joint=40 --eval-split test
```

Each finished job writes into `runs/<job>/sequence_eval/`:
`summary_test_tracker.json`, `summary_test_no_tracker.json` (raw output, as before),
`postprocess_sweep.json` (what was chosen on val and its TEST result) and
`summary_test_tracker_tuned.json` (tracker feedback with the chosen post-processing).

Copy back (laptop, PowerShell), for each job:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
scp -r "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_joint448_s42/sequence_eval" a410_joint448_s42\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_*/sequence_eval/*.json" .
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/logs/a410_*.log" logs\
```
(`raw_*` folders are large; the JSON summaries are what I need. Also copy
`runs/a410_simple_s42/sequence_eval/*.json` from round 1.)

---

# Round 3 (14 Oct): Chain-of-Focus, both versions in parallel on GPU 0

1. Laptop (PowerShell): send the new code
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/
```

2. DGX: unpack over the old code and run the new tests (about 1 minute; data and runs are not touched)
```
ssh akshita_sit@100.89.45.1
cd /raid/scratch/akshita_sit/drishti
python3 -m zipfile -e drishti_code.zip .
docker run --rm --gpus '"device=0"' -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python -m pytest -q tests/test_focus.py
```

3. DGX: start both versions on GPU 0 in the background (about 6 GB of GPU memory each)
```
bash code/dgx/run_job.sh 0 a410_focus_fixed_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_focus_fixed.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_focus_fixed_s42 --seed 42 --schedule joint --epochs joint=40 --eval-split test

bash code/dgx/run_job.sh 0 a410_focus_adaptive_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_focus_adaptive.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_focus_adaptive_s42 --seed 42 --schedule joint --epochs joint=40 --eval-split test
```

3b. DGX: version 3 (layer-mix head) on GPU 0 as well, and the layer probe (no training, about 30 min)
```
bash code/dgx/run_job.sh 0 a410_layermix_s42 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_layermix.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_layermix_s42 --seed 42 --schedule joint --epochs joint=40 --eval-split test

bash code/dgx/run_job.sh 0 probe_joint448 scripts/probe_layers.py \
  --config /workspace/runs/a410_joint448_s42/config.yaml \
  --checkpoint /workspace/runs/a410_joint448_s42/joint/checkpoints/joint_best.pt \
  --frames-root /workspace/data/antiuav410_frames --split val \
  --out /workspace/runs/a410_joint448_s42/layer_probe.json
```
All jobs on GPU 0 together: about 6 GB per training run, 2 GB per evaluation or probe.

4. DGX: monitor
```
docker ps
nvidia-smi -i 0
tail -c 3000 runs/logs/a410_layermix_s42.log | tr '\r' '\n' | tail -3
cat runs/logs/probe_joint448.log
tail -c 3000 runs/logs/a410_focus_fixed_s42.log | tr '\r' '\n' | tail -3
tail -c 3000 runs/logs/a410_focus_adaptive_s42.log | tr '\r' '\n' | tail -3
ls runs/a410_focus_*/ALL_STAGES_COMPLETE.txt
```

5. Laptop (PowerShell): copy the results back when both are done
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
mkdir a410_focus_fixed_s42\sequence_eval, a410_focus_adaptive_s42\sequence_eval -Force
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_focus_fixed_s42/sequence_eval/*.json" a410_focus_fixed_s42\sequence_eval\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_focus_adaptive_s42/sequence_eval/*.json" a410_focus_adaptive_s42\sequence_eval\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/logs/a410_focus_*.log" logs\
mkdir a410_layermix_s42\sequence_eval -Force
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_layermix_s42/sequence_eval/*.json" a410_layermix_s42\sequence_eval\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_joint448_s42/layer_probe.json" a410_joint448_s42\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/logs/a410_layermix_s42.log" logs\
```

---

# Round 4 (15 Oct): version 4 (memory crops, more crops, smoothing, recurrent heatmap, 2 focus steps)

Use **GPU 1** (or any free GPU: change the first number). Every job is capped at 16 CPU cores.

1. Laptop (PowerShell): send the new code
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/
```

2. DGX: unpack and test
```
ssh akshita_sit@100.89.45.1
cd /raid/scratch/akshita_sit/drishti
python3 -m zipfile -e drishti_code.zip .
docker run --rm --gpus '"device=1"' -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code drishti:latest \
  python -m pytest -q tests/test_v4.py tests/test_focus.py
```

3. DGX: evaluation only, on the Chain-of-Focus (fixed) model you already trained. No training.
Each writes to its own folder `runs/a410_focus_fixed_s42/sequence_eval_<tag>`; smoothing is chosen on
val inside every one of them.
```
bash code/dgx/run_job.sh 1 ff_eval_base scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml \
  --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_focus_fixed_s42 \
  --schedule joint --eval-only --eval-split test --eval-tag base

bash code/dgx/run_job.sh 1 ff_eval_mem scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml \
  --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_focus_fixed_s42 \
  --schedule joint --eval-only --eval-split test --eval-tag mem --seq-eval-args=--memory

bash code/dgx/run_job.sh 1 ff_eval_crops12 scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml \
  --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_focus_fixed_s42 \
  --schedule joint --eval-only --eval-split test --eval-tag crops12 --seq-eval-args "--num-crops 12"

bash code/dgx/run_job.sh 1 ff_eval_mem_crops12 scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml \
  --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_focus_fixed_s42 \
  --schedule joint --eval-only --eval-split test --eval-tag mem_crops12 --seq-eval-args "--memory --num-crops 12"
```

4. DGX: train version 4 (recurrent heatmap + 2 focus steps), then evaluate it with and without memory
```
bash code/dgx/run_job.sh 1 a410_v4_s42 scripts/run_all_stages.py --config configs/dgx_antiuav410_v4.yaml \
  --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_v4_s42 \
  --seed 42 --schedule joint --epochs joint=40 --eval-split test
```
When it shows `ALL DONE`, also run it with memory crops:
```
bash code/dgx/run_job.sh 1 a410_v4_eval_mem scripts/run_all_stages.py --config configs/dgx_antiuav410_v4.yaml \
  --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_v4_s42 \
  --schedule joint --eval-only --eval-split test --eval-tag mem --seq-eval-args=--memory
```

5. DGX: monitor
```
docker ps
nvidia-smi -i 1
for j in ff_eval_base ff_eval_mem ff_eval_crops12 ff_eval_mem_crops12 a410_v4_s42; do echo "== $j"; tail -c 3000 runs/logs/$j.log | tr '\r' '\n' | grep -avE '^\s*$' | tail -1; done
```
Each evaluation job takes about 4 to 6 hours (val and test, the sweep, two tuned runs); training about 13 hours.
If the DGX is busy, run the four evaluation jobs two at a time.

6. Laptop (PowerShell): copy the results back
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
foreach ($t in "base","mem","crops12","mem_crops12") { mkdir "a410_focus_fixed_s42\sequence_eval_$t" -Force; scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_focus_fixed_s42/sequence_eval_$t/*.json" "a410_focus_fixed_s42\sequence_eval_$t\" }
mkdir a410_v4_s42\sequence_eval, a410_v4_s42\sequence_eval_mem -Force
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_v4_s42/sequence_eval/*.json" a410_v4_s42\sequence_eval\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_v4_s42/sequence_eval_mem/*.json" a410_v4_s42\sequence_eval_mem\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/logs/a410_v4_s42.log" logs\
```

---

# Baseline 1 (15 Oct): YOLO11 on Anti-UAV410 (Track-2 detector, no first-frame box)

Uses its own image (`drishti-yolo`) and the same scoring scripts as our model, so the threshold, top-1,
duplicate merging and smoothing are chosen on val in exactly the same way. Runs on GPU 1.

1. Laptop (PowerShell): send the two new files (only additions, so running jobs are not affected)
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\dgx_transfer
ssh akshita_sit@100.89.45.1 "mkdir -p /raid/scratch/akshita_sit/drishti/code_v4/scripts/baselines"
scp yolo_antiuav.py akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/code_v4/scripts/baselines/
scp Dockerfile.yolo akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/code_v4/dgx/
```

2. DGX: build the YOLO image (once, a few minutes; needs internet for Ultralytics and the COCO weights)
```
cd /raid/scratch/akshita_sit/drishti
docker build -t drishti-yolo:latest -f code_v4/dgx/Dockerfile.yolo code_v4/dgx
```

3. DGX: 10-minute smoke test (1 epoch, few images, 2 sequences) to check the whole chain works
```
DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 yolo_smoke \
  scripts/baselines/yolo_antiuav.py --stage all --model yolo11n.pt --imgsz 640 --epochs 1 \
  --train-stride 50 --val-stride 50 --max-seqs 2 \
  --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410_smoke \
  --out /workspace/runs/baseline_yolo_smoke
tail -f runs/logs/yolo_smoke.log        # wait for ALL DONE (Ctrl+C to stop watching)
```

4. DGX: the real runs (YOLO11n and YOLO11s at 640; about 3 to 6 hours each, including predicting every frame)
```
DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 baseline_yolo11s_640 \
  scripts/baselines/yolo_antiuav.py --stage all --model yolo11s.pt --imgsz 640 --epochs 40 \
  --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410 \
  --out /workspace/runs/baseline_yolo11s_640

DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 baseline_yolo11n_640 \
  scripts/baselines/yolo_antiuav.py --stage all --model yolo11n.pt --imgsz 640 --epochs 40 \
  --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410 \
  --out /workspace/runs/baseline_yolo11n_640
```
Start the second one a few minutes after the first, so the first has finished writing the shared YOLO
view (`data/yolo_a410`).

5. Laptop (PowerShell): copy the results back
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
foreach ($m in "yolo11s_640","yolo11n_640") { mkdir "baseline_$m" -Force; scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/baseline_$m/*.json" "baseline_$m\" }
```
Each folder gets `score_test_raw.json`, `postprocess_sweep.json` (val-chosen post-processing applied to
test) and `cost.json` (parameters, GFLOPs, latency).

---

# Track-1 evaluation (15 Oct): our model given the first-frame box, for a fair comparison with the trackers

1. Laptop (PowerShell): send the one updated file (adds an option only; running jobs are not affected)
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\dgx_transfer
scp run_sequence_eval.py akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/code_v4/drishti_v2/experiments/
```

2. DGX: evaluate both Chain-of-Focus models in Track-1 mode on GPU 1 (no training; about 4 to 6 hours each)
```
cd /raid/scratch/akshita_sit/drishti
DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 ff_eval_track1 scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_focus_fixed_s42 --schedule joint --eval-only --eval-split test --eval-tag track1 --seq-eval-args=--init-first-box

DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 fa_eval_track1 scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_adaptive.yaml --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_focus_adaptive_s42 --schedule joint --eval-only --eval-split test --eval-tag track1 --seq-eval-args=--init-first-box
```

3. Laptop (PowerShell): copy back
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
mkdir a410_focus_fixed_s42\sequence_eval_track1, a410_focus_adaptive_s42\sequence_eval_track1 -Force
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_focus_fixed_s42/sequence_eval_track1/*.json" a410_focus_fixed_s42\sequence_eval_track1\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_focus_adaptive_s42/sequence_eval_track1/*.json" a410_focus_adaptive_s42\sequence_eval_track1\
```
The Track-1 number to compare with the trackers is in `summary_test_tracker_tuned.json`.

**Track 1 on top of version 4 (decided 15 Oct: run this instead of the two focus jobs above, once
`a410_v4_s42` shows ALL DONE).** `--init-first-box` switches memory crops on automatically.
```
DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 v4_eval_track1 scripts/run_all_stages.py --config configs/dgx_antiuav410_v4.yaml --frames-root /workspace/data/antiuav410_frames --run-dir /workspace/runs/a410_v4_s42 --schedule joint --eval-only --eval-split test --eval-tag track1 --seq-eval-args=--init-first-box
```
Copy back:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
mkdir a410_v4_s42\sequence_eval_track1 -Force
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/a410_v4_s42/sequence_eval_track1/*.json" a410_v4_s42\sequence_eval_track1\
```
(Superseded on 16 Oct: version 4 failed; Track 1 now runs on Chain-of-Focus fixed + memory + 12 crops,
see round 5.)

---

# Round 5 (16 Oct): pending items 1-3 and version 5, all on GPU 1, one command each

All earlier jobs have finished, so it is safe to unpack the new code over `code_v4`.

1. Laptop (PowerShell)
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/drishti_code_v5.zip
```

2. DGX: unpack over `code_v4` and test
```
ssh akshita_sit@100.89.45.1
cd /raid/scratch/akshita_sit/drishti
mkdir -p v5_tmp && python3 -m zipfile -e drishti_code_v5.zip v5_tmp && cp -r v5_tmp/code/. code_v4/ && rm -rf v5_tmp
docker run --rm --gpus '"device=1"' -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code_v4 drishti:latest python -m pytest -q tests/test_v4.py
```

3. DGX: start everything (both commands return immediately; jobs run in the background)
```
bash code_v4/dgx/run_pending.sh 1
bash code_v4/dgx/run_v5.sh 1
docker ps
```
Jobs: `baseline_yolo11n_448`, `baseline_yolo11n_320` (about 2 to 4 h each), `ff_eval_track1_c12`,
`v5_guided`, `v5_fused`, `v5_fused_noref` (about 4 to 6 h each), `analysis_by_attribute` (about 30 min).

4. Laptop (PowerShell): copy everything back when `docker ps` no longer lists them
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($s in "448","320") { mkdir "baseline_yolo11n_$s" -Force; scp "$H/baseline_yolo11n_$s/*.json" "baseline_yolo11n_$s\" }
foreach ($t in "track1_c12","v5_guided","v5_fused","v5_fused_noref") { mkdir "a410_focus_fixed_s42\sequence_eval_$t" -Force; scp "$H/a410_focus_fixed_s42/sequence_eval_$t/*.json" "a410_focus_fixed_s42\sequence_eval_$t\" }
scp "$H/analysis_by_attribute.json" .
scp "$H/logs/baseline_yolo11n_*.log" logs\
scp "$H/logs/analysis_by_attribute.log" logs\
```

## Round 6 (17 Oct): calibrated fusion + version 6 (trained arbiter)

1. Laptop (PowerShell): send the new code
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/drishti_code_v6.zip
```

2. DGX: unpack over code_v4 (no v4/v5 job is running any more, so this is safe) and test
```
ssh akshita_sit@100.89.45.1
cd /raid/scratch/akshita_sit/drishti
mkdir -p v6_tmp && python3 -m zipfile -e drishti_code_v6.zip v6_tmp && cp -r v6_tmp/code/. code_v4/ && rm -rf v6_tmp
docker run --rm --gpus '"device=1"' -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code_v4 drishti:latest python -m pytest -q tests/test_v6.py
```

3. DGX: start everything (one background script; it waits for each step itself)
```
nohup bash code_v4/dgx/run_v6.sh 1 > runs/logs/run_v6.log 2>&1 &
tail -f runs/logs/run_v6.log          # Ctrl+C stops watching, not the jobs
```
Order: `fusion_sweep` (minutes) | `v6_yolo_train` (1 to 2 h) with `v6_dump_val`, `v6_dump_test`
(about 4 to 6 h each, in parallel) | `v6_dump_train` (about 5 to 8 h) | four `v6_arbiter_*` (under 1 h each).
Each job has its own log in `runs/logs/`. If anything stops, re-run step 3; finished parts are skipped.

4. Laptop (PowerShell): copy back when the log says ALL DONE (the fusion sweep can be copied as soon as it ends)
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
scp "$H/a410_focus_fixed_s42/fusion_sweep.json" a410_focus_fixed_s42\
foreach ($f in "train_attn","train_mlp","val_attn","val_mlp") { mkdir "v6_arbiter_$f" -Force; scp "$H/v6_arbiter_$f/*" "v6_arbiter_$f\" }
scp "$H/logs/fusion_sweep.log" "$H/logs/run_v6.log" "$H/logs/v6_*.log" logs\
```

## Round 6b (17 Oct): frame-dynamics YOLO11n baseline

Same as the YOLO11n-640 baseline, but the input is [frame, |difference to previous frame|, |difference to 3
frames back|] (the input-level motion idea of the 2025 challenge winner). It needs its own YOLO view
(`yolo_a410_dyn`, about 4 GB, built by the job itself, about 1 h). Can run on GPU 1 next to run_v6.sh
(YOLO11n needs about 10 GB of GPU memory). Total about 6 to 10 h.
```
cd /raid/scratch/akshita_sit/drishti
DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 baseline_yolo11n_640_dyn \
  scripts/baselines/yolo_antiuav.py --stage all --frame-dynamics --model yolo11n.pt --imgsz 640 --epochs 40 \
  --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410_dyn \
  --out /workspace/runs/baseline_yolo11n_640_dyn
tail -f runs/logs/baseline_yolo11n_640_dyn.log      # wait for ALL DONE
```
Copy back (laptop, PowerShell):
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
mkdir baseline_yolo11n_640_dyn -Force
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/baseline_yolo11n_640_dyn/*.json" baseline_yolo11n_640_dyn\
scp "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs/logs/baseline_yolo11n_640_dyn.log" logs\
```

## Round 6c (17 Oct): single-frame detector baselines (the comparison set of Gaze-DETR's Table III)

YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n, YOLO12n, YOLO26n (40 epochs) and RT-DETR-l (20 epochs), all at 512.
One queue: about 2 days. Two queues side by side on the same free GPU (a and b): about 1 day.
YOLO26 needs a newer Ultralytics, so build one extra image first (a few minutes).
```
cd /raid/scratch/akshita_sit/drishti
docker build -t drishti-yolo26:latest -f code_v4/dgx/Dockerfile.yolo26 code_v4/dgx
G=2      # a free GPU (not GPU 0, which belongs to someone else)
nohup bash code_v4/dgx/run_detector_baselines.sh $G a > runs/logs/run_detector_baselines_a.log 2>&1 &
nohup bash code_v4/dgx/run_detector_baselines.sh $G b > runs/logs/run_detector_baselines_b.log 2>&1 &
tail -f runs/logs/run_detector_baselines_a.log runs/logs/run_detector_baselines_b.log
```
Copy back (laptop, PowerShell), any time; finished models have a cost.json:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($m in "yolo11n","yolov8n","yolov10n","yolo12n","yolov9t","yolo26n","rtdetr_l") { mkdir "baseline_${m}_512" -Force; scp "$H/baseline_${m}_512/*.json" "baseline_${m}_512\" }
scp "$H/logs/run_detector_baselines_*.log" logs\
```

## Round 7 (17 Oct): does our motion network add anything on top of frame-dynamics YOLO?

Evaluation only, GPU 1, about 5 to 6 h. Needs `runs/baseline_yolo11n_640_dyn/raw_val` and `raw_test` (they exist).
1. Laptop (PowerShell): send the code
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/drishti_code_v7.zip
```
2. DGX: unpack over code_v4 and start
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v7_tmp && mkdir -p v7_tmp && python3 -m zipfile -e drishti_code_v7.zip v7_tmp && cp -r v7_tmp/code/. code_v4/ && rm -rf v7_tmp
nohup bash code_v4/dgx/run_v6_dyn.sh 1 > runs/logs/run_v6_dyn.log 2>&1 &
sleep 30; tail -n 5 runs/logs/run_v6_dyn.log; docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'dyn'
```
3. Laptop (PowerShell): copy back when `runs/logs/run_v6_dyn.log` says ALL DONE
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
mkdir a410_focus_fixed_s42\sequence_eval_v5dyn_fused_noref -Force
scp "$H/a410_focus_fixed_s42/sequence_eval_v5dyn_fused_noref/*.json" a410_focus_fixed_s42\sequence_eval_v5dyn_fused_noref\
scp "$H/a410_focus_fixed_s42/fusion_sweep_dyn.json" a410_focus_fixed_s42\
foreach ($a in "mlp","attn") { mkdir "v6dyn_arbiter_val_$a" -Force; scp "$H/v6dyn_arbiter_val_$a/*" "v6dyn_arbiter_val_$a\" }
scp "$H/logs/run_v6_dyn.log" "$H/logs/v5dyn_fused_noref.log" "$H/logs/fusion_sweep_dyn.log" "$H/logs/v6dyn_*.log" logs\
```

## Round 8 (17 Oct): ceiling analysis + version 7 refiner, side by side with round 7 on GPU 1

About 4 to 5 h. Needs the frame-dynamics YOLO run (exists) and the plain-YOLO candidates (exist).
1. Laptop (PowerShell): send the code
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/drishti_code_v7.zip
```
2. DGX: unpack over code_v4 (safe while round 7 runs) and start
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v7_tmp && mkdir -p v7_tmp && python3 -m zipfile -e drishti_code_v7.zip v7_tmp && cp -r v7_tmp/code/. code_v4/ && rm -rf v7_tmp
nohup bash code_v4/dgx/run_v7.sh 1 > runs/logs/run_v7.log 2>&1 &
sleep 30; tail -n 5 runs/logs/run_v7.log; docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'analyze|v7_'
```
The ceiling numbers appear within minutes: `cat runs/logs/analyze_candidates_test.log`.
3. Laptop (PowerShell): copy back when `runs/logs/run_v7.log` says ALL DONE
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
mkdir v6_candidates, baseline_yolo11n_640_dyn_refined -Force
scp "$H/v6_candidates/analysis_*.json" v6_candidates\
scp "$H/baseline_yolo11n_640_dyn_refined/*.json" baseline_yolo11n_640_dyn_refined\
scp "$H/logs/run_v7.log" "$H/logs/analyze_candidates_*.log" "$H/logs/v7_*.log" logs\
```

## Round 9 (17 Oct): motion-encoding ablation for YOLO11n + annotation-noise check

Start these on the free GPU once the detector-baseline queues there have finished (`docker ps` shows no `_512`
containers). Two queues side by side, about 12 to 15 h in total: `winner`, `dyn_k5`, `box15` (queue a) and
`ts`, `dyn_k8`, `flow` (queue b). Each one is the frame-dynamics YOLO11n recipe with one change; `winner` is
the 2025 challenge winner's exact input (signed differences to t-1 and t-2), the paper's baseline row.
```
cd /raid/scratch/akshita_sit/drishti
G=3
nohup bash code_v4/dgx/run_motion_inputs.sh $G a > runs/logs/run_motion_inputs_a.log 2>&1 &
nohup bash code_v4/dgx/run_motion_inputs.sh $G b > runs/logs/run_motion_inputs_b.log 2>&1 &
tail -f runs/logs/run_motion_inputs_a.log runs/logs/run_motion_inputs_b.log
```
The annotation-noise check needs no GPU (about 20 min for test, CPU) and can start any time:
```
DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 annotation_noise_test \
  scripts/check_annotation_noise.py --frames-root /workspace/data/antiuav410_frames --split test --out /workspace/runs/annotation_noise_test.json
cat runs/logs/annotation_noise_test.log
```
Copy back (laptop, PowerShell):
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($t in "dyn_k5","dyn_k8","flow","ts","box15") { mkdir "baseline_yolo11n_640_$t" -Force; scp "$H/baseline_yolo11n_640_$t/*.json" "baseline_yolo11n_640_$t\" }
scp "$H/annotation_noise_test.json" .
scp "$H/logs/run_motion_inputs_*.log" "$H/logs/annotation_noise_test.log" logs\
```

---

# MASTER PLAN (17 Oct, night): rounds 7, 8 and 9 in one transfer

What the DGX has now: round-6 code (v6 done) and the detector-baseline queues on GPU 3. This one zip adds
round 7 (our network on top of frame-dynamics YOLO), round 8 (ceiling + v7 refiner), round 9 (motion-input
ablation + annotation-noise check). GPU 0 belongs to someone else; we use GPU 1 and GPU 3.

| Step | Where | What | Time | Starts |
| --- | --- | --- | --- | --- |
| A | laptop | scp the zip | 1 min | now |
| B | DGX | unpack over code_v4, quick test | 2 min | now |
| C | GPU 1 | round 7: `run_v6_dyn.sh` (3 jobs, ~6 GB) | 5-6 h | now |
| D | GPU 1 | round 8: `run_v7.sh` (ceiling + refiner), beside C | 4-5 h | now |
| E | CPU | annotation-noise check | 20 min | now |
| F | GPU 3 | round 9: `run_motion_inputs.sh` a + b, auto-starts when the `_512` queues finish | 12-15 h | automatic |
| G | GPU 1 | optional: ceiling on the FULL candidate pool (for the decoder idea), after C | 4-6 h | after C |

A. Laptop (PowerShell)
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/drishti_code_v9.zip
```

B. DGX: unpack and check (safe while jobs run; they already loaded their code)
```
ssh akshita_sit@100.89.45.1
cd /raid/scratch/akshita_sit/drishti
rm -rf v9_tmp && mkdir -p v9_tmp && python3 -m zipfile -e drishti_code_v9.zip v9_tmp && cp -r v9_tmp/code/. code_v4/ && rm -rf v9_tmp
ls code_v4/dgx/run_v6_dyn.sh code_v4/dgx/run_v7.sh code_v4/dgx/run_motion_inputs.sh code_v4/scripts/check_annotation_noise.py
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace/code_v4 drishti:latest python -m pytest -q tests/test_v6.py
docker ps --format 'table {{.Names}}\t{{.Status}}'        # what is running right now
```

C + D + E. DGX: start rounds 7 and 8 on GPU 1 and the noise check (all return immediately)
```
nohup bash code_v4/dgx/run_v6_dyn.sh 1 > runs/logs/run_v6_dyn.log 2>&1 &
nohup bash code_v4/dgx/run_v7.sh 1 > runs/logs/run_v7.log 2>&1 &
DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 annotation_noise_test \
  scripts/check_annotation_noise.py --frames-root /workspace/data/antiuav410_frames --split test --out /workspace/runs/annotation_noise_test.json
sleep 60; docker ps --format 'table {{.Names}}\t{{.Status}}'
```
Expected after a minute: `v5dyn_fused_noref`, `v6dyn_dump_val`, `v6dyn_dump_test`, `analyze_candidates_val`,
`analyze_candidates_test`, `v7_yolo_dyn_train`, `annotation_noise_test` all "Up".

F. DGX: round 9 on GPU 3, waiting by itself for the detector-baseline queues to finish
```
nohup bash -c 'while docker ps --format "{{.Names}}" | grep -q "_512$"; do sleep 600; done; \
  bash code_v4/dgx/run_motion_inputs.sh 3 a > runs/logs/run_motion_inputs_a.log 2>&1 & \
  bash code_v4/dgx/run_motion_inputs.sh 3 b > runs/logs/run_motion_inputs_b.log 2>&1 & wait' > runs/logs/run_motion_inputs_wait.log 2>&1 &
```
(If GPU 3 is already free, this starts within 10 minutes.)

G. DGX, optional, after C has finished: the ceiling on the full candidate pool (10 YOLO boxes per frame at any
score, instead of the top 3 at >= 0.05). Tells whether sequence decoding (backlog idea 11) can reach the misses.
```
DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 v6dyn_dump_test_full -m drishti_v2.experiments.dump_candidates \
  --config configs/dgx_antiuav410_focus_fixed.yaml --checkpoint /workspace/runs/a410_focus_fixed_s42/joint/checkpoints/joint_best.pt \
  --frames-root /workspace/data/antiuav410_frames --split test --ext-topk 10 --ext-min-score 0.001 \
  --ext-dets "/workspace/runs/baseline_yolo11n_640_dyn/raw_{split}" --out /workspace/runs/v6_candidates_dyn_full
# when it is done:
DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 analyze_candidates_dyn_full scripts/analyze_candidates.py \
  --candidates /workspace/runs/v6_candidates_dyn_full --split test --out /workspace/runs/v6_candidates_dyn_full/analysis_test.json
```

Monitor (any time)
```
docker ps --format 'table {{.Names}}\t{{.Status}}'
tail -n 3 runs/logs/run_v6_dyn.log runs/logs/run_v7.log runs/logs/run_motion_inputs_a.log runs/logs/run_motion_inputs_b.log
cat runs/logs/analyze_candidates_test.log runs/logs/annotation_noise_test.log      # the two quick answers (minutes)
grep -h "^TEST" runs/logs/fusion_sweep_dyn.log runs/logs/v6dyn_arbiter_*.log runs/logs/v7_refiner.log 2>/dev/null
```
Done when: `run_v6_dyn.log`, `run_v7.log`, `run_motion_inputs_a.log` and `_b.log` each end with ALL DONE.

Copy back (laptop, PowerShell), all at once when everything is done; any line can also be run earlier
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
# round 7
mkdir a410_focus_fixed_s42\sequence_eval_v5dyn_fused_noref, v6dyn_arbiter_val_mlp, v6dyn_arbiter_val_attn -Force
scp "$H/a410_focus_fixed_s42/sequence_eval_v5dyn_fused_noref/*.json" a410_focus_fixed_s42\sequence_eval_v5dyn_fused_noref\
scp "$H/a410_focus_fixed_s42/fusion_sweep_dyn.json" a410_focus_fixed_s42\
foreach ($a in "mlp","attn") { scp "$H/v6dyn_arbiter_val_$a/*" "v6dyn_arbiter_val_$a\" }
# round 8
mkdir v6_candidates, v6_candidates_dyn, v6_candidates_dyn_full, baseline_yolo11n_640_dyn_refined -Force
scp "$H/v6_candidates/analysis_*.json" v6_candidates\
scp "$H/v6_candidates_dyn/analysis_*.json" v6_candidates_dyn\
scp "$H/v6_candidates_dyn_full/analysis_*.json" v6_candidates_dyn_full\
scp "$H/baseline_yolo11n_640_dyn_refined/*.json" baseline_yolo11n_640_dyn_refined\
scp "$H/annotation_noise_test.json" .
# round 9 + detector baselines
foreach ($t in "winner","dyn_k5","dyn_k8","flow","ts","box15") { mkdir "baseline_yolo11n_640_$t" -Force; scp "$H/baseline_yolo11n_640_$t/*.json" "baseline_yolo11n_640_$t\" }
foreach ($m in "yolo11n","yolov8n","yolov10n","yolo12n","yolov9t","yolo26n","rtdetr_l") { mkdir "baseline_${m}_512" -Force; scp "$H/baseline_${m}_512/*.json" "baseline_${m}_512\" }
# logs
scp "$H/logs/run_v6_dyn.log" "$H/logs/run_v7.log" "$H/logs/run_motion_inputs_*.log" "$H/logs/run_detector_baselines_*.log" logs\
scp "$H/logs/v5dyn_*.log" "$H/logs/v6dyn_*.log" "$H/logs/v7_*.log" "$H/logs/fusion_sweep_dyn.log" "$H/logs/analyze_candidates_*.log" "$H/logs/annotation_noise_test.log" logs\
scp "$H/logs/baseline_yolo11n_640_*.log" "$H/logs/baseline_*_512.log" logs\
```

## Round 10 (18 Oct): the decisive measurement, after rounds 7 and 8 (see CHANGES section 19)

Rounds 7 and 8 are done (round 8's refiner failed on a bug, fixed in this zip). Pending on GPU 1, in this order:
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v10_tmp && mkdir -p v10_tmp && python3 -m zipfile -e drishti_code_v10.zip v10_tmp && cp -r v10_tmp/code/. code_v4/ && rm -rf v10_tmp
nohup bash code_v4/dgx/run_ceiling_full.sh 1 > runs/logs/run_ceiling_full.log 2>&1 &      # 4 to 6 h, the decisive one
nohup bash code_v4/dgx/run_v7.sh 1 > runs/logs/run_v7.log 2>&1 &                          # refiner re-run, 4 to 5 h, beside it
DRISHTI_IMAGE=drishti-yolo:latest DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh 1 annotation_noise_test \
  scripts/check_annotation_noise.py --frames-root /workspace/data/antiuav410_frames --split test --out /workspace/runs/annotation_noise_test.json
```
Round 9 (motion inputs on GPU 3, self-starting after the `_512` baselines) is unchanged: step F of the master plan.
Quick answers (minutes): `cat runs/logs/annotation_noise_test.log runs/logs/analyze_candidates_dyn_test.log`.
Copy back when `run_ceiling_full.log` and `run_v7.log` say ALL DONE:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
mkdir v6_candidates_dyn, v6_candidates_dyn_full, baseline_yolo11n_640_dyn_refined -Force
scp "$H/v6_candidates_dyn/analysis_*.json" v6_candidates_dyn\
scp "$H/v6_candidates_dyn_full/analysis_*.json" v6_candidates_dyn_full\
scp "$H/baseline_yolo11n_640_dyn_refined/*.json" baseline_yolo11n_640_dyn_refined\
scp "$H/annotation_noise_test.json" .
scp "$H/logs/run_ceiling_full.log" "$H/logs/analyze_candidates_*.log" "$H/logs/run_v7.log" "$H/logs/v7_*.log" "$H/logs/annotation_noise_test.log" logs\
```

## Round 11 (19 Oct): the two-eye system with the k=8 appearance eye (evaluation only, GPU 1, 5 to 6 h)

Needs `runs/baseline_yolo11n_640_dyn_k8/raw_{val,test}` (exist). Also dumps the full candidate pool for the decoder.
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v11_tmp && mkdir -p v11_tmp && python3 -m zipfile -e drishti_code_v11.zip v11_tmp && cp -r v11_tmp/code/. code_v4/ && rm -rf v11_tmp
nohup bash code_v4/dgx/run_fusion_variant.sh 1 dyn_k8 > runs/logs/run_fusion_dyn_k8.log 2>&1 &
sleep 60; docker ps --format 'table {{.Names}}\t{{.Status}}'      # expect v5dyn_k8_fused_noref, cand_dyn_k8_val, cand_dyn_k8_test
```
Copy back when `runs/logs/run_fusion_dyn_k8.log` says ALL DONE:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
mkdir a410_focus_fixed_s42\sequence_eval_v5dyn_k8_fused_noref, candidates_dyn_k8 -Force
scp "$H/a410_focus_fixed_s42/sequence_eval_v5dyn_k8_fused_noref/*.json" a410_focus_fixed_s42\sequence_eval_v5dyn_k8_fused_noref\
scp "$H/a410_focus_fixed_s42/fusion_sweep_dyn_k8.json" a410_focus_fixed_s42\
scp "$H/candidates_dyn_k8/analysis_test.json" candidates_dyn_k8\
scp "$H/logs/run_fusion_dyn_k8.log" "$H/logs/v5dyn_k8_*.log" "$H/logs/fusion_sweep_dyn_k8.log" "$H/logs/cand_dyn_k8_*.log" "$H/logs/ceiling_dyn_k8_test.log" logs\
```

## Round 12 (19 Oct): paper material in parallel, on any free GPU (independent of the decoder)

1. YOLO extras, two queues (about 12 to 14 h together): gap 12 and 16, seeds 1 and 2 of gap 8, gap 8 on
   YOLO26n and YOLO11s, one more seed of gap 5. Needs the `drishti-yolo26` image for the YOLO26n run.
```
cd /raid/scratch/akshita_sit/drishti
G=3        # a free GPU
nohup bash code_v4/dgx/run_extras.sh $G a > runs/logs/run_extras_a.log 2>&1 &
nohup bash code_v4/dgx/run_extras.sh $G b > runs/logs/run_extras_b.log 2>&1 &
```
2. Seed 43 of our motion network (the long pole: about 13 h training + 4 h evaluation, about 6 GB). The
   decoder is post-processing, so this seed stays valid whatever the decoder turns out to be.
```
DRISHTI_CODE=code_v4 bash code_v4/dgx/run_job.sh $G a410_focus_fixed_s43 scripts/run_all_stages.py \
  --config configs/dgx_antiuav410_focus_fixed.yaml --frames-root /workspace/data/antiuav410_frames \
  --run-dir /workspace/runs/a410_focus_fixed_s43 --seed 43 --schedule joint --epochs joint=40 --eval-split test
```
Copy back (laptop, PowerShell), when the queue logs say ALL DONE and `runs/a410_focus_fixed_s43/ALL_STAGES_COMPLETE.txt` exists:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($t in "yolo11n_640_dyn_k12","yolo11n_640_dyn_k16","yolo11n_640_dyn_k8_s1","yolo11n_640_dyn_k8_s2","yolo26n_640_dyn_k8","yolo11s_640_dyn_k8","yolo11n_640_dyn_k5_s1") { mkdir "baseline_$t" -Force; scp "$H/baseline_$t/*.json" "baseline_$t\" }
mkdir a410_focus_fixed_s43\sequence_eval -Force
scp "$H/a410_focus_fixed_s43/sequence_eval/*.json" a410_focus_fixed_s43\sequence_eval\
scp "$H/logs/run_extras_*.log" "$H/logs/baseline_yolo*_dyn_k*.log" "$H/logs/a410_focus_fixed_s43.log" logs\
```

## Round 13 (20 Oct): everything that is left, on GPU 1 and GPU 3 (see CHANGES section 22)

1. Laptop: send the code
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects
scp dgx_transfer\drishti_code.zip akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/drishti_code_v13.zip
```
2. DGX: unpack, then the smoke test (builds two new images, then about 30-45 min)
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v13_tmp && mkdir -p v13_tmp && python3 -m zipfile -e drishti_code_v13.zip v13_tmp && cp -r v13_tmp/code/. code_v4/ && rm -rf v13_tmp
nohup bash code_v4/dgx/run_remaining.sh smoke 1 > runs/logs/run_remaining_smoke.log 2>&1 &
tail -f runs/logs/run_remaining_smoke.log            # wait for "ALL DONE (smoke)" and the smoke summary
```
3. DGX: when all three smoke jobs say done, start the two long queues
```
nohup bash code_v4/dgx/run_remaining.sh gpu3 3 > runs/logs/run_remaining_gpu3.log 2>&1 &
nohup bash code_v4/dgx/run_remaining.sh gpu1 1 > runs/logs/run_remaining_gpu1.log 2>&1 &
```
GPU 3: DINO, Deformable-DETR, CHAL (~2 days). GPU 1: full system seeds 2 and 3, plain YOLO26n at 640, then SSTNet (~1.5 days).
Monitor: `tail -n 5 runs/logs/run_remaining_gpu1.log runs/logs/run_remaining_gpu3.log; docker ps`.
If a smoke job fails, send me the last lines it prints; the long queues skip nothing that failed.

## Round 13, final queue (20 Oct): everything in one go, results in about 2 days

After the smoke test (step 2 of round 13) shows all three done, start all three queues together:
```
cd /raid/scratch/akshita_sit/drishti
nohup bash code_v4/dgx/run_remaining.sh gpu3 3   > runs/logs/run_remaining_gpu3.log 2>&1 &    # DINO, Deformable-DETR, CHAL      ~2 days
nohup bash code_v4/dgx/run_remaining.sh ablate 3 > runs/logs/run_remaining_ablate.log 2>&1 &  # 3 ablation rows + system cost   ~6 h
nohup bash code_v4/dgx/run_remaining.sh gpu1 1   > runs/logs/run_remaining_gpu1.log 2>&1 &    # system seeds 2-3, YOLO26n, SSTNet ~1.5 days
```
Monitor: `tail -n 4 runs/logs/run_remaining_*.log; docker ps --format 'table {{.Names}}\t{{.Status}}'`.

Copy back (laptop, PowerShell), any time; unfinished items just error:
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($b in "dino_r50","deformable_detr_r50","chal","sstnet","yolo26n_640") { mkdir "baseline_$b" -Force; scp "$H/baseline_$b/*.json" "baseline_$b\" }
foreach ($c in "dyn_k8_s1_s43","dyn_k8_s2_s44","dyn_k8_nomem","dyn_k8_crops8","dyn_k8_joint448_s42") { mkdir "candidates_$c" -Force; scp "$H/candidates_$c/*.json" "candidates_$c\" }
foreach ($s in "s43","s44") { scp "$H/a410_focus_fixed_$s/fusion_sweep_dyn_k8_*.json" "a410_focus_fixed_$s\" }
scp "$H/system_cost.json" .
scp "$H/logs/run_remaining_*.log" "$H/logs/run_fusion_*.log" "$H/logs/run_ablation_*.log" "$H/logs/decode_*.log" "$H/logs/baseline_dino_r50.log" "$H/logs/baseline_deformable_detr_r50.log" "$H/logs/baseline_chal.log" "$H/logs/baseline_sstnet.log" "$H/logs/baseline_yolo26n_640.log" "$H/logs/system_cost.log" logs\
```

## Round 14 (21 Oct): the untried shortlist ideas on gap-8 YOLO11n (backlog 14, 16, 17, 7, + resolution 960)

Any free GPU (they are small; GPU 1 is fine next to the other jobs). About 10 to 12 h for both queues.
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v16_tmp && mkdir -p v16_tmp && python3 -m zipfile -e drishti_code_v16.zip v16_tmp && cp -r v16_tmp/code/. code_v4/ && rm -rf v16_tmp
nohup bash code_v4/dgx/run_ideas.sh 1 a > runs/logs/run_ideas_a.log 2>&1 &
nohup bash code_v4/dgx/run_ideas.sh 1 b > runs/logs/run_ideas_b.log 2>&1 &
```
Copy back (laptop, PowerShell):
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($t in "yolo11n_640_delta","yolo11n_640_dyn_k8_blank","yolo11n_640_dyn_k8_tta","yolo11n_640_bgmed","yolo11n_960_dyn_k8") { mkdir "baseline_$t" -Force; scp "$H/baseline_$t/*.json" "baseline_$t\" }
scp "$H/logs/run_ideas_*.log" logs\
```

## Round 15 (21 Oct): the seven improvement levers in parallel (see CHANGES section 28)

Two queues, about 15 h each; use GPU 1 and GPU 3 (if GPU 3 still runs Deformable-DETR / CHAL, both queues on GPU 1 is fine).
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v17_tmp && mkdir -p v17_tmp && python3 -m zipfile -e drishti_code_v17.zip v17_tmp && cp -r v17_tmp/code/. code_v4/ && rm -rf v17_tmp
nohup bash code_v4/dgx/run_levers.sh 1 a > runs/logs/run_levers_a.log 2>&1 &
nohup bash code_v4/dgx/run_levers.sh 3 b > runs/logs/run_levers_b.log 2>&1 &
```
Each queue prints a comparison table at the end (reference: gap 8 = 69.18, seeds 69.17 +- 0.18).
Copy back (laptop, PowerShell):
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($t in "yolo11n_640_dyn_k8_all","yolo11n_640_dyn_k8_rect","yolo11n_640_dyn_k8_zoom","yolo11n_640_diff2_3_12","yolo11n_800_dyn_k8","yolo11n_640_dyn_k8_clahe","yolo11n_640_dyn_k8_over","yolo11n_640_dyn_k8_pred800") { mkdir "baseline_$t" -Force | Out-Null; scp "$H/baseline_$t/*.json" "baseline_$t\" }
scp "$H/logs/run_levers_*.log" logs\
```

## Round 16 (22 Oct): the new-framework pieces on GPUs 0, 1 and 2 (see CHANGES section 29)

Three queues, 6 to 8 h each, started together (round 15 can keep running beside them).
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v18_tmp && mkdir -p v18_tmp && python3 -m zipfile -e drishti_code_v18.zip v18_tmp && cp -r v18_tmp/code/. code_v4/ && rm -rf v18_tmp
nohup bash code_v4/dgx/run_framework.sh 0 g0 > runs/logs/run_framework_g0.log 2>&1 &
nohup bash code_v4/dgx/run_framework.sh 1 g1 > runs/logs/run_framework_g1.log 2>&1 &
nohup bash code_v4/dgx/run_framework.sh 2 g2 > runs/logs/run_framework_g2.log 2>&1 &
```
Each queue prints a comparison table at the end (reference gap 8 = 69.18, seeds +- 0.18).
Copy back (laptop, PowerShell):
```
cd C:\Users\akshita_RF\Documents\Akshita_Projects\drishti-master\runs
$H = "akshita_sit@100.89.45.1:/raid/scratch/akshita_sit/drishti/runs"
foreach ($t in "yolo11n_640_dyn_k8_comp","yolo11n_640_dyn_k8_comp_nwd","yolo11n_640_k8_comp_tprof","yolo11n_640_k8_comp_lcm","yolo11n_640_dyn_k8_nwd","yolov8n_640_dyn_k8","yolov8n_p2_640_dyn_k8") { mkdir "baseline_$t" -Force | Out-Null; scp "$H/baseline_$t/*.json" "baseline_$t\" }
scp "$H/logs/run_framework_*.log" logs\
```

## Round 17 (22 Oct): combine the winning pieces (compensation + NWD + gap 12, seeds, P2 variant, NWD constant)

Three queues, 4 to 8 h each; c1 waits by itself for c0's shared view. Reference gap 8 = 69.18.
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v19_tmp && mkdir -p v19_tmp && python3 -m zipfile -e drishti_code_v19.zip v19_tmp && cp -r v19_tmp/code/. code_v4/ && rm -rf v19_tmp
nohup bash code_v4/dgx/run_combo.sh 0 c0 > runs/logs/run_combo_c0.log 2>&1 &
nohup bash code_v4/dgx/run_combo.sh 1 c1 > runs/logs/run_combo_c1.log 2>&1 &
nohup bash code_v4/dgx/run_combo.sh 3 c2 > runs/logs/run_combo_c2.log 2>&1 &
```
Copy back (laptop, PowerShell): the usual `baseline_*` loop (all runs), plus `scp "$H/logs/run_combo_*.log" logs\`.

## Round 18 (22 Oct): one command for everything left, on one GPU (~16-18 h)

Runs zoom in the background, then the combination queues c2 -> c0 -> c1 in order, and prints the final table.
```
cd /raid/scratch/akshita_sit/drishti
rm -rf v20_tmp && mkdir -p v20_tmp && python3 -m zipfile -e drishti_code_v20.zip v20_tmp && cp -r v20_tmp/code/. code_v4/ && rm -rf v20_tmp
nohup bash code_v4/dgx/run_night.sh 1 > runs/logs/run_night.log 2>&1 &
tail -f runs/logs/run_night.log
```
Copy back: the usual `baseline_*` loop, plus `scp "$H/logs/run_night.log" "$H/logs/run_combo_*.log" logs\`.
