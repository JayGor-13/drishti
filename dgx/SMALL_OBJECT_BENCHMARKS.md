# Drishti small-object benchmark runs

Build from the handover's `code` directory. Run on a Linux DGX with Docker and NVIDIA Container Toolkit. The image contains the Python code, the pinned detection evaluators, and the tools for public downloads.

```bash
docker build --target benchmark -t drishti-small-objects -f dgx/Dockerfile.small_objects .
```

Set `DRISHTI_ROOT` to the same DGX workspace used by the previous jobs. It must be writable by your user. The default location is `/raid/scratch/$USER/drishti`. The runner uses one GPU, with GPU 0 visible inside the container.


## Running on an Apple Silicon Mac

Run these commands from the handover's `code` directory with native arm64 Python 3.9 through 3.12. The prediction command defaults to CPU and batch size one. Use `--device mps` to try the Apple GPU.

```bash
python3 -m venv .venv-mac
source .venv-mac/bin/activate
python -m pip install "torch==2.8.0" "torchvision==0.23.0" \
  "ultralytics==8.3.228" "numpy==1.26.4" "opencv-python==4.11.0.86" \
  "pycocotools==2.0.10"
```

### Svanström thermal drone videos

Run these commands from the handover's `code` directory with the Mac Python environment active. Download the single 311 MB ZIP from [Zenodo](https://zenodo.org/records/5500576). The converter accepts the ZIP or the folder your browser unpacked from it and writes lossless frames to disk. Allow about 12 GB of free disk space. [Dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8573135/)

```bash
python -m pip install "mcos-decoder==0.1.0"
python scripts/baselines/prepare_svanstrom.py \
  "$HOME/Downloads/DroneDetectionThesis-Drone-detection-dataset-e7a6eaf" --out ../data/Svanstrom-IR
```

If you have the ZIP in Downloads, use `"$HOME/Downloads/Drone-detection-dataset-v1.0.0.zip"` as the input path.

Run a two-clip pilot with the saved Anti-UAV410 checkpoint.

```bash
python scripts/baselines/small_object_benchmark.py --stage predict --dataset svanstrom \
  --data-root ../data/Svanstrom-IR --weights ../weights/final_yolo11n_comp_k12_nwd_s42.pt \
  --variants aligned-long --device cpu --limit 2 --out ../runs/mac-svanstrom-pilot
```

Once the pilot writes `predictions.json`, run the full split and score it.

```bash
python scripts/baselines/small_object_benchmark.py --stage predict --dataset svanstrom \
  --data-root ../data/Svanstrom-IR --weights ../weights/final_yolo11n_comp_k12_nwd_s42.pt \
  --variants aligned-long --device cpu --out ../runs/mac-svanstrom-full

python scripts/baselines/small_object_benchmark.py --stage score --dataset svanstrom \
  --data-root ../data/Svanstrom-IR --predictions ../runs/mac-svanstrom-full/predictions.json \
  --out ../runs/mac-scores
```

This zero-shot detection protocol scores 153 thermal drone clips and treats 208 thermal clips of other aircraft and birds as drone-negative. It excludes drone clips 067, 096, 101, and 104 because the decoded label and video frame counts differ. It uses every frame in the retained clips and reports COCO box AP, AP50, AP75, and AP for small boxes. Compare methods on these same frames with the same scoring protocol. For the paper, describe this as cross-dataset transfer for thermal drone detection.

### ITSDT-15K moving infrared targets

Extract the organized ITSDT-15K release under `../data/ITSDT-15K`. The runner uses `annotations/instances_test2017.json` and the numbered frames under `images/`. Keep the original frame order and any history frames inside the test clip folders. The `--class-agnostic` flag evaluates the saved single-class UAV detector as a generic infrared target detector. [SeViL release and published scores](https://github.com/UESTC-nnLab/SeViL)

```bash
python scripts/baselines/small_object_benchmark.py --stage predict --dataset itsdt-15k \
  --data-root ../data/ITSDT-15K --weights ../weights/final_yolo11n_comp_k12_nwd_s42.pt \
  --variants aligned-long --class-agnostic --device mps --predict-batch 8 \
  --out ../runs/mac-itsdt-15k-yolo11n-s42

python scripts/baselines/small_object_benchmark.py --stage score --dataset itsdt-15k \
  --data-root ../data/ITSDT-15K --predictions ../runs/mac-itsdt-15k-yolo11n-s42/predictions.json \
  --out ../runs/mac-scores
```

Report COCO AP, AP50, AP75, and small-object AP for the untouched Anti-UAV410 checkpoints. Keep results for methods trained on ITSDT-15K in a separate comparison group.

### Fine-tune the Anti-UAV410 checkpoint

The following runs initialize YOLO11n from the saved seed-42 weights and retain the aligned 12-frame input and relative NWD loss. ITSDT-15K uses the published 40 development clips with six reserved for validation, plus the 20 official test clips. For Svanström, the runner holds out the later 20% of clip IDs within each source class for test, uses the preceding 15% for validation, and skips one clip at each boundary. Clip IDs group frames, and the source release gives no recording-session IDs for confirming session-level separation. Test clips and annotations are listed in each run directory.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python scripts/baselines/small_object_benchmark.py --stage run --dataset itsdt-15k \
  --data-root ../data/ITSDT-15K --model ../weights/final_yolo11n_comp_k12_nwd_s42.pt \
  --variants aligned-long-nwd --device mps --epochs 3 --train-stride 12 --val-stride 8 \
  --batch 8 --predict-batch 8 --workers 0 --finetune-lr 0.0005 --out ../runs/mac-finetune

python scripts/baselines/small_object_benchmark.py --stage run --dataset svanstrom \
  --data-root ../data/Svanstrom-IR --model ../weights/final_yolo11n_comp_k12_nwd_s42.pt \
  --variants aligned-long-nwd --device mps --epochs 3 --train-stride 32 --val-stride 16 \
  --batch 8 --predict-batch 8 --workers 0 --finetune-lr 0.0005 --out ../runs/mac-finetune
```

These three-epoch runs use temporally subsampled training and validation frames for fitting and checkpoint selection. The saved checkpoint is scored on every held-out test frame. For a zero-shot comparison on the Svanström fine-tuning split, pass the run's `test_annotations.json` to both the frozen checkpoint's `--stage predict` and `--stage score` using `--test-json`.

### IRDST-R thermal target videos

Stage the IRDST-R test frames and CHAL test annotations under `../data/IRDST-R/`. The expected annotation filename is `instances_test2017.json`. Use `--test-json /path/to/annotations.json` if it differs. Keep the original paths and numbered history frames inside each clip. Run a two-clip pilot with the saved Anti-UAV410 checkpoint. [CHAL release](https://github.com/UESTC-nnLab/CHAL)

```bash
python scripts/baselines/small_object_benchmark.py --stage predict --dataset irdst-r \
  --data-root ../data/IRDST-R --weights ../weights/final_yolo11n_comp_k12_nwd_s42.pt \
  --variants aligned-long --class-agnostic --device cpu --limit 2 \
  --out ../runs/mac-irdst-pilot
```

The single-class option explicitly maps the checkpoint's one output to the test release's single scored target class. Label this experiment **zero-shot, class-agnostic thermal target detection**. For the full run, omit `--limit 2` and use `--out ../runs/mac-irdst-full`. Then score its detections.

```bash
python scripts/baselines/small_object_benchmark.py --stage score --dataset irdst-r \
  --data-root ../data/IRDST-R --predictions ../runs/mac-irdst-full/predictions.json \
  --out ../runs/mac-scores
```

### Use a compatible multiclass checkpoint on XS-VID v2

Stage `../data/XS-VID-v2/annotations/protocols/paper_detection_test.json` and the referenced test frames under `../data/XS-VID-v2/images/`. For a pilot, `--limit 2` selects the first two clips by annotation path. Pass an Ultralytics YOLO detection checkpoint whose class names cover the scored categories, and select the input it was trained with. Replace the example checkpoint path with the one you have. [XS-VID release](https://gjhhust.github.io/XS-VID/DOWNLOAD.html)

```bash
python scripts/baselines/small_object_benchmark.py --stage predict --dataset xsvid \
  --data-root ../data --weights ../weights/xsvid_aligned_long_best.pt \
  --variants aligned-long --device cpu --limit 2 --out ../runs/mac-xsvid-pilot
```

Use `--variants rgb` with an RGB checkpoint. The handover checkpoints have one class, `uav`, and the runner rejects them for multiclass XS-VID AP. The released YOLOFT-L checkpoint is a separate CUDA reference. When compatible YOLO weights are available, omit `--limit` for the full run and change the output folder to `../runs/mac-xsvid-full`.

```bash
mkdir -p ../data/evaluators
curl -fL 'https://raw.githubusercontent.com/gjhhust/YOLOFT/9e51ac319baf0d3793694e770e049585648bae42/ultralytics/data/cocoeval_xs_vid.py' \
  -o ../data/evaluators/xsvid.py

python scripts/baselines/small_object_benchmark.py --stage score --dataset xsvid \
  --data-root ../data --predictions ../runs/mac-xsvid-full/predictions.json \
  --coco-evaluator ../data/evaluators/xsvid.py --out ../runs/mac-scores
```

Prediction writes `predictions.json` and `inference.json` under `--out`. Scoring prints the path of `metrics.json` and checks the prediction manifest for split mismatches and pilot runs. COCO scoring can use annotations and predictions alone. Preserve the manifest when moving results between machines. VisDrone prediction can use `--offline` with the extracted VID test split. The VisDrone scorer also needs the test media and its Octave toolkit.

## XS-VID v2 training on the DGX

Start one container to download the dataset, build the training views, train the six input variants, predict every evaluation frame, and score the official Detection protocol.

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -u "$(id -u):$(id -g)" -v "${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}:/workspace" \
  drishti-small-objects --dataset xsvid
```

The first run uses seed 42, YOLO11n at 640, 40 epochs, and a training stride of four. The two temporal gaps are two and twelve native frames. Training and validation labels come from the released `train_dev.json` and `val_dev.json`. The final paper-protocol result uses `paper_detection_test.json`. The runner records the exact splits and annotation hashes. It trains from the ordinary COCO YOLO initialization and creates the multiclass head for the benchmark. Published reference rows use the authors' training data budgets and image resolutions. Use the paired rows in this runner to measure Drishti's effect under identical conditions. [XS-VID download and protocol](https://gjhhust.github.io/XS-VID/DOWNLOAD.html)

Rerun the same command with `--variants rgb gray aligned-long --seeds 42 1 2` to repeat the key rows. Seed 42 is reused. Use `--variants aligned-long-nwd --seeds 42 1 2` for the separate scale-adaptive NWD row. The score from each model's held-out development split is in `val/metrics.json`. Use those scores to decide which variant to carry to VisDrone.

The download and its extraction each take about 34 GB. Allow additional space for the training views and prediction files. Use `--xs-hub ms` to download from ModelScope, or `--offline` to use a complete staged dataset under `data/XS-VID-v2`. Initial YOLO weights are downloaded separately when absent from the mounted workspace. Use `--model /workspace/pretrained/yolo11n.pt` to supply them.

## VisDrone-VID transfer

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -u "$(id -u):$(id -g)" -v "${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}:/workspace" \
  drishti-small-objects --dataset visdrone --variants rgb aligned-long --seeds 42 1 2
```

This uses the VID train, validation, and test-dev splits. The official VID toolkit produces AP, AP50, AP75, and AR. The separately labeled `diagnostic_*` metrics use the XS-VID area bins with a COCO-style evaluator. Use those diagnostic columns for comparisons between our own runs. The YOLOFT paper reports on VisDrone-VID validation with its own setup. Use `--vis-eval val` to evaluate the same video split with an internal development holdout from train, and align evaluator details before making a direct numerical comparison. [Official VisDrone downloads](https://github.com/VisDrone/VisDrone-Dataset), [Official VID evaluation toolkit](https://github.com/VisDrone/VisDrone2018-VID-toolkit), [YOLOFT paper](https://arxiv.org/html/2407.18137)

The container tries the official Google Drive links. If Drive rate-limits them, stage the official ZIP files in `data/` as `VisDrone2019-VID-train.zip`, `VisDrone2019-VID-val.zip`, and `VisDrone2019-VID-test-dev.zip`, or stage the extracted folders with the same stems and pass `--offline`.

## IRDST-R and AI-TOD-v2

Stage these releases before running the container. IRDST-R uses CHAL's files and access instructions. AI-TOD-v2 uses AI-TOD images and the revised AI-TOD-v2 COCO annotations. Supply the root relative to which the annotation image paths resolve, and pass explicit JSON paths if the release filenames differ.

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -u "$(id -u):$(id -g)" -v "${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}:/workspace" \
  drishti-small-objects --dataset irdst-r --data-root /workspace/data/IRDST-R

docker run --rm --gpus '"device=0"' --ipc=host \
  -u "$(id -u):$(id -g)" -v "${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}:/workspace" \
  drishti-small-objects --dataset aitod-v2 --data-root /workspace/data/AI-TOD-v2 \
  --train-json /workspace/data/AI-TOD-v2/annotations/trainval.json \
  --test-json /workspace/data/AI-TOD-v2/annotations/test.json
```

IRDST-R compares ordinary appearance input with aligned long-gap input and exports COCO AP plus CHAL's precision, recall, and F1 calculation. AI-TOD-v2 tests standard YOLO, fixed-scale NWD assignment, relative NWD assignment, and the full relative NWD assignment and box loss. The fixed-scale and relative assignment rows isolate the choice of scale with the same YOLO assigner. NWD-RKA is a separate published method with its own ranking rule. [IRDST-R release and protocol](https://github.com/UESTC-nnLab/CHAL), [AI-TOD-v2 release and evaluator](https://github.com/Chasel-Tsui/mmdet-aitod)

## Reproduce the released YOLOFT-L result

The upstream CUDA operators need a separate image target. Prepare XS-VID once with the benchmark container using `--dataset xsvid --stage data` or a full run, then build and run this reference container.

```bash
docker build --target yoloft -t drishti-yoloft-reference -f dgx/Dockerfile.small_objects .

docker run --rm --gpus '"device=0"' --ipc=host \
  -u "$(id -u):$(id -g)" -v "${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}:/workspace" \
  drishti-yoloft-reference
```

The reference runner downloads the released weights and uses the official XS-VID evaluation commands. It writes its results under `runs/small_objects/yoloft-reference/`. To score an existing COCO prediction JSON from any method, pass `--stage score --predictions /workspace/path/to/predictions.json` to the benchmark image with the relevant dataset flags. [Official YOLOFT workflow](https://github.com/gjhhust/YOLOFT#vid-traintest)

## Outputs and checks

- `runs/small_objects/<dataset>/<configuration>/manifest.json` records the settings, clip membership, annotation hashes, and available FPS metadata. `SUMMARY.md` aggregates completed test runs by variant, shows the change from RGB over shared seeds with the pair count in parentheses, and includes published XS-VID rows for reference.
- Each variant and seed has a checkpoint record, train logs, validation and test COCO predictions and metrics, and measured batch-one latency including input encoding and NMS. VisDrone also writes its native submission text.
- Repeating a command reuses completed training, predictions, and scores. An interrupted training attempt starts a new training attempt. Changes to the configuration or annotation contents get a different output directory.
- For a small pilot after staging the data, pass `--limit 2 --epochs 1 --variants rgb aligned-long`. The outputs are labeled as a pilot. Use `--stage data` to validate the inputs only, or `--stage prepare` to build the views without training.
- The single-frame RGB and grayscale rows help separate motion gains from the effect of removing color. Ignored boxes are excluded from YOLO labels, and their regions are masked in training images while preserving labeled objects. Evaluation runs on the unmasked frames with the benchmark's original ignore annotations.
