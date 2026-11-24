# DRISHTI-CORE Local Test Guide (Maximum 10 Videos)

This guide explains how to test DRISHTI-CORE locally on Windows using no more than 10 labeled videos. It is written for someone with no prior knowledge of the project.

The recommended workflow uses:

- 8 labeled videos for training.
- 2 labeled videos for validation.
- Unit tests to verify individual modules.
- A synthetic preflight to verify the complete pipeline.
- Four sequential training stages.
- Final evaluation and visual smoke tests.

Do not use `train_full.py`, `train_single_video.py`, or `run_inference.py` for this local check. The configurable training, evaluation, and smoke-test entry points are used below.

## 1. Required video and annotation files

Every video must have a matching Anti-UAV annotation JSON file. An unlabeled MP4 cannot be used for training or accurate evaluation.

For infrared data, each video folder must contain:

```text
infrared.mp4
infrared.json
```

For visible/RGB data, each video folder must contain:

```text
visible.mp4
visible.json
```

The expected JSON structure is approximately:

```json
{
  "gt_rect": [
    [x, y, width, height],
    [],
    [x, y, width, height]
  ],
  "exist": [1, 0, 1]
}
```

Important requirements:

- Coordinates must be in pixels of the original video.
- The default box format is `xywh`: left position, top position, width, and height.
- There should be one annotation entry per video frame.
- An empty list and `exist: 0` mean that the target is not visible.
- Do not mix infrared and visible videos in the same run.

## 2. Open PowerShell and set the paths

Open PowerShell and run:

```powershell
$PROJECT = "C:\Users\jaygo\Desktop\DESKTOP\Research Papers\DRISHTI-CORE"

# Change these two paths to a location with enough free disk space.
$RAW = "D:\DRISHTI_LOCAL\raw"
$FRAMES = "D:\DRISHTI_LOCAL\frames"

$CFG = "$PROJECT\configs\local_10videos.yaml"
$PY = "$PROJECT\.venv\Scripts\python.exe"
$RUN = "$PROJECT\results\local_10videos"

# Use "infrared" or "visible".
$MODALITY = "infrared"

# Automatically uses CUDA when available, otherwise CPU.
$DEVICE = "auto"

Set-Location $PROJECT
```

Normally, only `$RAW`, `$FRAMES`, and `$MODALITY` need to be changed.

Paths containing spaces are supported because the commands use PowerShell variables.

PowerShell variables only last in the current PowerShell window. If the window is closed, run this section again before continuing.

## 3. Create or verify the Python environment

First check whether the existing environment works:

```powershell
& $PY --version
```

If that command says that the file does not exist, create the environment:

```powershell
py -3.11 -m venv .venv
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
```

Check PyTorch and GPU availability:

```powershell
& $PY -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

CPU mode works, but training will be considerably slower.

## 4. Run all unit tests first

From the project directory, run:

```powershell
& $PY -m pytest -q
```

Expected result in the current codebase:

```text
36 passed
```

Do not start real-video training if these tests fail.

The tests cover:

- LDMI and Sobel motion inputs.
- MotionCNN and motion gating.
- Dense, guided, edge, grid, and motion crop proposals.
- Temporal fusion.
- Sparse mixture of experts and router backpropagation.
- Detection and subpixel-offset losses.
- Video augmentation and annotation updates.
- Average precision calculation.
- Tracker assignment and constant-velocity behavior.
- Checkpoint saving and resume behavior.
- End-to-end pipeline tensor shapes.

## 5. Create a local configuration

Copy the main configuration and open the copy:

```powershell
Copy-Item ".\configs\default.yaml" $CFG
notepad $CFG
```

Make changes only in `configs\local_10videos.yaml`. Do not modify `configs\default.yaml`, and do not add duplicate YAML keys.

For infrared videos, use:

```yaml
image_channels: 1
modality: infrared
```

For visible videos, use:

```yaml
image_channels: 3
modality: visible
```

Change the local-test settings to:

```yaml
clip_stride: 64
frame_stride: 1

train_batch_size: 1
eval_batch_size: 1
num_workers: 0

checkpoint: null
device: auto

output_dir: results/local_10videos

smoke_max_frames: 48
smoke_train_steps: 5
smoke_output_video: results/local_10videos/smoke/train_video_01/bounding_boxes.mp4
```

Keep these settings at their existing values initially:

```yaml
image_height: 448
image_width: 448
temporal_window: 5
crop_size: 64
crop_scales: [1.0, 2.0, 4.0]
```

`clip_stride: 64` creates a clip every 64 frames. It makes this a relatively quick functional test while still using every selected video. For full training, change it back to `4`.

If CUDA reports an out-of-memory error, change the following settings before starting any training stage:

```yaml
image_height: 224
image_width: 224
crop_size: 32
train_batch_size: 1
eval_batch_size: 1
```

Use the same configuration for all four training stages. Do not change architecture-related values such as channel counts, temporal dimensions, expert counts, or feature dimensions between stages.

### Optional permanent data paths

The commands in this guide pass data paths through `$RAW` and `$FRAMES`, so these configuration fields can remain empty:

```yaml
data_root:
frames_root:
```

If permanent paths are preferred, Windows paths can be written with forward slashes:

```yaml
data_root: "D:/DRISHTI_LOCAL/raw"
frames_root: "D:/DRISHTI_LOCAL/frames"
```

Command-line paths override configuration paths.

## 6. Arrange no more than 10 videos

For ten videos, use eight for training and two for validation:

```text
D:\DRISHTI_LOCAL\raw\
├── train\
│   ├── video_01\
│   │   ├── infrared.mp4
│   │   └── infrared.json
│   ├── video_02\
│   │   ├── infrared.mp4
│   │   └── infrared.json
│   ...
│   └── video_08\
│       ├── infrared.mp4
│       └── infrared.json
└── val\
    ├── video_09\
    │   ├── infrared.mp4
    │   └── infrared.json
    └── video_10\
        ├── infrared.mp4
        └── infrared.json
```

For visible data, replace `infrared.mp4` and `infrared.json` with `visible.mp4` and `visible.json`.

If fewer videos are available:

- Five videos: four training and one validation.
- Two videos: one training and one validation.
- Always keep at least one validation sequence.
- Do not copy more than ten sequence folders into this test dataset.
- Do not put the same video in both training and validation.

Count the selected sequences:

```powershell
& $PY .\scripts\prepare_dataset.py $RAW
```

For ten videos, expect:

```text
train: 8
val: 2
test: 0
```

This command only counts folders. It does not fully validate the video and annotation contents.

## 7. Extract frames

Pre-extracted frames are faster and more reliable for training than repeatedly seeking inside MP4 files.

Run:

```powershell
& $PY .\scripts\extract_frames.py `
    --data-root $RAW `
    --output-root $FRAMES `
    --splits train val `
    --modalities $MODALITY `
    --workers 2
```

The first output should report:

```text
'tasks': 10
```

The number should equal the number of videos selected.

If it reports `tasks: 0`, check that:

- `$RAW` points to the folder containing `train` and `val`.
- Filenames are exactly `infrared.mp4` and `infrared.json`, or exactly `visible.mp4` and `visible.json`.
- `$MODALITY` matches the filenames.

Check the extracted result:

```powershell
& $PY .\scripts\prepare_dataset.py $FRAMES

Get-ChildItem $FRAMES -Recurse -Filter "*.jpg" | Measure-Object
Get-ChildItem $FRAMES -Recurse -Filter "$MODALITY.json" | Measure-Object
```

The annotation count should equal the number of selected videos. The frame count should be greater than zero.

The extracted structure should look like:

```text
D:\DRISHTI_LOCAL\frames\
├── train\
│   └── video_01\
│       ├── infrared\
│       │   ├── 000000.jpg
│       │   ├── 000001.jpg
│       │   └── ...
│       └── infrared.json
└── val\
    └── video_09\
        ├── infrared\
        │   ├── 000000.jpg
        │   └── ...
        └── infrared.json
```

## 8. Test the data loader

Run:

```powershell
& $PY -c "import sys; from drishti_v2.models import DRISHTIConfig; from drishti_v2.experiments.common import build_loader; c=DRISHTIConfig.from_yaml(sys.argv[1]); loader=build_loader(c,None,'train',1,frames_root=sys.argv[2],modality=c.modality); b=next(iter(loader)); print('Frame tensor:',tuple(b['frames'].shape)); print('Clips in batch:',len(b['targets'])); print('Frames in first clip:',len(b['targets'][0])); print('First target boxes:',b['targets'][0][0]['boxes'])" $CFG $FRAMES
```

For infrared data at the default resolution, expect:

```text
Frame tensor: (1, 5, 1, 448, 448)
Frames in first clip: 5
```

For visible data, expect:

```text
Frame tensor: (1, 5, 3, 448, 448)
Frames in first clip: 5
```

An empty first target box is allowed if the drone is invisible in that particular frame.

## 9. Run a synthetic preflight

This confirms that the complete model can execute before real data is used.

Run synthetic evaluation:

```powershell
& $PY -m drishti_v2.experiments.run_eval `
    --config $CFG `
    --synthetic `
    --device $DEVICE `
    --output "$RUN\preflight\synthetic_metrics.json"
```

Run synthetic Stage 1 training:

```powershell
& $PY -m drishti_v2.experiments.run_training `
    --config $CFG `
    --synthetic `
    --device $DEVICE `
    --stage stage1 `
    --epochs 1 `
    --output-dir "$RUN\preflight\synthetic_stage1"
```

Both commands should complete without exceptions, `NaN` losses, `Infinity` values, or tensor-shape errors.

## 10. Record an untrained real-data baseline

This creates a random-model baseline to compare with the final trained model:

```powershell
& $PY -m drishti_v2.experiments.run_eval `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --split val `
    --device $DEVICE `
    --output "$RUN\baseline_eval\metrics.json"
```

Low or zero mAP is expected because this model is untrained.

## 11. Train all four stages

Run the following stages in sequence. Do not skip directly from Stage 1 to final fine-tuning.

### Stage 1: motion and detector

Stage 1 trains the MotionCNN, motion gate, detection head, heatmap, bounding-box, and subpixel-offset paths.

```powershell
& $PY -m drishti_v2.experiments.run_training `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --device $DEVICE `
    --stage stage1 `
    --epochs 2 `
    --output-dir "$RUN\stage1"
```

Confirm that the checkpoint exists and inspect recent steps:

```powershell
Test-Path "$RUN\stage1\checkpoints\stage1_best.pt"
Get-Content "$RUN\stage1\stage1_steps.jsonl" | Select-Object -Last 3
```

`Test-Path` should print `True`.

### Stage 2: temporal fusion

Stage 2 loads the Stage 1 weights and trains the temporal module.

```powershell
& $PY -m drishti_v2.experiments.run_training `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --device $DEVICE `
    --stage stage2 `
    --epochs 1 `
    --checkpoint "$RUN\stage1\checkpoints\stage1_best.pt" `
    --output-dir "$RUN\stage2"
```

Check the result:

```powershell
Test-Path "$RUN\stage2\checkpoints\stage2_best.pt"
Get-Content "$RUN\stage2\stage2_steps.jsonl" | Select-Object -Last 3
```

### Stage 3: mixture of experts

Stage 3 loads the Stage 2 weights and trains the sparse MoE and router.

```powershell
& $PY -m drishti_v2.experiments.run_training `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --device $DEVICE `
    --stage stage3 `
    --epochs 1 `
    --checkpoint "$RUN\stage2\checkpoints\stage2_best.pt" `
    --output-dir "$RUN\stage3"
```

Check the result:

```powershell
Test-Path "$RUN\stage3\checkpoints\stage3_best.pt"
Get-Content "$RUN\stage3\stage3_steps.jsonl" | Select-Object -Last 3
```

### Final end-to-end fine-tuning

The command-line name of the final stage is `finetune`, not `stage4`.

```powershell
& $PY -m drishti_v2.experiments.run_training `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --device $DEVICE `
    --stage finetune `
    --epochs 1 `
    --checkpoint "$RUN\stage3\checkpoints\stage3_best.pt" `
    --output-dir "$RUN\finetune"
```

Check the result:

```powershell
Test-Path "$RUN\finetune\checkpoints\finetune_best.pt"
Get-Content "$RUN\finetune\finetune_steps.jsonl" | Select-Object -Last 3
```

Each stage should create files similar to:

```text
checkpoints\<stage>_best.pt
checkpoints\<stage>_latest.pt
history.csv
<stage>_steps.jsonl
<stage>_eval_metrics.json
training_curves.png
metrics_bar.png
moe_diagnostics.png
```

Some loss fields being zero is expected because only the relevant modules are trained in each stage. For example, Stage 1 does not train the temporal or MoE modules.

### Continuing an interrupted stage

Use `--resume-checkpoint` only when continuing the same stage. For example, to continue Stage 1:

```powershell
& $PY -m drishti_v2.experiments.run_training `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --device $DEVICE `
    --stage stage1 `
    --epochs 3 `
    --resume-checkpoint "$RUN\stage1\checkpoints\stage1_latest.pt" `
    --output-dir "$RUN\stage1"
```

Here, `--epochs 3` means the final total epoch number, not three additional epochs.

Use `--checkpoint` when moving from one stage to the next. Use `--resume-checkpoint` only when resuming the same interrupted stage.

## 12. Run final validation evaluation

Set the final checkpoint and run evaluation:

```powershell
$FINAL = "$RUN\finetune\checkpoints\finetune_best.pt"

& $PY -m drishti_v2.experiments.run_eval `
    --config $CFG `
    --frames-root $FRAMES `
    --modality $MODALITY `
    --clip-stride 64 `
    --split val `
    --checkpoint $FINAL `
    --device $DEVICE `
    --output "$RUN\final_eval\metrics.json"
```

Inspect the metrics:

```powershell
Get-Content "$RUN\final_eval\metrics.json"
```

Also inspect:

```text
results\local_10videos\final_eval\metrics_bar.png
results\local_10videos\final_eval\moe_diagnostics.png
```

Compare the final metrics with:

```text
results\local_10videos\baseline_eval\metrics.json
```

## 13. Run end-to-end visual smoke tests

Smoke mode reads the original MP4 and annotation directly. It tests raw video decoding, complete model inference, backward propagation when enabled, streaming temporal state, tracker guidance, MoE diagnostics, metrics, and video rendering.

First, edit `configs\local_10videos.yaml` and set the trained checkpoint. Use forward slashes in the YAML path:

```yaml
checkpoint: "results/local_10videos/finetune/checkpoints/finetune_best.pt"
```

### Functional smoke test on a training video

Use:

```yaml
smoke_train_steps: 5
smoke_max_frames: 48
smoke_output_video: "results/local_10videos/smoke/train_video_01/bounding_boxes.mp4"
```

Run:

```powershell
& $PY .\main.py `
    --config $CFG `
    --mode smoke `
    --sequence-dir "$RAW\train\video_01" `
    --device $DEVICE
```

This performs five end-to-end gradient steps before rendering. It is a functional test and not an unbiased model-quality measurement.

### Honest visual test on a validation video

Change the configuration to:

```yaml
smoke_train_steps: 0
smoke_output_video: "results/local_10videos/smoke/val_video_09/bounding_boxes.mp4"
```

Run:

```powershell
& $PY .\main.py `
    --config $CFG `
    --mode smoke `
    --sequence-dir "$RAW\val\video_09" `
    --device $DEVICE
```

For the second validation video, change the output path to `val_video_10` and pass its sequence directory.

Each smoke directory should contain:

```text
bounding_boxes.mp4
smoke_summary.json
smoke_metrics.png
smoke_moe_diagnostics.png
```

`smoke_training_curves.png` is produced when `smoke_train_steps` is greater than zero.

When watching the output video:

- Ground-truth boxes are marked with the label `gt`.
- Predictions are marked with their confidence score.
- Predictions should generally follow the target rather than jumping randomly.
- Ground-truth boxes must align with the visible drone.
- Misaligned ground truth normally indicates incorrect coordinates, annotation format, or frame indexing.

## 14. How to verify every module

| Module | Where it is tested | What to check |
|---|---|---|
| LDMI and Sobel motion input | Unit tests and Stage 1 | No channel or shape errors; finite heatmap and motion-displacement losses |
| MotionCNN and motion gate | Stage 1 | Finite motion losses and normally nonzero gradient norms |
| Crop proposals and multiscale encoder | Unit tests and every forward pass | No crop-budget, channel, or tensor-shape errors |
| Detection and subpixel-offset heads | Stage 1 and fine-tuning | Finite classification, bounding-box, and offset losses |
| Temporal fusion | Stage 2 and fine-tuning | Finite temporal-consistency and trajectory-smoothness losses |
| Sparse MoE and router | Stage 3 and fine-tuning | Finite balance loss, router z-loss, entropy, and load-balance metrics |
| Tracker and streaming state | Smoke video | Predictions should move continuously and follow the target |
| Augmentation | Unit tests and training loader | Training starts without corrupted boxes or metadata |
| AP and evaluation metrics | Unit tests and evaluation | Metrics contain mAP50, mAP75, precision, recall, and F1 |
| Checkpoint and resume | Unit tests and stage outputs | Best/latest checkpoints exist and later stages load them |
| Visualization | Smoke mode | MP4 and PNG files open correctly |

For each `<stage>_steps.jsonl`, verify that:

- `loss/total` is a normal finite number.
- `grad/global_norm` is finite and is normally greater than zero in most steps.
- No value is `NaN` or `Infinity`.
- `perf/throughput_samples_sec` is greater than zero.
- GPU memory is reported when CUDA is used.

Inspect recent records with:

```powershell
Get-Content "$RUN\stage1\stage1_steps.jsonl" | Select-Object -Last 3
Get-Content "$RUN\stage2\stage2_steps.jsonl" | Select-Object -Last 3
Get-Content "$RUN\stage3\stage3_steps.jsonl" | Select-Object -Last 3
Get-Content "$RUN\finetune\finetune_steps.jsonl" | Select-Object -Last 3
```

Search for invalid numerical values:

```powershell
Get-ChildItem $RUN -Recurse -Filter "*_steps.jsonl" | Select-String -Pattern "NaN|Infinity"
```

No output from this search is the expected result.

For the MoE module, check that:

- `expert_utilization` contains eight values.
- The values are not all zero.
- One expert does not receive almost 100% of tokens throughout the entire run.
- `router_entropy`, `load_balance_cv`, and `token_drop_rate` are finite.

Important evaluation metrics include:

- `map50`: higher is better.
- `map75`: higher is better and stricter than mAP50.
- `precision`: higher means fewer false detections.
- `recall`: higher means fewer missed targets.
- `false_positives_per_image`: lower is better.
- `heatmap_peak_distance`: lower is better.
- `heatmap_peak_within_5px`: higher is better.
- `motion_direction_accuracy`: higher is better.
- `temporal_iou`: higher indicates more consistent boxes between frames.
- `trajectory_smoothness`: higher indicates smoother predicted motion.
- `detection_flicker_rate`: lower is better.
- `antiuav_tracking_accuracy`: higher is better.

## 15. How to interpret the result

There are three different levels of success:

1. **Software success:** all 36 tests pass, commands finish without exceptions, losses are finite, and every stage produces checkpoints.
2. **Pipeline success:** smoke mode produces a valid annotated MP4, metrics JSON, plots, and MoE diagnostics.
3. **Model-quality success:** predictions overlap the target and validation metrics improve over the untrained baseline.

With only ten videos and one or two epochs, low or even zero mAP can still occur. That alone does not prove that a module is broken.

A likely module or data problem is indicated by:

- Exceptions or tensor-shape errors.
- Missing checkpoints or output files.
- `NaN` or `Infinity` losses.
- Permanently zero gradients in the module's active training stage.
- Annotation boxes that do not align with the target.
- Output videos that cannot be opened.
- A completely collapsed MoE router for the entire run.

## 16. Common errors and fixes

### `No extracted Anti-UAV samples found`

Check the `$FRAMES` path, split folders, modality folder names, JSON filenames, and whether frame extraction produced JPG files.

### `No infrared.mp4/json pair found`

The smoke sequence directory is wrong, or the files are not named exactly `infrared.mp4` and `infrared.json`.

For visible data, the names must be `visible.mp4` and `visible.json`.

### Channel or convolution shape error

Check the configuration:

- Infrared: `image_channels: 1` and `modality: infrared`.
- Visible: `image_channels: 3` and `modality: visible`.

Also check `$MODALITY` in PowerShell.

### CUDA out-of-memory error

Use batch size 1. If necessary, reduce image dimensions to 224 and crop size to 32, or run with:

```powershell
$DEVICE = "cpu"
```

Do not change the configuration halfway through a staged run.

### Training hangs on Windows

Confirm that the configuration contains:

```yaml
num_workers: 0
```

### Smoke mode tries to load a missing checkpoint

Before the first untrained smoke run, use:

```yaml
checkpoint: null
```

After training, replace it with the actual final checkpoint path.

### Metrics remain low

Ten videos and a few epochs are intended to validate the software pipeline, not produce a publication-quality model. Check the loss trends, output video, annotation alignment, and final metrics relative to the untrained baseline.

## 17. Starting a clean second run

Use a different result directory so that old CSV and JSONL records are not mixed with the new experiment:

```powershell
$RUN = "$PROJECT\results\local_10videos_run2"
```

Also update these configuration paths when needed:

```yaml
output_dir: results/local_10videos_run2
smoke_output_video: results/local_10videos_run2/smoke/train_video_01/bounding_boxes.mp4
```

## Final acceptance checklist

- [ ] No more than 10 videos were selected.
- [ ] Training and validation contain different videos.
- [ ] Every MP4 has a matching annotation JSON.
- [ ] `pytest -q` reports 36 passing tests.
- [ ] Frame extraction reports the expected number of tasks.
- [ ] The data-loader tensor shape matches the selected modality.
- [ ] Synthetic evaluation and training finish successfully.
- [ ] Stage 1 creates `stage1_best.pt`.
- [ ] Stage 2 creates `stage2_best.pt`.
- [ ] Stage 3 creates `stage3_best.pt`.
- [ ] Fine-tuning creates `finetune_best.pt`.
- [ ] No loss or metric contains `NaN` or `Infinity`.
- [ ] Final evaluation creates metrics and diagnostic plots.
- [ ] The smoke-test MP4 opens correctly.
- [ ] Ground-truth boxes align with the target.
- [ ] Predictions and tracking behave consistently enough for a small-data functional test.
