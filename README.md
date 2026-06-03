# T-MoE Anti-UAV Detector

This repository now implements the Anti-UAV application described in
`TMoE_AntiDrone_Architecture.md`: a sparse, motion-conditioned video detector
with a LocateAnything-compatible semantic pathway, X3D-style motion pathway,
modality-aware top-2 MoE routing, event token cache, CFCR loss, and a simple
drone/no-drone detection head.

Dataset source:

```text
https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV
```

## What Is Implemented

- LocateAnything-style patch semantic encoder with a local lightweight stand-in.
- X3D-style motion encoder producing patch motion embeddings and confidence.
- Router matching `softmax(W_r [semantic concat motion])`.
- 8-expert top-2 MoE with optional dense-upcycling mode.
- Event-based temporal token cache with the architecture threshold default `0.15`.
- CFCR loss with cosine semantic alignment between adjacent frames.
- Patch-wise drone/no-drone logits and normalized `cx, cy, w, h` box regression.
- COCO-format Anti-UAV loader using `torchvision.datasets.CocoDetection`.
- Synthetic smoke dataset for quick CPU verification without downloading data.

## Quick Start

```powershell
python -m pytest -q
.\run_experiment.ps1
```

The default smoke run trains one epoch on synthetic moving drone boxes and
writes outputs to `results/`.

## Training With The ModelScope Anti-UAV Data

Download/extract the ModelScope dataset locally, then point the runner at the
COCO image roots and annotation files:

```powershell
python experiment.py `
  --full `
  --stage sparse `
  --train-image-root D:\AntiUAV\train2017 `
  --train-ann-file D:\AntiUAV\annotations_train.json `
  --val-image-root D:\AntiUAV\val2017 `
  --val-ann-file D:\AntiUAV\annotations_val.json `
  --height 448 `
  --width 448 `
  --patch-grid-size 28 `
  --num-frames 9 `
  --hidden-dim 1024 `
  --ffn-dim 4096 `
  --batch-size 8 `
  --device cuda
```

The dataset access path follows the requested COCO pattern:

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CocoDetection(
    root="path/to/train2017",
    annFile="path/to/annotations_train.json",
    transforms=data_transforms,
)
val_dataset = datasets.CocoDetection(
    root="path/to/val2017",
    annFile="path/to/annotations_val.json",
    transforms=data_transforms,
)
```

The repository wraps this with temporal-window assembly and patch target
assignment in `train/antiuav.py`.

## Training Stages

Dense upcycling:

```powershell
python experiment.py --full --stage dense ...
```

Sparse routing with cache and CFCR:

```powershell
python experiment.py --full --stage sparse ...
```

Outputs include checkpoints, `train_history.csv`, `eval_summary.json`, loss
curves, expert activation heatmaps, motion confidence heatmaps, and token-level
routing heatmaps.

## Project Layout

```text
models/
  motion_encoder.py
  router.py
  cache.py
  moe_layer.py
  tmoe_model.py
train/
  antiuav.py
  loss.py
  trainer.py
tests/
  test_modules.py
experiment.py
run_experiment.ps1
run_experiment.sh
TMoE_AntiDrone_Architecture.md
```
