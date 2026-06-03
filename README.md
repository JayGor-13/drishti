# T-MoE-LLaVA 2.0 ActivityNetQA Experiment

This repository contains a CPU-runnable PyTorch scaffold for the final proposal in `T-MoE-LLaVA_Final_Proposal.md`, plus a single ActivityNetQA experiment runner.

## What Is Implemented

- Kinematic motion encoder with an X3D-style adapter contract.
- Temporally-aware router with causal context and optional motion conditioning.
- Event-based token cache for static visual patches.
- Homogeneous top-k SwiGLU MoE experts with optional LoRA adapters.
- CFCR, load-balancing, autoregressive, and expert-orthogonalization losses.
- Minimal end-to-end video-language model wrapper and ActivityNetQA trainer.
- Single experiment entrypoint with smoke/full modes.
- Visual reports for losses, cache efficiency, routing entropy, expert activation, routing probabilities, and motion confidence.
- Unit tests for routing, caching, CFCR, sequence assembly, expert divergence, and LM logits.

## Quick Start

```powershell
python -m pytest -q
Copy-Item .env.example .env
# Put your Hugging Face token in .env as HF_TOKEN=...
.\run_experiment.ps1
```

`smoke = true` is the default in `experiment.py`: it runs 1 epoch on 5% of ActivityNetQA and still executes the full train, evaluate, checkpoint, and visualization pipeline.

For the full configured run:

```powershell
.\run_experiment.ps1 -Full
```

If you have the actual ActivityNet clips locally, pass them to the Python runner:

```powershell
python experiment.py --smoke --video-root D:\path\to\activitynet_videos
```

Bash is also kept for future use:

```bash
./run_experiment.sh --smoke
./run_experiment.sh --full
```

All outputs are written to `results/`.

## Training From Local Metadata And Video Chunks

For Colab/Kaggle, keep your ActivityNetQA metadata parquet and extracted video
chunk folders in Drive or notebook storage. The runner reads the metadata file,
scans the `--video-root` folder recursively, keeps only QA rows whose videos are
present in that folder, and trains on those examples.

Example data layout:

```text
/content/drive/MyDrive/ActivityNetQA/
  metadata/activitynetqa.parquet
  videos/videos_chunked_01/all_test/videos_chunked_01/*.mp4
  videos/videos_chunked_02/all_test/videos_chunked_02/*.mp4
  results/
```

Train on one extracted video chunk folder:

```bash
python experiment.py \
  --full \
  --metadata-file /content/drive/MyDrive/ActivityNetQA/metadata/activitynetqa.parquet \
  --video-root /content/drive/MyDrive/ActivityNetQA/videos/videos_chunked_01 \
  --require-real-videos \
  --epochs 1 \
  --batch-size 2 \
  --device cuda \
  --results-dir /content/drive/MyDrive/ActivityNetQA/results
```

Resume training on the next video chunk folder:

```bash
python experiment.py \
  --full \
  --metadata-file /content/drive/MyDrive/ActivityNetQA/metadata/activitynetqa.parquet \
  --video-root /content/drive/MyDrive/ActivityNetQA/videos/videos_chunked_02 \
  --require-real-videos \
  --epochs 1 \
  --batch-size 2 \
  --device cuda \
  --results-dir /content/drive/MyDrive/ActivityNetQA/results \
  --resume-checkpoint /content/drive/MyDrive/ActivityNetQA/results/checkpoints/latest.pt
```

Checkpoints are saved after every epoch in `results/checkpoints/`:

```text
epoch_001.pt
latest.pt
tmoe_micro_final.pt
```

If neither `torchvision` nor OpenCV can decode a clip, real-video runs fail
instead of silently using proxy frames. On Colab/Kaggle, install a decoder if
needed:

```bash
pip install torchvision opencv-python-headless
```

## Project Layout

```text
models/
  motion_encoder.py
  router.py
  cache.py
  moe_layer.py
  tmoe_model.py
train/
  activitynetqa.py
  loss.py
  trainer.py
tests/
  test_modules.py
experiment.py
run_experiment.ps1
run_experiment.sh
```

The current visual and motion backbones are intentionally lightweight adapters. They are shaped so CLIP-Large, X3D-Tiny, QLoRA/NF4, and a real LLM backbone can be swapped in without changing the higher-level routing, cache, and loss contracts.
