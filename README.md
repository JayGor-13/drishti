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

## Training On ActivityNetQA Video Shards

The Hugging Face dataset keeps the QA metadata separate from the large video
archives. The runner loads the small parquet QA table once, then can download
and extract only the video zip shards you ask for.

Train on one shard:

```bash
python experiment.py --full --video-shards 1 --epochs 1 --batch-size 2 --device cuda
```

Train on several shards one after another:

```bash
python experiment.py --full --video-shards 1-3 --epochs 1 --batch-size 2 --device cuda
```

Train through all 28 shards while cleaning extracted folders after each shard:

```bash
python experiment.py --full --all-video-shards --epochs 1 --batch-size 2 --device cuda --cleanup-extracted-shards
```

By default, shard zips are downloaded into `hf_cache/activitynetqa`, extracted
under `hf_cache/activitynetqa/extracted`, and the zip is deleted after extraction
to save disk. Add `--keep-shard-zip` if you have enough storage and want to keep
the downloaded archives.

Checkpoints are saved after every epoch in `results/checkpoints/`. In shard mode
you will see files like:

```text
shard_01_epoch_001.pt
after_shard_01.pt
latest.pt
tmoe_micro_final.pt
```

To resume from the most recent checkpoint:

```bash
python experiment.py --full --video-shards 2-3 --epochs 1 --batch-size 2 --device cuda --resume-checkpoint results/checkpoints/latest.pt
```

For local videos you already extracted yourself:

```bash
python experiment.py --full --video-root /path/to/extracted/videos --require-real-videos --epochs 1 --batch-size 2 --device cuda
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
