# T-MoE-LLaVA 2.0 Micro-MoE Scaffold

This repository now contains a CPU-runnable PyTorch scaffold for the implementation plan in `implementation_plan.md`.

## What Is Implemented

- Kinematic motion encoder with an X3D-style adapter contract.
- Temporally-aware router with causal context and optional motion conditioning.
- Event-based token cache for static visual patches.
- Homogeneous top-k SwiGLU MoE experts with optional LoRA adapters.
- CFCR, load-balancing, autoregressive, and expert-orthogonalization losses.
- Minimal end-to-end video-language model wrapper and trainer.
- Synthetic ablation runner for cache, routing, CFCR, and orthogonalization variants.
- Unit tests for routing, caching, CFCR, sequence assembly, expert divergence, and LM logits.

## Quick Start

```powershell
python -m pytest -q
python run_pipeline.py
python run_ablations.py
```

The smoke pipeline uses small dimensions by default, repeats frames to trigger the static-token cache path, and prints logits plus MoE cache statistics.

## Project Layout

```text
models/
  motion_encoder.py
  router.py
  cache.py
  moe_layer.py
  tmoe_model.py
train/
  loss.py
  trainer.py
tests/
  test_modules.py
run_pipeline.py
run_ablations.py
```

The current visual and motion backbones are intentionally lightweight adapters. They are shaped so CLIP-Large, X3D-Tiny, QLoRA/NF4, and a real LLM backbone can be swapped in without changing the higher-level routing, cache, and loss contracts.
