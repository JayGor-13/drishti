# DRISHTI-CORE v2

Production PyTorch implementation of the causal motion-guided sparse MoE detector described in `Architecture.md` and `implementation_plan.md`.

## Quick Smoke Run

```bash
python -m drishti_v2.experiments.run_eval --config configs/default.yaml --synthetic --device cpu
python -m drishti_v2.experiments.run_training --config configs/default.yaml --synthetic --device cpu --stage stage1 --epochs 1
python scripts/benchmark_latency.py --config configs/default.yaml --iters 5 --device cpu
```

## Dataset Layout

```text
data_root/
  train/<sequence>/visible/*.jpg
  train/<sequence>/visible.json
  val/<sequence>/visible/*.jpg
  val/<sequence>/visible.json
```

Annotations follow Anti-UAV style:

```json
{"gt_rect": [[x, y, w, h]], "exist": [1]}
```

## Main Modules

- `drishti_v2.models`: LDMI, MotionCNN, crop proposal engine, crop encoder, temporal fusion, sparse MoE, detection head, full pipeline.
- `drishti_v2.data`: Anti-UAV loader, synthetic loader, collator, augmentations.
- `drishti_v2.training`: staged freezing, loss, trainer, scheduler.
- `drishti_v2.evaluation`: detection metrics and evaluator.
- `drishti_v2.tracker`: constant-velocity inference tracker.

All experiment scripts print metrics/results and write artifacts under `results/`.
