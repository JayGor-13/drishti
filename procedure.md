# DRISHTI-CORE v1.0

## Title

DRISHTI: Motion-Guided Sparse Temporal Mixture-of-Experts for Tiny UAV Detection

## Mission

Build a publishable AAAI/NeurIPS research model that serves as the core
perception engine for future Counter-UAS systems.

## Dataset

Primary dataset:

- Anti-UAV

Optional future datasets:

- Anti-UAV410
- Drone-vs-Bird

## Architecture

```text
3 RGB frames [t-1, t, t+1]
  -> Stage 1: MotionCNN
  -> Stage 2: Top-k crop proposal
  -> Stage 3: LocateAnything encoder
  -> Motion score concatenation
  -> Stage 4: Temporal fusion transformer
  -> Stage 5: Sparse mixture of experts
  -> Stage 6: Detection head
  -> bounding boxes + confidence scores
```

This DRISHTI-CORE version intentionally stops at Anti-UAV detection. Swarm
tracking, graph behavior classification, and threat scoring require additional
swarm-specific labels and are not part of v1.0.

## Stage 1: MotionCNN

Input:

```text
[B, 9, H, W]
```

Architecture:

```text
Conv2d(9 -> 32) -> BN -> ReLU
Conv2d(32 -> 64) -> BN -> ReLU
Conv2d(64 -> 64) -> BN -> ReLU
Conv2d(64 -> 1) -> Sigmoid
```

Output:

- motion heatmap

Purpose:

- motion localization
- background suppression
- tiny UAV candidate discovery

## Stage 2: Top-K Crop Proposal

Input:

- motion heatmap

Operation:

- top-k peak extraction
- `K = 8`

Output:

```text
crops:        [B*8, 3, 64, 64]
motion_score: [B*8, 1]
```

Purpose:

- reduce search space
- focus computation on likely targets

## Stage 3: LocateAnything Encoder

Input:

```text
[B*8, 3, 64, 64]
```

Backbone:

- LocateAnything
- pretrained weights
- initially frozen

Output:

```text
[B*8, 256]
```

Motion score concatenation:

```text
256 + 1 -> [B*8, 257]
```

Purpose:

- visual representation learning
- tiny object feature extraction

## Stage 4: Temporal Fusion Transformer

Input:

```text
[B, 5, 8, 257]
```

Architecture:

```text
Transformer encoder
d_model = 257
nhead = 4
layers = 2
ffn = 512
dropout = 0.1
Linear(257 -> 256)
```

Output:

```text
[B, 8, 256]
```

Purpose:

- velocity understanding
- acceleration understanding
- temporal consistency
- occlusion recovery
- false-positive suppression

## Stage 5: Sparse Mixture of Experts

Input:

```text
[B*8, 256]
```

Router:

```text
Linear(256 -> 8) -> Softmax -> Top-2 routing
```

Experts:

```text
8 experts
Each expert:
Linear(256 -> 512) -> ReLU -> Linear(512 -> 256)
```

Output:

```text
[B*8, 256]
```

Purpose:

- scenario specialization
- sparse computation
- edge efficiency

## Stage 6: Detection Head

Input:

```text
[B*8, 256]
```

Output:

```text
[x, y, w, h, obj, conf]
```

Purpose:

- tiny UAV detection

## Training Stage 1: Detector Training

Train:

- MotionCNN
- Detection head

Freeze:

- LocateAnything encoder
- Temporal fusion
- MoE

Dataset:

- Anti-UAV

Loss:

```text
MSE heatmap + BCE objectness + 2.0 * SmoothL1 bounding box
```

Output:

```text
detector_best.pt
```

## Training Stage 2: Temporal Training

Load:

- `detector_best.pt`

Train:

- Temporal fusion transformer

Freeze:

- MotionCNN
- LocateAnything encoder
- Detection head
- MoE

Loss:

```text
MSE heatmap + BCE objectness + 2.0 * SmoothL1 bounding box
```

Output:

```text
temporal_best.pt
```

## Training Stage 3: MoE Training

Load:

- `temporal_best.pt`

Train:

- MoE router
- MoE experts

Freeze:

- MotionCNN
- LocateAnything encoder
- Temporal fusion transformer
- Detection head

Loss:

```text
detection_loss + load_balance_loss
```

Output:

```text
moe_best.pt
```

## Current Scope

DRISHTI-CORE v1.0 supports Anti-UAV detection only. It does not train or run
swarm behavior classification because Anti-UAV does not provide the required
swarm behavior labels.
