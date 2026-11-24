# DRISHTI-CORE v2 — Architectural Refactoring Strategy
### Synthesized from Motion Matters, TAD, and UTTracker Comparative Analysis

---

## Background

After performing start-to-end architectural comparisons of DRISHTI-CORE v2 against three
Anti-UAV CVPRW 2023 papers — **Motion Matters**, **TAD / TAD-Lightning**, and **UTTracker**
— the following critical gaps, bugs, and improvement opportunities have been identified.

This document groups all changes into **Priority Tiers** and maps each change to the exact
file and line it touches.

---

## User Review Required

> [!IMPORTANT]
> Changes in **Priority 1 (Critical Bugs)** are not optional — they are correctness fixes
> that will cause silent training failure or wrong metrics. They should be merged before
> any experiment run.

> [!WARNING]
> Changes in **Priority 2 (High-Impact Architectural)** add new modules or modify
> tensor flow. They will break checkpoint compatibility unless the new weights are
> treated as an additive branch. A config flag guards each one so old behavior is
> preserved.

> [!NOTE]
> Changes in **Priority 3 (Nice-to-Have / Research Exploration)** are optional
> improvements inspired by each comparison paper. They should be gated behind ablation
> experiments.

---

## Open Questions

> [!IMPORTANT]
> **Q1.** Should `CausalTemporalFusion` be fixed by (a) switching to global cross-crop attention
> or (b) adding a learned crop-identity embedding to tolerate index mismatch? Option (a) is
> more principled; option (b) preserves the causal per-crop structure.
>
> **Q2.** For the Sobel gradient channel in LDMI: should it be appended as a 16th channel
> (zero cost to change `MotionCNN` in-channels) or replace the `disappearance/appearance`
> channels (keeps the 15-channel shape)?
>
> **Q3.** The `del augment` line in `AntiUAVDataset` was intentional API cleanup — do you
> want a proper augmentation pipeline implemented now or deferred?

---

## Proposed Changes

---

### Tier 1 — Critical Bug Fixes (Must Fix Before Training)

These four bugs produce **silent wrong behavior** during training and evaluation.

---

#### [MODIFY] [temporal_fusion.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/temporal_fusion.py)

**Bug:** `CausalTemporalFusion.forward` reshapes `[B, T, K, D] → [B×K, T, D]`, then
applies per-crop-index causal attention. This assumes crop `k` at time `t-1` is the same
spatial location as crop `k` at time `t`. Since `CropProposalEngine` re-proposes new crop
centers every frame, indices are NOT aligned — the transformer fuses unrelated spatial
locations.

**Fix:** Inject a learned **crop-identity embedding** (option b, pending Q1) before the
`input_proj`. This embedding is conditioned on `source_label` (MOTION=0, EDGE=1, GRID=2,
GUIDED=3, PAD=4), not crop index. Source labels are stable across time — a MOTION-source
crop at `t` is still "a motion peak" at `t-1`, giving the transformer a meaningful signal
to attend to.

```diff
# temporal_fusion.py — proposed changes
+ self.source_embed = nn.Embedding(5, out_dim)   # 5 source types

def forward(self, sequence: Tensor, source_labels: Tensor | None = None) -> Tensor:
    ...
    x = self.input_proj(x) + self.pos_embed[:, -time:]
+   if source_labels is not None:
+       # source_labels: [B, K] — broadcast over time axis
+       se = self.source_embed(source_labels)          # [B, K, D]
+       se = se.unsqueeze(1).expand(-1, time, -1, -1)  # [B, T, K, D]
+       se = se.permute(0, 2, 1, 3).reshape(B*K, T, D) # [B*K, T, D]
+       x = x + se
```

**Pipeline change:** `pipeline.py:_forward_single` must pass `proposal.source_labels`
to the `temporal` call.

---

#### [MODIFY] [metrics.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/evaluation/metrics.py)

**Bug:** `map50` is computed as `precision * recall` which is mathematically wrong.
mAP@50 is the area under the Precision-Recall curve, not a single point product.

**Fix:** Use the correct trapezoid-rule AUC over the sorted PR curve.

```diff
- map50 = precision * recall   # WRONG
+ # Sort by recall, integrate with trapezoid rule
+ sorted_idx = recall.argsort()
+ map50 = torch.trapz(precision[sorted_idx], recall[sorted_idx])
```

---

#### [MODIFY] [dataset.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/data/dataset.py)

**Bug (line 425):** `del augment` silently discards the augmentation flag.
`AntiUAVDataset.__init__` advertises an `augment: bool` parameter but immediately deletes
it, meaning **no data augmentation ever runs during training**.

**Fix:** Implement a proper augmentation pipeline. Minimal version:
- Random horizontal flip (p=0.5)
- Random brightness/contrast jitter (±10%)
- Random temporal reversal of the frame clip (p=0.3)
- Box coordinates must be updated consistently with frame transforms.

```diff
- del augment
+ self.augment = augment  # store; apply in __getitem__
```

---

#### [MODIFY] [stage_losses.py — assign_crops](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/stage_losses.py#L57-L90)

**Bug (lines 83-88):** Target box regression computes `crop_scale` by dividing the
*predicted* global box size by the *predicted* crop-relative box size. This is a
chicken-and-egg problem — the target should be constructed entirely from GT geometry,
not from the model's current (noisy) predictions.

**Fix:** Compute the crop-relative GT box from geometry alone:

```diff
- global_size = boxes_cpu[batch_idx, crop_idx_item, 2:].clamp_min(1e-6)
- crop_size = crop_boxes_cpu[batch_idx, crop_idx_item, 2:].clamp_min(1e-6)
- crop_scale = global_size / crop_size
- rel_xy = (gt[:2] - centers_cpu[batch_idx, crop_idx_item]) / crop_scale + 0.5
- rel_wh = gt[2:] / crop_scale
+ # Geometry-only: use crop_size pixels in normalized coords
+ crop_frac_w = config.crop_size / float(frame_width)   # passed in
+ crop_frac_h = config.crop_size / float(frame_height)
+ rel_xy = (gt[:2] - centers_cpu[batch_idx, crop_idx_item])
+ rel_xy = rel_xy / torch.tensor([crop_frac_w, crop_frac_h]) + 0.5
+ rel_wh = gt[2:] / torch.tensor([crop_frac_w, crop_frac_h])
```

---

### Tier 2 — High-Impact Architectural Improvements

These are research-grade improvements that directly address weaknesses identified by
comparing against TAD, Motion Matters, and UTTracker.

---

#### [MODIFY] [detection_head.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/detection_head.py) — Add Sub-Pixel Center Offset Head (TAD-inspired)

**Motivation:** TAD uses an explicit `(Δx, Δy)` offset head to correct quantization error
from spatial downsampling. DRISHTI's `DetectionHead` regresses a full `[cx, cy, w, h]` box
but the crop center itself (from `CropProposalEngine`) has discretization error from the
`112×112` heatmap grid (factor of 4 from the full `448×448` frame).

**Change:** Add a 2-output `offset_head` branch that predicts a sub-pixel correction
`(Δcx, Δcy) ∈ [-0.5, 0.5]` relative to the grid cell. Apply it inside
`_boxes_to_global` in `pipeline.py`.

```python
# New branch in DetectionHead
self.offset_head = nn.Sequential(
    nn.LayerNorm(feature_dim),
    nn.Linear(feature_dim, 2),
    nn.Tanh(),   # output in [-1, 1], scale to [-0.5, 0.5] heatmap cells
)

def forward(self, features):
    logits = self.objectness_head(features)
    boxes = self.box_head(features)
    offsets = self.offset_head(features) * 0.5  # sub-pixel correction
    return logits, boxes, offsets
```

Config flag: `use_subpixel_offset: bool = True`

---

#### [MODIFY] [ldmi.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/ldmi.py) — Add Sobel Gradient Edge Channel (Motion-Matters-inspired)

**Motivation:** Motion Matters' key insight was that `Canny(Diff_i)` provided crisp
boundary definitions that boosted recall on tiny targets. LDMI currently provides signed
residuals but no explicit high-frequency gradient channel.

**Change:** Compute a parameter-free Sobel gradient magnitude on `d_new` and append it
as a **16th channel**. This requires `MotionCNN.in_channels` to accept 16 instead of 15,
controlled by a config flag.

```python
# In LocalDifferentialMotion.forward():
# After computing d_new:
sobel_x = F.conv2d(d_new.mean(1, keepdim=True), SOBEL_X.to(d_new), padding=1)
sobel_y = F.conv2d(d_new.mean(1, keepdim=True), SOBEL_Y.to(d_new), padding=1)
edge = (sobel_x**2 + sobel_y**2).sqrt()  # [B, 1, H, W]
# Append to output cat:
return torch.cat([r_old, m_old, s_old, f_curr, s_new, m_new, r_new,
                  disappearance, appearance, edge], dim=1)  # 16 channels
```

SOBEL_X and SOBEL_Y are constant tensors, not learned parameters.
Config flag: `ldmi_use_sobel_edge: bool = False` (default off for backward compat).
Config property update: `motion_input_channels` must account for the extra channel.

---

#### [MODIFY] [crop_proposal.py — _get_motion_centers](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/crop_proposal.py#L51-L64) — GPU Max-Pool Peak Extraction (TAD-inspired)

**Motivation:** TAD achieves 342 FPS by replacing CPU NMS with a `3×3` Max Pooling peak
filter on the GPU heatmap. DRISHTI's `_get_motion_centers` already uses this pattern
(line 58: `F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)`) — but the
surrounding code still does a CPU-bound `torch.topk` after a `torch.where`. This can
be GPU-only.

**Change:** The existing logic is 90% correct. Refine to ensure no implicit CPU sync:
- Replace `peaks.flatten(1)` → `peaks.view(batch, -1)` (identical, but explicit)
- Ensure `scores` and `indices` stay on GPU until after `centers` are computed
- Verify `bool(tensor)` calls do not force sync in hot path

This is a micro-optimization — mark as cleanup.

---

#### [MODIFY] [motion_cnn.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/motion_cnn.py) — Lightweight FPN Structure (Motion-Matters-inspired)

**Motivation:** Motion Matters used multi-resolution spatial resizing (0.5×, 1×, 2×) to
detect drones of varying apparent sizes. DRISHTI uses a single sequential `MotionCNN` with
stride-2 downsampling, which loses multi-scale response.

**Change:** Add an optional multi-scale branch to `MotionCNN` that fuses the feature map
at two additional resolutions (`56×56` and `28×28`) before generating the final `112×112`
heatmap. This is a lightweight FPN-style merge.

```
LDMI [B, 15, 448, 448]
  └─► Conv(s=2) → [B, 32, 224, 224]   scale_4
      └─► Conv(s=2) → [B, 64, 112, 112] scale_8  ← main heatmap path
          └─► Conv(s=2) → [B, 64, 56, 56]  scale_16
              └─► Upsample → [B, 64, 112, 112]
              ↕ Lateral sum
          [B, 64, 112, 112]
          └─► Conv(1×1) → [B, 1, 112, 112] final heatmap
```

Parameter overhead: ~15K additional params. Config flag: `motion_cnn_use_fpn: bool = False`.

---

#### [MODIFY] [moe.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/moe.py) — Source-Conditioned Expert Bias (novel)

**Motivation (synthesized):** DRISHTI uses 8 MoE experts for 8 crop tokens. As the
Skeptic/Reviewer pointed out, if num_experts == num_crops the router may degenerate to
1-to-1 assignment. Inject a source-type bias into the router logits to break symmetry and
specialize experts by crop source (MOTION, EDGE, GRID, GUIDED).

**Change:** Add a 4×8 source-to-expert bias matrix (32 params) added to router logits
before softmax. During training (Stage 3) this is learned jointly.

```python
self.source_bias = nn.Embedding(5, num_experts)  # 5 source types → 8 expert logits

def forward(self, x, source_labels=None):
    router_logits = self.router(x_flat)
    if source_labels is not None:
        router_logits = router_logits + self.source_bias(source_labels.reshape(-1))
```

Config flag: `moe_use_source_bias: bool = True`

---

### Tier 3 — Nice-to-Have / Exploratory

These require separate ablation experiments before merging into main.

---

#### Add Morphological Dilation Layer to MotionCNN (UTTracker-inspired)

UTTracker's DSOD uses morphological Open/Close operations to sharpen sub-6×6 pixel
drone peaks from background noise. Add an optional `MaxPool2d(3,1,1)` → `Sigmoid`
layer after the final `Conv(1×1)` in `MotionCNN` to mimic morphological dilation.
This sharpens heatmap peaks for very small targets.

Config flag: `motion_cnn_morphological: bool = False`

---

#### Lightweight Homography Fallback in LDMI (UTTracker-inspired)

UTTracker uses LoFTR + RANSAC homography matrix H for fast camera pan correction.
LDMI handles this parameter-free, but only for rotational flow. For fast translational
pans (>30px/frame), LDMI may bleed through.

**Proposal:** Detect pan magnitude by computing the mean of `|d_new|` over the full
frame. If it exceeds a threshold (e.g. >0.05 pixel intensity shift), optionally invoke
OpenCV ORB + RANSAC (<2ms) to warp the frame before LDMI. This is a CPU pre-step,
not part of the GPU pipeline.

This is complex and optional — defer to ablation.

---

#### Data Augmentation Pipeline (dataset.py)

Since the `del augment` bug disables all augmentation, implement a proper pipeline:

| Augmentation | Probability | Note |
|---|---|---|
| Horizontal flip | 0.5 | Mirror box cx |
| Temporal reversal | 0.3 | Reverse frame sequence |
| Brightness jitter ±10% | 0.5 | Clamp to [0,1] |
| Gaussian noise σ=0.01 | 0.3 | Simulates sensor noise |
| Scale jitter 0.8–1.2× | 0.3 | Crop and resize |

All transforms must be applied consistently across all T frames of the clip and the
corresponding box annotations.

---

## Implementation Order

Execute in this order to avoid regressions:

```
Step 1. Fix metrics.py (mAP computation)          — 5 min, zero risk
Step 2. Fix dataset.py (del augment → store)       — 10 min, zero risk
Step 3. Fix stage_losses.py (box target geometry)  — 20 min, test with synthetic dataset
Step 4. Fix temporal_fusion.py (source embedding)  — 30 min, test shapes
Step 5. Add subpixel offset head (detection_head)  — 20 min, additive
Step 6. Add Sobel channel to ldmi.py               — 30 min, config gated
Step 7. Add source bias to moe.py                  — 15 min, additive
Step 8. Add FPN to motion_cnn.py                   — 45 min, config gated
Step 9. Data augmentation pipeline                 — 1–2 hours
Step 10. Ablation experiments for Tier 3
```

---

## Verification Plan

### Automated Tests
```bash
# Smoke run (should complete without NaN/error):
python -m drishti_v2.experiments.smoke_run

# Shape test — confirm no tensor dimension mismatches after changes:
python -c "
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.models.config import DRISHTIConfig
import torch
cfg = DRISHTIConfig()
model = DRISHTIPipeline(cfg)
frames = torch.randn(2, 5, 3, 448, 448)
out = model(frames, frame_index=0)
print('objectness_logits:', out.objectness_logits.shape)  # expect [2, 8, 1]
print('boxes:', out.boxes.shape)                          # expect [2, 8, 4]
print('PASS')
"

# Loss test — confirm no NaN losses on synthetic data:
python -c "
from drishti_v2.data.dataset import SyntheticAntiUAVDataset
from drishti_v2.training.stage_losses import StageLossFactory
# ... run one forward+loss step
"
```

### Manual Verification
- After fixing `metrics.py`: confirm `map50 ≤ 1.0` and non-trivially > `precision * recall`
- After fixing `stage_losses.py`: inspect `box_targets` values — they should be in `[0, 1]` and geometrically reasonable
- After `temporal_fusion` fix: confirm the causal mask is still upper-triangular (no future leakage)
- After `detection_head` offset branch: confirm `offsets.abs().max() ≤ 0.5`
