# DRISHTI-CORE v2: Master Implementation Plan

---

# Section 1: Complete Model Architecture

## 1.1 Full Forward Pass Data Flow

```
Input: frames [B, T=5, C=3, H=448, W=448]
                │
    ┌───────────▼───────────────────────────────────────────────────┐
    │  Per-frame loop  (t = 0 → T−1)                               │
    │                                                               │
    │  Step 1: Triplet Construction                                 │
    │  [B, 9, 448, 448]  ← cat(f_{t-2}, f_{t-1}, f_t)            │
    │                                                               │
    │  Step 2: LDMI v2  (parameter-free)                           │
    │  [B, 15, 448, 448] ← signed residuals + magnitudes           │
    │                       + scale maps + transition cues          │
    │                                                               │
    │  Step 3: MotionCNN                                            │
    │  [B, 1, 112, 112]  ← motion heatmap                         │
    │                                                               │
    │  Step 4: MotionGate                                           │
    │  [B]               ← motion confidence score                 │
    │           │                                                   │
    │      trust? ──── YES ──► CropProposalEngine (selective K=8)  │
    │           └──── NO  ──► CropProposalEngine (dense K=16)     │
    │                                                               │
    │  Step 5: CropEncoder                                          │
    │  [B, K, 256]       ← per-crop CNN features                  │
    │                                                               │
    │  Step 6: Augment                                             │
    │  [B, K, 257]       ← cat(encoded, heatmap_score)            │
    └───────────────────────────────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────────────────┐
    │  Step 7: CausalTemporalFusion                                 │
    │  [B, T, K, 257] → [B, K, 256]                               │
    │  Causal transformer: each crop-track reads its past          │
    └───────────────────────────────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────────────────┐
    │  Step 8: SparseMoE                                            │
    │  [B, K, 256] → [B, K, 256]                                  │
    │  Top-2 of 8 experts per crop token                           │
    └───────────────────────────────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────────────────┐
    │  Step 9: DetectionHead                                        │
    │  [B, K, 256] → objectness [B, K, 1] + crop_boxes [B, K, 4] │
    └───────────────────────────────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────────────────┐
    │  Step 10: Global Box Mapping                                  │
    │  crop_boxes [B, K, 4] → boxes [B, K, 4]  (full-frame coords)│
    └───────────────────────────────────────────────────────────────┘

Output: PipelineOutput (heatmap, boxes, objectness_logits, ...)
```

---

## 1.2 Module-by-Module Architecture Reference

### Module 1: LocalDifferentialMotion (LDMI v2)

**File**: [ldmi.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/ldmi.py)
**Parameters**: 0 (fully parameter-free)
**Input**: `[B, 9, H, W]` — triplet of 3 frames concatenated along channel axis
**Output**: `[B, 15, H, W]`

| Op | Description | Output shape |
|---|---|---|
| Split triplet | `f_old, f_prev, f_curr = triplet.split(3, dim=1)` | `3 × [B, 3, H, W]` |
| `d_old = f_prev − f_old` | Raw frame difference, past interval | `[B, 3, H, W]` |
| `d_new = f_curr − f_prev` | Raw frame difference, recent interval | `[B, 3, H, W]` |
| `avg_pool2d(d, k)` for k ∈ {7,15,31,63} | Local mean at 4 scales | `4 × [B, 3, H, W]` |
| `r = d − local_mean` (signed) | Signed residual at each scale | `4 × [B, 3, H, W]` |
| `argmax(abs(r), dim=scales)` | Pick best-responding scale per pixel | `[B, 3, H, W]` index |
| `r_old, r_new` | Sign-preserving max-abs residual | `2 × [B, 3, H, W]` |
| `m_old = ‖d_old‖₂`, `m_new = ‖d_new‖₂` | Raw motion magnitude | `2 × [B, 1, H, W]` |
| `s_old, s_new` = normalised scale index | Object size hint | `2 × [B, 1, H, W]` |
| `D = relu(r̂_old − r̂_new)` | Disappearance (occlusion cue) | `[B, 1, H, W]` |
| `A = relu(r̂_new − r̂_old)` | Appearance (new object cue) | `[B, 1, H, W]` |
| `cat([r_old, m_old, s_old, f_curr, s_new, m_new, r_new, D, A])` | Final output | `[B, 15, H, W]` |

**Output channel breakdown** (RGB, C=3):

```
Ch  0– 2:  r_old  (signed motion contrast, past interval)
Ch  3:     m_old  (raw L2 motion magnitude, past interval)
Ch  4:     s_old  (normalised best-scale index, past interval)
Ch  5– 7:  f_curr (current frame appearance)
Ch  8:     s_new  (normalised best-scale index, recent interval)
Ch  9:     m_new  (raw L2 motion magnitude, recent interval)
Ch 10–12:  r_new  (signed motion contrast, recent interval)
Ch 13:     D      (disappearance / occlusion onset)
Ch 14:     A      (appearance / new object onset)
```

---

### Module 2: MotionCNN

**File**: [motion_cnn.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/motion_cnn.py)
**Parameters**: ~50K (updated first layer adds ~1,728 params)
**Input**: `[B, 15, 448, 448]`
**Output**: `[B, 1, 112, 112]`

| Layer | Op | Input → Output | Notes |
|---|---|---|---|
| conv1 | Conv2d(15→32, k=3, s=2, p=1) + BN + ReLU | `[B,15,448,448]→[B,32,224,224]` | Was 9→32 |
| conv2 | Conv2d(32→64, k=3, s=2, p=1) + BN + ReLU | `[B,32,224,224]→[B,64,112,112]` | Unchanged |
| conv3 | Conv2d(64→64, k=3, s=1, p=1) + BN + ReLU | `[B,64,112,112]→[B,64,112,112]` | Unchanged |
| conv4 | Conv2d(64→1, k=1) + Sigmoid | `[B,64,112,112]→[B,1,112,112]` | Unchanged |

**Training supervision**: GT heatmaps are Gaussian blobs (σ=2) at GT box centers, at 1/4 resolution.

---

### Module 3: MotionGate

**File**: `drishti_v2/models/motion_gate.py` *(NEW)*
**Parameters**: 129
**Input**: `[B, 1, 112, 112]` — heatmap from MotionCNN
**Output**: `[B]` — confidence ∈ (0, 1)

| Step | Op | Output |
|---|---|---|
| Flatten heatmap | `h = heatmap.view(B, -1)` | `[B, 12544]` |
| Extract 6 stats | max, mean, std, entropy, top1−top2 gap, active fraction | `[B, 6]` |
| MLP layer 1 | Linear(6→16) + ReLU | `[B, 16]` |
| MLP layer 2 | Linear(16→1) + Sigmoid | `[B, 1]` |
| Squeeze | `.squeeze(-1)` | `[B]` |

**Decision**: if `confidence < threshold (default 0.5)` → use dense crop mode.

---

### Module 4: CropProposalEngine

**File**: [crop_proposal.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/crop_proposal.py)
**Parameters**: 0 (pure algorithmic)

**Selective mode** (normal, K=8):
```
Priority fill order:
  1. GUIDED  (tracker predictions)  → up to 6 slots
  2. GRID    (fixed positions)       → up to 4 slots (every scan_period=4 frames)
  3. EDGE    (frame borders)         → up to 2 slots
  4. MOTION  (heatmap NMS peaks)     → remaining slots
  5. PAD     (frame center)          → fill remaining
```

**Dense mode** (fallback, K=16 for grid_size=4):
```
Priority fill order:
  1. GUIDED  (tracker predictions)   → up to available slots
  2. GRID    (4×4 uniform grid)      → fill remaining
     Positions: (i/(n+1), j/(n+1)) for i,j ∈ {1,2,3,4}
```

**Crop extraction**: `_extract_crops(frame, centers)` — bilinear interpolate a 64×64 patch from the full-res frame at each center location.

**Output**:
- `crops`: `[B×K, 3, 64, 64]`
- `centers`: `[B, K, 2]` — normalised (cx, cy)
- `scores`: `[B, K]` — heatmap value at each center
- `source_labels`: `[B, K]` — integer in {0=MOTION, 1=EDGE, 2=GRID, 3=GUIDED, 4=PAD}

---

### Module 5: CropEncoder

**File**: [crop_encoder.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/crop_encoder.py)
**Parameters**: ~300K
**Input**: `[B×K, 3, 64, 64]`
**Output**: `[B, K, 256]`

| Layer | Op | Input → Output |
|---|---|---|
| conv1 | Conv2d(3→32, k=3, s=1, p=1) + BN + ReLU | `[B×K,3,64,64]→[B×K,32,64,64]` |
| conv2 | Conv2d(32→64, k=3, s=2, p=1) + BN + ReLU | `[B×K,32,64,64]→[B×K,64,32,32]` |
| conv3 | Conv2d(64→128, k=3, s=2, p=1) + BN + ReLU | `[B×K,64,32,32]→[B×K,128,16,16]` |
| pool | AdaptiveAvgPool2d(1) | `[B×K,128,16,16]→[B×K,128,1,1]` |
| flatten | `.view(B×K, 128)` | `[B×K,128]` |
| fc | Linear(128→256) + ReLU | `[B×K,256]` |
| reshape | `.view(B, K, 256)` | `[B,K,256]` |

After encoding: append heatmap score → `[B, K, 257]`.

---

### Module 6: CausalTemporalFusion

**File**: [temporal_fusion.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/temporal_fusion.py)
**Parameters**: ~500K
**Input**: `[B, T=5, K=8, 257]`
**Output**: `[B, K=8, 256]`

| Layer | Op | Input → Output |
|---|---|---|
| reshape | `[B,T,K,D]→[B*K,T,D]` | `[B*K, 5, 257]` |
| input_proj | Linear(257→256) + pos_embed | `[B*K, 5, 256]` |
| causal mask | upper-triangular bool mask | `[5, 5]` |
| TransformerEncoder | 2 layers, nhead=4, ffn=512 | `[B*K, 5, 256]` |
| extract present | `encoded[:, -1]` | `[B*K, 256]` |
| LayerNorm | normalize | `[B*K, 256]` |
| reshape | `.view(B, K, 256)` | `[B, K, 256]` |

**Current limitation** (noted for future work): reshaping forces crop index k at time t to attend only to crop k at past times. The index assignment is arbitrary and can mismatch when CropProposalEngine re-proposes different locations each frame.

---

### Module 7: SparseMoE

**File**: [moe.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/moe.py)
**Parameters**: ~1.1M (router + 8 experts)
**Input**: `[B, K=8, 256]`
**Output**: `[B, K=8, 256]` + `MoEDiagnostics`

| Sub-module | Architecture | Params |
|---|---|---|
| Router | Linear(256→8, bias=False) | 2,048 |
| Expert × 8 | Linear(256→512) + GELU + Dropout(0.1) + Linear(512→256) | 8 × 131,584 = 1,052,672 |
| **Total** | — | **~1.05M** |

**Routing forward pass**:
1. Flatten: `x_flat = x.reshape(B*K, 256)` → `[N=B*K, 256]`
2. Router logits: `logits = router(x_flat)` → `[N, 8]`
3. Probabilities: `probs = softmax(logits)` → `[N, 8]`
4. Top-2 selection: `top_probs, top_indices = probs.topk(2)` → `[N, 2]`
5. Normalise: `top_weights = top_probs / sum(top_probs)`
6. Expert computation: `out[i] = w1 * expert_a(x_i) + w2 * expert_b(x_i)`
7. Reshape: `.view(B, K, 256)`

**Diagnostics** (all computed here, passed to loss functions):
- `balance_loss` (current, used everywhere)
- `router_entropy` (existing logging field)
- `router_logits` (NEW — needed for z-loss in Stage 3)

---

### Module 8: DetectionHead

**File**: [detection_head.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/detection_head.py)
**Parameters**: ~130K
**Input**: `[B, K, 256]`
**Output**: `objectness_logits [B, K, 1]` + `crop_boxes [B, K, 4]`

| Branch | Architecture |
|---|---|
| Objectness | LayerNorm(256) → Linear(256→1) → (raw logit) |
| Box | LayerNorm(256) → Linear(256→256) → GELU → Linear(256→4) → Sigmoid |

Box output is in crop-relative normalised coordinates `[cx, cy, w, h] ∈ [0, 1]`.

---

### Module 9: SimpleTracker (Inference Only)

**File**: [tracker.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/tracker/tracker.py)
**Parameters**: 0 (algorithmic)

At inference only — not used during training.

| Step | Operation |
|---|---|
| `predict()` | Move each track center by its velocity estimate: `center += velocity` |
| `update(boxes, logits)` | Match detections to tracks by Euclidean distance (threshold=0.15) |
| New tracks | Birth for unmatched detections with score > 0.3 |
| Dead tracks | Kill tracks not matched for >15 frames (`max_coast=15`) |
| `get_guided_centers()` | Return `[1, num_tracks, 2]` of predicted positions → CropProposalEngine GUIDED slot |

---

# Section 2: LDMI v2 + Adaptive Gating Changes

## 2.1 Summary of All Code Changes (LDMI + Gating)

| File | Action | Lines Affected | New Params |
|---|---|---|---|
| [ldmi.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/ldmi.py) | Full rewrite | All 46 lines | 0 |
| [motion_cnn.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/motion_cnn.py) | `in_channels` 9→15 | Line 16 | +1,728 |
| **motion_gate.py** | **NEW FILE** | — | 129 |
| [crop_proposal.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/crop_proposal.py) | Add `forward_dense()` | After line 64 | 0 |
| [pipeline.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/pipeline.py) | Wire gate + adaptive mode | Lines 37–170 | 0 |
| [config.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/config.py) | Add 4 new fields, update scales | Lines 18–30 | — |
| [stage_control.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/stage_control.py) | Include `motion_gate` in stage1 | Line 22 | — |
| [\_\_init\_\_.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/__init__.py) | Export `MotionGate` | Lines 3–7 | — |

Mathematical derivations: see [implementation_plan.md](file:///C:/Users/jaygo/.gemini/antigravity/brain/e6394a49-7659-4c6f-8105-8369feb56fae/implementation_plan.md) sections 1.2–1.5.

---

# Section 3: Stage-Specific Loss Functions

## 3.1 Mathematical Definition of All Loss Primitives

### 3.1.1 Sigmoid Focal Loss (replaces BCE in all stages)

**Problem with current BCE at [losses.py line 71](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/losses.py#L71)**:

```python
cls_loss = F.binary_cross_entropy_with_logits(output.objectness_logits, labels)
```

BCE treats all K crops equally. With K=8 and typically 1 positive crop (the one containing the UAV), the ratio is 7:1 negative:positive. The network learns to predict "no object" everywhere — this minimises loss because 7/8 labels are zero and predicting zero is always correct for them.

**Focal Loss** (Lin et al., RetinaNet 2017):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t = \sigma(\text{logit})$ if label=1, else $p_t = 1 - \sigma(\text{logit})$, and:
- $(1 - p_t)^\gamma$ is the **modulating factor** — when the model predicts correctly with high confidence ($p_t \to 1$), this term $\to 0$ and the loss is down-weighted. When the model is wrong or uncertain, the loss stays high.
- $\alpha_t$ is the class-balance weight: $\alpha$ for positives, $1-\alpha$ for negatives.

**Effect**: Easy, well-classified negatives contribute negligibly to gradient. Hard misclassified examples dominate training.

**With our crop setup** (K=8, ~1-2 positives):
- At initialization, model predicts $p \approx 0.5$ everywhere. Negative BCE gradient is $0.5$ per sample → all 7 negatives contribute as much as the 1 positive.
- With Focal ($\gamma=2$): same initialization gives modulating factor $(1-0.5)^2 = 0.25$ → negatives contribute $0.25\times$ as much. After a few epochs, easy negatives are down-weighted further.

**Recommended values**: $\gamma = 2.0$, $\alpha = 0.25$ (positives are up-weighted since they're rare).

**Numerically stable implementation**:
$$\text{FL}(\text{logit}, y) = \alpha_t \cdot (1-p_t)^\gamma \cdot \max(\text{logit}, 0) - \text{logit} \cdot y + \log(1 + e^{-|\text{logit}|})$$

This avoids computing $\sigma(\text{logit})$ directly, which can overflow.

---

### 3.1.2 Heatmap Focal Loss (replaces MSE on heatmaps)

**Problem with current MSE at [losses.py line 68](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/losses.py#L68)**:

```python
heatmap_loss = F.mse_loss(output.heatmap, gt_heatmap)
```

MSE treats all pixels equally. The GT heatmap has Gaussian peaks (σ=2) at GT centers and zeros everywhere else. The heatmap is 112×112 = 12,544 pixels. The Gaussian with σ=2 covers roughly $\pi \times 6^2 \approx 113$ pixels. So ~99% of pixels are zero in the GT.

MSE penalises any predicted pixel > 0 in the background equally with any predicted pixel < 1 at the peak center. There is no distinction between "almost-background" and "peak region."

**CornerNet/CenterNet Heatmap Focal Loss** (Law & Deng 2018):

For each pixel $(x,y)$, let $\hat{y}_{xy}$ be the predicted heatmap value and $y_{xy}$ be the GT value:

$$\mathcal{L}_{\text{hm}} = \frac{-1}{N} \sum_{x,y} \begin{cases} (1 - \hat{y}_{xy})^\alpha \log(\hat{y}_{xy}) & \text{if } y_{xy} = 1 \\ (1 - y_{xy})^\beta (\hat{y}_{xy})^\alpha \log(1 - \hat{y}_{xy}) & \text{otherwise} \end{cases}$$

where $N$ = number of GT keypoints (objects), $\alpha = 2$, $\beta = 4$.

**Key mechanics**:
- **Peak pixels** ($y_{xy} = 1$): Standard log-loss, but $(1 - \hat{y})^\alpha$ down-weights easy cases (when $\hat{y} \approx 1$ already, $(1-1)^2 = 0$ contribution).
- **Background pixels near the peak** ($y_{xy} \in (0, 1)$ from Gaussian falloff): $(1 - y_{xy})^\beta$ suppresses the penalty. A pixel with $y_{xy} = 0.5$ (half-way up the Gaussian) contributes $(1-0.5)^4 = 0.0625\times$ the normal penalty.
- **Background pixels far from peaks** ($y_{xy} \approx 0$): $(1 - 0)^4 = 1$ full penalty. The network must predict near-zero here.

**Why $\beta=4$**: Aggressively suppresses penalty in the Gaussian falloff region. The model is not punished for "leaking" some heatmap activation near (but not at) a GT center.

---

### 3.1.3 Complete IoU Loss (replaces Smooth L1)

**Problem with current Smooth L1 at [losses.py line 74](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/losses.py#L74)**:

```python
bbox_loss = F.smooth_l1_loss(output.crop_boxes[positive], box_targets[positive])
```

Smooth L1 treats each coordinate independently. It does not know that $(cx, cy, w, h)$ form a geometric box — a small error in $cx$ has the same cost as the same error in $w$, regardless of the IoU consequence.

**CIoU Loss** (Zheng et al., 2020):

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha_v \cdot v$$

where:

**IoU term**:
$$\text{IoU} = \frac{|\mathbf{b} \cap \mathbf{b}^{gt}|}{|\mathbf{b} \cup \mathbf{b}^{gt}|}$$

**Center distance penalty**:
$$\frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} = \frac{(cx - cx^{gt})^2 + (cy - cy^{gt})^2}{c^2}$$

where $c$ = diagonal of the smallest box enclosing both predictions and GT.

This term is **zero only when centers coincide**, and is normalized by the enclosing diagonal so it's scale-invariant. It drives the predicted center toward the GT center even when IoU = 0 (no overlap at all — Smooth L1 still provides gradient, but it's coordinate-wise, not geometry-aware).

**Aspect ratio consistency**:
$$v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$$

$$\alpha_v = \frac{v}{(1 - \text{IoU}) + v}$$

$v$ measures the difference in aspect ratio. $\alpha_v$ is an adaptive trade-off: when IoU is already high, aspect ratio correction is emphasized. When IoU is low (boxes don't overlap), center alignment is prioritized.

**Why CIoU over GIoU or DIoU**:
- GIoU: only adds a penalty for the non-overlapping area — doesn't penalize misaligned centers when boxes overlap.
- DIoU: adds center distance but no aspect ratio — can converge to wrong proportions.
- CIoU: center distance + aspect ratio + IoU — all three geometric properties.

**Our setting**: Boxes are in crop-relative $[cx, cy, w, h] \in [0, 1]^4$ coordinates. CIoU is invariant to scale, so the crop-relative space is fine. The "enclosing diagonal" $c$ is computed in this same normalised space.

---

### 3.1.4 Motion Displacement Loss (Stage 1 only)

**Goal**: Validate that the LDMI + MotionCNN pipeline captures the **direction and magnitude** of UAV motion, not just its position at a single frame.

**Formulation**:

For consecutive frames $t$ and $t-1$ within a clip, let:
- $\hat{\mathbf{p}}_t \in \mathbb{R}^2$ = predicted heatmap peak location at time $t$ (argmax of heatmap, in normalised coords)
- $\mathbf{g}_t \in \mathbb{R}^2$ = GT box center at time $t$ (from annotation)

Define **predicted displacement** between consecutive frames:
$$\hat{\mathbf{d}}_t = \hat{\mathbf{p}}_t - \hat{\mathbf{p}}_{t-1}$$

Define **GT displacement**:
$$\mathbf{d}_t = \mathbf{g}_t - \mathbf{g}_{t-1}$$

**Motion Displacement Loss** over all T−1 consecutive pairs:
$$\mathcal{L}_{\text{motion}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left\|\hat{\mathbf{d}}_t - \mathbf{d}_t\right\|_2^2$$

**What this forces**: If the heatmap peak at t=0 is at (0.45, 0.62) and at t=1 is at (0.47, 0.64), the predicted displacement is (+0.02, +0.02). If the GT displacement is (+0.02, +0.02), the loss is zero. If the peak drifts due to noise/false response, the loss is nonzero.

**Implementation requirement**: The loss must receive targets for **all frames**, not just the last frame. Current `_last_targets()` pattern at [losses.py line 27](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/losses.py#L27) must be extended.

> [!WARNING]
> `argmax` on the heatmap is non-differentiable. Two options:
> 1. **Soft argmax** (differentiable): $\hat{\mathbf{p}} = \sum_{xy} \text{softmax}(\mathcal{H} / \tau)_{xy} \cdot (x, y)$. With small temperature $\tau$, this approximates argmax while remaining differentiable.
> 2. **Stop-gradient on peak, L2 on heatmap near peak**: Compute non-differentiable argmax, then penalise the heatmap value at that location relative to adjacent locations. Less clean but faster.
>
> **Recommendation**: Option 1 (soft argmax, $\tau = 0.1$).

---

### 3.1.5 Temporal Consistency Loss (Stage 2 only)

**Goal**: Penalise objectness score flickering — a crop that contains the UAV at time $t$ should also have high objectness at time $t-1$ if the UAV was there.

**Formulation**:

Let $s_t^{(k)} = \sigma(\text{logit}_t^{(k)}) \in (0,1)$ be the objectness score for crop $k$ at time $t$.

$$\mathcal{L}_{\text{consist}} = \frac{1}{K(T-1)} \sum_{k=1}^{K} \sum_{t=1}^{T-1} \left(s_t^{(k)} - s_{t-1}^{(k)}\right)^2$$

This is the **squared score difference** between adjacent frames for the same crop index.

**Issue with this formulation**: Crop $k$ at time $t$ and crop $k$ at time $t-1$ may be at different spatial locations (the re-proposal problem). A score change may be geometrically valid if the crop moved.

**Mitigation**: Weight the loss by the spatial proximity of the two crops:

$$w_{t,k} = \exp\left(-\frac{\|\mathbf{c}_t^{(k)} - \mathbf{c}_{t-1}^{(k)}\|_2^2}{2\sigma_{\text{spatial}}^2}\right)$$

Crops that moved far (different proposals) contribute little. Crops that stayed near the same location (stable proposals or GUIDED crops) contribute fully.

$$\mathcal{L}_{\text{consist}} = \frac{1}{K(T-1)} \sum_{k,t} w_{t,k} \cdot \left(s_t^{(k)} - s_{t-1}^{(k)}\right)^2$$

**Effect**: Reduces detection flickering ("object present → absent → present" oscillation over frames), which directly improves tracking stability.

---

### 3.1.6 Trajectory Smoothness Loss (Stage 2 only)

**Goal**: Penalise physically impossible accelerations in the predicted box trajectory. UAVs have bounded accelerations — sudden jumps between frames indicate prediction errors, not real motion.

**Formulation**:

For each crop $k$ across frames, define:

$$\Delta_t^{(k)} = \mathbf{box}_t^{(k)} - \mathbf{box}_{t-1}^{(k)} \quad \text{(velocity at time } t\text{)}$$

$$\Delta^2_t{}^{(k)} = \Delta_t^{(k)} - \Delta_{t-1}^{(k)} \quad \text{(acceleration at time } t\text{)}$$

$$\mathcal{L}_{\text{smooth}} = \frac{1}{K(T-2)} \sum_{k=1}^{K} \sum_{t=2}^{T-1} \left\|\Delta^2_t{}^{(k)}\right\|_2^2$$

Minimising this encourages **constant-velocity prediction** — the predicted box positions form a straight trajectory unless the model has strong evidence for curvature.

**Important**: Only apply to positive crops (crops assigned a GT box). Penalising acceleration of background crops is meaningless.

$$\mathcal{L}_{\text{smooth}} = \frac{1}{\sum_k \mathbf{1}[\text{pos}_k] \cdot (T-2)} \sum_{\text{pos } k} \sum_{t=2}^{T-1} \left\|\Delta^2_t{}^{(k)}\right\|_2^2$$

---

### 3.1.7 Router Z-Loss (Stage 3 only)

**Problem**: The router at [moe.py line 69](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/moe.py#L69) can produce arbitrarily large logits. After softmax, large logits cause near-one routing probabilities for the top expert — the router becomes deterministic and stops load-balancing. This also causes numerical instability in the softmax computation (overflow).

**Router Z-Loss** (ST-MoE, Zoph et al. 2022):

$$\mathcal{L}_z = \frac{1}{N} \sum_{i=1}^{N} \left(\log \sum_{j=1}^{E} e^{x_{ij}}\right)^2$$

where $x_{ij}$ are the raw router logits for token $i$ and expert $j$.

**Interpretation**: $\log \sum_j e^{x_j} = \text{logsumexp}(x)$ is the "soft maximum" of the logits. Squaring it and summing penalises large logit magnitudes directly — if all logits are small ($x_j \approx 0$), $\text{logsumexp} \approx \log E$, which is a constant. If any logit is large (e.g., $x_1 = 10$), $\text{logsumexp} \approx 10$ and the squared penalty is 100.

**Mathematical consequence**: Minimising $\mathcal{L}_z$ prevents the router from learning to produce very confident single-expert assignments. The routing distribution stays more spread out, maintaining expert diversity and numerical stability.

**Coefficient**: Literature suggests $\lambda_z \approx 10^{-3}$. Small enough to not override the detection loss, large enough to regularise logit magnitude.

**Implementation in [moe.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/moe.py)** — add to `MoEDiagnostics` and compute before softmax:

```python
# Compute BEFORE softmax (need raw logits)
router_logits = self.router(x_flat)                           # [N, E]
z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() # scalar

probs = torch.softmax(router_logits, dim=-1)
# ... rest unchanged
```

Add `router_z_loss: Tensor` field to `MoEDiagnostics` dataclass.

---

## 3.2 Stage-Specific Loss Compositions

### Stage 1 Loss — Spatial Detector

**What's training**: MotionCNN + CropEncoder + DetectionHead + MotionGate

**Formula**:
$$\mathcal{L}_{\text{Stage1}} = \underbrace{w_{hm} \cdot \mathcal{L}_{\text{HeatmapFocal}}}_{\text{1.0}} + \underbrace{w_{cls} \cdot \mathcal{L}_{\text{SigmoidFocal}}}_{\text{1.0}} + \underbrace{w_{box} \cdot \mathcal{L}_{\text{CIoU}}}_{\text{2.0}} + \underbrace{w_{motion} \cdot \mathcal{L}_{\text{motion}}}_{\text{0.5}} + \underbrace{w_{gate} \cdot \mathcal{L}_{\text{gate}}}_{\text{0.01}}$$

| Term | Weight | Purpose |
|---|---|---|
| $\mathcal{L}_{\text{HeatmapFocal}}$ | 1.0 | Teach MotionCNN to produce clean peaked heatmaps |
| $\mathcal{L}_{\text{SigmoidFocal}}$ | 1.0 | Teach DetectionHead to assign objectness; Focal prevents imbalance collapse |
| $\mathcal{L}_{\text{CIoU}}$ | 2.0 | Teach DetectionHead to regress tight boxes; geometry-aware |
| $\mathcal{L}_{\text{motion}}$ | 0.5 | Validate LDMI+MotionCNN captures motion direction and magnitude |
| $\mathcal{L}_{\text{gate}} = \frac{1}{B}\sum(1 - g)$ | 0.01 | Sparsity: discourage gate from always triggering dense mode |

**Requires**:
- All frame targets in the clip (for $\mathcal{L}_{\text{motion}}$)
- Heatmaps from all timesteps (for soft-argmax peak extraction)

---

### Stage 2 Loss — Temporal Fusion

**What's training**: CausalTemporalFusion (MotionCNN, encoder, head, gate are frozen)

**Formula**:
$$\mathcal{L}_{\text{Stage2}} = \underbrace{w_{hm} \cdot \mathcal{L}_{\text{HeatmapFocal}}}_{\text{0.5}} + \underbrace{w_{cls} \cdot \mathcal{L}_{\text{SigmoidFocal}}}_{\text{1.0}} + \underbrace{w_{box} \cdot \mathcal{L}_{\text{CIoU}}}_{\text{2.0}} + \underbrace{w_{tc} \cdot \mathcal{L}_{\text{consist}}}_{\text{0.3}} + \underbrace{w_{sm} \cdot \mathcal{L}_{\text{smooth}}}_{\text{0.1}}$$

| Term | Weight | Why this weight vs Stage 1 |
|---|---|---|
| $\mathcal{L}_{\text{HeatmapFocal}}$ | 0.5 | MotionCNN is frozen — this still propagates but is read-only signal |
| $\mathcal{L}_{\text{SigmoidFocal}}$ | 1.0 | Detection must still work — same weight |
| $\mathcal{L}_{\text{CIoU}}$ | 2.0 | Box accuracy — same weight |
| $\mathcal{L}_{\text{consist}}$ | 0.3 | New: temporal consistency across frames |
| $\mathcal{L}_{\text{smooth}}$ | 0.1 | New: trajectory smoothness on positive crops |

**Note on heatmap loss in Stage 2**: MotionCNN is frozen, so $\mathcal{L}_{\text{HeatmapFocal}}$ won't update it. However the heatmap loss signal still informs the training loop logging and confirms spatial accuracy hasn't degraded. Weight reduced to 0.5 to keep it as a diagnostic signal without pulling gradient toward frozen layers.

---

### Stage 3 Loss — MoE Routing

**What's training**: SparseMoE (everything else frozen)

**Formula**:
$$\mathcal{L}_{\text{Stage3}} = \underbrace{w_{cls} \cdot \mathcal{L}_{\text{SigmoidFocal}}}_{\text{1.0}} + \underbrace{w_{box} \cdot \mathcal{L}_{\text{CIoU}}}_{\text{2.0}} + \underbrace{w_{bal} \cdot \mathcal{L}_{\text{balance}}}_{\text{0.01}} + \underbrace{w_z \cdot \mathcal{L}_z}_{\text{0.001}}$$

| Term | Weight | Purpose |
|---|---|---|
| $\mathcal{L}_{\text{SigmoidFocal}}$ | 1.0 | MoE must improve or maintain detection accuracy |
| $\mathcal{L}_{\text{CIoU}}$ | 2.0 | MoE must improve or maintain box accuracy |
| $\mathcal{L}_{\text{balance}}$ | 0.01 | Existing load-balancing: $E \sum f_j \bar{p}_j$ |
| $\mathcal{L}_z$ | 0.001 | Router Z-Loss: prevent logit explosion, improve stability |

**Why no heatmap loss in Stage 3**: Heatmap supervision targets MotionCNN which is frozen. Including it adds computation without useful gradient.

**Why no temporal losses in Stage 3**: Temporal modules are frozen. These losses provide no useful gradient for the MoE.

---

### Stage 4 Loss — End-to-End Finetune

**What's training**: Everything

**Formula**:
$$\mathcal{L}_{\text{Stage4}} = \underbrace{0.5 \cdot \mathcal{L}_{\text{HeatmapFocal}}}_{\text{spatial}} + \underbrace{1.0 \cdot \mathcal{L}_{\text{SigmoidFocal}}}_{\text{cls}} + \underbrace{2.0 \cdot \mathcal{L}_{\text{CIoU}}}_{\text{box}} + \underbrace{0.3 \cdot \mathcal{L}_{\text{motion}}}_{\text{motion}} + \underbrace{0.15 \cdot \mathcal{L}_{\text{consist}}}_{\text{temporal}} + \underbrace{0.05 \cdot \mathcal{L}_{\text{smooth}}}_{\text{smooth}} + \underbrace{0.01 \cdot \mathcal{L}_{\text{balance}}}_{\text{moe}} + \underbrace{0.001 \cdot \mathcal{L}_z}_{\text{z-loss}}$$

Auxiliary weights ($\mathcal{L}_{\text{motion}}$, $\mathcal{L}_{\text{consist}}$, $\mathcal{L}_{\text{smooth}}$) are reduced from Stage 1/2 values. In the final joint finetuning, the **detection signal** ($\mathcal{L}_{\text{Focal}}$, $\mathcal{L}_{\text{CIoU}}$) must dominate. Auxiliary terms guide but should not overwhelm.

---

## 3.3 Complete File Change Map

### NEW: [drishti_v2/training/focal_loss.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/focal_loss.py)

```python
"""Sigmoid Focal Loss and Heatmap Focal Loss implementations."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    """
    Sigmoid Focal Loss for binary classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: raw logits [*]
        targets: binary labels in {0, 1} [*]
        gamma: focusing parameter (default 2.0)
        alpha: class balance weight for positives (default 0.25)

    Returns:
        scalar mean loss
    """
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)       # p if y=1, 1-p if y=0
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    modulating = (1 - p_t).pow(gamma)
    # Numerically stable BCE: max(logits,0) - logits*y + log(1+exp(-|logits|))
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = alpha_t * modulating * bce
    return loss.mean()


def heatmap_focal_loss(pred: Tensor, gt: Tensor, alpha: float = 2.0, beta: float = 4.0) -> Tensor:
    """
    CornerNet/CenterNet Heatmap Focal Loss.

    L = -(1/N) * sum_{xy} {
        (1 - pred)^alpha * log(pred)               if gt == 1
        (1 - gt)^beta * pred^alpha * log(1-pred)   otherwise
    }

    Args:
        pred: predicted heatmap [B, 1, H, W] in (0, 1) — after Sigmoid
        gt:   GT heatmap [B, 1, H, W] in [0, 1] — Gaussian blobs at object centers
        alpha: focusing exponent (default 2.0)
        beta:  background suppression exponent (default 4.0)

    Returns:
        scalar mean loss
    """
    pred = pred.clamp(1e-6, 1 - 1e-6)
    pos_mask = (gt == 1.0).float()
    neg_mask = 1.0 - pos_mask
    n = pos_mask.sum().clamp_min(1)

    pos_loss = (1 - pred).pow(alpha) * torch.log(pred) * pos_mask
    neg_loss = (1 - gt).pow(beta) * pred.pow(alpha) * torch.log(1 - pred) * neg_mask

    return -(pos_loss.sum() + neg_loss.sum()) / n
```

---

### NEW: [drishti_v2/training/ciou_loss.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/ciou_loss.py)

```python
"""Complete IoU Loss for bounding box regression."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def ciou_loss(pred_boxes: Tensor, gt_boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """
    CIoU Loss = 1 - IoU + rho^2(b, b_gt)/c^2 + alpha_v * v
    Operates on [cx, cy, w, h] normalised coordinates.

    Args:
        pred_boxes: [N, 4] predicted boxes in [cx, cy, w, h]
        gt_boxes:   [N, 4] GT boxes in [cx, cy, w, h]

    Returns:
        scalar mean CIoU loss
    """
    # Convert to [x1, y1, x2, y2]
    def to_xyxy(b):
        return torch.stack([b[..., 0] - b[..., 2] / 2,
                            b[..., 1] - b[..., 3] / 2,
                            b[..., 0] + b[..., 2] / 2,
                            b[..., 1] + b[..., 3] / 2], dim=-1)

    p_xyxy = to_xyxy(pred_boxes)
    g_xyxy = to_xyxy(gt_boxes)

    # IoU
    inter_x1 = torch.max(p_xyxy[..., 0], g_xyxy[..., 0])
    inter_y1 = torch.max(p_xyxy[..., 1], g_xyxy[..., 1])
    inter_x2 = torch.min(p_xyxy[..., 2], g_xyxy[..., 2])
    inter_y2 = torch.min(p_xyxy[..., 3], g_xyxy[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp_min(0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0)
    inter_area = inter_w * inter_h
    pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
    gt_area = gt_boxes[..., 2] * gt_boxes[..., 3]
    union_area = pred_area + gt_area - inter_area + eps
    iou = inter_area / union_area

    # Enclosing box diagonal
    enc_x1 = torch.min(p_xyxy[..., 0], g_xyxy[..., 0])
    enc_y1 = torch.min(p_xyxy[..., 1], g_xyxy[..., 1])
    enc_x2 = torch.max(p_xyxy[..., 2], g_xyxy[..., 2])
    enc_y2 = torch.max(p_xyxy[..., 3], g_xyxy[..., 3])
    c2 = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    # Center distance
    rho2 = (pred_boxes[..., 0] - gt_boxes[..., 0]).pow(2) + \
           (pred_boxes[..., 1] - gt_boxes[..., 1]).pow(2)

    # Aspect ratio
    v = (4 / math.pi ** 2) * (
        torch.atan(gt_boxes[..., 2] / gt_boxes[..., 3].clamp_min(eps)) -
        torch.atan(pred_boxes[..., 2] / pred_boxes[..., 3].clamp_min(eps))
    ).pow(2)
    with torch.no_grad():
        alpha_v = v / ((1 - iou) + v + eps)

    loss = 1 - iou + rho2 / c2 + alpha_v * v
    return loss.mean()
```

---

### NEW: [drishti_v2/training/motion_loss.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/motion_loss.py)

```python
"""Motion Displacement Loss — validates heatmap motion tracking."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def soft_argmax2d(heatmap: Tensor, temperature: float = 0.1) -> Tensor:
    """
    Differentiable 2D argmax via softmax expectation.
    Returns expected (x, y) position in normalised [0, 1] coords.

    Args:
        heatmap: [B, 1, H, W]
        temperature: softmax temperature (lower = sharper, closer to hard argmax)

    Returns:
        [B, 2] expected positions (cx, cy) in [0, 1]
    """
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1) / temperature
    weights = F.softmax(flat, dim=-1)

    # Create coordinate grids
    ys = torch.linspace(0, 1, H, device=heatmap.device)
    xs = torch.linspace(0, 1, W, device=heatmap.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(1, -1).expand(B, -1)
    grid_y = grid_y.reshape(1, -1).expand(B, -1)

    cx = (weights * grid_x).sum(dim=-1)
    cy = (weights * grid_y).sum(dim=-1)
    return torch.stack([cx, cy], dim=-1)   # [B, 2]


def motion_displacement_loss(
    heatmaps: list[Tensor],
    all_targets: list[list[dict]],
    temperature: float = 0.1,
) -> Tensor:
    """
    L_motion = (1/(T-1)) * sum_t ||d_pred_t - d_gt_t||^2

    Args:
        heatmaps: list of T tensors, each [B, 1, H, W]
        all_targets: list of B clips, each a list of T per-frame target dicts
                     each dict has "boxes" key with [N, 4] GT boxes in [cx,cy,w,h]
        temperature: soft-argmax temperature

    Returns:
        scalar loss
    """
    T = len(heatmaps)
    if T < 2:
        return heatmaps[0].sum() * 0.0

    # Extract GT centers per frame [B, 2] — use first GT box center per frame
    B = heatmaps[0].shape[0]
    device = heatmaps[0].device

    gt_centers = []
    for t in range(T):
        centers_t = []
        for b in range(B):
            boxes = all_targets[b][t].get("boxes", torch.empty(0, 4))
            if boxes.numel() > 0:
                centers_t.append(boxes[0, :2].to(device))  # use first box center
            else:
                centers_t.append(torch.zeros(2, device=device))
        gt_centers.append(torch.stack(centers_t, dim=0))   # [B, 2]

    # Extract predicted heatmap peaks (soft argmax)
    pred_peaks = [soft_argmax2d(hm, temperature) for hm in heatmaps]  # list of [B, 2]

    total_loss = heatmaps[0].sum() * 0.0
    for t in range(1, T):
        d_pred = pred_peaks[t] - pred_peaks[t - 1]     # [B, 2]
        d_gt = gt_centers[t] - gt_centers[t - 1]       # [B, 2]
        total_loss = total_loss + (d_pred - d_gt).pow(2).sum(dim=-1).mean()

    return total_loss / (T - 1)
```

---

### NEW: [drishti_v2/training/temporal_loss.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/temporal_loss.py)

```python
"""Temporal Consistency Loss and Trajectory Smoothness Loss."""
from __future__ import annotations
import torch
from torch import Tensor


def temporal_consistency_loss(
    logits_seq: list[Tensor],
    centers_seq: list[Tensor],
    sigma_spatial: float = 0.1,
) -> Tensor:
    """
    L_consist = (1/K(T-1)) * sum_{k,t} w_{t,k} * (s_t^k - s_{t-1}^k)^2
    Spatially-weighted score consistency across adjacent frames.

    Args:
        logits_seq: list of T tensors, each [B, K, 1] — objectness logits
        centers_seq: list of T tensors, each [B, K, 2] — crop centers
        sigma_spatial: spatial distance scale for weighting

    Returns:
        scalar loss
    """
    T = len(logits_seq)
    if T < 2:
        return logits_seq[0].sum() * 0.0

    scores_seq = [torch.sigmoid(l.squeeze(-1)) for l in logits_seq]  # list of [B, K]
    total = logits_seq[0].sum() * 0.0

    for t in range(1, T):
        # Spatial proximity weight
        dist = (centers_seq[t] - centers_seq[t - 1]).pow(2).sum(dim=-1)  # [B, K]
        w = torch.exp(-dist / (2 * sigma_spatial ** 2))

        diff = (scores_seq[t] - scores_seq[t - 1]).pow(2)  # [B, K]
        total = total + (w * diff).mean()

    return total / (T - 1)


def trajectory_smoothness_loss(
    boxes_seq: list[Tensor],
    labels_seq: list[Tensor],
) -> Tensor:
    """
    L_smooth = (1/K_pos(T-2)) * sum_{pos k, t} ||Delta^2_t^k||^2

    Args:
        boxes_seq: list of T tensors, each [B, K, 4] — predicted boxes [cx,cy,w,h]
        labels_seq: list of T tensors, each [B, K] — binary positive labels

    Returns:
        scalar loss
    """
    T = len(boxes_seq)
    if T < 3:
        return boxes_seq[0].sum() * 0.0

    # Only apply to crops that are positive in at least one frame
    pos_mask = torch.stack(labels_seq, dim=0).any(dim=0)  # [B, K]

    total = boxes_seq[0].sum() * 0.0
    count = 0

    for t in range(2, T):
        v_t = boxes_seq[t] - boxes_seq[t - 1]          # [B, K, 4] velocity
        v_t1 = boxes_seq[t - 1] - boxes_seq[t - 2]     # [B, K, 4] prev velocity
        acc = (v_t - v_t1).pow(2).sum(dim=-1)           # [B, K] acceleration magnitude

        masked_acc = acc * pos_mask.float()
        total = total + masked_acc.sum()
        count += pos_mask.float().sum().item()

    if count == 0:
        return total
    return total / max(count, 1)
```

---

### NEW: [drishti_v2/training/stage_losses.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/stage_losses.py)

```python
"""Stage-specific loss functions for DRISHTI-CORE v2."""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.models.pipeline import PipelineOutput
from drishti_v2.training.focal_loss import sigmoid_focal_loss, heatmap_focal_loss
from drishti_v2.training.ciou_loss import ciou_loss
from drishti_v2.training.motion_loss import motion_displacement_loss
from drishti_v2.training.temporal_loss import temporal_consistency_loss, trajectory_smoothness_loss


# ──────────────────────────────────────────────
# Shared utility: crop-GT assignment
# (Refactored from DRISHTILoss._assign_crops)
# ──────────────────────────────────────────────

def assign_crops(output: PipelineOutput, targets: list[dict]) -> tuple[Tensor, Tensor]:
    """Assigns GT boxes to nearest crop centers. Returns labels and box targets."""
    batch, num_crops, _ = output.proposal_centers.shape
    labels = output.objectness_logits.new_zeros(batch, num_crops, 1)
    box_targets = output.crop_boxes.detach().new_zeros(batch, num_crops, 4)
    for b_idx, target in enumerate(targets):
        boxes = target.get("boxes", torch.empty(0, 4)).to(output.proposal_centers.device)
        if boxes.numel() == 0:
            continue
        centers = output.proposal_centers[b_idx]
        distances = torch.cdist(centers, boxes[:, :2])
        crop_indices = distances.argmin(dim=0).unique()
        for crop_idx in crop_indices:
            gt_idx = distances[crop_idx].argmin()
            gt = boxes[gt_idx]
            labels[b_idx, crop_idx, 0] = 1.0
            global_pred_size = output.boxes[b_idx, crop_idx, 2:].clamp_min(1e-6)
            crop_scale = global_pred_size / output.crop_boxes[b_idx, crop_idx, 2:].clamp_min(1e-6)
            rel_xy = (gt[:2] - centers[crop_idx]) / crop_scale + 0.5
            rel_wh = gt[2:] / crop_scale
            box_targets[b_idx, crop_idx] = torch.cat([rel_xy, rel_wh]).clamp(0.0, 1.0)
    return labels, box_targets


def make_gt_heatmaps(targets: list[dict], heatmap_size: tuple[int, int], device: torch.device) -> Tensor:
    """Build GT heatmaps from target dicts."""
    return torch.stack([MotionCNN.make_gt_heatmap(
        t.get("boxes", torch.empty(0, 4)).to(device), heatmap_size,
    ) for t in targets], dim=0)


# ──────────────────────────────────────────────
# Stage 1: Spatial Detector
# ──────────────────────────────────────────────

class Stage1Loss(nn.Module):
    def __init__(self, w_hm=1.0, w_cls=1.0, w_box=2.0, w_motion=0.5,
                 w_gate=0.01, focal_gamma=2.0, focal_alpha=0.25,
                 hm_alpha=2.0, hm_beta=4.0, motion_temperature=0.1):
        super().__init__()
        self.w_hm, self.w_cls, self.w_box = w_hm, w_cls, w_box
        self.w_motion, self.w_gate = w_motion, w_gate
        self.gamma, self.alpha = focal_gamma, focal_alpha
        self.hm_alpha, self.hm_beta = hm_alpha, hm_beta
        self.motion_temp = motion_temperature

    def forward(self, output: PipelineOutput, targets: list,
                all_heatmaps: list[Tensor] | None = None) -> dict[str, Tensor]:
        last_targets = [clip[-1] for clip in targets] if isinstance(targets[0], list) else targets
        hm_size = tuple(output.heatmap.shape[-2:])

        gt_hm = make_gt_heatmaps(last_targets, hm_size, output.heatmap.device).to(output.heatmap.dtype)
        hm_loss = heatmap_focal_loss(output.heatmap, gt_hm, self.hm_alpha, self.hm_beta)

        labels, box_targets = assign_crops(output, last_targets)
        cls_loss = sigmoid_focal_loss(output.objectness_logits, labels, self.gamma, self.alpha)

        positive = labels.squeeze(-1) > 0.5
        box_loss = ciou_loss(output.crop_boxes[positive], box_targets[positive]) \
            if positive.any() else output.objectness_logits.sum() * 0.0

        motion_loss = motion_displacement_loss(all_heatmaps, targets, self.motion_temp) \
            if all_heatmaps is not None and isinstance(targets[0], list) \
            else output.objectness_logits.sum() * 0.0

        gate_loss = output.objectness_logits.sum() * 0.0  # placeholder — gate sparsity added in pipeline

        total = (self.w_hm * hm_loss + self.w_cls * cls_loss +
                 self.w_box * box_loss + self.w_motion * motion_loss +
                 self.w_gate * gate_loss)

        return {"loss": total, "heatmap": hm_loss, "cls": cls_loss,
                "bbox": box_loss, "motion_disp": motion_loss,
                "balance": output.balance_loss}


# ──────────────────────────────────────────────
# Stage 2: Temporal Fusion
# ──────────────────────────────────────────────

class Stage2Loss(nn.Module):
    def __init__(self, w_hm=0.5, w_cls=1.0, w_box=2.0, w_tc=0.3, w_sm=0.1,
                 focal_gamma=2.0, focal_alpha=0.25, hm_alpha=2.0, hm_beta=4.0):
        super().__init__()
        self.w_hm, self.w_cls, self.w_box = w_hm, w_cls, w_box
        self.w_tc, self.w_sm = w_tc, w_sm
        self.gamma, self.alpha = focal_gamma, focal_alpha
        self.hm_alpha, self.hm_beta = hm_alpha, hm_beta

    def forward(self, output: PipelineOutput, targets: list,
                logits_seq: list[Tensor] | None = None,
                centers_seq: list[Tensor] | None = None,
                boxes_seq: list[Tensor] | None = None) -> dict[str, Tensor]:
        last_targets = [clip[-1] for clip in targets] if isinstance(targets[0], list) else targets
        hm_size = tuple(output.heatmap.shape[-2:])

        gt_hm = make_gt_heatmaps(last_targets, hm_size, output.heatmap.device).to(output.heatmap.dtype)
        hm_loss = heatmap_focal_loss(output.heatmap, gt_hm, self.hm_alpha, self.hm_beta)

        labels, box_targets = assign_crops(output, last_targets)
        cls_loss = sigmoid_focal_loss(output.objectness_logits, labels, self.gamma, self.alpha)
        positive = labels.squeeze(-1) > 0.5
        box_loss = ciou_loss(output.crop_boxes[positive], box_targets[positive]) \
            if positive.any() else output.objectness_logits.sum() * 0.0

        tc_loss = temporal_consistency_loss(logits_seq, centers_seq) \
            if logits_seq is not None and centers_seq is not None \
            else output.objectness_logits.sum() * 0.0

        labels_seq = [labels] * len(boxes_seq) if boxes_seq is not None else None
        sm_loss = trajectory_smoothness_loss(boxes_seq, labels_seq) \
            if boxes_seq is not None else output.objectness_logits.sum() * 0.0

        total = (self.w_hm * hm_loss + self.w_cls * cls_loss + self.w_box * box_loss +
                 self.w_tc * tc_loss + self.w_sm * sm_loss)
        return {"loss": total, "heatmap": hm_loss, "cls": cls_loss, "bbox": box_loss,
                "temporal_consist": tc_loss, "traj_smooth": sm_loss,
                "balance": output.balance_loss}


# ──────────────────────────────────────────────
# Stage 3: MoE Routing
# ──────────────────────────────────────────────

class Stage3Loss(nn.Module):
    def __init__(self, w_cls=1.0, w_box=2.0, w_bal=0.01, w_zloss=0.001,
                 focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.w_cls, self.w_box = w_cls, w_box
        self.w_bal, self.w_zloss = w_bal, w_zloss
        self.gamma, self.alpha = focal_gamma, focal_alpha

    def forward(self, output: PipelineOutput, targets: list) -> dict[str, Tensor]:
        last_targets = [clip[-1] for clip in targets] if isinstance(targets[0], list) else targets
        labels, box_targets = assign_crops(output, last_targets)
        cls_loss = sigmoid_focal_loss(output.objectness_logits, labels, self.gamma, self.alpha)
        positive = labels.squeeze(-1) > 0.5
        box_loss = ciou_loss(output.crop_boxes[positive], box_targets[positive]) \
            if positive.any() else output.objectness_logits.sum() * 0.0
        balance = output.balance_loss
        z_loss = output.moe_diagnostics.router_z_loss if hasattr(output.moe_diagnostics, "router_z_loss") \
            else output.balance_loss * 0.0

        total = self.w_cls * cls_loss + self.w_box * box_loss + self.w_bal * balance + self.w_zloss * z_loss
        return {"loss": total, "cls": cls_loss, "bbox": box_loss,
                "balance": balance, "z_loss": z_loss}


# ──────────────────────────────────────────────
# Stage 4: End-to-End Finetune
# ──────────────────────────────────────────────

class Stage4Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = Stage1Loss(w_motion=0.3, w_gate=0.01)
        self.s2 = Stage2Loss(w_hm=0.5, w_tc=0.15, w_sm=0.05)
        self.s3 = Stage3Loss(w_bal=0.01, w_zloss=0.001)

    def forward(self, output: PipelineOutput, targets: list, **kwargs) -> dict[str, Tensor]:
        d1 = self.s1.forward(output, targets, kwargs.get("all_heatmaps"))
        d2 = self.s2.forward(output, targets, kwargs.get("logits_seq"),
                              kwargs.get("centers_seq"), kwargs.get("boxes_seq"))
        d3 = self.s3.forward(output, targets)
        total = d1["loss"] + d2["temporal_consist"] + d2["traj_smooth"] + d3["z_loss"]
        return {"loss": total, **{f"s1_{k}": v for k, v in d1.items()},
                **{f"s2_{k}": v for k, v in d2.items()},
                **{f"s3_{k}": v for k, v in d3.items()}}


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class StageLossFactory:
    @staticmethod
    def make_loss(stage: str, **kwargs) -> nn.Module:
        stage = stage.lower()
        if stage in {"stage1", "detector"}:
            return Stage1Loss(**{k: v for k, v in kwargs.items()
                                 if k in Stage1Loss.__init__.__code__.co_varnames})
        if stage in {"stage2", "temporal"}:
            return Stage2Loss(**{k: v for k, v in kwargs.items()
                                 if k in Stage2Loss.__init__.__code__.co_varnames})
        if stage in {"stage3", "moe"}:
            return Stage3Loss(**{k: v for k, v in kwargs.items()
                                 if k in Stage3Loss.__init__.__code__.co_varnames})
        if stage in {"stage4", "finetune", "e2e", "all"}:
            return Stage4Loss()
        raise ValueError(f"Unknown stage: {stage}")
```

---

### MODIFY: [drishti_v2/models/moe.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/moe.py)

Add `router_z_loss` field to `MoEDiagnostics` (line 25):

```diff
 @dataclass
 class MoEDiagnostics:
     balance_loss: Tensor
     expert_utilization: Tensor
     routing_probabilities: Tensor
     router_entropy: Tensor
     token_drop_rate: Tensor
     expert_reuse_frequency: Tensor
     load_balance_cv: Tensor
+    router_z_loss: Tensor           # NEW: logsumexp squared penalty
```

In `SparseMoE.forward` (after line 68, before `probs = softmax(...)`):

```diff
+    # Router Z-Loss (computed on raw logits, before softmax)
+    router_logits = self.router(x_flat)                              # [N, E]
+    z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()   # scalar
-    probs = torch.softmax(self.router(x_flat), dim=-1)
+    probs = torch.softmax(router_logits, dim=-1)
```

Pass `router_z_loss=z_loss.detach()` to `MoEDiagnostics` in both the dense and sparse branches.

---

### MODIFY: [drishti_v2/training/losses.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/losses.py)

Add deprecation warning at top of `DRISHTILoss.__init__`:

```diff
+import warnings
 class DRISHTILoss(nn.Module):
     def __init__(self, ...):
+        warnings.warn(
+            "DRISHTILoss is deprecated. Use StageLossFactory.make_loss(stage) instead.",
+            DeprecationWarning, stacklevel=2,
+        )
         super().__init__()
         ...
```

---

### MODIFY: [drishti_v2/training/trainer.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/trainer.py)

Key changes:

1. **Line 87** — change type hint from `DRISHTILoss` to `nn.Module`:
```diff
-    loss_fn: DRISHTILoss,
+    loss_fn: nn.Module,
```

2. **Line 205** — extend accumulator for new loss keys:
```diff
-    accum = {"loss": 0.0, "heatmap": 0.0, "cls": 0.0, "bbox": 0.0, "balance": 0.0}
+    accum = {k: 0.0 for k in [
+        "loss", "heatmap", "cls", "bbox", "balance",
+        "motion_disp", "temporal_consist", "traj_smooth", "z_loss",
+    ]}
```

3. **Line 227** — pass stage-appropriate extras to loss:
```diff
-    losses = self.loss_fn(output, batch["targets"])
+    loss_kwargs = {"targets": batch["targets"]}
+    if stage in {"stage1", "detector", "stage4", "finetune", "e2e", "all"}:
+        loss_kwargs["all_heatmaps"] = getattr(output, "_all_heatmaps", None)
+    losses = self.loss_fn(output, **loss_kwargs)
```

4. **Lines 264–269** — extend per-step JSONL log to include new keys:
```diff
     "loss/total": round(float(losses["loss"].detach().cpu()), 6),
     "loss/heatmap": round(float(losses.get("heatmap", losses["loss"]*0).detach().cpu()), 6),
     "loss/cls": round(float(losses.get("cls", losses["loss"]*0).detach().cpu()), 6),
     "loss/bbox": round(float(losses.get("bbox", losses["loss"]*0).detach().cpu()), 6),
     "loss/balance": round(float(losses.get("balance", losses["loss"]*0).detach().cpu()), 6),
+    "loss/motion_disp": round(float(losses.get("motion_disp", losses["loss"]*0).detach().cpu()), 6),
+    "loss/temporal_consist": round(float(losses.get("temporal_consist", losses["loss"]*0).detach().cpu()), 6),
+    "loss/traj_smooth": round(float(losses.get("traj_smooth", losses["loss"]*0).detach().cpu()), 6),
+    "loss/z_loss": round(float(losses.get("z_loss", losses["loss"]*0).detach().cpu()), 6),
```

---

### MODIFY: [drishti_v2/evaluation/metrics.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/evaluation/metrics.py)

Add new metric functions after line 90:

```python
def heatmap_peak_metrics(
    heatmaps: list[Tensor],          # [B, 1, H, W] per batch
    targets: list[dict[str, Tensor]],
    pixel_threshold_frac: float = 5.0 / 448,
) -> dict[str, float]:
    """
    - heatmap_peak_distance: mean L2 distance between heatmap peak and GT center
    - heatmap_peak_within_threshold: fraction of frames where peak is within threshold distance
    """
    distances = []
    within = []
    for hm, target in zip(heatmaps, targets):
        boxes = target.get("boxes", torch.empty(0, 4))
        if boxes.numel() == 0:
            continue
        gt_center = boxes[0, :2]  # use first GT box
        # Hard argmax for evaluation (not training)
        H, W = hm.shape[-2:]
        flat_idx = hm.view(-1).argmax()
        cy = (flat_idx // W).float() / H
        cx = (flat_idx % W).float() / W
        pred_center = torch.tensor([cx, cy])
        dist = (pred_center - gt_center.cpu()).pow(2).sum().sqrt().item()
        distances.append(dist)
        within.append(float(dist < pixel_threshold_frac))
    return {
        "heatmap_peak_distance": float(sum(distances) / max(len(distances), 1)),
        "heatmap_peak_within_5px": float(sum(within) / max(len(within), 1)),
    }


def motion_direction_accuracy(
    pred_displacements: list[Tensor],   # list of [2] vectors
    gt_displacements: list[Tensor],
) -> dict[str, float]:
    """Cosine similarity between predicted and GT motion direction vectors."""
    sims = []
    for pd, gd in zip(pred_displacements, gt_displacements):
        cos_sim = F.cosine_similarity(pd.unsqueeze(0), gd.unsqueeze(0)).item()
        sims.append(cos_sim)
    return {"motion_direction_accuracy": float(sum(sims) / max(len(sims), 1))}
```

---

### MODIFY: [configs/default.yaml](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/configs/default.yaml)

Add new loss weight configs:

```yaml
# Stage 1: Spatial Detector losses
focal_gamma: 2.0
focal_alpha: 0.25
heatmap_focal_alpha: 2.0
heatmap_focal_beta: 4.0
w_motion_displacement: 0.5
w_gate_sparsity: 0.01

# Stage 2: Temporal Fusion losses
w_temporal_consistency: 0.3
w_trajectory_smoothness: 0.1
sigma_spatial_consist: 0.1

# Stage 3: MoE losses
moe_balance_weight: 0.01
router_z_loss_weight: 0.001

# Motion gating
use_motion_gate: true
motion_gate_hidden: 16
motion_gate_threshold: 0.5
dense_grid_size: 4

# LDMI scales (updated)
ldmi_scales: [7, 15, 31, 63]
```

---

# Section 4: What Is Learned in Each Stage (and Why)

## Stage 1 — Spatial Detector (80 epochs, lr=1e-4)

**Trainable**: MotionCNN + CropEncoder + DetectionHead + MotionGate
**Frozen**: CausalTemporalFusion + SparseMoE

### What the model learns

**MotionCNN** receives 15-channel LDMI output and learns to map it to a peaked heatmap.

Mathematically, it learns a function $f_\theta: \mathbb{R}^{15 \times H \times W} \to [0,1]^{H/4 \times W/4}$ such that:
$$f_\theta(\text{LDMI}(F_{t-2}, F_{t-1}, F_t)) \approx \mathcal{G}_\sigma(\text{gt\_center})$$

The HeatmapFocalLoss guides it to produce sharp peaks at UAV locations ($(1-\hat{y})^\alpha$ penalises flat predictions at peak locations) while the background weight $(1-y)^\beta$ suppresses false positives without aggressively penalising the Gaussian falloff region.

**CropEncoder** learns to produce features that distinguish UAV crops from background crops.

The SigmoidFocalLoss guides it: $\mathcal{L}_{\text{FL}}$ down-weights easy negative crops (the 7/8 that obviously contain no UAV) and forces the model to concentrate on hard cases. Without Focal, the gradient would be dominated by the 7 easy negatives → encoder learns "predict everything as background."

**DetectionHead** learns from CIoU to produce geometrically precise boxes.

The centre-distance term $\rho^2/c^2$ pushes the predicted center toward the GT center even when IoU = 0. At initialisation, random boxes often have zero overlap with GT — Smooth L1 would have large gradient that doesn't account for geometry, but CIoU's center-distance provides a sensible gradient direction regardless.

**MotionGate** learns the 6 heatmap statistics that predict whether the motion signal is trustworthy. The sparsity regulariser $\mathcal{L}_{\text{gate}} = (1 - g)$ prevents it from always outputting low confidence.

**MotionDisplacementLoss** forces the heatmap peak trajectory to match the GT box trajectory:

$$\min_\theta \frac{1}{T-1} \sum_t \|\hat{\mathbf{d}}_t - \mathbf{d}_t\|_2^2$$

This ensures LDMI + MotionCNN captures not just "where is the UAV" but "how is it moving."

---

## Stage 2 — Temporal Fusion (30 epochs, lr=5e-5)

**Trainable**: CausalTemporalFusion
**Frozen**: MotionCNN + CropEncoder + DetectionHead + MotionGate + SparseMoE

### What the model learns

At this stage, the spatial modules produce stable features (learned in Stage 1). The transformer now sees consistent input and can learn meaningful temporal patterns.

**CausalTemporalFusion** learns to aggregate crop feature histories. Its causal attention over T timesteps:

$$\text{Attn}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d}} + M_{\text{causal}}\right) V_{1:t}$$

The attention weights learn to extract relevant historical context. A crop containing the UAV at time $t$ should attend strongly to its own history at $t-1, t-2$ (where it also contained the UAV) and weakly to times when the crop was a background GRID scan.

**TemporalConsistencyLoss** explicitly penalises score oscillation. A crop that scores 0.9 (positive) at $t=3$ but 0.05 (negative) at $t=4$ then 0.8 (positive) at $t=5$ gets a high penalty. The model learns to propagate confidence across time.

**TrajectorySmoothLoss** penalises acceleration. If the model predicts box at $(0.45, 0.62)$ at $t=3$ and $(0.47, 0.64)$ at $t=4$, velocity = $(+0.02, +0.02)$. If it then predicts $(0.41, 0.61)$ at $t=5$, velocity becomes $(-0.06, -0.03)$, acceleration = $(-0.08, -0.05)$ — very large. The smooth loss penalises this, encouraging the temporal module to produce smooth, physically plausible trajectories.

---

## Stage 3 — MoE Routing (20 epochs, lr=1e-5)

**Trainable**: SparseMoE
**Frozen**: Everything else

### What the model learns

At this stage, the temporal module produces stable fused features. The MoE learns to route these features to specialized experts.

**SparseMoE router** learns $\mathbf{W}_r \in \mathbb{R}^{8 \times 256}$ such that crops requiring different processing strategies are routed to different experts:

$$p_i = \text{softmax}(\mathbf{W}_r \mathbf{x}_i)$$

With the SigmoidFocalLoss and CIoULoss, expert outputs must be useful for detection. The router is indirectly trained to route in ways that maximise detection accuracy.

**Load-balance loss** ($\mathcal{L}_{\text{balance}} = E \sum_j f_j \bar{p}_j$) prevents routing collapse where all tokens go to 2 experts and the other 6 are never trained.

**Router Z-Loss** prevents the router logits from growing large. Large logits → near-deterministic routing → experts stop receiving diverse training signal → experts don't specialise → the MoE degrades to a single expert.

$$\mathcal{L}_z = \frac{1}{N} \sum_i (\text{logsumexp}(\mathbf{x}_i))^2$$

By penalising large logsumexp values, the router logits stay moderate, probabilities stay spread across multiple experts, and each expert receives enough training signal to develop genuine specialisation.

---

## Stage 4 — End-to-End Finetune (10 epochs, lr=2e-6)

**Trainable**: Everything

### What the model learns

Joint optimisation with a very small learning rate. Each module makes small coordinated adjustments that wouldn't emerge from stage-wise training:

- **MotionCNN** can slightly adjust its heatmap peaks based on feedback from the full downstream pipeline
- **CropEncoder** can refine features based on what the temporal module and MoE have learned to use
- **CausalTemporalFusion** can refine temporal patterns based on improved spatial features
- **SparseMoE** can refine routing based on improved temporal features
- **DetectionHead** adjusts based on all upstream refinements

The combined Stage4Loss uses all auxiliary terms at reduced weights, ensuring the detection signal ($\mathcal{L}_{\text{Focal}}$, $\mathcal{L}_{\text{CIoU}}$) still dominates.

---

# Section 5: Complete File Change Summary

## All New Files

| File | Purpose |
|---|---|
| `drishti_v2/models/motion_gate.py` | Learned heatmap statistics → motion confidence |
| `drishti_v2/training/focal_loss.py` | Sigmoid Focal Loss + Heatmap Focal Loss |
| `drishti_v2/training/ciou_loss.py` | Complete IoU Loss |
| `drishti_v2/training/motion_loss.py` | Motion Displacement Loss |
| `drishti_v2/training/temporal_loss.py` | Temporal Consistency + Trajectory Smoothness |
| `drishti_v2/training/stage_losses.py` | Stage1–4 losses + StageLossFactory |

## All Modified Files

| File | Change | Lines |
|---|---|---|
| [ldmi.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/ldmi.py) | Full rewrite — 9ch→15ch | All |
| [motion_cnn.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/motion_cnn.py) | `in_channels` 9→15 | Line 16 |
| [moe.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/moe.py) | Add `router_z_loss` to diagnostics | Lines 10–26, 68–70 |
| [crop_proposal.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/crop_proposal.py) | Add `forward_dense()` | After line 64 |
| [pipeline.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/pipeline.py) | Wire MotionGate, adaptive mode | Lines 37–170 |
| [config.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/config.py) | New fields, update scales | Lines 18–30 |
| [stage_control.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/stage_control.py) | MotionGate in stage1 | Line 22 |
| [losses.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/losses.py) | Add deprecation warning | Lines 14–21 |
| [trainer.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/training/trainer.py) | Accept `nn.Module`, log new keys | Lines 87, 205, 227, 264–269 |
| [evaluator.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/evaluation/evaluator.py) | Stage-aware eval | Lines 23–41 |
| [metrics.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/evaluation/metrics.py) | New metric functions | After line 90 |
| [\_\_init\_\_.py](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/drishti_v2/models/__init__.py) | Export MotionGate | Lines 3–7 |
| [default.yaml](file:///c:/Users/jaygo/Desktop/DESKTOP/Research%20Papers/DRISHTI-CORE/configs/default.yaml) | New loss weight fields | Throughout |

## Parameter Count Impact

| Module | Before | After | Delta |
|---|---|---|---|
| LDMI | 0 | 0 | 0 |
| MotionCNN first conv | 9×32×9=2,592 | 15×32×9=4,320 | +1,728 |
| MotionGate (NEW) | 0 | 129 | +129 |
| MoE (z_loss is computed, not params) | 1,054,720 | 1,054,720 | 0 |
| Loss primitives (no params) | — | — | 0 |
| **Total delta** | — | — | **+1,857** |
