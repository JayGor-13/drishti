# T-MoE-LLaVA for Anti-Drone Detection
## Full Architecture & Training Plan (AAAI Workshop Scope)

---

## 0. What We Are Building

A **sparse, motion-conditioned video model** that detects hostile drones from a
ground-facing camera feed. The model runs efficiently on edge hardware (Jetson
Orin Nano) by skipping compute on static sky regions and only activating full
MoE routing where motion is detected.

**Single claim for the workshop:**
> "Motion-conditioned sparse routing achieves competitive detection accuracy
> vs. dense baselines at significantly lower FLOPs, with a structural advantage
> on tiny drone targets where appearance is near-zero."

**Dataset:** Anti-UAV (CVPR) — 410 RGB+IR video sequences of hostile drones.
**Baselines:** YOLOv8-S fine-tuned, RT-DETR fine-tuned, Video-LLaVA dense.

---

## 1. Full Model Architecture

### 1.1 High-Level Overview

```
Video Frame Sequence  f_{t-w} ... f_t ... f_{t+w}
         │
         ├─────────────────────┬──────────────────────
         │                     │
   [Pathway 1]           [Pathway 2]
   Semantic Tokens        Motion Embeddings
   LocateAnything         X3D-Tiny
   V^t ∈ R^{N×D}         δ^t ∈ R^{N×D}
         │                     │
         └──────────┬──────────┘
                    │
            [Modality-Aware Router]
            P_i^t = softmax(W_r [x_i^t ⊕ δ_i^t])
            m_i^t = σ(W_m δ_i^t)  ← motion confidence scalar
                    │
         ┌──────────┴──────────┐
         │                     │
   m_i^t < ε            m_i^t ≥ ε
   (static token)        (motion token)
         │                     │
   [Temporal Cache]    [MoE Layer]
   Reuse expert        8 experts, Top-2 routing
   output from t-1     Full computation
         │                     │
         └──────────┬──────────┘
                    │
            [Detection Head]
            Bounding boxes + confidence
            per drone per frame
```

---

### 1.2 Component 1 — Semantic Pathway: LocateAnything (replaces CLIP)

**What it is:**
NVIDIA's LocateAnything-3B. Vision-language grounding model, open weights on
HuggingFace. Trained on 12M images, 785M bounding boxes. Uses Parallel Box
Decoding (PBD) — predicts boxes in one forward pass, not token-by-token.

**Why it replaces CLIP here:**
CLIP gives generic image embeddings. LocateAnything gives
*grounding-aware* embeddings — it already understands spatial object
relationships. For a detection task this is strictly better as a backbone.

**What it produces:**
Visual tokens V^t ∈ R^{N×D} where N = number of spatial patches, D = 1024.

**In practice:**
- Input: single frame f_t, resized to 448×448
- Output: patch-level embeddings, one per 16×16 region = 784 tokens
- Frozen in Stage 1, partially unfrozen in Stage 2+

**Key note:**
LocateAnything is non-commercial research license. For the AAAI paper this is
fine. For iDEX deployment you either negotiate with NVIDIA or swap back to
CLIP-Large which is Apache 2.0. Keep this noted.

---

### 1.3 Component 2 — Motion Pathway: X3D-Tiny

**What it is:**
A lightweight 3D CNN designed for video. "Tiny" variant has ~2M parameters.
Operates on a temporal window of frames, not a single frame.

**What it does:**
Extracts per-patch motion features from a window of 2w+1 frames centered on t.

**Formally (from your proposal):**
```
δ_i^t = X3D-Tiny(f_{t-w : t+w})_i
```
where i indexes spatial patch, w=4 (9-frame window in practice).

**Why not optical flow:**
Your proposal correctly identified that naive frame differences
|f_t - f_{t-1}| fail when the camera moves (pans/tilts). X3D-Tiny implicitly
learns ego-motion disentanglement during training. It sees the whole temporal
window and learns what is true object motion vs. camera motion.

**What it produces:**
Motion embeddings δ^t ∈ R^{N×D_m} where D_m = 256, then projected to D=1024
to match semantic pathway via a linear projection head W_proj.

**Motion confidence scalar (from your proposal):**
```
m_i^t = σ(W_m δ_i^t) ∈ [0, 1]
```
This is the gating signal. One scalar per patch per frame.
- m close to 0 = static patch (sky, ground, trees) → cache it
- m close to 1 = moving patch (drone) → route through MoE

---

### 1.4 Component 3 — Modality-Aware Router

**What it does:**
Takes the concatenation of semantic token and motion embedding for patch i at
time t, and produces a routing probability distribution over 8 experts.

**Formally (from your proposal):**
```
P_i^t = softmax(W_r [x_i^t ⊕ δ_i^t])
```
where ⊕ is concatenation, W_r ∈ R^{8 × 2D}.

**The modality-awareness:**
- For video tokens: router sees [semantic ⊕ motion] — full context
- For text tokens (if any): router sees [semantic ⊕ zeros] — standard routing
  This is the "modality-aware" part from your proposal. Text tokens
  don't have motion, so they route differently by design.

**Top-2 routing:**
Each token is sent to its top-2 experts by probability weight.
Final output = weighted sum of the two expert outputs.

---

### 1.5 Component 4 — Event-Based Temporal Token Cache

**What it does:**
This is your O(1) efficiency claim. Directly from your proposal:

```
IF m_i^t < ε:
    output_i^t = output_i^{t-1}   ← just copy from last frame
ELSE:
    output_i^t = MoE(x_i^t, δ_i^t)  ← full computation
```

**What ε is:**
A threshold hyperparameter. In practice set ε = 0.15 and tune on validation.
Too low and you cache too aggressively (miss slow drones).
Too high and you lose the efficiency benefit.

**In a drone video:**
Sky = ~70-80% of pixels = cached every frame.
Drone patch = always above ε = always computed.
This is how you show the sparse activation visualization in the demo.

---

### 1.6 Component 5 — MoE Layer: 8 Implicit Experts, Top-2 Routing

**Structure:**
8 expert MLPs, each identical in architecture (2-layer FFN, hidden dim 4096).
Initialized identically + small Gaussian noise (from your proposal Stage III).

**Why implicit, not explicit:**
Your proposal resolved this debate correctly. Explicitly forcing experts to be
"kinematic" or "semantic" requires contrastive losses that are unstable and
violate MoE best practices. Instead, experts naturally specialize through
training — some will learn to handle motion blur, some will learn to handle
tiny distant drones, some will handle occlusion. You don't force it.

**Load balancing:**
Standard auxiliary loss L_aux from Switch Transformer:
```
L_aux = α · Σ_e f_e · p_e
```
where f_e = fraction of tokens routed to expert e,
      p_e = mean routing probability for expert e.
This prevents all tokens collapsing to one expert.

---

### 1.7 Component 6 — CFCR Loss

**From your proposal, refined:**
```
L_cfcr = (1/T·N) Σ_{t=1}^{T-1} Σ_{i,j} A_{i,j} · (1 - m_i^t) · JSD(P_i^t || P_j^{t+1})
```

**What each term means:**
- `(1 - m_i^t)`: only enforce consistency on STATIC patches (low motion).
  Moving patches (drones) are ALLOWED to change their routing — they're
  genuinely dynamic. Static patches (sky) should route consistently.
- `A_{i,j}`: attention-based alignment matrix. Tracks which patch j in
  frame t+1 corresponds to patch i in frame t. Handles slight camera movement.
- `JSD(...)`: Jensen-Shannon Divergence between routing distributions.
  Chosen over L2 because it's bounded [0,1] and avoids vanishing gradients
  on the probability simplex (your proposal's reasoning).

**What this achieves for drone detection:**
Without CFCR, the router can assign the same sky patch to different experts
each frame (routing instability). With CFCR, sky patches route consistently,
which means when a drone appears in a previously-static patch, the sudden
routing change is a detectable signal. The router effectively learns to be
surprised by drones.

---

### 1.8 Component 7 — Detection Head

**For the AAAI workshop, keep this simple:**
A lightweight detection head on top of MoE outputs:
- Linear projection → class scores (drone / no-drone, 2 classes)
- Bounding box regression (cx, cy, w, h) per patch
- NMS post-processing

This is NOT an LLM. No autoregressive decoding for the workshop.
You're outputting bounding boxes, not text.
The language output (threat summary) is the iDEX Phase 2 addition.

**Total Loss:**
```
L_total = L_det + α·L_aux + β·L_cfcr
```
where L_det = standard detection loss (focal loss + GIoU box loss)
      L_aux = MoE load balancing
      L_cfcr = temporal routing consistency

---

### 1.9 Parameter Count Summary

| Component            | Params   | Frozen? (Stage 1) |
|----------------------|----------|-------------------|
| LocateAnything-3B    | ~3B      | Yes               |
| X3D-Tiny             | ~2M      | No — trains first |
| Projection heads     | ~8M      | No                |
| Router W_r           | ~16M     | No                |
| 8 Expert MLPs        | ~800M    | Initialized Stage III |
| Detection head       | ~4M      | No                |
| **Active at inference** | **~400-600M** | — |

Active params at inference = only 2 of 8 experts fire per token,
plus the cached tokens don't compute at all.
This is your efficiency story.

---

## 2. Training Plan — Anti-UAV Dataset, AAAI Scope

### Dataset Setup

**Anti-UAV (CVPR):**
- 410 video sequences, RGB + IR, ground-to-air
- Split: 240 train / 80 val / 90 test (standard split)
- Labels: bounding box per frame per drone
- Download: https://anti-uav.github.io/

**What you use:** RGB sequences only for the workshop.
IR is future work / iDEX Phase 2.

**Preprocessing:**
- Extract frames at 25fps
- Resize to 448×448
- Temporal window: 9 frames (w=4), stride 1
- Augmentation: horizontal flip, color jitter, random crop
  Do NOT use mosaic augmentation — it destroys temporal coherence

---

### Stage 1 — Motion Encoder Pretraining

**Goal:** X3D-Tiny learns what drone motion looks like vs. camera motion.

**What trains:** X3D-Tiny + projection head W_proj only.
Everything else frozen.

**Supervision:** Self-supervised optical flow prediction.
X3D-Tiny predicts optical flow between frames t and t+1.
Supervised by RAFT (pretrained optical flow model) as pseudo-labels.
No ground truth bounding boxes needed here.

**Why this matters:**
A drone 6 pixels wide has almost no appearance signal.
But it moves differently from clouds, trees, camera shake.
This stage teaches your model that distinction before detection training.

**Hyperparameters:**
```
Optimizer:     AdamW, lr=1e-4
Batch size:    16 sequences (each = 9 frames)
Epochs:        30
Loss:          L2 between predicted and RAFT flow
Hardware:      Single A100 (Google Colab Pro) is enough
Time:          ~18 hours
```

---

### Stage 2 — Dense Full Model Training

**Goal:** Train the full model as a DENSE network first (no MoE yet).
The MoE structure exists but all tokens go through all experts equally.
This is the "dense upcycling" from your proposal.

**What trains:** Everything except LocateAnything backbone (still frozen).
Trains: X3D-Tiny, router, all 8 experts (as dense), detection head.

**Why dense first:**
If you initialize MoE and immediately apply sparse routing, experts collapse.
Training dense first gives all experts a meaningful initialization
before they start competing for tokens.

**Loss:**
```
L_total = L_det + α·L_aux   (no CFCR yet, β=0)
```

**Hyperparameters:**
```
Optimizer:     AdamW, lr=3e-5 (low to preserve LocateAnything features)
Batch size:    8 sequences
Epochs:        20
α (aux):       0.01
Hardware:      A100
Time:          ~24 hours
```

---

### Stage 3 — MoE Sparse Routing + CFCR Activation

**Goal:** Switch on sparse routing and temporal consistency loss.

**What changes:**
- Top-2 routing activates (tokens now compete for experts)
- Temporal Token Cache activates (ε=0.15)
- CFCR loss activates with linear warmup

**MoE initialization (from your proposal):**
Copy dense MLP weights into all 8 experts, then add small Gaussian noise
σ=0.01 to break symmetry. This is critical — without noise all experts
are identical and routing never differentiates.

**CFCR warmup (from your proposal):**
Linear warmup of β from 0 → 0.1 over 500 steps.
This prevents CFCR from fighting load-balancing during initialization.
```
β(step) = 0.1 × min(step/500, 1.0)
```

**Computing A_{i,j} in practice:**
Use cosine similarity between LocateAnything features of adjacent frames
to build the alignment matrix. Drones move fast so their patches will have
LOW similarity (correctly marked as dynamic). Sky patches have HIGH similarity
(correctly marked as static → consistency enforced).

**Loss:**
```
L_total = L_det + α·L_aux + β(step)·L_cfcr
```

**Hyperparameters:**
```
Optimizer:     AdamW, lr=1e-5
Batch size:    8 sequences
Epochs:        15
α (aux):       0.01
β (cfcr):      warmup 0→0.1 over 500 steps
ε (cache):     0.15
Hardware:      A100
Time:          ~20 hours
```

---

### Stage 4 — Partial LocateAnything Unfreeze

**Goal:** Fine-tune the last 4 transformer blocks of LocateAnything
on the Anti-UAV domain. The model was trained on natural scenes,
robotics, and GUI — NOT on sky/drone footage. Domain gap needs closing.

**What trains:** Last 4 blocks of LocateAnything + everything from Stage 3.

**Why not full unfreeze:**
3B parameter full unfreeze on a small dataset (240 sequences) = catastrophic
forgetting of grounding knowledge. Last 4 blocks = domain adaptation
without forgetting.

**Hyperparameters:**
```
Optimizer:     AdamW, lr=5e-6 (very low for backbone)
Batch size:    4 sequences (memory constrained)
Epochs:        10
Gradient clipping: 1.0
Hardware:      A100
Time:          ~16 hours
```

**Total training time: ~80 hours on one A100.**
Google Colab Pro+ gives ~50 compute units/month, each unit ~1 hour A100.
You'll need 2 months of credits, or use Kaggle (free 30hr/week GPU quota).

---

## 3. Evaluation

### What You Measure

| Metric | What it shows |
|--------|---------------|
| mAP@50 | Detection accuracy vs. baselines |
| mAP@50 on tiny drones (<32px) | Where your motion pathway wins |
| GFLOPs per frame | Efficiency claim |
| % tokens cached per frame | Sparse activation story |
| FPS on Jetson Orin Nano | Edge deployment claim |
| FPS on Jetson (30+ min) | Thermal stability (important!) |

### Ablation Table (Required for AAAI)

| Model variant | mAP@50 | GFLOPs | Tiny mAP |
|---------------|--------|--------|----------|
| YOLOv8-S baseline | — | — | — |
| RT-DETR baseline | — | — | — |
| Ours, no CFCR (β=0) | — | — | — |
| Ours, no cache (ε=1) | — | — | — |
| Ours, no motion path | — | — | — |
| **Ours, full model** | — | — | — |

Each row removes one component. This proves every architectural
decision was justified — which is what AAAI reviewers specifically look for.

---

## 4. What You Do NOT Build For The Workshop

These are explicitly Phase 2 (iDEX):
- Language output / threat summary text
- Swarm behavioral classification
- IR pathway
- Multi-drone identity tracking
- Synthetic AirSim swarm data

Keep scope tight. One claim, proven cleanly.

---

## 5. Open Release (Critical for AAAI Acceptance)

Per AAAI's own guidance, papers releasing open datasets or reproducible
code are especially encouraged. You release:

1. Fine-tuned model weights on HuggingFace
2. Training code on GitHub
3. Sparse activation visualization tool
   (the visual showing cached sky tokens vs. active drone tokens)

The visualization alone will be cited by other papers.
It makes your efficiency claim tangible to non-ML reviewers.
