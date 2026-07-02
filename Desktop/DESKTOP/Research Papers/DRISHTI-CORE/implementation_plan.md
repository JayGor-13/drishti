# DRISHTI-CORE v2 — Production Implementation Plan
**Classification:** Internal Engineering Specification  
**Version:** 2.0.0  
**Status:** Approved for Implementation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Directory Structure](#2-project-directory-structure)
3. [Dependencies & Environment](#3-dependencies--environment)
4. [Data Pipeline](#4-data-pipeline)
5. [Module-by-Module Implementation Blueprint](#5-module-by-module-implementation-blueprint)
   - 5.1 Config System
   - 5.2 Local Differential Motion Invariant (LDMI)
   - 5.3 MotionCNN
   - 5.4 Multi-Source Crop Proposal Engine
   - 5.5 Frozen Crop Encoder
   - 5.6 Temporal Fusion Transformer
   - 5.7 Sparse Mixture-of-Experts
   - 5.8 Detection Head
   - 5.9 Full Pipeline
   - 5.10 Inference Tracker
6. [Loss Functions](#6-loss-functions)
7. [Activation Functions](#7-activation-functions)
8. [Hyperparameter Specification & Justification](#8-hyperparameter-specification--justification)
9. [Staged Training Procedure](#9-staged-training-procedure)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Ablation Study Design](#11-ablation-study-design)
12. [Baseline Comparisons](#12-baseline-comparisons)
13. [Failure Case Analysis](#13-failure-case-analysis)
14. [Compute Budget Plan](#14-compute-budget-plan)
15. [Reproducibility Checklist](#15-reproducibility-checklist)

---

## 1. Overview

### 1.1 The Problem Statement
Standard video object detectors fail on tiny UAV targets captured by a moving camera because:
1. They confuse camera-induced background motion with target motion.
2. They cannot detect targets that are stationary in pixel space while the camera follows them.
3. They use non-causal future frames, preventing real-time deployment.
4. They lack a re-acquisition mechanism for targets occluded by structures.

### 1.2 Our Solution in One Paragraph
DRISHTI-CORE v2 decouples background motion from target motion using a **parameter-free Local Differential Motion Invariant (LDMI)** layer applied before any learned component. A **MotionCNN** converts the filtered signal into a spatial anomaly heatmap. An **adaptive crop proposal engine** allocates 8 crops per frame across four sources — tracker-guided, heatmap-driven, frame-edge surveillance, and periodic interior scanning — ensuring complete spatial coverage without redundant computation. A **frozen crop encoder** maps visual patches to feature vectors. A **causal temporal fusion transformer** integrates temporal context across 5 past frames. A **sparse top-2 MoE** provides model capacity at low active parameter cost. A **detection head** regresses bounding boxes and objectness. An **inference-time tracker** maintains multi-target state and feeds predicted coordinates back into the crop engine. Every component operates causally (zero look-ahead).

### 1.3 Implementation Philosophy
- **From Scratch:** No dependency on any pre-existing DRISHTI codebase. This plan defines the complete implementation starting point.
- **Production Quality:** Every module is a standalone, testable, documented class. Every public method has a typed signature.
- **Reproducible:** Fixed seeds, logged hyperparameters, deterministic data loading.
- **Modular by Design:** You can swap out any module (e.g., replace MotionCNN with an optical-flow estimate) without touching other modules.

---

## 2. Project Directory Structure

```
drishti_v2/
│
├── configs/                          # All YAML configuration files
│   ├── default.yaml                  # Base config
│   ├── ablation_no_ldmi.yaml
│   ├── ablation_no_edge_crops.yaml
│   ├── ablation_dense_moe.yaml
│   ├── ablation_e2e_training.yaml
│   └── sweep_crops_and_period.yaml
│
├── data/                             # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py                    # AntiUAVDataset class
│   ├── collator.py                   # DRISHTICollator class
│   ├── augmentations.py              # VideoAugmentation class
│   └── utils.py                      # box format conversions, normalization
│
├── models/                           # All model modules
│   ├── __init__.py
│   ├── config.py                     # DRISHTIConfig dataclass
│   ├── ldmi.py                       # LocalDifferentialMotion class
│   ├── motion_cnn.py                 # MotionCNN class
│   ├── crop_proposal.py              # CropProposalEngine class
│   ├── crop_encoder.py               # CropEncoder class
│   ├── temporal_fusion.py            # CausalTemporalFusion class
│   ├── moe.py                        # SparseMoE + Expert classes
│   ├── detection_head.py             # DetectionHead class
│   └── pipeline.py                   # DRISHTIPipeline (assembles all above)
│
├── tracker/
│   ├── __init__.py
│   └── tracker.py                    # Track + SimpleTracker classes
│
├── training/
│   ├── __init__.py
│   ├── losses.py                     # All loss functions
│   ├── trainer.py                    # DRISHTITrainer class
│   ├── scheduler.py                  # LR scheduler factory
│   └── stage_control.py             # Freeze/unfreeze logic per stage
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                    # All metric computations
│   ├── evaluator.py                  # DRISHTIEvaluator class
│   └── visualize.py                  # Bounding box & heatmap visualization
│
├── experiments/
│   ├── __init__.py
│   ├── run_training.py               # Main training entry point
│   ├── run_eval.py                   # Standalone evaluation script
│   ├── run_ablation.py               # Ablation sweep runner
│   └── run_inference.py              # Real-time inference script
│
├── tests/
│   ├── test_ldmi.py
│   ├── test_motion_cnn.py
│   ├── test_crop_proposal.py
│   ├── test_temporal_fusion.py
│   ├── test_moe.py
│   ├── test_detection_head.py
│   ├── test_pipeline.py
│   ├── test_tracker.py
│   └── test_losses.py
│
├── scripts/
│   ├── download_antiuav.sh          # Dataset download helper
│   ├── prepare_dataset.py           # Frame extraction & annotation parsing
│   └── benchmark_latency.py         # FPS/GFLOPs profiling script
│
├── checkpoints/                      # Saved model weights (gitignored)
├── logs/                             # TensorBoard / W&B logs (gitignored)
├── results/                          # Evaluation outputs, CSVs, plots
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. Dependencies & Environment

### 3.1 `requirements.txt`
```
# Core deep learning
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Video & image processing
opencv-python-headless>=4.8.0
Pillow>=10.0.0
imageio>=2.31.0

# Scientific computing
numpy>=1.24.0
scipy>=1.11.0

# Data loading & management
pycocotools>=2.0.7
h5py>=3.9.0

# Configuration management
omegaconf>=2.3.0
hydra-core>=1.3.0

# Experiment tracking
tensorboard>=2.14.0
wandb>=0.15.0

# Evaluation
torchmetrics>=1.2.0
motmetrics>=1.4.0    # for Multi-Object Tracking metrics

# Profiling & benchmarking
fvcore>=0.1.5.post20221221    # for FLOPs counting
thop>=0.1.1                   # alternative FLOPs counter

# Utilities
tqdm>=4.66.0
pyyaml>=6.0.1
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tabulate>=0.9.0

# Code quality
pytest>=7.4.0
black>=23.9.0
isort>=5.12.0
mypy>=1.5.0
```

### 3.2 Python & CUDA
- Python: 3.10+
- CUDA: 11.8+ (for PyTorch 2.x)
- Recommended: `conda env create -f environment.yml`

### 3.3 Environment Setup Commands
```bash
conda create -n drishti python=3.10
conda activate drishti
pip install -r requirements.txt
python setup.py develop
```

---

## 4. Data Pipeline

### 4.1 `data/dataset.py` — `AntiUAVDataset`

**Class Purpose:** Load Anti-UAV RGB video sequences as fixed-length temporal windows with their annotations.

```python
class AntiUAVDataset(torch.utils.data.Dataset):
    """
    Anti-UAV dataset loader for temporal video windows.
    
    Expected directory layout:
        data_root/
            train/
                <sequence_name>/
                    visible/
                        000000.jpg
                        000001.jpg
                        ...
                    visible.json   # {"gt_rect": [[x,y,w,h],...], "exist": [1,1,0,...]}
            val/
                ...
            test/
                ...
    
    Every item returned is a temporal clip of length `num_frames` with
    the ground-truth annotation for every frame in the clip.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str,                       # "train" | "val" | "test"
        num_frames: int = 5,              # Temporal window length
        frame_size: tuple[int, int] = (448, 448),
        clip_stride: int = 4,             # Stride between clips
        frame_stride: int = 1,            # Stride between frames within a clip
        modality: str = "visible",        # "visible" | "infrared"
        box_format: str = "xywh",         # Input annotation format
        augment: bool = True,
        sequence_filter: list[str] | None = None,  # Filter to specific sequences
    ) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Returns:
            {
                "frames": Tensor [num_frames, 3, H, W],   # normalized [0,1]
                "targets": list[dict],                     # per-frame GT
                    each dict: {
                        "boxes": Tensor [N, 4],            # normalized [cx,cy,w,h]
                        "labels": Tensor [N],              # all 1s (drone class)
                        "visible": bool,                   # is target visible?
                    }
                "meta": {
                    "sequence": str,
                    "frame_indices": list[int],
                }
            }
        """
        ...
```

### 4.2 `data/collator.py` — `DRISHTICollator`

```python
class DRISHTICollator:
    """Collates variable-length target lists into padded batch tensors."""
    
    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        """
        Returns:
            {
                "frames": Tensor [B, T, 3, H, W],
                "targets": list[list[dict]],     # [B][T] — not padded, left as lists
                "meta": list[dict],
            }
        """
        ...
```

### 4.3 `data/augmentations.py` — `VideoAugmentation`

```python
class VideoAugmentation:
    """
    Applies consistent spatial augmentations across all frames of a video clip.
    All transforms are applied identically to every frame in the temporal window
    to preserve temporal consistency.
    
    Training augmentations:
        - RandomHorizontalFlip(p=0.5)
        - RandomColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        - RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0), p=0.2)
        - RandomErasing(p=0.2, scale=(0.02, 0.1))   # simulate partial occlusion
    
    Inference augmentations:
        - None (only normalize)
    """
    
    def __init__(self, train: bool = True) -> None:
        ...
    
    def __call__(
        self,
        frames: list[Tensor],     # list of [3, H, W] raw tensors
        targets: list[dict],
    ) -> tuple[list[Tensor], list[dict]]:
        ...
```

---

## 5. Module-by-Module Implementation Blueprint

### 5.1 `models/config.py` — `DRISHTIConfig`

```python
@dataclass
class DRISHTIConfig:
    """
    Master configuration for the DRISHTI-CORE v2 pipeline.
    All hyperparameter values are listed here with their defaults and justification.
    """
    
    # ─── Image Properties ────────────────────────────────────────────────────
    image_channels: int = 3
    image_height: int = 448
    image_width: int = 448
    
    # ─── LDMI Parameters ─────────────────────────────────────────────────────
    ldmi_scales: tuple[int, ...] = (15, 31)
    # Justification: Scale 15 catches small drones (<10px); scale 31 catches
    # medium drones (10-30px). Larger scales risk including target in the
    # averaging window itself, reducing the residual signal.
    
    # ─── MotionCNN Parameters ────────────────────────────────────────────────
    motion_cnn_channels: tuple[int, ...] = (32, 64, 64)
    # Architecture: 9 -> 32 -> 64 -> 64 -> 1 with strides (2, 2, 1)
    # Produces heatmap at H/4 x W/4 resolution (112x112 for 448 input)
    
    # ─── Crop Proposal Parameters ────────────────────────────────────────────
    num_crops: int = 8
    # MUST SATISFY: num_crops >= num_guided + num_edge + num_motion_min
    
    crop_size: int = 64             # pixels, square crop extracted from current frame
    border_width_frac: float = 0.07 # fraction of frame width/height as edge zone
    scan_period: int = 4            # Global interior scan every N frames
    # Ablation target: sweep over scan_period ∈ {2, 4, 8, 16}
    
    # ─── Crop Encoder Parameters ─────────────────────────────────────────────
    encoder_feature_dim: int = 256
    encoder_frozen: bool = True     # Frozen during Stage 1 and 2 training
    
    # ─── Temporal Fusion Parameters ──────────────────────────────────────────
    temporal_window: int = 5        # Number of past frames in context
    temporal_heads: int = 4
    temporal_layers: int = 2
    temporal_ffn_dim: int = 512
    temporal_dropout: float = 0.1
    
    # ─── MoE Parameters ──────────────────────────────────────────────────────
    num_experts: int = 8
    top_k: int = 2
    expert_ffn_dim: int = 512
    moe_dropout: float = 0.1
    moe_balance_weight: float = 0.01   # Weight of auxiliary load-balance loss
    
    # ─── Detection Head Parameters ───────────────────────────────────────────
    head_hidden_dim: int = 256
    objectness_threshold: float = 0.3  # Inference confidence gate
    
    # ─── Tracker Parameters ──────────────────────────────────────────────────
    tracker_dist_threshold: float = 0.15  # Normalized Euclidean distance gate
    tracker_max_coast: int = 15           # Frames before track is deleted
    tracker_birth_threshold: float = 0.3  # Min confidence for new track birth
```

---

### 5.2 `models/ldmi.py` — `LocalDifferentialMotion`

**Class Purpose:** Non-learnable, parameter-free preprocessing module that computes camera-motion-invariant anomaly residuals.

```python
class LocalDifferentialMotion(nn.Module):
    """
    Local Differential Motion Invariant (LDMI) Preprocessing Layer.
    
    Decouples target motion from camera ego-motion by computing the
    per-pixel deviation from local spatial neighbourhood motion.
    
    Algorithm:
        1. Compute frame differences: d_old = f_{t-1} - f_{t-2}
                                      d_new = f_t - f_{t-1}
        2. Average-pool each difference at multiple spatial scales
           to estimate local uniform background motion.
        3. Subtract pooled average from original difference to get residual:
              r = |d - AvgPool(d, k)|
        4. Fuse multi-scale residuals via element-wise max.
        5. Return [r_old, f_t, r_new] — same shape as input triplet.
    
    Mathematical guarantee:
        For any uniform translation vector v across the neighbourhood,
        r = |v - avg(v, v, ..., v)| = |v - v| = 0
        Therefore background motion always suppressed to 0.
        Only pixels with non-local motion survive.
    
    Parameters:
        image_channels (int): Number of channels per frame (default: 3)
        scales (tuple[int, ...]): Average pooling kernel sizes (default: (15, 31))
    
    Learnable parameters: 0
    FLOPs: ~2 * len(scales) * H * W * C avg_pool operations
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        scales: tuple[int, ...] = (15, 31),
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.scales = scales
        # No learnable parameters — registered as module for device tracking
    
    def _compute_residual(self, diff: Tensor) -> Tensor:
        """
        Compute the multi-scale local differential residual.
        
        Args:
            diff: Frame difference tensor [B, C, H, W]
        
        Returns:
            residual: Max-fused absolute residual [B, C, H, W]
                      Values near 0 = background motion
                      Values near 1 = anomalous motion (likely target)
        """
        residuals = []
        for k in self.scales:
            padding = k // 2
            local_mean = F.avg_pool2d(
                diff,
                kernel_size=k,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
            residuals.append(torch.abs(diff - local_mean))
        
        # Element-wise max across scales — captures all target size regimes
        fused = residuals[0]
        for r in residuals[1:]:
            fused = torch.max(fused, r)
        return fused
    
    def forward(self, triplet: Tensor) -> Tensor:
        """
        Args:
            triplet: Causal frame triplet [B, C*3, H, W]
                     channels 0:C   = f_{t-2}
                     channels C:2C  = f_{t-1}
                     channels 2C:3C = f_t
        
        Returns:
            filtered: [B, C*3, H, W]
                      channels 0:C   = residual r_{t-1}  (older anomaly)
                      channels C:2C  = f_t               (raw appearance)
                      channels 2C:3C = residual r_t      (recent anomaly)
        
        Note:
            The raw f_t is preserved in the center channels so the
            downstream MotionCNN has access to appearance context.
            This prevents it from firing on motion artifacts alone.
        """
        C = self.image_channels
        f_old  = triplet[:, 0:C]        # f_{t-2}
        f_prev = triplet[:, C:2*C]      # f_{t-1}
        f_curr = triplet[:, 2*C:3*C]    # f_t
        
        d_old = f_prev - f_old          # older motion
        d_new = f_curr - f_prev         # recent motion
        
        r_old = self._compute_residual(d_old)
        r_new = self._compute_residual(d_new)
        
        return torch.cat([r_old, f_curr, r_new], dim=1)
```

---

### 5.3 `models/motion_cnn.py` — `MotionCNN`

**Class Purpose:** Learnable CNN that converts the LDMI-filtered triplet into a 2D spatial heatmap of anomalous motion.

```python
class MotionCNN(nn.Module):
    """
    Convolutional anomaly localizer.
    
    Takes the LDMI-filtered triplet and produces a single-channel heatmap
    where high values indicate likely target presence.
    
    Architecture:
        Input: [B, C*3, H, W]    (default 9 channels for RGB)
        L1: Conv(9->32, k=3, s=2, p=1) + BN + ReLU  -> [B, 32, H/2, W/2]
        L2: Conv(32->64, k=3, s=2, p=1) + BN + ReLU -> [B, 64, H/4, W/4]
        L3: Conv(64->64, k=3, s=1, p=1) + BN + ReLU -> [B, 64, H/4, W/4]
        L4: Conv(64->1, k=1, s=1, p=0) + Sigmoid    -> [B, 1, H/4, W/4]
    
    Training signal:
        Supervised by a Gaussian heatmap centered on GT bounding boxes.
        Loss: MSE between predicted heatmap and GT Gaussian heatmap.
    
    Parameters:
        image_channels (int): Channels per frame (default: 3)
        hidden_channels (tuple[int, ...]): Feature map sizes (default: (32, 64, 64))
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: tuple[int, ...] = (32, 64, 64),
    ) -> None:
        super().__init__()
        in_channels = image_channels * 3  # triplet input
        
        layers = []
        for idx, out_ch in enumerate(hidden_channels):
            stride = 2 if idx < 2 else 1   # only first two layers downsample
            layers += [
                nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_ch
        
        # Final 1x1 projection to single channel
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=1, bias=True))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, filtered_triplet: Tensor) -> Tensor:
        """
        Args:
            filtered_triplet: [B, C*3, H, W] — output of LocalDifferentialMotion
        
        Returns:
            heatmap: [B, 1, H//4, W//4] — values in [0, 1]
        """
        return self.net(filtered_triplet)
    
    @staticmethod
    def make_gt_heatmap(
        boxes: Tensor,
        heatmap_size: tuple[int, int],
        sigma: float = 2.0,
    ) -> Tensor:
        """
        Generates a Gaussian heatmap ground truth from bounding box annotations.
        
        Args:
            boxes: Normalized [cx, cy, w, h] boxes [N, 4]
            heatmap_size: (H_h, W_h) of the heatmap output
            sigma: Gaussian spread (default: 2.0 heatmap pixels)
        
        Returns:
            heatmap: [1, H_h, W_h] with values in [0, 1]
        """
        H_h, W_h = heatmap_size
        heatmap = torch.zeros(1, H_h, W_h)
        for box in boxes:
            cx = int(box[0] * W_h)
            cy = int(box[1] * H_h)
            for y in range(H_h):
                for x in range(W_h):
                    heatmap[0, y, x] = max(
                        heatmap[0, y, x],
                        torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                    )
        return heatmap
```

---

### 5.4 `models/crop_proposal.py` — `CropProposalEngine`

**Class Purpose:** The central scheduling and routing module. It decides WHERE to place the 8 crop windows every frame.

```python
@dataclass
class ProposalOutput:
    """All outputs from the crop proposal stage."""
    crops: Tensor              # [B*K, C, crop_h, crop_w] — extracted image patches
    centers: Tensor            # [B, K, 2] — normalized (x,y) centers
    scores: Tensor             # [B, K] — heatmap value at each crop center
    source_labels: Tensor      # [B, K] — 0=motion, 1=edge, 2=grid, 3=guided
    heatmap: Tensor            # [B, 1, H_h, W_h] — full anomaly heatmap


class CropProposalEngine(nn.Module):
    """
    Multi-Source Crop Attention Proposal Engine.
    
    Assembles exactly `num_crops` (default: 8) crop coordinates per frame
    from four sources, ordered by priority:
    
      1. GUIDED   — coordinates predicted by the inference tracker (highest trust)
      2. MOTION   — top-k peaks from the MotionCNN heatmap
      3. EDGE     — boundary surveillance coordinates
      4. GRID     — periodic interior sweep coordinates
    
    The priority ordering ensures that tracker feedback always gets slots first.
    Remaining slots are filled by motion peaks, then edge positions, then grid.
    
    On training: tracker is absent, so allocation is MOTION + EDGE + GRID.
    On inference: full allocation with guided coordinates from the tracker.
    
    Arguments:
        config (DRISHTIConfig): Pipeline configuration object.
    """
    
    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        self.config = config
        
        # Pre-compute the static grid positions for interior sweep
        # 2x2 grid covering the inner 60% of the frame
        self._interior_grid = [
            (0.30, 0.30), (0.30, 0.70),
            (0.70, 0.30), (0.70, 0.70),
        ]
        
        # Pre-compute edge midpoints for both alternating patterns
        bw = config.border_width_frac
        self._edge_horizontal = [(bw / 2, 0.5), (1 - bw / 2, 0.5)]   # Left, Right
        self._edge_vertical   = [(0.5, bw / 2), (0.5, 1 - bw / 2)]   # Top, Bottom
    
    def _get_motion_centers(
        self, heatmap: Tensor, n: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Extract top-n peak coordinates from the heatmap using non-maximum suppression.
        
        Args:
            heatmap: [B, 1, H_h, W_h]
            n: Number of peaks to extract per batch item
        
        Returns:
            centers: [B, n, 2] — normalized (x,y) coords in [0,1]
            scores:  [B, n]    — heatmap confidence at each center
        """
        B, _, H_h, W_h = heatmap.shape
        # Suppress non-maxima with max pooling trick
        suppressed = F.max_pool2d(
            heatmap, kernel_size=3, stride=1, padding=1
        )
        peaks = (heatmap == suppressed).float() * heatmap
        flat = peaks.view(B, -1)
        scores, indices = torch.topk(flat, k=n, dim=-1)
        
        row = (indices // W_h).float() / H_h
        col = (indices % W_h).float() / W_h
        centers = torch.stack([col, row], dim=-1)   # (x, y)
        return centers, scores
    
    def _get_edge_centers(
        self, frame_index: int, batch_size: int, device: torch.device,
    ) -> Tensor:
        """
        Returns two border midpoint coordinates, alternating horizontal/vertical.
        
        Args:
            frame_index: Global frame index to determine alternation
            batch_size: B
            device: Tensor device
        
        Returns:
            centers: [B, 2, 2] — two (x,y) pairs per batch item
        """
        pattern = self._edge_horizontal if frame_index % 2 == 1 else self._edge_vertical
        centers = torch.tensor(pattern, device=device)        # [2, 2]
        return centers.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _get_grid_centers(
        self, batch_size: int, device: torch.device,
    ) -> Tensor:
        """
        Returns the 4 static interior grid centers.
        
        Returns:
            centers: [B, 4, 2]
        """
        centers = torch.tensor(self._interior_grid, device=device)   # [4, 2]
        return centers.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _extract_crops(
        self, frame: Tensor, centers: Tensor,
    ) -> Tensor:
        """
        Extract fixed-size patches from the current frame at given coordinates.
        
        Args:
            frame: Current frame [B, C, H, W]
            centers: [B, K, 2] normalized (x,y) centers
        
        Returns:
            crops: [B*K, C, crop_h, crop_w]
        """
        B, C, H, W = frame.shape
        K = centers.shape[1]
        crop_h = crop_w = self.config.crop_size
        
        # Convert normalized to pixel coordinates
        px = (centers[..., 0] * W).long()
        py = (centers[..., 1] * H).long()
        
        crops = []
        for b in range(B):
            for k in range(K):
                x0 = px[b, k] - crop_w // 2
                y0 = py[b, k] - crop_h // 2
                # Use replicate padding at boundaries
                crop = F.pad(
                    frame[b],
                    pad=(
                        max(0, -x0), max(0, x0 + crop_w - W),
                        max(0, -y0), max(0, y0 + crop_h - H),
                    ),
                    mode="replicate",
                )
                x0c = max(x0, 0)
                y0c = max(y0, 0)
                crops.append(crop[:, y0c:y0c + crop_h, x0c:x0c + crop_w])
        
        return torch.stack(crops)
    
    def forward(
        self,
        frame: Tensor,
        heatmap: Tensor,
        frame_index: int,
        guided_centers: Tensor | None = None,
    ) -> ProposalOutput:
        """
        Assemble crop proposals from all four sources.
        
        Args:
            frame: Current frame [B, C, H, W]
            heatmap: MotionCNN output [B, 1, H_h, W_h]
            frame_index: Global frame counter (for edge alternation + grid scheduling)
            guided_centers: Optional [B, K_guided, 2] from tracker. None during training.
        
        Returns:
            ProposalOutput with crops and metadata
        """
        B = frame.shape[0]
        K = self.config.num_crops
        device = frame.device
        is_scan_frame = (frame_index % self.config.scan_period == 0)
        
        all_centers = []
        all_scores = []
        all_sources = []
        
        # 1. GUIDED CROPS (highest priority)
        n_guided = 0
        if guided_centers is not None:
            n_guided = min(guided_centers.shape[1], K - 2)  # always reserve ≥2 for discovery
            all_centers.append(guided_centers[:, :n_guided])
            all_scores.append(torch.ones(B, n_guided, device=device))
            all_sources.extend([3] * n_guided)  # 3 = guided
        
        remaining = K - n_guided
        
        # 2. GRID CROPS on scan frames
        n_grid = 0
        if is_scan_frame:
            grid = self._get_grid_centers(B, device)
            n_grid = min(4, remaining - 2)   # always reserve ≥2 for motion/edge
            all_centers.append(grid[:, :n_grid])
            all_scores.append(torch.zeros(B, n_grid, device=device))
            all_sources.extend([2] * n_grid)  # 2 = grid
        
        remaining = K - n_guided - n_grid
        
        # 3. EDGE CROPS
        edge = self._get_edge_centers(frame_index, B, device)   # [B, 2, 2]
        n_edge = min(2, remaining - 1)   # always reserve ≥1 for motion
        all_centers.append(edge[:, :n_edge])
        all_scores.append(torch.zeros(B, n_edge, device=device))
        all_sources.extend([1] * n_edge)   # 1 = edge
        
        remaining = K - n_guided - n_grid - n_edge
        
        # 4. MOTION CROPS (fill remainder from heatmap peaks)
        if remaining > 0:
            motion_centers, motion_scores = self._get_motion_centers(heatmap, remaining)
            all_centers.append(motion_centers)
            all_scores.append(motion_scores)
            all_sources.extend([0] * remaining)  # 0 = motion
        
        # Assemble
        centers = torch.cat(all_centers, dim=1)    # [B, K, 2]
        scores  = torch.cat(all_scores, dim=1)     # [B, K]
        source_labels = torch.tensor(all_sources, device=device).unsqueeze(0).expand(B, -1)
        
        # Extract image patches
        crops = self._extract_crops(frame, centers)   # [B*K, C, crop_size, crop_size]
        
        return ProposalOutput(
            crops=crops,
            centers=centers,
            scores=scores,
            source_labels=source_labels,
            heatmap=heatmap,
        )
```

---

### 5.5 `models/crop_encoder.py` — `CropEncoder`

```python
class CropEncoder(nn.Module):
    """
    Visual feature extractor for 64x64 crop patches.
    
    Maps each crop to a fixed-dimensional feature vector.
    Frozen during Stage 1 (Detector) and Stage 2 (Temporal) training.
    
    Architecture:
        Conv(3->64,  k=3, s=1, p=1) + BN + ReLU  -> [64, 64, 64]
        Conv(64->128, k=3, s=2, p=1) + BN + ReLU -> [128, 32, 32]
        Conv(128->256,k=3, s=2, p=1) + BN + ReLU -> [256, 16, 16]
        AdaptiveAvgPool2d(1)                       -> [256, 1, 1]
        Flatten + Linear(256->256)                 -> [256]
    
    Parameters:
        out_dim (int): Output feature dimension (default: 256)
    
    Learnable parameters: ~600K
    """
    
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.frozen = False
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, out_dim)
    
    def forward(self, crops: Tensor) -> Tensor:
        """
        Args:
            crops: [B*K, 3, 64, 64]
        
        Returns:
            features: [B*K, out_dim]
        """
        x = self.backbone(crops).flatten(1)
        return self.head(x)
    
    def freeze(self) -> None:
        """Freeze all parameters. Called before Stage 1 training begins."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.frozen = True
    
    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self.frozen = False
```

---

### 5.6 `models/temporal_fusion.py` — `CausalTemporalFusion`

```python
class CausalTemporalFusion(nn.Module):
    """
    Causal Temporal Fusion Transformer.
    
    Fuses crop features across a past temporal window of length T=5.
    Strictly causal — the model at time t only attends to {t-4,...,t}.
    A causal attention mask is used during training to enforce this.
    
    Input format:
        Feature sequence [B, T, K, D+1] where:
            T = temporal window (5)
            K = number of crops (8)
            D = encoder feature dim (256)
            +1 = scalar heatmap score appended to each crop's feature
    
    Processing:
        1. Reshape to [B*K, T, D+1]  — treat each crop independently over time
        2. Add learnable temporal positional embedding
        3. Apply causal self-attention mask (lower-triangular)
        4. Pass through 2 Transformer encoder blocks
        5. Extract the last token (present timestep)
        6. Project [D+1] -> [D]
        7. Reshape back to [B, K, D]
    
    Why causal mask?
        During training, the sequence is built from frames [t-T+1,...,t].
        Without a causal mask, the transformer can attend to future frames
        within the sequence. The mask prevents this, matching inference behaviour
        where only past frames are available.
    
    Parameters:
        feature_dim (int):   Crop encoder output dimension + 1 (default: 257)
        out_dim (int):       Output feature dimension (default: 256)
        nhead (int):         Attention heads (default: 4)
        num_layers (int):    Transformer encoder depth (default: 2)
        ffn_dim (int):       FFN inner width (default: 512)
        dropout (float):     Attention and FFN dropout (default: 0.1)
        max_seq_len (int):   Maximum temporal window (default: 5)
    """
    
    def __init__(
        self,
        feature_dim: int = 257,
        out_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 5,
    ) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Learnable positional embedding for temporal positions
        self.pos_embed = nn.Embedding(max_seq_len, feature_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Causal mask (cached)
        self.register_buffer(
            "_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(max_seq_len),
        )
        
        self.proj = nn.Linear(feature_dim, out_dim)
    
    def forward(self, sequence: Tensor) -> Tensor:
        """
        Args:
            sequence: [B, T, K, D+1] — temporal feature sequence
        
        Returns:
            fused: [B, K, D] — features for the current timestep only
        """
        B, T, K, D = sequence.shape
        
        # Reshape: treat each crop as independent temporal sequence
        x = sequence.permute(0, 2, 1, 3).reshape(B * K, T, D)  # [B*K, T, D]
        
        # Add temporal positional embeddings
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)
        
        # Causal self-attention
        mask = self._causal_mask[:T, :T]
        x = self.transformer(x, mask=mask)  # [B*K, T, D]
        
        # Extract present-timestep token and project
        present = x[:, -1, :]              # [B*K, D]
        out = self.proj(present)           # [B*K, out_dim]
        
        return out.reshape(B, K, -1)       # [B, K, out_dim]
```

---

### 5.7 `models/moe.py` — `Expert` & `SparseMoE`

```python
class Expert(nn.Module):
    """
    Single FFN Expert module.
    
    Architecture: Linear(D->4D) -> GELU -> Dropout -> Linear(4D->D)
    
    Note: GELU is used instead of ReLU here because it provides
    smoother gradients for the expert output, which stabilizes
    routing diversity during early training.
    """
    
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    """
    Sparse Top-K Mixture-of-Experts with auxiliary load-balancing.
    
    Routes each input token to the top_k experts by learned routing probabilities.
    Computes a weighted combination of their outputs.
    
    Auxiliary Loss (Switch Transformer formulation):
        L_balance = n_experts * sum_e(f_e * p_e)
        where:
            f_e = fraction of tokens dispatched to expert e
            p_e = mean routing probability assigned to expert e
        
        This penalises routing collapse (all tokens going to one expert)
        by maximising routing entropy.
    
    Parameters:
        d_model (int):      Input/output feature dimension (default: 256)
        num_experts (int):  Total number of experts (default: 8)
        top_k (int):        Number of experts activated per token (default: 2)
        ffn_dim (int):      Expert FFN inner dimension (default: 512)
        dropout (float):    Expert dropout (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, ffn_dim, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, K, D] or [N, D]
        
        Returns:
            out: Same shape as x — transformed features
            balance_loss: Scalar auxiliary loss for load balancing
        """
        *dims, D = x.shape
        x_flat = x.reshape(-1, D)   # [N, D]
        N = x_flat.shape[0]
        
        # Routing probabilities
        logits = self.router(x_flat)                        # [N, E]
        probs = torch.softmax(logits, dim=-1)               # [N, E]
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)  # [N, k]
        
        # Normalize top-k weights to sum to 1
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [N, k]
        
        # Compute outputs for each selected expert
        out = torch.zeros_like(x_flat)
        for rank in range(self.top_k):
            expert_idx = topk_indices[:, rank]     # [N]
            weights = topk_weights[:, rank]         # [N]
            
            for e in range(self.num_experts):
                token_mask = (expert_idx == e)
                if not token_mask.any():
                    continue
                expert_out = self.experts[e](x_flat[token_mask])   # [M, D]
                out[token_mask] += weights[token_mask].unsqueeze(-1) * expert_out
        
        # Auxiliary load-balance loss
        # f_e: fraction of tokens routed to expert e
        f_e = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            f_e[e] = (topk_indices == e).float().mean()
        
        # p_e: mean routing probability for expert e
        p_e = probs.mean(dim=0)   # [E]
        
        balance_loss = (self.num_experts * (f_e * p_e).sum())
        
        return out.reshape(*dims, D), balance_loss
```

---

### 5.8 `models/detection_head.py` — `DetectionHead`

```python
class DetectionHead(nn.Module):
    """
    Detection head that maps per-crop features to object predictions.
    
    For each crop, independently predicts:
        - objectness: probability this crop contains a target (scalar)
        - box:        bounding box [cx, cy, w, h] relative to crop boundaries
                      normalized to [0,1] in crop-space
    
    Architecture:
        Objectness branch:
            LayerNorm -> Linear(D->1)  (no activation; BCEWithLogitsLoss)
        
        Box regression branch:
            LayerNorm -> Linear(D->D) -> GELU -> Linear(D->4) -> Sigmoid
    
    Why separate branches?
        Objectness and box regression have very different gradient magnitudes.
        Shared weights would cause the dominant task (objectness) to suppress
        the box regression signals.
    
    Why Sigmoid for box?
        Forces outputs to [0,1] = valid normalized crop-space coordinates.
    """
    
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        
        self.objectness_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )
        
        self.box_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 4),
            nn.Sigmoid(),
        )
    
    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            features: [B, K, D]
        
        Returns:
            objectness_logits: [B, K, 1]  — raw (pre-sigmoid) objectness scores
            boxes: [B, K, 4]              — normalized [cx, cy, w, h] in crop-space
        """
        return self.objectness_head(features), self.box_head(features)
```

---

### 5.9 `models/pipeline.py` — `DRISHTIPipeline`

```python
@dataclass
class PipelineOutput:
    heatmap: Tensor              # [B, 1, H_h, W_h]
    proposal_centers: Tensor     # [B, K, 2]
    proposal_scores: Tensor      # [B, K]
    proposal_sources: Tensor     # [B, K]  source label per crop
    crop_features: Tensor        # [B, K, D]
    fused_features: Tensor       # [B, K, D]
    moe_features: Tensor         # [B, K, D]
    objectness_logits: Tensor    # [B, K, 1]
    boxes: Tensor                # [B, K, 4]
    balance_loss: Tensor         # scalar


class DRISHTIPipeline(nn.Module):
    """
    Full DRISHTI-CORE v2 Pipeline (training mode).
    
    Assembles all modules in sequence. Maintains a rolling temporal buffer
    to provide the causal temporal fusion stage with past feature context.
    
    Call forward() for training (causal temporal window over clip).
    Call forward_stream() for real-time inference frame-by-frame.
    """
    
    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        self.config = config
        
        self.ldmi         = LocalDifferentialMotion(config.image_channels, config.ldmi_scales)
        self.motion_cnn   = MotionCNN(config.image_channels, config.motion_cnn_channels)
        self.crop_engine  = CropProposalEngine(config)
        self.encoder      = CropEncoder(config.encoder_feature_dim)
        self.temporal     = CausalTemporalFusion(
            feature_dim=config.encoder_feature_dim + 1,  # +1 for score
            out_dim=config.encoder_feature_dim,
            nhead=config.temporal_heads,
            num_layers=config.temporal_layers,
            ffn_dim=config.temporal_ffn_dim,
            dropout=config.temporal_dropout,
            max_seq_len=config.temporal_window,
        )
        self.moe          = SparseMoE(
            d_model=config.encoder_feature_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            ffn_dim=config.expert_ffn_dim,
            dropout=config.moe_dropout,
        )
        self.head         = DetectionHead(config.encoder_feature_dim)
        
        # Rolling feature buffer for temporal fusion
        # Maintained as a deque of [B, K, D+1] tensors
        self._feature_buffer: list[Tensor] = []
    
    def forward(
        self,
        frames: Tensor,               # [B, T, C, H, W] — full clip
        frame_index: int = 0,
        guided_centers: Tensor | None = None,
    ) -> PipelineOutput:
        """Training forward pass over a full temporal clip."""
        B, T, C, H, W = frames.shape
        
        # Build temporal feature sequence
        all_features = []
        last_heatmap = None
        last_centers = None
        last_scores = None
        last_sources = None
        last_logits = None
        last_boxes = None
        
        for t in range(T):
            # Step 1: Build causal triplet [t-2, t-1, t]
            t0 = max(0, t - 2)
            t1 = max(0, t - 1)
            triplet = torch.cat([frames[:, t0], frames[:, t1], frames[:, t]], dim=1)
            
            # Step 2: LDMI filter
            filtered = self.ldmi(triplet)
            
            # Step 3: Generate heatmap
            heatmap = self.motion_cnn(filtered)
            
            # Step 4: Crop proposals
            proposal = self.crop_engine(
                frame=frames[:, t],
                heatmap=heatmap,
                frame_index=frame_index + t,
                guided_centers=guided_centers if t == T - 1 else None,
            )
            
            # Step 5: Encode crops
            encoded = self.encoder(proposal.crops)        # [B*K, D]
            encoded = encoded.reshape(B, self.config.num_crops, -1)  # [B, K, D]
            
            # Append motion scores to features
            scores = proposal.scores.unsqueeze(-1)        # [B, K, 1]
            augmented = torch.cat([encoded, scores], dim=-1)  # [B, K, D+1]
            all_features.append(augmented)
            
            last_heatmap = heatmap
            last_centers = proposal.centers
            last_scores = proposal.scores
            last_sources = proposal.source_labels
        
        # Step 6: Temporal fusion on the last T frames
        sequence = torch.stack(all_features, dim=1)   # [B, T, K, D+1]
        fused = self.temporal(sequence)               # [B, K, D]
        
        # Step 7: Sparse MoE
        moe_out, balance_loss = self.moe(fused)       # [B, K, D]
        
        # Step 8: Detection head
        logits, boxes = self.head(moe_out)            # logits: [B,K,1], boxes: [B,K,4]
        
        return PipelineOutput(
            heatmap=last_heatmap,
            proposal_centers=last_centers,
            proposal_scores=last_scores,
            proposal_sources=last_sources,
            crop_features=encoded,
            fused_features=fused,
            moe_features=moe_out,
            objectness_logits=logits,
            boxes=boxes,
            balance_loss=balance_loss,
        )
```

---

### 5.10 `tracker/tracker.py` — `Track` & `SimpleTracker`

```python
@dataclass
class Track:
    """Single target track state."""
    track_id: int
    center: Tensor       # [2] normalized (x,y)
    size: Tensor         # [2] normalized (w,h)
    velocity: Tensor     # [2] in normalized units/frame
    confidence: float
    age: int = 0
    coast_count: int = 0
    hit_count: int = 1


class SimpleTracker:
    """
    Inference-time multi-target state tracker.
    
    Maintains a list of Track objects across frames.
    Associates incoming detections to tracks using Euclidean distance gating.
    
    Track Lifecycle:
        BIRTH   -> confidence > birth_threshold, no matching track found
        ACTIVE  -> confirmed after hit_count >= 1 (simplified; could require 3 in stricter mode)
        COAST   -> no matching detection within dist_threshold; keep predicting
        DEAD    -> coast_count > max_coast; removed from table
    
    Output:
        get_guided_centers() returns the predicted center for each active
        track at time t+1, which are fed into CropProposalEngine as guided crops.
    """
    
    def __init__(
        self,
        dist_threshold: float = 0.15,
        max_coast: int = 15,
        birth_threshold: float = 0.3,
    ) -> None:
        self.dist_threshold = dist_threshold
        self.max_coast = max_coast
        self.birth_threshold = birth_threshold
        self.tracks: list[Track] = []
        self._next_id = 0
    
    def predict(self) -> None:
        """
        Constant-velocity state projection.
        Called BEFORE processing detections for the current frame.
        """
        for track in self.tracks:
            track.center = (track.center + track.velocity).clamp(0.0, 1.0)
            track.coast_count += 1
            track.age += 1
    
    def update(self, boxes: Tensor, logits: Tensor) -> None:
        """
        Associate detections to tracks and update state.
        
        Args:
            boxes: [K, 4] normalized [cx, cy, w, h] from detection head
            logits: [K, 1] objectness logits
        """
        confs = torch.sigmoid(logits.squeeze(-1))    # [K]
        high_conf_mask = confs > self.birth_threshold
        det_boxes = boxes[high_conf_mask]
        det_confs = confs[high_conf_mask]
        
        matched_det = set()
        matched_track = set()
        
        # Greedy distance matching
        for t_idx, track in enumerate(self.tracks):
            best_dist = float("inf")
            best_d_idx = -1
            for d_idx, det in enumerate(det_boxes):
                if d_idx in matched_det:
                    continue
                dist = torch.norm(track.center - det[:2]).item()
                if dist < best_dist:
                    best_dist = dist
                    best_d_idx = d_idx
            
            if best_dist < self.dist_threshold and best_d_idx >= 0:
                # Update matched track
                new_center = det_boxes[best_d_idx, :2]
                track.velocity = new_center - track.center
                track.center = new_center
                track.size = det_boxes[best_d_idx, 2:]
                track.confidence = det_confs[best_d_idx].item()
                track.coast_count = 0
                track.hit_count += 1
                matched_det.add(best_d_idx)
                matched_track.add(t_idx)
        
        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.coast_count <= self.max_coast]
        
        # Birth new tracks from unmatched detections
        for d_idx, det in enumerate(det_boxes):
            if d_idx not in matched_det:
                self.tracks.append(Track(
                    track_id=self._next_id,
                    center=det[:2].clone(),
                    size=det[2:].clone(),
                    velocity=torch.zeros(2),
                    confidence=det_confs[d_idx].item(),
                ))
                self._next_id += 1
    
    def get_guided_centers(self) -> Tensor | None:
        """
        Returns predicted positions of all active tracks.
        
        Returns:
            centers: [1, N_tracks, 2] or None if no tracks
        """
        if not self.tracks:
            return None
        centers = torch.stack([t.center for t in self.tracks]).unsqueeze(0)
        return centers
    
    def reset(self) -> None:
        """Clear all tracks (e.g., between evaluation sequences)."""
        self.tracks = []
        self._next_id = 0
```

---

## 6. Loss Functions

All losses live in `training/losses.py`.

```python
class DRISHTILoss(nn.Module):
    """
    Combined multi-task loss for DRISHTI-CORE v2.
    
    Components:
        L_heatmap:  MSE between predicted heatmap and GT Gaussian heatmap.
                    Supervision signal for the MotionCNN to localize targets.
        
        L_cls:      Binary Cross Entropy with Logits between predicted
                    objectness and per-crop binary assignment from GT boxes.
                    Positive: crop whose center is nearest to any GT box center.
                    Negative: all other crops.
        
        L_bbox:     Smooth L1 Loss between predicted box offsets and GT offsets,
                    computed ONLY on positive crops (L_cls label = 1).
                    Using Smooth L1 rather than L1 for robustness to outlier predictions
                    in early training.
        
        L_balance:  MoE auxiliary load-balancing loss (from SparseMoE.forward()).
                    Prevents routing collapse.
    
    Total Loss:
        L = w_h * L_heatmap + w_c * L_cls + w_b * L_bbox + w_m * L_balance
    
    Default weights (from procedure.md):
        w_h = 1.0, w_c = 1.0, w_b = 2.0, w_m = 0.01
    
    Why these weights?
        L_bbox gets 2x because small coordinate errors in tiny targets have
        large IoU impact. The box regression task needs stronger signal.
        L_balance gets 0.01 because it is an auxiliary regularizer — too high
        would force uniform routing even when specialization is beneficial.
    """
    
    def __init__(
        self,
        w_heatmap: float = 1.0,
        w_cls: float = 1.0,
        w_bbox: float = 2.0,
        w_balance: float = 0.01,
    ) -> None:
        ...
    
    def forward(
        self,
        output: PipelineOutput,
        targets: list[dict],            # list of [B] per-frame GT dicts
        heatmap_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        """
        Returns:
            {
                "loss": total weighted loss (scalar, differentiable)
                "heatmap": L_heatmap
                "cls": L_cls
                "bbox": L_bbox
                "balance": L_balance
            }
        """
        ...
```

---

## 7. Activation Functions

| Location | Activation | Reason |
|---|---|---|
| MotionCNN (hidden layers) | ReLU | Standard; inlcuded with BN for training stability |
| MotionCNN (output) | Sigmoid | Forces output to [0,1] — needed for MSE heatmap loss supervision |
| CropEncoder (hidden) | ReLU | Standard; paired with BatchNorm |
| CausalTemporalFusion (FFN) | ReLU (via PyTorch default TransformerEncoderLayer) | Standard transformer practice |
| Expert FFN | GELU | Smoother gradients → better routing diversity than ReLU in MoE |
| DetectionHead (objectness) | None (logit) | BCEWithLogitsLoss absorbs sigmoid numerically stably |
| DetectionHead (box) | Sigmoid | Forces box coordinates to [0,1] — valid normalized space |

---

## 8. Hyperparameter Specification & Justification

| Hyperparameter | Default Value | Ablation Range | Justification |
|---|---|---|---|
| `image_size` | 448×448 | fixed | Anti-UAV standard; large enough for tiny target resolution |
| `temporal_window` | 5 | {3, 5, 7} | 5 frames ≈ 0.2 sec @25fps. Enough for velocity estimation without excessive memory |
| `num_crops` | 8 | {4, 8, 16} | **Must be ablated.** 8 balances coverage and compute. |
| `crop_size` | 64 | {32, 64, 128} | 64 × 64 is 1/7 of image width — large enough for tiny target |
| `scan_period` | 4 | {2, 4, 8, 16} | **Must be ablated.** Every 4th frame = 6.25% compute overhead |
| `border_width_frac` | 0.07 | {0.05, 0.07, 0.10} | 7% ≈ 31px of 448px — sufficient zone for boundary detection |
| `ldmi_scales` | (15, 31) | {(15,), (31,), (15,31), (15,31,63)} | Two scales cover <10px and <30px targets. 63 captures larger objects |
| `num_experts` | 8 | {4, 8, 16} | Matches num_crops — one potential expert specialization per crop source |
| `top_k` | 2 | {1, 2, 4} | Top-2 → 25% active parameters. Top-1 collapses; top-4 reduces savings |
| `encoder_feature_dim` | 256 | fixed | Standard intermediate dimension; sufficient for small dataset scale |
| `temporal_heads` | 4 | {2, 4, 8} | 4 heads for 257-dim = 64-dim per head — efficient |
| `temporal_layers` | 2 | {1, 2, 4} | Shallow is intentional — this is a refinement stage, not a backbone |
| `moe_balance_weight` | 0.01 | {0.001, 0.01, 0.1} | Standard Switch Transformer value |
| `objectness_threshold` | 0.3 | {0.2, 0.3, 0.5} | Tuned on val set; lower → more recall, higher → more precision |
| `tracker_dist_threshold` | 0.15 | {0.05, 0.10, 0.15} | 0.15 = 67px at 448 resolution; large enough for typical UAV velocity |
| `tracker_max_coast` | 15 | {5, 10, 15, 25} | 15 frames = 0.6s @25fps; minimum time behind a building edge |

---

## 9. Staged Training Procedure

### Stage 1: Detector Pre-training
```
Trainable:   MotionCNN, DetectionHead
Frozen:      CropEncoder, TemporalFusion, MoE
Loss:        L_heatmap + L_cls + L_bbox  (no balance loss — MoE frozen)
Optimizer:   AdamW(lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
Scheduler:   CosineAnnealingLR(T_max=80, eta_min=1e-6)
Epochs:      80
Batch Size:  16 clips
Input:       Single-frame triplet (temporal window = 1, no temporal fusion)
Notes:       TemporalFusion receives a single padded sequence of identical features.
             This teaches the MotionCNN and head to detect from pure spatial evidence.
Output:      checkpoints/stage1_best.pt
```

### Stage 2: Temporal Integration
```
Trainable:   CausalTemporalFusion
Frozen:      MotionCNN, CropEncoder, MoE, DetectionHead
Loss:        L_cls + L_bbox  (heatmap supervision no longer needed here)
Optimizer:   AdamW(lr=5e-5, weight_decay=1e-4)
Scheduler:   CosineAnnealingLR(T_max=30, eta_min=1e-7)
Epochs:      30
Batch Size:  8 clips (T=5, higher memory)
Init:        Load checkpoints/stage1_best.pt
Notes:       The stage-1 detector provides stable, high-quality crop encodings.
             The transformer can now learn WHEN a crop is interesting over time.
Output:      checkpoints/stage2_best.pt
```

### Stage 3: MoE Specialization
```
Trainable:   SparseMoE (router + experts)
Frozen:      Everything else
Loss:        L_cls + L_bbox + w_m * L_balance
Optimizer:   AdamW(lr=1e-5, weight_decay=1e-4)
Scheduler:   CosineAnnealingLR(T_max=20, eta_min=1e-8)
Epochs:      20
Batch Size:  8 clips
Init:        Load checkpoints/stage2_best.pt
Notes:       Low LR because we are fine-tuning a single routing module.
             Monitor expert utilization (each expert should receive >5% of tokens).
             If routing collapse occurs, increase w_m.
Output:      checkpoints/stage3_best.pt
```

### Full End-to-End Fine-tuning (Optional)
```
Trainable:   All modules
Loss:        Full L_heatmap + L_cls + L_bbox + L_balance
Optimizer:   AdamW(lr=2e-6, weight_decay=1e-4)
Scheduler:   CosineAnnealingLR(T_max=10, eta_min=1e-9)
Epochs:      10
Notes:       Very low LR to prevent catastrophic forgetting of stage-learned features.
Output:      checkpoints/final_best.pt
```

### Training Logging
At every epoch, log:
```
train_loss, train_heatmap_loss, train_cls_loss, train_bbox_loss, train_balance_loss
val_loss, val_mAP@50, val_mAP@50:95, val_precision, val_recall, val_f1
active_expert_fraction (how many experts received >1% tokens this epoch)
learning_rate
```

---

## 10. Evaluation Protocol

All evaluations use a fixed confidence threshold (tuned on val → deployed on test):

### Detection Metrics
| Metric | Definition | Significance |
|---|---|---|
| **mAP@0.50** | Mean Average Precision at IoU threshold 0.50 | Primary detection metric |
| **mAP@0.50:0.95** | COCO-style mAP averaged over IoU thresholds 0.50–0.95 | Localization quality |
| **Precision@τ** | TP / (TP + FP) at confidence τ=0.3 | Spurious detection rate |
| **Recall@τ** | TP / (TP + FN) at confidence τ=0.3 | Miss rate |
| **F1@τ** | 2 × P × R / (P + R) at confidence τ=0.3 | Balanced accuracy |
| **False Positives per Image** | FP count / total frames | Background false alarm rate |
| **Frame-1 Recall** | Recall on clip-first-frames only | Cold start performance |
| **Edge Entry Recall** | Recall on frames where target first enters the frame boundary | Boundary detection |

### Tracking Metrics
| Metric | Definition |
|---|---|
| **Success Plot AUC** | Area under target overlap (IoU) success curve from 0 to 1 |
| **Precision Plot @20px** | % of frames where predicted center is within 20 pixels of GT center |
| **Occlusion Recovery Rate** | % of occluded targets successfully re-acquired within N=10 frames |
| **Mean Frames-to-Reacquire** | Mean number of frames to re-detect after occlusion end |

### Efficiency Metrics
| Metric | Tool |
|---|---|
| **GFLOPs per frame** | `fvcore.nn.FlopCountAnalysis` |
| **FPS (GPU)** | `torch.cuda.Event` timing |
| **FPS (Edge)** | Measured on Jetson Orin Nano |
| **Active parameters** | Parameters of top-k activated experts only |
| **Total parameters** | `sum(p.numel() for p in model.parameters())` |
| **Peak GPU memory (MB)** | `torch.cuda.max_memory_allocated()` |
| **Energy per frame (mJ)** | Power × (1/FPS), measured via `tegrastats` on Jetson |

### Robustness Metrics
| Metric | Protocol |
|---|---|
| **mAP vs. target pixel area** | Bin GT boxes by pixel area, compute mAP per bin |
| **mAP vs. ego-motion speed** | Segment video by camera velocity (from metadata), compute mAP per segment |

---

## 11. Ablation Study Design

Each ablation changes exactly ONE component. All other settings are default.

### Ablation 1: LDMI Filter (Core Contribution)
| Config | LDMI | Expected mAP@50 | Expected Frame-1 Recall |
|---|---|---|---|
| Baseline (No LDMI) | Off — raw frame concatenation | Lowest | Low |
| Single Scale k=15 | Single scale | Medium | Medium |
| Single Scale k=31 | Single scale | Medium | Medium |
| **Full (k=15, 31)** | **Multi-scale** | **Highest** | **Highest** |

*Expected finding:* LDMI provides the largest single improvement. Multi-scale outperforms single-scale because it handles both small and medium targets.

### Ablation 2: Crop Sources
| Config | Sources Active | Expected mAP@50 | Frame-1 Recall |
|---|---|---|---|
| Motion Only | Motion peaks only | Medium | Very Low |
| Edge Only | Edge crops only | Low | High |
| Grid Only | Grid crops only | Low | Medium |
| Motion + Grid | No edge | Medium | Medium |
| Motion + Edge | No grid | Medium | High |
| **Full (Motion+Edge+Grid+Guided)** | **All** | **Highest** | **Highest** |

*Expected finding:* Each source contributes independently. Edge crops primarily benefit new-target entry recall. Grid crops primarily benefit occlusion recovery. Combined > any single source.

### Ablation 3: Dense vs. Sparse MoE
| Config | MoE Type | GFLOPs | mAP@50 |
|---|---|---|---|
| Dense FFN | Single FFN (no routing) | Highest | ~Same |
| MoE Top-1 | top_k = 1 | Lowest | Lower |
| **MoE Top-2** | **top_k = 2** | **Medium** | **~Dense** |
| MoE Top-4 | top_k = 4 | High | ~Dense |

*Expected finding:* Top-2 MoE matches dense accuracy at significantly lower GFLOPs.

### Ablation 4: Training Stages
| Config | Training Strategy | mAP@50 |
|---|---|---|
| End-to-End (all modules jointly) | Single training run | Lower |
| Stage 1 Only (no temporal, no MoE) | Only detector | Baseline |
| Stage 1+2 (detector + temporal) | Two stages | Better |
| **Stage 1+2+3 (full staged)** | **Three stages** | **Highest** |

*Expected finding:* Staged training outperforms end-to-end because each stage prevents gradient interference between components.

### Hyperparameter Sweep: `num_crops` and `scan_period`
Run a grid sweep to justify default values:

| `num_crops` | `scan_period` | mAP@50 | Occlusion Recovery | GFLOPs |
|---|---|---|---|---|
| 4 | 4 | ? | ? | Lowest |
| 8 | 2 | ? | ? | Higher |
| **8** | **4** | **?** | **?** | **Medium** |
| 8 | 8 | ? | ? | Lower |
| 8 | 16 | ? | ? | Lowest |
| 16 | 4 | ? | ? | Highest |

*Selection criterion:* Best mAP@50 × Occlusion Recovery product at lowest GFLOPs.

---

## 12. Baseline Comparisons

### Required Baselines (for publication)

| Baseline | Source | Reason |
|---|---|---|
| **Anti-UAV v1/v2 Paper baseline** | CVPR Anti-UAV paper | Direct comparison on the benchmark we use |
| **YOLOv8-S** | Ultralytics | Standard tiny object detection SOTA |
| **RT-DETR-S** | PaddleDetection | Real-time transformer-based detector |
| **SORT + YOLOv8** | SORT paper | Classic tracking baseline |
| **ByteTrack + YOLOv8** | ByteTrack paper | Strong tracking baseline |
| **DRISHTI-CORE v2 (ours)** | This work | Full proposed method |

For each baseline:
- Train on the same Anti-UAV train split
- Evaluate on the same Anti-UAV val split with the same evaluation protocol
- Report all metrics from Section 10

---

## 13. Failure Case Analysis

After the main evaluation run, perform a structured failure analysis:

### Failure Mode 1: Dense Moving Backgrounds (e.g., trees in wind)
- **Protocol:** Filter test clips where background motion frequency > threshold.
- **Metric:** mAP@50 on this subset vs. full set.
- **Expected finding:** LDMI may introduce false positives if tree-leaf motion is locally non-uniform. Document the frequency.

### Failure Mode 2: Target Smaller Than LDMI Scale
- **Protocol:** Filter clips where GT box area < 25 px².
- **Metric:** Recall on this subset.
- **Fix:** Add smaller LDMI scale (k=7) for sub-pixel target detection.

### Failure Mode 3: Long Occlusion (> 15 frames)
- **Protocol:** Filter clips with occlusions lasting > `max_coast` frames.
- **Metric:** Track resumption rate on this subset.
- **Expected finding:** Tracker prunes the track before re-emergence. This is a known limitation.

### Failure Mode 4: Swarm Scenarios (> 4 Simultaneous Targets)
- **Protocol:** Manually identify multi-target sequences in the Anti-UAV dataset.
- **Metric:** Multi-target detection rate and track accuracy.
- **Expected finding:** Crop budget is the bottleneck. With 8 crops and 4+ targets, each target receives ≤1 crop.

---

## 14. Compute Budget Plan

| Task | Estimated GPU Hours (A100) | Notes |
|---|---|---|
| Stage 1: Detector (80 epochs) | ~12 hours | Single GPU, batch=16 |
| Stage 2: Temporal (30 epochs) | ~6 hours | Single GPU, batch=8 |
| Stage 3: MoE (20 epochs) | ~3 hours | Single GPU, batch=8 |
| E2E Fine-tuning (10 epochs) | ~2 hours | Optional |
| Baseline: YOLOv8-S | ~4 hours | Standard training |
| Baseline: RT-DETR-S | ~8 hours | Standard training |
| Ablation: LDMI (4 configs) | ~4 × 6 = 24 hours | Each = Stage 1 only |
| Ablation: Crop sources (5 configs) | ~5 × 21 = 105 hours | Full staged training |
| Ablation: MoE type (3 configs) | ~3 × 5 = 15 hours | Stage 3 only |
| Ablation: Training stages (4 configs) | ~4 × 21 = 84 hours | Full pipeline per config |
| Hyperparameter sweep (6 configs) | ~6 × 21 = 126 hours | Full pipeline per config |
| **Total (estimated)** | **~289 hours** | **~12 A100 days** |

> **Budget Note:** The ablation sweeps dominate. Prioritize: LDMI ablation (24h) + Crop source ablation (105h) as the core contributions. MoE and staging ablations are secondary.

---

## 15. Reproducibility Checklist

```yaml
Reproducibility:
  - Set seeds: torch.manual_seed(42), numpy.random.seed(42), random.seed(42)
  - Enable deterministic: torch.backends.cudnn.deterministic = True
  - Disable benchmark: torch.backends.cudnn.benchmark = False
  - Log full config: Dump DRISHTIConfig to YAML at experiment start
  - Log git hash: Record git commit hash in experiment directory
  - Save best checkpoint: by val mAP@50, saved every epoch
  - Save last checkpoint: always, for resuming
  - Log hardware: GPU model, CUDA version, driver version, RAM
  - Dataset fingerprint: MD5 hash of train.txt and val.txt file lists
  - Pin DataLoader workers: num_workers=4, pin_memory=True
```
