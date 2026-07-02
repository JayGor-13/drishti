# DRISHTI-CORE v2: Causal Motion-Guided Sparse MoE for Tiny UAV Detection
## Comprehensive Technical Architecture & Engineering Specification

This document provides a detailed, mathematical, and layer-by-layer engineering specification of the **DRISHTI-CORE v2** video target detector. It is designed to serve as a comprehensive blueprint for model implementation, verification, and academic presentation.

---

## 1. Academic Aim & Engineering Objectives

### 1.1 Academic Aim
To resolve the fundamental detection-efficiency tradeoff for tiny object detection in video feeds from moving sensors, specifically addressing:
1. **Motion Signal Ambiguity:** Decoupling camera-induced background motion from target motion without the $O(N^2)$ computational complexity of dense optical flow.
2. **Zero-Velocity Blindness:** Detecting targets that are stationary relative to the frame (e.g., when the camera tracks the target).
3. **Temporal Causality:** Eliminating look-ahead bias to ensure latency-free, frame-by-frame streaming inference.
4. **Spatial Occlusions:** Re-acquiring targets that temporarily disappear behind structural occlusions (e.g., buildings, foliage) and emerge along unpredicted trajectories.

### 1.2 Quantitative Objectives
* **Causal Latency:** Zero-frame future look-ahead. Inference on frame $f_t$ depends only on $\{f_{t-k}\}_{k=0}^4$.
* **Edge Throughput:** Achieve $\ge 30$ FPS on a Jetson Orin Nano (15W power budget).
* **Compute Reduction:** Reduce active parameter execution and overall GFLOPs by $\ge 60\%$ relative to dense baselines.

---

## 2. Mathematical Formulations & Physical Invariants

### 2.1 The Local Spatial Invariant (LDMI)
Let $f_t(x, y)$ be the intensity of pixel $(x,y)$ at time step $t$. Under perspective projection, camera movement with angular velocity $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ generates a rotational flow field $\mathbf{v}^{rot}$:
$$\mathbf{v}^{rot}(x,y) = \begin{bmatrix} -f\omega_y + y\omega_z - \frac{xy}{f}\omega_x + \frac{x^2}{f}\omega_y \\ f\omega_x - x\omega_z - \frac{y^2}{f}\omega_x + \frac{xy}{f}\omega_y \end{bmatrix}$$
For a tiny target at distance $D$ much larger than the camera focal length $f$, the spatial footprint of the target is small, covering a patch $\mathcal{N}$ of size $K \times K$ pixels. For any pixel $(x', y') \in \mathcal{N}$ centered around $(x_0, y_0)$, the coordinates satisfy:
$$x' = x_0 + \delta x, \quad y' = y_0 + \delta y \quad (\delta x, \delta y \ll f)$$
Substituting these into the flow equation reveals that the spatial derivatives of the camera-induced flow are bounded by $O(1/f)$:
$$\mathbf{v}^{rot}(x_0 + \delta x, y_0 + \delta y) = \mathbf{v}^{rot}(x_0, y_0) + \mathbf{J}(x_0, y_0) \begin{bmatrix} \delta x \\ \delta y \end{bmatrix}$$
where the Jacobian $\mathbf{J}$ elements are scaled by $1/f$. Thus, for typical focal lengths, the background flow field is **locally uniform** (approximated as a constant local translation vector):
$$\mathbf{v}^{rot}(x', y') \approx \mathbf{v}^{rot}(x_0, y_0)$$
Let $\mathbf{v}(x,y)$ be the observed pixel difference vector $d_t(x,y) = f_t(x,y) - f_{t-1}(x,y)$. The local differential residual $r_t(x,y)$ is formulated as:
$$r_t(x,y) = d_t(x,y) - \frac{1}{|\mathcal{N}(x,y)|} \sum_{(x',y') \in \mathcal{N}(x,y)} d_t(x',y')$$
* **For Background Pixels:** Since $d_t(x',y') \approx \mathbf{v}^{rot}(x_0, y_0)$ for all pixels in $\mathcal{N}$, the subtraction yields:
  $$r_t(x,y) \approx \mathbf{v}^{rot}(x_0, y_0) - \mathbf{v}^{rot}(x_0, y_0) \approx 0$$
* **For Target Pixels:** The target violates the local uniformity assumption. Its local residual becomes:
  $$r_t(x_T, y_T) \approx \mathbf{v}^{target}(x_T, y_T) - \mathbf{v}^{rot}(x_0, y_0) \neq 0$$
This non-parametric filter suppresses uniform background motion, rendering the residual map camera-motion-invariant.

---

### 2.2 Boundary Coordinate Parameterization (Edge Attention)
To continuously monitor targets entering the frame, we define four border zones along the edges of the normalized coordinate plane $[0, 1]^2$. Let $W_{border} \in (0, 0.5)$ be the normalized margin width (default $0.07$, representing $7\%$ of the frame width).

```
   ┌────────────────────────────────────────────────────────┐
   │                       TOP ZONE                         │
   ├──────┬──────────────────────────────────────────┬──────┤
   │      │                                          │      │
   │ LEFT │                                          │RIGHT │
   │ ZONE │                 INTERIOR                 │ ZONE │
   │      │                                          │      │
   ├──────┴──────────────────────────────────────────┴──────┤
   │                      BOTTOM ZONE                       │
   └────────────────────────────────────────────────────────┘
```

The centers for crop extraction are sampled from the edge midpoints dynamically:
* **Horizontal Edges:**
  $$\mathbf{p}_{left} = \left[ \frac{W_{border}}{2},\ 0.5 \right], \quad \mathbf{p}_{right} = \left[ 1 - \frac{W_{border}}{2},\ 0.5 \right]$$
* **Vertical Edges:**
  $$\mathbf{p}_{top} = \left[ 0.5,\ \frac{W_{border}}{2} \right], \quad \mathbf{p}_{bottom} = \left[ 0.5,\ 1 - \frac{W_{border}}{2} \right]$$

---

### 2.3 Periodic Interior Grid Sweep
During tracking, the target can disappear behind structural occlusions and change direction. To prevent permanent lock loss without wasting GPU memory, we define a periodic spatial grid search.
Let $N_{scan} \in \mathbb{Z}^+$ be the global scan period (default $N_{scan} = 4$). At every frame index $t$ where $t \pmod {N_{scan}} = 0$, we allocate $4$ crops to tile the inner $60\%$ of the screen:
$$\mathbf{p}_{grid} = \Big\{ [0.3, 0.3], [0.3, 0.7], [0.7, 0.3], [0.7, 0.7] \Big\}$$
This ensures that any drone emerging from behind central buildings/trees is captured within 4 frames, regardless of its path.

---

## 3. Layer-by-Layer Module Specifications

### Module 1: Causal Triplet Extractor
This module constructs the temporal input tensor.
* **Input Shape:** Video buffer tensor $B \times T_{buffer} \times C \times H \times W$ (typically $B \times 5 \times 3 \times 448 \times 448$).
* **Output Shape:** Causal triplet tensor $[B, 9, H, W]$.
* **Operation:** Extracts three frames relative to the present index $t$: $[f_{t-2}, f_{t-1}, f_t]$ and concatenates them along the channel dimension.
  $$T_t = \text{Concat}\big(f_{t-2},\ f_{t-1},\ f_t;\ \text{dim}=1\big)$$

---

### Module 2: Local Differential Motion Invariant (LDMI) Preprocessing
This module performs background motion subtraction.
* **Input Shape:** Causal triplet tensor $T_t \in \mathbb{R}^{B \times 9 \times H \times W}$ (where channels $0:3 = f_{t-2}$, $3:6 = f_{t-1}$, $6:9 = f_t$).
* **Output Shape:** Filtered triplet tensor $p_t \in \mathbb{R}^{B \times 9 \times H \times W}$.
* **Layer Details:**
  1. **Differences:** Compute $d_{t-1} = f_{t-1} - f_{t-2}$ and $d_t = f_t - f_{t-1}$ (shape: $B \times 3 \times H \times W$).
  2. **Multi-Scale Average Pooling:** Pass $d_{t-1}$ and $d_t$ through parallel 2D average pooling layers with padding to maintain spatial resolution:
     $$\bar{d}_{t, 15} = \text{AvgPool2d}(d_t, \text{kernel}=15, \text{stride}=1, \text{padding}=7)$$
     $$\bar{d}_{t, 31} = \text{AvgPool2d}(d_t, \text{kernel}=31, \text{stride}=1, \text{padding}=15)$$
  3. **Residual Fusion:**
     $$r_t = \max(|d_t - \bar{d}_{t, 15}|,\ |d_t - \bar{d}_{t, 31}|) \in \mathbb{R}^{B \times 3 \times H \times W}$$
  4. **Tri-Channel Output:** Concatenate $r_{t-1}$, $f_t$ (raw appearance), and $r_t$:
     $$p_t = \text{Concat}(r_{t-1},\ f_t,\ r_t;\ \text{dim}=1)$$

---

### Module 3: MotionCNN
A convolutional layer block that processes the filtered triplet to localize anomalies.
* **Input Shape:** $p_t \in \mathbb{R}^{B \times 9 \times H \times W}$.
* **Output Shape:** Heatmap $M_t \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}$.
* **Layer Configurations:**
  
  | Layer Index | Operation | Input Channels | Output Channels | Kernel Size | Stride | Padding | Activation |
  |---|---|---|---|---|---|---|---|
  | L3.1 | Conv2d | 9 | 32 | 3 | 2 | 1 | BatchNorm + ReLU |
  | L3.2 | Conv2d | 32 | 64 | 3 | 2 | 1 | BatchNorm + ReLU |
  | L3.3 | Conv2d | 64 | 64 | 3 | 1 | 1 | BatchNorm + ReLU |
  | L3.4 | Conv2d | 64 | 1 | 1 | 1 | 0 | Sigmoid |

---

### Module 4: Multi-Source Crop Attention Proposal
* **Input:** Heatmap $M_t \in \mathbb{R}^{B \times 1 \times H_h \times W_h}$, active tracking predictions, border definitions.
* **Output:** Crop coordinates tensor $[B, 8, 2]$ (normalized $[x, y]$ centers) and extracted crops tensor $[B \cdot 8, 3, 64, 64]$.
* **Coordinate Extraction Logic:**
  1. **Motion Proposals ($K_{motion}$):** Apply a local maximum filter to $M_t$ (e.g., via `MaxPool2d` of kernel size 3, stride 1, padding 1). Coords where the output matches the input are extracted. The top peaks with scores $> 0.1$ are retained.
  2. **Edge Proposals ($K_{edge}$):** Pre-parameterized border midpoints.
  3. **Grid Proposals ($K_{grid}$):** Fixed coordinates tiling the inner $60\%$ of the screen.
  4. **Guided Proposals ($K_{guided}$):** Target coordinates predicted by the state tracker.
* **Extraction:** For each center, crop a $64 \times 64$ patch from the current frame $f_t$ using replicate padding if coordinates exceed boundaries.

---

### Module 5: Frozen Crop Encoder
* **Input Shape:** $\text{crops} \in \mathbb{R}^{B \cdot 8 \times 3 \times 64 \times 64}$.
* **Output Shape:** Crop features tensor $[B, 8, 256]$.
* **Structure:** A lightweight 3-stage CNN frozen during detector training:
  - `Conv2d(3 -> 64, k=3, s=1, p=1)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU`
  - `Conv2d(64 -> 128, k=3, s=2, p=1)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` (downsamples to 32x32)
  - `Conv2d(128 -> 256, k=3, s=2, p=1)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` (downsamples to 16x16)
  - `AdaptiveAvgPool2d(1)` $\rightarrow$ Flatten $\rightarrow$ `Linear(256 -> 256)`

---

### Module 6: Causal Temporal Fusion Transformer
Fuses temporal context across past crop features.
* **Input Shape:** Sequence of crops over time: $[B, 5, 8, 257]$ (where 257 includes the concatenated heatmap score of each crop).
* **Output Shape:** Fused features tensor $[B, 8, 256]$.
* **Structure:** 
  1. Reshape features to $[B \cdot 8, 5, 257]$ (treating crops independently over the temporal dimension).
  2. Add learnable 1D temporal positional embeddings.
  3. Process through 2 Transformer blocks:
     - Multi-Head Self-Attention (`embed_dim=257`, `nhead=4`, `dropout=0.1`).
     - Feed-Forward Block (`linear_dim=512`, `dropout=0.1`, `LayerNorm`).
  4. Extract the present token (index $t$, the last token in sequence) and project:
     $$\text{fused} = \text{Linear}_{257 \rightarrow 256}(\text{tokens}[:, -1]) \in \mathbb{R}^{B \cdot 8 \times 256}$$
  5. Reshape back to $[B, 8, 256]$.

---

### Module 7: Sparse Mixture-of-Experts (MoE)
Performs sparse computation on the fused features.
* **Input Shape:** Fused features tensor $[B \cdot 8, 256]$.
* **Output Shape:** Routed features tensor $[B \cdot 8, 256]$.
* **Structure:**
  1. **Router:** A linear layer $W_{router} \in \mathbb{R}^{256 \times 8}$. Compute routing probabilities:
     $$P_i = \text{softmax}(W_{router} \cdot x_i) \in \mathbb{R}^8$$
  2. **Top-2 Selection:** Identify the top 2 indices and their weights:
     $$G_i = \text{top2}(P_i)$$
     $$w_i = \frac{G_i}{\sum G_i}$$
  3. **Conditional Execution:** Route each token $x_i$ to its selected experts. Each expert FFN is:
     $$\text{Expert}_e(x) = \text{Linear}_{512 \rightarrow 256}(\text{ReLU}(\text{Linear}_{256 \rightarrow 512}(x)))$$
  4. **Recombination:** Multiply expert outputs by their routing weights and sum:
     $$y_i = \sum_{e \in G_i} w_{i, e} \cdot \text{Expert}_e(x_i)$$

---

### Module 8: Detection Head
* **Input Shape:** Routed features tensor $[B, 8, 256]$.
* **Output Shapes:** Logits ($B \times 8 \times 1$), boxes ($B \times 8 \times 4$).
* **Structure:**
  - **Objectness:** `LayerNorm` $\rightarrow$ `Linear(256 -> 1)`.
  - **Box Regression:** `LayerNorm` $\rightarrow$ `Linear(256 -> 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(256 -> 4)` $\rightarrow$ `Sigmoid`. Returns $[cx, cy, w, h]$ offsets relative to the crop window.

---

### Module 9: Simple Tracker (Inference-Only)
* **Input:** Boxes and logits output from Module 8.
* **Output:** Track Table updates and guided centers for the next frame.
* **Algorithm:** Constant-velocity prediction combined with Euclidean distance assignment.

---

## 4. Dynamic Scheduling & Crop Allocation Algorithm

The following pseudocode details how the crop coordinate budget is dynamically scheduled to cover tracking, edge scanning, and periodic global scans:

```python
def allocate_crop_budget(frame_index, active_tracks, heatmap_peaks, config):
    """
    Args:
        frame_index (int): Sequential index of current frame t
        active_tracks (list[Track]): Table of current confirmed targets
        heatmap_peaks (Tensor): Peaks extracted from MotionCNN [N, 2]
        config (Config): Model configuration parameters
    Returns:
        crops (list[tuple[float, float]]): List of exactly 8 normalized coordinate centers
    """
    crop_list = []
    MAX_CROPS = 8
    
    # Step 1: Guided Crop Allocation (Prioritize active tracks)
    num_tracks = len(active_tracks)
    if num_tracks > 0:
        # Give each track up to 2 crops (center + velocity look-ahead)
        crops_per_track = min(2, (MAX_CROPS - 2) // num_tracks)
        for track in active_tracks:
            # Crop 1: Center prediction
            crop_list.append(track.predicted_center)
            # Crop 2: Look-ahead (pos + vel * dt)
            if crops_per_track > 1:
                look_ahead = track.predicted_center + track.velocity
                crop_list.append(clip_coordinates(look_ahead))
                
    K_guided = len(crop_list)
    
    # Step 2: Determine if frame is a Global Scan Frame
    is_global_scan = (frame_index % config.scan_period == 0)
    
    # Step 3: Allocate Grid Proposals (Periodic interior scan)
    K_grid = 0
    if is_global_scan:
        # Tile 4 central region centers
        grid_centers = [(0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)]
        # Add as many as budget allows
        available_slots = MAX_CROPS - len(crop_list)
        grid_slots = min(4, available_slots)
        crop_list.extend(grid_centers[:grid_slots])
        K_grid = grid_slots

    # Step 4: Allocate Edge proposals (Continuous entry check)
    # Odd frames check Left/Right, Even frames check Top/Bottom
    edge_centers = []
    if frame_index % 2 == 1:
        edge_centers = [(config.border_width / 2.0, 0.5), (1.0 - config.border_width / 2.0, 0.5)]
    else:
        edge_centers = [(0.5, config.border_width / 2.0), (0.5, 1.0 - config.border_width / 2.0)]
        
    available_slots = MAX_CROPS - len(crop_list)
    edge_slots = min(len(edge_centers), available_slots)
    crop_list.extend(edge_centers[:edge_slots])
    K_edge = edge_slots
    
    # Step 5: Allocate Motion Proposals (Fill remaining slots with heatmap peaks)
    available_slots = MAX_CROPS - len(crop_list)
    if available_slots > 0:
        peaks_to_add = min(len(heatmap_peaks), available_slots)
        crop_list.extend(heatmap_peaks[:peaks_to_add])
        
    # Step 6: Fallback (Pad with center frame if budget not met)
    while len(crop_list) < MAX_CROPS:
        crop_list.append((0.5, 0.5))
        
    return crop_list[:MAX_CROPS]
```

---

## 5. Multi-Target Tracker & Distance Gating

The tracker maintains target tracks across frames during inference using Euclidean distance matching.

```python
class Track:
    def __init__(self, track_id, box, confidence):
        self.id = track_id
        self.center = box[:2]      # [cx, cy]
        self.size = box[2:]        # [w, h]
        self.velocity = torch.zeros(2)
        self.confidence = confidence
        self.coast_count = 0
        self.hit_count = 1
        
    def predict(self):
        """Constant-velocity state projection."""
        self.center = self.center + self.velocity
        
    def update(self, box, confidence):
        """State measurement update."""
        new_center = box[:2]
        # Calculate velocity: displacement since last frame
        self.velocity = new_center - self.center
        self.center = new_center
        self.size = box[2:]
        self.confidence = confidence
        self.coast_count = 0
        self.hit_count += 1

class SimpleTracker:
    def __init__(self, dist_threshold=0.15, max_coast=15):
        self.tracks = []
        self.next_id = 0
        self.dist_threshold = dist_threshold
        self.max_coast = max_coast

    def update_tracks(self, detections, confidences):
        """
        Args:
            detections (Tensor): Bounding boxes [N_det, 4]
            confidences (Tensor): Confidence scores [N_det]
        """
        # Step 1: Predict positions for existing tracks
        for track in self.tracks:
            track.predict()
            
        # Step 2: Distance Matrix Calculation (Euclidean distance on centers)
        matched_detections = set()
        matched_tracks = set()
        
        if len(self.tracks) > 0 and len(detections) > 0:
            for track_idx, track in enumerate(self.tracks):
                best_dist = float('inf')
                best_det_idx = -1
                for det_idx, det in enumerate(detections):
                    if det_idx in matched_detections:
                        continue
                    dist = torch.norm(track.center - det[:2])
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = det_idx
                
                # Step 3: Distance Gating
                if best_dist < self.dist_threshold:
                    track.update(detections[best_det_idx], confidences[best_det_idx])
                    matched_detections.add(best_det_idx)
                    matched_tracks.add(track_idx)
                    
        # Step 4: Handle Unmatched Tracks (Coast or Prune)
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track.coast_count += 1
                
        # Remove tracks that coasted too long
        self.tracks = [t for t in self.tracks if t.coast_count < self.max_coast]
        
        # Step 5: Handle Unmatched Detections (Birth new tracks)
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections and confidences[det_idx] > 0.3:
                new_track = Track(self.next_id, det, confidences[det_idx])
                self.tracks.append(new_track)
                self.next_id += 1
```

---

## 6. Detailed Training Procedures

The model is trained in a staged sequence to avoid gradient interference and MoE routing collapse.

```
       [ Stage 1: Detector Training ]
       - Trainable: MotionCNN + Detection Head
       - Frozen: Crop Encoder + Temporal Fusion + MoE
       - Loss: Heatmap MSE + BCE Objectness + SmoothL1 Bounding Box
       - Output Checkpoint: detector_best.pt
                     │
                     ▼
       [ Stage 2: Temporal Training ]
       - Load: detector_best.pt
       - Trainable: Temporal Fusion Transformer
       - Frozen: MotionCNN + Crop Encoder + MoE + Detection Head
       - Loss: BCE Objectness + SmoothL1 Bounding Box
       - Output Checkpoint: temporal_best.pt
                     │
                     ▼
       [ Stage 3: MoE Training ]
       - Load: temporal_best.pt
       - Trainable: MoE Router + Experts
       - Frozen: MotionCNN + Crop Encoder + Temporal Fusion + Detection Head
       - Loss: Detector Loss + Aux Load-Balancing Loss
       - Output Checkpoint: moe_best.pt
```

### Staged Training Settings
1. **Detector Stage:** AdamW optimizer with $LR = 1\times 10^{-4}$, weight decay $1\times 10^{-4}$, trained for 80 epochs.
2. **Temporal Stage:** AdamW optimizer with $LR = 5\times 10^{-5}$, trained for 30 epochs.
3. **MoE Stage:** AdamW optimizer with $LR = 1\times 10^{-5}$, trained for 20 epochs. The auxiliary load balancing loss weight is set to $w_{moe} = 0.01$.

---

## 7. Complete Tensor Flow Specification

The following table traces the tensor shapes through the entire v2 architecture:

| Block / Stage | Input Tensor Shape | Operation | Output Tensor Shape |
|---|---|---|---|
| **Causal Triplet Extractor** | $B \times T_{buffer} \times 3 \times 448 \times 448$ | Window Index Extraction | $B \times 9 \times 448 \times 448$ |
| **Local Differential Preproc (LDMI)** | $B \times 9 \times 448 \times 448$ | Frame Difference + AvgPool | $B \times 9 \times 448 \times 448$ |
| **MotionCNN** | $B \times 9 \times 448 \times 448$ | 2D Convolutions + Sigmoid | $B \times 1 \times 112 \times 112$ |
| **Multi-Source Crop Attention** | $B \times 1 \times 112 \times 112$ | Coordinate Extraction & Cropping | $B \cdot 8 \times 3 \times 64 \times 64$ |
| **Frozen Crop Encoder** | $B \cdot 8 \times 3 \times 64 \times 64$ | Frozen CNN Feature Mapping | $B \times 8 \times 256$ |
| **Score Concatenation** | $B \times 8 \times 256$ | Concatenate Motion Value | $B \times 8 \times 257$ |
| **Causal Temporal Fusion** | $B \times 5 \times 8 \times 257$ | Sequence Self-Attention | $B \times 8 \times 256$ |
| **Sparse Mixture-of-Experts** | $B \times 8 \times 256$ | Softmax Routing + Expert FFNs | $B \times 8 \times 256$ |
| **Detection Head** | $B \times 8 \times 256$ | Linear Box & Logit Heads | Logits: $B \times 8 \times 1$<br>Boxes: $B \times 8 \times 4$ |
