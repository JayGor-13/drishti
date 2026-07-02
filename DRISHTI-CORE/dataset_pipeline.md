# DRISHTI-CORE v2 — Dataset Pipeline Documentation
**File:** `docs/dataset.md`  
**Version:** 2.0.0  
**Scope:** Everything from raw dataset access → frame extraction → preprocessing → model input

This document is the single source of truth for the data pipeline. Any engineer implementing the v2 system from scratch should be able to reproduce the exact data flow described here.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Raw Dataset Structure](#2-raw-dataset-structure)
3. [Annotation Format](#3-annotation-format)
4. [Frame Extraction Pipeline](#4-frame-extraction-pipeline)
5. [Extracted Frame Directory Layout](#5-extracted-frame-directory-layout)
6. [Dataset Classes — Detailed Reference](#6-dataset-classes--detailed-reference)
   - 6.1 AntiUAVRGBTVideoDataset (Video Mode)
   - 6.2 AntiUAVExtractedFrameDataset (Pre-extracted Mode)
   - 6.3 ModelScopeAntiUAVCocoDataset (COCO Mode)
   - 6.4 SyntheticAntiUAVDataset (Smoke Test Mode)
7. [Temporal Window Construction](#7-temporal-window-construction)
8. [Frame Decoding & Resizing](#8-frame-decoding--resizing)
9. [Bounding Box Format Conventions](#9-bounding-box-format-conventions)
10. [Target Construction](#10-target-construction)
11. [Collator — Batching Strategy](#11-collator--batching-strategy)
12. [DataLoader Configuration](#12-dataloader-configuration)
13. [Dataset Selection Logic in experiment.py](#13-dataset-selection-logic-in-experimentpy)
14. [Configuration Parameters Reference](#14-configuration-parameters-reference)
15. [Adding a New Dataset](#15-adding-a-new-dataset)

---

## 1. Dataset Overview

### 1.1 Primary Dataset: Anti-UAV RGBT
The Anti-UAV challenge dataset (CVPR 2020+) is the primary training and evaluation benchmark. It consists of:
- **410 sequences** of real-world RGB (visible) and IR (infrared) UAV videos captured from ground-facing cameras.
- Each sequence is an `.mp4` video with a paired `.json` annotation file.
- Modalities: `visible` (RGB) and `infrared` (thermal IR).
- Target: Single drone per sequence (the challenge is single-target tracking).

**Official source:** `https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV`

### 1.2 Secondary Dataset: COCO-Format (ModelScope)
The ModelScope platform also provides a COCO-format version where all frames are stored as individual images with annotations in the standard COCO JSON schema. This mode is supported for flexibility but requires `torchvision`.

### 1.3 Smoke Test / Development Mode: SyntheticAntiUAVDataset
A fully synthetic, dependency-free dataset for unit testing and rapid pipeline verification. It generates random background frames with a bright, moving rectangular target and valid bounding-box annotations. No external data required.

---

## 2. Raw Dataset Structure

After downloading and extracting the Anti-UAV dataset, the expected directory layout is:

```
<data_root>/
├── train/
│   ├── 20190925_111759_1_8/
│   │   ├── infrared.mp4
│   │   ├── infrared.json
│   │   ├── visible.mp4
│   │   └── visible.json
│   ├── 20190925_112035_1_8/
│   │   ├── infrared.mp4
│   │   ├── infrared.json
│   │   ├── visible.mp4
│   │   └── visible.json
│   └── ...
├── val/
│   └── ...                      # same structure as train/
└── test/
    └── ...                      # same structure as train/
```

**Key rules about the raw structure:**
- Each `<sequence_name>/` directory is a distinct video sequence.
- The `.mp4` file holds the raw video frames.
- The `.json` file is the annotation for that sequence (one annotation per frame).
- Both modalities (`infrared` and `visible`) live **in the same sequence folder**, using their modality name as a prefix.

### 2.1 Alternative Annotation Locations
Some Anti-UAV dataset versions store annotations separately under a `label_new/` subfolder. The loader handles this via `_annotation_candidates()`, which searches multiple possible paths in order:

```
Lookup order for annotation file:
  1. <sequence_dir>/<modality>.json                            (e.g., visible.json)
  2. <data_root>/label_new/<split>/<name>/<modality>.json
  3. <data_root>/label_new/<name>/<modality>.json
  4. <data_root>/label_new/<split>/<name>_<modality>.json
  5. <data_root>/label_new/<split>/<modality>_<name>.json
  6. <data_root>/label_new/<split>/<name>.json
  7. <data_root>/label_new/<name>_<modality>.json
  8. <data_root>/label_new/<modality>_<name>.json
  9. <data_root>/label_new/<name>.json
```

The loader uses the first path that exists. This makes it compatible with multiple official dataset release versions.

---

## 3. Annotation Format

Each `.json` annotation file is read by `_read_antiuav_json()` (in `train/antiuav.py`). The function supports two top-level JSON structures:

### 3.1 List Format (Older Releases)
```json
[
  [20, 30, 50, 60],
  [22, 32, 50, 60],
  [],
  ...
]
```
Each element is a frame-level bounding box in `[x, y, w, h]` format. An empty list `[]` means the target is not visible in that frame.

### 3.2 Dictionary Format (Newer Releases)
```json
{
  "gt_rect": [[20, 30, 50, 60], [22, 32, 50, 60], [], ...],
  "exist":   [1, 1, 0, ...]
}
```
Supported key aliases for bounding boxes: `gt_rect`, `gt_bbox`, `bbox`, `bboxes`, `boxes`  
Supported key aliases for existence: `exist`, `exists`, `target_visible`, `presence`

### 3.3 Annotation Parsing Rules

```python
def _read_antiuav_json(path: Path) -> tuple[tuple[Any, ...], tuple[bool, ...]]:
    """
    Returns:
        boxes:  tuple of per-frame raw box values (list of 4 floats, or empty list)
        exists: tuple of per-frame boolean visibility flags
    
    Existence fallback:
        If the JSON has no 'exist' key, existence is inferred from the box itself:
        a box is considered "present" if its width AND height are both > 0.
    """
```

**Parsing edge cases handled:**
- `exists` list shorter than `boxes` list: tail is filled by checking box area.
- `boxes` list shorter than `exists` list: tail boxes filled with empty lists.
- Non-boolean values in `exist` (e.g., `0`, `"false"`, `"null"`): converted to `bool`.
- Multiple candidate boxes per frame: handled by `_candidate_boxes()` which detects nested lists.
- Malformed boxes (non-numeric, length < 4, zero area): silently skipped.

---

## 4. Frame Extraction Pipeline

Reading frames directly from `.mp4` files on every training step is slow (random seeks are expensive). The recommended workflow is to **pre-extract all frames to individual JPEG files** once, and then use `AntiUAVExtractedFrameDataset` for training.

### 4.1 Extraction Script: `train/extract_frames.py`

This script is invoked as a module. It discovers all `<modality>.mp4` files under `data_root`, extracts them frame-by-frame via OpenCV, and writes the frames to `output_root`.

**Full command:**
```bash
python -m train.extract_frames \
  --data-root   /path/to/antiuav_raw \
  --output-root /path/to/antiuav_frames \
  --splits      train val test \
  --modalities  visible \
  --workers     4 \
  --image-ext   .jpg \
  --jpeg-quality 95
```

**CLI arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-root` | `Path` | required | Root directory of the raw Anti-UAV dataset |
| `--output-root` | `Path` | required | Root directory where frames will be written |
| `--splits` | `list[str]` | `["train", "val", "test"]` | Splits to extract |
| `--modalities` | `list[str]` | `["visible"]` | One or both of `visible`, `infrared` |
| `--workers` | `int` | `4` | Number of parallel threads |
| `--image-ext` | `str` | `.jpg` | Frame file extension: `.jpg` or `.png` |
| `--jpeg-quality` | `int` | `95` | JPEG quality when `--image-ext .jpg` |
| `--overwrite` | `flag` | `False` | Re-extract even if frames already exist |

### 4.2 Extraction Algorithm (`extract_task`)

For each sequence/modality pair:

```
1. Check if already extracted (skip if output dir has any .jpg files and --overwrite not set)
2. Open video with cv2.VideoCapture
3. Read frames sequentially (cap.read() in a while loop)
4. Write each frame as: <output_frame_dir>/<frame_count:06d><image_ext>
   e.g., output/train/20190925_111759_1_8/visible/000000.jpg
5. Copy annotation JSON alongside the frame directory:
   output/train/20190925_111759_1_8/visible.json
6. Release the VideoCapture handle
```

**Why copy the annotation?**
The extracted-frame dataset needs to find the annotation file next to the frame directory. Copying it avoids maintaining two separate paths during training.

**Why ThreadPoolExecutor?**
Video decoding is mostly I/O-bound (disk reads). Multiple threads can decode different sequences concurrently without GIL contention. Use `--workers 4` on a fast NVMe SSD; reduce to `--workers 1` on slow HDDs.

**Frame naming convention:**
Frames are named with zero-padded 6-digit indices: `000000.jpg`, `000001.jpg`, etc. This ensures alphabetical sort == chronological order.

---

## 5. Extracted Frame Directory Layout

After running `extract_frames.py`, the output directory has this structure:

```
<output_root>/
├── train/
│   ├── 20190925_111759_1_8/
│   │   ├── visible/
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── visible.json
│   ├── 20190925_112035_1_8/
│   │   ├── visible/
│   │   │   └── ...
│   │   └── visible.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Supported frame file extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## 6. Dataset Classes — Detailed Reference

All classes are in `train/antiuav.py` and exported from `train/__init__.py`.

---

### 6.1 `AntiUAVRGBTVideoDataset` — Video Mode

**When to use:** When frames have NOT been pre-extracted. Training directly from `.mp4` files. Slower but requires no pre-processing step.

**Class signature:**
```python
class AntiUAVRGBTVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,        # Root dir containing split/ subdirectories
        split: str = "train",         # "train" | "val" | "test"
        modality: str = "infrared",   # "infrared" | "visible"
        num_frames: int = 9,          # Temporal window length
        height: int = 448,            # Output frame height (resize target)
        width: int = 448,             # Output frame width (resize target)
        clip_stride: int = 4,         # Step between consecutive clips
        frame_stride: int = 1,        # Step between frames within a clip
        image_channels: int = 1,      # 1 (grayscale) or 3 (RGB)
        box_format: str = "xywh",     # Input annotation format
        label_dir_name: str = "label_new",  # Fallback annotation dir name
        sequence_ids: set[str] | None = None,  # Filter to subset of sequences
    )
```

**Internal data flow:**
1. `__init__` calls `_discover_sequences()` to find all valid `(video_path, annotation_path)` pairs.
2. `_discover_sequences()` iterates over `split_dir/`, calls `_find_annotation()` for each.
3. `_find_annotation()` tries all 9 candidate paths in order.
4. Valid sequences are loaded into `AntiUAVVideoSequence` dataclass instances.
5. `samples` list is built: all valid `(seq_idx, start_frame)` pairs given `clip_stride`.

**`__getitem__` flow:**
```
Input: integer index idx
  → look up (seq_idx, start) from self.samples[idx]
  → compute frame_indices = _frame_indices(start, sequence)
  → _read_frames(video_path, frame_indices) using OpenCV
  → per-frame: _target_from_box(box, exists, image_size)
  → return dict with "frames", "frame_targets", "image_ids", "sequence", "dataset_url"
```

**`_read_frames` implementation detail:**  
Uses `cv2.VideoCapture`. For each requested frame index, calls `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` to seek to the exact frame. This is a **random seek** and is expensive on large files. For efficiency, it is better to use `AntiUAVExtractedFrameDataset`.

**Fallback on failed read:**  
If `cap.read()` returns `ok=False`, the previous successfully decoded frame is reused. If no frame has been decoded yet, a zero tensor of the target size is returned.

---

### 6.2 `AntiUAVExtractedFrameDataset` — Pre-extracted Mode

**When to use:** After running `train/extract_frames.py`. This is the **recommended mode for training** — it reads JPEG files sequentially, which is much faster than random video seeks.

**Class signature:**
```python
class AntiUAVExtractedFrameDataset(Dataset):
    def __init__(
        self,
        frames_root: str | Path,      # Root dir of extracted frames
        split: str = "train",
        modality: str = "visible",
        num_frames: int = 5,
        height: int = 448,
        width: int = 448,
        clip_stride: int = 4,
        frame_stride: int = 1,
        image_channels: int = 3,
        box_format: str = "xywh",
        sequence_ids: set[str] | None = None,
    )
```

**Key difference from video mode:**  
Instead of `video_path`, sequences are represented by `frame_paths: tuple[Path, ...]` — a sorted tuple of all extracted frame file paths. Frame access is a simple `cv2.imread()` on the exact path, no seeking required.

**Internal data flow:**
1. `_discover_sequences()` iterates over `split_dir/`, looks for `<modality>/` subdirectory and `<modality>.json`.
2. Within each sequence, all files with supported extensions are collected and sorted alphabetically (giving chronological order).
3. Samples are built as in video mode.

**`__getitem__` flow:**
```
Input: integer index idx
  → look up (seq_idx, start) from self.samples[idx]
  → compute frame_indices = _frame_indices(start, sequence)
  → for each idx: _read_frame(sequence.frame_paths[idx]) using cv2.imread
  → stack frames: torch.stack(frames)  → [T, C, H, W]
  → return dict with "frames", "frame_targets", etc.
```

---

### 6.3 `ModelScopeAntiUAVCocoDataset` — COCO Mode

**When to use:** When the dataset has been downloaded in COCO format from ModelScope (individual frame images + COCO annotations JSON).

**Dependency:** Requires `torchvision` and `pycocotools`.

```python
class ModelScopeAntiUAVCocoDataset(Dataset):
    def __init__(
        self,
        root: str | Path,       # Directory containing image files
        ann_file: str | Path,   # Path to COCO-format annotations JSON
        num_frames: int = 9,    # Temporal window radius = num_frames // 2
        height: int = 448,
        width: int = 448,
    )
```

**Temporal window construction in COCO mode:**  
Unlike the video-based datasets, COCO images are not ordered sequentially by default. The dataset sorts all images by `file_name` (alphabetically) to infer chronological order. The temporal window is then a **centered** window of `num_frames` around the requested index:

```python
def _window_indices(self, center_index: int) -> list[int]:
    radius = self.num_frames // 2
    last = len(self.order) - 1
    return [min(max(center_index + offset, 0), last) for offset in range(-radius, radius + 1)]
```

> **⚠️ Note:** COCO mode is inherently **non-causal** (uses `radius` frames in both directions). It should only be used for exploratory experiments, not production training.

---

### 6.4 `SyntheticAntiUAVDataset` — Smoke Test Mode

**When to use:** Unit tests, CI pipelines, quick architecture verification without any external dataset.

```python
class SyntheticAntiUAVDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 16,
        num_frames: int = 5,
        height: int = 64,
        width: int = 64,
        image_channels: int = 3,
    )
```

**Generation algorithm:**
1. For each sample `i`, seed a deterministic generator: `torch.Generator().manual_seed(10_000 + i)`.
2. Generate background frames: `0.08 * torch.rand(T, C, H, W)` — dark, near-zero noise.
3. Randomly select a starting position `(start_x, start_y)` and a vertical drift rate `drift_y ∈ {-1, 0, 1}`.
4. For each frame `t`, paint a bright rectangle at `(start_x + t, start_y + drift_y * t)`.
   - Color: `[0.95, 0.92, 0.35]` (bright yellowish) for RGB; `[0.95]` for grayscale.
5. Generate ground truth bounding box from the rectangle coordinates.
6. Return frames as `[T, C, H, W]` tensor clamped to `[0, 1]`.

**Why this design?**  
The target moves by exactly 1 pixel per frame horizontally. This tests that the temporal fusion module can learn directional velocity. The drift in `y` tests multi-axis tracking.

---

## 7. Temporal Window Construction

### 7.1 How Samples Are Indexed

For any sequence of length `N_frames`, clips are generated at intervals of `clip_stride`:

```
window_span = (num_frames - 1) * frame_stride + 1
max_start   = max(0, N_frames - window_span)
starts      = range(0, max_start + 1, clip_stride)
# Always include max_start to ensure the last clip is included
if starts[-1] != max_start:
    starts.append(max_start)
```

Each `(seq_idx, start)` pair is one sample in `self.samples`.

**Example:**
- Sequence length: 100 frames
- `num_frames=5`, `frame_stride=1`, `clip_stride=4`
- `window_span = (5-1)*1 + 1 = 5`
- `max_start = 100 - 5 = 95`
- `starts = [0, 4, 8, ..., 92, 95]` → 25 clips

### 7.2 Frame Index Computation

Given a clip starting at `start`, frame indices within the clip are:

```python
def _frame_indices(self, start: int, sequence) -> list[int]:
    last = max(sequence.num_frames - 1, 0)
    return [
        min(start + frame_idx * self.frame_stride, last)
        for frame_idx in range(self.num_frames)
    ]
```

The `min(..., last)` clamp ensures that if a window overshoots the end of a short sequence, the last valid frame is repeated.

**Example:**
- `start=95`, `num_frames=5`, `frame_stride=1`, sequence length=100
- indices: `[95, 96, 97, 98, 99]` ✓

**Short sequence edge case:**
- Sequence length=3, `num_frames=5`, `start=0`
- indices: `[0, 1, 2, 2, 2]` (frame 2 repeated twice)

---

## 8. Frame Decoding & Resizing

### 8.1 For Video Mode (OpenCV VideoCapture)

```python
def _cv_frame_to_tensor(frame: np.ndarray, image_channels: int) -> Tensor:
    if frame.ndim == 2:
        # Grayscale native (infrared FLIR cameras)
        tensor = torch.from_numpy(frame.copy()).float().div(255.0).unsqueeze(0)
        if image_channels == 3:
            tensor = tensor.repeat(3, 1, 1)  # repeat grayscale into 3 channels
        return tensor
    
    if image_channels == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(gray.copy()).float().div(255.0).unsqueeze(0)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).float().div(255.0).permute(2, 0, 1)
```

**Key steps:**
1. OpenCV reads frames as BGR uint8 `np.ndarray` by default.
2. Convert to RGB channel order using `cv2.COLOR_BGR2RGB`.
3. Normalize to `[0.0, 1.0]` by dividing by 255.
4. Permute from `[H, W, C]` to `[C, H, W]` (PyTorch convention).

**Infrared mode (`image_channels=1`):**
- IR frames are often saved as 8-bit grayscale. They are read, converted to single-channel float, and kept as `[1, H, W]`.
- If `image_channels=3` with IR, the grayscale channel is tripled: `[1, H, W] → [3, H, W]`. This allows using RGB-pretrained encoders with IR frames.

### 8.2 Resizing

All frames are resized to `(height, width)` using bilinear interpolation:

```python
def _resize_chw(image: Tensor, height: int, width: int) -> Tensor:
    return F.interpolate(
        image.unsqueeze(0),           # add batch dim for interpolate
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                      # remove batch dim
```

Resizing is applied after tensor conversion, in float `[0,1]` space.

**Note:** For the full Anti-UAV dataset, frames are typically 640×512 (IR) or 1920×1080 (visible). They are downsampled to the configured `height × width` (default 448×448 in production, 64×64 in smoke tests).

---

## 9. Bounding Box Format Conventions

### 9.1 Raw Annotation Format
The Anti-UAV JSON files provide boxes in `[x, y, w, h]` format where `x, y` is the **top-left corner** in absolute pixel coordinates.

### 9.2 Model Input Format
The pipeline expects normalized **center-point** format: `[cx, cy, w, h]` where all values are in `[0.0, 1.0]`.

**Conversion (in `_target_from_antiuav_box`):**
```python
# Input: [x, y, w, h] in absolute pixels
cx = (x + box_w / 2.0) / max(image_width, 1)   # normalize center x
cy = (y + box_h / 2.0) / max(image_height, 1)  # normalize center y
w  = box_w / max(image_width, 1)                 # normalize width
h  = box_h / max(image_height, 1)               # normalize height
# All values clamped to [0.0, 1.0]
```

### 9.3 XYXY Support
The loader also supports `xyxy` format input via `box_format="xyxy"`:
```python
if box_format == "xyxy":
    x1, y1, x2, y2 = values
    x, y, box_w, box_h = x1, y1, x2 - x1, y2 - y1
```

### 9.4 Validation Rules
A bounding box is accepted only if:
- It has at least 4 numeric values
- `box_w > 0` and `box_h > 0` (non-zero area)
- The `exists` flag for the frame is `True`

---

## 10. Target Construction

### 10.1 `_empty_target()` — Invisible Frame
When the target is not visible (`exists=False`) or the box is invalid:
```python
def _empty_target() -> dict[str, Tensor]:
    return {
        "boxes":  torch.zeros(0, 4, dtype=torch.float32),  # [0, 4]
        "labels": torch.zeros(0, dtype=torch.long),         # [0]
    }
```

### 10.2 Visible Frame Target
```python
{
    "boxes":  torch.tensor([[cx, cy, w, h], ...], dtype=torch.float32),  # [N, 4]
    "labels": torch.ones(N, dtype=torch.long),                           # [N] — all class 1
}
```

In Anti-UAV, there is always at most **one target per frame** (`N ≤ 1`). The schema supports `N > 1` for future multi-target extension.

### 10.3 Per-frame Target List in `__getitem__`
Each sample returns a list of `num_frames` targets, one per temporal position:
```python
"frame_targets": [
    target_at_frame_0,   # dict with "boxes" and "labels"
    target_at_frame_1,
    ...
    target_at_frame_{T-1}
]
```

---

## 11. Collator — Batching Strategy

### 11.1 `DRISHTICollator`

The collator stacks frames into a batch tensor and collects target lists.

```python
class DRISHTICollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "frames": torch.stack([item["frames"] for item in batch]),
            # Shape: [B, T, C, H, W]
            
            "frame_targets": [item["frame_targets"] for item in batch],
            # Shape: list of length B, each is list of T target dicts
            # Not padded — kept as nested Python lists
            
            "image_ids": [item["image_ids"] for item in batch],
            # Shape: list of length B, each is list of T image ID strings
            
            "dataset_url": batch[0].get("dataset_url", MODELSCOPE_ANTI_UAV_URL),
        }
```

**Why are frame_targets NOT padded into a tensor?**  
The number of GT boxes per frame varies (0 or 1 for Anti-UAV, potentially more in swarm extensions). A jagged list is much simpler than a padded tensor with a mask. Loss functions iterate over the list directly.

### 11.2 `AntiUAVDetectionCollator`

An alternative collator that converts the per-frame target lists into patch-level tensors indexed by spatial patch grid. This is used for the patch-classification training variant.

```python
class AntiUAVDetectionCollator:
    def __init__(self, patch_grid_size: int) -> None:
        self.patch_grid_size = patch_grid_size
    
    def __call__(self, batch) -> dict[str, Any]:
        # Returns:
        # "frames":         [B, T, C, H, W]
        # "class_targets":  [B, T, P*P]        where P = patch_grid_size
        # "box_targets":    [B, T, P*P, 4]
        # "box_mask":       [B, T, P*P] bool
        # "frame_targets":  list[B][T] (raw)
```

The patch assignment logic: for each GT box with center `(cx, cy)`, the corresponding patch index is:
```python
grid_x = int(cx * patch_grid_size)    # column index
grid_y = int(cy * patch_grid_size)    # row index
patch_idx = grid_y * patch_grid_size + grid_x
```

---

## 12. DataLoader Configuration

The recommended DataLoader settings for each mode are:

```python
# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,      # 16 for Stage 1, 8 for Stage 2+
    shuffle=True,                      # Always shuffle training set
    collate_fn=DRISHTICollator(),
    num_workers=config.num_workers,    # 4 recommended for SSD storage
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(config.num_workers > 0),
    drop_last=False,
)

# Validation / Test DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,                     # Never shuffle eval set
    collate_fn=DRISHTICollator(),
    num_workers=config.num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(config.num_workers > 0),
    drop_last=False,
)
```

**`persistent_workers=True` rationale:**  
When `num_workers > 0`, worker processes are kept alive between batches. This avoids the overhead of forking and loading the dataset into each worker process at every epoch.

**`pin_memory=True` rationale:**  
Pre-allocates page-locked (pinned) CPU memory for the batch. This allows the CUDA `memcpy` to device to happen asynchronously, reducing GPU idle time.

---

## 13. Dataset Selection Logic in `experiment.py`

The `make_dataloaders()` function in `experiment.py` implements a priority-ordered selection among all four dataset modes:

```
Priority 1: --frames-root is provided
            → Use AntiUAVExtractedFrameDataset
            → FASTEST — recommended for all training runs

Priority 2: --data-root is provided
            → Use AntiUAVRGBTVideoDataset
            → Slower — reads directly from .mp4 files

Priority 3: --train-image-root AND --train-ann-file are both provided
            → Use ModelScopeAntiUAVCocoDataset
            → Requires torchvision, non-causal windows

Priority 4: --smoke flag is set (no data paths provided)
            → Use SyntheticAntiUAVDataset
            → For unit tests and smoke tests only

Otherwise: Raise ValueError with instructions
```

### 13.1 Smoke Mode Configuration
```python
train_dataset = SyntheticAntiUAVDataset(
    num_samples=config.smoke_train_samples,  # default: 8
    num_frames=config.num_frames,            # default: 5
    height=config.frame_height,              # default: 64
    width=config.frame_width,               # default: 64
    image_channels=config.image_channels,   # default: 3
)
```

---

## 14. Configuration Parameters Reference

All parameters are controlled by `ExperimentConfig` in `experiment.py`:

| Parameter | CLI Flag | Default | Unit | Description |
|---|---|---|---|---|
| `frames_root` | `--frames-root` | `None` | path | Root of pre-extracted frames |
| `data_root` | `--data-root` | `None` | path | Root of raw Anti-UAV `.mp4` files |
| `train_split` | `--train-split` | `"train"` | string | Split name used as subdirectory |
| `val_split` | `--val-split` | `"test"` | string | Val/test split name |
| `modality` | `--modality` | `"visible"` | enum | `"visible"` or `"infrared"` |
| `num_frames` | `--num-frames` | `5` | frames | Temporal window length `T` |
| `clip_stride` | `--clip-stride` | `4` | frames | Gap between clip start positions |
| `frame_stride` | `--frame-stride` | `1` | frames | Gap between frames within a clip |
| `frame_height` | `--height` | `64` (smoke) / `448` (full) | pixels | Output frame height |
| `frame_width` | `--width` | `64` (smoke) / `448` (full) | pixels | Output frame width |
| `image_channels` | `--image-channels` | `3` | channels | 1 (grayscale) or 3 (RGB) |
| `box_format` | `--box-format` | `"xywh"` | enum | Input annotation box format |
| `batch_size` | `--batch-size` | `2` | samples | Clips per batch |
| `num_workers` | `--num-workers` | `0` | processes | DataLoader worker processes |

---

## 15. Adding a New Dataset

To add a new dataset (e.g., a different UAV detection dataset):

1. **Create a new Dataset class** in `train/antiuav.py` that subclasses `torch.utils.data.Dataset`.

2. **Match the `__getitem__` return schema** exactly:
   ```python
   {
       "frames": Tensor,          # [T, C, H, W], float32, [0.0, 1.0]
       "frame_targets": list,     # list of T dicts, each: {"boxes": [N,4], "labels": [N]}
       "image_ids": list,         # list of T unique string identifiers
       "dataset_url": str,        # source URL for provenance tracking
   }
   ```
   The `DRISHTICollator` and loss functions depend on this exact contract.

3. **Box format:** Ensure `boxes` are in normalized `[cx, cy, w, h]` format with values in `[0.0, 1.0]`.

4. **Export the class** from `train/__init__.py`.

5. **Add selection logic** in `make_dataloaders()` in `experiment.py`.

6. **Add a smoke test** in `tests/` verifying:
   - Correct tensor shapes.
   - No infinite values.
   - Boxes within `[0, 1]`.
   - Length consistency between `frames` and `frame_targets`.
