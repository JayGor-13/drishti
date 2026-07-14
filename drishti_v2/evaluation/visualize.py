from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def save_detection_figure(frame: Tensor, boxes: Tensor, scores: Tensor, path: str | Path, threshold: float = 0.3) -> None:
    """Save a simple detection overlay for debugging."""

    image = frame.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for box, score in zip(boxes.detach().cpu(), scores.detach().cpu()):
        if float(score) < threshold:
            continue
        cx, cy, bw, bh = box.tolist()
        x0 = (cx - bw / 2) * width
        y0 = (cy - bh / 2) * height
        rect = plt.Rectangle((x0, y0), bw * width, bh * height, fill=False, color="lime", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x0, y0, f"{float(score):.2f}", color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.6})
    ax.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_metrics_bar(metrics: dict[str, float], path: str | Path, title: str = "DRISHTI metrics") -> None:
    """Save a compact bar chart for scalar metrics."""

    scalar_items = [(key, value) for key, value in metrics.items() if isinstance(value, (int, float))]
    if not scalar_items:
        return
    keys, values = zip(*scalar_items)
    fig_width = max(8, len(keys) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(range(len(values)), values, color="#4C9AFF")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    """Plot train/validation scalar history after a run."""

    if not history:
        return
    epochs = [row.get("epoch", idx + 1) for idx, row in enumerate(history)]
    keys = [key for key in sorted(history[-1]) if key.startswith(("train_", "val_")) and key != "train_gate"]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key in keys:
        values = [row.get(key) for row in history]
        if any(value is not None for value in values):
            ax.plot(epochs, [0.0 if value is None else value for value in values], marker="o", label=key)
    ax.set_xlabel("epoch")
    ax.set_title("Training progress")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_moe_diagnostics(metrics: dict[str, float | list[float]], path: str | Path) -> None:
    """Visualize MoE expert utilization and scalar router diagnostics."""

    utilization = metrics.get("expert_utilization")
    if not isinstance(utilization, list):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(utilization)), utilization, color="#36B37E")
    axes[0].set_title("Expert utilization")
    axes[0].set_xlabel("expert")
    axes[0].set_ylabel("fraction")
    scalar_keys = ["load_balance_cv", "router_entropy", "token_drop_rate", "expert_overlap"]
    scalar_values = [float(metrics.get(key, 0.0)) for key in scalar_keys]
    axes[1].bar(range(len(scalar_keys)), scalar_values, color="#FFAB00")
    axes[1].set_xticks(range(len(scalar_keys)))
    axes[1].set_xticklabels(scalar_keys, rotation=35, ha="right")
    axes[1].set_title("Router diagnostics")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_boxes_on_image(
    image: Tensor | np.ndarray,
    boxes: Tensor,
    scores: Tensor,
    threshold: float = 0.3,
    top_k: int = 1,
    gt_boxes: Tensor | None = None,
) -> np.ndarray:
    """Return an RGB uint8 image with predictions and optional GT boxes drawn."""

    import cv2

    if isinstance(image, Tensor):
        array = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        array = (array * 255).astype(np.uint8)
    else:
        array = image.copy()
        if array.dtype != np.uint8:
            array = np.clip(array * 255, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    height, width = array.shape[:2]
    output = array.copy()

    if gt_boxes is not None:
        for box in gt_boxes.detach().cpu():
            _draw_cxcywh(output, box, width, height, color=(0, 220, 255), label="gt")

    if boxes.numel() > 0 and scores.numel() > 0:
        order = torch.argsort(scores.detach().cpu(), descending=True)
        drawn = 0
        for index in order:
            score = float(scores[index])
            if score < threshold and drawn >= top_k:
                continue
            _draw_cxcywh(output, boxes[index].detach().cpu(), width, height, color=(60, 255, 60), label=f"{score:.2f}")
            drawn += 1
            if drawn >= top_k and score < threshold:
                break
    return output


def _draw_cxcywh(image: np.ndarray, box: Tensor, width: int, height: int, color: tuple[int, int, int], label: str) -> None:
    import cv2

    cx, cy, bw, bh = [float(value) for value in box.tolist()]
    x0 = int(max(0, min(width - 1, (cx - bw / 2.0) * width)))
    y0 = int(max(0, min(height - 1, (cy - bh / 2.0) * height)))
    x1 = int(max(0, min(width - 1, (cx + bw / 2.0) * width)))
    y1 = int(max(0, min(height - 1, (cy + bh / 2.0) * height)))
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
