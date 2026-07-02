from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
