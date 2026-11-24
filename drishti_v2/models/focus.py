"""Chain-of-Focus refinement: "where to look, then how closely to look".

After the detector has produced one box per crop, the top-K candidates get a closer look:

    box  ->  focus window centred on the box (``focus_expand`` x its size, in pixels)
         ->  re-crop that window from the full frame at the encoder's input size (a zoom-in)
         ->  shared crop encoder  ->  focus head  ->  refined box (+ a "stop" logit)

* ``focus_mode: fixed``     one focus step for each of the top-K candidates (version 1).
* ``focus_mode: adaptive``  up to ``focus_max_steps`` steps; a halting head predicts whether the
  current box is already tight (IoU >= ``focus_halt_iou``); the candidate stops as soon as it says
  so (version 2, adaptive compute per frame, inspired by Chain-of-Focus for VLMs, arXiv:2505.15436).

Training needs no reinforcement learning: input boxes are the ground truth jittered in position and
size (plus the model's own top prediction), the regression target is the ground truth, and the halting
target is "IoU(input box, GT) >= focus_halt_iou".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _box_iou_pairwise(a: Tensor, b: Tensor) -> Tensor:
    """IoU of matching rows of two [N, 4] cx,cy,w,h tensors."""

    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp_min(0).prod(-1)
    union = a[:, 2:].prod(-1) + b[:, 2:].prod(-1) - inter
    return inter / union.clamp_min(1e-9)


class FocusRefiner(nn.Module):
    def __init__(self, config, encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self._encoder = [encoder]  # shared with the pipeline; kept out of this module's parameters
        dim, hidden = config.encoder_feature_dim, config.head_hidden_dim
        self.stem = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.box_head = nn.Linear(hidden, 4)
        self.halt_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)  # starts as "box = window centre", refined by training

    @property
    def encoder(self) -> nn.Module:
        return self._encoder[0]

    # ----------------------------------------------------------------- geometry
    def windows_for(self, boxes: Tensor, frame_hw: tuple[int, int]) -> Tensor:
        """Window size in pixels for each cx,cy,w,h box (normalized coords), clamped."""

        height, width = frame_hw
        side = torch.maximum(boxes[..., 2] * (width - 1), boxes[..., 3] * (height - 1)) * self.config.focus_expand
        return side.clamp(self.config.focus_min_window_px, self.config.focus_max_window_px)

    def extract(self, frames: Tensor, frame_index: Tensor, centers: Tensor, window_px: Tensor) -> Tensor:
        """Zoomed crops [N, C*len(scales), S, S]; window i comes from frames[frame_index[i]]."""

        _, _, height, width = frames.shape
        size = self.config.crop_size
        steps = torch.linspace(-0.5, 0.5, size, device=frames.device, dtype=frames.dtype)
        crops = []
        for scale in self.config.crop_scales:
            span = window_px.to(frames.dtype) * scale  # [N] pixels
            gx = centers[:, 0, None] * 2 - 1 + 2 * steps[None, :] * span[:, None] / max(width - 1, 1)
            gy = centers[:, 1, None] * 2 - 1 + 2 * steps[None, :] * span[:, None] / max(height - 1, 1)
            grid = torch.stack(torch.broadcast_tensors(gx[:, None, :], gy[:, :, None]), dim=-1)  # [N, S, S, 2]
            src = frames[frame_index]
            crops.append(F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True))
        return torch.cat(crops, dim=1)

    def predict(self, frames: Tensor, frame_index: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One focus step for N input boxes -> (refined boxes [N,4], halt logits [N], window px [N])."""

        height, width = frames.shape[-2:]
        window_px = self.windows_for(boxes, (height, width))
        crops = self.extract(frames, frame_index, boxes[:, :2].clamp(0, 1), window_px)
        feats = self.stem(self.encoder(crops))
        raw = self.box_head(feats)
        rel_xy = torch.sigmoid(raw[:, :2])
        rel_wh = F.softplus(raw[:, 2:] + 0.5413) * (1.0 / self.config.focus_expand)  # init: wh = box size
        span = torch.stack([window_px / max(width - 1, 1), window_px / max(height - 1, 1)], dim=-1)
        centers = boxes[:, :2] + (rel_xy - 0.5) * span
        refined = torch.cat([centers, rel_wh * span], dim=-1)
        refined = torch.cat([refined[:, :2].clamp(0, 1), refined[:, 2:].clamp(1e-4, 1)], dim=-1)
        return refined, self.halt_head(feats).squeeze(-1), window_px

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def refine(self, frames: Tensor, boxes: Tensor, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Refine the top-K boxes of each frame. Returns (boxes [B,K',4], focus windows used per frame [B])."""

        batch, num = boxes.shape[:2]
        k = min(self.config.focus_topk, num)
        top = logits.squeeze(-1).topk(k, dim=1).indices  # [B, k]
        frame_index = torch.arange(batch, device=boxes.device).repeat_interleave(k)
        current = boxes.gather(1, top[..., None].expand(-1, -1, 4)).reshape(-1, 4)
        active = torch.ones(len(current), dtype=torch.bool, device=boxes.device)
        used = torch.zeros(len(current), device=boxes.device)
        adaptive = self.config.focus_mode == "adaptive"
        for _ in range(self.config.focus_max_steps):
            if not active.any():
                break
            idx = active.nonzero().squeeze(-1)
            refined, halt, _ = self.predict(frames, frame_index[idx], current[idx])
            used[idx] += 1
            stop = (torch.sigmoid(halt) > 0.5) if adaptive else torch.zeros_like(halt, dtype=torch.bool)
            move = ~stop
            current[idx[move]] = refined[move]
            active[idx[stop]] = False
        out = boxes.clone()
        out.scatter_(1, top[..., None].expand(-1, -1, 4), current.view(batch, k, 4))
        return out, used.view(batch, k).sum(dim=1)

    # ----------------------------------------------------------------- training
    def training_loss(self, frames: Tensor, pred_boxes: Tensor, pred_logits: Tensor, targets: list) -> dict[str, Tensor]:
        """Supervised focus loss on the last frame of each clip (no RL)."""

        from drishti_v2.training.ciou_loss import ciou_loss  # lazy: avoids a models<->training import cycle

        device = frames.device
        inputs, gts, frame_index = [], [], []
        for b, clip in enumerate(targets):
            target = clip[-1] if isinstance(clip, list) else clip
            gt = target.get("boxes", torch.empty(0, 4))
            if gt.numel() == 0:
                continue
            gt = gt[0].to(device=device, dtype=frames.dtype)
            n = self.config.focus_train_windows
            size = gt[2:].clamp_min(1e-3)
            jitter_c = gt[:2] + torch.randn(n, 2, device=device) * size * 0.5
            jitter_s = size * torch.exp(torch.randn(n, 2, device=device) * 0.35)
            jittered = torch.cat([jitter_c.clamp(0, 1), jitter_s], dim=-1)
            top = pred_boxes[b, pred_logits[b].squeeze(-1).argmax()].detach()[None]
            inputs.append(torch.cat([jittered, top], dim=0))
            gts.append(gt.expand(n + 1, 4))
            frame_index.append(torch.full((n + 1,), b, dtype=torch.long, device=device))
        zero = pred_boxes.sum() * 0.0
        if not inputs:
            return {"focus": zero, "focus_halt": zero}
        current, gt, frame_index = torch.cat(inputs), torch.cat(gts), torch.cat(frame_index)

        box_losses, halt_losses = [], []
        steps = self.config.focus_max_steps  # fixed mode with 2 steps (idea E) trains both steps
        for _ in range(steps):
            refined, halt, window_px = self.predict(frames, frame_index, current)
            height, width = frames.shape[-2:]
            span = torch.stack([window_px / max(width - 1, 1), window_px / max(height - 1, 1)], dim=-1)
            inside = ((gt[:, :2] - current[:, :2]).abs() < span / 2).all(dim=-1)  # GT visible in the window
            if inside.any():
                box_losses.append(ciou_loss(refined[inside], gt[inside]) + F.l1_loss(refined[inside], gt[inside]) * 10.0)
            good = (_box_iou_pairwise(current, gt) >= self.config.focus_halt_iou).to(halt.dtype)
            halt_losses.append(F.binary_cross_entropy_with_logits(halt, good))
            current = refined.detach()
        focus = torch.stack(box_losses).mean() if box_losses else zero
        halt_loss = torch.stack(halt_losses).mean() if self.config.focus_mode == "adaptive" else zero
        return {"focus": focus, "focus_halt": halt_loss}
