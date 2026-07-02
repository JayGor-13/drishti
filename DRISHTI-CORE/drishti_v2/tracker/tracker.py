from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Track:
    track_id: int
    center: Tensor
    size: Tensor
    velocity: Tensor
    confidence: float
    age: int = 0
    coast_count: int = 0
    hit_count: int = 1


class SimpleTracker:
    """Constant-velocity Euclidean-gated multi-target tracker."""

    def __init__(self, dist_threshold: float = 0.15, max_coast: int = 15, birth_threshold: float = 0.3) -> None:
        self.dist_threshold = dist_threshold
        self.max_coast = max_coast
        self.birth_threshold = birth_threshold
        self.tracks: list[Track] = []
        self._next_id = 0

    def predict(self) -> None:
        for track in self.tracks:
            track.center = (track.center + track.velocity.to(track.center.device)).clamp(0.0, 1.0)
            track.coast_count += 1
            track.age += 1

    def update(self, boxes: Tensor, logits: Tensor) -> None:
        if boxes.numel() == 0:
            self.tracks = [track for track in self.tracks if track.coast_count <= self.max_coast]
            return
        confs = torch.sigmoid(logits.squeeze(-1))
        keep = confs > self.birth_threshold
        det_boxes = boxes[keep].detach()
        det_confs = confs[keep].detach()
        matched_det: set[int] = set()
        matched_track: set[int] = set()

        for track_idx, track in enumerate(self.tracks):
            best_dist = float("inf")
            best_det = -1
            for det_idx, det in enumerate(det_boxes):
                if det_idx in matched_det:
                    continue
                dist = torch.norm(track.center.to(det.device) - det[:2]).item()
                if dist < best_dist:
                    best_dist = dist
                    best_det = det_idx
            if best_det >= 0 and best_dist < self.dist_threshold:
                new_center = det_boxes[best_det, :2].clone()
                track.velocity = (new_center - track.center.to(new_center.device)).detach()
                track.center = new_center
                track.size = det_boxes[best_det, 2:].clone()
                track.confidence = float(det_confs[best_det].item())
                track.coast_count = 0
                track.hit_count += 1
                matched_det.add(best_det)
                matched_track.add(track_idx)

        self.tracks = [track for idx, track in enumerate(self.tracks) if idx in matched_track or track.coast_count <= self.max_coast]

        for det_idx, det in enumerate(det_boxes):
            if det_idx not in matched_det:
                self.tracks.append(
                    Track(
                        track_id=self._next_id,
                        center=det[:2].clone(),
                        size=det[2:].clone(),
                        velocity=torch.zeros(2, device=det.device),
                        confidence=float(det_confs[det_idx].item()),
                    )
                )
                self._next_id += 1

    def get_guided_centers(self) -> Tensor | None:
        if not self.tracks:
            return None
        return torch.stack([track.center for track in self.tracks], dim=0).unsqueeze(0)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0
