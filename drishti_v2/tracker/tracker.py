from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from drishti_v2.assignment import linear_sum_assignment


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
    last_unpredicted_center: Tensor | None = None


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
            track.last_unpredicted_center = track.center.clone()
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

        if self.tracks and det_boxes.numel() > 0:
            track_centers = torch.stack([track.center.to(det_boxes.device) for track in self.tracks])
            cost = torch.cdist(track_centers, det_boxes[:, :2])
            track_indices, detection_indices = linear_sum_assignment(cost)
        else:
            track_indices = torch.empty(0, dtype=torch.long, device=det_boxes.device)
            detection_indices = torch.empty(0, dtype=torch.long, device=det_boxes.device)

        for track_idx, best_det in zip(track_indices.tolist(), detection_indices.tolist()):
            if float(cost[track_idx, best_det]) < self.dist_threshold:
                track = self.tracks[track_idx]
                new_center = det_boxes[best_det, :2].clone()
                previous_center = track.last_unpredicted_center
                if previous_center is None:
                    previous_center = track.center
                track.velocity = (new_center - previous_center.to(new_center.device)).detach()
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


class TargetMemory:
    """Remember where the single drone was last seen confidently, and keep looking there (idea A).

    The motion heatmap misses hovering or slow drones (no motion, no peak). This memory keeps a
    constant-velocity estimate of the last confident detection and, for up to ``max_age`` frames
    after it, offers crop centres at the predicted position (and the last seen position) so the
    detector keeps looking where the drone is likely to be even when the motion cue disappears.
    """

    def __init__(self, threshold: float = 0.3, max_age: int = 30, momentum: float = 0.6) -> None:
        self.threshold = threshold
        self.max_age = max_age
        self.momentum = momentum
        self.reset()

    def reset(self) -> None:
        self.center: Tensor | None = None
        self.velocity = torch.zeros(2)
        self.age = 0

    def update(self, boxes: Tensor, scores: Tensor) -> None:
        """Feed this frame's output (any number of boxes); only the best confident box is used."""

        if scores.numel() > 0 and float(scores.max()) >= self.threshold:
            best = boxes[int(scores.argmax()), :2].detach().cpu().float()
            if self.center is not None and self.age <= self.max_age:
                step = (best - self.center) / max(self.age, 1)
                self.velocity = self.momentum * self.velocity + (1 - self.momentum) * step
            self.center = best
            self.age = 1
        elif self.center is not None:
            self.age += 1

    def guided_centers(self) -> Tensor | None:
        """[1, M, 2] crop centres for the next frame, or None when memory has expired."""

        if self.center is None or self.age > self.max_age:
            return None
        predicted = (self.center + self.velocity * self.age).clamp(0.0, 1.0)
        return torch.stack([predicted, self.center.clamp(0.0, 1.0)]).unsqueeze(0)
