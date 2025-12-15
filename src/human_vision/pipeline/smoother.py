from __future__ import annotations

from typing import Iterable, List

from ..results import Keypoint2D


class EmaSmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._last: List[Keypoint2D] | None = None

    def smooth(self, keypoints: Iterable[Keypoint2D]) -> List[Keypoint2D]:
        kp_list = list(keypoints)
        if self._last is None:
            self._last = kp_list
            return kp_list
        smoothed: List[Keypoint2D] = []
        for prev, curr in zip(self._last, kp_list):
            smoothed.append(
                Keypoint2D(
                    x=self.alpha * curr.x + (1 - self.alpha) * prev.x,
                    y=self.alpha * curr.y + (1 - self.alpha) * prev.y,
                    score=curr.score,
                    name=curr.name or prev.name,
                )
            )
        self._last = smoothed
        return smoothed
