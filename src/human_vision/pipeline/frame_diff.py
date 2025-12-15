from __future__ import annotations

import numpy as np


class FrameDiffer:
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._last: np.ndarray | None = None

    def has_significant_change(self, frame: np.ndarray) -> bool:
        if self._last is None:
            self._last = frame.copy()
            return True
        diff = np.mean(np.abs(frame.astype(float) - self._last.astype(float))) / 255.0
        self._last = frame.copy()
        return diff > self.threshold
