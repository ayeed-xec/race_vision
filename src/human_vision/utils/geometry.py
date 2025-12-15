from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def letterbox_resize(frame: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = frame.shape[:2]
    target_h, target_w = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
    import cv2  # type: ignore

    resized_part = cv2.resize(frame, (new_w, new_h))
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    resized[top : top + new_h, left : left + new_w] = resized_part
    return resized, scale, (left, top)


def bbox_from_landmarks(points: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)
