from __future__ import annotations

import numpy as np


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        return frame[..., :3]
    return frame
