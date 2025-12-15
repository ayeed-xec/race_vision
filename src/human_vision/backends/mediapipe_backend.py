from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import MissingDependencyError
from ..core.contracts import IModelAdapter
from ..models.registry import ModelSpec
from ..models.store import ModelStore
from ..results import FaceResult, HandResult, PoseResult, SegmentationResult


class _EmptyAdapter:
    thread_safe = True

    def __init__(self, capability: str):
        self.capability = capability

    def warmup(self) -> None:
        return None

    def infer(self, frame: np.ndarray, roi=None):  # type: ignore[override]
        if self.capability == "face":
            return FaceResult()
        if self.capability == "hand":
            return HandResult()
        if self.capability == "pose":
            return PoseResult()
        if self.capability == "segmentation":
            return SegmentationResult(mask=np.zeros(frame.shape[:2], dtype=np.uint8))
        return None

    def close(self) -> None:
        return None


class MediaPipeBackend:
    def build_adapter(self, spec: ModelSpec, store: ModelStore) -> IModelAdapter:  # type: ignore[override]
        try:
            import mediapipe  # noqa: F401
        except Exception:
            # degrade gracefully
            raise MissingDependencyError(
                "Optional dependency 'mediapipe' missing. Install via pip install human_vision[mediapipe]"
            )
        # For simplicity return empty adapter; real implementation would wrap mediapipe solutions
        return _EmptyAdapter(spec.capability.value)
