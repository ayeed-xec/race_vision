from __future__ import annotations

import numpy as np

from ..core.errors import MissingDependencyError
from ..models.registry import ModelSpec
from ..models.store import ModelStore
from ..results import PosePerson, PoseResult, Keypoint2D


class _TFLiteAdapter:
    thread_safe = True

    def __init__(self, spec: ModelSpec, store: ModelStore):
        self.spec = spec
        self.store = store

    def warmup(self) -> None:
        return None

    def infer(self, frame: np.ndarray, roi=None):  # type: ignore[override]
        # produce dummy 17-keypoint pose
        keypoints = [Keypoint2D(x=0.5, y=0.5, score=0.0) for _ in range(17)]
        return PoseResult(persons=[PosePerson(keypoints=keypoints, score=0.0)])

    def close(self) -> None:
        return None


class TFLiteBackend:
    def build_adapter(self, spec: ModelSpec, store: ModelStore):  # type: ignore[override]
        try:
            import tflite_runtime  # type: ignore  # noqa: F401
        except Exception:
            raise MissingDependencyError(
                "Optional dependency 'tflite-runtime' missing. Install via pip install human_vision[tflite]"
            )
        store.ensure(spec)
        return _TFLiteAdapter(spec, store)
