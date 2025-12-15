from __future__ import annotations

import numpy as np

from ..core.errors import MissingDependencyError, MissingModelError
from ..models.registry import ModelSpec
from ..models.store import ModelStore
from ..results import FaceResult


class _OnnxAdapter:
    thread_safe = True

    def __init__(self, spec: ModelSpec, path):
        self.spec = spec
        self.path = path

    def warmup(self) -> None:
        return None

    def infer(self, frame: np.ndarray, roi=None):  # type: ignore[override]
        # placeholder returning empty results
        if self.spec.capability == self.spec.capability.FACE:
            return FaceResult()
        return None

    def close(self) -> None:
        return None


class OnnxBackend:
    def build_adapter(self, spec: ModelSpec, store: ModelStore):  # type: ignore[override]
        try:
            import onnxruntime  # type: ignore  # noqa: F401
        except Exception:
            raise MissingDependencyError(
                "Optional dependency 'onnxruntime' missing. Install via pip install human_vision[onnx]"
            )
        path = store.get_path(spec.model_id)
        if not path.exists():
            raise MissingModelError(spec.model_id, str(path))
        return _OnnxAdapter(spec, path)
