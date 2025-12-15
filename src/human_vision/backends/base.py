from __future__ import annotations

from typing import Dict, Type

from ..core.types import BackendKind
from .mediapipe_backend import MediaPipeBackend
from .tflite_backend import TFLiteBackend
from .onnx_backend import OnnxBackend


class BackendFactory:
    _registry: Dict[BackendKind, object] = {
        BackendKind.MEDIAPIPE: MediaPipeBackend(),
        BackendKind.TFLITE: TFLiteBackend(),
        BackendKind.ONNX: OnnxBackend(),
    }

    @classmethod
    def get(cls, kind: BackendKind):
        return cls._registry[kind]
