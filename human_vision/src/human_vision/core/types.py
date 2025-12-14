from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Tuple


class Capability(str, Enum):
    FACE = "face"
    POSE = "pose"
    HAND = "hand"
    SEGMENTATION = "segmentation"
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    LIVENESS = "liveness"


class BackendKind(str, Enum):
    MEDIAPIPE = "mediapipe"
    TFLITE = "tflite"
    ONNX = "onnx"
    REPO_ASSETS = "repo_assets"


class ModelDownload(NamedTuple):
    type: str
    url: str
    path_in_zip: str | None = None
    sha256: str | None = None


class FrameShape(NamedTuple):
    height: int
    width: int
    channels: int


SizeHW = Tuple[int, int]
