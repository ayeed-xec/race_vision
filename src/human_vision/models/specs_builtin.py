from __future__ import annotations

from .registry import ModelRegistry, ModelSpec
from ..core.types import BackendKind, Capability, ModelDownload


SPEC_LIST = [
    # Pose
    ModelSpec(
        model_id="movenet-lightning",
        capability=Capability.POSE,
        backend_kind=BackendKind.TFLITE,
        enabled_by_default=True,
        download=ModelDownload(
            type="tfhub",
            url="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3",
        ),
        input_size=(192, 192),
    ),
    ModelSpec(
        model_id="movenet-thunder",
        capability=Capability.POSE,
        backend_kind=BackendKind.TFLITE,
        enabled_by_default=False,
        download=ModelDownload(
            type="tfhub",
            url="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3",
        ),
        input_size=(256, 256),
    ),
    ModelSpec("movenet-multipose", Capability.POSE, BackendKind.TFLITE, False),
    ModelSpec("posenet", Capability.POSE, BackendKind.MEDIAPIPE, False),
    ModelSpec("blazepose-lite", Capability.POSE, BackendKind.MEDIAPIPE, False),
    ModelSpec("blazepose-full", Capability.POSE, BackendKind.MEDIAPIPE, False),
    ModelSpec("blazepose-heavy", Capability.POSE, BackendKind.MEDIAPIPE, False),
    ModelSpec("efficientpose", Capability.POSE, BackendKind.ONNX, False),
    # Hands
    ModelSpec("handtrack", Capability.HAND, BackendKind.MEDIAPIPE, True),
    ModelSpec("handdetect", Capability.HAND, BackendKind.MEDIAPIPE, True),
    ModelSpec("handskeleton", Capability.HAND, BackendKind.MEDIAPIPE, True),
    ModelSpec("handlandmark-full", Capability.HAND, BackendKind.MEDIAPIPE, True),
    ModelSpec("handlandmark-lite", Capability.HAND, BackendKind.MEDIAPIPE, True),
    ModelSpec("handlandmark-sparse", Capability.HAND, BackendKind.MEDIAPIPE, False),
    # Face
    ModelSpec("blazeface", Capability.FACE, BackendKind.MEDIAPIPE, True),
    ModelSpec("blazeface-back", Capability.FACE, BackendKind.MEDIAPIPE, False),
    ModelSpec("blazeface-front", Capability.FACE, BackendKind.MEDIAPIPE, False),
    ModelSpec("facemesh", Capability.FACE, BackendKind.MEDIAPIPE, True),
    ModelSpec("facemesh-attention", Capability.FACE, BackendKind.MEDIAPIPE, False),
    ModelSpec("iris", Capability.FACE, BackendKind.MEDIAPIPE, True),
    ModelSpec("faceres", Capability.FACE, BackendKind.ONNX, False),
    ModelSpec("faceres-deep", Capability.FACE, BackendKind.ONNX, False),
    ModelSpec("faceboxes", Capability.FACE, BackendKind.ONNX, False),
    ModelSpec("mobileface", Capability.FACE, BackendKind.ONNX, False),
    ModelSpec("mobilefacenet", Capability.FACE, BackendKind.ONNX, False),
    # Attributes
    ModelSpec("emotion", Capability.ATTRIBUTE, BackendKind.ONNX, False),
    ModelSpec("age", Capability.ATTRIBUTE, BackendKind.ONNX, False),
    ModelSpec("gender", Capability.ATTRIBUTE, BackendKind.ONNX, False),
    ModelSpec("gender-ssrnet-imdb", Capability.ATTRIBUTE, BackendKind.ONNX, False),
    # Segmentation
    ModelSpec("selfie", Capability.SEGMENTATION, BackendKind.MEDIAPIPE, True),
    ModelSpec("meet", Capability.SEGMENTATION, BackendKind.MEDIAPIPE, False),
    ModelSpec("rvm", Capability.SEGMENTATION, BackendKind.ONNX, False),
    # Objects
    ModelSpec("mb3-centernet", Capability.OBJECT, BackendKind.ONNX, False),
    ModelSpec("nanodet", Capability.OBJECT, BackendKind.ONNX, False),
    # Liveness
    ModelSpec("liveness", Capability.LIVENESS, BackendKind.ONNX, False),
    ModelSpec("antispoof/realfake", Capability.LIVENESS, BackendKind.ONNX, False),
]


REGISTRY = ModelRegistry(SPEC_LIST)
