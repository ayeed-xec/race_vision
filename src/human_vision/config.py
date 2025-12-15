from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from .core.types import Capability


class CapabilityConfig(BaseModel):
    enabled: bool = True
    model_id: Optional[str] = None
    fps_limit: Optional[float] = None
    every_n_frames: Optional[int] = None


class RuntimeConfig(BaseModel):
    mode: str = Field("threaded", description="sequential|threaded|multiprocess")
    max_workers: int = 4


class FrameDiffConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.01


class SmoothingConfig(BaseModel):
    enabled: bool = True
    alpha: float = 0.5


class BackendConfig(BaseModel):
    onnx_intra_op_threads: Optional[int] = None
    tflite_num_threads: Optional[int] = None


class VisionConfig(BaseModel):
    capabilities: Dict[Capability, CapabilityConfig] = Field(
        default_factory=lambda: {cap: CapabilityConfig() for cap in Capability}
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    frame_diff: FrameDiffConfig = Field(default_factory=FrameDiffConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    model_config = {
        "use_enum_values": True,
    }
