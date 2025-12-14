from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


class Box2D(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: Optional[float] = None
    label: Optional[str] = None

    model_config = {"frozen": True}


class Keypoint2D(BaseModel):
    x: float
    y: float
    score: Optional[float] = None
    name: Optional[str] = None

    model_config = {"frozen": True}


class FaceResult(BaseModel):
    boxes: List[Box2D] = Field(default_factory=list)
    mesh_landmarks: List[List[Keypoint2D]] = Field(default_factory=list)
    iris_landmarks: Optional[List[List[Keypoint2D]]] = None
    embedding: Optional[List[float]] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    liveness: Optional[str] = None
    antispoof: Optional[str] = None


class PosePerson(BaseModel):
    keypoints: List[Keypoint2D] = Field(default_factory=list)
    score: Optional[float] = None


class PoseResult(BaseModel):
    persons: List[PosePerson] = Field(default_factory=list)
    segmentation_mask: Optional[np.ndarray] = None

    model_config = {"arbitrary_types_allowed": True}


class HandInstance(BaseModel):
    landmarks: List[Keypoint2D] = Field(default_factory=list)
    handedness: Optional[str] = None
    score: Optional[float] = None


class HandResult(BaseModel):
    hands: List[HandInstance] = Field(default_factory=list)


class ObjectResult(BaseModel):
    boxes: List[Box2D] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    mask: Optional[np.ndarray] = None
    type: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class SchedulerDebug(BaseModel):
    mode: str
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    dropped_frames: int = 0
    notes: List[str] = Field(default_factory=list)


class VisionDebug(BaseModel):
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    scheduler: SchedulerDebug = Field(default_factory=lambda: SchedulerDebug(mode="sequential"))
    dropped_frames: int = 0
    notes: List[str] = Field(default_factory=list)


class VisionResult(BaseModel):
    timestamp_ms: Optional[int] = None
    frame_id: Optional[int] = None
    fps_estimate: Optional[float] = None
    face: Optional[FaceResult] = None
    pose: Optional[PoseResult] = None
    hands: Optional[HandResult] = None
    objects: Optional[ObjectResult] = None
    segmentation: Optional[SegmentationResult] = None
    debug: VisionDebug = Field(default_factory=VisionDebug)

    model_config = {"arbitrary_types_allowed": True}
