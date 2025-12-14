from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.types import Capability
from ..models.registry import ModelSpec


@runtime_checkable
class IModelAdapter(Protocol):
    thread_safe: bool

    def warmup(self) -> None: ...

    def infer(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None): ...

    def close(self) -> None: ...


@runtime_checkable
class IBackend(Protocol):
    def build_adapter(self, spec: ModelSpec, store: "ModelStore") -> IModelAdapter: ...


@runtime_checkable
class IScheduler(Protocol):
    def run(self, tasks: dict[str, callable]) -> dict[str, object]: ...

    def close(self) -> None: ...


class StageExecution(Protocol):
    capability: Capability

    def __call__(self) -> object: ...
