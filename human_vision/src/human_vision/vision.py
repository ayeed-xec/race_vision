from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import VisionConfig
from .core.errors import HumanVisionError, MissingDependencyError, MissingModelError
from .core.lifecycle import Closeable
from .core.types import Capability
from .logging import get_logger
from .models.specs_builtin import REGISTRY
from .models.store import ModelStore
from .results import VisionResult
from .pipeline.runner import PipelineRunner
from .pipeline.stages import Stage
from .backends.base import BackendFactory


class Vision(Closeable):
    def __init__(self, config: Optional[VisionConfig] = None, store: Optional[ModelStore] = None):
        self.config = config or VisionConfig()
        self.store = store or ModelStore()
        self.registry = REGISTRY
        self.runner = PipelineRunner(self.config)
        self.logger = get_logger()
        super().__init__(self.close)

    def __enter__(self):
        return self

    def analyze(self, frame: np.ndarray, *, timestamp_ms: int | None = None) -> VisionResult:
        stages = []
        for capability, cap_conf in self.config.capabilities.items():
            if not cap_conf.enabled:
                continue
            stages.append(
                Stage(
                    name=capability.value,
                    capability=capability,
                    runner=lambda cap=capability: self._run_capability(cap, frame),
                )
            )
        outputs, debug = self.runner.run(stages)
        result = VisionResult(timestamp_ms=timestamp_ms, debug=debug)
        result.pose = outputs.get(Capability.POSE)
        result.face = outputs.get(Capability.FACE)
        result.hands = outputs.get(Capability.HAND)
        result.segmentation = outputs.get(Capability.SEGMENTATION)
        result.objects = outputs.get(Capability.OBJECT)
        return result

    def _run_capability(self, capability: Capability, frame: np.ndarray):
        spec = self.registry.resolve_by_capability(capability, self.config.capabilities[capability].model_id)
        if spec is None:
            return None
        backend = BackendFactory.get(spec.backend_kind)
        try:
            adapter = backend.build_adapter(spec, self.store)
            adapter.warmup()
            return adapter.infer(frame)
        except (MissingDependencyError, MissingModelError) as exc:
            self.logger.warning(str(exc))
            return None
        except HumanVisionError:
            raise

    def close(self) -> None:  # type: ignore[override]
        try:
            self.runner.close()
        except Exception:
            pass
