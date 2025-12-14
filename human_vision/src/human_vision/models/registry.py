from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.types import BackendKind, Capability, ModelDownload


@dataclass
class ModelSpec:
    model_id: str
    capability: Capability
    backend_kind: BackendKind
    enabled_by_default: bool = True
    download: Optional[ModelDownload] = None
    input_size: tuple[int, int] | None = None
    notes: Optional[str] = None
    license: Optional[str] = None


class ModelRegistry:
    def __init__(self, specs: List[ModelSpec]):
        self._specs: Dict[str, ModelSpec] = {spec.model_id: spec for spec in specs}

    def list_models(self) -> List[ModelSpec]:
        return list(self._specs.values())

    def get(self, model_id: str) -> ModelSpec:
        return self._specs[model_id]

    def resolve_default(self, capability: Capability) -> Optional[ModelSpec]:
        for spec in self._specs.values():
            if spec.capability == capability and spec.enabled_by_default:
                return spec
        return None

    def resolve_by_capability(
        self, capability: Capability, preferred_model_id: Optional[str] = None
    ) -> Optional[ModelSpec]:
        if preferred_model_id:
            return self._specs.get(preferred_model_id)
        return self.resolve_default(capability)
