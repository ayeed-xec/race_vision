from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.types import Capability


@dataclass
class Stage:
    name: str
    capability: Capability
    runner: Callable[[], object]

    def __call__(self) -> object:
        return self.runner()
