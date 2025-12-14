"""Minimal Pydantic compatibility layer for offline testing.
This stub provides just enough surface area for the library configuration
models used in the test-suite. It intentionally omits validation and most
advanced features of real Pydantic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class FieldInfo:
    default: Any = None
    default_factory: Optional[Callable[[], Any]] = None
    description: Optional[str] = None


def Field(default: Any = None, *, description: str | None = None, default_factory: Callable[[], Any] | None = None):
    return FieldInfo(default=default, default_factory=default_factory, description=description)


class BaseModel:
    model_config: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # capture annotations for default creation
        cls.__field_defaults__ = {}
        for name, annotation in getattr(cls, "__annotations__", {}).items():
            value = getattr(cls, name, None)
            cls.__field_defaults__[name] = value

    def __init__(self, **data: Any):
        for name, default in getattr(self, "__field_defaults__", {}).items():
            if isinstance(default, FieldInfo):
                if name in data:
                    value = data[name]
                elif default.default_factory is not None:
                    value = default.default_factory()
                else:
                    value = default.default
            else:
                value = data.get(name, default)
            setattr(self, name, value)
        # accept extra keys without validation
        for name, value in data.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def model_dump(self) -> Dict[str, Any]:
        return self.__dict__.copy()


__all__ = ["BaseModel", "Field", "FieldInfo"]
