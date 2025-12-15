from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

from ..core.errors import MissingDependencyError


def optional_import(name: str, package: Optional[str] = None) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"Optional dependency '{name}' is missing. Install via pip install human_vision[{package or name}]"
        ) from exc
