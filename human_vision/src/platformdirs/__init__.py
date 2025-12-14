"""Minimal platformdirs shim for offline testing."""
from __future__ import annotations

from pathlib import Path

def user_cache_dir(appname: str = "", appauthor: str | None = None) -> str:
    base = Path("/tmp") / appname
    base.mkdir(parents=True, exist_ok=True)
    return str(base)

__all__ = ["user_cache_dir"]
