from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, Optional


class Closeable(AbstractContextManager):
    """Base context manager to ensure cleanup."""

    def __init__(self, closer: Optional[Callable[[], None]] = None):
        self._closer = closer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        if self._closer:
            self._closer()
