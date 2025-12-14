"""Minimal requests shim for offline testing."""
from __future__ import annotations

from typing import Any


class Response:
    def __init__(self, content: bytes = b"", status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def iter_content(self, chunk_size: int = 1024):  # pragma: no cover - unused in tests
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


def get(url: str, *args: Any, **kwargs: Any) -> Response:  # pragma: no cover - monkeypatched in tests
    raise RuntimeError("requests.get is not available in offline stub")


__all__ = ["get", "Response"]
