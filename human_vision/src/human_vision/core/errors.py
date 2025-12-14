from __future__ import annotations

class HumanVisionError(Exception):
    """Base error for the library."""


class MissingDependencyError(HumanVisionError):
    """Raised when an optional dependency is missing."""


class MissingModelError(HumanVisionError):
    """Raised when a model weight is missing from the local store."""

    def __init__(self, model_id: str, expected_path: str):
        super().__init__(f"Model '{model_id}' is missing. Expected at {expected_path}")
        self.model_id = model_id
        self.expected_path = expected_path


class DownloadError(HumanVisionError):
    """Raised when downloads fail."""
