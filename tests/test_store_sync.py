import io
import zipfile
from pathlib import Path

import pytest

from human_vision.models.store import ModelStore
from human_vision.models.sync import sync_repo_models
from human_vision.models.specs_builtin import REGISTRY
from human_vision.core.types import ModelDownload


class DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code


def test_model_store_ensure_download(monkeypatch, tmp_path):
    spec = REGISTRY.get("movenet-lightning")
    spec = type("Spec", (), spec.__dict__.copy())()
    spec.download = ModelDownload(type="tfhub", url="http://example.com")
    payload = b"fake-tflite"

    def fake_get(url, timeout=10):  # noqa: ARG001
        return DummyResponse(payload)

    monkeypatch.setattr("requests.get", fake_get)
    store = ModelStore(base_dir=tmp_path)
    path = store.ensure(spec)
    assert (path / "model.tflite").read_bytes() == payload


def test_sync_repo_models(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("repo-main/models/model.txt", "hello")
    buf.seek(0)

    def fake_get(url, stream=True, timeout=10):  # noqa: ARG001
        return DummyResponse(buf.getvalue())

    monkeypatch.setattr("requests.get", fake_get)
    target = sync_repo_models("https://github.com/example/repo", base_dir=tmp_path)
    assert (target / "models" / "model.txt").read_text() == "hello"
    assert (target / "manifest.json").exists()
