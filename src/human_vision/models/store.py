from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import requests
from platformdirs import user_cache_dir

from ..core.errors import DownloadError, MissingModelError
from ..core.types import ModelDownload
from .registry import ModelSpec


class ModelStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(user_cache_dir("human_vision"))
        self.models_dir = self.base_dir / "models"
        self.repo_dir = self.base_dir / "repo_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.repo_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def is_present(self, model_id: str) -> bool:
        return self.get_path(model_id).exists()

    def ensure(self, spec: ModelSpec) -> Path:
        target = self.get_path(spec.model_id)
        if target.exists():
            return target
        if spec.download is None:
            raise MissingModelError(spec.model_id, str(target))
        self._download(spec.download, target)
        return target

    def _download(self, download: ModelDownload, target: Path) -> None:
        if download.type == "tfhub" or download.type == "github-zip":
            resp = requests.get(download.url, timeout=10)
            if resp.status_code != 200:
                raise DownloadError(f"Failed to download {download.url}")
            target.mkdir(parents=True, exist_ok=True)
            if download.type == "tfhub":
                with open(target / "model.tflite", "wb") as f:
                    f.write(resp.content)
            else:
                import io

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    members = [m for m in zf.namelist() if m.startswith(download.path_in_zip or "models/")]
                    for member in members:
                        if member.endswith("/"):
                            continue
                        dest = target / Path(member).name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(member))
        else:
            raise DownloadError(f"Unsupported download type {download.type}")
