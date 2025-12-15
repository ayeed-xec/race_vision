from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..core.errors import DownloadError


def _repo_slug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path = parsed.path.rstrip("/")
    return path.split("/")[-1]


def sync_repo_models(repo_url: str, branch: str = "main", base_dir: Path | None = None) -> Path:
    slug = _repo_slug(repo_url)
    zip_url = f"{repo_url}/archive/{branch}.zip"
    resp = requests.get(zip_url, stream=True, timeout=10)
    if resp.status_code != 200:
        raise DownloadError(f"Failed to download repo archive from {zip_url}")
    base = Path(base_dir) if base_dir else Path.home() / ".cache" / "human_vision" / "repo_models"
    target_dir = base / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    content = resp.content
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{slug}-{branch}/models/")]
        extracted = []
        for member in members:
            if member.endswith("/"):
                continue
            rel = Path(member).relative_to(f"{slug}-{branch}/models")
            dest = target_dir / "models" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            extracted.append({"file": str(rel), "size": dest.stat().st_size})
    manifest = {"repo": repo_url, "branch": branch, "files": extracted}
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return target_dir
