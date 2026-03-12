from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path


def _normalize_windows_path(raw_path: Path) -> str:
    path_text = str(raw_path)
    if os.name == "nt" and path_text.startswith("\\\\?\\"):
        return path_text[4:]
    return path_text


def _workspace_mkdtemp(suffix: str | None = None, prefix: str | None = None, dir: str | None = None) -> str:
    base_dir = Path(dir or tempfile.gettempdir())
    base_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = prefix or "tmp"
    name_suffix = suffix or ""

    for _ in range(128):
        candidate = base_dir / f"{name_prefix}{uuid.uuid4().hex[:8]}{name_suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return str(candidate)
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not allocate a temporary directory under {base_dir}")


_REPO_ROOT = Path(_normalize_windows_path(Path(__file__).resolve().parent))
_PYTEST_TMPDIR = _REPO_ROOT / ".tmp" / "pytest-temp"
_PYTEST_TMPDIR.mkdir(parents=True, exist_ok=True)

for _env_key in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_env_key] = str(_PYTEST_TMPDIR)

tempfile.tempdir = str(_PYTEST_TMPDIR)
tempfile.mkdtemp = _workspace_mkdtemp
