"""App settings IO for A.L.G (same workspace settings.json as R.F.G)."""
from __future__ import annotations

import json
from pathlib import Path

from rfalib.constants import DEFAULT_SETTINGS, default_workspace


def settings_path(workspace: Path | None = None) -> Path:
    base = Path(workspace) if workspace else default_workspace()
    return base / "settings.json"


def load_settings() -> dict:
    path = settings_path()
    data = dict(DEFAULT_SETTINGS)
    data.setdefault("labeler", "")
    if path.exists():
        try:
            data.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass  # corrupted settings -> defaults (fail-open, data survives)
    return data


def save_settings(settings: dict) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def resolve_workspace(settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    return Path(settings.get("workspace") or default_workspace())


def ensure_workspace(settings: dict | None = None) -> Path:
    from rfalib.constants import WORKSPACE_SUBDIRS
    ws = resolve_workspace(settings)
    ws.mkdir(parents=True, exist_ok=True)
    for sub in WORKSPACE_SUBDIRS:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    return ws
