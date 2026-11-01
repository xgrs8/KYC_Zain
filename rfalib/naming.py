"""Naming, counters, batches and splits (plan 6.6).

- Per-type counters live in <workspace>/state.json and NEVER reset
  (R14); ID_01 two-digit zero padding, unpadded from 100 (ID_100).
- Batch folders: generated/<YYYY-MM-DD_NN>/ (own counter).
- split in {train, val, test} = deterministic bucket of sha1(case_id ?? doc_id)
  at 80/10/10 (D5).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import threading
from pathlib import Path

from .constants import DOC_PREFIX, DOC_TYPES, PAD_WIDTH, SPLITS

_LOCK = threading.Lock()

_SOURCE_NAME_RE = re.compile(r"^(ID|TAX|LIC)_(\d+)")


# ---------------------------------------------------------------- counters

def _load_state(workspace: Path) -> dict:
    state_file = workspace / "state.json"
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    data.setdefault("next", {})
    for t in DOC_TYPES:
        data["next"].setdefault(t, 1)
    data.setdefault("batch", {})
    data.setdefault("batch_next", 1)
    return data


def _save_state(workspace: Path, data: dict) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    tmp = workspace / "state.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(workspace / "state.json")


def doc_name(doc_type: str, number: int) -> str:
    """ID_01 / TAX_07 / LIC_100 - two-digit padding, unpadded from 100."""
    return f"{DOC_PREFIX[doc_type]}_{number:0{PAD_WIDTH}d}" if number < 100 \
        else f"{DOC_PREFIX[doc_type]}_{number}"


def peek_next(workspace: Path, doc_type: str) -> int:
    return _load_state(workspace)["next"][doc_type]


def take_next(workspace: Path, doc_type: str) -> int:
    """Reserve the next counter for a doc type (advances state.json)."""
    with _LOCK:
        data = _load_state(workspace)
        n = data["next"][doc_type]
        data["next"][doc_type] = n + 1
        _save_state(workspace, data)
    return n


def reset_counters(workspace: Path, doc_type: str | None = None) -> None:
    """Reset counters to 1 (deliberate UI action - never automatic)."""
    with _LOCK:
        data = _load_state(workspace)
        for t in ([doc_type] if doc_type else DOC_TYPES):
            data["next"][t] = 1
        _save_state(workspace, data)


def set_counter(workspace: Path, doc_type: str, value: int) -> None:
    with _LOCK:
        data = _load_state(workspace)
        data["next"][doc_type] = max(1, int(value))
        _save_state(workspace, data)


# ---------------------------------------------------------------- batches

def next_batch_name(workspace: Path) -> str:
    """generated/<YYYY-MM-DD_NN>/ - one counter per day, never reset."""
    with _LOCK:
        data = _load_state(workspace)
        today = _dt.date.today().isoformat()
        data["batch"][today] = data["batch"].get(today, 0) + 1
        _save_state(workspace, data)
        return f"{today}_{data['batch'][today]:02d}"


# ---------------------------------------------------------------- splits (D5)

def split_for(entity_id: str) -> str:
    """Deterministic 80/10/10 bucket of sha1(entity_id).

    All variants of one document and all documents of one case share the
    same entity_id, hence the same split.
    """
    digest = hashlib.sha1(entity_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return SPLITS[0]
    if bucket < 90:
        return SPLITS[1]
    return SPLITS[2]


# ---------------------------------------------------------------- A.L.G naming

def derive_output_name(source_name: str, doc_type: str, workspace: Path) -> str:
    """A.L.G output base name (6.6).

    When the uploaded source matches ^(ID|TAX|LIC)_\\d+ the merged output
    reuses that stem (ID_01_front.png + ID_01_back.png -> ID_01); otherwise
    the chosen type's next counter is taken.
    """
    m = _SOURCE_NAME_RE.match(source_name)
    if m and m.group(1) == doc_type:
        return doc_name(doc_type, int(m.group(2)))
    n = take_next(workspace, doc_type)
    return doc_name(doc_type, n)


def source_matches_type(source_name: str, doc_type: str) -> bool:
    m = _SOURCE_NAME_RE.match(source_name)
    return bool(m and m.group(1) == doc_type)
