"""Sidecar and JSON IO (plan 6.5, D12).

All JSON files are written UTF-8, ensure_ascii=False, no BOM, "\\n"
newlines, 2-space indent - and round-trip byte-exact (T-SCHEMA).

Truth sidecar  (<doc>.truth.json): source values printed on the image.
Label sidecar  (<doc>.label.sidecar.json): labeling metadata.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

from .constants import (
    APP_VERSION,
    CONDITION_TAGS,
    DOC_TYPES,
    FIELD_STATUS,
    RFALIB_VERSION,
    SPLITS,
)

_BOM = "\ufeff"
_FORBIDDEN_FRAGMENTS = (  # placeholder scanning for CI-side hygiene
    "TODO", "XXX", "TBD", "FIXME", "undefined", "NaN", "【",
)


# ---------------------------------------------------------------- JSON IO

def dumps(obj: Any) -> str:
    """Canonical JSON text: UTF-8, unescaped Arabic, 2-space indent, \\n."""
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def write_json(path: Path, obj: Any) -> None:
    """Atomic canonical write; no BOM ever."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(dumps(obj).encode("utf-8"))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"BOM found in {path} - sidecars must be BOM-free")
    return json.loads(raw.decode("utf-8"))


def check_placeholders(obj: Any) -> list[str]:
    """Fail-closed scan for leftover placeholders in string values."""
    found: list[str] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")
        elif isinstance(node, str):
            for frag in _FORBIDDEN_FRAGMENTS:
                if frag in node:
                    found.append(f"{where}: contains {frag!r}")

    walk(obj, "$")
    return found


# ---------------------------------------------------------------- builders

def _now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def truth_sidecar(*, doc_id: str, doc_type: str, case_id: str | None,
                  batch: str, seed: str, doc_code: str, split: str,
                  printed: dict[str, Any],
                  intended: dict[str, Any] | None = None,
                  defects: list[str] | None = None,
                  merge: dict[str, Any] | None = None,
                  files: list[str] | None = None) -> dict[str, Any]:
    """Build the truth sidecar object (6.5)."""
    if split not in SPLITS:
        raise ValueError(f"bad split {split!r}")
    if doc_type not in DOC_TYPES:
        raise ValueError(f"bad doc type {doc_type!r}")
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "case_id": case_id,
        "batch": batch,
        "seed": seed,
        "synthetic": True,
        "doc_code": doc_code,
        "split": split,
        "printed": printed,
        "intended": intended or {},
        "defects": defects or [],
        "merge": merge or {},
        "files": files or [],
        "generated_at": _now_iso(),
        "generator": f"rfalg {APP_VERSION} / rfalib {RFALIB_VERSION}",
    }


def label_sidecar(*, doc_id: str, photo: str, doc_type: str, labeler: str,
                  field_status: dict[str, str],
                  truth_disagreements: list[dict[str, str]] | None = None,
                  conditions: list[str] | None = None,
                  truth_ref: str | None = None, split: str | None = None,
                  synthetic: bool = False,
                  skipped: str | None = None,
                  warnings_overridden: list[str] | None = None) -> dict[str, Any]:
    """Build the label sidecar object (6.5)."""
    for key, status in field_status.items():
        if status not in FIELD_STATUS:
            raise ValueError(f"bad field status {status!r} for {key}")
    for tag in conditions or []:
        if tag not in CONDITION_TAGS:
            raise ValueError(f"bad condition tag {tag!r}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"bad split {split!r}")
    out = {
        "doc_id": doc_id,
        "photo": photo,
        "doc_type": doc_type,
        "labeler": labeler,
        "labeled_at": _now_iso(),
        "field_status": field_status,
        "truth_disagreements": truth_disagreements or [],
        "conditions": conditions or [],
        "truth_ref": truth_ref,
        "split": split,
        "synthetic": synthetic,
    }
    if skipped is not None:
        out["skipped"] = skipped
    if warnings_overridden:
        out["warnings_overridden"] = warnings_overridden
    return out


# ---------------------------------------------------------------- lookup

def find_truth_sidecar(image_path: Path) -> Path | None:
    """Find the truth sidecar associated with a source image.

    For ID_01_front.png / ID_01_back.png looks for ID_01.truth.json;
    for TAX_01.png looks for TAX_01.truth.json.
    """
    p = Path(image_path)
    m = re.match(r"^((?:ID|TAX|LIC)_\d+)", p.name)
    stem = m.group(1) if m else p.stem
    for cand in (p.parent / f"{stem}.truth.json",
                 p.with_name(stem).with_suffix(".truth.json")):
        if cand.exists():
            return cand
    return None


def printed_values(truth: dict[str, Any]) -> dict[str, str]:
    """The `printed` map of a truth sidecar as plain str dict."""
    return {k: str(v) for k, v in (truth.get("printed") or {}).items()}
