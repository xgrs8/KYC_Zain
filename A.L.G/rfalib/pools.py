"""Name / value pools (plan 6.2, protocol B4 and rule 10).

Pools are loaded from RFG/data/names/*.txt (one entry per line, `#`
comments allowed), deduplicated via the 6.7 normalization, and screened
against data/blocklist.txt - a blocklist hit raises (fail-closed).
Provenance is documented in data/names/SOURCES.md.
"""
from __future__ import annotations

import random
from pathlib import Path

from .constants import POOL_FILES
from .normalize import normalize


class Pools:
    """Validated, deduplicated, blocklist-screened value pools."""

    KEYS = ("given_m", "given_f", "surnames", "companies", "activities",
            "purposes", "cities", "areas", "streets", "authorities",
            "company_types")

    def __init__(self, values: dict[str, list[str]],
                 blocklist: list[str], pool_dir: Path):
        self._values = values
        self.blocklist = blocklist
        self.pool_dir = Path(pool_dir)

    # -- access -----------------------------------------------------------
    def __getattr__(self, key: str) -> list[str]:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self.__dict__["_values"][key]
        except KeyError as exc:
            raise AttributeError(f"no pool named {key!r}") from exc

    def get(self, key: str) -> list[str]:
        return self._values[key]

    def pick(self, rng: random.Random, key: str) -> str:
        return rng.choice(self._values[key])

    def __repr__(self) -> str:  # pragma: no cover
        sizes = {k: len(v) for k, v in self._values.items()}
        return f"Pools({sizes})"


def _read_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_pools(pool_dir: str | Path,
               blocklist_file: str | Path | None = None) -> Pools:
    """Load, deduplicate and screen every pool (fail-closed on problems)."""
    pool_dir = Path(pool_dir)
    if blocklist_file is None:
        blocklist_file = pool_dir.parent / "blocklist.txt"
    blocklist_file = Path(blocklist_file)

    block_norms: set[str] = set()
    if blocklist_file.exists():
        for entry in _read_lines(blocklist_file):
            block_norms.update({normalize(entry), normalize(entry).replace(" ", "")})

    values: dict[str, list[str]] = {}
    problems: list[str] = []
    for fname in POOL_FILES:
        key = fname[:-4]
        path = pool_dir / fname
        if not path.exists():
            problems.append(f"missing pool file: {path}")
            values[key] = []
            continue
        seen: set[str] = set()
        entries: list[str] = []
        for entry in _read_lines(path):
            n = normalize(entry)
            if not n:
                continue
            if n in seen:  # duplicate-pool detection (6.7)
                continue
            if n in block_norms or n.replace(" ", "") in block_norms:
                problems.append(f"blocklist hit in {fname}: {entry!r}")
                continue
            seen.add(n)
            entries.append(entry)
        values[key] = entries

    empty = [k for k, v in values.items() if not v]
    if problems or empty:
        raise ValueError(
            "pool validation failed: " + "; ".join(problems +
                [f"pool {k!r} is empty" for k in empty]))
    return Pools(values, sorted(block_norms), pool_dir)
