"""Persistent state of the MCP server (``<workspace>/alg_mcp_state.json``).

This file is separate from the desktop app's own ``alg_queue.json`` so the two can never overwrite
each other. All writes go through ``Store.tx()``: one lock (thread + cross-process), load, modify,
atomic save.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

STATE_VERSION = 1


class StateLockTimeout(RuntimeError):
    pass


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 20.0, stale: float = 60.0) -> Iterator[None]:
    """Cross-process lock using an exclusively created lock file (works on Windows and Linux)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale:   # a crashed process left it behind
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - start > timeout:
                raise StateLockTimeout(f"could not lock {lock_path} within {timeout}s")
            time.sleep(0.02)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def make_sid(doc_type: str, name: str, front: str, back: str | None) -> str:
    h = hashlib.sha1(f"{doc_type}|{name}|{front}|{back or ''}".encode("utf-8")).hexdigest()[:8]
    return f"{doc_type}-{h}"


def new_draft() -> dict:
    return {"fields": {}, "reasons": {}, "notes": {}, "pending_format": {}, "exceptions": {},
            "conditions": [], "views": {}, "zoom_views": 0, "updated": time.time()}


def new_sample(doc_type: str, name: str, front: str, back: str | None) -> dict:
    return {
        "sid": make_sid(doc_type, name, front, back), "doc_type": doc_type, "name": name,
        "front": front, "back": back, "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "waiting",            # waiting | done | skipped | needs_human
        "claims": {},                   # agent -> {"at": t, "expires": t}
        "drafts": {},                   # agent -> new_draft()
        "submissions": [],              # finished attempts, one per agent
        "disagreements": [],
        "result": None, "skip": None, "last_error": None,
    }


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock_path = self.path.with_suffix(".lock")
        self._tlock = threading.RLock()
        self._depth = 0
        self._state: dict | None = None

    # ------------------------------------------------------------------ io
    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                data.setdefault("samples", {})
                return data
            except (json.JSONDecodeError, OSError):
                backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
                try:
                    self.path.replace(backup)       # keep the broken file for inspection
                except OSError:
                    pass
        return {"version": STATE_VERSION, "samples": {}}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ api
    @contextmanager
    def tx(self) -> Iterator[dict]:
        """Locked read-modify-write. Re-entrant inside one thread."""
        with self._tlock:
            if self._depth > 0:
                yield self._state            # nested call: reuse the open state
                return
            with _file_lock(self._lock_path):
                self._state = self._load()
                self._depth = 1
                try:
                    yield self._state
                    self._save(self._state)
                finally:
                    self._depth = 0
                    self._state = None

    def snapshot(self) -> dict:
        """Read-only view (no lock needed: writes are atomic replaces)."""
        with self._tlock:
            return self._load()
