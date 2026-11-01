"""A.L.G sample queue (plan 8.2): states, persistence, ordering.

States: Waiting / In progress / Done / Skipped (skip requires a reason,
stored in the label sidecar).  The queue persists in
<workspace>/alg_queue.json so a labeling session survives restarts.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rfalib.constants import QUEUE_STATES


@dataclass
class Sample:
    sample_id: str
    doc_type: str                 # ID | TAX | LIC
    name: str                     # display name / base stem
    front: str | None = None      # absolute path to the front image
    back: str | None = None       # absolute path to the back image (ID only)
    state: str = "Waiting"
    added_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    skip_reason: str | None = None
    output_name: str | None = None   # set when Done (e.g. ID_01)

    def matches_state(self, state: str) -> bool:
        return self.state == state


class QueueStore:
    """Persisted queue of labeling samples."""

    def __init__(self, workspace: Path):
        self.path = Path(workspace) / "alg_queue.json"
        self.samples: list[Sample] = []
        self.load()

    # ------------------------------------------------------------ persistence
    def load(self) -> None:
        self.samples = []
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for item in data.get("samples", []):
                    self.samples.append(Sample(**item))
            except (json.JSONDecodeError, TypeError, OSError):
                pass  # corrupted queue -> start fresh (fail-open)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"samples": [asdict(s) for s in self.samples]},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------ operations
    def add(self, doc_type: str, name: str, front: str,
            back: str | None = None) -> Sample:
        sample = Sample(sample_id=uuid.uuid4().hex[:12], doc_type=doc_type,
                        name=name, front=front, back=back)
        self.samples.append(sample)
        self.save()
        return sample

    def get(self, sample_id: str) -> Sample | None:
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        return None

    def set_state(self, sample: Sample, state: str, output_name: str | None = None,
                  skip_reason: str | None = None) -> None:
        if state not in QUEUE_STATES:
            raise ValueError(f"bad state {state!r}")
        sample.state = state
        if output_name is not None:
            sample.output_name = output_name
        if skip_reason is not None:
            sample.skip_reason = skip_reason
        self.save()

    def next_waiting(self) -> Sample | None:
        for s in self.samples:
            if s.state == "Waiting":
                return s
        return None

    def by_state(self, state: str) -> list[Sample]:
        return [s for s in self.samples if s.state == state]

    def clear_session(self) -> None:
        """Clear Current Session: drop non-Done entries (Done labels stay)."""
        self.samples = [s for s in self.samples if s.state == "Done"]
        self.save()
