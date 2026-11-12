"""The contract between the Gate Block and Laya, and the backends that answer it.

Laya is a *decision model*: it does not write text. It is handed a state (the compact JSON built by
``laya_state.py``) and a fixed set of typed questions, and returns a probability for every option of every
question in one forward pass. This module defines those questions and wraps whatever answers them behind
one small interface::

    backend.decide(state_json, QUESTIONS) -> BackendAnswer

Backends: :class:`LayaBackend` (the real model, through the ``laya`` package) and :class:`StaticBackend`
(tests, dry runs). With no backend at all the router uses the deterministic rules.

Checked against laya 0.4.31: ``laya.load(path_or_id)`` returns an agent; ``agent.predict(state, questions)``
returns ``{"answers": {qid: {"choice", "probabilities", ...}}, "usage": {"truncated", ...}}``; every
question gets its own token budget for its options, so three questions do not compete for space.

IMPORTANT: the options of a ``choice`` question are positional for Laya. Never reorder or rename the
labels below once Laya has been fine-tuned on them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .types import DOCRES_TASKS, REJECT_REASONS

QUESTIONS: Dict[str, Dict[str, Any]] = {
    "decision": {
        "type": "choice",
        "instructions": "What should happen to this document photo?",
        "criteria": {
            "accept": "readable as it is, send to OCR",
            "enhance": "has a defect that restoration can fix",
            "reject": "unusable, ask for a new photo",
        },
    },
    "reject_reason": {
        "type": "choice",
        "instructions": "If the photo must be retaken, what is the main reason?",
        "criteria": {
            "blur": "blurry",
            "glare": "light reflection hides text",
            "dark": "too dark",
            "overexposed": "too bright, text washed out",
            "shadow": "shadow over the document",
            "cut_off": "part of the document is outside the photo",
            "angle": "taken at a steep angle",
            "too_small": "document too far or too small",
            "obstructed": "a finger or object covers text",
            "other": "another quality problem",
        },
    },
    "docres_task": {
        "type": "choice",
        "instructions": "If the photo can be restored, which single restoration helps most?",
        "criteria": {
            "dewarping": "flatten curled or folded paper",
            "deshadowing": "remove a shadow",
            "appearance": "fix a dark or low-contrast photo",
            "deblurring": "sharpen a blurry photo",
            "binarization": "clean black-and-white text",
        },
    },
}
DECISIONS = ("accept", "enhance", "reject")
assert tuple(QUESTIONS["decision"]["criteria"]) == DECISIONS
assert tuple(QUESTIONS["reject_reason"]["criteria"]) == REJECT_REASONS
assert tuple(QUESTIONS["docres_task"]["criteria"]) == DOCRES_TASKS


class BackendError(RuntimeError):
    """The decision backend is unavailable or returned something unusable."""


def validate_questions(questions: Dict[str, Dict[str, Any]]) -> None:
    """Check a question set against the rules Laya enforces for ``choice`` questions."""
    if not questions:
        raise ValueError("no questions")
    for qid, spec in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("question ids must be non-empty strings")
        if not isinstance(spec, dict) or spec.get("type") != "choice":
            raise ValueError(f"{qid}: the gate only asks 'choice' questions")
        ins = spec.get("instructions")
        if not isinstance(ins, str) or not ins.strip():
            raise ValueError(f"{qid}: instructions must be a non-empty string")
        crit = spec.get("criteria")
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError(f"{qid}: criteria must be a dict of at least two label -> description entries")
        for label, desc in crit.items():
            if not isinstance(label, str) or not label.strip() or label != label.strip():
                raise ValueError(f"{qid}: bad option label {label!r}")
            if label.lower() in ("true", "false", "yes", "no"):
                raise ValueError(f"{qid}: boolean-word labels confuse Laya; use a meaningful label instead of {label!r}")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"{qid}.{label}: the description must be a non-empty string")


@dataclass
class BackendAnswer:
    probs: Dict[str, Dict[str, float]]          # question id -> option -> probability (each row sums to 1)
    unsure: bool = False                        # the backend itself says "do not rely on this" (abstain / review)
    note: str = ""
    raw: Optional[Dict[str, Any]] = None

    def top(self, qid: str) -> str:
        row = self.probs[qid]
        return max(row, key=row.get)


def normalise_answer(probs: Dict[str, Dict[str, float]], questions: Dict[str, Dict[str, Any]] = QUESTIONS) -> Dict[str, Dict[str, float]]:
    """Validate raw probabilities and normalise each row: every option present, finite, non-negative, sum > 0."""
    out: Dict[str, Dict[str, float]] = {}
    for qid, spec in questions.items():
        if not isinstance(probs.get(qid), dict):
            raise BackendError(f"the answer has no probabilities for question {qid!r}")
        row: Dict[str, float] = {}
        for label in spec["criteria"]:
            if label not in probs[qid]:
                raise BackendError(f"the answer to {qid!r} has no probability for option {label!r}")
            try:
                v = float(probs[qid][label])
            except (TypeError, ValueError) as exc:
                raise BackendError(f"the probability of {qid}.{label} is not a number") from exc
            if not math.isfinite(v) or v < 0:
                raise BackendError(f"invalid probability {v} for {qid}.{label}")
            row[label] = v
        total = sum(row.values())
        if total <= 0:
            raise BackendError(f"the probabilities for {qid!r} sum to zero")
        out[qid] = {k: v / total for k, v in row.items()}
    return out


class DecisionBackend(Protocol):
    name: str

    def decide(self, state_json: str, questions: Dict[str, Dict[str, Any]]) -> BackendAnswer: ...


def onehot(decision: str, reason: str = "other", task: str = "deblurring", confidence: float = 0.9) -> Dict[str, Dict[str, float]]:
    """Probabilities that put ``confidence`` on the given choices (for tests and dry runs)."""
    def row(labels, pick):
        rest = (1.0 - confidence) / (len(labels) - 1)
        return {label: (confidence if label == pick else rest) for label in labels}
    return {"decision": row(DECISIONS, decision), "reject_reason": row(REJECT_REASONS, reason),
            "docres_task": row(DOCRES_TASKS, task)}


class StaticBackend:
    """Returns configured probabilities: a dict, or a callable ``f(state_dict) -> dict``. A key ``"_unsure"``
    set to true in the returned dict marks the answer as one the backend does not stand behind."""

    name = "static"

    def __init__(self, answer):
        self._answer = answer
        self.calls: List[str] = []

    def decide(self, state_json: str, questions: Dict[str, Dict[str, Any]]) -> BackendAnswer:
        self.calls.append(state_json)
        raw = self._answer(json.loads(state_json)) if callable(self._answer) else self._answer
        if not isinstance(raw, dict):
            raise BackendError("the static backend must be given a dict of probabilities")
        probs = {k: v for k, v in raw.items() if not str(k).startswith("_")}
        return BackendAnswer(probs=normalise_answer(probs, questions), unsure=bool(raw.get("_unsure", False)), raw=raw)


class LayaBackend:
    """The real Laya model.

    ``model`` is a local checkpoint folder (the one holding ``rl_agent_config.json`` and
    ``model.safetensors``) or a Hugging Face id such as ``convaiinnovations/laya``. ``calibration`` is an
    optional calibration file saved by Laya. An already loaded agent can be passed as ``agent``.

    The state is sent as the exact JSON text produced by :func:`laya_state.state_to_json`, so it is the same
    text the model was fine-tuned on. A state that Laya had to truncate is refused: an answer about half a
    state is not an answer.
    """

    name = "laya"

    def __init__(self, model: str = "convaiinnovations/laya", device: Optional[str] = None,
                 calibration: Optional[str] = None, agent: Any = None):
        if agent is None:
            try:
                import laya  # type: ignore
            except ImportError as exc:
                raise BackendError("the 'laya' package is not installed (pip install laya)") from exc
            try:
                kwargs: Dict[str, Any] = {}
                if device:
                    kwargs["device"] = device
                if calibration:
                    kwargs["calibration"] = calibration
                agent = laya.load(model, **kwargs)
            except Exception as exc:
                raise BackendError(f"could not load Laya from {model!r}: {exc}") from exc
        self.agent = agent
        self.model = model

    def decide(self, state_json: str, questions: Dict[str, Dict[str, Any]]) -> BackendAnswer:
        try:
            out = self.agent.predict(state_json, questions)
        except Exception as exc:
            raise BackendError(f"Laya failed: {exc}") from exc
        if not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
            raise BackendError("unexpected Laya output: no 'answers'")
        usage = out.get("usage") or {}
        if usage.get("truncated"):
            raise BackendError(f"Laya truncated the state ({usage.get('state_tokens_dropped')} tokens dropped): "
                               "lower laya.max_state_chars")
        probs: Dict[str, Dict[str, float]] = {}
        unsure = False
        for qid, spec in questions.items():
            a = out["answers"].get(qid)
            if not isinstance(a, dict):
                raise BackendError(f"Laya returned nothing for question {qid!r}")
            row = a.get("probabilities")
            if not isinstance(row, dict):
                raise BackendError(f"Laya returned no probabilities for question {qid!r}")
            probs[qid] = {str(k): v for k, v in row.items()}
            if qid == "decision" and (a.get("abstain") is True or a.get("route") == "review"):
                unsure = True            # reported by the vegaml engine; absent on the plain checkpoints
        return BackendAnswer(probs=normalise_answer(probs, questions), unsure=unsure,
                             note="laya abstained / asked for review" if unsure else "", raw=out)
