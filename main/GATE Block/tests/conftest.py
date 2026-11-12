"""Shared fixtures. Photos are simulated from the real reference templates (``gate_block/synth.py``):
a fresh fictional variant of the template is photographed on a background with the requested defect."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate_block import Gate, synth  # noqa: E402
from gate_block.laya_client import DECISIONS, QUESTIONS  # noqa: E402
from gate_block.quality import FieldQuality, MetricValue, SideQuality  # noqa: E402
from gate_block.recognition import RecognitionResult  # noqa: E402
from gate_block.report import DocumentAnalysis  # noqa: E402
from gate_block.types import DEFECTS  # noqa: E402

TEMPLATE_KEYS = ("national_id/front", "national_id/back", "tax_card/single", "business_license/single")
_PHOTOS: Dict[tuple, np.ndarray] = {}


@pytest.fixture(scope="session")
def gate() -> Gate:
    """The gate with the shipped config and templates, deterministic rules only (no Laya)."""
    return Gate()


@pytest.fixture(scope="session")
def pack(gate):
    return gate.pack


def make_photo(gate: Gate, key: str = "tax_card/single", seed: int = 3, variant_seed: Optional[int] = None, **params) -> np.ndarray:
    """A simulated phone photo (BGR) of a fictional variant of template ``key``; cached for the session."""
    cache_key = (key, seed, variant_seed, tuple(sorted((k, repr(v)) for k, v in params.items())))
    if cache_key not in _PHOTOS:
        doc_type, side = key.split("/")
        t = gate.pack.get(doc_type, side)
        variant = synth.variant_from_template(t.bgr, {k: f.box for k, f in t.fields.items()}, seed if variant_seed is None else variant_seed)
        if t.size[1] > t.size[0]:
            params.setdefault("frame", (1200, 1600))
        _PHOTOS[cache_key] = synth.capture(variant, synth.CaptureParams(seed=seed, **params))[0]
    return _PHOTOS[cache_key].copy()


@pytest.fixture(scope="session")
def photo(gate):
    def _photo(key: str = "tax_card/single", seed: int = 3, **params) -> np.ndarray:
        return make_photo(gate, key, seed, **params)
    return _photo


# ---------------------------------------------------------------------- synthetic analyses (no images)
def fake_side(side: str = "single", doc_type: str = "tax_card", status: str = "recognized", reference: str = "template",
              confidence: float = 0.9, fields: Optional[Dict[str, FieldQuality]] = None, glare_spread: float = 0.0,
              **severities: str):
    """A SideQuality + RecognitionResult with the given defect severities (everything else "none")."""
    unknown = set(severities) - set(DEFECTS)
    assert not unknown, f"unknown defects {unknown}"
    metrics = {d: MetricValue(0.0, severities.get(d, "none")) for d in DEFECTS}
    metrics["glare"].extra["spread"] = glare_spread
    sq = SideQuality(key=f"{doc_type}/{side}", doc_type=doc_type, side=side, reference=reference, recognition_status=status,
                     match_confidence=confidence, metrics=metrics, fields=dict(fields or {}), geometry={})
    rec = RecognitionResult(status=status, doc_type=doc_type, side=side, confidence=confidence, best=None, runner_up=None)
    return sq, rec


def fake_field(name: str, critical: bool = True, **causes: str) -> FieldQuality:
    """A field with the given causes, e.g. ``fake_field("taxpayer_number", glare="severe")``."""
    if not causes:
        return FieldQuality(name, critical, "good", "none", None, {}, {})
    worst = max(causes.values(), key=("none", "mild", "moderate", "severe").index)
    cause = next(c for c, s in causes.items() if s == worst)
    status = "outside" if cause == "cut_off" and worst in ("moderate", "severe") else "poor" if worst in ("moderate", "severe") else "fair"
    return FieldQuality(name, critical, status, worst, cause, dict(causes), {})


def fake_analysis(fields: Optional[Dict[str, FieldQuality]] = None, status: str = "recognized", reference: str = "template",
                  confidence: float = 0.9, glare_spread: float = 0.0, **severities: str) -> DocumentAnalysis:
    """A one-sided tax-card analysis with the given defect severities."""
    sq, rec = fake_side(status=status, reference=reference, confidence=confidence, fields=fields, glare_spread=glare_spread, **severities)
    return DocumentAnalysis("tax_card", {"single": sq}, {"single": rec})


# ---------------------------------------------------------------------- a stand-in for the real Laya agent
class FakeLayaAgent:
    """Answers with the same structure as ``laya.load(...).predict(state, questions)`` (laya 0.4.31).

    ``choose(state_text) -> {question_id: label}`` decides the labels; the rest of the probability mass is
    spread evenly. ``usage`` and the vegaml-engine extras can be overridden to exercise the client."""

    def __init__(self, choose=None, confidence: float = 0.9, usage=None, extras=None, fail: Optional[Exception] = None):
        self.choose = choose or (lambda state: {"decision": "accept", "reject_reason": "other", "docres_task": "deblurring"})
        self.confidence = confidence
        self.usage = usage or {}
        self.extras = extras or {}
        self.fail = fail
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.fail is not None:
            raise self.fail
        picks = self.choose(state)
        answers = {}
        for qid, spec in questions.items():
            labels = list(spec["criteria"])
            pick = picks.get(qid, labels[0])
            rest = (1.0 - self.confidence) / (len(labels) - 1)
            probs = {label: (self.confidence if label == pick else rest) for label in labels}
            answers[qid] = dict({"type": "choice", "choice": pick, "probabilities": probs, "confidence": self.confidence,
                                 "answer_confidence": self.confidence, "action": pick}, **self.extras.get(qid, {}))
        usage = dict({"input_tokens": 120, "state_tokens": 40, "state_tokens_dropped": 0, "truncated": False,
                      "truncated_questions": []}, **self.usage)
        return {"model": "fake-laya", "answers": answers, "usage": usage}

    system_one = predict


assert tuple(QUESTIONS["decision"]["criteria"]) == DECISIONS


# ---------------------------------------------------------------------- measured photos (cached for the session)
_MEASURED: Dict[tuple, tuple] = {}


@pytest.fixture(scope="session")
def measured(gate):
    """``measured(key, **capture_params) -> (recognition, side_quality)`` for a simulated photo."""
    from gate_block.imageio import load_image
    from gate_block.quality import analyze_side
    from gate_block.recognition import recognize

    def _measured(key: str = "tax_card/single", seed: int = 3, **params):
        cache_key = (key, seed, tuple(sorted((k, repr(v)) for k, v in params.items())))
        if cache_key not in _MEASURED:
            loaded = load_image(make_photo(gate, key, seed, **params), gate.cfg.image)
            rec = recognize(loaded, gate.pack, gate.cfg)
            _MEASURED[cache_key] = (rec, analyze_side(loaded, rec, gate.pack, gate.cfg))
        return _MEASURED[cache_key]
    return _measured
