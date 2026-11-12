"""Training data for Laya: the gate's decisions as rows Laya's trainer reads.

Why this exists
---------------
Laya's published checkpoints are general decision models. Asked the gate's three questions as they come,
the multilingual checkpoint names sensible retake reasons and DocRes tasks, but it does not judge
*severity*: it answered "accept" for almost every state, severely blurred or glared photos included. The
gate therefore lists in the state the branches the measurements permit (``allowed``, from
``rules.permitted``), so Laya's job is the judgement between them, and Laya has to be fine-tuned on the
gate's decision policy before it can make that judgement. The fine-tuning itself is done outside this
repository; this module makes its data.

The labels come from the deterministic rules (the same policy the gate uses without Laya), applied to
1. measurement states the gate really produced: simulated photos of the templates and your own photos, and
2. many synthetic measurement states (random combinations of defects and field problems), so every
   combination Laya can meet has been seen; ``--synthetic-enhance`` adds states where two or more DocRes
   tasks are permitted, where the choice of the task matters.
Every row is ``{"state", "questions", "expected", "meta"}``, the format ``laya.train.finetune`` reads. To
make Laya *better* than the rules later, label photos by what really happened downstream (did TeleOCR read
the photo correctly, did DocRes help) and train on those labels (``export-laya --labels``).

    python -m gate_block laya-data --out laya_data [--photos PHOTOS_DIR] [--simulate 300] [--synthetic 4000] [--synthetic-enhance 1700]
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..laya_client import QUESTIONS
from ..laya_state import build_state, state_to_json
from ..quality import FieldQuality, MetricValue, SideQuality
from ..recognition import RecognitionResult
from ..report import DocumentAnalysis
from ..rules import permitted, rules_decision
from ..types import DEFECTS, DOCRES_TASKS, SEV_RANK
from .export_laya_dataset import (
    _is_eval,
    expected_from,
    row_from_result,
    simulated_results,
)
from .samples import iter_samples, run_sample

_SEVS = ("mild", "moderate", "severe")
# the field cause a whole-document defect shows up as, in the fields it reaches
_FIELD_CAUSE = {"blur": "blur", "shadow": "shadow", "glare": "glare", "cut": "cut_off", "obstruction": "obstructed",
                "low_contrast": "low_contrast", "dark": "low_contrast"}


# ---------------------------------------------------------------------- synthetic measurement states
def _field(name: str, critical: bool, causes: Dict[str, str]) -> FieldQuality:
    if not causes:
        return FieldQuality(name, critical, "good", "none", None, {}, {})
    worst = max(causes.values(), key=lambda s: SEV_RANK[s])
    cause = next(c for c, s in causes.items() if s == worst)
    status = ("outside" if cause == "cut_off" and SEV_RANK[worst] >= 2 else
              "poor" if SEV_RANK[worst] >= 2 else "fair")
    return FieldQuality(name, critical, status, worst, cause, dict(causes), {})


def _near(rng: random.Random, sev: str) -> str:
    """A field severity close to the document's: the same, or one step either way."""
    i = _SEVS.index(sev) + rng.choice((-1, 0, 0, 1))
    return _SEVS[min(max(i, 0), 2)] if i >= 0 else "none"


def random_side(rng: random.Random, template, n_defects: Optional[int] = None) -> Tuple[SideQuality, RecognitionResult]:
    """One side with a random but plausible combination of defects and field problems."""
    sev = {d: "none" for d in DEFECTS}
    k = rng.choice((0, 0, 1, 1, 1, 1, 2, 2, 3)) if n_defects is None else n_defects
    for d in rng.sample(DEFECTS, k):
        if d == "curl" and template.rigid:
            continue
        sev[d] = rng.choices(_SEVS, weights=(0.45, 0.33, 0.22))[0]       # most real problems are small
    if rng.random() < 0.03:
        sev["blur"] = "unknown"
    spread = rng.choice((0.0, 0.05, 0.2, 0.7)) if sev["glare"] != "none" else 0.0
    metrics = {d: MetricValue(0.0 if s != "unknown" else None, s) for d, s in sev.items()}
    metrics["glare"].extra["spread"] = spread
    fields: Dict[str, FieldQuality] = {}
    for name, spec in template.fields.items():
        causes: Dict[str, str] = {}
        for d, cause in _FIELD_CAUSE.items():
            s = sev[d]
            if s in _SEVS and rng.random() < {"blur": 0.8, "dark": 0.3}.get(d, 0.35):
                fs = _near(rng, s)
                if fs != "none":
                    causes[cause] = fs if cause not in causes else max(causes[cause], fs, key=lambda x: SEV_RANK[x])
        if rng.random() < 0.04:                                    # a problem of this field alone
            causes[rng.choice(("blur", "glare", "shadow", "obstructed", "low_contrast", "cut_off"))] = rng.choice(_SEVS)
        fields[name] = _field(name, spec.critical, causes)
    sq = SideQuality(key=template.key, doc_type=template.doc_type, side=template.side, reference="template",
                     recognition_status="recognized", match_confidence=rng.uniform(0.6, 1.0), metrics=metrics,
                     fields=fields, geometry={})
    rec = RecognitionResult(status="recognized", doc_type=template.doc_type, side=template.side,
                            confidence=sq.match_confidence, best=None, runner_up=None)
    return sq, rec


def random_analysis(rng: random.Random, pack, cfg) -> DocumentAnalysis:
    doc_type = rng.choice(sorted({t.doc_type for t in pack.templates}))
    sides, recs = {}, {}
    templates = [t for t in pack.templates if t.doc_type == doc_type]
    clean_other = rng.random() < 0.5                                # two-sided: often only one side is bad
    for i, t in enumerate(templates):
        sq, rec = random_side(rng, t, 0 if (clean_other and i > 0) else None)
        sides[t.side] = sq
        recs[t.side] = rec
    return DocumentAnalysis(doc_type, sides, recs, field_loss_at=cfg.rules.field_loss_at)


def synthetic_rows(gate, n: int, seed: int = 0, enhance_choice: bool = False) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """``n`` rows from random measurement states, labelled by the deterministic rules.

    ``enhance_choice``: keep only states where the rules enhance and the measurements permit two or more
    DocRes tasks, i.e. where the choice of the task matters (rare among random states, so drawn on
    purpose); such rows are marked ``synthetic_enhance``."""
    rng = random.Random(seed)
    cfg = gate.cfg
    origin = "synthetic_enhance" if enhance_choice else "synthetic"
    made = tries = 0
    while made < n and tries < (n * 400 if enhance_choice else n):
        tries += 1
        analysis = random_analysis(rng, gate.pack, cfg)
        rnd = rng.choice((0, 0, 0, 0, 1, 1, 2)) if cfg.routing.max_enhance_rounds else 0
        rnd = min(rnd, cfg.routing.max_enhance_rounds)
        usual = [t for t in DOCRES_TASKS if t not in gate.tree.rules_never_choose]
        applied = rng.sample(usual, min(rnd, len(usual)))           # tasks earlier rounds would have run
        out = rules_decision(analysis, cfg, gate.tree, applied, rnd)
        if out.decision not in ("accept", "enhance", "reject"):
            continue                                                 # "review" is not one of Laya's options
        allowed = permitted(analysis, cfg, gate.tree, applied, rnd)
        if not allowed:
            continue                                                 # nothing to choose: Laya is not asked
        if enhance_choice and (out.decision != "enhance"
                               or sum(a.startswith("enhance:") for a in allowed) < 2):
            continue
        state = state_to_json(build_state(analysis, applied, rnd, cfg.laya.max_state_chars, allowed=allowed))
        name = f"synth_enhance_{made:06d}" if enhance_choice else f"synth_{tries - 1:06d}"
        yield name, {"state": state, "questions": QUESTIONS,
                     "expected": expected_from(out.decision, out.reason, out.operation),
                     "meta": {"label_source": "rules", "origin": origin}}
        made += 1


# ---------------------------------------------------------------------- the data set
def build_dataset(gate, out_dir, photos=None, simulate: int = 0, synthetic: int = 4000, eval_fraction: float = 0.15,
                  seed: int = 0, max_copies: int = 3, echo=print, synthetic_enhance: int = 0) -> Dict[str, Any]:
    """Write ``train.jsonl`` and ``eval.jsonl`` (rows for ``laya.train.finetune``) and return a summary.

    Identical states are kept at most ``max_copies`` times (common states would otherwise crowd out the
    rare ones). When one state received different labels (possible when two analyses differ only in what
    the state leaves out), the majority label is kept and the conflict is counted."""
    if gate.backend is not None:
        raise ValueError("build the data set with a gate that has no Laya backend")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, Dict[str, Any]]] = []
    skipped: Counter = Counter()
    if photos:
        for sample in iter_samples(photos):
            try:
                res = run_sample(gate, sample)
            except Exception as exc:  # one unreadable photo must not stop the export
                skipped[f"error_{type(exc).__name__}"] += 1
                continue
            row, why = row_from_result(res)
            if row is None:
                skipped[why] += 1
                continue
            row["meta"]["origin"] = "photo"
            rows.append((f"photo:{sample.name}", row))
    if simulate:
        for name, res in simulated_results(gate, simulate, seed):
            row, why = row_from_result(res)
            if row is None:
                skipped[why] += 1
                continue
            row["meta"]["origin"] = "simulated_photo"
            rows.append((name, row))
    rows.extend(synthetic_rows(gate, synthetic, seed + 1))
    if synthetic_enhance:
        rows.extend(synthetic_rows(gate, synthetic_enhance, seed + 2, enhance_choice=True))

    by_state: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for name, row in rows:
        by_state[row["state"]].append((name, row))
    conflicts = 0
    split: Dict[str, List[Dict[str, Any]]] = {"train": [], "eval": []}
    for state, items in by_state.items():
        labels = Counter(json.dumps(r["expected"], sort_keys=True) for _, r in items)
        if len(labels) > 1:
            conflicts += 1
        keep = json.loads(labels.most_common(1)[0][0])
        kept = [(n, dict(r, expected=keep)) for n, r in items][:max_copies]
        # one state goes entirely to train or to eval: the evaluation never sees a state it was trained on
        part = "eval" if _is_eval(state, eval_fraction, seed) else "train"
        for name, row in kept:
            row["meta"] = dict(row["meta"], sample=name)
            split[part].append(row)
    random.Random(seed).shuffle(split["train"])
    summary: Dict[str, Any] = {"rows_before_dedup": len(rows), "unique_states": len(by_state),
                               "states_with_conflicting_labels": conflicts, "skipped": dict(skipped)}
    for part, items in split.items():
        with open(out / f"{part}.jsonl", "w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[part] = {
            "rows": len(items),
            "origin": dict(Counter(r["meta"]["origin"] for r in items)),
            "decision": dict(Counter(r["expected"]["decision"] for r in items)),
            "reject_reason": dict(Counter(r["expected"].get("reject_reason") for r in items if "reject_reason" in r["expected"])),
            "docres_task": dict(Counter(r["expected"].get("docres_task") for r in items if "docres_task" in r["expected"])),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    echo(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


__all__ = ["random_analysis", "synthetic_rows", "build_dataset"]
