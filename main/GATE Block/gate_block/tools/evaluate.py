"""Measure the gate against labelled photos: the rules alone, Laya alone, and the two together.

    python -m gate_block evaluate PHOTOS_DIR --labels labels.csv [--laya PATH] [--out evaluation.json]
    python -m gate_block evaluate --simulate 300 --laya PATH          (agreement with the rules only)

``labels.csv`` has the columns ``sample, decision, reason, task`` (the same file ``export-laya`` reads):
what *should* happen to each photo. Without it, the first folder below PHOTOS_DIR is used when it is named
``accept``, ``enhance`` or ``reject``. Simulated photos (``--simulate``) have no independent label: they are
labelled by the rules, so for them the report only says how far Laya agrees with the rules.

Three deciders are compared on the same photos, in one pass:

    rules      the deterministic rules alone (what the gate does without ``--laya``)
    laya_raw   Laya's own recommendation, before Python validates it (recognised documents only)
    combined   what the gate really does with ``--laya``: Laya's recommendation after validation

Reported for each: decision accuracy, false acceptance rate (a photo that should not go straight to OCR was
accepted), false rejection rate (a usable photo was sent back to the user), DocRes task accuracy on the
photos both sides call "enhance", unnecessary DocRes runs, how often a person is asked, and the accuracy per
folder. Laya's time per decision and the gate's time per photo are reported too.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..laya_client import DECISIONS
from .export_laya_dataset import read_labels, simulated_results
from .samples import iter_samples, run_sample


def _split(code: Optional[str]) -> Dict[str, Optional[str]]:
    route, _, detail = (code or "").partition(":")
    return {"decision": route or None, "detail": detail or None}


def _laya_raw(decision: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    """Laya's unvalidated recommendation, from the probabilities stored with the decision."""
    probs = decision.get("laya_probabilities")
    if not probs or not decision.get("laya_eligible", True):
        return None
    top = max(probs["decision"], key=probs["decision"].get)
    key = {"reject": "reject_reason", "enhance": "docres_task"}.get(top)
    return {"decision": top, "detail": max(probs[key], key=probs[key].get) if key else None}


def collect(gate, root=None, labels_csv=None, simulate: int = 0, seed: int = 0, progress=None) -> List[Dict[str, Any]]:
    """One record per photo: the label and what each decider says."""
    labels = read_labels(labels_csv) if labels_csv else {}
    records: List[Dict[str, Any]] = []

    def add(name: str, folder: str, res, label_source: Optional[str] = None) -> None:
        label = labels.get(name) or labels.get(Path(name).name)
        if label is not None:
            expected = {"decision": label["decision"], "detail": label["reason"] or label["task"] or None}
            source = "labels"
        elif folder in DECISIONS:
            expected, source = {"decision": folder, "detail": None}, "folder"
        elif label_source == "rules":
            expected, source = _split((res.decision or {}).get("rules_baseline") or res.status_code), "rules"
        else:
            expected, source = None, None
        dec = res.decision or {}
        rec = {"sample": name, "folder": folder, "expected": expected, "label_source": source,
               "combined": _split(res.status_code), "rules": _split(dec.get("rules_baseline") or res.status_code),
               "laya_raw": _laya_raw(dec), "decided_by": dec.get("decided_by", res.stage), "overrides": dec.get("overrides", []),
               "total_ms": res.timings_ms.get("total"), "decision_ms": res.timings_ms.get("decision")}
        records.append(rec)
        if progress:
            progress(rec)

    if root:
        for sample in iter_samples(root):
            try:
                add(sample.name, sample.label, run_sample(gate, sample))
            except Exception as exc:  # one unreadable file must not stop an evaluation
                records.append({"sample": sample.name, "folder": sample.label, "error": f"{type(exc).__name__}: {exc}"})
    for name, res in simulated_results(gate, simulate, seed) if simulate else ():
        add(name, "simulated", res, label_source="rules")
    return records


def _score(pairs: List[tuple]) -> Dict[str, Any]:
    """Metrics for one decider. ``pairs`` = [(expected, predicted, folder), ...] with a known expected decision."""
    n = len(pairs)
    out: Dict[str, Any] = {"photos": n}
    if not n:
        return out
    correct = sum(1 for e, p, _ in pairs if p["decision"] == e["decision"])
    should_not_accept = [(e, p) for e, p, _ in pairs if e["decision"] != "accept"]
    usable = [(e, p) for e, p, _ in pairs if e["decision"] != "reject"]
    both_enhance = [(e, p) for e, p, _ in pairs if e["decision"] == p["decision"] == "enhance" and e["detail"]]
    both_reject = [(e, p) for e, p, _ in pairs if e["decision"] == p["decision"] == "reject" and e["detail"]]

    def rate(k: int, m: int) -> Optional[float]:
        return None if not m else round(k / m, 4)

    out.update({
        "decision_accuracy": rate(correct, n),
        "false_acceptance_rate": rate(sum(1 for _, p in should_not_accept if p["decision"] == "accept"), len(should_not_accept)),
        "false_rejection_rate": rate(sum(1 for _, p in usable if p["decision"] == "reject"), len(usable)),
        "docres_task_accuracy": rate(sum(1 for e, p in both_enhance if p["detail"] == e["detail"]), len(both_enhance)),
        "reject_reason_accuracy": rate(sum(1 for e, p in both_reject if p["detail"] == e["detail"]), len(both_reject)),
        "unnecessary_docres_rate": rate(sum(1 for e, p, _ in pairs if e["decision"] == "accept" and p["decision"] == "enhance"),
                                        sum(1 for e, _, _ in pairs if e["decision"] == "accept")),
        "review_rate": rate(sum(1 for _, p, _ in pairs if p["decision"] == "review"), n),
        "confusion": {f"{e}->{p}": c for (e, p), c in sorted(Counter((e["decision"], p["decision"]) for e, p, _ in pairs).items())},
    })
    by_folder: Dict[str, List[bool]] = defaultdict(list)
    for e, p, folder in pairs:
        by_folder[folder].append(p["decision"] == e["decision"])
    out["accuracy_by_folder"] = {f: {"photos": len(v), "accuracy": round(sum(v) / len(v), 4)} for f, v in sorted(by_folder.items())}
    return out


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    labelled = [r for r in ok if r["expected"]]
    sources = Counter(r["label_source"] for r in labelled)
    summary: Dict[str, Any] = {
        "photos": len(records), "errors": len(records) - len(ok), "labelled": len(labelled), "label_sources": dict(sources),
        "laya_connected": any(r["laya_raw"] for r in ok),
        "decided_by": dict(Counter(r["decided_by"] for r in ok)),
        "rules": _score([(r["expected"], r["rules"], r["folder"]) for r in labelled]),
        "laya_raw": _score([(r["expected"], r["laya_raw"], r["folder"]) for r in labelled if r["laya_raw"]]),
        "combined": _score([(r["expected"], r["combined"], r["folder"]) for r in labelled]),
    }
    with_laya = [r for r in ok if r["laya_raw"]]
    if with_laya:
        summary["laya_agrees_with_rules"] = round(sum(1 for r in with_laya if r["laya_raw"]["decision"] == r["rules"]["decision"]) / len(with_laya), 4)
        summary["laya_overridden"] = dict(Counter(o.split(" ")[0] for r in with_laya for o in r["overrides"]))
    times = [r["total_ms"] for r in ok if r.get("total_ms") is not None]
    dtimes = [r["decision_ms"] for r in ok if r.get("decision_ms") is not None]
    if times:
        summary["mean_ms_per_photo"] = round(sum(times) / len(times), 1)
    if dtimes:
        summary["mean_ms_per_decision"] = round(sum(dtimes) / len(dtimes), 2)
    notes: List[str] = []
    if sources.get("rules") and len(sources) == 1:
        notes.append("every label comes from the rules themselves: 'rules' is 100 % by construction; read laya_raw / combined as agreement with the rules")
    elif sources.get("rules"):
        notes.append("simulated photos are labelled by the rules: they favour the rules")
    if len(labelled) < 50:
        notes.append(f"only {len(labelled)} labelled photos: the rates are rough")
    if not summary["laya_connected"]:
        notes.append("no Laya backend connected (--laya): 'combined' equals 'rules'")
    summary["notes"] = notes
    return summary


def run(gate, root=None, labels_csv=None, simulate: int = 0, seed: int = 0, out_path=None, echo=print) -> Dict[str, Any]:
    records = collect(gate, root, labels_csv, simulate, seed)
    summary = summarise(records)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({"summary": summary, "photos": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    echo(json.dumps(summary, ensure_ascii=False, indent=2))
    for note in summary["notes"]:
        echo(f"NOTE: {note}")
    return {"summary": summary, "photos": records}
