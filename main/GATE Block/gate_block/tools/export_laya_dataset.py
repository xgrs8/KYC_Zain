"""Build a fine-tuning set for Laya from your photos (and, optionally, simulated ones).

    python -m gate_block export-laya PHOTOS_DIR --out laya_data [--labels labels.csv] [--simulate 400]

Each row is what Laya's trainer reads (``laya-train --data train.jsonl``)::

    {"state": "<compact JSON, exactly as sent at inference>",
     "questions": {...the three questions of laya_client.QUESTIONS...},
     "expected": {"decision": "enhance", "docres_task": "deblurring"}}

Only the questions that apply are labelled: ``reject_reason`` on reject rows, ``docres_task`` on enhance rows.

Where the labels come from, best first:

1. ``--labels labels.csv`` with the columns ``sample, decision, reason, task`` (reason / task may be empty).
   Use it for labels from OCR results: accept if TeleOCR read the photo correctly, enhance + the task that
   made it readable, reject + the reason otherwise. These are the labels that can make Laya better than
   the rules.
2. Otherwise the deterministic rules label the row ("silver" labels). A Laya trained only on these learns to
   imitate the rules: good for checking the whole loop, not a way to beat them.

Every photo of a recognised document becomes a row, because that is exactly what Laya is asked about at
inference, including the photos a safety rule rejects (they teach the reject branch and its reasons).
Photos Gate 1 does not recognise are never shown to Laya, so they are not exported (the summary counts
them). A label that contradicts a safety rule (for example "accept" for a photo with a cut-off edge) is
kept but listed in the summary: at inference the safety rule wins, so check the photo or the thresholds.

The split into train / eval is by *group*, never by photo: all photos of one document stay on the same
side. The group is taken from the file name with ``--group-regex`` (default: the name up to the first
"__" or the whole name).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..laya_client import DECISIONS, QUESTIONS
from ..types import DOCRES_TASKS, REJECT_REASONS
from .samples import iter_samples, run_sample


def read_labels(path) -> Dict[str, Dict[str, str]]:
    labels: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            name = (row.get("sample") or "").strip().replace("\\", "/")
            decision = (row.get("decision") or "").strip().lower()
            reason = (row.get("reason") or "").strip().lower()
            task = (row.get("task") or "").strip().lower()
            if not name:
                raise ValueError(f"{path} line {i}: empty 'sample'")
            if decision not in DECISIONS:
                raise ValueError(f"{path} line {i}: decision must be one of {DECISIONS}, got {decision!r}")
            if decision == "reject" and reason not in REJECT_REASONS:
                raise ValueError(f"{path} line {i}: a reject row needs a reason from {REJECT_REASONS}, got {reason!r}")
            if decision == "enhance" and task not in DOCRES_TASKS:
                raise ValueError(f"{path} line {i}: an enhance row needs a task from {DOCRES_TASKS}, got {task!r}")
            labels[name] = {"decision": decision, "reason": reason, "task": task}
    return labels


def expected_from(decision: str, reason: Optional[str], task: Optional[str]) -> Dict[str, str]:
    exp = {"decision": decision}
    if decision == "reject":
        exp["reject_reason"] = reason or "other"
    elif decision == "enhance":
        exp["docres_task"] = task
    return exp


def row_from_result(res, label: Optional[Dict[str, str]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """A training row for one gate result, or ``(None, why it was skipped)``."""
    if not res.laya_state_json or not res.decision:
        return None, "stopped_before_quality"
    if not res.decision.get("laya_eligible", True):
        return None, "laya_not_asked"                 # not recognised, or nothing to choose from
    if label is not None:
        exp = expected_from(label["decision"], label["reason"], label["task"])
        source = "labels"
    else:
        if res.route not in DECISIONS:
            return None, f"rules_said_{res.route}"
        if res.route == "reject" and res.reason not in REJECT_REASONS:
            return None, f"reason_{res.reason}"
        exp = expected_from(res.route, res.reason, res.operation)
        source = "rules"
    meta = {"label_source": source, "rules": res.status_code, "safety_rule": res.decision["decided_by"] == "hard_rule"}
    return {"state": res.laya_state_json, "questions": QUESTIONS, "expected": exp, "meta": meta}, "ok"


def _group_of(name: str, pattern: Optional[str]) -> str:
    stem = Path(name).name
    if pattern:
        m = re.search(pattern, stem)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return stem.split("__")[0]


def _is_eval(group: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2 ** 32 < fraction


def simulated_results(gate, n: int, seed: int = 0) -> Iterator[Tuple[str, Any]]:
    """``n`` simulated photos of the reference templates with random defects, run through the gate."""
    from .. import synth
    rng = random.Random(seed)
    templates = gate.pack.templates
    for i in range(n):
        t = templates[i % len(templates)]
        kw: Dict[str, Any] = dict(seed=rng.randrange(10 ** 6), rot_deg=rng.uniform(-10, 10), persp=rng.uniform(0, 0.06),
                                  doc_frac=rng.uniform(0.5, 0.9), bg=rng.choice(["wood", "plain", "cloth", "clutter", "dark"]),
                                  noise=rng.uniform(0, 3),
                                  # every phone renders tones and colours a little differently
                                  gamma=rng.uniform(0.8, 1.25), contrast=rng.uniform(0.9, 1.2), flare=rng.uniform(0.0, 0.05),
                                  saturation=rng.uniform(0.9, 1.3), tint=(rng.uniform(0.92, 1.08), 1.0, rng.uniform(0.92, 1.08)))
        if t.size[1] > t.size[0]:
            kw["frame"] = (1200, 1600)
        for _ in range(rng.choice([0, 1, 1, 2])):                 # zero to two defects per photo
            kind = rng.choice(["blur", "dark", "shadow", "glare", "curl", "noise", "motion", "small"])
            if kind == "blur":
                kw["blur_sigma"] = rng.uniform(1.0, 5.0)
            elif kind == "dark":
                kw["exposure"] = rng.uniform(0.3, 0.8)
            elif kind == "shadow":
                kw["shadow"] = rng.uniform(0.2, 0.8)
            elif kind == "glare":
                kw["glare"], kw["glare_xy"] = rng.uniform(0.3, 1.0), (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
            elif kind == "curl" and not t.rigid:
                kw["curl"] = rng.uniform(0.02, 0.07)
            elif kind == "noise":
                kw["noise"] = rng.uniform(6, 16)
            elif kind == "motion":
                kw["motion_len"] = rng.choice([7, 11, 15])
            elif kind == "small":
                kw["doc_frac"] = rng.uniform(0.35, 0.5)
        variant = synth.variant_from_template(t.bgr, {k: f.box for k, f in t.fields.items()}, rng.randrange(10 ** 6))
        photo, _ = synth.capture(variant, synth.CaptureParams(**kw))
        yield f"sim_{i:05d}__{t.doc_type}_{t.side}", gate.process_image(photo, expected=t.doc_type)


def run(gate, root, out_dir, labels_csv=None, eval_fraction: float = 0.15, group_regex: Optional[str] = None,
        simulate: int = 0, seed: int = 0, echo=print) -> Dict[str, Any]:
    if gate.backend is not None:
        raise ValueError("export with a gate that has no Laya backend: the rows must not depend on the model being trained")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = read_labels(labels_csv) if labels_csv else {}
    used_labels = set()
    rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "eval": []}
    skipped: Counter = Counter()
    sources: Counter = Counter()
    conflicts: List[str] = []

    def add(name: str, res) -> None:
        label = labels.get(name) or labels.get(Path(name).name)
        if label is not None:
            used_labels.add(name if name in labels else Path(name).name)
        row, why = row_from_result(res, label)
        if row is None:
            skipped[why] += 1
            return
        row["meta"]["sample"] = name
        sources[row["meta"]["label_source"]] += 1
        if row["meta"]["safety_rule"] and row["expected"]["decision"] != "reject":
            conflicts.append(name)
        rows["eval" if _is_eval(_group_of(name, group_regex), eval_fraction, seed) else "train"].append(row)

    if root:
        for sample in iter_samples(root):
            try:
                add(sample.name, run_sample(gate, sample))
            except Exception as exc:
                skipped[f"error_{type(exc).__name__}"] += 1
    for name, res in simulated_results(gate, simulate, seed) if simulate else ():
        add(name, res)

    summary: Dict[str, Any] = {"skipped": dict(skipped), "label_sources": dict(sources),
                               "labels_not_matched": sorted(set(labels) - used_labels),
                               "labels_contradicting_a_safety_rule": sorted(conflicts)}
    for split, items in rows.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = {
            "rows": len(items),
            "decision": dict(Counter(r["expected"]["decision"] for r in items)),
            "reject_reason": dict(Counter(r["expected"]["reject_reason"] for r in items if "reject_reason" in r["expected"])),
            "docres_task": dict(Counter(r["expected"]["docres_task"] for r in items if "docres_task" in r["expected"])),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    echo(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["labels_not_matched"]:
        echo(f"WARNING: {len(summary['labels_not_matched'])} rows of the labels file matched no photo (see summary.json)")
    if conflicts:
        echo(f"WARNING: {len(conflicts)} labels say accept/enhance where a safety rule rejects (see summary.json)")
    if len(set(r["expected"]["decision"] for r in rows["train"])) < 3:
        echo("WARNING: the training set does not contain all three decisions; add photos (or --simulate) before training")
    echo(f"Wrote {out / 'train.jsonl'} and {out / 'eval.jsonl'}")
    return summary
