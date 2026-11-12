"""Calibrate the quality thresholds on your own photos.

    python -m gate_block calibrate PHOTOS_DIR --out calibration

Put photos you consider GOOD (sharp, evenly lit, whole document visible) in ``PHOTOS_DIR/good/``. Other
folders (``blur/``, ``dark/``, ``glare/`` ...) are optional and are reported next to the good ones. See
``gate_block/tools/samples.py`` for the file names that are understood.

What it does: runs the gate on every photo, writes every measured value to ``measurements.csv``, prints how
the good photos are distributed and how the current thresholds route each folder, and writes
``suggested_thresholds.yaml``: for each metric the "mild" threshold is moved just outside what 95 % of the
good photos show, so that good photos stop being sent to DocRes; "moderate" and "severe" keep their distance
from "mild". It never loosens a threshold that the good photos already pass.

The better calibration, once Block 3 exists, is against OCR results: label each photo by whether TeleOCR
read it correctly and move the thresholds to where accuracy drops. The CSV is the input for that too.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..types import DEFECTS
from .samples import iter_samples, run_sample

# metric -> (config key, True when higher is worse)
THRESHOLD_KEYS = {
    "blur": ("blur_ratio", True), "dark": ("dark_ratio", False), "low_contrast": ("contrast_ratio", False),
    "shadow": ("shadow_depth", True), "glare": ("glare_content", True), "low_resolution": ("text_px", False),
    "curl": ("curl", True), "noise": ("noise_sigma", True),
}


def measure_folder(gate, root, workdir=None, progress=None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in iter_samples(root):
        try:
            res = run_sample(gate, sample, workdir)
        except Exception as exc:  # one unreadable file must not stop a calibration run
            rows.append({"sample": sample.name, "label": sample.label, "side": "", "status_code": f"error:{type(exc).__name__}",
                         "route": "error", "recognition": "", "decided_by": ""})
            continue
        if progress:
            progress(sample, res)
        base = {"sample": sample.name, "label": sample.label, "status_code": res.status_code, "route": res.route,
                "decided_by": (res.decision or {}).get("decided_by", res.stage)}
        if not res.report:
            rows.append({**base, "side": "", "recognition": res.stage})
            continue
        for side, metrics in res.report["quality_by_side"].items():
            row = {**base, "side": side, "recognition": res.report["document_recognition"][side]["status"],
                   "reference": res.report["measurement_reference"][side]}
            for d in DEFECTS:
                row[d] = metrics[d]["value"]
                row[f"{d}_severity"] = metrics[d]["severity"]
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path) -> None:
    keys = ["sample", "label", "side", "route", "status_code", "decided_by", "recognition", "reference"]
    keys += [k for d in DEFECTS for k in (d, f"{d}_severity")]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def suggest(rows: List[Dict[str, Any]], cfg, good_label: str = "good") -> Dict[str, Any]:
    """Thresholds suggested by the good photos (template-measured sides only)."""
    good = [r for r in rows if r.get("label") == good_label and r.get("reference") == "template"]
    out: Dict[str, Any] = {"n_good_sides": len(good), "quality": {}, "notes": []}
    if len(good) < 20:
        out["notes"].append(f"only {len(good)} good sides measured; use at least 20 (more is better) before trusting this")
    if not good:
        return out
    q = cfg.quality
    # exposure_ref: what "normal brightness" is for these cameras (the dark metric is brightness / exposure_ref)
    exposure = [float(r["dark"]) * q.exposure_ref for r in good if r.get("dark") is not None]
    new_ref = q.exposure_ref
    if exposure:
        median = float(np.median(exposure))
        if abs(median / q.exposure_ref - 1.0) > 0.05:                         # a real difference, not measurement scatter
            new_ref = round(median, 3)
            out["quality"]["exposure_ref"] = new_ref
    for metric, (key, higher_worse) in THRESHOLD_KEYS.items():
        vals = np.array([float(r[metric]) for r in good if r.get(metric) is not None], float)
        if metric == "dark" and exposure:
            vals = np.array(exposure) / new_ref                                # as it will read after the change
        if len(vals) < 5:
            continue
        mild, moderate, severe = getattr(q, key)
        if higher_worse:
            edge = float(np.percentile(vals, 95)) * 1.10
            if edge > mild:
                step = edge - mild
                out["quality"][key] = [round(edge, 4), round(moderate + step, 4), round(severe + step, 4)]
        else:
            edge = float(np.percentile(vals, 5)) * 0.92
            if edge < mild:
                step = mild - edge
                new = [round(edge, 4), round(max(moderate - step, edge * 0.8), 4), round(max(severe - step, edge * 0.6), 4)]
                if new[0] > new[1] > new[2] > 0:
                    out["quality"][key] = new
    return out


def summarise(rows: List[Dict[str, Any]], good_label: str = "good") -> str:
    lines: List[str] = []
    by_label: Dict[str, Counter] = defaultdict(Counter)
    seen = set()
    for r in rows:
        if (r["sample"], r["label"]) in seen:
            continue
        seen.add((r["sample"], r["label"]))
        by_label[r["label"]][r["status_code"]] += 1
    lines.append("How the current thresholds route each folder:")
    for label in sorted(by_label):
        total = sum(by_label[label].values())
        parts = ", ".join(f"{k} {v}" for k, v in by_label[label].most_common())
        lines.append(f"  {label:14s} {total:4d} photos: {parts}")
    good = [r for r in rows if r.get("label") == good_label and r.get("reference") == "template"]
    if good:
        lines.append("")
        lines.append(f"Measured values on the '{good_label}' photos ({len(good)} sides):")
        lines.append(f"  {'metric':15s} {'p05':>8s} {'p50':>8s} {'p95':>8s} {'max':>8s}")
        for d in DEFECTS:
            vals = np.array([float(r[d]) for r in good if r.get(d) is not None], float)
            if len(vals):
                lines.append(f"  {d:15s} {np.percentile(vals, 5):8.3f} {np.percentile(vals, 50):8.3f} {np.percentile(vals, 95):8.3f} {vals.max():8.3f}")
    return "\n".join(lines)


def run(gate, root, out_dir, good_label: str = "good", workdir: Optional[str] = None, echo=print) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = measure_folder(gate, root, workdir, progress=lambda s, r: echo(f"  {s.label}/{Path(s.name).name}: {r.status_code}"))
    write_csv(rows, out / "measurements.csv")
    suggestion = suggest(rows, gate.cfg, good_label)
    with open(out / "suggested_thresholds.yaml", "w", encoding="utf-8") as fh:
        fh.write("# Suggested by `python -m gate_block calibrate`. Copy the values you agree with into\n"
                 "# config/quality_thresholds.yaml (keys not listed here keep their current values).\n")
        yaml.safe_dump({k: v for k, v in suggestion["quality"].items()}, fh, sort_keys=False)
    echo("")
    echo(summarise(rows, good_label))
    echo("")
    for note in suggestion["notes"]:
        echo(f"NOTE: {note}")
    if suggestion["quality"]:
        echo(f"Suggested changes written to {out / 'suggested_thresholds.yaml'}:")
        for k, v in suggestion["quality"].items():
            echo(f"  {k}: {v}")
    else:
        echo("No threshold change suggested: the good photos already pass the current thresholds.")
    echo(f"All measurements: {out / 'measurements.csv'}")
    return {"rows": rows, "suggestion": suggestion}
