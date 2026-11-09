"""Human-side tools for AI labels: audit sheet, approve/reject, promote to ``labeled/``, compare with human labels.

AI labels live in ``<workspace>/ai_labeled`` (same files as the app writes, plus ``<base>.label.ai.json``).
Nothing reaches ``labeled/`` (what you train on) until you promote it.
"""
from __future__ import annotations

import base64
import html
import io
import json
import random
import shutil
import time
from pathlib import Path

from PIL import Image

from rfalib.normalize import same

from . import images
from .config import Config
from .protocol import UNREADABLE

SUFFIXES = (".png", ".jpg", ".label.json", ".label.sidecar.json", ".label.ai.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ai_labels(cfg: Config) -> list[dict]:
    out = []
    for audit_path in sorted(cfg.out_dir.glob("*.label.ai.json")):
        base = audit_path.name[: -len(".label.ai.json")]
        audit = _read(audit_path)
        out.append({"base": base, "audit": audit, "audit_path": audit_path,
                    "label_path": cfg.out_dir / f"{base}.label.json", "image_path": cfg.out_dir / f"{base}.png"})
    return out


def _thumb(path: Path, max_side: int = 1100) -> str:
    im = images.load_oriented(path)
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def review_sheet(cfg: Config, n: int = 30, seed: int = 1, only: str = "needs_review",
                 out: Path | None = None) -> tuple[Path, int]:
    items = load_ai_labels(cfg)
    if only != "all":
        items = [x for x in items if x["audit"]["review"]["state"] == only]
    random.Random(seed).shuffle(items)
    items = items[:n]
    rows = []
    for x in items:
        a = x["audit"]
        body = []
        for key, f in a["fields"].items():
            cls = {"agrees": "ok", "disagrees": "bad", "agent_unreadable": "warn"}.get(f.get("reference", ""), "")
            if f["status"] == "unreadable":
                cls = cls or "unr"
            note = f.get("reason") or f.get("reference") or ""
            if f.get("format_exception"):
                note += f" | format exception: {f['format_exception']}"
            body.append(f"<tr class='{cls}'><td>{html.escape(key)}</td><td dir='auto'>{html.escape(f['value'])}</td>"
                        f"<td>{html.escape(note)}</td></tr>")
        reasons = "".join(f"<li>{html.escape(r)}</li>" for r in a["review"]["reasons"]) or "<li>none</li>"
        rows.append(f"""<section><h2>{html.escape(x['base'])} <small>{a['doc_type']} · {html.escape(a['labeler'])} ·
          state: {a['review']['state']}</small></h2>
          <div class='cols'><img src='data:image/jpeg;base64,{_thumb(x['image_path'])}'>
          <div><ul class='why'>{reasons}</ul><table><tr><th>field</th><th>value</th><th>note</th></tr>{''.join(body)}</table></div></div>
          </section>""")
    page = f"""<!doctype html><meta charset='utf-8'><title>A.L.G AI label review</title>
<style>body{{font:14px system-ui;margin:20px;max-width:1500px}} .cols{{display:flex;gap:20px;align-items:flex-start}}
img{{max-width:48%;border:1px solid #ccc}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:3px 8px}}
tr.ok td:first-child{{background:#e7f6e7}} tr.bad{{background:#fde2e2}} tr.warn{{background:#fff3cd}} tr.unr{{background:#eef}}
section{{margin-bottom:34px;border-bottom:2px solid #888;padding-bottom:18px}} small{{color:#666;font-weight:normal}}
.why{{color:#a00;margin:0 0 8px}}</style>
<h1>AI label review ({len(items)} labels, filter: {html.escape(only)})</h1>
<p>Compare every value with the image. Note the wrong ones, then run
<code>alg-mcp reject BASE ...</code> and <code>alg-mcp approve BASE ...</code>.
<b>Look at the image first, then at the AI's value</b> (pre-filled values bias reviewers).</p>{''.join(rows)}"""
    out = Path(out or (cfg.workspace / "ai_review.html"))
    out.write_text(page, encoding="utf-8")
    return out, len(items)


def set_review(cfg: Config, bases: list[str], state: str, who: str = "human") -> dict:
    if state not in ("approved", "rejected"):
        raise ValueError("state must be approved or rejected")
    done, missing = [], []
    for base in bases:
        p = cfg.out_dir / f"{base}.label.ai.json"
        if not p.exists():
            missing.append(base)
            continue
        audit = _read(p)
        audit["review"].update(state=state, decided_by=who, decided_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        p.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        done.append(base)
    return {"updated": done, "not_found": missing}


def promote(cfg: Config, auto_ok: bool = False, approved: bool = False, dry_run: bool = False) -> dict:
    if cfg.output == "labeled":
        raise ValueError("output is already 'labeled': nothing to promote")
    want = set()
    if auto_ok:
        want.add("auto_ok")
    if approved:
        want.add("approved")
    if not want:
        raise ValueError("choose what to promote: --auto-ok and/or --approved")
    promoted, conflicts, skipped = [], [], 0
    for x in load_ai_labels(cfg):
        if x["audit"]["review"]["state"] not in want:
            skipped += 1
            continue
        base = x["base"]
        targets = [cfg.human_dir / f"{base}{s}" for s in SUFFIXES]
        if any(t.exists() for t in targets):
            conflicts.append(base)                 # the human version wins; never overwrite
            continue
        if not dry_run:
            cfg.human_dir.mkdir(parents=True, exist_ok=True)
            for s in SUFFIXES:
                src = cfg.out_dir / f"{base}{s}"
                if src.exists():
                    shutil.copy2(src, cfg.human_dir / src.name)
            audit = x["audit"]
            audit["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            x["audit_path"].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        promoted.append(base)
    return {"promoted": promoted, "conflicts_not_overwritten": conflicts,
            "left_in_review": skipped, "dry_run": dry_run}


def compare(cfg: Config, ai_dir: Path | None = None, human_dir: Path | None = None, max_examples: int = 25) -> dict:
    """Field-by-field agreement between AI labels and human labels of the SAME image (matched by image hash)."""
    ai_dir = Path(ai_dir or cfg.out_dir)
    human_dir = Path(human_dir or cfg.human_dir)
    human = {}
    for lp in human_dir.glob("*.label.json"):
        png = lp.with_name(lp.name[: -len(".label.json")] + ".png")
        if png.exists():
            human[images.sha256_file(png)] = lp
    tally = {"agree": 0, "differ": 0, "both_unreadable": 0, "ai_over_abstains": 0, "ai_over_reads": 0}
    per_field: dict[str, dict[str, int]] = {}
    examples, matched = [], 0
    for lp in sorted(ai_dir.glob("*.label.json")):
        png = lp.with_name(lp.name[: -len(".label.json")] + ".png")
        if not png.exists() or (h := images.sha256_file(png)) not in human:
            continue
        matched += 1
        ai_f, hu_f = _read(lp)["fields"], _read(human[h])["fields"]
        for key, hv in hu_f.items():
            av = ai_f.get(key)
            if av is None:
                continue
            if hv == UNREADABLE and av == UNREADABLE:
                kind = "both_unreadable"
            elif hv == UNREADABLE:
                kind = "ai_over_reads"
            elif av == UNREADABLE:
                kind = "ai_over_abstains"
            else:
                kind = "agree" if same(av, hv) else "differ"
            tally[kind] += 1
            per_field.setdefault(key, {k: 0 for k in tally})[kind] += 1
            if kind in ("differ", "ai_over_reads") and len(examples) < max_examples:
                examples.append({"label": lp.name[: -len(".label.json")], "field": key, "human": hv, "ai": av, "kind": kind})
    total = sum(tally.values())
    return {"images_matched": matched, "fields_compared": total, "tally": tally,
            "exact_agreement": round(tally["agree"] / total, 4) if total else None,
            "dangerous_ai_over_reads": tally["ai_over_reads"], "per_field": per_field, "examples": examples}
