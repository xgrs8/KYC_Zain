#!/usr/bin/env python3
"""Publish the GATE Block's Laya training data to Hugging Face as a public dataset, with an honest card.

    python publish_dataset.py --data data/v1 --stage hf_dataset                    # build and look at it
    python publish_dataset.py --data data/v1 --push RADAXGRS/kyc-gate-laya-decisions   # and publish it

``--data`` is a folder written by ``python -m gate_block laya-data`` (train.jsonl, eval.jsonl,
summary.json; the .gz versions kept in the repository work too). The card is written from the data itself
(counts per origin, decision, reason and task), so it cannot drift from what is uploaded. Pushing needs a
Hugging Face token with write access in HF_TOKEN (or a prior ``hf auth login``). A dataset that already
exists is left alone unless ``--update`` is given.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_URL = "https://github.com/xgrs8/kyc_zain"
ORIGINS = {
    "synthetic": "measurement states drawn at random over the gate's severity scales "
                 "(`gate_block.tools.laya_training.random_analysis`); no image behind them",
    "synthetic_enhance": "drawn the same way, but kept only where the rules choose *enhance* and two or more DocRes "
                         "tasks are permitted (the cases where the choice of the task matters)",
    "simulated_photo": "synthetic photos made from the four document templates by `gate_block.synth` (blur, camera "
                       "shake, glare, shadow, darkness, curl, cut-off, a finger, steep angle, distance) and measured "
                       "by the gate",
    "photo": "real image files measured by the gate: renders of the synthetic print run and phone photos of a "
             "printed licence of that run (see below)",
}


def load(folder: Path, name: str) -> List[Dict[str, Any]]:
    plain, packed = folder / name, folder / (name + ".gz")
    if plain.exists():
        text = plain.read_text(encoding="utf-8")
    elif packed.exists():
        text = gzip.decompress(packed.read_bytes()).decode("utf-8")
    else:
        raise SystemExit(f"{folder}: neither {name} nor {name}.gz")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def table(counter: Counter, head: str) -> str:
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in counter.most_common())
    return f"| {head} | rows |\n|---|---|\n{rows}"


def card(train: List[Dict], evals: List[Dict], summary: Dict[str, Any], provenance: Dict[str, Any]) -> str:
    def counts(rows, key):
        return Counter(r["expected"].get(key) for r in rows if r["expected"].get(key))
    origin_t = Counter((r.get("meta") or {}).get("origin", "?") for r in train)
    origin_e = Counter((r.get("meta") or {}).get("origin", "?") for r in evals)
    photos = sorted({(r.get("meta") or {}).get("sample", "") for r in train + evals
                     if (r.get("meta") or {}).get("origin") == "photo"})
    origin_rows = "\n".join(f"| `{o}` | {origin_t.get(o, 0)} | {origin_e.get(o, 0)} | {ORIGINS.get(o, '')} |"
                            for o in sorted(set(origin_t) | set(origin_e)))
    return f"""---
license: cc-by-4.0
language: [en]
pretty_name: KYC GATE Block decisions for Laya
task_categories: [text-classification]
tags: [kyc, document-quality, laya, decision-making, synthetic, rule-labelled]
size_categories: [1K<n<10K]
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: eval.jsonl
---

# KYC GATE Block decisions for Laya

Training data for the decision model of the **GATE Block**, Block 1 of a KYC document agent for Iraqi
documents (national ID, tax card, company licence). Each row is the quality **measurement state** of one
document photo, the three questions the gate asks, and the answer the gate's **rules** give:

* `decision`: accept (send to OCR) / enhance (send to DocRes) / reject (ask for a new photo)
* `reject_reason`: blur, glare, dark, overexposed, shadow, cut_off, angle, too_small, obstructed, other
* `docres_task`: dewarping, deshadowing, appearance, deblurring, binarization

The rows are in the format `laya.train.finetune` reads (`state`, `questions`, `expected`, `meta`). There are
**no images** in this dataset and no personal data: the states are severities and field statuses only.

## Where the labels come from (the source of truth)

The labels are **not** human annotations and **not** OCR outcomes. They are the decisions of the GATE
Block's deterministic rules (`config/rules.yaml` and `config/decision_tree.yaml` in
[{REPO_URL.split('/', 3)[-1]}]({REPO_URL})) applied to the measurements of each state. The state lists
the branches the measurements permit (`allowed`); the label is the branch the rules choose among them.
A model trained on this data therefore learns to **reproduce the rules**; it cannot be better than they are,
and its confidence means "agrees with the rules", not "the document will be read correctly".

## Where the rows come from

| origin | train | validation | what it is |
|---|---|---|---|
{origin_rows}

The `photo` rows were measured on these files: {', '.join(f'`{p}`' for p in photos) or 'none'}.

{chr(10).join(f'* `{k}` {v}' for k, v in (provenance.get('photos') or {}).items())}

The renders
(`dataset/...`) are the digital images of the synthetic print run made by Z.ai's *Synthetic KYC Dataset
Builder* (90 fictional identities; the ID cards have no portrait); `real/...` are phone photos of one printed
and cut-out licence of that run, in the conditions the names say. The print run itself has generation
errors (Syrian currency and addresses on the licences, a placeholder surname on the IDs ...): they do not
matter here, since only quality measurements are recorded, but see the A.L.G README in the repository before
using that print run for OCR.

## How it was made

```
{provenance.get('command', 'python -m gate_block laya-data ...')}
```

{provenance.get('code', 'GATE Block')} ({provenance.get('commit', 'commit not recorded')}). Identical states are
kept at most three times; a state belongs entirely to train or to validation (hash of the state, validation
share {provenance.get('eval_fraction', 0.15)}), so no validation state was ever trained on. States that
received different labels: {summary.get('states_with_conflicting_labels', 'n/a')}.

## Counts

{table(counts(train, 'decision'), 'decision (train)')}

{table(counts(train, 'reject_reason'), 'reject_reason (train)')}

{table(counts(train, 'docres_task'), 'docres_task (train)')}

Validation: {len(evals)} rows ({dict(counts(evals, 'decision'))}).

## Limitations

* Rule labels: the thresholds behind them were tuned on templates, simulated photos, the print run's renders
  and ten real photos, not on customer uploads.
* Most rows are synthetic states; their distribution is not the distribution of real uploads.
* Few real photos, all of one printed licence; no real photos of the ID or the tax card.
* Fictional documents only.

## Use

```python
from datasets import load_dataset
ds = load_dataset("{provenance.get('repo', 'RADAXGRS/kyc-gate-laya-decisions')}")
```

Training: `training/train_laya.py` in the repository (SFT then RL with proper-scoring-rule rewards, whole
model, from `convaiinnovations/laya-multilingual`).
"""


def build(data: Path, stage: Path, provenance: Dict[str, Any]) -> Dict[str, Any]:
    train, evals = load(data, "train.jsonl"), load(data, "eval.jsonl")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8")) if (data / "summary.json").exists() else {}
    overlap = {r["state"] for r in train} & {r["state"] for r in evals}
    if overlap:
        raise SystemExit(f"{len(overlap)} state(s) are in both train and validation: refusing to publish")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with open(stage / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if summary:
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (stage / "README.md").write_text(card(train, evals, summary, provenance), encoding="utf-8")
    return {"train": len(train), "validation": len(evals), "folder": str(stage)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--stage", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--push", help="dataset repo id, e.g. RADAXGRS/kyc-gate-laya-decisions")
    ap.add_argument("--update", action="store_true", help="upload even if the dataset already exists")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args(argv)
    prov_file = a.data / "provenance.json"
    provenance = json.loads(prov_file.read_text(encoding="utf-8")) if prov_file.exists() else {}
    if a.push:
        provenance["repo"] = a.push
    res = build(a.data, a.stage, provenance)
    print(json.dumps(res))
    if not a.push:
        return 0
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if api.repo_exists(a.push, repo_type="dataset") and not a.update:
        print(f"{a.push} already exists: left as it is (pass --update to upload again)")
        return 0
    api.create_repo(a.push, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.push, repo_type="dataset", folder_path=str(a.stage),
                      commit_message=f"GATE Block decisions: {res['train']} train / {res['validation']} validation rows")
    print(f"published: https://huggingface.co/datasets/{a.push}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
