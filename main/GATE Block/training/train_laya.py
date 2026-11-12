#!/usr/bin/env python3
"""Fine-tune Laya for the GATE Block's decision, from a fresh base checkpoint (Kaggle GPU or any machine).

    python train_laya.py --out run1                       # pulls everything from Hugging Face
    python train_laya.py --out run1 --push RADAXGRS/laya-kyc-gate   # and publishes it if every gate passes

What it does
  1. pulls a fresh base checkpoint from Hugging Face (convaiinnovations/laya-multilingual, Apache-2.0)
  2. pulls the training data from Hugging Face (RADAXGRS/kyc-gate-laya-decisions): measurement states of
     document photos with the GATE Block's decision as the label
  3. stage 1, supervised fine-tuning (SFT, loss ``soft-ce``): the whole model, encoder included, learns the
     job: accept / enhance / reject, the reject reason and the DocRes task
  4. stage 2, reinforcement learning (RL, loss ``rlcd``, Laya's own objective): a GRPO-style policy gradient
     over noisy samples of Laya's own probabilities, each rewarded with proper scoring rules (spherical score
     + ranked probability score), plus the supervised term. A proper scoring rule pays most for the
     probability that matches how often the answer is right, so this stage trains the confidence, not only
     the choice. Laya then fits one temperature per question on a calibration slice it keeps back. (Laya's
     authors saw no gain from the RL term on their own benchmark, so its value here is whatever the held-out
     calibration numbers of the two stages show: both are in the report.)
  5. measures the result on the held-out split: accuracy per question, false accepts, expected calibration
     error, and the reliability table ("of the answers given with ~95 % confidence, how many were right")
  6. pushes the model to Hugging Face when ``--push`` is given: public only if every gate passes; a failed
     run is not published (``--private-on-fail`` keeps it as a private repo for inspection)

What the confidence means here: the labels are the GATE Block's rules, so "95 % confident" means "agrees
with the rules 95 % of the time". The confidence that the KYC system needs at the end ("this document was
read correctly") can only be measured against real outcomes (TeleOCR results on held-out photos); see
``README.md`` in this folder, phase 2.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE = "convaiinnovations/laya-multilingual"
DEFAULT_DATA = "RADAXGRS/kyc-gate-laya-decisions"
GATES = {
    "decision_accuracy_min": 0.98,      # the job
    "false_accepts_max": 0,             # a photo the rules reject or enhance must never be accepted
    "decision_ece_max": 0.05,           # calibration of the decision probabilities
    "high_conf_threshold": 0.95,        # answers given with at least this confidence ...
    "high_conf_accuracy_min": 0.95,     # ... must be right at least this often
    "high_conf_min_n": 30,              # (checked only with enough of them)
}
BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001)


# ---------------------------------------------------------------------- inputs
def resolve(source: str, kind: str, dest: Path, revision: Optional[str], token: Optional[str]) -> Tuple[Path, Dict]:
    """A local folder is used as it is; anything else is a Hugging Face repo id, downloaded at a pinned commit."""
    if Path(source).is_dir():
        return Path(source), {"source": str(source), "kind": "local folder"}
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    info = api.dataset_info(source, revision=revision) if kind == "dataset" else api.model_info(source, revision=revision)
    path = snapshot_download(source, repo_type=kind, revision=info.sha, local_dir=str(dest), token=token)
    return Path(path), {"source": source, "kind": f"huggingface {kind}", "revision": info.sha}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def small_subset(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """n rows that still contain every decision (for a smoke test)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["expected"]["decision"]].append(r)
    out: List[Dict[str, Any]] = []
    while len(out) < min(n, len(rows)):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop(rng.randrange(len(by[k]))))
    return out


# ---------------------------------------------------------------------- training
def train_stage(data: Path, model_dir: Path, out_dir: Path, loss: str, epochs: int, a, eval_file: Path,
                encoder_lr: float, head_lr: float) -> Dict[str, Any]:
    from laya.train import TrainConfig, finetune
    cfg = TrainConfig(epochs=epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum, encoder_lr=encoder_lr,
                      head_lr=head_lr, loss=loss, freeze_encoder=a.freeze_encoder, eval_data=str(eval_file),
                      seed=a.seed, target_error=a.target_error)
    t0 = time.time()
    summary = finetune(str(data), str(model_dir), str(out_dir), cfg, device=a.device)
    shutil.rmtree(out_dir / "checkpoint_latest", ignore_errors=True)         # only kept for crash recovery
    return {"loss": loss, "epochs": epochs, "encoder_lr": None if a.freeze_encoder else encoder_lr, "head_lr": head_lr,
            "minutes": round((time.time() - t0) / 60, 1),
            "laya_summary": {k: v for k, v in summary.items() if isinstance(v, (int, float, str, dict, list))}}


# ---------------------------------------------------------------------- evaluation
def reliability(pairs: List[Tuple[float, bool]]) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    rows, ece, n = [], 0.0, len(pairs)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        inside = [(c, ok) for c, ok in pairs if lo <= c < hi]
        if not inside:
            continue
        conf = sum(c for c, _ in inside) / len(inside)
        acc = sum(ok for _, ok in inside) / len(inside)
        ece += len(inside) / n * abs(conf - acc)
        rows.append({"confidence": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": len(inside),
                     "mean_confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return rows, (round(ece, 4) if n else None)


def evaluate(model_dir: Path, rows: List[Dict[str, Any]], device: str, batch_size: int,
             threshold: float) -> Dict[str, Any]:
    import laya
    agent = laya.load(str(model_dir), device=device)
    questions = rows[0]["questions"]
    t0 = time.time()
    preds = agent.predict_batch([r["state"] for r in rows], questions, batch_size=batch_size)
    hit, tot, pairs = Counter(), Counter(), defaultdict(list)
    confusion, by_origin, by_origin_hit = Counter(), Counter(), Counter()
    false_accepts = []
    for r, p in zip(rows, preds):
        answers = p["answers"]
        for q, exp in r["expected"].items():
            probs = answers[q]["probabilities"]
            top = max(probs, key=probs.get)
            ok = top == exp
            tot[q] += 1
            hit[q] += ok
            pairs[q].append((float(probs[top]), ok))
            if q == "decision":
                confusion[f"{exp}->{top}"] += 1
                origin = (r.get("meta") or {}).get("origin", "?")
                by_origin[origin] += 1
                by_origin_hit[origin] += ok
                if top == "accept" and exp != "accept":
                    false_accepts.append({"state": r["state"], "expected": exp, "p_accept": probs["accept"]})
    out: Dict[str, Any] = {"rows": len(rows), "seconds": round(time.time() - t0, 1),
                           "accuracy": {q: round(hit[q] / tot[q], 4) for q in tot}, "n": dict(tot),
                           "decision_confusion": dict(sorted(confusion.items())),
                           "decision_accuracy_by_origin": {o: f"{by_origin_hit[o]}/{by_origin[o]}" for o in by_origin},
                           "false_accepts": len(false_accepts), "false_accept_examples": false_accepts[:5],
                           "reliability": {}, "ece": {}}
    for q, pr in pairs.items():
        out["reliability"][q], out["ece"][q] = reliability(pr)
    high = [(c, ok) for c, ok in pairs["decision"] if c >= threshold]
    out["high_confidence"] = {"threshold": threshold, "n": len(high),
                              "accuracy": round(sum(ok for _, ok in high) / len(high), 4) if high else None,
                              "share_of_answers": round(len(high) / max(1, len(pairs["decision"])), 4)}
    return out


def check_gates(m: Dict[str, Any], g: Dict[str, float]) -> List[str]:
    failed = []
    if m["accuracy"].get("decision", 0.0) < g["decision_accuracy_min"]:
        failed.append(f"decision accuracy {m['accuracy'].get('decision')} < {g['decision_accuracy_min']}")
    if m["false_accepts"] > g["false_accepts_max"]:
        failed.append(f"{m['false_accepts']} false accept(s)")
    ece = m["ece"].get("decision")
    if ece is None or ece > g["decision_ece_max"]:
        failed.append(f"decision ECE {ece} > {g['decision_ece_max']}")
    hc = m["high_confidence"]
    if hc["n"] >= g["high_conf_min_n"] and (hc["accuracy"] or 0.0) < g["high_conf_accuracy_min"]:
        failed.append(f"answers with confidence >= {hc['threshold']} are right only {hc['accuracy']} of the time")
    return failed


# ---------------------------------------------------------------------- publishing
def model_card(repo: str, report: Dict[str, Any]) -> str:
    m = report["evaluation"]
    acc = m["accuracy"]
    rel = "\n".join(f"| {r['confidence']} | {r['n']} | {r['mean_confidence']:.3f} | {r['accuracy']:.3f} |"
                    for r in m["reliability"]["decision"])
    base, data = report["base"], report["data"]

    def where(info: Dict[str, Any], kind: str) -> str:
        if info.get("kind") == "local folder":
            return f"a local folder ({Path(info['source']).name})"
        url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{info.get('source')}"
        return f"[`{info.get('source')}`]({url}) at `{info.get('revision')}`"

    stages = "\n".join(f"| {i + 1} | {s['loss']} | {s['epochs']} | {s['encoder_lr'] or 'frozen'} | {s['head_lr']} "
                       f"| {s['minutes']} |" for i, s in enumerate(report["stages"]))
    return f"""---
license: apache-2.0
library_name: laya
base_model: {DEFAULT_BASE}
datasets:
- {report['data'].get('source', DEFAULT_DATA)}
language: [ar, en]
tags: [laya, kyc, document-quality, decision-model, calibrated-decisions]
---

# {repo.split('/')[-1]}

The decision model of the **GATE Block** (Block 1 of the KYC document agent): given the quality measurements
of one photo of an Iraqi national ID, tax card or company licence, it chooses **accept** (send to OCR),
**enhance** (send to DocRes, and which restoration task) or **reject** (ask for a new photo, and why).
The GATE Block validates every answer against the measurements before it is used, and its safety rules
always win.

* base: {where(base, 'model')}
* data: {where(data, 'dataset')}
* trained with `training/train_laya.py` of [xgrs8/kyc_zain](https://github.com/xgrs8/kyc_zain): {report.get('trained', 'whole model')}

| stage | loss | epochs | encoder lr | head lr | minutes |
|---|---|---|---|---|---|
{stages}

## Held-out results ({m['rows']} states never seen in training)

| question | accuracy |
|---|---|
| decision | {acc.get('decision')} |
| reject reason | {acc.get('reject_reason')} |
| DocRes task | {acc.get('docres_task')} |

False accepts (a photo the rules reject or enhance, accepted by the model): **{m['false_accepts']}**.
Decision ECE: **{m['ece'].get('decision')}**. Answers with confidence >= {m['high_confidence']['threshold']}:
{m['high_confidence']['n']}, right {m['high_confidence']['accuracy']} of the time.

| decision confidence | n | mean confidence | accuracy |
|---|---|---|---|
{rel}

## What the confidence means

The labels are the GATE Block's deterministic rules applied to the measurements, so the probabilities say
how likely the rules' answer is, not whether OCR will read the document. A probability for "this document
will be read correctly" has to be fitted on real outcomes (OCR results on held-out photos) before it can go
into the final confidence formula of the KYC system.

## Use

```python
import laya
agent = laya.load("{repo}")
# with the GATE Block: Gate(backend=LayaBackend("{repo}"))
```

Gates checked before this model was published: {json.dumps(report['gates'])}.
"""


def push(out_dir: Path, repo: str, token: Optional[str], public: bool, card: str, report: Dict[str, Any]) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=not public, exist_ok=True)
    if public:
        try:
            api.update_repo_settings(repo, private=False)
        except AttributeError:                                       # older huggingface_hub
            api.update_repo_visibility(repo, private=False)
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    (out_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    api.upload_folder(repo_id=repo, folder_path=str(out_dir), repo_type="model",
                      commit_message=f"GATE Block decision model: decision accuracy {report['evaluation']['accuracy'].get('decision')}")
    return f"https://huggingface.co/{repo}"


# ---------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE, help="Hugging Face model id, or a local checkpoint folder")
    ap.add_argument("--base-revision")
    ap.add_argument("--data", default=DEFAULT_DATA, help="Hugging Face dataset id, or a local folder with train/eval jsonl")
    ap.add_argument("--data-revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-epochs", type=int, default=3)
    ap.add_argument("--rl-epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--target-error", type=float, default=0.05, help="abstention thresholds Laya fits after training")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="train on N rows and evaluate on N/2 (a quick check, no push)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train the decision head only (the whole model needs a GPU or about 10 GB of RAM)")
    ap.add_argument("--push", help="Hugging Face model repo to publish to, e.g. RADAXGRS/laya-kyc-gate")
    ap.add_argument("--private-on-fail", action="store_true", help="push a run that fails the gates as a private repo")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    try:                                    # T4 / P100 have no bfloat16: let Laya run its forward passes in fp16
        import torch
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            os.environ.setdefault("LAYA_CUDA_AMP", "fp16")
    except ImportError:
        pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base_dir, base_info = resolve(a.base, "model", out / "base", a.base_revision, token)
    data_dir, data_info = resolve(a.data, "dataset", out / "data", a.data_revision, token)
    train_rows, eval_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "eval.jsonl")
    if a.smoke:
        train_rows, eval_rows = small_subset(train_rows, a.smoke, a.seed), small_subset(eval_rows, max(8, a.smoke // 2), a.seed)
    train_file, eval_file = write_jsonl(out / "work" / "train.jsonl", train_rows), write_jsonl(out / "work" / "eval.jsonl", eval_rows)
    print(f"base {base_info}\ndata {data_info}: {len(train_rows)} train / {len(eval_rows)} held-out rows", flush=True)

    stages = []
    sft_dir, final_dir = out / "stage1_sft", out / "model"
    stages.append(train_stage(train_file, base_dir, sft_dir, "soft-ce", a.sft_epochs, a, eval_file, 2.5e-5, 1e-4))
    if a.rl_epochs:
        stages.append(train_stage(train_file, sft_dir, final_dir, "rlcd", a.rl_epochs, a, eval_file, 1e-5, 5e-5))
    else:
        shutil.copytree(sft_dir, final_dir, dirs_exist_ok=True)
    ev = evaluate(final_dir, eval_rows, "cpu" if a.device == "cpu" else a.device if a.device != "auto" else None,
                  a.eval_batch, GATES["high_conf_threshold"])
    failed = check_gates(ev, GATES)
    report = {"base": base_info, "data": data_info, "train_rows": len(train_rows), "eval_rows": len(eval_rows),
              "trained": "decision head only" if a.freeze_encoder else "whole model (encoder and head)",
              "stages": stages, "evaluation": ev, "gates": GATES, "gates_failed": failed, "smoke": bool(a.smoke)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: ev[k] for k in ("accuracy", "false_accepts", "ece", "high_confidence", "decision_confusion")},
                     indent=1), flush=True)
    print("gates: " + ("all passed" if not failed else "FAILED: " + "; ".join(failed)), flush=True)
    if a.smoke:
        print("smoke run: every step ran. A few rows and one epoch cannot pass the gates; that is expected "
              "and nothing is published.", flush=True)
        return 0
    if a.push and not a.smoke:
        if failed and not a.private_on_fail:
            print("not published: the gates failed", flush=True)
            return 1
        url = push(final_dir, a.push, token, public=not failed, card=model_card(a.push, report), report=report)
        print(f"published {'publicly' if not failed else 'as a PRIVATE repo'}: {url}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
