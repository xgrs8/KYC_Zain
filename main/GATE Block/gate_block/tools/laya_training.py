"""Teach Laya the gate's decisions, and package the result so it can be put on any computer.

Why this exists
---------------
Laya's published checkpoints are general decision models. Asked the gate's three questions as they come,
the multilingual checkpoint names sensible retake reasons and DocRes tasks, but it does not judge
*severity*: it answered "accept" for severely blurred or glared photos and "reject" for clean ones. The
gate therefore (1) lists in the state the branches the measurements permit (``allowed``, from
``rules.permitted``), so Laya's job is the judgement between them, and (2) fine-tunes Laya's decision head
on the gate's decision policy.

The labels come from the deterministic rules (the same policy the gate uses without Laya), applied to
1. measurement states the gate really produced: simulated photos of the templates and your own photos, and
2. many synthetic measurement states (random combinations of defects and field problems), so every
   combination Laya can meet has been seen.
Laya then makes the decision itself; the router still checks it against the measurements. To make Laya
*better* than the rules later, label photos by what really happened downstream (did TeleOCR read the photo
correctly, did DocRes help) and train on those labels (``export-laya --labels``).

Only part of the decision head is trained: the encoder of the base checkpoint is left as it is. So the
result is small: ``laya-pack`` writes the trained head to one file, and ``laya-assemble`` rebuilds the full
checkpoint from that file plus the base checkpoint on another computer, after checking that the base is
the one the head was trained on.

    python -m gate_block laya-data   --out laya_data [--photos PHOTOS_DIR] [--simulate 300] [--synthetic 4000]
    python -m gate_block laya-train  --base PATH/TO/Multilingual --data laya_data --out gate_laya
    python -m gate_block laya-pack   gate_laya --out gate_laya_head.safetensors
    python -m gate_block laya-assemble --base PATH/TO/Multilingual --head gate_laya_head.safetensors --out gate_laya
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
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

HEAD_FORMAT = "gate_block.laya_head/1"
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
    """Write ``train.jsonl`` and ``eval.jsonl`` for ``laya-train`` and return a summary.

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


# ---------------------------------------------------------------------- training
class EncoderCache:
    """Remember what a *frozen* encoder returns for each training sequence.

    ``laya.train`` runs the whole encoder on every item in every epoch, also when the encoder is frozen.
    Frozen (no gradient, eval mode, no dropout) and with the options in their fixed order, the encoder's
    output for a sequence never changes, so it is computed once and reused: from the second epoch on only
    the small decision head runs, which makes a CPU run several times faster. Outputs are kept in float16
    (``max_bytes`` caps the memory; sequences beyond it are simply recomputed). Calibration and evaluation
    run the model in eval mode through Laya's own encoder path and are not affected.
    """

    def __init__(self, encoder, max_bytes: int = 3 * 1024 ** 3):
        self.encoder = encoder
        self.inner = encoder.forward
        self.max_bytes = max_bytes
        self.store: Dict[bytes, Any] = {}
        self.bytes = 0
        self.hits = self.misses = 0

    def install(self) -> None:
        self.encoder.forward = self.forward          # an instance attribute: the weights / state_dict are untouched

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        from types import SimpleNamespace

        import torch

        if kwargs or input_ids is None or attention_mask is None or torch.is_grad_enabled() and any(
                p.requires_grad for p in self.encoder.parameters()):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        lengths = attention_mask.sum(1).tolist()
        keys = []
        for i, n in enumerate(lengths):
            n = int(n)
            prefix = bool(attention_mask[i, :n].all()) and (n == attention_mask.shape[1] or not bool(attention_mask[i, n:].any()))
            keys.append(input_ids[i, :n].cpu().numpy().tobytes() if prefix else None)
        missing = [i for i, k in enumerate(keys) if k is None or k not in self.store]
        out = None
        if missing:
            with torch.no_grad():
                h_new = self.inner(input_ids=input_ids[missing], attention_mask=attention_mask[missing]).last_hidden_state
            out = torch.zeros(input_ids.shape + (h_new.shape[-1],), dtype=h_new.dtype, device=h_new.device)
            for j, i in enumerate(missing):
                out[i] = h_new[j]
                n, k = int(lengths[i]), keys[i]
                if k is not None and self.bytes < self.max_bytes:
                    v = h_new[j, :n].detach().to(torch.float16).cpu().clone()
                    self.store[k] = v
                    self.bytes += v.numel() * 2
            self.misses += len(missing)
        for i, k in enumerate(keys):
            if i in missing:
                continue
            v = self.store[k]
            if out is None:
                out = torch.zeros(input_ids.shape + (v.shape[-1],), dtype=torch.float32, device=input_ids.device)
            out[i, :v.shape[0]] = v.to(out.dtype)
            self.hits += 1
        return SimpleNamespace(last_hidden_state=out)


def _soft_ce(logits, target, mask):
    import torch
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    return -(target * logp * mask).sum(-1).mean()


def cached_head_train(model, tok, items, config, device, max_len, head_max_len, on_epoch_end=None, parallel=False,
                     log=print, batch_size: int = 32, check_items: int = 24, layers: str = "last"):
    """A drop-in for ``laya.train.train_model`` that trains the **last head layer and the scorer** only.

    Laya's decision head is two transformer layers over the encoder's output, read at the option markers.
    With the encoder and the first head layer frozen, the input of the last layer never changes, so it is
    computed once (sorted by length: almost no padding). The last layer then only has to produce the
    outputs that are read — the option markers and token 0 — while it still attends to every token. That
    is mathematically the same layer, about ten times cheaper per step than running the whole head, which
    is what makes enough optimizer steps affordable on a CPU.

    ``layers="all"`` trains the whole head (every head layer, the question-type embedding and the scorer)
    the same way: then only the encoder's output is cached, and the last layer still computes only what
    is read.

    Before training, the shortcut is checked against the model's own forward pass on ``check_items``
    items; training stops with an error if they disagree.
    """
    import math as _math

    import torch
    from laya.common import collate_items
    from laya.train import encode_item

    if parallel:
        raise ValueError("cached_head_train supports the sequential option layout only")
    if not items:
        raise ValueError("no training items")
    torch.manual_seed(config.seed)
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if layers not in ("last", "all"):
        raise ValueError("layers must be 'last' or 'all'")
    last = model.head.layers[-1]
    front = list(model.head.layers[:-1])
    trainable = list(last.parameters()) + list(model.scorer.parameters())
    if layers == "all":
        trainable += [prm for layer in front for prm in layer.parameters()] + list(model.type_emb.parameters())
    for prm in trainable:
        prm.requires_grad_(True)

    encoded = [encode_item(tok, it, max_len, head_max_len) for it in items]       # canonical option order
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
    feats: List[Any] = [None] * len(encoded)
    with torch.no_grad():                                      # encoder + first head layer(s), once
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            b = collate_items([[encoded[i] for i in idx]], tok.pad_token_id)
            att = b["attention_mask"].to(device)
            h = model.encoder(input_ids=b["input_ids"].to(device), attention_mask=att).last_hidden_state
            if layers == "last":
                h = h + model.type_emb(b["qtype"].to(device))[:, None, :]
                for layer in front:
                    h = layer(h, src_key_padding_mask=~att.bool())
            for j, i in enumerate(idx):
                feats[i] = h[j, :len(encoded[i]["ids"])].to(torch.float16).cpu()
    log(f"cached_head_train: features of {len(feats)} items computed (training {layers} head layers)")

    def batch_forward(idx):
        n = len(idx)
        L = max(feats[i].shape[0] for i in idx)
        kmax = max(len(encoded[i]["markers"]) for i in idx)
        x = torch.zeros(n, L, feats[idx[0]].shape[1], dtype=torch.float32, device=device)
        pad = torch.ones(n, L, dtype=torch.bool, device=device)
        pos = torch.zeros(n, kmax + 1, dtype=torch.long, device=device)
        mmask = torch.zeros(n, kmax, dtype=torch.bool, device=device)
        target = torch.zeros(n, kmax, dtype=torch.float32, device=device)
        qtype = torch.zeros(n, dtype=torch.long, device=device)
        for j, i in enumerate(idx):
            f, e = feats[i], encoded[i]
            x[j, :f.shape[0]] = f.to(device=device, dtype=torch.float32)
            pad[j, :f.shape[0]] = False
            k = len(e["markers"])
            pos[j, 1:k + 1] = torch.tensor(e["markers"], device=device)
            mmask[j, :k] = True
            target[j, :k] = torch.tensor(e["target"][:k], device=device)
            qtype[j] = int(e["qtype"])
        if layers == "all":
            x = x + model.type_emb(qtype)[:, None, :]
            for layer in front:
                x = layer(x, src_key_padding_mask=pad)
        xn = last.norm1(x)
        sel = torch.gather(x, 1, pos[:, :, None].expand(-1, -1, x.shape[-1]))
        q = torch.gather(xn, 1, pos[:, :, None].expand(-1, -1, x.shape[-1]))
        attn = last.self_attn(q, xn, xn, key_padding_mask=pad, need_weights=False)[0]
        y = sel + last.dropout1(attn)
        y = y + last._ff_block(last.norm2(y))
        logits = model.scorer(y[:, 1:]).squeeze(-1).float().masked_fill(~mmask, -1e4)
        return logits, target, mmask

    # the shortcut must give what the full model gives
    check = order[:: max(1, len(order) // max(1, check_items))][:check_items]
    with torch.no_grad():
        mine, _, mm = batch_forward(check)
        b = collate_items([[encoded[i] for i in check]], tok.pad_token_id)
        ref, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                       b["marker_mask"].to(device), b["qtype"].to(device))
        diff = float((torch.softmax(mine, -1) - torch.softmax(ref.float().masked_fill(~mm, -1e4), -1)).abs().max())
    log(f"cached_head_train: shortcut vs full forward, max probability difference {diff:.2e}")
    if diff > 2e-2:
        raise RuntimeError(f"the last-layer shortcut does not reproduce the model ({diff:.3g}); use method='head'")

    optimizer = torch.optim.AdamW(trainable, lr=config.head_lr, weight_decay=config.weight_decay)
    steps_per_epoch = _math.ceil(len(order) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * config.epochs,
                                                           eta_min=config.min_lr)
    rng = random.Random(config.seed)
    history = []
    for epoch in range(config.epochs):
        last.train()
        model.scorer.train()
        for layer in front:
            layer.train(layers == "all")
        # batches of similar length, in random order
        jittered = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]["ids"]) + rng.uniform(0, 24)))
        batches = [jittered[k:k + batch_size] for k in range(0, len(jittered), batch_size)]
        rng.shuffle(batches)
        total = correct = seen = 0.0
        per_k: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0])
        for idx in batches:
            logits, target, mmask = batch_forward(idx)
            loss = _soft_ce(logits, target, mmask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(idx)
            hits = (logits.argmax(-1) == target.argmax(-1)).tolist()
            for j, i in enumerate(idx):
                per_k[len(encoded[i]["markers"])][0] += hits[j]
                per_k[len(encoded[i]["markers"])][1] += 1
            correct += sum(hits)
            seen += len(idx)
        mean = total / max(seen, 1)
        history.append(mean)
        by_k = " ".join(f"{k}-way {v[0] / max(v[1], 1):.3f}" for k, v in sorted(per_k.items()))
        log("epoch %d/%d mean loss %.4f train accuracy %.4f (%s)" % (epoch + 1, config.epochs, mean, correct / max(seen, 1), by_k))
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean)                          # Laya saves checkpoint_latest: a stopped run keeps its progress
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return history


def train(base_dir, data_dir, out_dir, epochs: int = 7, micro_batch: int = 32, grad_accum: int = 1,
          head_lr: float = 1e-3, method: str = "cached-head", device: str = "auto", seed: int = 0,
          loss: str = "soft-ce", finish_only: bool = False, log=print) -> Dict[str, Any]:
    """Fine-tune ``base_dir`` on ``data_dir/train.jsonl`` (evaluated on ``eval.jsonl``) into ``out_dir``.

    Everything except the optimisation itself is Laya's own ``laya.train.finetune``: reading the data,
    holding out the calibration slice, the evaluation before and after, fitting the temperatures and the
    abstention thresholds, and saving a checkpoint ``laya.load`` reads. ``method``:

    * ``last-layer``: the encoder and the first head layer stay as they are; the last head layer and the
      scorer are trained (``cached_head_train``) — the fastest on a CPU;
    * ``cached-head``: the whole decision head is trained on the cached encoder output, with the same
      shortcuts (``cached_head_train(layers="all")``);
    * ``head``: the whole decision head is trained by Laya's own loop with the encoder frozen (its output
      cached between epochs, ``EncoderCache``);
    * ``full``: the encoder is trained too (a GPU job; the result cannot be packed as a small head).

    ``finish_only``: no optimisation at all, only Laya's calibration, evaluation and saving. Use it with the
    ``checkpoint_latest`` folder of a run that was stopped after its last epoch as ``base_dir`` (the same
    data, so the calibration slice is the same; ``epochs`` etc. are recorded as given).
    """
    if method not in ("last-layer", "cached-head", "head", "full"):
        raise ValueError("method must be last-layer, cached-head, head or full")
    try:
        import laya.train as laya_train
    except ImportError as exc:  # laya is optional for the gate itself
        raise RuntimeError("the 'laya' package is needed to train (pip install laya)") from exc
    data = Path(data_dir)
    config = laya_train.TrainConfig(epochs=epochs, micro_batch=micro_batch, grad_accum=grad_accum, head_lr=head_lr,
                                    freeze_encoder=method != "full", seed=seed, loss=loss,
                                    eval_data=str(data / "eval.jsonl"), log_every=50)
    original_load, original_train = laya_train.load_checkpoint, laya_train.train_model
    caches: List[EncoderCache] = []

    def load_with_cache(model_dir):
        model, tok, cfg = original_load(model_dir)
        cache = EncoderCache(model.encoder)
        cache.install()
        caches.append(cache)
        return model, tok, cfg

    def fast(model, tok, items, cfg, dev, max_len, head_max_len, on_epoch_end=None, parallel=False):
        return cached_head_train(model, tok, items, cfg, dev, max_len, head_max_len, on_epoch_end, parallel,
                                log=log, batch_size=micro_batch, layers="all" if method == "cached-head" else "last")

    def nothing(model, tok, items, cfg, dev, max_len, head_max_len, on_epoch_end=None, parallel=False):
        log("finish_only: no training; calibrating, evaluating and saving the given checkpoint")
        model.eval()
        return []

    if finish_only:
        laya_train.train_model = nothing
    elif method == "head":
        laya_train.load_checkpoint = load_with_cache
    elif method in ("last-layer", "cached-head"):
        laya_train.train_model = fast
    try:
        out = laya_train.finetune(str(data / "train.jsonl"), str(base_dir), str(out_dir), config=config, device=device)
    finally:
        laya_train.load_checkpoint, laya_train.train_model = original_load, original_train
    for c in caches:
        log(f"encoder cache: {len(c.store)} sequences, {c.bytes / 2 ** 20:.0f} MiB, {c.hits} reused, {c.misses} computed")
    return out


# ---------------------------------------------------------------------- packaging
def _encoder_digest(tensors: Dict[str, Any]) -> str:
    """A fingerprint of the encoder weights (names, shapes, dtypes and values)."""
    import torch
    h = hashlib.sha256()
    for name in sorted(k for k in tensors if k.startswith("encoder.")):
        t = tensors[name]
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def pack_head(checkpoint_dir, out_file, base_dir=None) -> Dict[str, Any]:
    """Write the trained decision head (every tensor outside the encoder) and the checkpoint config to one
    file. With ``base_dir`` it is also checked that the encoder is exactly the base's (nothing else
    changed in training, so ``assemble`` will rebuild exactly this checkpoint)."""
    from safetensors.torch import load_file, save_file
    ckpt = Path(checkpoint_dir)
    tensors = load_file(str(ckpt / "model.safetensors"))
    digest = _encoder_digest(tensors)
    if base_dir is not None:
        base = load_file(str(Path(base_dir) / "model.safetensors"))
        if _encoder_digest(base) != digest:
            raise ValueError("the encoder of this checkpoint differs from the base: it was not trained with a frozen "
                             "encoder, so the head alone cannot rebuild it")
    head = {k: v for k, v in tensors.items() if not k.startswith("encoder.")}
    config = json.loads((ckpt / "rl_agent_config.json").read_text(encoding="utf-8"))
    meta = {"format": HEAD_FORMAT, "base_encoder_sha256": digest, "rl_agent_config": json.dumps(config, ensure_ascii=False)}
    for extra in ("calibration.json",):
        if (ckpt / extra).is_file():
            meta[extra] = (ckpt / extra).read_text(encoding="utf-8")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    save_file(head, str(out_file), metadata=meta)
    return {"tensors": len(head), "bytes": Path(out_file).stat().st_size, "base_encoder_sha256": digest}


def assemble(base_dir, head_file, out_dir) -> Dict[str, Any]:
    """Rebuild the fine-tuned checkpoint from the base checkpoint and a packed head (see :func:`pack_head`).
    Refuses a base whose encoder is not the one the head was trained on."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    base, out = Path(base_dir), Path(out_dir)
    with safe_open(str(head_file), framework="pt") as fh:
        meta = fh.metadata() or {}
    if meta.get("format") != HEAD_FORMAT:
        raise ValueError(f"{head_file} is not a packed gate head (format {meta.get('format')!r})")
    tensors = load_file(str(base / "model.safetensors"))
    if _encoder_digest(tensors) != meta["base_encoder_sha256"]:
        raise ValueError("this base checkpoint is not the one the head was trained on (encoder fingerprint differs)")
    head = load_file(str(head_file))
    missing = sorted(k for k in tensors if not k.startswith("encoder.") and k not in head)
    if missing:
        raise ValueError(f"the head file lacks tensors of the base checkpoint: {missing[:5]}")
    tensors.update(head)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("encoder", "tokenizer"):
        if (base / sub).is_dir():
            shutil.copytree(base / sub, out / sub, dirs_exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"))
    (out / "rl_agent_config.json").write_text(json.dumps(json.loads(meta["rl_agent_config"]), indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    for extra in ("calibration.json",):
        if extra in meta:
            (out / extra).write_text(meta[extra], encoding="utf-8")
    return {"out": str(out), "tensors": len(tensors)}


__all__ = ["random_analysis", "synthetic_rows", "build_dataset", "train", "pack_head", "assemble"]
