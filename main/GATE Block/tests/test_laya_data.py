"""Laya's training data: the gate's decisions as rows, and the dataset in datasets/laya/ with its card."""
import gzip
import json
import random
import sys
from pathlib import Path

import pytest

from gate_block.laya_client import QUESTIONS
from gate_block.laya_state import STATE_KEYS
from gate_block.tools import laya_data as LT
from gate_block.types import DOCRES_TASKS, REJECT_REASONS


def test_random_analyses_cover_every_document_and_side(gate):
    rng = random.Random(1)
    seen = set()
    for _ in range(60):
        a = LT.random_analysis(rng, gate.pack, gate.cfg)
        assert a.all_recognized and a.doc_type in ("national_id", "tax_card", "business_license")
        assert set(a.sides) == ({"front", "back"} if a.doc_type == "national_id" else {"single"})
        for side, sq in a.sides.items():
            assert set(sq.fields) == set(gate.pack.get(a.doc_type, side).fields)     # the real field names
        seen.add(a.doc_type)
    assert len(seen) == 3


def test_enhance_choice_rows_are_the_ones_where_the_task_matters(gate):
    rows = [r for _, r in LT.synthetic_rows(gate, 40, seed=11, enhance_choice=True)]
    assert len(rows) == 40
    for r in rows:
        allowed = json.loads(r["state"])["allowed"]
        assert r["expected"]["decision"] == "enhance" and r["meta"]["origin"] == "synthetic_enhance"
        assert sum(a.startswith("enhance:") for a in allowed) >= 2
        assert "enhance:" + r["expected"]["docres_task"] in allowed


def test_synthetic_rows_are_labelled_by_the_rules_and_cover_every_answer(gate):
    rows = [r for _, r in LT.synthetic_rows(gate, 3000, seed=3)]
    decisions = {r["expected"]["decision"] for r in rows}
    reasons = {r["expected"].get("reject_reason") for r in rows} - {None}
    tasks = {r["expected"].get("docres_task") for r in rows} - {None}
    assert decisions == {"accept", "enhance", "reject"}
    assert reasons >= set(REJECT_REASONS) - {"other"}
    assert tasks == set(DOCRES_TASKS) - set(gate.tree.rules_never_choose)       # the rules never pick binarization
    for r in rows:
        assert r["questions"] == QUESTIONS
        state = json.loads(r["state"])
        assert set(state) <= {"doc", "round", "done", "glare_all", "unk", "crit", "crit_more", "oth", "allowed"} | set(STATE_KEYS.values())
        assert state["allowed"] and (r["expected"]["decision"] + (":" + (r["expected"].get("docres_task") or r["expected"].get("reject_reason")) if r["expected"]["decision"] != "accept" else "")) in state["allowed"]
        exp = r["expected"]
        assert ("reject_reason" in exp) == (exp["decision"] == "reject")
        assert ("docres_task" in exp) == (exp["decision"] == "enhance")
        for qid, label in exp.items():
            assert label in QUESTIONS[qid]["criteria"]


def test_synthetic_labels_are_the_rules_decision(gate, monkeypatch):
    """Every label is what ``rules_decision`` says for the generated analysis and round."""
    seen = []
    real = LT.rules_decision

    def spy(analysis, cfg, tree, applied, rnd):
        out = real(analysis, cfg, tree, applied, rnd)
        seen.append((out, LT.permitted(analysis, cfg, tree, applied, rnd)))
        return out

    monkeypatch.setattr(LT, "rules_decision", spy)
    rows = [r for _, r in LT.synthetic_rows(gate, 300, seed=6)]
    kept = [o for o, allowed in seen if o.decision != "review" and allowed]      # Laya is asked only with options
    assert len(rows) == len(kept)
    for row, out in zip(rows, kept):
        assert row["expected"] == LT.expected_from(out.decision, out.reason, out.operation)
    again = [r for _, r in LT.synthetic_rows(gate, 300, seed=6)]
    assert again == rows                                                            # deterministic


def test_build_dataset_dedups_states_and_never_splits_one_state(gate, tmp_path):
    summary = LT.build_dataset(gate, tmp_path, synthetic=1500, simulate=0, eval_fraction=0.2, seed=2, max_copies=2,
                               echo=lambda *_: None)
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    evals = [json.loads(line) for line in (tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["train"]["rows"] == len(train) and summary["eval"]["rows"] == len(evals) and evals
    assert not ({r["state"] for r in train} & {r["state"] for r in evals})       # evaluation states are unseen
    counts = {}
    for r in train + evals:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
        assert set(r) == {"state", "questions", "expected", "meta"}
    assert max(counts.values()) <= 2
    labels = {}
    for r in train + evals:                                                      # one state, one label
        assert labels.setdefault(r["state"], json.dumps(r["expected"], sort_keys=True)) == json.dumps(r["expected"], sort_keys=True)
    with pytest.raises(ValueError):
        from gate_block import Gate, StaticBackend, onehot
        LT.build_dataset(Gate(backend=StaticBackend(onehot("accept"))), tmp_path, synthetic=10)


# ---------------------------------------------------------------------- the dataset folder and its card
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "laya"
sys.path.insert(0, str(DATA_DIR))
import publish_dataset as PD  # noqa: E402

_Q = {"decision": {"type": "choice", "instructions": "x", "criteria": {"accept": "a", "enhance": "e", "reject": "r"}}}


def _row(state: str, decision: str, origin: str = "synthetic", **extra) -> dict:
    return {"state": state, "questions": _Q, "expected": {"decision": decision, **extra},
            "meta": {"label_source": "rules", "origin": origin}}


def test_dataset_card_is_built_from_the_data(tmp_path):
    data = tmp_path / "v1"
    data.mkdir()
    train = [_row("a", "reject", reject_reason="blur"), _row("b", "enhance", "synthetic_enhance", docres_task="deblurring"),
             dict(_row("c", "accept", "photo"), meta={"origin": "photo", "sample": "photo:real/table"})]
    evals = [_row("d", "accept")]
    (data / "train.jsonl.gz").write_bytes(gzip.compress("\n".join(json.dumps(r) for r in train).encode()))
    (data / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in evals), encoding="utf-8")
    (data / "provenance.json").write_text(json.dumps({"command": "python -m gate_block laya-data --seed 0"}))
    out = PD.build(data, tmp_path / "stage", json.loads((data / "provenance.json").read_text()))
    assert out == {"train": 3, "validation": 1, "folder": str(tmp_path / "stage")}
    text = (tmp_path / "stage" / "README.md").read_text(encoding="utf-8")
    assert "not** human annotations" in text and "`photo:real/table`" in text
    assert "| `synthetic_enhance` | 1 | 0 |" in text and "laya-data --seed 0" in text
    assert len((tmp_path / "stage" / "train.jsonl").read_text().splitlines()) == 3


def test_dataset_with_a_state_in_both_splits_is_refused(tmp_path):
    data = tmp_path / "v1"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps(_row("same", "accept")))
    (data / "eval.jsonl").write_text(json.dumps(_row("same", "accept")))
    with pytest.raises(SystemExit, match="both train and validation"):
        PD.build(data, tmp_path / "stage", {})


def test_the_shipped_dataset_is_split_cleanly_and_labelled_by_the_rules(tmp_path):
    out = PD.build(DATA_DIR / "v1", tmp_path / "stage", {})
    assert out["train"] > 3000 and out["validation"] > 500
    train = [json.loads(line) for line in (tmp_path / "stage" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["meta"]["label_source"] for r in train} == {"rules"}
    assert {r["expected"]["decision"] for r in train} == {"accept", "enhance", "reject"}
