"""The Kaggle training script and the dataset publisher (no model is loaded here: the parts around it)."""
import gzip
import json
import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import publish_dataset as PD  # noqa: E402
import train_laya as TL  # noqa: E402

QUESTIONS = {"decision": {"type": "choice", "instructions": "x", "criteria": {"accept": "a", "enhance": "e", "reject": "r"}}}


def _row(state: str, decision: str, origin: str = "synthetic", **extra) -> dict:
    return {"state": state, "questions": QUESTIONS, "expected": {"decision": decision, **extra},
            "meta": {"label_source": "rules", "origin": origin}}


def test_reliability_table_and_ece():
    pairs = [(0.97, True)] * 19 + [(0.97, False)] + [(0.6, True), (0.6, False)]
    rows, ece = TL.reliability(pairs)
    top = next(r for r in rows if r["confidence"] == "0.95-0.99")
    assert top["n"] == 20 and top["accuracy"] == 0.95
    assert ece == pytest.approx((20 / 22) * abs(0.97 - 0.95) + (2 / 22) * abs(0.6 - 0.5), abs=1e-4)


def test_gates():
    good = {"accuracy": {"decision": 0.99}, "false_accepts": 0, "ece": {"decision": 0.01},
            "high_confidence": {"threshold": 0.95, "n": 400, "accuracy": 0.99}}
    assert TL.check_gates(good, TL.GATES) == []
    bad = dict(good, false_accepts=1, high_confidence={"threshold": 0.95, "n": 400, "accuracy": 0.9})
    failed = TL.check_gates(bad, TL.GATES)
    assert any("false accept" in f for f in failed) and any("right only" in f for f in failed)
    few = dict(good, high_confidence={"threshold": 0.95, "n": 5, "accuracy": 0.6})      # too few to judge
    assert TL.check_gates(few, TL.GATES) == []


def test_smoke_subset_keeps_every_decision():
    rows = [_row(f"s{i}", d) for i, d in enumerate(["reject"] * 50 + ["enhance"] * 20 + ["accept"] * 5)]
    sub = TL.small_subset(rows, 12, 0)
    assert len(sub) == 12 and {r["expected"]["decision"] for r in sub} == {"accept", "enhance", "reject"}


def test_push_is_public_only_when_the_gates_pass(tmp_path, monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_repo(self, repo, repo_type, private, exist_ok):
            calls.append(("create", repo, private))

        def update_repo_settings(self, repo, private):
            calls.append(("visibility", private))

        def upload_folder(self, repo_id, folder_path, repo_type, commit_message):
            calls.append(("upload", repo_id, sorted(p.name for p in Path(folder_path).iterdir())))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    report = {"base": {"source": TL.DEFAULT_BASE, "revision": "abc"}, "data": {"source": TL.DEFAULT_DATA, "revision": "def"},
              "stages": [{"loss": "soft-ce", "epochs": 3, "encoder_lr": 2.5e-5, "head_lr": 1e-4, "minutes": 20.0}],
              "evaluation": {"rows": 10, "accuracy": {"decision": 1.0}, "false_accepts": 0, "ece": {"decision": 0.01},
                             "high_confidence": {"threshold": 0.95, "n": 10, "accuracy": 1.0},
                             "reliability": {"decision": [{"confidence": "0.95-0.99", "n": 10, "mean_confidence": 0.97,
                                                           "accuracy": 1.0}]}},
              "gates": TL.GATES}
    (tmp_path / "model.safetensors").write_bytes(b"x")
    card = TL.model_card("RADAXGRS/laya-kyc-gate", report)
    assert "base_model: convaiinnovations/laya-multilingual" in card and "| decision | 1.0 |" in card
    TL.push(tmp_path, "RADAXGRS/laya-kyc-gate", "tok", True, card, report)
    assert ("create", "RADAXGRS/laya-kyc-gate", False) in calls and ("visibility", False) in calls
    assert ("upload", "RADAXGRS/laya-kyc-gate", ["README.md", "model.safetensors", "training_report.json"]) in calls


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
