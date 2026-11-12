"""Teaching Laya the gate's decisions: the training set, and packing / rebuilding the trained checkpoint."""
import json
import random

import pytest

from gate_block.laya_client import QUESTIONS
from gate_block.laya_state import STATE_KEYS
from gate_block.tools import laya_training as LT
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


# ---------------------------------------------------------------------- packing and rebuilding
@pytest.fixture()
def fake_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    g = torch.Generator().manual_seed(0)
    encoder = {"encoder.layer.0.weight": torch.randn(8, 8, generator=g).half(), "encoder.embed.weight": torch.randn(20, 8, generator=g).half()}
    base_head = {"head.layers.0.weight": torch.randn(8, 8, generator=g).half(), "temperature": torch.ones(3)}
    tuned_head = {"head.layers.0.weight": torch.randn(8, 8, generator=g).half(), "temperature": torch.tensor([1.5, 1.0, 2.0])}

    def write(d, head, config):
        d.mkdir()
        (d / "encoder").mkdir()
        (d / "tokenizer").mkdir()
        (d / "encoder" / "config.json").write_text("{}", encoding="utf-8")
        (d / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        (d / "rl_agent_config.json").write_text(json.dumps(config), encoding="utf-8")
        st.save_file({**encoder, **head}, str(d / "model.safetensors"))
        return d

    base = write(tmp_path / "base", base_head, {"encoder": "x", "temperature": [1, 1, 1]})
    tuned = write(tmp_path / "tuned", tuned_head, {"encoder": "x", "temperature": [1.5, 1.0, 2.0], "training": {"epochs": 4}})
    return base, tuned, torch, st


def test_pack_and_assemble_rebuild_the_tuned_checkpoint(fake_checkpoints, tmp_path):
    base, tuned, torch, st = fake_checkpoints
    head = tmp_path / "head.safetensors"
    info = LT.pack_head(tuned, head, base_dir=base)
    assert info["tensors"] == 2 and head.stat().st_size < (tuned / "model.safetensors").stat().st_size
    out = tmp_path / "rebuilt"
    LT.assemble(base, head, out)
    a, b = st.load_file(str(tuned / "model.safetensors")), st.load_file(str(out / "model.safetensors"))
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert json.loads((out / "rl_agent_config.json").read_text(encoding="utf-8")) == json.loads((tuned / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert (out / "encoder" / "config.json").is_file() and (out / "tokenizer" / "tokenizer.json").is_file()


def test_assemble_refuses_the_wrong_base(fake_checkpoints, tmp_path):
    base, tuned, torch, st = fake_checkpoints
    head = tmp_path / "head.safetensors"
    LT.pack_head(tuned, head)
    other = st.load_file(str(base / "model.safetensors"))
    other["encoder.layer.0.weight"] = other["encoder.layer.0.weight"] + 1
    st.save_file(other, str(base / "model.safetensors"))
    with pytest.raises(ValueError, match="not the one"):
        LT.assemble(base, head, tmp_path / "rebuilt")
    with pytest.raises(ValueError, match="differs from the base"):
        LT.pack_head(tuned, tmp_path / "h2.safetensors", base_dir=base)
    with pytest.raises(ValueError, match="not a packed gate head"):
        LT.assemble(base, base / "model.safetensors", tmp_path / "rebuilt2")


# ---------------------------------------------------------------------- the frozen-encoder cache
def test_encoder_cache_returns_what_the_encoder_returns_and_reuses_it():
    torch = pytest.importorskip("torch")
    calls = []

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(50, 4)
            self.emb.weight.requires_grad_(False)

        def forward(self, input_ids=None, attention_mask=None):
            calls.append(input_ids.shape[0])
            h = self.emb(input_ids) * attention_mask[..., None] + attention_mask.sum(1)[:, None, None] * 0.01
            return type("Out", (), {"last_hidden_state": h})()

    enc = Encoder()
    reference = Encoder()
    reference.load_state_dict(enc.state_dict())
    cache = LT.EncoderCache(enc)
    cache.install()
    ids = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    first = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    want = reference(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(first[0, :3], want[0, :3]) and torch.allclose(first[1], want[1])
    # the same sequences again, in another batch with other padding: nothing is recomputed
    ids2 = torch.tensor([[8, 9, 10, 11, 12, 0], [5, 6, 7, 0, 0, 0]])
    mask2 = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]])
    n_calls = len(calls)
    again = enc(input_ids=ids2, attention_mask=mask2).last_hidden_state
    assert len(calls) == n_calls and cache.hits == 2
    assert torch.allclose(again[0, :5], want[1], atol=1e-2) and torch.allclose(again[1, :3], want[0, :3], atol=1e-2)
    assert set(enc.state_dict()) == set(reference.state_dict())                 # the weights are not touched
    # a trainable encoder is never cached
    enc.emb.weight.requires_grad_(True)
    enc(input_ids=ids, attention_mask=mask)
    assert len(calls) == n_calls + 1


# ---------------------------------------------------------------------- the cached head trainer
class _TinyTokenizer:
    """Just enough of a tokenizer for Laya's sequence builder: one id per word, stable across calls."""
    mask_token, pad_token_id, cls_token_id, sep_token_id, mask_token_id = "[MASK]", 0, 1, 2, 3

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, **_):
        ids = [self.vocab.setdefault(w, 4 + len(self.vocab) % 400) for w in str(text).replace('"', " ").split()]
        return {"input_ids": ids}


def _tiny_laya():
    torch = pytest.importorskip("torch")
    common = pytest.importorskip("laya.common")
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=64)
            self.emb = torch.nn.Embedding(512, 64)
            self.pos = torch.nn.Embedding(256, 64)          # the option markers are one token: positions tell them apart
            layer = torch.nn.TransformerEncoderLayer(64, 2, 128, 0.0, batch_first=True)
            self.body = torch.nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

        def forward(self, input_ids=None, attention_mask=None):
            x = self.emb(input_ids) + self.pos(torch.arange(input_ids.shape[1]))[None]
            h = self.body(x, src_key_padding_mask=~attention_mask.bool())
            return SimpleNamespace(last_hidden_state=h)

    torch.manual_seed(0)
    return common.DecisionModel(Encoder(), head_layers=2, n_act=2), _TinyTokenizer()


@pytest.mark.parametrize("layers", ["last", "all"])
def test_cached_head_train_reproduces_the_model_and_learns(layers):
    torch = pytest.importorskip("torch")
    laya_train = pytest.importorskip("laya.train")
    model, tok = _tiny_laya()
    q = {"decision": QUESTIONS["decision"]}
    rows = [{"state": json.dumps({"doc": "tax_card", "allowed": allowed}), "questions": q,
             "expected": {"decision": allowed[0].split(":")[0]}}
            for allowed in (["accept"], ["enhance:deblurring"], ["reject:blur"]) for _ in range(8)]
    items, skipped = laya_train.items_from_rows(tok, rows, 256, 128)
    assert len(items) == 24 and not skipped
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    first_before = {k: v.clone() for k, v in model.head.layers[0].state_dict().items()}
    config = laya_train.TrainConfig(epochs=100, head_lr=1e-3, weight_decay=0.0, seed=0)
    logs = []
    history = LT.cached_head_train(model, tok, items, config, torch.device("cpu"), 256, 128, layers=layers,
                                   batch_size=24, log=logs.append)
    assert any("shortcut vs full forward" in m for m in logs)          # checked against Laya's own forward pass
    assert history[-1] < 0.5 * history[0]
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in encoder_before.items())
    first_same = all(torch.equal(v, model.head.layers[0].state_dict()[k]) for k, v in first_before.items())
    assert first_same == (layers == "last")
    # the trained model, run the normal way, answers what it was taught
    from laya.common import collate_items
    enc = [laya_train.encode_item(tok, it, 256, 128) for it in items]
    b = collate_items([enc], tok.pad_token_id)
    with torch.no_grad():
        logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    pred = logits.masked_fill(~b["marker_mask"], -1e4).argmax(-1)
    gold = torch.tensor([max(range(3), key=lambda k: e["target"][k]) for e in enc])
    assert float((pred == gold).float().mean()) >= 0.95
