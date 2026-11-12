"""The contract with Laya: the questions, the compact state, and the client around the real package.

The real model is not needed here: ``FakeLayaAgent`` answers with the structure ``laya`` 0.4.31 returns
from ``agent.predict(state, questions)``.
"""
import json

import pytest

from gate_block import Gate
from gate_block.config import GateConfig
from gate_block.laya_client import (
    DECISIONS,
    QUESTIONS,
    BackendError,
    LayaBackend,
    StaticBackend,
    normalise_answer,
    onehot,
    validate_questions,
)
from gate_block.laya_state import STATE_KEYS, build_state, state_to_json
from gate_block.types import DEFECTS, DOCRES_TASKS, REJECT_REASONS

from .conftest import FakeLayaAgent, fake_analysis, fake_field


# ---------------------------------------------------------------------- the questions
def test_questions_are_valid_and_cover_the_decision_tree():
    validate_questions(QUESTIONS)
    assert tuple(QUESTIONS["decision"]["criteria"]) == DECISIONS == ("accept", "enhance", "reject")
    assert tuple(QUESTIONS["reject_reason"]["criteria"]) == REJECT_REASONS
    assert tuple(QUESTIONS["docres_task"]["criteria"]) == DOCRES_TASKS
    # the five tasks are exactly the --task values of the DocRes repository
    assert set(DOCRES_TASKS) == {"dewarping", "deshadowing", "appearance", "deblurring", "binarization"}


def test_questions_are_short_enough_for_the_per_question_budget():
    # Laya gives each question its own budget for instructions + options (192 tokens on the English model).
    # Words are a safe lower bound for tokens; staying under 100 words leaves a wide margin.
    for qid, spec in QUESTIONS.items():
        words = len(spec["instructions"].split()) + sum(len(f"{k}: {v}".split()) for k, v in spec["criteria"].items())
        assert words < 100, (qid, words)


@pytest.mark.parametrize("bad", [
    {},
    {"q": {"type": "boolean", "instructions": "x", "criteria": {"a": "b", "c": "d"}}},
    {"q": {"type": "choice", "instructions": "", "criteria": {"a": "b", "c": "d"}}},
    {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "b"}}},
    {"q": {"type": "choice", "instructions": "x", "criteria": {"yes": "b", "no": "d"}}},      # boolean words confuse Laya
    {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "", "c": "d"}}},
])
def test_bad_question_sets_are_refused(bad):
    with pytest.raises(ValueError):
        validate_questions(bad)


# ---------------------------------------------------------------------- the state
def test_state_lists_only_what_is_wrong():
    assert build_state(fake_analysis()) == {"doc": "tax_card"}
    st = build_state(fake_analysis(blur="moderate", shadow="mild"))
    assert st == {"doc": "tax_card", "blur": "moderate", "shadow": "mild"}
    assert state_to_json(st) == '{"doc":"tax_card","blur":"moderate","shadow":"mild"}'


def test_state_vocabulary_is_fixed():
    st = build_state(fake_analysis(**{d: "severe" for d in DEFECTS}))
    assert set(st) - {"doc"} == set(STATE_KEYS.values())
    assert set(st.values()) - {"tax_card"} == {"severe"}


def test_background_facts_appear_only_when_they_stand_out():
    assert build_state(fake_analysis(perspective="mild", obstruction="mild")) == {"doc": "tax_card"}
    assert build_state(fake_analysis(perspective="moderate"))["persp"] == "moderate"


def test_unknown_is_never_reported_as_none():
    st = build_state(fake_analysis(blur="unknown"))
    assert "blur" not in st and st["unk"] == ["blur"]


def test_fields_rounds_and_overexposure_in_the_state():
    fields = {"taxpayer_number": fake_field("taxpayer_number", glare="severe"),
              "taxpayer_name": fake_field("taxpayer_name", shadow="moderate"),
              "economic_activity": fake_field("economic_activity", critical=False, obstructed="moderate"),
              "issue_date": fake_field("issue_date", critical=False)}
    st = build_state(fake_analysis(fields=fields, glare="severe", glare_spread=0.8), applied_ops=["deblurring"], round_index=1)
    assert st["crit"] == {"taxpayer_number": "glare:severe", "taxpayer_name": "shadow:moderate"}
    assert st["oth"] == ["obstructed:moderate"] and st["round"] == 1 and st["done"] == ["deblurring"] and st["glare_all"] is True


def test_state_is_deterministic_compact_and_capped():
    a = fake_analysis(blur="moderate", fields={f"field_{i}": fake_field(f"field_{i}", glare="severe") for i in range(40)})
    assert state_to_json(build_state(a)) == state_to_json(build_state(a))
    for cap in (200, 400, 700):
        st = build_state(a, max_chars=cap)
        text = state_to_json(st)
        assert len(text) <= cap and json.loads(text)["blur"] == "moderate"       # the defects survive, fields are summarised
        assert "crit_more" in st


# ---------------------------------------------------------------------- the client
def test_laya_backend_reads_the_real_output_shape():
    agent = FakeLayaAgent(lambda state: {"decision": "enhance", "reject_reason": "blur", "docres_task": "deshadowing"}, confidence=0.8)
    backend = LayaBackend(agent=agent)
    ans = backend.decide('{"doc":"tax_card","shadow":"moderate"}', QUESTIONS)
    assert ans.top("decision") == "enhance" and ans.top("docres_task") == "deshadowing" and not ans.unsure
    assert ans.probs["decision"]["enhance"] == pytest.approx(0.8)
    for row in ans.probs.values():
        assert sum(row.values()) == pytest.approx(1.0)
    state, questions = agent.calls[0]
    assert state == '{"doc":"tax_card","shadow":"moderate"}' and questions is QUESTIONS     # the exact text, the exact questions


def test_a_truncated_state_is_not_an_answer():
    agent = FakeLayaAgent(usage={"truncated": True, "state_tokens_dropped": 37})
    with pytest.raises(BackendError, match="truncated"):
        LayaBackend(agent=agent).decide("{}", QUESTIONS)


def test_abstain_and_review_signals_mark_the_answer_unsure():
    for extra in ({"abstain": True}, {"route": "review"}):
        ans = LayaBackend(agent=FakeLayaAgent(extras={"decision": extra})).decide("{}", QUESTIONS)
        assert ans.unsure
    ans = LayaBackend(agent=FakeLayaAgent(extras={"decision": {"route": "auto", "abstain": False}})).decide("{}", QUESTIONS)
    assert not ans.unsure


def test_broken_model_output_raises_backend_error():
    class Broken:
        def __init__(self, out):
            self.out = out

        def predict(self, state, questions):
            if isinstance(self.out, Exception):
                raise self.out
            return self.out

    good = FakeLayaAgent().predict("{}", QUESTIONS)
    no_question = json.loads(json.dumps(good))
    del no_question["answers"]["docres_task"]
    no_probs = json.loads(json.dumps(good))
    del no_probs["answers"]["decision"]["probabilities"]
    missing_option = json.loads(json.dumps(good))
    del missing_option["answers"]["decision"]["probabilities"]["reject"]
    negative = json.loads(json.dumps(good))
    negative["answers"]["decision"]["probabilities"]["reject"] = -0.2
    all_zero = json.loads(json.dumps(good))
    all_zero["answers"]["decision"]["probabilities"] = {k: 0.0 for k in DECISIONS}
    for out in (RuntimeError("CUDA out of memory"), None, "accept", {}, {"answers": []}, no_question, no_probs, missing_option, negative, all_zero):
        with pytest.raises(BackendError):
            LayaBackend(agent=Broken(out)).decide("{}", QUESTIONS)


def test_missing_laya_package_or_checkpoint_is_reported_clearly(tmp_path):
    with pytest.raises(BackendError):
        LayaBackend(str(tmp_path / "no_such_checkpoint"))


def test_normalise_answer_and_onehot():
    probs = normalise_answer({"decision": {"accept": 2, "enhance": 1, "reject": 1},
                              "reject_reason": onehot("reject")["reject_reason"], "docres_task": onehot("enhance")["docres_task"]})
    assert probs["decision"] == {"accept": 0.5, "enhance": 0.25, "reject": 0.25}
    row = onehot("reject", reason="glare", confidence=0.7)
    assert row["decision"]["reject"] == 0.7 and row["reject_reason"]["glare"] == 0.7
    assert sum(row["reject_reason"].values()) == pytest.approx(1.0)


def test_static_backend_accepts_a_function_of_the_state():
    backend = StaticBackend(lambda state: onehot("reject" if state.get("blur") == "severe" else "accept"))
    assert backend.decide('{"doc":"x","blur":"severe"}', QUESTIONS).top("decision") == "reject"
    assert backend.decide('{"doc":"x"}', QUESTIONS).top("decision") == "accept"


# ---------------------------------------------------------------------- the gate with a (fake) Laya connected
def test_gate_with_laya_connected_end_to_end(photo):
    def policy(state_text):
        state = json.loads(state_text)
        if state.get("shadow"):
            return {"decision": "enhance", "docres_task": "deshadowing", "reject_reason": "shadow"}
        return {"decision": "accept", "docres_task": "deblurring", "reject_reason": "other"}

    agent = FakeLayaAgent(policy)
    gate = Gate(backend=LayaBackend(agent=agent))
    clean = gate.process_image(photo(), expected="tax_card")
    assert clean.status_code == "accept" and clean.decision["decided_by"] == "laya" and clean.decision["laya_choice"] == "accept"
    shaded = gate.process_image(photo(shadow=0.55), expected="tax_card")
    assert shaded.status_code == "enhance:deshadowing" and shaded.decision["decided_by"] == "laya"
    assert shaded.decision["laya_probabilities"]["docres_task"]["deshadowing"] == pytest.approx(0.9)
    cut = gate.process_image(photo(cut=0.10), expected="tax_card")                  # the model says accept; the safety rule says no
    assert cut.status_code == "reject:cut_off" and cut.decision["decided_by"] == "hard_rule"
    assert agent.calls[-1][0] == cut.laya_state_json                               # Laya saw exactly the reported state
    json.dumps(cut.to_dict(), ensure_ascii=False)


def test_low_confidence_threshold_comes_from_the_config(photo):
    cfg = GateConfig.load_dir(GateConfig.default_dir())
    cfg.laya.min_decision_confidence = 0.95
    gate = Gate(config=cfg, backend=LayaBackend(agent=FakeLayaAgent(confidence=0.9)))
    res = gate.process_image(photo(), expected="tax_card")
    assert res.status_code == "accept" and res.decision["decided_by"] == "rules_fallback"
