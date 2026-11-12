"""The decision layer on synthetic measurements (no images): the rules, and the router around Laya.

The central promise tested here: whatever the model answers - stuck on one option, malformed, unsure,
crashing - the gate never accepts a photo the measurements condemn, never rejects or "enhances" a clean
one, and never collapses into a single outcome.
"""
import itertools
import random

import pytest

from gate_block.config import GateConfig
from gate_block.decision_tree import load_tree, operation_evidence, reason_evidence
from gate_block.laya_client import BackendError, StaticBackend, onehot
from gate_block.laya_state import build_state, state_to_json
from gate_block.messages import DEFAULT_MESSAGES_AR
from gate_block.router import decide
from gate_block.rules import acceptable, hard_reject, rules_decision
from gate_block.types import DEFECTS, DOCRES_TASKS, REJECT_REASONS, ROUTES

from .conftest import fake_analysis, fake_field

CFG = GateConfig.from_dict(None)
TREE = load_tree(None)


def rules(analysis, applied=(), rnd=0, cfg=CFG):
    out = rules_decision(analysis, cfg, TREE, applied, rnd)
    return out.decision + (f":{out.operation or out.reason}" if (out.operation or out.reason) else "")


def routed(analysis, backend, applied=(), rnd=0, cfg=CFG):
    state = state_to_json(build_state(analysis, applied, rnd, cfg.laya.max_state_chars))
    return decide(analysis, state, backend, cfg, TREE, applied, rnd)


def code(d):
    return d.route + (f":{d.operation or d.reason}" if (d.operation or d.reason) else "")


class Crashing:
    name = "crashing"

    def decide(self, state_json, questions):
        raise BackendError("the model process died")


# ---------------------------------------------------------------------- the rules, case by case
@pytest.mark.parametrize("severities, expected", [
    ({}, "accept"),
    ({"blur": "mild"}, "enhance:deblurring"),
    ({"blur": "moderate"}, "enhance:deblurring"),
    ({"blur": "severe"}, "reject:blur"),
    ({"shadow": "mild"}, "enhance:deshadowing"),
    ({"shadow": "severe"}, "enhance:deshadowing"),              # DocRes has a task for shadows of any depth
    ({"dark": "mild"}, "enhance:appearance"),
    ({"dark": "severe"}, "reject:dark"),
    ({"low_contrast": "mild"}, "enhance:appearance"),           # never binarization on the rules' own initiative
    ({"low_contrast": "severe"}, "reject:other"),
    ({"curl": "moderate"}, "enhance:dewarping"),
    ({"cut": "mild"}, "reject:cut_off"),
    ({"low_resolution": "mild"}, "accept"),
    ({"low_resolution": "moderate"}, "reject:too_small"),
    ({"glare": "moderate"}, "reject:glare"),                    # nothing restores it, and it is not acceptable
    ({"glare": "severe"}, "reject:glare"),
    ({"noise": "moderate"}, "reject:dark"),                     # heavy noise = too little light
    ({"noise": "severe"}, "reject:dark"),
    ({"glare": "moderate", "noise": "moderate"}, "reject:glare"),
    ({"perspective": "severe"}, "accept"),                      # a tilted but sharp photo is fine
    ({"perspective": "severe", "low_resolution": "mild"}, "reject:angle"),
    ({"obstruction": "moderate"}, "accept"),                    # a thumb on the edge: only covered fields count
    ({"obstruction": "severe"}, "reject:obstructed"),           # ... but not something over a fifth of the document
    ({"blur": "mild", "shadow": "mild"}, "enhance:deblurring"),         # equal evidence: the tree's order
    ({"blur": "mild", "shadow": "moderate"}, "enhance:deshadowing"),    # stronger evidence wins
    ({"curl": "mild", "blur": "mild"}, "enhance:dewarping"),
    ({"blur": "severe", "cut": "mild"}, "reject:cut_off"),              # what DocRes cannot fix is named first
])
def test_rules_cases(severities, expected):
    assert rules(fake_analysis(**severities)) == expected


def test_glare_over_most_of_the_document_is_called_overexposure():
    assert rules(fake_analysis(glare="severe", glare_spread=0.7)) == "reject:overexposed"


def test_field_level_rules():
    lost = fake_analysis(fields={"taxpayer_number": fake_field("taxpayer_number", glare="severe")})
    assert rules(lost) == "reject:glare"
    covered = fake_analysis(fields={"taxpayer_number": fake_field("taxpayer_number", obstructed="moderate")})
    assert rules(covered) == "reject:obstructed"
    outside = fake_analysis(fields={"expiry_date": fake_field("expiry_date", cut_off="moderate")})
    assert rules(outside) == "reject:cut_off"
    # a critical field in a shadow can be restored: DocRes first, and "shadow" if that fails
    shaded = fake_analysis(fields={"taxpayer_number": fake_field("taxpayer_number", shadow="moderate")})
    assert rules(shaded) == "enhance:deshadowing"
    assert rules(shaded, applied=["deshadowing"], rnd=1) == "reject:shadow"
    # a slightly affected field is not a reason for anything
    fair = fake_analysis(fields={"taxpayer_number": fake_field("taxpayer_number", glare="mild")})
    assert rules(fair) == "accept"


def test_noncritical_field_loss_is_configurable():
    a = fake_analysis(fields={"economic_activity": fake_field("economic_activity", critical=False, obstructed="severe")})
    assert rules(a) == "reject:obstructed"
    lenient = GateConfig.from_dict({"rules": {"reject_on_noncritical_field_loss": False}})
    assert rules(a, cfg=lenient) == "accept"


def test_rounds_and_applied_tasks():
    a = fake_analysis(blur="mild", shadow="mild")
    assert rules(a) == "enhance:deblurring"
    assert rules(a, applied=["deblurring"], rnd=1) == "enhance:deshadowing"       # never the same task twice
    assert rules(a, applied=["deblurring", "deshadowing"], rnd=2) == "accept"     # only mild defects left
    b = fake_analysis(blur="moderate")
    assert rules(b, applied=["deblurring"], rnd=1) == "reject:blur"               # restoration did not help
    assert rules(b, rnd=CFG.routing.max_enhance_rounds) == "reject:blur"          # no round left
    review = GateConfig.from_dict({"routing": {"on_enhancement_failure": "review"}})
    assert rules(b, applied=["deblurring"], rnd=1, cfg=review) == "review:enhancement_not_enough"
    none = GateConfig.from_dict({"routing": {"max_enhance_rounds": 0}})
    assert rules(fake_analysis(blur="mild"), cfg=none) == "accept"
    assert rules(b, cfg=none) == "reject:blur"


def test_enhance_threshold_is_configurable():
    cfg = GateConfig.from_dict({"rules": {"enhance_at": "moderate"}})
    assert rules(fake_analysis(blur="mild"), cfg=cfg) == "accept"
    assert rules(fake_analysis(blur="moderate"), cfg=cfg) == "enhance:deblurring"


def test_unmeasurable_sharpness_goes_to_a_person_not_to_ocr():
    assert rules(fake_analysis(blur="unknown")) == "review:quality_unmeasurable"
    assert not acceptable(fake_analysis(blur="unknown"))


def test_unrecognised_documents_never_go_on():
    junk = fake_analysis(status="unsupported", reference="generic", confidence=0.05)
    assert rules(junk) == "reject:no_document"
    unclear = fake_analysis(status="uncertain", reference="generic", confidence=0.40)
    assert rules(unclear) == "review:recognition_uncertain"
    blurry = fake_analysis(status="uncertain", reference="generic", confidence=0.40, blur="severe")
    assert rules(blurry) == "reject:blur"                                         # a bad photo gets its real reason
    for a in (junk, unclear, blurry):
        assert not acceptable(a)


def test_review_can_be_switched_off():
    cfg = GateConfig.from_dict({"routing": {"review_enabled": False}})
    unclear = fake_analysis(status="uncertain", reference="generic", confidence=0.40)
    d = routed(unclear, None, cfg=cfg)
    assert code(d) == "reject:other" and d.overrides


# ---------------------------------------------------------------------- Laya's recommendation is honoured when permitted
def test_laya_decides_inside_what_the_measurements_permit():
    mild = fake_analysis(blur="mild")
    d = routed(mild, StaticBackend(onehot("accept")))
    assert code(d) == "accept" and d.decided_by == "laya" and d.confidence == pytest.approx(0.9)
    d = routed(mild, StaticBackend(onehot("enhance", task="deblurring")))
    assert code(d) == "enhance:deblurring" and d.decided_by == "laya"
    moderate = fake_analysis(blur="moderate", shadow="moderate")
    d = routed(moderate, StaticBackend(onehot("reject", reason="shadow")))
    assert code(d) == "reject:shadow" and d.decided_by == "laya"
    d = routed(moderate, StaticBackend(onehot("enhance", task="deshadowing")))
    assert code(d) == "enhance:deshadowing" and d.decided_by == "laya" and d.alternatives == ["deblurring"]
    faint = fake_analysis(low_contrast="mild")
    d = routed(faint, StaticBackend(onehot("enhance", task="binarization")))       # Laya may pick it, the rules never do
    assert code(d) == "enhance:binarization" and d.decided_by == "laya"
    assert d.rules_baseline == "enhance:appearance" and d.laya_choice == "enhance"


def test_laya_cannot_leave_the_permitted_set():
    clean, moderate = fake_analysis(), fake_analysis(blur="moderate")
    d = routed(clean, StaticBackend(onehot("reject", reason="blur")))
    assert code(d) == "accept" and d.decided_by == "rules_override" and d.overrides == ["reject_without_measured_reason"]
    d = routed(clean, StaticBackend(onehot("enhance", task="deblurring")))
    assert code(d) == "accept" and d.overrides == ["enhance_without_supporting_evidence"]
    d = routed(moderate, StaticBackend(onehot("accept")))
    assert code(d) == "enhance:deblurring" and d.overrides == ["accept_contradicted_by_measurements"]
    d = routed(moderate, StaticBackend(onehot("enhance", task="dewarping")))       # a task with no evidence behind it
    assert code(d) == "enhance:deblurring" and d.decided_by == "laya" and d.overrides == ["task_dewarping_without_supporting_evidence"]
    d = routed(moderate, StaticBackend(onehot("reject", reason="glare")))          # a reason that was not measured
    assert code(d) == "reject:blur" and d.overrides == ["reason_glare_without_supporting_evidence"]
    d = routed(moderate, StaticBackend(onehot("enhance", task="deblurring")), applied=["deblurring"], rnd=1)
    assert code(d) == "reject:blur" and d.decided_by == "rules_override"           # the same task is never run twice
    d = routed(fake_analysis(blur="mild"), StaticBackend(onehot("enhance", task="deblurring")), rnd=CFG.routing.max_enhance_rounds)
    assert code(d) == "accept" and d.overrides == ["no_enhancement_round_left"]


def test_safety_rules_cannot_be_overridden_and_laya_is_still_asked():
    cut = fake_analysis(cut="moderate", blur="moderate")
    for answer in (onehot("accept"), onehot("enhance", task="deblurring")):
        backend = StaticBackend(answer)
        d = routed(cut, backend)
        assert code(d) == "reject:cut_off" and d.decided_by == "hard_rule" and len(backend.calls) == 1
        assert d.overrides and "against_a_safety_rule" in d.overrides[0] and d.confidence is None
    d = routed(cut, StaticBackend(onehot("reject", reason="cut_off")))
    assert code(d) == "reject:cut_off" and d.decided_by == "laya" and d.overrides == [] and d.confidence == pytest.approx(0.9)
    d = routed(cut, StaticBackend(onehot("reject", reason="blur")))                # right branch, the safety rule names the reason
    assert code(d) == "reject:cut_off" and d.decided_by == "laya" and "replaced_by_the_safety_rule_reason_cut_off" in d.overrides[0]
    d = routed(cut, None)
    assert code(d) == "reject:cut_off" and d.decided_by == "hard_rule"
    d = routed(cut, Crashing())
    assert code(d) == "reject:cut_off" and d.decided_by == "hard_rule"


def test_unusable_answers_fall_back_to_the_rules():
    a = fake_analysis(shadow="moderate")
    for backend, needle in ((Crashing(), "laya_unavailable"), (StaticBackend(dict(onehot("accept"), _unsure=True)), "laya_abstained"),
                            (StaticBackend(onehot("accept", confidence=0.5)), "laya_low_confidence"),
                            (StaticBackend({"decision": {"accept": 1.0}}), "laya_unavailable"),               # options missing
                            (StaticBackend(dict(onehot("accept"), decision={"accept": float("nan"), "enhance": 0.0, "reject": 0.0})), "laya_unavailable"),
                            (StaticBackend("accept"), "laya_unavailable")):
        d = routed(a, backend)
        assert code(d) == "enhance:deshadowing" and d.decided_by == "rules_fallback", needle
        assert needle in d.overrides[0]
    ask_a_person = GateConfig.from_dict({"laya": {"on_uncertain": "review"}})
    d = routed(a, Crashing(), cfg=ask_a_person)
    assert code(d) == "review:laya_uncertain"
    d = routed(fake_analysis(cut="mild"), Crashing(), cfg=ask_a_person)            # ... but never instead of a safety rule
    assert code(d) == "reject:cut_off"


def test_laya_is_not_asked_about_unrecognised_photos():
    backend = StaticBackend(onehot("accept"))
    d = routed(fake_analysis(status="uncertain", reference="generic", confidence=0.4), backend)
    assert code(d) == "review:recognition_uncertain" and backend.calls == [] and d.laya_eligible is False
    d = routed(fake_analysis(status="unsupported", reference="generic", confidence=0.05), backend)
    assert code(d) == "reject:no_document" and backend.calls == []


def test_the_state_laya_sees_is_the_state_that_was_built():
    backend = StaticBackend(onehot("accept"))
    a = fake_analysis(blur="mild")
    state = state_to_json(build_state(a, (), 0, CFG.laya.max_state_chars))
    decide(a, state, backend, CFG, TREE)
    assert backend.calls == [state]


# ---------------------------------------------------------------------- invariants over many random documents
SEVS = ("none", "none", "none", "mild", "moderate", "severe")
FIELD_CAUSES = ("cut_off", "obstructed", "glare", "shadow", "blur", "low_contrast")


def random_analysis(rng: random.Random):
    severities = {d: rng.choice(SEVS) for d in DEFECTS if rng.random() < 0.35}
    if rng.random() < 0.03:
        severities["blur"] = "unknown"
    fields = {}
    for name, critical in (("taxpayer_number", True), ("taxpayer_name", True), ("economic_activity", False)):
        if rng.random() < 0.15:
            fields[name] = fake_field(name, critical, **{rng.choice(FIELD_CAUSES): rng.choice(("mild", "moderate", "severe"))})
        else:
            fields[name] = fake_field(name, critical)
    recognised = rng.random() < 0.9
    return fake_analysis(fields=fields, status="recognized" if recognised else rng.choice(("uncertain", "unsupported")),
                         reference="template" if recognised else "generic",
                         confidence=rng.uniform(0.6, 1.0) if recognised else rng.uniform(0.0, 0.5),
                         glare_spread=rng.choice((0.0, 0.1, 0.7)), **severities)


def backends():
    out = {"none": None, "crashing": Crashing(), "unsure": StaticBackend(dict(onehot("accept"), _unsure=True))}
    for decision in ("accept", "enhance", "reject"):
        out[f"stuck_{decision}"] = StaticBackend(onehot(decision, reason="glare", task="binarization", confidence=0.99))
    rng = random.Random(7)
    out["random"] = StaticBackend(lambda state: onehot(rng.choice(("accept", "enhance", "reject")), rng.choice(REJECT_REASONS),
                                                       rng.choice(DOCRES_TASKS), rng.choice((0.4, 0.7, 0.95))))
    return out


def test_invariants_hold_for_every_backend():
    rng = random.Random(2026)
    cases = [(random_analysis(rng), rng.choice((0, 0, 1, 2)), rng.choice(((), ("deblurring",), ("deshadowing", "appearance"))))
             for _ in range(1500)]
    for name, backend in backends().items():
        seen = set()
        for analysis, rnd, applied in cases:
            d = routed(analysis, backend, applied, rnd)
            seen.add(d.route)
            ctx = f"{name}: {analysis.defects} round {rnd} applied {applied} -> {code(d)} ({d.decided_by})"
            assert d.route in ROUTES, ctx
            hr = hard_reject(analysis, CFG, TREE)
            if not analysis.all_recognized:
                assert d.route in ("reject", "review"), ctx                      # never OCR, never DocRes
            elif hr is not None:
                assert code(d) == f"reject:{hr[0]}", ctx                         # safety rules always win
            if d.route == "accept":
                assert acceptable(analysis), ctx                                 # never accept against the measurements
            elif d.route == "enhance":
                assert d.operation in DOCRES_TASKS and d.operation not in applied, ctx
                assert d.operation in {e.name for e in operation_evidence(analysis, TREE)}, ctx
                assert rnd < CFG.routing.max_enhance_rounds and hr is None, ctx
            elif d.route == "reject":
                assert d.reason in DEFAULT_MESSAGES_AR and d.reason != "review", ctx
                if analysis.all_recognized and d.reason != "other":
                    assert d.reason in {e.name for e in reason_evidence(analysis, TREE)}, ctx   # a measured reason
            else:
                assert d.reason in ("recognition_uncertain", "quality_unmeasurable", "enhancement_not_enough", "laya_uncertain"), ctx
            assert d.rules_baseline == rules(analysis, applied, rnd), ctx
        assert {"accept", "enhance", "reject"} <= seen, f"{name} never produced {({'accept', 'enhance', 'reject'} - seen)}"


def test_a_clean_document_is_accepted_whatever_the_model_says():
    for name, backend in backends().items():
        d = routed(fake_analysis(), backend)
        assert code(d) == "accept", name


def test_a_condemned_document_is_rejected_whatever_the_model_says():
    for severities in ({"cut": "severe"}, {"blur": "severe"}, {"glare": "severe"}, {"low_resolution": "severe"}, {"dark": "severe"}):
        for name, backend in backends().items():
            assert routed(fake_analysis(**severities), backend).route == "reject", (name, severities)


def test_rules_are_deterministic():
    rng = random.Random(5)
    for _ in range(200):
        a = random_analysis(rng)
        assert rules(a) == rules(a)


def test_every_single_defect_and_severity_has_a_defined_outcome():
    for defect, sev in itertools.product(DEFECTS, ("mild", "moderate", "severe", "unknown")):
        d = routed(fake_analysis(**{defect: sev}), None)
        assert d.route in ROUTES and (d.route != "reject" or d.reason in REJECT_REASONS), (defect, sev)


# ---------------------------------------------------------------------- what Laya is offered = what is enforced
def test_permitted_is_exactly_what_the_router_lets_laya_decide():
    """``rules.permitted`` (listed in Laya's state as ``allowed``) and the router's validation agree: every
    permitted code chosen by Laya is executed as chosen, and every other code is not."""
    import random as _random

    from gate_block import Gate
    from gate_block.rules import permitted
    from gate_block.tools.laya_training import random_analysis
    from gate_block.types import DOCRES_TASKS, REJECT_REASONS

    gate = Gate()
    rng = _random.Random(11)
    every = ["accept"] + [f"enhance:{t}" for t in DOCRES_TASKS] + [f"reject:{r}" for r in REJECT_REASONS]
    checked = 0
    for _ in range(150):
        a = random_analysis(rng, gate.pack, CFG)
        rnd = rng.choice((0, 0, 1, 2))
        applied = rng.sample([t for t in DOCRES_TASKS if t != "binarization"], rnd)
        allowed = permitted(a, CFG, TREE, applied, rnd)
        state = build_state(a, applied, rnd, CFG.laya.max_state_chars, allowed=allowed)
        assert state["allowed"] == allowed and len(state_to_json(state)) <= CFG.laya.max_state_chars
        for c in every:
            route, _, detail = c.partition(":")
            probs = onehot(route, reason=detail or "other", task=detail or "deblurring", confidence=0.95)
            d = routed(a, StaticBackend(probs), applied, rnd)
            if c in allowed:
                assert d.decided_by == "laya" and code(d) == c, (c, allowed, code(d), d.decided_by)
            else:
                assert code(d) != c or d.decided_by != "laya", (c, allowed, code(d), d.decided_by)
            checked += 1
    assert checked == 150 * len(every)
