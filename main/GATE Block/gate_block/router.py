"""Laya recommends, Python validates and routes.

The measurements define which branches of the decision tree are *permitted* for a photo
(``rules.permitted``; Laya reads the same list as ``allowed`` in its state):

    accept    permitted when nothing worse than a mild defect was measured and no critical field is in trouble
    enhance   permitted when a DocRes task has evidence behind it, was not applied yet, and rounds remain
    reject    permitted when a retake reason was measured (moderate or worse)

Every photo of a recognised document with at least one permitted branch goes to Laya (when a Laya backend is
connected), and :func:`decide` checks Laya's choice against exactly those conditions before anything is
executed. Laya makes the judgement calls between permitted branches: a mild defect — OCR as it is, or
restore first? a moderate one — restore, or ask for a new photo? which task, which reason?

and, around that:

    * the hard safety rules cannot be overridden (document outside the frame, a field lost, damage beyond
      repair ...): the photo is rejected with the safety rule's reason whatever Laya recommends, and the
      result records whether Laya agreed;
    * a photo Gate 1 could not recognise, or one with no permitted branch at all (not good enough for OCR,
      nothing left to restore, no measured reason), is never shown to Laya: the deterministic rules settle it;
    * a failed, malformed, truncated, low-confidence or abstaining answer is not used: the rules decide
      (or a person, with ``laya.on_uncertain: review``);
    * when Laya picks a branch that is not permitted, the rules decide and the override is recorded.

So a stuck or untrained model cannot push a bad photo to OCR, cannot reject a clean one, and cannot send
everything to review.

``Decision.decided_by`` says who owned the outcome:

    laya             Laya recommended it and the measurements permit it
    hard_rule        a safety rule imposed it (Laya was not connected, not usable, or recommended otherwise)
    rules            no Laya backend is connected (or Gate 1 did not recognise the document)
    rules_fallback   Laya's answer was not usable
    rules_override   Laya recommended a branch the measurements do not permit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import GateConfig
from .decision_tree import Evidence, Tree, reason_evidence
from .laya_client import QUESTIONS, BackendAnswer, BackendError, DecisionBackend
from .report import DocumentAnalysis
from .rules import (
    RulesOutcome,
    acceptable,
    enhance_candidates,
    hard_reject,
    permitted,
    rules_decision,
)


@dataclass
class Decision:
    route: str                                   # accept | enhance | reject | review
    reason: Optional[str] = None                 # retake reason code, or the reason for a review
    operation: Optional[str] = None              # DocRes task (enhance only)
    confidence: Optional[float] = None           # Laya's probability for its choice (None when the rules decided)
    decided_by: str = "rules"                    # hard_rule | laya | rules | rules_fallback | rules_override
    affected_sides: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)   # other DocRes tasks with evidence, best first
    overrides: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    laya_probabilities: Optional[Dict[str, Dict[str, float]]] = None
    laya_choice: Optional[str] = None            # Laya's own recommendation (accept | enhance | reject), when it was asked
    laya_eligible: bool = True                   # false: Gate 1 did not recognise the document, Laya is never asked
    rules_baseline: Optional[str] = None         # what the deterministic rules alone would have done, e.g. "enhance:deblurring"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {k: getattr(self, k) for k in (
            "route", "reason", "operation", "decided_by", "affected_sides", "alternatives", "overrides", "trace")}
        d["confidence"] = None if self.confidence is None else round(self.confidence, 4)
        d["laya_choice"] = self.laya_choice
        d["laya_eligible"] = self.laya_eligible
        d["rules_baseline"] = self.rules_baseline
        if self.laya_probabilities is not None:
            d["laya_probabilities"] = {q: {k: round(v, 4) for k, v in row.items()} for q, row in self.laya_probabilities.items()}
        return d


def _sides(evidence: List[Evidence], name: Optional[str]) -> List[str]:
    return next((e.sides for e in evidence if e.name == name), [])


def _top(probs) -> Optional[str]:
    return None if not probs else max(probs["decision"], key=probs["decision"].get)


def _from_rules(analysis: DocumentAnalysis, tree: Tree, out: RulesOutcome, decided_by: str, overrides: List[str],
                trace: List[str], probs=None, applied_ops: Sequence[str] = ()) -> Decision:
    sides: List[str] = []
    alternatives: List[str] = []
    if out.decision == "reject":
        sides = _sides(reason_evidence(analysis, tree), out.reason) or analysis.unrecognized_sides()
    elif out.decision == "enhance":
        cands = enhance_candidates(analysis, tree, applied_ops)
        sides = _sides(cands, out.operation)
        alternatives = [e.name for e in cands if e.name != out.operation]
    elif out.decision == "review":
        sides = analysis.unrecognized_sides()
    return Decision(route=out.decision, reason=out.reason, operation=out.operation, decided_by=decided_by,
                    affected_sides=sides, alternatives=alternatives, overrides=overrides, trace=trace + out.why,
                    laya_probabilities=probs, laya_choice=_top(probs))


def _finalise(d: Decision, cfg: GateConfig) -> Decision:
    if d.route == "review" and not cfg.routing.review_enabled:
        d.overrides.append(f"review is disabled: '{d.reason}' became a retake request")
        d.route, d.reason = "reject", "other"
    return d


def decide(analysis: DocumentAnalysis, state_json: str, backend: Optional[DecisionBackend], cfg: GateConfig,
           tree: Tree, applied_ops: Sequence[str] = (), round_index: int = 0) -> Decision:
    """The validated decision for one document (see the module docstring)."""
    rules = rules_decision(analysis, cfg, tree, applied_ops, round_index)
    d = _decide(analysis, state_json, backend, cfg, tree, rules, applied_ops, round_index)
    baseline = _finalise(Decision(rules.decision, reason=rules.reason, operation=rules.operation), cfg)
    d.rules_baseline = baseline.route + (f":{baseline.operation or baseline.reason}" if (baseline.operation or baseline.reason) else "")
    return d


def _decide(analysis: DocumentAnalysis, state_json: str, backend: Optional[DecisionBackend], cfg: GateConfig,
            tree: Tree, rules: RulesOutcome, applied_ops: Sequence[str], round_index: int) -> Decision:

    # 1. a photo Gate 1 could not recognise never reaches Laya or OCR
    if not analysis.all_recognized:
        d = _from_rules(analysis, tree, rules, "rules", [], ["Gate 1 did not recognise the document: Laya is not asked"],
                        applied_ops=applied_ops)
        d.laya_eligible = False
        return _finalise(d, cfg)

    # 2. nothing to choose from (not good enough for OCR, nothing left to restore, no measured reason):
    #    the rules' fallback applies and Laya is not asked
    if not permitted(analysis, cfg, tree, applied_ops, round_index):
        d = _from_rules(analysis, tree, rules, "rules", [],
                        ["the measurements permit no branch to choose from: Laya is not asked"], applied_ops=applied_ops)
        d.laya_eligible = False
        return _finalise(d, cfg)

    # 3. the hard safety rules (evaluated now, enforced below whatever Laya says)
    hr = hard_reject(analysis, cfg, tree)

    def safety_reject(decided_by: str, trace: List[str], overrides: Optional[List[str]] = None, probs=None,
                      confidence: Optional[float] = None) -> Decision:
        return _finalise(Decision("reject", reason=hr[0], confidence=confidence, decided_by=decided_by,
                                  affected_sides=_sides(reason_evidence(analysis, tree), hr[0]),
                                  overrides=overrides or [], trace=trace, laya_probabilities=probs,
                                  laya_choice=_top(probs)), cfg)

    if backend is None:
        if hr is not None:
            return safety_reject("hard_rule", list(hr[1]))
        return _finalise(_from_rules(analysis, tree, rules, "rules", [], ["no Laya backend connected"], applied_ops=applied_ops), cfg)

    def fall_back(why: str, probs=None, confidence: Optional[float] = None) -> Decision:
        if hr is not None:                                   # the safety rule does not need Laya
            return safety_reject("hard_rule", list(hr[1]) + [f"{why}: Laya's answer was not used"], probs=probs)
        if cfg.laya.on_uncertain == "review":
            return _finalise(Decision("review", reason="laya_uncertain", confidence=confidence, decided_by="rules_fallback",
                                      overrides=[why], trace=[why], laya_probabilities=probs, laya_choice=_top(probs)), cfg)
        return _finalise(_from_rules(analysis, tree, rules, "rules_fallback", [why], [f"{why}: the rules decided"],
                                     probs, applied_ops), cfg)

    # 4. ask Laya: every photo of a recognised document with something to choose
    try:
        ans: BackendAnswer = backend.decide(state_json, QUESTIONS)
    except BackendError as exc:
        return fall_back(f"laya_unavailable ({exc})")
    probs = ans.probs
    row = probs["decision"]
    top = max(row, key=row.get)
    conf = float(row[top])
    trace = [f"Laya: {top} ({conf:.2f})"]
    if ans.unsure:
        return fall_back("laya_abstained", probs, conf)
    if conf < cfg.laya.min_decision_confidence:
        return fall_back(f"laya_low_confidence ({conf:.2f} < {cfg.laya.min_decision_confidence})", probs, conf)

    # 5. validation. A safety rule applies: the photo is retaken, with the safety rule's reason
    if hr is not None:
        if top != "reject":
            return safety_reject("hard_rule", trace + list(hr[1]), [f"laya_chose_{top}_against_a_safety_rule"], probs)
        reasons = probs["reject_reason"]
        wanted = max(reasons, key=reasons.get)
        notes = [] if wanted == hr[0] else [f"reason_{wanted}_replaced_by_the_safety_rule_reason_{hr[0]}"]
        return safety_reject("laya", trace + ["confirmed by a safety rule: " + "; ".join(hr[1])], notes, probs, conf)

    # 6. no safety rule applies: is Laya's branch permitted by the measurements?
    def override(why: str) -> Decision:
        return _finalise(_from_rules(analysis, tree, rules, "rules_override", [why], trace, probs, applied_ops), cfg)

    if top == "accept":
        if not acceptable(analysis):
            return override("accept_contradicted_by_measurements")
        return _finalise(Decision("accept", confidence=conf, decided_by="laya", trace=trace, laya_probabilities=probs,
                                  laya_choice=top), cfg)

    if top == "enhance":
        if round_index >= cfg.routing.max_enhance_rounds:
            return override("no_enhancement_round_left")
        cands = enhance_candidates(analysis, tree, applied_ops)
        if not cands:
            return override("enhance_without_supporting_evidence")
        tasks = probs["docres_task"]
        by_name = {e.name: e for e in cands}
        overrides: List[str] = []
        wanted = max(tasks, key=tasks.get)
        if wanted not in by_name:
            overrides.append(f"task_{wanted}_without_supporting_evidence")
        chosen = max(by_name, key=lambda n: (tasks[n], by_name[n].strength))
        return _finalise(Decision("enhance", operation=chosen, confidence=conf, decided_by="laya", overrides=overrides,
                                  affected_sides=by_name[chosen].sides, alternatives=[n for n in by_name if n != chosen],
                                  trace=trace + [f"task {chosen} ({tasks[chosen]:.2f})"], laya_probabilities=probs,
                                  laya_choice=top), cfg)

    # top == "reject"
    evidence = reason_evidence(analysis, tree)
    if not evidence:
        return override("reject_without_measured_reason")
    reasons = probs["reject_reason"]
    by_name = {e.name: e for e in evidence}
    overrides = []
    wanted = max(reasons, key=reasons.get)
    if wanted not in by_name:
        overrides.append(f"reason_{wanted}_without_supporting_evidence")
    chosen = max(by_name, key=lambda n: (reasons[n], by_name[n].strength))
    return _finalise(Decision("reject", reason=chosen, confidence=conf, decided_by="laya", overrides=overrides,
                              affected_sides=by_name[chosen].sides, trace=trace + [f"reason {chosen} ({reasons[chosen]:.2f})"],
                              laya_probabilities=probs, laya_choice=top), cfg)
