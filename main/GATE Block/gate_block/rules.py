"""The deterministic rule engine.

It has three jobs:

1. **Safety net.** :func:`hard_reject` lists the conditions under which a photo can never go on, whatever
   Laya says: part of the document is outside the frame, a field is covered or washed out, the photo is
   too damaged to restore.
2. **Fallback.** When Laya is not connected, fails, is unsure, or contradicts the measurements, the router
   uses :func:`rules_decision`, so the gate keeps working and cannot degenerate into "always accept" or
   "always reject".
3. **Baseline.** The same decision, from the same measurements, is the yardstick Laya's fine-tuned
   decisions have to beat.

Everything here reads the severities produced by ``quality.py`` and the thresholds in ``config/rules.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import GateConfig
from .decision_tree import Evidence, Tree, operation_evidence, reason_evidence
from .report import DocumentAnalysis
from .types import BLOCKING_DEFECTS, UNKNOWN, rank

RECOVERABLE = ("blur", "dark", "low_contrast")                     # DocRes has a task for these, up to a point
UNRECOVERABLE = ("cut", "obstruction", "low_resolution", "glare", "noise")   # DocRes has no task for these
_DEFECT_REASON = {"cut": "cut_off", "obstruction": "obstructed", "low_resolution": "too_small", "glare": "glare",
                  "noise": "dark", "blur": "blur", "dark": "dark", "low_contrast": "other", "shadow": "shadow"}


@dataclass
class RulesOutcome:
    decision: str                        # accept | enhance | reject | review
    reason: Optional[str] = None         # retake reason, or the reason for a review
    operation: Optional[str] = None      # DocRes task, for enhance
    why: List[str] = field(default_factory=list)


def _reason_for(defect: str, analysis: DocumentAnalysis) -> str:
    if defect == "glare" and analysis.global_overexposure:
        return "overexposed"
    if defect == "low_resolution" and analysis.at_least("perspective", "moderate"):
        return "angle"                       # the far side is small because of the camera angle: say that
    return _DEFECT_REASON.get(defect, "other")


def hard_reject(analysis: DocumentAnalysis, cfg: GateConfig, tree: Tree) -> Optional[Tuple[str, List[str]]]:
    """``(reason, explanation)`` when the photo must be retaken whatever else is true, else ``None``."""
    r = cfg.rules
    for defect in UNRECOVERABLE:                                   # fixed order: the report is deterministic
        level = r.reject_unrecoverable_at.get(defect)
        if level and analysis.at_least(defect, level):
            return _reason_for(defect, analysis), [f"{defect} is {analysis.sev(defect)}; DocRes cannot fix it (retake from {level})"]
    for defect in RECOVERABLE:                                     # a whole-photo cause explains lost fields best
        if analysis.at_least(defect, r.reject_recoverable_at):
            return _reason_for(defect, analysis), [f"{defect} is {analysis.sev(defect)}: too damaged to restore"]
    for issue in analysis.lost_fields():
        if issue.critical or r.reject_on_noncritical_field_loss:
            reason = issue.lost
            if reason == "glare" and analysis.global_overexposure:
                reason = "overexposed"
            elif reason == "other":                                # strokes vanished: name the cause if one was measured
                reason = next((_DEFECT_REASON[d] for d in ("blur", "dark", "shadow") if analysis.at_least(d, "moderate")), "other")
            kind = "critical field" if issue.critical else "field"
            return reason, [f"{kind} {issue.name} ({issue.side}) cannot be recovered: {issue.lost}"]
    moderate = [d for d in UNRECOVERABLE if d != "obstruction" and analysis.at_least(d, "moderate")]
    if len(moderate) >= r.max_moderate_unrecoverable:
        return _reason_for(moderate[0], analysis), [f"{len(moderate)} unrecoverable defects are moderate or worse: {moderate}"]
    if analysis.at_least("perspective", "severe") and analysis.at_least("low_resolution", "mild"):
        return "angle", ["extreme camera angle and too few pixels on the far side"]
    return None


def acceptable(analysis: DocumentAnalysis) -> bool:
    """Good enough for OCR as it is: every side recognised, no blocking defect beyond mild, no critical field
    in trouble, nothing that could not be measured."""
    if not analysis.all_recognized or analysis.critical_issues():
        return False
    sev = analysis.defects
    if sev["blur"] == UNKNOWN:                                      # no sharp edge anywhere in the text regions
        return False
    return not any(rank(sev[d]) >= 2 for d in BLOCKING_DEFECTS)


def enhance_candidates(analysis: DocumentAnalysis, tree: Tree, applied_ops: Iterable[str] = (),
                       allow_never_choose: bool = True) -> List[Evidence]:
    """DocRes tasks supported by the measurements and not applied yet, best first. Only a recognised
    document may go to the restoration model."""
    if not analysis.all_recognized:
        return []
    done = set(applied_ops)
    out = [e for e in operation_evidence(analysis, tree) if e.name not in done]
    if not allow_never_choose:
        out = [e for e in out if e.name not in tree.rules_never_choose]
    return out


def permitted(analysis: DocumentAnalysis, cfg: GateConfig, tree: Tree, applied_ops: Iterable[str] = (),
              round_index: int = 0) -> List[str]:
    """The branches of the decision tree the measurements permit, as codes: ``accept``, ``enhance:<task>``,
    ``reject:<reason>`` (best supported first within each branch). This is what Laya chooses from and what
    the router enforces afterwards (``router._decide`` applies exactly these conditions):

    * a safety rule applies: only ``reject:<its reason>``;
    * ``accept`` when the photo is good enough for OCR as it is;
    * ``enhance:<task>`` for every DocRes task whose defect was measured and that was not applied yet,
      while DocRes rounds remain;
    * ``reject:<reason>`` for every retake reason that was measured (moderate or worse).

    An unrecognised document has no options: Laya is not asked."""
    if not analysis.all_recognized:
        return []
    hr = hard_reject(analysis, cfg, tree)
    if hr is not None:
        return [f"reject:{hr[0]}"]
    out = ["accept"] if acceptable(analysis) else []
    if round_index < cfg.routing.max_enhance_rounds:
        out += [f"enhance:{e.name}" for e in enhance_candidates(analysis, tree, applied_ops)]
    out += [f"reject:{e.name}" for e in reason_evidence(analysis, tree)]
    return out


def unrecognized_outcome(analysis: DocumentAnalysis, cfg: GateConfig, tree: Tree) -> RulesOutcome:
    """Gate 1 could not confirm the document. Either the photo is bad (then the measured reason explains it
    and the user is asked to retake), or it is a good photo of something else (no document), or it is
    genuinely unclear (a person looks at it). It never goes on to OCR."""
    sides = analysis.unrecognized_sides()
    if analysis.recognition_evidence < cfg.recognition.unsupported_score:
        # next to no sign of a supported document: without a template the quality numbers mean little,
        # so the honest message is "no clear document", whatever else is wrong with the picture
        return RulesOutcome("reject", reason="no_document",
                            why=[f"nothing in the photo matches a template (evidence {analysis.recognition_evidence:.2f}, sides {sides})"])
    hr = hard_reject(analysis, cfg, tree)
    if hr is not None:
        return RulesOutcome("reject", reason=hr[0], why=[f"not recognised ({sides})"] + hr[1])
    reasons = reason_evidence(analysis, tree)
    if reasons:
        return RulesOutcome("reject", reason=reasons[0].name,
                            why=[f"not recognised ({sides}) and the photo has a measured problem: {reasons[0].name}"])
    return RulesOutcome("review", reason="recognition_uncertain",
                        why=[f"some evidence of a supported document ({analysis.recognition_evidence:.2f}) but not enough"])


def rules_decision(analysis: DocumentAnalysis, cfg: GateConfig, tree: Tree, applied_ops: Sequence[str] = (),
                   round_index: int = 0) -> RulesOutcome:
    if not analysis.all_recognized:
        return unrecognized_outcome(analysis, cfg, tree)
    hr = hard_reject(analysis, cfg, tree)
    if hr is not None:
        return RulesOutcome("reject", reason=hr[0], why=hr[1])
    need = rank(cfg.rules.enhance_at)
    cands = [e for e in enhance_candidates(analysis, tree, applied_ops, allow_never_choose=False) if e.strength >= need]
    if cands and round_index < cfg.routing.max_enhance_rounds:
        best = cands[0]
        return RulesOutcome("enhance", operation=best.name,
                            why=[f"{best.name}: evidence {best.strength:g} on {', '.join(best.sides) or 'the document'}"])
    if acceptable(analysis):
        return RulesOutcome("accept", why=["no defect needs action"])
    if analysis.sev("blur") == UNKNOWN and not reason_evidence(analysis, tree):
        return RulesOutcome("review", reason="quality_unmeasurable",
                            why=["sharpness could not be measured: no clear edge in the text regions"])
    # something is still wrong and no (further) restoration is possible
    reasons = reason_evidence(analysis, tree)
    why = [f"still not good enough after {round_index} enhancement round(s)" if round_index else "no restoration applies"]
    if cfg.routing.on_enhancement_failure == "reject":
        return RulesOutcome("reject", reason=reasons[0].name if reasons else "other", why=why)
    return RulesOutcome("review", reason="enhancement_not_enough", why=why)
