"""Combine the quality results of the side(s) of one document into one analysis and one JSON report.

``DocumentAnalysis`` is what every later stage reads (the Laya state, the rules, the decision tree, the
router), so "the worst severity over the sides", "which side is affected" and "which fields are lost" are
decided in exactly one place.

The JSON report keeps four kinds of value strictly apart, as the design brief asks: a *measured value*,
a 0-100 *quality score* (for people), a *severity word* (for decisions) and *unknown* (could not measure).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .quality import FieldQuality, SideQuality
from .recognition import RecognitionResult
from .types import DEFECTS, UNKNOWN, at_least, rank

SCHEMA_VERSION = 1


def field_loss_reason(f: FieldQuality, loss_at: str) -> Optional[str]:
    """Why the content of a field cannot be recovered by any restoration, as a retake reason, or ``None``.

    A field is lost when it is cut off, covered or in glare at ``loss_at`` or worse, or when its strokes
    have (almost) vanished: something no DocRes task brings back."""
    for cause in ("cut_off", "obstructed", "glare"):
        if at_least(f.causes.get(cause, "none"), loss_at):
            return cause
    if f.causes.get("low_contrast") == "severe":
        for cause in ("obstructed", "glare", "cut_off"):
            if cause in f.causes:
                return cause
        return "other"
    return None


@dataclass
class FieldIssue:
    side: str
    name: str
    critical: bool
    status: str                 # poor | outside
    cause: Optional[str]
    severity: str
    lost: Optional[str]         # retake reason when the content is unrecoverable, else None


@dataclass
class DocumentAnalysis:
    doc_type: Optional[str]
    sides: Dict[str, SideQuality]
    recognitions: Dict[str, RecognitionResult]
    field_loss_at: str = "moderate"
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ defects
    @property
    def defects(self) -> Dict[str, str]:
        """Worst *known* severity of every defect over the sides ("unknown" only if no side could measure it)."""
        out: Dict[str, str] = {}
        for d in DEFECTS:
            best, known = "none", False
            for sq in self.sides.values():
                sev = sq.metrics[d].severity
                if sev == UNKNOWN:
                    continue
                known = True
                if rank(sev) > rank(best):
                    best = sev
            out[d] = best if known else UNKNOWN
        return out

    def sev(self, defect: str) -> str:
        return self.defects[defect]

    def at_least(self, defect: str, level: str) -> bool:
        return at_least(self.sev(defect), level)

    def defect_sides(self, defect: str, level: str = "moderate") -> List[str]:
        """The sides on which ``defect`` is at least ``level``."""
        return [s for s, sq in self.sides.items() if at_least(sq.metrics[defect].severity, level)]

    @property
    def global_overexposure(self) -> bool:
        """Ink is washed out over half of the document or more: overexposure rather than a reflection."""
        return any(sq.metrics["glare"].extra.get("spread", 0.0) >= 0.5 for sq in self.sides.values())

    # ------------------------------------------------------------------ recognition
    @property
    def all_recognized(self) -> bool:
        """Every side was positively recognised and measured against its template."""
        return (bool(self.sides) and all(r.status == "recognized" for r in self.recognitions.values())
                and all(sq.reference == "template" for sq in self.sides.values()))

    @property
    def recognition_evidence(self) -> float:
        """The weakest side's combined recognition score (0..1)."""
        return min((r.confidence for r in self.recognitions.values()), default=0.0)

    def unrecognized_sides(self) -> List[str]:
        return [s for s, r in self.recognitions.items() if r.status != "recognized"]

    # ------------------------------------------------------------------ fields
    def field_issues(self) -> List[FieldIssue]:
        """Fields with a real problem: moderately or severely degraded, outside the frame, or lost."""
        out: List[FieldIssue] = []
        for side, sq in self.sides.items():
            for name, f in sq.fields.items():
                lost = field_loss_reason(f, self.field_loss_at)
                if f.status in ("poor", "outside") or lost is not None:
                    out.append(FieldIssue(side, name, f.critical, f.status, f.cause, f.severity, lost))
        return out

    def critical_issues(self) -> List[FieldIssue]:
        return [i for i in self.field_issues() if i.critical]

    def lost_fields(self) -> List[FieldIssue]:
        return [i for i in self.field_issues() if i.lost is not None]

    def fields_with(self, cause: str, level: str = "moderate", critical_only: bool = False) -> List[str]:
        """Sides holding a field that has ``cause`` at ``level`` or worse."""
        sides = []
        for side, sq in self.sides.items():
            for f in sq.fields.values():
                if (f.critical or not critical_only) and at_least(f.causes.get(cause, "none"), level):
                    sides.append(side)
                    break
        return sides

    # ------------------------------------------------------------------ report
    def to_report(self) -> Dict[str, Any]:
        recognition: Dict[str, Any] = {}
        for side, rec in self.recognitions.items():
            recognition[side] = {
                "status": rec.status, "document_type": rec.doc_type, "side": rec.side,
                "match_confidence": round(rec.confidence, 3),
                "template_used": None if rec.best is None else rec.best.template.key,
                "evidence": rec.evidence, "notes": list(rec.notes),
            }
        metrics: Dict[str, Any] = {}
        severities = self.defects
        for d in DEFECTS:
            worst_side, worst = None, None
            for side, sq in self.sides.items():
                m = sq.metrics[d]
                if worst is None or rank(m.severity) > rank(worst.severity):
                    worst_side, worst = side, m
            entry = worst.to_dict() if worst is not None else {}
            entry["severity"] = severities[d]
            entry["worst_side"] = worst_side if severities[d] not in ("none", UNKNOWN) else None
            metrics[d] = entry
        fields: Dict[str, Any] = {}
        many = len(self.sides) > 1
        for side, sq in self.sides.items():
            for name, f in sq.fields.items():
                entry = {"side": side, **f.to_dict()}
                entry["lost"] = field_loss_reason(f, self.field_loss_at)
                fields[f"{side}.{name}" if many else name] = entry
        return {
            "schema_version": SCHEMA_VERSION,
            "document_type": self.doc_type,
            "document_recognition": recognition,
            "document_geometry": {side: sq.geometry for side, sq in self.sides.items()},
            "quality_metrics": metrics,
            "quality_by_side": {side: {k: m.to_dict() for k, m in sq.metrics.items()} for side, sq in self.sides.items()},
            "field_quality": fields,
            "measurement_reference": {side: sq.reference for side, sq in self.sides.items()},
            "warnings": list(self.warnings) + [f"{side}: {w}" for side, sq in self.sides.items() for w in sq.warnings],
        }
