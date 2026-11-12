"""The decision tree: which outcomes the measurements support.

                                photo of a recognised document
                                             |
               +-----------------------------+-----------------------------+
               |                             |                             |
            ACCEPT                        ENHANCE                       REJECT
       readable as it is          a defect DocRes can fix         unusable: retake
         -> TeleOCR              -> DocRes with ONE task:        -> Iraqi-Arabic message
                                   dewarping | deshadowing |       for ONE reason:
                                   appearance | deblurring |       blur | glare | dark | overexposed |
                                   binarization                    shadow | cut_off | angle | too_small |
                                                                   obstructed | other

Laya *chooses* a branch and a leaf. This module says which branches and leaves the measurements *support*,
and the router never accepts a choice with no evidence behind it. A fourth outcome, REVIEW (send to a
person), is never offered to Laya: Python selects it when no safe decision can be made.

The five task names are exactly the values of ``--task`` in the DocRes repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import yaml

from .config import ConfigError
from .report import DocumentAnalysis
from .types import DOCRES_TASKS, REJECT_REASONS, rank

DEFAULT_TREE = {
    # tie-break between equally serious retake reasons: the first one listed wins
    "reason_priority": ["cut_off", "obstructed", "too_small", "glare", "overexposed", "dark", "blur", "shadow", "angle", "other"],
    # tie-break between equally well supported DocRes tasks
    "operation_order": ["dewarping", "deblurring", "deshadowing", "appearance", "binarization"],
    # tasks the deterministic rules never choose on their own (Laya may, when the evidence is there):
    # binarization turns the document into black and white, which destroys photos, stamps and colours
    "rules_never_choose": ["binarization"],
}
TASK_IS_FOR = {
    "dewarping": "curled or folded paper",
    "deblurring": "blur",
    "deshadowing": "a shadow across the document",
    "appearance": "a dark or low-contrast photo",
    "binarization": "faint text (black-and-white output)",
}


class TreeError(ConfigError):
    pass


@dataclass
class Tree:
    reason_priority: List[str]
    operation_order: List[str]
    rules_never_choose: List[str]

    def validate(self) -> None:
        if sorted(self.reason_priority) != sorted(REJECT_REASONS):
            raise TreeError(f"reason_priority must list each of {sorted(REJECT_REASONS)} exactly once")
        if sorted(self.operation_order) != sorted(DOCRES_TASKS):
            raise TreeError(f"operation_order must list each of {sorted(DOCRES_TASKS)} exactly once")
        for task in self.rules_never_choose:
            if task not in DOCRES_TASKS:
                raise TreeError(f"rules_never_choose: unknown task {task!r}")


def load_tree(config_dir: Union[str, Path, None] = None) -> Tree:
    data = {k: list(v) for k, v in DEFAULT_TREE.items()}
    if config_dir is not None:
        fp = Path(config_dir) / "decision_tree.yaml"
        if fp.is_file():
            with open(fp, "r", encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
            if not isinstance(user, dict):
                raise TreeError(f"{fp} must contain a mapping")
            for key, value in user.items():
                if key not in data:
                    raise TreeError(f"unknown key {key!r} in {fp} (valid: {sorted(data)})")
                if not isinstance(value, list):
                    raise TreeError(f"{key} in {fp} must be a list")
                data[key] = [str(x) for x in value]
    tree = Tree(**data)
    tree.validate()
    return tree


# ---------------------------------------------------------------------- evidence
@dataclass
class Evidence:
    name: str                 # a retake reason or a DocRes task
    strength: float           # severity rank (1 mild, 2 moderate, 3 severe) plus a small tie-breaker
    sides: List[str]          # the sides (front / back / single) where the cause shows


class _Collector:
    def __init__(self) -> None:
        self.items: Dict[str, Evidence] = {}

    def add(self, name: str, severity: str, sides: List[str], bonus: float = 0.0) -> None:
        r = rank(severity)
        if r < 1:
            return
        strength = r + bonus
        prev = self.items.get(name)
        if prev is None:
            self.items[name] = Evidence(name, strength, sorted(set(sides)))
        else:
            prev.strength = max(prev.strength, strength)
            prev.sides = sorted(set(prev.sides) | set(sides))


def reason_evidence(analysis: DocumentAnalysis, tree: Tree) -> List[Evidence]:
    """Reasons for a retake that the measurements support (moderate or worse), most serious first."""
    c = _Collector()
    lost = analysis.lost_fields()

    def lost_sides(reason: str) -> List[str]:
        return sorted({i.side for i in lost if i.lost == reason})

    if analysis.at_least("cut", "mild"):
        c.add("cut_off", max(analysis.sev("cut"), "moderate", key=rank), analysis.defect_sides("cut", "mild"), 0.5)
    if lost_sides("cut_off"):
        c.add("cut_off", "moderate", lost_sides("cut_off"), 0.5)
    if lost_sides("obstructed"):
        c.add("obstructed", "moderate", lost_sides("obstructed"), 0.4)
    if analysis.at_least("obstruction", "severe"):               # a hand / object over a large part of the page
        c.add("obstructed", "severe", analysis.defect_sides("obstruction", "severe"), 0.4)
    if analysis.at_least("low_resolution", "moderate"):
        c.add("too_small", analysis.sev("low_resolution"), analysis.defect_sides("low_resolution"), 0.1)
    glare_reason = "overexposed" if analysis.global_overexposure else "glare"
    if analysis.at_least("glare", "moderate"):
        c.add(glare_reason, analysis.sev("glare"), analysis.defect_sides("glare"), 0.1)
    if lost_sides("glare"):
        c.add(glare_reason, "moderate", lost_sides("glare"), 0.3)
    if analysis.at_least("dark", "moderate"):
        c.add("dark", analysis.sev("dark"), analysis.defect_sides("dark"), 0.1)
    if analysis.at_least("noise", "moderate"):                    # heavy noise is what too little light looks like
        c.add("dark", analysis.sev("noise"), analysis.defect_sides("noise"))
    if analysis.at_least("blur", "moderate"):
        c.add("blur", analysis.sev("blur"), analysis.defect_sides("blur"), 0.1)
    if analysis.fields_with("blur", "moderate", critical_only=True):
        c.add("blur", "moderate", analysis.fields_with("blur", "moderate", critical_only=True))
    if analysis.at_least("shadow", "moderate"):
        c.add("shadow", analysis.sev("shadow"), analysis.defect_sides("shadow"), 0.1)
    if analysis.fields_with("shadow", "moderate", critical_only=True):
        c.add("shadow", "moderate", analysis.fields_with("shadow", "moderate", critical_only=True))
    # a steep camera angle is a reason only when it has done damage (far side soft or low on pixels)
    if analysis.at_least("perspective", "moderate") and (analysis.at_least("low_resolution", "mild") or analysis.at_least("blur", "mild")):
        c.add("angle", analysis.sev("perspective"), analysis.defect_sides("perspective"))
    if analysis.at_least("low_contrast", "moderate"):
        c.add("other", analysis.sev("low_contrast"), analysis.defect_sides("low_contrast"))
    if lost_sides("other"):
        c.add("other", "moderate", lost_sides("other"))
    order = {r: i for i, r in enumerate(tree.reason_priority)}
    return sorted(c.items.values(), key=lambda e: (-e.strength, order[e.name]))


def operation_evidence(analysis: DocumentAnalysis, tree: Tree) -> List[Evidence]:
    """DocRes tasks the measurements support: the defect the task is made for is present (mild or worse)."""
    c = _Collector()
    c.add("deblurring", analysis.sev("blur"), analysis.defect_sides("blur", "mild"))
    c.add("deblurring", "moderate" if analysis.fields_with("blur", "moderate") else "none", analysis.fields_with("blur", "moderate"))
    c.add("deshadowing", analysis.sev("shadow"), analysis.defect_sides("shadow", "mild"))
    c.add("deshadowing", "moderate" if analysis.fields_with("shadow", "moderate") else "none", analysis.fields_with("shadow", "moderate"))
    c.add("dewarping", analysis.sev("curl"), analysis.defect_sides("curl", "mild"))
    c.add("appearance", analysis.sev("dark"), analysis.defect_sides("dark", "mild"))
    c.add("appearance", analysis.sev("low_contrast"), analysis.defect_sides("low_contrast", "mild"))
    c.add("appearance", "moderate" if analysis.fields_with("low_contrast", "moderate") else "none", analysis.fields_with("low_contrast", "moderate"))
    c.add("binarization", analysis.sev("low_contrast"), analysis.defect_sides("low_contrast", "mild"))
    order = {t: i for i, t in enumerate(tree.operation_order)}
    return sorted(c.items.values(), key=lambda e: (-e.strength, order[e.name]))


def describe_tree(tree: Tree) -> Dict[str, object]:
    """A plain description of the tree, for the docs and for ``python -m gate_block tree``."""
    return {
        "branches": {
            "accept": "readable as it is -> TeleOCR",
            "enhance": "a defect DocRes can fix -> DocRes, one task per round",
            "reject": "unusable -> retake, with the Iraqi-Arabic message for the reason",
            "review": "no safe decision -> a person (chosen by Python, never offered to Laya)",
        },
        "reject_reasons_by_priority": list(tree.reason_priority),
        "docres_tasks_by_priority": list(tree.operation_order),
        "task_is_for": dict(TASK_IS_FOR),
        "rules_never_choose": list(tree.rules_never_choose),
    }
