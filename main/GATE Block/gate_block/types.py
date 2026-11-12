"""Names shared by every part of the Gate Block.

Anything another block (DocRes, TeleOCR, the app) or Laya depends on by *name* lives here, so a typo
cannot silently create a new category somewhere else.
"""
from __future__ import annotations

from typing import Iterable

# --------------------------------------------------------------------------- severities
SEVERITIES = ("none", "mild", "moderate", "severe")
SEV_RANK = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
UNKNOWN = "unknown"   # a measurement that could not be made; never the same thing as "none"

# --------------------------------------------------------------------------- documents
DOC_TYPES = ("national_id", "tax_card", "business_license")
SIDES = ("front", "back", "single")

# --------------------------------------------------------------------------- DocRes
# Exactly the values accepted by `python inference.py --task ...` in the DocRes repository
# (ZZZHANG-jx/DocRes). Do not rename them. ("end2end" also exists there; the gate never asks for it.)
DOCRES_TASKS = ("dewarping", "deshadowing", "appearance", "deblurring", "binarization")

# --------------------------------------------------------------------------- routing
ROUTES = ("accept", "enhance", "reject", "review")
NEXT_BLOCKS = {"accept": "teleocr", "enhance": "docres", "reject": "retake", "review": "human_review"}

# Reasons for a retake that Laya may choose between (the reject branch of the decision tree).
# The ORDER IS POSITIONAL for a Laya `choice` question: never reorder or rename after fine-tuning.
REJECT_REASONS = (
    "blur", "glare", "dark", "overexposed", "shadow", "cut_off", "angle", "too_small", "obstructed", "other",
)
# Reasons decided by Gate 1 / intake alone (Laya is never asked about these).
GATE1_REASONS = ("no_document", "wrong_document", "wrong_side", "unreadable_file", "image_too_small", "blank_image")
REVIEW_REASONS = ("recognition_uncertain", "quality_unmeasurable", "enhancement_not_enough", "laya_uncertain")

# Every defect the quality gate measures. One severity word per defect goes into the report.
DEFECTS = (
    "cut",             # part of the document is outside the frame
    "low_resolution",  # too few pixels on the document
    "blur",
    "dark",
    "low_contrast",
    "shadow",
    "glare",
    "perspective",     # camera angle
    "curl",            # curled / folded paper (flexible documents only)
    "noise",
    "obstruction",     # finger / hand over the document
)
# Defects that matter only through their consequences, which are measured separately: a camera angle shows
# up as blur or low resolution on the far side, and a finger matters only where it covers a field. They are
# reported, but on their own they neither block "accept" nor justify a retake.
INFORMATIONAL_DEFECTS = ("perspective", "obstruction")
BLOCKING_DEFECTS = tuple(d for d in DEFECTS if d not in INFORMATIONAL_DEFECTS)


def rank(sev: str) -> int:
    """Rank of a severity word; 'unknown' (and anything else) is -1."""
    return SEV_RANK.get(sev, -1)


def at_least(sev: str, level: str) -> bool:
    """True when `sev` is a *known* severity at or above `level` (boundary inclusive)."""
    r = rank(sev)
    return r >= 0 and r >= SEV_RANK[level]


def worst(sevs: Iterable[str]) -> str:
    """Worst known severity of a collection; 'none' when it holds no known value."""
    best = "none"
    for s in sevs:
        if rank(s) > rank(best):
            best = s
    return best
