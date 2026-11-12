"""The compact JSON state that Laya reads: words, not numbers.

Laya is a text encoder with a small context, so the state lists only what is wrong. A key that is absent
means "none"; a measurement that could not be made is listed under ``unk`` and is never silently treated
as "none". The same function builds the state for inference and for the training export, and the
serialisation (:func:`state_to_json`) is fixed, so the text Laya is fine-tuned on is the text it gets later.

    {"doc":"national_id","blur":"moderate","shadow":"mild","crit":{"name":"blur:moderate"},"oth":["blur:moderate"]}

Keys: ``doc`` document type; one key per defect that is present (``cut`` ``res`` ``blur`` ``dark``
``contrast`` ``shadow`` ``glare`` ``persp`` ``curl`` ``noise`` ``obstr``) with its severity word;
``glare_all`` when the ink is washed out over most of the document; ``unk`` defects that could not be
measured; ``crit`` critical fields with a problem as ``cause:severity``; ``oth`` the problems of the other
fields (``cause:severity``, worst first, at most four); ``round`` and ``done`` after a DocRes pass (which
tasks were already applied); ``allowed`` the branches of the decision tree the measurements permit
(``rules.permitted``: ``accept``, ``enhance:<task>``, ``reject:<reason>``). Laya chooses among them — the
judgement calls (a mild defect: OCR as it is, or restore first? a moderate one: restore, or ask for a new
photo? which task, which reason?) — and the router checks the choice against the same conditions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .report import DocumentAnalysis
from .types import DEFECTS, DOCRES_TASKS, INFORMATIONAL_DEFECTS, UNKNOWN, rank

STATE_KEYS = {
    "cut": "cut", "low_resolution": "res", "blur": "blur", "dark": "dark", "low_contrast": "contrast",
    "shadow": "shadow", "glare": "glare", "perspective": "persp", "curl": "curl", "noise": "noise",
    "obstruction": "obstr",
}
assert set(STATE_KEYS) == set(DEFECTS)


def build_state(analysis: DocumentAnalysis, applied_ops: Iterable[str] = (), round_index: int = 0,
                max_chars: int = 700, allowed: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    st: Dict[str, Any] = {"doc": analysis.doc_type or "unknown"}
    if round_index:
        st["round"] = int(round_index)
    done = [o for o in applied_ops if o in DOCRES_TASKS]
    if done:
        st["done"] = done
    sev = analysis.defects
    for d in DEFECTS:
        if rank(sev[d]) >= (2 if d in INFORMATIONAL_DEFECTS else 1):     # background facts only when they stand out
            st[STATE_KEYS[d]] = sev[d]
    if "glare" in st and analysis.global_overexposure:
        st["glare_all"] = True
    unknown = [STATE_KEYS[d] for d in DEFECTS if sev[d] == UNKNOWN]
    if unknown:
        st["unk"] = unknown
    crit: Dict[str, str] = {}
    other: List[str] = []
    for issue in sorted(analysis.field_issues(), key=lambda i: (-rank(i.severity), i.name)):
        what = f"{issue.lost or issue.cause or 'unclear'}:{issue.severity}"
        if issue.critical:
            crit.setdefault(issue.name, what)
        elif what not in other:
            other.append(what)
    if crit:
        st["crit"] = crit
    if other:
        st["oth"] = other[:4]
    if allowed is not None:
        st["allowed"] = list(allowed)          # the branches the measurements permit (rules.permitted)
    return _fit(st, max_chars)


def state_to_json(state: Dict[str, Any]) -> str:
    """The one serialisation used for training and for inference (compact, keys in the order built)."""
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


def _fit(state: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """Drop the least informative parts until the JSON fits ``max_chars`` (a string is never cut in two)."""
    st = dict(state)
    if len(state_to_json(st)) <= max_chars:
        return st
    st.pop("oth", None)
    while len(state_to_json(st)) > max_chars and len(st.get("crit", {})) > 1:
        st["crit"] = dict(list(st["crit"].items())[:-1])           # the least severe critical field goes first
        st["crit_more"] = len(state.get("crit", {})) - len(st["crit"])
    if len(state_to_json(st)) > max_chars:
        st.pop("unk", None)
    if len(state_to_json(st)) > max_chars:
        raise ValueError(f"cannot fit the Laya state into {max_chars} characters")
    return st
