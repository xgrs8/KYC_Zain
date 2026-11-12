"""Block 1 - the Gate.

    upload -> Gate 1 (is it the right document?) -> [national ID: merge front + back]
           -> quality gate (measure) -> JSON report -> Laya (choose) -> Python (validate and route)
           -> TeleOCR | DocRes | retake message | human review

Entry points of :class:`Gate`:

    process_image(src, expected=...)   one photo: a tax card, a business licence, or one side of an ID
    process_id(front, back)            national ID: Gate 1 on both uploads, merge, quality gate on the merge
    process_merged_id(src)             an ID that is already merged (the dataset's ID_23.png, a DocRes output)
    process_submission({...})          id_front + id_back + tax_card + business_license in one call
    recheck(enhanced, previous)        verify a DocRes output before it goes on to OCR
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from .config import GateConfig
from .decision_tree import Tree, load_tree
from .id_merge import MergedID, merge_id, split_merged
from .imageio import ImageLoadError, LoadedImage, load_image, to_bgr
from .laya_client import DecisionBackend
from .laya_state import build_state, state_to_json
from .messages import load_messages, message_for
from .quality import SideQuality, analyze_side
from .recognition import RecognitionResult, recognize, transfer_recognition
from .report import DocumentAnalysis
from .router import Decision, decide
from .rules import acceptable, permitted
from .templates import TemplatePack
from .types import DOC_TYPES, GATE1_REASONS, NEXT_BLOCKS, rank

Source = Union[str, "os.PathLike[str]", bytes, bytearray, np.ndarray]

_LOAD_ERROR_MESSAGE = {"unreadable": "unreadable_file", "too_large": "unreadable_file",
                       "too_small": "image_too_small", "blank": "blank_image"}
# the defects each DocRes task is meant to reduce (used to verify an enhancement)
OP_TARGETS = {
    "deblurring": ("blur",), "deshadowing": ("shadow",), "dewarping": ("curl",),
    "appearance": ("dark", "low_contrast"), "binarization": ("low_contrast",),
}


@dataclass
class GateResult:
    route: str                                   # accept | enhance | reject | review
    next_block: str                              # teleocr | docres | retake | human_review
    stage: str                                   # intake | gate1 | quality  (where the outcome was decided)
    status_code: str                             # accept | enhance:deblurring | reject:blur | review:...
    document_type: Optional[str] = None
    reason: Optional[str] = None
    operation: Optional[str] = None
    message_ar: Optional[str] = None             # Iraqi-Arabic message (reject and review)
    message_ar_with_side: Optional[str] = None   # the same, naming the ID side when only one is affected
    affected_sides: List[str] = field(default_factory=list)
    docres_request: Optional[Dict[str, Any]] = None
    teleocr_request: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    gate1: Dict[str, Any] = field(default_factory=dict)
    report: Optional[Dict[str, Any]] = None
    laya_state: Optional[Dict[str, Any]] = None
    laya_state_json: Optional[str] = None
    image_path: Optional[str] = None             # the image this result is about, when it is on disk
    kind: str = "photo"                          # photo | merged_id
    regions: Optional[Dict[str, List[int]]] = None   # merged ID: pixel rows [y0, y1) of each side
    applied_ops: List[str] = field(default_factory=list)
    round_index: int = 0
    acceptable_as_is: bool = False               # true: the image could go to OCR without any enhancement
    recheck: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    image: Optional[np.ndarray] = None           # the image to hand on (BGR); never serialised

    _FIELDS = ("route", "next_block", "stage", "status_code", "document_type", "reason", "operation", "message_ar",
               "message_ar_with_side", "affected_sides", "docres_request", "teleocr_request", "decision", "gate1",
               "report", "laya_state", "laya_state_json", "image_path", "kind", "regions", "applied_ops",
               "round_index", "acceptable_as_is", "recheck", "warnings", "timings_ms")

    def to_dict(self, include_report: bool = True) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self._FIELDS}
        if not include_report:
            d.pop("report")
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateResult":
        """Rebuild a result from its JSON form (to recheck a DocRes output in a later process)."""
        missing = [k for k in ("route", "next_block", "stage", "status_code") if k not in data]
        if missing:
            raise ValueError(f"not a gate result: missing {missing}")
        return cls(**{k: data[k] for k in cls._FIELDS if k in data})


def _gate1_summary(rec: RecognitionResult) -> Dict[str, Any]:
    return {"status": rec.status, "document_type": rec.doc_type, "side": rec.side,
            "match_confidence": round(rec.confidence, 3),
            "template_used": None if rec.best is None else rec.best.template.key, "notes": list(rec.notes)}


class Gate:
    """The Gate Block. Create it once (loading the templates takes about a second) and reuse it."""

    def __init__(self, config: Optional[GateConfig] = None, templates_dir: Union[str, Path, None] = None,
                 backend: Optional[DecisionBackend] = None, config_dir: Union[str, Path, None] = None):
        cdir: Optional[Path] = Path(config_dir) if config_dir else GateConfig.default_dir()
        if not cdir.is_dir():
            if config_dir:
                raise FileNotFoundError(f"config directory not found: {cdir}")
            cdir = None
        if config is None:
            config = GateConfig.load_dir(cdir) if cdir else GateConfig.from_dict(None)
        self.cfg = config
        self.config_dir = cdir
        self.messages = load_messages(cdir)
        self.tree: Tree = load_tree(cdir)
        self.backend = backend
        root = cdir.parent if cdir else Path(__file__).resolve().parent.parent
        tdir = Path(templates_dir) if templates_dir else Path(self.cfg.paths.templates_dir)
        if not tdir.is_absolute() and not tdir.is_dir():
            tdir = root / tdir
        self.pack = TemplatePack.load(tdir, self.cfg)

    # ------------------------------------------------------------------ small helpers
    def _reject_early(self, code: str, stage: str, gate1: Dict[str, Any], sides: Sequence[str] = (),
                      doc_type: Optional[str] = None, t0: Optional[float] = None) -> GateResult:
        sides = list(sides)
        return GateResult(route="reject", next_block=NEXT_BLOCKS["reject"], stage=stage, status_code=f"reject:{code}",
                          document_type=doc_type, reason=code, message_ar=self.messages[code],
                          message_ar_with_side=message_for(code, self.messages, sides), affected_sides=sides,
                          gate1=gate1, timings_ms={"total": round((time.perf_counter() - t0) * 1000.0, 1)} if t0 else {})

    @staticmethod
    def _save(img: np.ndarray, workdir: Union[str, Path, None], name: str) -> Optional[str]:
        if workdir is None:
            return None
        Path(workdir).mkdir(parents=True, exist_ok=True)
        path = Path(workdir) / name
        if not cv2.imwrite(str(path), img):
            raise OSError(f"could not write {path}")
        return str(path)

    def _finish(self, analysis: DocumentAnalysis, gate1: Dict[str, Any], image_path: Optional[str],
                image: Optional[np.ndarray], applied_ops: Sequence[str], round_index: int, kind: str,
                regions: Optional[Dict[str, List[int]]], t0: float, timings: Dict[str, float]) -> GateResult:
        state = build_state(analysis, applied_ops, round_index, self.cfg.laya.max_state_chars,
                            allowed=permitted(analysis, self.cfg, self.tree, applied_ops, round_index))
        state_json = state_to_json(state)
        t_dec = time.perf_counter()
        dec: Decision = decide(analysis, state_json, self.backend, self.cfg, self.tree, applied_ops, round_index)
        timings["decision"] = round((time.perf_counter() - t_dec) * 1000.0, 1)
        report = analysis.to_report()
        code = dec.reason if dec.route in ("reject", "review") else dec.operation
        res = GateResult(
            route=dec.route, next_block=NEXT_BLOCKS[dec.route],
            stage="gate1" if dec.reason in GATE1_REASONS or dec.reason == "recognition_uncertain" else "quality",
            status_code=dec.route + (f":{code}" if code else ""), document_type=analysis.doc_type, reason=dec.reason,
            operation=dec.operation, affected_sides=list(dec.affected_sides), decision=dec.to_dict(), gate1=gate1,
            report=report, laya_state=state, laya_state_json=state_json, image_path=image_path, kind=kind,
            regions=regions, applied_ops=list(applied_ops), round_index=round_index,
            acceptable_as_is=acceptable(analysis), warnings=list(report["warnings"]), image=image)
        if dec.route == "reject":
            res.message_ar = self.messages[dec.reason]
            res.message_ar_with_side = message_for(dec.reason, self.messages, dec.affected_sides)
        elif dec.route == "review":
            res.message_ar = res.message_ar_with_side = self.messages["review"]
        elif dec.route == "enhance":
            res.docres_request = {
                "task": dec.operation, "image": image_path, "kind": kind, "round": round_index + 1,
                "applied_before": list(applied_ops), "alternatives": list(dec.alternatives),
                "max_side": self.cfg.routing.docres_max_side,
                # merged ID: run DocRes on the affected side(s) only and leave the white band untouched
                "regions": ({s: regions[s] for s in dec.affected_sides if s in regions} or dict(regions)) if regions else None,
            }
        else:
            res.teleocr_request = {"image": image_path, "kind": kind, "document_type": analysis.doc_type}
        timings["total"] = round((time.perf_counter() - t0) * 1000.0, 1)
        res.timings_ms = timings
        return res

    # ------------------------------------------------------------------ one photo
    def process_image(self, src: Source, expected: Optional[str] = None, workdir: Union[str, Path, None] = None,
                      applied_ops: Sequence[str] = (), round_index: int = 0, name: str = "upload") -> GateResult:
        """Gate a single photo: a tax card, a business licence, or one side of an ID.

        ``expected`` (national_id | tax_card | business_license) makes Gate 1 refuse a photo that is
        recognised as a different document."""
        if expected is not None and expected not in DOC_TYPES:
            raise ValueError(f"expected must be one of {DOC_TYPES}, got {expected!r}")
        t0 = time.perf_counter()
        try:
            loaded = load_image(src, self.cfg.image)
        except ImageLoadError as exc:
            return self._reject_early(_LOAD_ERROR_MESSAGE[exc.code], "intake", {"error": str(exc)}, doc_type=expected, t0=t0)
        rec = recognize(loaded, self.pack, self.cfg)
        t1 = time.perf_counter()
        gate1 = {"upload": _gate1_summary(rec)}
        if rec.status == "unsupported":
            return self._reject_early("no_document", "gate1", gate1, doc_type=expected, t0=t0)
        if expected and rec.status == "recognized" and rec.doc_type != expected:
            return self._reject_early("wrong_document", "gate1", gate1, doc_type=expected, t0=t0)
        side = rec.side if rec.status == "recognized" else "single"
        sq = analyze_side(loaded, rec, self.pack, self.cfg)
        t2 = time.perf_counter()
        doc_type = rec.doc_type if rec.status == "recognized" else expected
        analysis = DocumentAnalysis(doc_type, {side: sq}, {side: rec}, field_loss_at=self.cfg.rules.field_loss_at)
        path = os.fspath(src) if isinstance(src, (str, os.PathLike)) else self._save(loaded.work, workdir, f"{name}.png")
        timings = {"gate1": round((t1 - t0) * 1000.0, 1), "quality": round((t2 - t1) * 1000.0, 1)}
        return self._finish(analysis, gate1, path, loaded.work, applied_ops, round_index, "photo", None, t0, timings)

    # ------------------------------------------------------------------ national ID
    @staticmethod
    def _id_side_problem(rec: RecognitionResult, wanted_side: str) -> Optional[str]:
        """Gate-1 verdict for one ID upload: ``None`` when it may go on, else the message code."""
        if rec.status == "unsupported":
            return "no_document"
        if rec.status == "recognized":
            if rec.doc_type != "national_id":
                return "wrong_document"
            if rec.side != wanted_side:
                return "wrong_side"
        return None

    def process_id(self, front: Source, back: Source, workdir: Union[str, Path, None] = None,
                   applied_ops: Sequence[str] = (), round_index: int = 0, name: str = "ID") -> GateResult:
        """National ID: Gate 1 on both uploads, merge (front on top, back below), quality gate on the merge."""
        t0 = time.perf_counter()
        uploads: Dict[str, Source] = {"front": front, "back": back}
        loaded: Dict[str, LoadedImage] = {}
        recs: Dict[str, RecognitionResult] = {}
        for side in ("front", "back"):
            try:
                loaded[side] = load_image(uploads[side], self.cfg.image)
            except ImageLoadError as exc:
                return self._reject_early(_LOAD_ERROR_MESSAGE[exc.code], "intake", {"error": str(exc), "side": side},
                                          sides=[side], doc_type="national_id", t0=t0)
            recs[side] = recognize(loaded[side], self.pack, self.cfg)
        gate1 = {s: _gate1_summary(r) for s, r in recs.items()}
        warnings: List[str] = []
        problems = {s: self._id_side_problem(recs[s], s) for s in ("front", "back")}
        if problems["front"] == "wrong_side" and problems["back"] == "wrong_side" and self.cfg.routing.auto_swap_sides:
            uploads = {"front": uploads["back"], "back": uploads["front"]}
            loaded = {"front": loaded["back"], "back": loaded["front"]}
            recs = {"front": recs["back"], "back": recs["front"]}
            gate1 = {s: _gate1_summary(r) for s, r in recs.items()}
            gate1["swapped"] = True
            warnings.append("the front and back were uploaded the wrong way round and have been swapped")
        else:
            for side in ("front", "back"):
                if problems[side]:
                    return self._reject_early(problems[side], "gate1", gate1, sides=[side], doc_type="national_id", t0=t0)
        t1 = time.perf_counter()
        try:
            merged = merge_id(uploads["front"], uploads["back"], self.cfg.id_merge, self.cfg.image.max_pixels)
        except ImageLoadError as exc:
            return self._reject_early(_LOAD_ERROR_MESSAGE[exc.code], "intake", {"error": str(exc)}, doc_type="national_id", t0=t0)
        res = self._assess_merged(merged, gate1, workdir, applied_ops, round_index, name, t0,
                                  {"gate1": round((t1 - t0) * 1000.0, 1)}, prior=(loaded, recs))
        res.warnings = warnings + res.warnings
        return res

    def process_merged_id(self, src: Source, workdir: Union[str, Path, None] = None, applied_ops: Sequence[str] = (),
                          round_index: int = 0, name: str = "ID", regions: Optional[Dict[str, Sequence[int]]] = None) -> GateResult:
        """Gate an ID that is already merged (front on top, white band, back below).

        ``regions`` ({"front": [y0, y1], "back": [y0, y1]}) may be given when the rows of the two sides are
        known (a DocRes output of a merged image); otherwise the white band is located."""
        t0 = time.perf_counter()
        try:
            arr, _ = to_bgr(src, self.cfg.image.max_pixels)
            if regions:
                (f0, f1), (b0, b1) = regions["front"], regions["back"]
                if not (0 <= f0 < f1 <= b0 < b1 <= arr.shape[0]):
                    raise ImageLoadError("unreadable", f"regions {dict(regions)} do not fit an image of height {arr.shape[0]}")
                merged = MergedID(image=arr, front_rows=(int(f0), int(f1)), gap_rows=(int(f1), int(b0)), back_rows=(int(b0), int(b1)))
            else:
                merged = split_merged(arr, self.cfg.id_merge)
        except ImageLoadError as exc:
            return self._reject_early(_LOAD_ERROR_MESSAGE.get(exc.code, "unreadable_file"), "intake", {"error": str(exc)},
                                      doc_type="national_id", t0=t0)
        path = os.fspath(src) if isinstance(src, (str, os.PathLike)) else None
        return self._assess_merged(merged, {}, workdir, applied_ops, round_index, name, t0, {}, prior=None, image_path=path)

    def _assess_merged(self, merged: MergedID, gate1: Dict[str, Any], workdir, applied_ops, round_index, name, t0,
                       timings: Dict[str, float], prior=None, image_path: Optional[str] = None) -> GateResult:
        t_q = time.perf_counter()
        sides: Dict[str, SideQuality] = {}
        recs: Dict[str, RecognitionResult] = {}
        for side, half in (("front", merged.front()), ("back", merged.back())):
            try:
                half_loaded = load_image(np.ascontiguousarray(half), self.cfg.image)
            except ImageLoadError as exc:
                return self._reject_early(_LOAD_ERROR_MESSAGE[exc.code], "intake", {"error": str(exc), "side": side},
                                          sides=[side], doc_type="national_id", t0=t0)
            factor = 1.0
            if prior is not None:
                # the merge only resized the upload: reuse what Gate 1 found instead of searching again
                src_loaded, src_recs = prior
                rec = transfer_recognition(src_recs[side], src_loaded[side], half_loaded)
                factor = min(1.0, src_loaded[side].original_size[0] / float(merged.width))   # an enlarged copy holds no extra detail
            else:
                rec = recognize(half_loaded, self.pack, self.cfg)
                gate1[side] = _gate1_summary(rec)
                problem = self._id_side_problem(rec, side)
                if problem:
                    return self._reject_early(problem, "gate1", gate1, sides=[side], doc_type="national_id", t0=t0)
            recs[side] = rec
            sides[side] = analyze_side(half_loaded, rec, self.pack, self.cfg, resolution_factor=factor)
        timings["quality"] = round((time.perf_counter() - t_q) * 1000.0, 1)
        path = image_path or self._save(merged.image, workdir, f"{name}_merged.png")
        regions = {"front": list(merged.front_rows), "back": list(merged.back_rows)}
        analysis = DocumentAnalysis("national_id", sides, recs, field_loss_at=self.cfg.rules.field_loss_at)
        return self._finish(analysis, gate1, path, merged.image, applied_ops, round_index, "merged_id", regions, t0, timings)

    # ------------------------------------------------------------------ a whole submission
    def process_submission(self, uploads: Dict[str, Source], workdir: Union[str, Path, None] = None) -> Dict[str, Any]:
        """Gate whatever was uploaded of ``id_front`` + ``id_back``, ``tax_card`` and ``business_license``."""
        unknown = set(uploads) - {"id_front", "id_back", "tax_card", "business_license"}
        if unknown:
            raise ValueError(f"unknown upload slot(s) {sorted(unknown)}; use id_front, id_back, tax_card, business_license")
        docs: Dict[str, GateResult] = {}
        if "id_front" in uploads and "id_back" in uploads:
            docs["national_id"] = self.process_id(uploads["id_front"], uploads["id_back"], workdir=workdir)
        elif "id_front" in uploads or "id_back" in uploads:
            missing = "back" if "id_front" in uploads else "front"
            docs["national_id"] = self._reject_early("no_document", "intake", {"missing_upload": missing}, sides=[missing],
                                                     doc_type="national_id")
        for key in ("tax_card", "business_license"):
            if key in uploads:
                docs[key] = self.process_image(uploads[key], expected=key, workdir=workdir, name=key)
        routes = [r.route for r in docs.values()]
        overall = next((r for r in ("reject", "review", "enhance", "accept") if r in routes), "reject")
        return {"overall": overall, "next_block": NEXT_BLOCKS[overall], "documents": docs}

    # ------------------------------------------------------------------ verify a DocRes output
    def _harmful(self, defect: str, severity: str) -> bool:
        """Does a defect at this severity change the outcome (moderate or worse, or enough for a safety rule)?"""
        level = self.cfg.rules.reject_unrecoverable_at.get(defect)
        return rank(severity) >= 2 or (level is not None and rank(severity) >= rank(level))

    def recheck(self, enhanced: Source, previous: GateResult, workdir: Union[str, Path, None] = None) -> GateResult:
        """Verify a DocRes output before OCR.

        The gate runs again on the enhanced image, remembering the task that was applied, and returns the
        new result: accept (go to OCR), enhance (another task, while rounds remain), reject or review.

        The enhancement is treated as a failure when the defect it targeted got worse, when it introduced
        another harmful defect, or when the document can no longer be recognised. Then the image *before*
        this enhancement is used again: it goes to OCR if it was acceptable as it was, otherwise the user is
        asked to retake (or a person reviews it, with ``routing.on_enhancement_failure: review``). A photo
        that was acceptable before an enhancement is never rejected because of that enhancement."""
        if previous.route != "enhance" or not previous.operation:
            raise ValueError("recheck needs the result of an 'enhance' decision")
        op = previous.operation
        applied = list(previous.applied_ops) + [op]
        rnd = previous.round_index + 1
        name = f"enhanced_r{rnd}"                          # never overwrite the image of an earlier round
        if previous.kind == "merged_id":
            regions = None
            if previous.regions:
                try:                                       # same height as before: the rows of the two sides are known
                    arr, _ = to_bgr(enhanced, self.cfg.image.max_pixels)
                    if arr.shape[0] == int(previous.regions["back"][1]):
                        regions = previous.regions
                except ImageLoadError:
                    pass                                   # reported by process_merged_id below
            res = self.process_merged_id(enhanced, workdir=workdir, applied_ops=applied, round_index=rnd, name=name,
                                         regions=regions)
        else:
            res = self.process_image(enhanced, expected=previous.document_type, workdir=workdir, applied_ops=applied,
                                     round_index=rnd, name=name)
        before = (previous.report or {}).get("quality_metrics", {})
        after = (res.report or {}).get("quality_metrics", {})
        targets = OP_TARGETS.get(op, ())
        worsened: List[str] = []
        improved: Optional[bool] = None
        if before and after:
            improved = sum(max(rank(after[d]["severity"]), 0) for d in targets) < sum(max(rank(before[d]["severity"]), 0) for d in targets)
            for d in after:
                sev_after = after[d]["severity"]
                r_after, r_before = rank(sev_after), rank(before.get(d, {}).get("severity", "none"))
                if r_after > r_before and (d in targets or self._harmful(d, sev_after)):
                    worsened.append(d)
        elif not before:
            res.warnings.append("the previous result carries no measurement report: the enhancement could not be compared with it")
        if res.stage in ("intake", "gate1") and previous.stage == "quality":
            worsened.append("recognition")                 # the enhanced image is no longer recognised
        res.recheck = {"operation": op, "targets": list(targets), "improved": improved, "worsened": worsened,
                       "used": "enhanced", "enhanced_status": res.status_code}
        why = None
        if worsened:
            why = f"{op} made the image worse ({', '.join(worsened)})"
        elif res.route in ("reject", "review") and previous.acceptable_as_is:
            why = f"{op} did not help and the image was acceptable before it"
        if why is None:
            return res

        # fall back to the image as it was before this enhancement
        res.recheck["used"] = "original"
        res.warnings.append(f"{why}: using the image from before the enhancement")
        res.operation = res.docres_request = res.teleocr_request = None
        res.message_ar = res.message_ar_with_side = None
        res.image, res.image_path, res.kind, res.regions = previous.image, previous.image_path, previous.kind, previous.regions
        res.report, res.laya_state, res.laya_state_json = previous.report, previous.laya_state, previous.laya_state_json
        res.gate1, res.document_type, res.acceptable_as_is = previous.gate1, previous.document_type, previous.acceptable_as_is
        res.stage, res.affected_sides = "quality", []
        if previous.acceptable_as_is:
            res.route, res.reason = "accept", None
            res.teleocr_request = {"image": previous.image_path, "kind": previous.kind, "document_type": previous.document_type}
        elif self.cfg.routing.on_enhancement_failure == "review" and self.cfg.routing.review_enabled:
            res.route, res.reason = "review", "enhancement_not_enough"
            res.message_ar = res.message_ar_with_side = self.messages["review"]
        else:
            serious = [d for d in targets if rank(before.get(d, {}).get("severity", "none")) >= 2]
            reason = {"blur": "blur", "shadow": "shadow", "dark": "dark"}.get(serious[0], "other") if serious else "other"
            res.route, res.reason, res.affected_sides = "reject", reason, list(previous.affected_sides)
            res.message_ar = self.messages[reason]
            res.message_ar_with_side = message_for(reason, self.messages, previous.affected_sides)
        res.status_code = res.route + (f":{res.reason}" if res.reason else "")
        res.next_block = NEXT_BLOCKS[res.route]
        res.decision = dict(res.decision or {}, route=res.route, reason=res.reason, operation=None, decided_by="recheck",
                            affected_sides=list(res.affected_sides), alternatives=[],
                            trace=list((res.decision or {}).get("trace", [])) + [f"{why}: fell back to the image before the enhancement"])
        return res
