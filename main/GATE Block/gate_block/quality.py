"""Quality gate: measure what is wrong with ONE photographed document side.

The gate *measures and reports*. It never decides what to do with the photo: that is the job of Laya and
the router (``laya_client.py``, ``router.py``, ``rules.py``).

Two modes:

    template   the photo was aligned to a reference template, so every measurement is made on the
               rectified document *relative to the template* (the same thresholds then work for every
               design) and every field box is inspected on its own.
    generic    Gate 1 found no usable alignment (a very blurry photo, for instance): only a few
               whole-picture measurements are possible. Everything else is reported as "unknown" -
               never as "none" and never as "missing".

Every defect gets one severity word: none | mild | moderate | severe | unknown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import GateConfig, Thr
from .geometry import (
    clip_polygon,
    outside_fraction,
    polygon_area,
    project_points,
    quad_sides,
    warp_canonical,
    warp_homography,
)
from .imageio import LoadedImage
from .measures import (
    boxes_to_cell_mask,
    colour_correct,
    colour_gains,
    contrast_range,
    edge_sharpness,
    finger_mask,
    lighting_stats,
    local_scales,
    motion_length,
    noise_sigma,
    quad_perspective,
    saturated_mask,
    sigma_equivalent,
    stroke_contrast,
    structure_agreement,
    to_gray,
)
from .recognition import Hypothesis, RecognitionResult
from .templates import Template, TemplatePack
from .types import DEFECTS, UNKNOWN, rank

# when several causes hit a field equally hard, the one named first is reported
CAUSE_PRIORITY = ("cut_off", "obstructed", "glare", "shadow", "blur", "low_contrast")
GRID = (24, 15)          # lighting / glare grid laid over the rectified document (columns, rows)


# ---------------------------------------------------------------------- severity helpers
def sev_high(value: Optional[float], thr: Thr) -> str:
    """Higher is worse; value >= threshold -> that severity (boundary inclusive)."""
    if value is None or not math.isfinite(value):
        return UNKNOWN
    mild, moderate, severe = thr
    return "severe" if value >= severe else "moderate" if value >= moderate else "mild" if value >= mild else "none"


def sev_low(value: Optional[float], thr: Thr) -> str:
    """Lower is worse; value <= threshold -> that severity (boundary inclusive)."""
    if value is None or not math.isfinite(value):
        return UNKNOWN
    mild, moderate, severe = thr
    return "severe" if value <= severe else "moderate" if value <= moderate else "mild" if value <= mild else "none"


def info_score(value: Optional[float], thr: Thr, higher_worse: bool) -> Optional[float]:
    """A 0-100 score for people reading the report (never used for decisions): 100 = perfect,
    about 23 at the 'severe' threshold, 0 well beyond it."""
    if value is None or not math.isfinite(value):
        return None
    severe = thr[2]
    if higher_worse:
        s = 1.0 - value / max(severe * 1.3, 1e-9)
    else:
        bad = severe * 0.7
        s = (value - bad) / max(1.0 - bad, 1e-9)
    return float(round(100.0 * min(1.0, max(0.0, s)), 1))


# ---------------------------------------------------------------------- result types
@dataclass
class MetricValue:
    value: Optional[float]         # the measured number (None: could not be measured / not applicable)
    severity: str                  # none | mild | moderate | severe | unknown
    score: Optional[float] = None  # 0-100, informational
    reliable: bool = True
    note: str = ""
    extra: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"value": None if self.value is None else round(float(self.value), 4),
                             "severity": self.severity, "quality_score": self.score, "reliable": self.reliable}
        if self.note:
            d["note"] = self.note
        if self.extra:
            d["extra"] = {k: (None if v is None else round(float(v), 4)) for k, v in self.extra.items()}
        return d


@dataclass
class FieldQuality:
    name: str
    critical: bool
    status: str                    # good | fair | poor | outside | unknown
    severity: str                  # worst severity among the causes
    cause: Optional[str]           # dominant cause, see CAUSE_PRIORITY
    causes: Dict[str, str]         # every cause that is not "none" -> its severity
    detail: Dict[str, Optional[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {"critical": self.critical, "status": self.status, "severity": self.severity, "cause": self.cause,
                "causes": dict(self.causes),
                "measures": {k: (None if v is None else round(float(v), 4)) for k, v in self.detail.items()}}


@dataclass
class SideQuality:
    key: str                                   # "national_id/front", or "unknown"
    doc_type: Optional[str]
    side: Optional[str]
    reference: str                             # template | generic
    recognition_status: str
    match_confidence: float
    metrics: Dict[str, MetricValue]
    fields: Dict[str, FieldQuality]
    geometry: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    rectified: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "doc_type": self.doc_type, "side": self.side, "reference": self.reference,
                "recognition": {"status": self.recognition_status, "match_confidence": round(self.match_confidence, 3)},
                "geometry": self.geometry,
                "quality_metrics": {k: m.to_dict() for k, m in self.metrics.items()},
                "field_quality": {k: f.to_dict() for k, f in self.fields.items()},
                "warnings": list(self.warnings)}


def _unknown(note: str = "") -> MetricValue:
    return MetricValue(None, UNKNOWN, None, reliable=False, note=note)


# ---------------------------------------------------------------------- glare
def _static_ink(t: Template) -> np.ndarray:
    """Deep printed ink of the template outside the variable boxes: places where we *know* dark ink is."""
    g = np.clip(t.gray, 0, 255).astype(np.uint8)
    background = cv2.medianBlur(cv2.dilate(g, np.ones((7, 7), np.uint8)), 21)
    ink = (background.astype(np.int16) - g.astype(np.int16)) > 50
    w, h = t.size
    for (x0, y0, x1, y1) in t.variable_boxes():
        ink[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = False
    return ink


def _glare(r_bgr: np.ndarray, t: Template, cfg: GateConfig, precise: bool):
    """Glare = a highlight that washes out ink. Bright paper alone is not glare, so only places where the
    template has ink count: an ink pixel is lost when everything within a few pixels of it is clipped.

    Returns (share of the static ink that is lost, share of the grid cells that are glare cells,
    boolean GRID mask of the glare cells, raw clipped mask)."""
    qc = cfg.quality
    sat = saturated_mask(r_bgr, qc.glare_v, qc.glare_s)
    k = 5 if precise else 15                       # tolerance to misalignment: larger without feature alignment
    lost = cv2.erode(sat, np.ones((k, k), np.uint8)).astype(bool) & _static_ink(t)
    ink = _static_ink(t)
    n_ink = int(ink.sum())
    share = float(lost.sum()) / n_ink if n_ink else 0.0
    gx, gy = GRID
    h, w = ink.shape
    ink_cells = cv2.resize(ink.astype(np.float32), (gx, gy), interpolation=cv2.INTER_AREA) * (w / gx) * (h / gy)
    lost_cells = cv2.resize(lost.astype(np.float32), (gx, gy), interpolation=cv2.INTER_AREA) * (w / gx) * (h / gy)
    cells = (ink_cells >= 15) & (lost_cells >= 0.5 * ink_cells)
    spread = float(cells.mean())
    cells = cv2.dilate(cells.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # ink-free neighbours inherit
    return share, spread, cells, sat.astype(bool)


# ---------------------------------------------------------------------- curl (flexible documents)
def _gradient_image(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return m / (float(np.percentile(m, 95)) + 1e-6)


def nonrigid_displacement(r_gray: np.ndarray, t: Template, grid: Tuple[int, int] = (6, 4)) -> Optional[float]:
    """How far parts of the document sit from where a flat sheet would put them: the 90th percentile of the
    block-wise shift between the rectified photo and the template (phase correlation of gradient images
    over static content), as a share of the long side. About 0 for a flat sheet; grows with curl and folds."""
    w, h = t.size
    tg, pg = _gradient_image(t.gray), _gradient_image(r_gray)
    variable = np.zeros((h, w), np.uint8)
    for (x0, y0, x1, y1) in t.variable_boxes():
        variable[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = 1
    gx, gy = grid if w >= h else (grid[1], grid[0])
    shifts = []
    for j in range(gy):
        for i in range(gx):
            x0, x1, y0, y1 = i * w // gx, (i + 1) * w // gx, j * h // gy, (j + 1) * h // gy
            if variable[y0:y1, x0:x1].mean() > 0.35:
                continue
            tb, pb = tg[y0:y1, x0:x1], pg[y0:y1, x0:x1]
            if float(tb.std()) < 0.08 or float(pb.std()) < 0.03:
                continue
            window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(tb.astype(np.float32), pb.astype(np.float32), window)
            if response >= 0.06:
                shifts.append((dx, dy))
    if len(shifts) < 6:
        return None
    d = np.asarray(shifts, np.float64)
    d -= np.median(d, axis=0)
    return float(np.percentile(np.hypot(d[:, 0], d[:, 1]), 90) / max(w, h))


# ---------------------------------------------------------------------- one field
def _field_quality(name: str, t: Template, r_bgr: np.ndarray, r_gray: np.ndarray, covered: np.ndarray,
                   lit_curve: Tuple[np.ndarray, np.ndarray], exposure: float, glare_cells: np.ndarray, sat: np.ndarray,
                   h_img_from_canon: np.ndarray, work_size: Tuple[int, int], tol: float, sigma_res: float,
                   doc_sigma: Optional[float], cfg: GateConfig) -> FieldQuality:
    qc = cfg.quality
    spec = t.fields[name]
    x0, y0, x1, y1 = t.field_box_px(name, qc.field_pad)
    rg, tg = r_gray[y0:y1, x0:x1], t.gray[y0:y1, x0:x1]
    detail: Dict[str, Optional[float]] = {}
    causes: Dict[str, str] = {}

    def add(cause: str, sev: str) -> None:
        if sev not in ("none", UNKNOWN):
            causes[cause] = sev

    # is the box inside the photo at all?
    poly = project_points(h_img_from_canon, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) + tol
    detail["outside_fraction"] = outside_fraction(poly, work_size[0] - 1 + 2 * tol, work_size[1] - 1 + 2 * tol)
    add("cut_off", sev_high(detail["outside_fraction"], qc.field_outside))

    # sharpness, against this field's own blur curve and relative to the size of this field's text. A
    # heavily blurred box keeps too few strong edges to be measured on its own: the blur measured over all
    # the text of the document stands in for it then (blur is a property of the whole photo).
    sharp, _ = edge_sharpness(rg, min_pixels=qc.field_min_edge_pixels)
    sigmas, curve = t.field_curves.get(name, (t.blur_sigmas, t.blur_curve))
    raw_sigma = sigma_equivalent(sharp, sigmas, curve) if sharp is not None else doc_sigma
    if raw_sigma is not None and doc_sigma is not None:
        # One box holds a few words, and they are not the template's words: its own reading scatters. Blur
        # comes from the lens and the hand and changes slowly across a document, so a box may read blurrier
        # than the whole text area only within a margin (focus falling off towards one edge).
        raw_sigma = min(raw_sigma, 1.5 * doc_sigma + 0.5)
    sigma = _net_blur(raw_sigma, sigma_res)
    detail["blur_sigma"] = sigma
    detail["blur_ratio"] = None if sigma is None else sigma / max(t.field_text_height.get(name, t.text_height), 1.0)
    add("blur", sev_high(detail["blur_ratio"], qc.blur_ratio))

    # Stroke contrast: faded or washed-out print. Blur and dim light lower the stroke contrast too, but those
    # are reported as blur, dark and shadow; so the template is blurred to the blur measured here and the
    # contrast is taken relative to the light falling on this very box.
    tg_ref = cv2.GaussianBlur(tg, (0, 0), raw_sigma) if raw_sigma and raw_sigma > 0.3 else tg
    box_light = (float(np.median(rg)) + 1.0) / (float(np.median(tg)) + 1.0)
    detail["contrast_ratio"] = (stroke_contrast(rg) / max(stroke_contrast(tg_ref), 20.0)) / min(max(box_light, 0.15), 1.0)
    add("low_contrast", sev_low(detail["contrast_ratio"], qc.contrast_ratio))

    # local shadow: this field against well-lit print of the same brightness elsewhere in the photo
    paper = float(np.median(tg))
    lit = float(np.interp(paper, lit_curve[0], lit_curve[1]))
    detail["local_light_ratio"] = (float(np.median(rg)) + 1.0) / (paper + 1.0) / max(lit, 1e-6)
    add("shadow", sev_high(1.0 - detail["local_light_ratio"], qc.shadow_depth))

    # glare: the part of the box that lies in a glare cell and is itself washed out
    gx, gy = GRID
    w, h = t.size
    washed = 0.0
    for j in range(int(y0 * gy / h), min(gy - 1, int((y1 - 1) * gy / h)) + 1):
        for i in range(int(x0 * gx / w), min(gx - 1, int((x1 - 1) * gx / w)) + 1):
            if not glare_cells[j, i]:
                continue
            cx0, cx1 = max(x0, i * w // gx), min(x1, (i + 1) * w // gx)
            cy0, cy1 = max(y0, j * h // gy), min(y1, (j + 1) * h // gy)
            part = sat[cy0:cy1, cx0:cx1]
            if part.size and float(part.mean()) >= 0.85:
                washed += part.size
    box_sat = float(sat[y0:y1, x0:x1].mean()) if rg.size else 0.0
    glare_frac = washed / float(max(rg.size, 1))
    if box_sat >= 0.97 and stroke_contrast(rg) < 15.0:      # a clipped box with no ink at all: fully washed out
        glare_frac = 1.0
    detail["glare_fraction"] = glare_frac
    add("glare", sev_high(glare_frac, qc.field_glare))

    # a hand or an object over the box
    detail["covered_fraction"] = float(covered[y0:y1, x0:x1].mean()) if rg.size else 0.0
    add("obstructed", sev_high(detail["covered_fraction"], qc.field_covered))

    cause, worst_rank = None, 0
    for c in CAUSE_PRIORITY:
        if c in causes and rank(causes[c]) > worst_rank:
            cause, worst_rank = c, rank(causes[c])
    severity = causes[cause] if cause else "none"
    if cause == "cut_off" and worst_rank >= 2:
        status = "outside"
    else:
        status = {"none": "good", "mild": "fair", "moderate": "poor", "severe": "poor"}[severity]
    return FieldQuality(name=name, critical=spec.critical, status=status, severity=severity, cause=cause,
                        causes=causes, detail=detail)


# ---------------------------------------------------------------------- blur and covering objects
def _net_blur(sigma: Optional[float], sigma_res: float) -> Optional[float]:
    """Blur beyond what the pixel count of the photo explains (canonical px).

    A sharp photo taken from far away is soft once it is brought to canonical size, simply because it holds
    few pixels. That is a resolution problem (reported as ``low_resolution``), not a blurry photo: nothing a
    deblurring model does can add those pixels. Blurs add in quadrature, so the share explained by the
    resolution is taken out the same way."""
    if sigma is None:
        return None
    return math.sqrt(max(sigma * sigma - sigma_res * sigma_res, 0.0))


def _resolution_blur(scale: float, lens_sigma_px: float) -> float:
    """Blur (Gaussian sigma, canonical px) of a *sharp* photo whose pixels are ``scale`` canonical px wide.

    The camera's softness (``lens_sigma_px`` photo px) and, when the photo is enlarged to the canonical size
    (``scale`` < 1), the linear interpolation, whose triangle kernel has a sigma of 1/sqrt(6) photo px."""
    scale = max(scale, 1e-3)
    interp = 1.0 / math.sqrt(6.0) if scale < 1.0 else 0.0
    return math.sqrt(lens_sigma_px ** 2 + interp ** 2) / scale


def _covered_by_objects(r_gray: np.ndarray, t: Template, sigma_eq: Optional[float], sat: np.ndarray,
                        inside: np.ndarray, cfg: GateConfig, displacement: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Where an object lies on the document, found by missing print (see ``measures.structure_agreement``).

    Returns ``(covered, share)``: a float map at canonical size (1 = covered) and the share of the comparable
    cells that are covered. Cells that are washed out by glare or lie outside the photo are not counted
    here: those are glare and a cut-off document. The outer ring of cells is left out, as paper edges lift
    and alignment is least exact there; a thumb on the very edge covers nothing that is read."""
    qc = cfg.quality
    w, h = t.size
    static = np.ones((h, w), bool)
    for (x0, y0, x1, y1) in t.variable_boxes():
        static[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = False
    blur = max(1.0, sigma_eq or 0.0)
    shift = 6
    if not t.rigid:                       # paper bends: print may sit further from where a flat sheet puts it
        shift = int(min(80, 14 + 1.5 * (displacement or 0.0) * max(w, h)))
    ncc = structure_agreement(r_gray, t.gray, static, _static_ink(t), tpl_blur=blur, max_shift=shift)
    gy, gx = ncc.shape
    cell = lambda m: cv2.resize(m.astype(np.float32), (gx, gy), interpolation=cv2.INTER_AREA)   # noqa: E731
    usable = ~np.isnan(ncc) & (cell(sat) < 0.5) & (cell(inside) > 0.9)
    ring = np.zeros((gy, gx), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    valid = usable & ~ring
    score = np.nan_to_num(ncc, nan=1.0)
    lost = valid & (score < qc.structure_lost)
    weak = valid & (score < qc.structure_weak)
    grow = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0                # noqa: E731
    lost |= weak & grow(lost)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lost.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < qc.structure_min_cells:
            lost[labels == k] = False
    # An object that reaches in from outside also covers the edge. Edge cells cannot start a covered patch
    # (a lifted paper edge looks the same there), but they join one that is already established inside.
    ring_lost = usable & ring & (score < qc.structure_lost)
    for _ in range(max(gx, gy)):
        joined = ring_lost & grow(lost) & ~lost
        if not joined.any():
            break
        lost |= joined
    valid = valid | lost
    share = float(lost.sum()) / float(max(int(valid.sum()), 1))
    # cells without comparable print (field values, blank paper) take the state of their neighbours
    value = np.where(valid, lost.astype(np.float32), 0.0)
    weight = valid.astype(np.float32)
    for _ in range(2):
        num = cv2.blur(value * weight, (3, 3), borderType=cv2.BORDER_REPLICATE)
        den = cv2.blur(weight, (3, 3), borderType=cv2.BORDER_REPLICATE)
        fill = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        value = np.where(weight > 0, value, fill)
        weight = np.maximum(weight, (den > 1e-6).astype(np.float32))
    covered = cv2.resize(value.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(covered, 0.0, 1.0), share


# ---------------------------------------------------------------------- template-aligned analysis
def usable_hypothesis(rec: RecognitionResult, cfg: GateConfig) -> Optional[Hypothesis]:
    """The alignment to measure against, if Gate 1 produced one good enough to trust."""
    b = rec.best
    if b is None or b.h_canon_from_work is None:
        return None
    rc = cfg.recognition
    if rec.status == "recognized":
        return b
    if rec.status == "uncertain" and (b.inliers >= rc.min_inliers or (b.lf >= rc.lf_strong and b.cand_score >= 0.5)):
        return b
    return None


def _analyze_aligned(loaded: LoadedImage, rec: RecognitionResult, hyp: Hypothesis, cfg: GateConfig,
                     keep_rectified: bool, resolution_factor: float) -> SideQuality:
    qc = cfg.quality
    t = hyp.template
    w, h = t.size
    ww, wh = loaded.work_size
    h_c = np.asarray(hyp.h_canon_from_work, np.float64)
    h_img = np.linalg.inv(h_c)
    r_bgr = warp_homography(loaded.work, h_c, t.size)
    r_gray = to_gray(r_bgr)
    warnings: List[str] = []
    precise = hyp.aligned_by_features
    if not precise:
        warnings.append("aligned by the document outline only (no feature match): field positions are approximate")
    tol = 0.004 * max(ww, wh)

    # ---- geometry
    corners = project_points(h_img, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    out_frac = outside_fraction(corners + tol, ww - 1 + 2 * tol, wh - 1 + 2 * tol)
    visible = clip_polygon(corners, ww - 1, wh - 1)
    frame_fill = (polygon_area(visible) if len(visible) >= 3 else 0.0) / float(ww * wh)
    persp = quad_perspective(corners)
    scales = local_scales(h_img, t.size)                       # work px per canonical px: 4 corners + centre
    to_upload = resolution_factor / loaded.work_scale          # work px -> px of the real upload
    min_scale = float(min(scales)) * to_upload
    text_px = min_scale * t.text_height                        # smallest printed text, in px of the upload
    top_edge = corners[1] - corners[0]
    geometry = {
        "frame": "cut" if out_frac >= qc.cut_fraction[0] else "complete",
        "outside_fraction": round(float(out_frac), 5),
        "frame_fill": round(float(frame_fill), 4),
        "corners_work_px": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
        "rotation_deg": round(float(math.degrees(math.atan2(top_edge[1], top_edge[0]))), 1),
        "perspective": {"side_mismatch": round(persp["side_mismatch"], 4), "max_angle_dev_deg": round(persp["angle_dev_deg"], 2)},
        "min_scale": round(min_scale, 3),
        "text_height_px": round(text_px, 2),
        "original_size_px": list(loaded.original_size),
        "alignment": "features" if precise else "outline",
    }

    # ---- sharpness of the text regions (needed first: everything structural is compared at this blur)
    sharp, _ = edge_sharpness(r_gray, mask=t.text_mask)
    sigma_raw = None if sharp is None else sigma_equivalent(sharp, t.blur_sigmas, t.blur_curve)
    # A camera movement smears the strokes it crosses while those along it stay crisp, and a contrast-
    # normalised edge reading is blind to that (see ``measures.motion_length``). The movement's own
    # Gaussian equivalent (a box of length L has sigma L / sqrt(12)) counts when it is the larger blur.
    in_photo = cv2.warpPerspective(np.full((wh, ww), 255, np.uint8), h_c, (w, h), flags=cv2.INTER_NEAREST) > 0
    motion_mask = in_photo if t.text_mask is None else (in_photo & t.text_mask)
    motion_px, motion_angle = (motion_length(r_gray, motion_mask, t.motion_rulers)
                               if t.motion_rulers and motion_mask.sum() >= 400 else (0.0, 0))
    if sigma_raw is not None:
        sigma_raw = max(sigma_raw, motion_px / math.sqrt(12.0))
    if sigma_raw is None:
        warnings.append("blur could not be measured (too few strong edges in the text regions)")
    # the blur a sharp photo of this size has anyway (canonical px): the camera's own softness, plus the
    # interpolation when the photo holds fewer pixels than the canonical image and is enlarged to it
    detail_scale = float(np.median(scales)) * min(1.0, resolution_factor)
    sigma_res = _resolution_blur(detail_scale, qc.lens_sigma_px)
    sigma_net = _net_blur(sigma_raw, sigma_res)
    blur_ratio = None if sigma_net is None else sigma_net / max(t.text_height, 1.0)
    blur_sev = sev_high(blur_ratio, qc.blur_ratio)

    # ---- light: first pass, to know how dark the photo is
    glare_share, glare_spread, glare_cells, sat = _glare(r_bgr, t, cfg, precise)
    base_ignore = boxes_to_cell_mask(t.ignore_boxes, *GRID) if t.ignore_boxes else np.zeros((GRID[1], GRID[0]), bool)
    base_ignore[0, :] = base_ignore[-1, :] = True
    base_ignore[:, 0] = base_ignore[:, -1] = True              # outer ring: alignment tolerance, not shadow
    exposure_first = lighting_stats(r_gray, t.gray, GRID[0], GRID[1], base_ignore)["exposure_ratio"] / qc.exposure_ref

    # ---- something lying on the document: a hand (skin colour) or any object (the print is missing there)
    skin_reliable = exposure_first >= qc.dark_ratio[2]         # colours mean little in a severely dark photo
    if skin_reliable:
        finger = finger_mask(colour_correct(r_bgr, colour_gains(r_bgr, t.bgr)), t.skin)
        for (bx0, by0, bx1, by1) in t.ignore_boxes:            # a portrait in the photo slot is skin, not a finger
            finger[int(by0 * h):int(math.ceil(by1 * h)), int(bx0 * w):int(math.ceil(bx1 * w))] = False
    else:
        finger = np.zeros((h, w), bool)
        warnings.append("too dark to judge colours: a covering hand is detected by missing print only")
    inside = in_photo
    displacement = None if t.rigid else nonrigid_displacement(r_gray, t)
    # Paper curled so much that its print sits far from where a flat sheet puts it cannot be checked for
    # covering objects: the print is "missing" where the bend moved it, and the background shows where the
    # sheet was expected. Then coverage is not measured now; it is measured after dewarping (recheck).
    too_curled = (displacement is not None and rank(blur_sev) < 2
                  and rank(sev_high(displacement, qc.curl)) >= 2)
    if too_curled:
        objects, object_share = np.zeros((h, w), np.float32), 0.0
        finger = np.zeros((h, w), bool)
        warnings.append("paper curled too much to look for covering objects: checked again after dewarping")
    else:
        objects, object_share = _covered_by_objects(r_gray, t, sigma_raw, sat, inside, cfg, displacement)
    skin_share = float(finger.mean())
    covered = np.maximum(objects, finger.astype(np.float32))
    obstruction = max(skin_share, object_share)

    # ---- light: a covering hand or object is not a shadow
    ignore = base_ignore | (cv2.resize(covered, GRID, interpolation=cv2.INTER_AREA) >= 0.3)
    light = lighting_stats(r_gray, t.gray, GRID[0], GRID[1], ignore)
    exposure_abs = light["exposure_ratio"]
    exposure = exposure_abs / qc.exposure_ref
    contrast_rel = (contrast_range(r_gray) / max(t.contrast, 1.0)) / max(min(exposure, 1.0), 0.3)

    # ---- noise, curl
    noise = math.sqrt(max(noise_sigma(r_gray) ** 2 - t.noise ** 2, 0.0))
    if t.rigid:
        curl = MetricValue(None, "none", None, note="not applicable: rigid card")
    elif rank(blur_sev) >= 2:
        curl = _unknown("cannot be measured on a blurred photo")
    else:
        curl = MetricValue(displacement, sev_high(displacement, qc.curl), info_score(displacement, qc.curl, True),
                           reliable=displacement is not None)

    m: Dict[str, MetricValue] = {
        "cut": MetricValue(out_frac, sev_high(out_frac, qc.cut_fraction), info_score(out_frac, qc.cut_fraction, True)),
        # `value` is the height of the smallest printed text in pixels of the upload: what a reader gets to see
        "low_resolution": MetricValue(text_px, sev_low(text_px, qc.text_px), info_score(text_px, qc.text_px, False),
                                      extra={"min_scale": min_scale, "text_height_canonical_px": t.text_height}),
        # `value` is blur as a share of the height of the smallest printed text (what decides legibility),
        # after taking out the softness that the pixel count of the photo explains
        "blur": MetricValue(blur_ratio, blur_sev, info_score(blur_ratio, qc.blur_ratio, True), reliable=blur_ratio is not None,
                            extra={"sigma_px": sigma_net, "sigma_measured_px": sigma_raw, "sigma_from_resolution_px": sigma_res,
                                   "motion_px": round(motion_px, 1), "motion_direction_deg": motion_angle if motion_px else None}),   # 0: along the lines
        # Brightness is one measurement: `value` is the photo's brightness relative to the template. Too dark
        # is a defect. Too bright is not one by itself: what overexposure destroys is ink, and lost ink is
        # measured as glare (the share of the paper that is clipped is given for information only).
        "dark": MetricValue(exposure, sev_low(exposure, qc.dark_ratio), info_score(exposure, qc.dark_ratio, False),
                            extra={"clipped_fraction": float(sat.mean())}),
        "low_contrast": MetricValue(contrast_rel, sev_low(contrast_rel, qc.contrast_ratio),
                                    info_score(contrast_rel, qc.contrast_ratio, False)),
        "shadow": MetricValue(light["shadow_depth"], sev_high(light["shadow_depth"], qc.shadow_depth),
                              info_score(light["shadow_depth"], qc.shadow_depth, True)),
        "glare": MetricValue(glare_share, sev_high(glare_share, qc.glare_content), info_score(glare_share, qc.glare_content, True),
                             reliable=precise, extra={"spread": glare_spread}),
        "perspective": MetricValue(persp["side_mismatch"], sev_high(persp["side_mismatch"], qc.perspective),
                                   info_score(persp["side_mismatch"], qc.perspective, True)),
        "curl": curl,
        "noise": MetricValue(noise, sev_high(noise, qc.noise_sigma), info_score(noise, qc.noise_sigma, True)),
        # share of the document covered by a hand (skin colour) or by any object (missing print)
        "obstruction": (_unknown("paper curled: measured after dewarping") if too_curled else
                        MetricValue(obstruction, sev_high(obstruction, qc.obstruction), info_score(obstruction, qc.obstruction, True),
                                    extra={"hand": skin_share, "object": object_share})),
    }
    fields = {name: _field_quality(name, t, r_bgr, r_gray, covered, light["lit_curve"], exposure, glare_cells, sat,
                                   h_img, (ww, wh), tol, sigma_res, sigma_raw, cfg) for name in t.fields}
    return SideQuality(key=t.key, doc_type=t.doc_type, side=t.side, reference="template",
                       recognition_status=rec.status, match_confidence=rec.confidence, metrics=m, fields=fields,
                       geometry=geometry, warnings=warnings, rectified=r_bgr if keep_rectified else None)


# ---------------------------------------------------------------------- generic analysis
def _analyze_generic(loaded: LoadedImage, rec: RecognitionResult, pack: TemplatePack, cfg: GateConfig,
                     keep_rectified: bool, resolution_factor: float) -> SideQuality:
    qc = cfg.quality
    ww, wh = loaded.work_size
    b = rec.best
    quad = (np.asarray(b.quad_small, np.float64) * loaded.small_to_work if b is not None
            else np.array([[0, 0], [ww - 1, 0], [ww - 1, wh - 1], [0, wh - 1]], np.float64))
    top, right, bottom, left = quad_sides(quad)
    wid, hei = (top + bottom) / 2.0, (left + right) / 2.0
    long_side = cfg.image.canonical_long_side
    size = ((long_side, max(8, int(round(long_side * hei / max(wid, 1.0))))) if wid >= hei
            else (max(8, int(round(long_side * wid / max(hei, 1.0)))), long_side))
    r_bgr, _ = warp_canonical(loaded.work, quad.astype(np.float32), size)
    r_gray = to_gray(r_bgr)

    # generic references: the median over the template pack
    curve = np.median(np.stack([t.blur_curve for t in pack.templates]), axis=0)
    sigmas = pack.templates[0].blur_sigmas
    ref_contrast = float(np.median([t.contrast for t in pack.templates]))
    ref_lum = float(np.median([float(np.median(t.gray)) for t in pack.templates]))
    exposure = (float(np.median(r_gray)) + 1.0) / (ref_lum + 1.0) / qc.exposure_ref
    contrast_rel = (contrast_range(r_gray) / max(ref_contrast, 1.0)) / max(min(exposure, 1.0), 0.3)
    persp = quad_perspective(quad)
    scale_work = (wid / size[0] + hei / size[1]) / 2.0
    scale = scale_work / loaded.work_scale * resolution_factor
    text_px = scale * qc.default_text_height
    sharp, _ = edge_sharpness(r_gray)
    sigma_raw = None if sharp is None else sigma_equivalent(sharp, sigmas, curve)
    sigma_net = _net_blur(sigma_raw, _resolution_blur(scale_work * min(1.0, resolution_factor), qc.lens_sigma_px))
    blur_ratio = None if sigma_net is None else sigma_net / qc.default_text_height
    noise = noise_sigma(r_gray)
    m: Dict[str, MetricValue] = {
        "cut": _unknown("needs a template alignment"),
        "low_resolution": MetricValue(text_px, sev_low(text_px, qc.text_px), info_score(text_px, qc.text_px, False), reliable=False,
                                      note="text size assumed (no template)", extra={"min_scale": scale}),
        "blur": MetricValue(blur_ratio, sev_high(blur_ratio, qc.blur_ratio), info_score(blur_ratio, qc.blur_ratio, True),
                            reliable=False, note="whole picture against a generic curve: approximate",
                            extra={"sigma_px": sigma_net, "sigma_measured_px": sigma_raw}),
        "dark": MetricValue(exposure, sev_low(exposure, qc.dark_ratio), info_score(exposure, qc.dark_ratio, False), reliable=False),
        "low_contrast": MetricValue(contrast_rel, sev_low(contrast_rel, qc.contrast_ratio),
                                    info_score(contrast_rel, qc.contrast_ratio, False), reliable=False),
        "shadow": _unknown("a dark design band cannot be told from a shadow without a template"),
        "glare": _unknown("bright paper cannot be told from glare without a template"),
        "perspective": MetricValue(persp["side_mismatch"], sev_high(persp["side_mismatch"], qc.perspective),
                                   info_score(persp["side_mismatch"], qc.perspective, True), reliable=False),
        "curl": _unknown("needs a template alignment"),
        "noise": MetricValue(noise, sev_high(noise, qc.noise_sigma), info_score(noise, qc.noise_sigma, True), reliable=False),
        "obstruction": _unknown("needs a template alignment"),
    }
    geometry = {"frame": "unknown", "outside_fraction": None,
                "frame_fill": round(polygon_area(clip_polygon(quad, ww - 1, wh - 1)) / float(ww * wh), 4),
                "corners_work_px": [[round(float(x), 1), round(float(y), 1)] for x, y in quad], "rotation_deg": None,
                "perspective": {"side_mismatch": round(persp["side_mismatch"], 4), "max_angle_dev_deg": round(persp["angle_dev_deg"], 2)},
                "min_scale": round(scale, 3), "text_height_px": round(text_px, 2),
                "original_size_px": list(loaded.original_size), "alignment": "none"}
    return SideQuality(key="unknown", doc_type=None, side=None, reference="generic", recognition_status=rec.status,
                       match_confidence=rec.confidence, metrics=m, fields={}, geometry=geometry,
                       warnings=["no usable template alignment: whole-picture measurements only, fields unknown"],
                       rectified=r_bgr if keep_rectified else None)


def analyze_side(loaded: LoadedImage, rec: RecognitionResult, pack: TemplatePack, cfg: GateConfig,
                 keep_rectified: bool = False, resolution_factor: float = 1.0) -> SideQuality:
    """Measure the quality of one photographed document side.

    ``resolution_factor`` scales the measured pixel density when ``loaded`` is an enlarged copy of the real
    upload (the halves of a merged ID are resized to a fixed width): an upscaled photo holds no more detail
    than the original did."""
    hyp = usable_hypothesis(rec, cfg)
    if hyp is not None:
        sq = _analyze_aligned(loaded, rec, hyp, cfg, keep_rectified, resolution_factor)
    else:
        sq = _analyze_generic(loaded, rec, pack, cfg, keep_rectified, resolution_factor)
    assert set(sq.metrics) == set(DEFECTS), "every defect must be reported"
    return sq
