"""Gate 1: is the upload one of the supported documents?

The outcome is never a bare yes / no:

    recognized   strong evidence that the photo shows a supported document (type and side known)
    unsupported  reliable evidence that it does NOT: no document-like object, or a sharp, detailed picture
                 that matches none of the templates
    uncertain    not enough evidence either way (typically a blurry, dark or cropped photo); these are not
                 rejected here - the quality gate measures them and the pipeline then decides

Evidence comes from three independent sources, so the failure of one (ORB features on a blurry photo, for
instance) never decides the outcome on its own:

    geometry   a document-shaped quadrilateral (aspect ratio, straight sides, a real outline)
    layout     low-frequency similarity of the rectified photo to each template; tolerant of blur, and the
               fields whose content changes from one document to the next are masked out
    features   ORB matches against the template's *static* content, checked with a RANSAC homography

Matching a template shows that the picture *looks like* the document. It says nothing about whether the
document is genuine: this is not an anti-forgery check.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import GateConfig
from .geometry import (
    Candidate,
    extend_cut_quad,
    find_candidates,
    polygon_area,
    project_points,
    quad_homography,
    warp_canonical,
    warp_homography,
)
from .imageio import LoadedImage
from .measures import edge_sharpness, to_gray
from .templates import (
    Template,
    TemplatePack,
    chroma_histogram,
    make_orb,
    preprocess_for_orb,
    thumbnail_signature,
)


@dataclass
class Hypothesis:
    template: Template
    quad_small: np.ndarray                # (4, 2) document corners in small-image px, in the template's orientation
    lf: float                             # layout similarity, 0..1
    cand_score: float                     # geometric plausibility, 0..1
    source: str                           # edges | mask | fullframe | features
    inliers: int = 0                      # RANSAC inliers that passed the reliability checks
    raw_inliers: int = 0                  # inliers before those checks (diagnostics)
    inlier_ratio: float = 0.0
    n_keypoints: int = 0
    residual_px: Optional[float] = None
    h_canon_from_work: Optional[np.ndarray] = None    # work-image px -> canonical px
    aligned_by_features: bool = False     # the alignment was set by feature matches, not just by the outline
    score: float = 0.0
    verified: bool = False

    @property
    def key(self) -> str:
        return self.template.key


@dataclass
class RecognitionResult:
    status: str                           # recognized | uncertain | unsupported
    doc_type: Optional[str]               # of the best hypothesis (also set when uncertain)
    side: Optional[str]
    confidence: float                     # combined evidence score of the best hypothesis, 0..1
    best: Optional[Hypothesis]
    runner_up: Optional[Hypothesis]
    evidence: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def key(self) -> Optional[str]:
        return None if self.doc_type is None else f"{self.doc_type}/{self.side}"


# ---------------------------------------------------------------------- layout similarity
def weighted_ncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    ws = float(w.sum())
    if ws <= 1e-6:
        return 0.0
    ma, mb = float((w * a).sum() / ws), float((w * b).sum() / ws)
    cov = float((w * (a - ma) * (b - mb)).sum() / ws)
    va, vb = float((w * (a - ma) ** 2).sum() / ws), float((w * (b - mb) ** 2).sum() / ws)
    return cov / float(np.sqrt(va * vb + 1e-9))


def lf_similarity(img_small: np.ndarray, quad_small: np.ndarray, t: Template) -> float:
    """Low-frequency layout similarity (0..1) between the quad region of the photo and template ``t``."""
    tw, th = t.thumb_size
    mid = (tw * 3, th * 3)
    dst = np.array([[0, 0], [mid[0] - 1, 0], [mid[0] - 1, mid[1] - 1], [0, mid[1] - 1]], np.float32)
    hm = cv2.getPerspectiveTransform(np.asarray(quad_small, np.float32), dst)
    warp = cv2.warpPerspective(img_small, hm, mid, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gn, edge, ab = thumbnail_signature(warp, (tw, th))
    s_gray = max(0.0, weighted_ncc(gn, t.thumb_gray, t.thumb_weight))
    s_edge = max(0.0, weighted_ncc(edge, t.thumb_edge, t.thumb_weight))
    hist, coloured = chroma_histogram(ab, t.thumb_weight)
    if t.has_chroma and coloured:
        return 0.55 * s_gray + 0.30 * s_edge + 0.15 * float(np.minimum(hist, t.chroma_hist).sum())
    return (0.55 * s_gray + 0.30 * s_edge) / 0.85


# ---------------------------------------------------------------------- feature evidence
def _sane_refinement(h: np.ndarray, size: Tuple[int, int]) -> bool:
    """A refinement on an already rectified image must be close to the identity."""
    w, hh = size
    corners = np.array([[0, 0], [w, 0], [w, hh], [0, hh]], np.float64)
    p = project_points(h, corners)
    if not np.all(np.isfinite(p)):
        return False
    if float(np.max(np.linalg.norm(p - corners, axis=1))) > 0.25 * max(w, hh):
        return False
    return 0.6 <= polygon_area(p) / float(w * hh) <= 1.6


def _match(des, pts: np.ndarray, t: Template, cfg: GateConfig):
    """Ratio-test matches photo -> template and a RANSAC homography.

    Returns ``(inliers, good matches, H, median residual, spread)``. ``spread`` is the area covered by the
    inlier points relative to the area the template's keypoints span: inliers crowded into one spot are not
    evidence of a document."""
    if des is None or len(pts) < 8 or len(t.des) < 8:
        return 0, 0, None, None, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, t.des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.recognition.ratio * p[1].distance]
    if len(good) < 8:
        return 0, len(good), None, None, 0.0
    src = np.float32([pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = t.kp_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    thr = cfg.recognition.ransac_px * (1.0 if t.rigid else cfg.recognition.flexible_ransac_scale)
    hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr)
    if hm is None or mask is None:
        return 0, len(good), None, None, 0.0
    keep = mask.ravel() == 1
    n_in = int(keep.sum())
    if n_in < 4:
        return n_in, len(good), None, None, 0.0
    err = np.linalg.norm(cv2.perspectiveTransform(src[keep], hm) - dst[keep], axis=2).ravel()
    hull_area = float(cv2.contourArea(cv2.convexHull(dst[keep].reshape(-1, 1, 2))))
    spread = hull_area / float(t.size[0] * t.size[1] * t.kp_hull_frac)
    return n_in, len(good), hm, float(np.median(err)), float(spread)


def _detect(gray_u8: np.ndarray, n: int):
    kps, des = make_orb(n).detectAndCompute(preprocess_for_orb(gray_u8), None)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2) if kps else np.zeros((0, 2), np.float32)
    return pts, des


def _detect_spread(gray_u8: np.ndarray, n: int, cells: int = 8):
    """Up to ``n`` keypoints spread evenly over the picture.

    ORB keeps the strongest corners of the whole picture. On a busy background (a keyboard, a patterned
    cloth, a bright phone screen next to the document) those all land on the background and the document is
    left with almost none. Here the picture is divided into cells and every cell keeps its own strongest
    corners, so the document gets its share wherever it lies and whatever surrounds it."""
    img = preprocess_for_orb(gray_u8)
    orb = make_orb(n * 3)
    kps = list(orb.detect(img, None))
    if len(kps) > n:
        h, w = img.shape[:2]
        gx = max(1, cells)
        gy = max(1, int(round(cells * h / float(w))))
        per_cell = int(np.ceil(n / float(gx * gy)))
        buckets: Dict[Tuple[int, int], list] = {}
        for k in kps:
            buckets.setdefault((min(gx - 1, int(k.pt[0] * gx / w)), min(gy - 1, int(k.pt[1] * gy / h))), []).append(k)
        keep, rest = [], []
        for cell in buckets.values():
            cell.sort(key=lambda k: -k.response)
            keep += cell[:per_cell]
            rest += cell[per_cell:]
        if len(keep) < n:                                      # cells without texture leave room for the others
            rest.sort(key=lambda k: -k.response)
            keep += rest[: n - len(keep)]
        kps = keep
    kps, des = orb.compute(img, kps)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2) if kps else np.zeros((0, 2), np.float32)
    return pts, des


def _document_quad(h_canon_from_work: np.ndarray, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """The document's corners in work-image px implied by an alignment, or None if it is not a proper quad."""
    w, h = size
    try:
        quad = project_points(np.linalg.inv(h_canon_from_work), [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(quad)) or not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return None
    return quad


def _refine(loaded: LoadedImage, hyp: Hypothesis, cfg: GateConfig) -> None:
    """Match features on the photo rectified with the current alignment.

    * Matches that agree with the current alignment (a near-identity correction) count as evidence even
      when they sit in one part of the document (a card whose print is concentrated in its header).
    * The matches may *replace* the alignment only when they span enough of the document: a homography
      fitted to one corner extrapolates badly over the rest.
    * A large correction (the layout guess had the document the wrong way up, the outline was off) is
      accepted only on full feature evidence that spans the document."""
    t, rc = hyp.template, cfg.recognition
    warp = warp_homography(loaded.work, hyp.h_canon_from_work, t.size)
    pts, des = _detect(cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY), rc.orb_features)
    n_in, n_good, hr, res, spread = _match(des, pts, t, cfg)
    hyp.n_keypoints = max(hyp.n_keypoints, len(pts))
    hyp.raw_inliers = max(hyp.raw_inliers, n_in)
    if hr is None or n_in < 8 or n_in <= hyp.inliers:
        return
    near = _sane_refinement(hr, t.size)
    new_h = quad = None
    if spread >= rc.align_inlier_spread:
        cand_h = hr @ hyp.h_canon_from_work
        q = _document_quad(cand_h, t.size)
        ww, wh = loaded.work_size
        if q is not None and polygon_area(q) >= rc.min_area_frac * ww * wh:
            new_h, quad = cand_h, q
    if near:
        if spread < rc.min_inlier_spread:
            return
    elif new_h is None or n_in < rc.min_inliers:
        return
    hyp.inliers = n_in
    hyp.inlier_ratio = n_in / n_good if n_good else 0.0
    hyp.residual_px = res
    if new_h is not None:
        hyp.h_canon_from_work = new_h
        hyp.quad_small = (quad / loaded.small_to_work).astype(np.float32)
        hyp.lf = lf_similarity(loaded.small, hyp.quad_small, t)  # the layout score of the alignment actually used
        hyp.aligned_by_features = True


def _verify(loaded: LoadedImage, hyp: Hypothesis, cfg: GateConfig) -> None:
    """Feature verification of a quadrilateral hypothesis: rectify by the quad, then refine twice."""
    hyp.h_canon_from_work = quad_homography(hyp.quad_small * loaded.small_to_work, hyp.template.size)
    hyp.verified = True
    _refine(loaded, hyp, cfg)
    if hyp.aligned_by_features:
        _refine(loaded, hyp, cfg)               # second pass on the better alignment


def _direct_hypotheses(loaded: LoadedImage, pack: TemplatePack, cfg: GateConfig) -> Tuple[List[Hypothesis], int]:
    """Match the whole photo against every template; no quadrilateral needed (cluttered backgrounds, a
    document whose outline was not found). Returns the hypotheses and the largest raw inlier count."""
    rc = cfg.recognition
    img = loaded.small
    pts, des = _detect_spread(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rc.orb_search_features)
    out: List[Hypothesis] = []
    raw_max = 0
    s2w = loaded.small_to_work
    for t in pack.templates:
        n_in, n_good, hm, res, spread = _match(des, pts, t, cfg)
        raw_max = max(raw_max, n_in)
        if hm is None or n_in < rc.min_inliers:
            continue
        # matches that span the document fix its position; a strong cluster of matches in one part of it
        # is still a find, and the refinement below (features of the rectified document) settles the rest
        if spread < rc.align_inlier_spread and not (n_in >= rc.strong_inliers and spread >= 2 * rc.min_inlier_spread):
            continue
        try:
            inv = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            continue
        w, h = t.size
        quad = project_points(inv, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
        if not np.all(np.isfinite(quad)) or polygon_area(quad) < rc.min_area_frac * img.shape[0] * img.shape[1]:
            continue
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
            continue
        hyp = Hypothesis(template=t, quad_small=quad, lf=lf_similarity(img, quad, t), cand_score=0.5, source="features",
                         inliers=n_in, raw_inliers=n_in, inlier_ratio=n_in / max(n_good, 1), n_keypoints=len(pts),
                         residual_px=res, h_canon_from_work=hm @ np.diag([1.0 / s2w, 1.0 / s2w, 1.0]), verified=True,
                         aligned_by_features=True)
        _refine(loaded, hyp, cfg)
        out.append(hyp)
    return out, raw_max


def _sharp_detailed_picture(loaded: LoadedImage, hyp: Hypothesis, cfg: GateConfig) -> bool:
    """True when the candidate region is a well-lit, sharp, detailed picture. If such a picture still
    matches no template, that is reliable evidence of "not a supported document"; a dark or blurry one
    proves nothing. Sharpness is measured on strong edges only, so sensor noise cannot fake it."""
    rc = cfg.recognition
    warp, _ = warp_canonical(loaded.work, hyp.quad_small * loaded.small_to_work, hyp.template.size)
    g = to_gray(warp)
    if float(np.median(g)) < rc.rich_min_luminance:
        return False
    sharp, _ = edge_sharpness(g)
    if sharp is None or sharp < rc.rich_sharpness:
        return False
    pts, _ = _detect(np.clip(g, 0, 255).astype(np.uint8), rc.rich_keypoints * 2)
    return len(pts) >= rc.rich_keypoints


# ---------------------------------------------------------------------- decision
def _score(h: Hypothesis, cfg: GateConfig) -> float:
    rc = cfg.recognition
    f_feat = 0.0
    if h.inliers >= rc.min_inliers:
        f_feat = min(1.0, (h.inliers - rc.min_inliers) / float(rc.strong_inliers - rc.min_inliers))
    f_lf = min(1.0, max(0.0, (h.lf - rc.lf_min) / (rc.lf_strong - rc.lf_min)))
    return 0.45 * f_feat + 0.40 * f_lf + 0.15 * h.cand_score


def recognize(loaded: LoadedImage, pack: TemplatePack, cfg: GateConfig) -> RecognitionResult:
    rc = cfg.recognition
    img = loaded.small
    cands: List[Candidate] = find_candidates(img, pack.aspects(), rc)

    # every candidate x every template, in the best of the four orientations
    hyps: List[Hypothesis] = []
    for c in cands:
        for t in pack.templates:
            best_lf, best_q = -1.0, c.quad
            for rot in range(4):
                q = np.roll(c.quad, -rot, axis=0)
                lf = lf_similarity(img, q, t)
                if lf > best_lf:
                    best_lf, best_q = lf, q
            if c.source != "fullframe":
                # a document cut by the frame: its visible part is not the whole template
                longer = extend_cut_quad(best_q, t.aspect, img.shape[:2])
                if longer is not None:
                    lf = lf_similarity(img, longer, t)
                    if lf > best_lf + 0.02:
                        best_lf, best_q = lf, longer
            hyps.append(Hypothesis(template=t, quad_small=best_q, lf=best_lf, cand_score=c.score, source=c.source))

    # feature verification: the two best candidates of the three most promising templates
    by_template: Dict[str, List[Hypothesis]] = {}
    for h in hyps:
        by_template.setdefault(h.key, []).append(h)
    promise = lambda h: h.lf * (0.5 + 0.5 * h.cand_score)          # noqa: E731
    for group in sorted(by_template.values(), key=lambda g: -max(promise(h) for h in g))[:3]:
        for h in sorted(group, key=lambda h: -promise(h))[:2]:
            _verify(loaded, h, cfg)

    direct, direct_raw = _direct_hypotheses(loaded, pack, cfg)     # independent of the outline search
    hyps += direct
    for h in hyps:
        h.score = _score(h, cfg)
    hyps.sort(key=lambda h: h.score, reverse=True)
    if hyps and not hyps[0].verified:
        _verify(loaded, hyps[0], cfg)
        hyps[0].score = _score(hyps[0], cfg)
        hyps.sort(key=lambda h: h.score, reverse=True)
    if not hyps:
        return RecognitionResult("unsupported", None, None, 0.0, None, None, {}, ["no document-like object found"])

    best = hyps[0]
    second = next((h for h in hyps if h.key != best.key), None)
    margin = best.score - (second.score if second else 0.0)
    plausible_doc = any(c.source != "fullframe" and c.score >= 0.45 and c.parts.get("edge", 0.0) >= 0.5
                        and c.parts.get("contrast", 0.0) >= 0.5 for c in cands)
    evidence = {
        "score": round(best.score, 4), "layout_similarity": round(best.lf, 4), "inliers": float(best.inliers),
        "raw_inliers": float(best.raw_inliers), "inlier_ratio": round(best.inlier_ratio, 4),
        "keypoints": float(best.n_keypoints), "geometry": round(best.cand_score, 4), "margin": round(margin, 4),
        "direct_raw_inliers": float(direct_raw), "document_like_object": float(plausible_doc),
    }
    if best.residual_px is not None:
        evidence["residual_px"] = round(best.residual_px, 3)

    notes: List[str] = []
    feat_ok = best.inliers >= rc.min_inliers
    layout_only = best.lf >= rc.lf_certain and best.cand_score >= 0.6 and margin >= 0.25
    if (best.score >= rc.recognized_score and margin >= rc.margin and (feat_ok or best.lf >= rc.lf_strong)) or layout_only:
        status = "recognized"
    elif (best.score < rc.unsupported_score and best.inliers < max(3, rc.min_inliers // 3)
          and best.lf < rc.lf_min + 0.05 and direct_raw < rc.unsupported_max_direct_inliers
          and (not plausible_doc or _sharp_detailed_picture(loaded, best, cfg))):
        status = "unsupported"
        notes.append("no document-like object" if not plausible_doc else "a sharp picture that matches no template")
    else:
        status = "uncertain"
        if best.score >= rc.recognized_score and margin < rc.margin:
            notes.append("two templates are almost equally likely")
        if not feat_ok:
            notes.append("no feature evidence (blurry, dark, small or not a supported document)")
    return RecognitionResult(status=status, doc_type=best.template.doc_type, side=best.template.side,
                             confidence=float(min(1.0, max(0.0, best.score))), best=best, runner_up=second,
                             evidence=evidence, notes=notes)


def transfer_recognition(rec: RecognitionResult, src: LoadedImage, dst: LoadedImage) -> RecognitionResult:
    """Carry a recognition result over to a *resized copy of the same picture*.

    The ID merge only resizes each upload, so the document found in the upload sits at the same relative
    place in the merged half; recognising it a second time would be wasted work. Resizing maps pixel centres
    as x_dst = (x_src + 0.5) * a - 0.5, which is undone here exactly."""
    if rec.best is None:
        return rec
    (sw, sh), (dw, dh) = src.work_size, dst.work_size
    ax, ay = dw / float(sw), dh / float(sh)
    src_from_dst = np.array([[1.0 / ax, 0.0, 0.5 / ax - 0.5], [0.0, 1.0 / ay, 0.5 / ay - 0.5], [0.0, 0.0, 1.0]])

    def move(h: Optional[Hypothesis]) -> Optional[Hypothesis]:
        if h is None:
            return None
        quad_work = h.quad_small.astype(np.float64) * src.small_to_work
        quad_dst = (quad_work + 0.5) * np.array([ax, ay]) - 0.5
        return replace(h, quad_small=(quad_dst / dst.small_to_work).astype(np.float32),
                       h_canon_from_work=None if h.h_canon_from_work is None else h.h_canon_from_work @ src_from_dst)

    return replace(rec, best=move(rec.best), runner_up=move(rec.runner_up))
