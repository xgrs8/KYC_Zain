"""Quadrilateral geometry and the search for document-shaped objects in a photo."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import RecognitionCfg


# ------------------------------------------------------------------------------ basics
def order_quad(pts) -> np.ndarray:
    """Order 4 points clockwise on screen (y down), starting from the top-left-most one."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    k = int(np.argmin(p.sum(axis=1)))
    return np.roll(p, -k, axis=0).astype(np.float32)


def quad_sides(q) -> Tuple[float, float, float, float]:
    """(top, right, bottom, left) side lengths of a TL, TR, BR, BL quadrilateral."""
    q = np.asarray(q, np.float64)
    return (float(np.linalg.norm(q[1] - q[0])), float(np.linalg.norm(q[2] - q[1])),
            float(np.linalg.norm(q[3] - q[2])), float(np.linalg.norm(q[0] - q[3])))


def polygon_area(p) -> float:
    p = np.asarray(p, np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def rect_quad(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)


def clip_polygon(poly, w: float, h: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to the rectangle [0, w] x [0, h]."""
    pts = [tuple(map(float, p)) for p in np.asarray(poly, np.float64).reshape(-1, 2)]

    def clip(points, inside, intersect):
        res = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if inside(cur):
                if not inside(prev):
                    res.append(intersect(prev, cur))
                res.append(cur)
            elif inside(prev):
                res.append(intersect(prev, cur))
        return res

    def at_x(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def at_y(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    for inside, intersect in (
            (lambda p: p[0] >= 0.0, lambda a, b: at_x(a, b, 0.0)),
            (lambda p: p[0] <= w, lambda a, b: at_x(a, b, float(w))),
            (lambda p: p[1] >= 0.0, lambda a, b: at_y(a, b, 0.0)),
            (lambda p: p[1] <= h, lambda a, b: at_y(a, b, float(h)))):
        if not pts:
            break
        pts = clip(pts, inside, intersect)
    return np.asarray(pts, np.float64).reshape(-1, 2)


def outside_fraction(poly, w: float, h: float) -> float:
    """Share of the polygon's area that lies outside the rectangle [0, w] x [0, h]."""
    area = polygon_area(poly)
    if area <= 1e-9:
        return 0.0
    clipped = clip_polygon(poly, w, h)
    inside = polygon_area(clipped) if len(clipped) >= 3 else 0.0
    return float(min(1.0, max(0.0, 1.0 - inside / area)))


def quad_iou(a, b, shape: Tuple[int, int]) -> float:
    h, w = shape
    s = 160.0 / max(h, w)
    size = (int(w * s) + 2, int(h * s) + 2)
    ma = np.zeros((size[1], size[0]), np.uint8)
    mb = np.zeros_like(ma)
    cv2.fillConvexPoly(ma, np.round(np.asarray(a) * s).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.round(np.asarray(b) * s).astype(np.int32), 1)
    union = int(np.logical_or(ma, mb).sum())
    return int(np.logical_and(ma, mb).sum()) / union if union else 0.0


def project_points(h, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(h, np.float64)).reshape(-1, 2)


def quad_homography(quad, size: Tuple[int, int]) -> np.ndarray:
    """Homography that maps the ``quad`` (TL, TR, BR, BL) onto a ``size`` (w, h) canonical image."""
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst).astype(np.float64)


def warp_canonical(img: np.ndarray, quad, size: Tuple[int, int]):
    """Warp the document ``quad`` (TL, TR, BR, BL in ``img`` pixels) to a ``size`` (w, h) image.

    Returns ``(image, H)`` with ``H`` mapping ``img`` pixels to canonical pixels. A large downscale is done
    in two steps (area resize first) so that fine patterns do not alias."""
    w, h = size
    top, right, bottom, left = quad_sides(quad)
    scale = (((top + bottom) / 2.0) / w + ((left + right) / 2.0) / h) / 2.0     # image px per canonical px
    q = np.asarray(quad, np.float32)
    src, pre = img, np.eye(3)
    if scale > 1.6:
        f = 1.3 / scale
        src = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        q = q * f
        pre = np.diag([f, f, 1.0])
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    hm = cv2.getPerspectiveTransform(q, dst)
    flags = cv2.INTER_CUBIC if scale < 1.0 else cv2.INTER_LINEAR
    out = cv2.warpPerspective(src, hm, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return out, hm @ pre


def warp_homography(img: np.ndarray, h_canon_from_img, size: Tuple[int, int]) -> np.ndarray:
    """Warp ``img`` to a ``size`` (w, h) image with a known homography (image px -> canonical px), with the
    same two-step downscale as :func:`warp_canonical`."""
    hm = np.asarray(h_canon_from_img, np.float64)
    w, hh = size
    inv = np.linalg.inv(hm)
    p = inv @ np.array([w / 2.0, hh / 2.0, 1.0])
    den = p[2]
    dxdu, dxdv = (inv[0, 0] - (p[0] / den) * inv[2, 0]) / den, (inv[0, 1] - (p[0] / den) * inv[2, 1]) / den
    dydu, dydv = (inv[1, 0] - (p[1] / den) * inv[2, 0]) / den, (inv[1, 1] - (p[1] / den) * inv[2, 1]) / den
    scale = float(math.sqrt(abs(dxdu * dydv - dxdv * dydu)))                     # image px per canonical px
    src = img
    if scale > 1.6:
        f = 1.3 / scale
        src = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        hm = hm @ np.diag([1.0 / f, 1.0 / f, 1.0])
    flags = cv2.INTER_CUBIC if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.warpPerspective(src, hm, (w, hh), flags=flags, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------------------ candidates
@dataclass
class Candidate:
    quad: np.ndarray                  # (4, 2) float32: TL, TR, BR, BL
    score: float                      # plausibility of being a document, 0..1
    source: str                       # edges | mask | fullframe
    area_frac: float
    parts: Dict[str, float] = field(default_factory=dict)


def _contours(binary: np.ndarray, mode: int):
    return cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)[-2]


def _approx_quad(hull: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(hull, True)
    for eps in (0.012, 0.02, 0.03, 0.045, 0.065):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area > 1.0 and cv2.contourArea(hull) / rect_area >= 0.80:
        return cv2.boxPoints(rect).astype(np.float32)
    return None


def refine_quad_by_lines(hull: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Replace the corners of ``quad`` by the intersections of straight lines fitted to the hull along each
    side. A card with rounded corners gets its true (sharp) corners back instead of points on the arcs."""
    q = order_quad(quad).astype(np.float64)
    hp = hull.reshape(-1, 2).astype(np.float64)
    samples = []
    for i in range(len(hp)):                                   # sample the hull outline densely
        a, b = hp[i], hp[(i + 1) % len(hp)]
        n = max(2, int(np.linalg.norm(b - a) / 3.0))
        samples.append(a[None, :] + (b - a)[None, :] * np.linspace(0, 1, n, endpoint=False)[:, None])
    pts = np.vstack(samples)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length < 10:
            return quad
        t = ((pts - a) @ ab) / (length * length)
        dist = np.abs((pts - a)[:, 0] * ab[1] - (pts - a)[:, 1] * ab[0]) / length
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.04 * length)]
        if len(sel) < 8:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((float(vx), float(vy), float(x0), float(y0)))
    corners = []
    for i in range(4):                                         # corner i = side (i-1) x side i
        v1x, v1y, x1, y1 = lines[(i - 1) % 4]
        v2x, v2y, x2, y2 = lines[i]
        det = v1x * v2y - v1y * v2x
        if abs(det) < 1e-6:
            return quad
        s = ((x2 - x1) * v2y - (y2 - y1) * v2x) / det
        corners.append((x1 + s * v1x, y1 + s * v1y))
    out = np.asarray(corners, np.float32)
    if not np.all(np.isfinite(out)) or np.max(np.linalg.norm(out - q, axis=1)) > 0.2 * max(quad_sides(q)):
        return quad                                            # a wild intersection: keep the safe polygon
    return out


def _edge_hulls(gray: np.ndarray, min_area: float) -> List[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hi, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = max(40.0, float(hi))
    edges = cv2.Canny(blur, 0.5 * hi, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts = [c for c in _contours(edges, cv2.RETR_LIST) if cv2.contourArea(c) >= min_area]
    cnts.sort(key=cv2.contourArea, reverse=True)
    return [cv2.convexHull(c) for c in cnts[:12]]


def _mask_hulls(img: np.ndarray, gray: np.ndarray, lab: np.ndarray, min_area: float) -> List[np.ndarray]:
    h, w = gray.shape
    masks = []
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, m_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [m_light, 255 - m_light]
    sat = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    _, m_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [m_sat, 255 - m_sat]
    # distance from the background colour, modelled from the ring of pixels along the image border
    r = max(3, int(0.04 * min(h, w)))
    ring = np.concatenate([lab[:r].reshape(-1, 3), lab[-r:].reshape(-1, 3), lab[:, :r].reshape(-1, 3), lab[:, -r:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(ring, axis=0), axis=2)
    d8 = cv2.GaussianBlur(np.clip(dist * 4.0, 0, 255).astype(np.uint8), (7, 7), 0)
    _, m_bg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(m_bg)
    k = max(5, int(0.02 * min(h, w))) | 1
    k2 = max(3, k // 2) | 1
    hulls: List[np.ndarray] = []
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k2, k2), np.uint8))
        cnts = [c for c in _contours(m, cv2.RETR_EXTERNAL) if cv2.contourArea(c) >= min_area]
        cnts.sort(key=cv2.contourArea, reverse=True)
        hulls += [cv2.convexHull(c) for c in cnts[:3]]
    return hulls


def _gradient_map(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.dilate(np.sqrt(gx * gx + gy * gy), np.ones((7, 7), np.uint8))
    return mag / max(float(np.percentile(mag, 95)), 40.0)      # floor: a smooth ramp has no edges


def _edge_support(gmap: np.ndarray, quad: np.ndarray) -> float:
    """Share of the quad's outline that lies on a real image edge."""
    h, w = gmap.shape
    t = np.linspace(0.05, 0.95, 24)[:, None]
    p = np.vstack([quad[i][None, :] * (1 - t) + quad[(i + 1) % 4][None, :] * t for i in range(4)])
    xs = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
    return float(np.mean(gmap[ys, xs] >= 0.3))


def _boundary_contrast(lab: np.ndarray, quad: np.ndarray) -> float:
    """Share of outline samples where the colour just inside differs clearly from the colour just outside:
    a document lying on a surface. Texture and smooth gradients give a small share."""
    h, w = lab.shape[:2]
    q = order_quad(quad)
    off = 0.012 * min(h, w) + 3.0
    t = np.linspace(0.06, 0.94, 24)[:, None]
    hits = total = 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]], np.float32)                # points into the quad for a clockwise outline
        n /= max(float(np.linalg.norm(n)), 1e-6)
        pts = a[None, :] * (1 - t) + b[None, :] * t
        for pi, po in zip(pts + off * n, pts - off * n):
            xi, yi, xo, yo = int(round(pi[0])), int(round(pi[1])), int(round(po[0])), int(round(po[1]))
            if not (0 <= xi < w and 0 <= yi < h and 0 <= xo < w and 0 <= yo < h):
                continue                                       # outline at the frame: nothing outside to compare
            total += 1
            if float(np.linalg.norm(lab[yi, xi] - lab[yo, xo])) >= 10.0:
                hits += 1
    return hits / total if total >= 12 else 0.0


def score_quad(quad: np.ndarray, hull: Optional[np.ndarray], shape: Tuple[int, int], aspects: Sequence[float],
               gmap: np.ndarray, lab: np.ndarray, cfg: RecognitionCfg) -> Optional[Candidate]:
    h, w = shape
    quad = order_quad(quad)
    area = polygon_area(quad)
    frac = area / float(h * w)
    if frac < cfg.min_area_frac:
        return None
    top, right, bottom, left = quad_sides(quad)
    if min(top, right, bottom, left) < 20:
        return None
    wid, hei = (top + bottom) / 2.0, (right + left) / 2.0
    aspect = max(wid, hei) / max(min(wid, hei), 1e-6)
    err = min(abs(math.log(aspect / a)) for a in (aspects or [1.5]))
    f_aspect = math.exp(-0.5 * (err / cfg.aspect_sigma) ** 2)
    hull_area = polygon_area(hull.reshape(-1, 2)) if hull is not None else area
    f_rect = max(0.0, 1.0 - abs(1.0 - hull_area / max(area, 1e-6)) * 5.0)
    devs = []
    for i in range(4):
        a, b, c = quad[(i - 1) % 4], quad[i], quad[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9))
        devs.append(abs(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) - 90.0))
    f_angle = 1.0 - min(1.0, max(devs) / 45.0)
    f_edge = _edge_support(gmap, quad)
    f_contrast = _boundary_contrast(lab, quad)
    f_area = min(1.0, max(0.0, (frac - 0.03) / 0.35))
    score = 0.30 * f_aspect + 0.20 * f_rect + 0.15 * f_angle + 0.20 * f_edge + 0.15 * f_area
    return Candidate(quad=quad, score=float(score), source="", area_frac=float(frac),
                     parts={"aspect": f_aspect, "rect": f_rect, "angle": f_angle, "edge": f_edge,
                            "contrast": f_contrast, "area": f_area})


def find_candidates(img: np.ndarray, aspects: Sequence[float], cfg: RecognitionCfg) -> List[Candidate]:
    """Document-shaped quadrilaterals in ``img``, best first. A full-frame candidate is always included so
    that a document filling the whole photo (a scan, a tight crop) can still be examined."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    gmap = _gradient_map(gray)
    min_area = cfg.min_area_frac * h * w
    found: List[Candidate] = []
    for source, hulls in (("edges", _edge_hulls(gray, min_area)), ("mask", _mask_hulls(img, gray, lab, min_area))):
        for hull in hulls:
            quad = _approx_quad(hull)
            if quad is None:
                continue
            cand = score_quad(refine_quad_by_lines(hull, quad), hull, (h, w), aspects, gmap, lab, cfg)
            if cand is None:
                continue
            cand.source = source
            if cand.area_frac > 0.93:      # "the whole frame is the document" is not evidence of a document
                cand.score *= 0.5
            found.append(cand)
    full = score_quad(rect_quad(w, h), None, (h, w), aspects, gmap, lab, cfg)
    if full is not None:
        full.source = "fullframe"
        full.score *= 0.6
        found.append(full)
    found.sort(key=lambda c: c.score, reverse=True)
    kept: List[Candidate] = []
    for c in found:
        if all(quad_iou(c.quad, k.quad, (h, w)) < 0.85 for k in kept):
            kept.append(c)
        if len(kept) >= cfg.max_candidates:
            break
    return kept


def extend_cut_quad(quad, aspect: float, shape: Tuple[int, int], tol_frac: float = 0.015,
                    min_gain: float = 0.03) -> Optional[np.ndarray]:
    """Undo the "visible part looks like the whole document" mistake for a document cut by the frame.

    ``quad`` is TL, TR, BR, BL in the *template's* orientation and ``aspect`` the template's width / height.
    When a side of the quad lies along the edge of the image and the quad is too short across that side
    for the template's proportions, the document continues outside the frame: the quad is extended there
    until the proportions are right. Returns the extended quad, or ``None`` when nothing needs extending
    (or the cut cannot be located: sides on the frame in both directions).
    """
    h, w = shape
    q = np.asarray(quad, np.float64).copy()
    tol = tol_frac * min(h, w)

    def on_frame(a, b) -> bool:
        return bool((a[0] <= tol and b[0] <= tol) or (a[0] >= w - 1 - tol and b[0] >= w - 1 - tol)
                    or (a[1] <= tol and b[1] <= tol) or (a[1] >= h - 1 - tol and b[1] >= h - 1 - tol))

    top, right, bottom, left = on_frame(q[0], q[1]), on_frame(q[1], q[2]), on_frame(q[3], q[2]), on_frame(q[0], q[3])
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    height = (np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0])) / 2.0
    if min(width, height) < 1.0:
        return None
    if (left or right) and not (top or bottom):
        s = aspect * height / width                       # how much longer the width should be
        if s < 1.0 + min_gain:
            return None
        grow_right = (s - 1.0) * (1.0 if not left else 0.5 if right else 0.0)
        grow_left = (s - 1.0) * (1.0 if not right else 0.5 if left else 0.0)
        u_top, u_bot = q[1] - q[0], q[2] - q[3]
        q[1], q[2] = q[1] + u_top * grow_right, q[2] + u_bot * grow_right
        q[0], q[3] = q[0] - u_top * grow_left, q[3] - u_bot * grow_left
        return q.astype(np.float32)
    if (top or bottom) and not (left or right):
        s = (width / aspect) / height
        if s < 1.0 + min_gain:
            return None
        grow_bottom = (s - 1.0) * (1.0 if not top else 0.5 if bottom else 0.0)
        grow_top = (s - 1.0) * (1.0 if not bottom else 0.5 if top else 0.0)
        v_left, v_right = q[3] - q[0], q[2] - q[1]
        q[3], q[2] = q[3] + v_left * grow_bottom, q[2] + v_right * grow_bottom
        q[0], q[1] = q[0] - v_left * grow_top, q[1] - v_right * grow_top
        return q.astype(np.float32)
    return None
