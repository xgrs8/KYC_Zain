"""Pure measurement functions (numpy / OpenCV only, no decisions, no thresholds).

Each function measures one physical property of a *rectified* document photo and returns plain numbers;
``quality.py`` turns the numbers into severities. Wherever a value depends on the document's design
(sharpness of a guilloche background is not the sharpness of plain paper) the measurement is made
*relative to the template*, so the same thresholds work for every document type.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BLUR_SIGMAS: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ------------------------------------------------------------------------------ sharpness
def edge_sharpness(gray: np.ndarray, win: int = 11, rel_contrast: float = 0.4, min_abs_contrast: float = 10.0,
                   min_pixels: int = 60, mask: Optional[np.ndarray] = None) -> Tuple[Optional[float], int]:
    """Contrast-normalised edge sharpness.

    On strong edges (local maxima of the gradient whose local contrast is at least ``rel_contrast`` of the
    strongest contrast in the patch) take ``gradient magnitude / local contrast``. An ideal step gives 0.5;
    a step blurred by a Gaussian of sigma s gives about 0.4 / s. Dividing by the local contrast makes the
    value independent of exposure and of the ink/paper contrast; using strong edges only makes it
    insensitive to sensor noise.

    ``mask`` (boolean, same shape) restricts the measurement to a region, e.g. the text fields.

    Returns ``(median ratio, number of edge pixels used)``. The value is ``None`` when fewer than
    ``min_pixels`` strong-edge pixels exist (not enough text or lines to judge).
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.6)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    k = np.ones((win, win), np.uint8)
    contrast = cv2.dilate(g, k) - cv2.erode(g, k)
    local_max = mag >= (cv2.dilate(mag, np.ones((3, 3), np.uint8)) - 1e-6)
    region = contrast if mask is None else contrast[mask]
    if region.size == 0:
        return None, 0
    c_ref = float(np.percentile(region, 95))
    sel = local_max & (contrast >= max(min_abs_contrast, rel_contrast * c_ref))
    if mask is not None:
        sel &= mask
    n = int(sel.sum())
    if n < min_pixels:
        return None, n
    return float(np.median(mag[sel] / contrast[sel])), n


def blur_curve(gray: np.ndarray, sigmas: Sequence[float] = BLUR_SIGMAS,
               mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sharpness of ``gray`` after a Gaussian blur of each sigma, forced strictly decreasing so that it can
    be inverted. This is a template's own "ruler": which blur gives which sharpness on *this* design."""
    vals: List[float] = []
    for s in sigmas:
        g = gray if s <= 0 else cv2.GaussianBlur(gray, (0, 0), float(s))
        v, _ = edge_sharpness(g, min_pixels=20, mask=mask)
        vals.append(float(v) if v is not None else (vals[-1] if vals else 0.5))
    curve = np.minimum.accumulate(np.asarray(vals, np.float64))
    for i in range(1, len(curve)):
        if curve[i] >= curve[i - 1]:
            curve[i] = curve[i - 1] - 1e-6
    return np.asarray(sigmas, np.float64), curve


# ---------------------------------------------------------------------- motion blur
MOTION_LENGTHS: Tuple[int, ...] = (0, 3, 5, 7, 9, 12, 15, 20, 26, 34)    # smear lengths of the ruler, canonical px
MOTION_ANGLES: Tuple[int, ...] = (0, 45, 90, 135)                       # gradient directions, degrees (y down)


def directional_energy(gray: np.ndarray, mask: Optional[np.ndarray] = None, pre_blur: float = 1.0) -> Dict[int, float]:
    """Mean squared derivative of ``gray`` along each of ``MOTION_ANGLES`` (inside ``mask``).

    ``pre_blur`` keeps pixel noise and JPEG blocks, which have no direction, out of the comparison."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), pre_blur)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    if mask is not None:
        gx, gy = gx[mask], gy[mask]
    out: Dict[int, float] = {}
    for angle in MOTION_ANGLES:
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        d = gx * c + gy * s
        out[angle] = float(np.mean(d * d)) if d.size else 0.0
    return out


def energy_ratios(energy: Dict[int, float]) -> Dict[int, float]:
    """For each direction: its energy over the energy of the perpendicular direction."""
    return {a: energy[a] / max(energy[(a + 90) % 180], 1e-9) for a in MOTION_ANGLES}


def motion_smear(gray: np.ndarray, length: int, angle: int) -> np.ndarray:
    """``gray`` smeared by a camera movement of ``length`` px; ``angle`` is the direction of the movement
    in the same convention as the gradient directions of ``directional_energy``."""
    if length <= 1:
        return gray.astype(np.float32)
    k = np.zeros((length, length), np.float32)
    cv2.line(k, (0, length // 2), (length - 1, length // 2), 1.0, 1)
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), -float(angle), 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(float(k.sum()), 1e-9)
    return cv2.filter2D(gray.astype(np.float32), -1, k)


def motion_rulers(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
    """A template's motion "ruler": for each direction, the energy ratio after smearing the template along
    that direction by each of ``MOTION_LENGTHS`` (forced strictly decreasing so that it can be inverted)."""
    rulers: Dict[int, np.ndarray] = {}
    for angle in MOTION_ANGLES:
        vals = [energy_ratios(directional_energy(motion_smear(gray, n, angle), mask))[angle] for n in MOTION_LENGTHS]
        curve = np.minimum.accumulate(np.asarray(vals, np.float64))
        for i in range(1, len(curve)):
            if curve[i] >= curve[i - 1]:
                curve[i] = curve[i - 1] * (1 - 1e-6)
        rulers[angle] = curve
    return rulers


def motion_length(gray: np.ndarray, mask: Optional[np.ndarray], rulers: Dict[int, np.ndarray]) -> Tuple[float, int]:
    """Length (canonical px) and direction of the camera movement that best explains how much weaker the
    photo's derivatives are in one direction than across it, compared with the template's own balance.

    Edge sharpness is normalised by the local contrast, so a smear that wipes out the strokes it crosses
    hardly moves it: the strokes that run along the movement stay crisp and are the majority on a line of
    print. The balance of derivative energy between perpendicular directions does not have that blind spot,
    and evenly spread blur, exposure and ink colour leave it nearly unchanged."""
    ratios = energy_ratios(directional_energy(gray, mask))
    lengths = np.asarray(MOTION_LENGTHS, np.float64)
    best, best_angle = 0.0, MOTION_ANGLES[0]
    for angle, curve in rulers.items():
        v = ratios[angle]
        if v >= curve[0]:
            n = 0.0
        elif v <= curve[-1]:
            n = float(lengths[-1])
        else:
            n = float(np.interp(v, curve[::-1], lengths[::-1]))
        if n > best:
            best, best_angle = n, angle
    return best, best_angle


def sigma_equivalent(value: float, sigmas: np.ndarray, curve: np.ndarray) -> float:
    """Invert a blur curve: which Gaussian sigma would give this sharpness on the template?"""
    if value >= curve[0]:
        return 0.0
    if value <= curve[-1]:
        return float(sigmas[-1])
    return float(np.interp(value, curve[::-1], sigmas[::-1]))


# ------------------------------------------------------------------------------ noise
def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate on the flattest 40 % of the pixels, in 0-255 grey levels."""
    g = gray.astype(np.float32)
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = np.abs(cv2.filter2D(g, cv2.CV_32F, m))[2:-2, 2:-2]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[2:-2, 2:-2]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[2:-2, 2:-2]
    mag = np.sqrt(gx * gx + gy * gy)
    flat = mag <= np.percentile(mag, 40)
    if int(flat.sum()) < 100:
        flat = np.ones_like(flat, bool)
    return float(math.sqrt(math.pi / 2.0) / 6.0 * conv[flat].mean())


# ------------------------------------------------------------------------------ lighting
def cell_medians(gray: np.ndarray, gx: int, gy: int, cell: int = 16) -> np.ndarray:
    """Median luminance of a gx x gy grid laid over ``gray`` (the median ignores thin text strokes)."""
    small = cv2.resize(gray.astype(np.float32), (gx * cell, gy * cell), interpolation=cv2.INTER_AREA)
    blocks = small.reshape(gy, cell, gx, cell).transpose(0, 2, 1, 3).reshape(gy, gx, -1)
    return np.median(blocks, axis=-1)


def boxes_to_cell_mask(boxes: Sequence[Sequence[float]], gx: int, gy: int, min_cover: float = 0.35) -> np.ndarray:
    """Boolean (gy, gx) mask of the grid cells covered at least ``min_cover`` by any normalised box."""
    mask = np.zeros((gy, gx), bool)
    for (x0, y0, x1, y1) in boxes:
        for j in range(gy):
            for i in range(gx):
                cx0, cx1, cy0, cy1 = i / gx, (i + 1) / gx, j / gy, (j + 1) / gy
                ix = max(0.0, min(cx1, x1) - max(cx0, x0))
                iy = max(0.0, min(cy1, y1) - max(cy0, y0))
                if ix * iy / ((cx1 - cx0) * (cy1 - cy0)) >= min_cover:
                    mask[j, i] = True
    return mask


def lit_curve(tpl_cells: np.ndarray, ratio_cells: np.ndarray, step: float = 1.15,
              min_cells: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """The photo / template luminance ratio of *well-lit* cells, as a function of how bright the print is.

    A camera's tone curve (contrast, gamma, lens flare) changes that ratio with the brightness of the print
    itself: a dark header band comes out relatively darker or lighter than white paper. Measured against one
    common reference, such a band would look like a shadow. So the cells are grouped into brightness classes
    of the template (each spanning a factor ``step`` in luminance) and the well-lit level (80th percentile
    of the ratio) is taken per class. Returns ``(xs, ys)`` for ``np.interp``; a single point when the print
    has only one brightness class."""
    t = np.asarray(tpl_cells, np.float64).ravel()
    r = np.asarray(ratio_cells, np.float64).ravel()
    single = (np.array([float(np.median(t))]), np.array([float(np.percentile(r, 80))]))
    lo, hi = float(t.min()), float(t.max())
    if len(t) < 40 or (hi + 1.0) / (lo + 1.0) < 1.25:
        return single
    cls = np.floor(np.log((t + 1.0) / (lo + 1.0)) / math.log(step)).astype(int)
    xs: List[float] = []
    ys: List[float] = []
    sizes: List[int] = []
    for c in np.unique(cls):
        sel = cls == c
        if int(sel.sum()) >= min_cells:
            xs.append(float(np.median(t[sel])))
            ys.append(float(np.percentile(r[sel], 80)))
            sizes.append(int(sel.sum()))
    if not xs:
        return single
    big = [i for i, n in enumerate(sizes) if n >= 0.5 * max(sizes)]
    paper = ys[big[-1]]                                         # the brightest of the large classes: the paper
    # the curve may explain a tone difference, not a deep shadow lying on one whole brightness class
    return np.asarray(xs), np.clip(np.asarray(ys), 0.45 * paper, 2.2 * paper)


def lighting_stats(photo_gray: np.ndarray, tpl_gray: Optional[np.ndarray], gx: int = 24, gy: int = 15,
                   ignore: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Exposure and shadow statistics from per-cell luminance ratios.

    With a template the ratio photo / template is used, which removes the document's own design (dark
    header bands, tinted backgrounds), and each cell is compared with the well-lit cells of the same print
    brightness (:func:`lit_curve`), which removes the camera's tone curve. Without a template, cells are
    compared with the median cell of the photo itself, which cannot tell a dark design band from a shadow
    (the caller reports shadow as unknown then).

    Returns ``exposure_ratio`` (brightness of the paper relative to the template), ``shadow_depth`` (how
    much darker the darkest cells are than well-lit ones, 0..1), ``lit_ratio`` (the well-lit ratio at paper
    brightness) and ``lit_curve`` (``(xs, ys)`` or ``None``)."""
    pm = cell_medians(photo_gray, gx, gy)
    valid = np.ones(pm.shape, bool) if ignore is None else ~ignore
    if int(valid.sum()) < 8:
        valid = np.ones(pm.shape, bool)
    if tpl_gray is None:
        r = ((pm + 1.0) / (float(np.median(pm)) + 1.0))[valid]
        lit = float(np.percentile(r, 80))
        return {"exposure_ratio": float(np.median(r)), "lit_ratio": lit, "lit_curve": None,
                "shadow_depth": float(max(0.0, 1.0 - np.percentile(r, 5) / max(lit, 1e-6)))}
    tm = cell_medians(tpl_gray, gx, gy)
    ratio = (pm + 1.0) / (tm + 1.0)
    xs, ys = lit_curve(tm[valid], ratio[valid])
    # illumination of every cell relative to well-lit cells of the same print brightness. The darkest
    # cells are set against the 80th percentile, not the median: a shadow that covers most of the document
    # must still read as a shadow, and a glare spot must not turn the rest of the document into one.
    q = (ratio / np.maximum(np.interp(tm, xs, ys), 1e-6))[valid]
    paper = tm[valid] >= np.median(tm[valid])                   # the lighter half of the print
    return {
        "exposure_ratio": float(np.median(ratio[valid][paper])),
        "shadow_depth": float(max(0.0, 1.0 - np.percentile(q, 5) / max(float(np.percentile(q, 80)), 1e-6))),
        "lit_ratio": float(np.interp(np.percentile(tm[valid], 80), xs, ys)),
        "lit_curve": (xs, ys),
    }


def contrast_range(gray: np.ndarray) -> float:
    return float(np.percentile(gray, 95) - np.percentile(gray, 5))


def stroke_contrast(gray: np.ndarray, win: int = 7) -> float:
    """Contrast of the strongest strokes in a patch (98th percentile of the local max - min), independent
    of how much ink the patch holds: a short word and a long one give the same value."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.6)
    k = np.ones((win, win), np.uint8)
    return float(np.percentile(cv2.dilate(g, k) - cv2.erode(g, k), 98))


def content_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels that carry printed content (text, lines, emblems) rather than plain background."""
    g = np.clip(gray, 0, 255).astype(np.uint8)
    background = cv2.medianBlur(cv2.dilate(g, np.ones((7, 7), np.uint8)), 21)
    m = (cv2.absdiff(g, background) > 25).astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


# ------------------------------------------------------------------------------ glare / colour / skin
def saturated_mask(bgr: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    """Clipped highlights: very bright and nearly colourless pixels (isolated specks removed)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 2] >= v_min) & (hsv[:, :, 1] <= s_max)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] >= 225) & (hsv[:, :, 1] <= 60)).astype(np.uint8)


def colour_gains(photo_bgr: np.ndarray, tpl_bgr: np.ndarray, gx: int = 12, gy: int = 8) -> np.ndarray:
    """Per-channel gain (B, G, R) that brings the photo's colours to the template's: the median over a grid
    of the ratio of cell means. Dividing the photo by it removes the colour and brightness of the light
    (a warm lamp makes white paper look skin-coloured; this undoes that)."""
    p = cv2.resize(photo_bgr.astype(np.float32), (gx, gy), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    t = cv2.resize(tpl_bgr.astype(np.float32), (gx, gy), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    gains = np.median((p + 1.0) / (t + 1.0), axis=0)
    return np.clip(gains, 0.2, 5.0)


def colour_correct(photo_bgr: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.clip(photo_bgr.astype(np.float32) / gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of skin-coloured pixels (a finger or hand). Use it on a colour-corrected image
    (see :func:`colour_gains`): under a warm lamp white paper is skin-coloured too."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    return (cr >= 135) & (cr <= 178) & (cb >= 80) & (cb <= 130) & (y >= 50) & (sat >= 35)


def finger_mask(corrected_bgr: np.ndarray, template_skin: np.ndarray) -> np.ndarray:
    """Skin-coloured blobs in the (colour-corrected) rectified photo that the template does not have there."""
    m = (skin_mask(corrected_bgr) & ~template_skin).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


# ------------------------------------------------------------------------------ text size, print structure
def text_band_height(gray_patch: np.ndarray) -> Optional[float]:
    """Height in pixels of the line of text printed in a field box, or None when the box holds no clear
    dark-on-light text. The rows that carry ink are weighted by how much ink they carry, and the height is
    the span holding the middle 90 % of it: dots and marks above or below the letters (Arabic) neither
    break the line apart nor stretch it."""
    g = np.asarray(gray_patch, np.float32)
    if g.shape[0] < 6 or g.shape[1] < 6:
        return None
    lo, hi = np.percentile(g, [1, 90])                  # a short word inks only a few % of its box
    if hi - lo < 30:
        return None
    ink = (g < (lo + hi) / 2.0).sum(axis=1).astype(np.float64)
    ink[ink < 0.01 * g.shape[1]] = 0.0
    total = float(ink.sum())
    if total <= 0:
        return None
    cum = np.cumsum(ink) / total
    top = int(np.searchsorted(cum, 0.05))
    bottom = int(np.searchsorted(cum, 0.95))
    return float(bottom - top + 1)


def _local_structure(g: np.ndarray, pre_blur: float) -> np.ndarray:
    """Grey image with the local brightness and local contrast taken out: what is left is the printed
    structure, the same under a lamp, in a shadow or in a dim room."""
    g = g.astype(np.float32)
    if pre_blur > 0.05:
        g = cv2.GaussianBlur(g, (0, 0), pre_blur)
    mean = cv2.GaussianBlur(g, (0, 0), 12.0)
    var = cv2.GaussianBlur((g - mean) ** 2, (0, 0), 12.0)
    return (g - mean) / (np.sqrt(var) + 4.0)


def structure_grid(size: Tuple[int, int], cells_long: int = 16) -> Tuple[int, int]:
    """(columns, rows) of the grid used by :func:`structure_agreement` for a canonical image ``size``."""
    w, h = size
    if w >= h:
        return cells_long, max(2, int(round(cells_long * h / float(w))))
    return max(2, int(round(cells_long * w / float(h)))), cells_long


def structure_agreement(photo_gray: np.ndarray, tpl_gray: np.ndarray, static: np.ndarray, ink: np.ndarray,
                        cells_long: int = 16, tpl_blur: float = 1.0, max_shift: int = 6) -> np.ndarray:
    """Does the photo show the print the template has? One value per grid cell, 1 = the same structure,
    about 0 = something else is there; NaN where the template has no static print to compare.

    This is what separates a *covering object* from a *shadow*. A shadow darkens the paper but the print is
    still there, so the structure agrees; a phone, a hand or any other object lying on the document replaces
    the print, so it does not. Brightness and contrast are normalised away first, the template is blurred to
    the blur measured in the photo, and each cell may shift by ``max_shift`` px (alignment tolerance)."""
    h, w = tpl_gray.shape[:2]
    a = _local_structure(photo_gray, 1.0)
    t = tpl_gray.astype(np.float32)
    if tpl_blur > 0.05:
        t = cv2.GaussianBlur(t, (0, 0), tpl_blur)
    b = _local_structure(t, 1.0)
    a = np.where(static, a, 0.0).astype(np.float32)
    b = np.where(static, b, 0.0).astype(np.float32)
    gx, gy = structure_grid((w, h), cells_long)
    out = np.full((gy, gx), np.nan, np.float32)
    for j in range(gy):
        for i in range(gx):
            x0, x1, y0, y1 = i * w // gx, (i + 1) * w // gx, j * h // gy, (j + 1) * h // gy
            tb = b[y0:y1, x0:x1]
            if float(static[y0:y1, x0:x1].mean()) < 0.4 or float(ink[y0:y1, x0:x1].mean()) < 0.02 or float(tb.std()) < 0.2:
                continue
            wx0, wx1, wy0, wy1 = max(0, x0 - max_shift), min(w, x1 + max_shift), max(0, y0 - max_shift), min(h, y1 + max_shift)
            res = cv2.matchTemplate(a[wy0:wy1, wx0:wx1], tb, cv2.TM_CCOEFF_NORMED)
            if res.size:
                best = float(np.nanmax(res))
                out[j, i] = best if math.isfinite(best) else 0.0
    return out


# ------------------------------------------------------------------------------ geometry measures
def quad_perspective(quad: np.ndarray) -> Dict[str, float]:
    """Opposite-side length mismatch and worst corner-angle deviation of a TL, TR, BR, BL quadrilateral."""
    q = np.asarray(quad, np.float64)
    top, right = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
    bottom, left = np.linalg.norm(q[3] - q[2]), np.linalg.norm(q[0] - q[3])
    mismatch = max(abs(top - bottom) / max(top, bottom, 1e-9), abs(left - right) / max(left, right, 1e-9))
    devs = []
    for i in range(4):
        a, b, c = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9))
        devs.append(abs(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) - 90.0))
    return {"side_mismatch": float(mismatch), "angle_dev_deg": float(max(devs))}


def local_scales(h_img_from_canon: np.ndarray, size: Tuple[int, int]) -> List[float]:
    """sqrt(|det J|) of the canonical -> image homography at the 4 corners and the centre: the number of
    image pixels per canonical pixel there (1.0 or more = the photo holds at least canonical detail)."""
    w, h = size
    hm = np.asarray(h_img_from_canon, np.float64)
    out: List[float] = []
    for (u, v) in ((0, 0), (w, 0), (w, h), (0, h), (w / 2.0, h / 2.0)):
        den = hm[2, 0] * u + hm[2, 1] * v + hm[2, 2]
        if abs(den) < 1e-12:
            out.append(0.0)
            continue
        x = (hm[0, 0] * u + hm[0, 1] * v + hm[0, 2]) / den
        y = (hm[1, 0] * u + hm[1, 1] * v + hm[1, 2]) / den
        dxdu, dxdv = (hm[0, 0] - x * hm[2, 0]) / den, (hm[0, 1] - x * hm[2, 1]) / den
        dydu, dydv = (hm[1, 0] - y * hm[2, 0]) / den, (hm[1, 1] - y * hm[2, 1]) / den
        out.append(float(math.sqrt(abs(dxdu * dydv - dxdv * dydu))))
    return out
