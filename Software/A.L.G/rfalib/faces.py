"""Person/face detection and removal (plan D6, 7.4).

The ENTIRE person (head, shoulders, torso) is removed from every portrait
region - reduced to the plain photo backdrop, never blocked or blurred.

The gate runs on EVERY output image with OpenCV Haar frontal+profile
cascades at the plan's binding parameters (scaleFactor=1.08,
minNeighbors=5, minSize=24) at REFERENCE_WIDTH = 400 px (just under the
ID template's native 454 px - the E4 evidence regime: the template's own
Mr. Bean portrait and its ghost are detected there, its 16 px MRZ text
and guilloche artwork are not).  Because rendered Arabic/MRZ text
occasionally still produces weak sub-threshold responses (plan risk 8),
any detection is CONFIRMED by a stricter re-run at minNeighbors=8 on the
same downscaled image: real persons survive confirmation (the template's
portraits survive minNeighbors=10), weak text/artifact responses do not.
Confirmed detections fail the run and quarantine the sample
(fail-closed).  No histogram equalization (it amplifies flat text into
face-like contrast).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import FACE_MIN_NEIGHBORS, FACE_MIN_SIZE, FACE_SCALE_FACTOR

REFERENCE_WIDTH = 400   # E4-verified detection scale (persons yes, text no)
CONFIRM_NEIGHBORS = 8   # confirmation threshold; persons survive, artifacts die


def _cascade(name: str):
    path = Path(cv2.data.haarcascades) / name
    if not path.exists():  # pragma: no cover - opencv-python always ships them
        raise FileNotFoundError(f"cascade missing: {path}")
    return cv2.CascadeClassifier(str(path))


_FRONTAL = None
_PROFILE = None


def cascades():
    global _FRONTAL, _PROFILE
    if _FRONTAL is None:
        _FRONTAL = _cascade("haarcascade_frontalface_default.xml")
        _PROFILE = _cascade("haarcascade_profileface.xml")
    return _FRONTAL, _PROFILE


def _detect_all(gray: np.ndarray, neighbors: int) -> list[tuple[int, int, int, int]]:
    frontal, profile = cascades()
    kw = dict(scaleFactor=FACE_SCALE_FACTOR, minNeighbors=neighbors,
              minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE))
    found: list[tuple[int, int, int, int]] = []

    def _lst(res):
        try:
            return res.tolist()
        except AttributeError:
            return []

    for x, y, w, h in _lst(frontal.detectMultiScale(gray, **kw)):
        found.append((int(x), int(y), int(w), int(h)))
    for x, y, w, h in _lst(profile.detectMultiScale(gray, **kw)):
        found.append((int(x), int(y), int(w), int(h)))
    width = gray.shape[1]
    flipped = cv2.flip(gray, 1)
    for x, y, w, h in _lst(profile.detectMultiScale(flipped, **kw)):
        found.append((width - int(x) - int(w), int(y), int(w), int(h)))
    return found


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def detect_persons(image_bgr: np.ndarray,
                   reference_width: int | None = None) -> list[tuple[int, int, int, int]]:
    """Detect faces/persons: frontal cascade + profile cascade on the image
    and on its horizontal mirror (covers both profile orientations), at the
    plan's binding parameters, then CONFIRMED at minNeighbors=8.

    When `reference_width` is given and the image is wider, it is
    area-downscaled to that width first (template-native scale).
    Returned rects are in the ORIGINAL image coordinates."""
    img = image_bgr
    scale = 1.0
    if reference_width and img.shape[1] > reference_width:
        scale = img.shape[1] / reference_width
        img = cv2.resize(img, (reference_width,
                               max(1, round(img.shape[0] / scale))),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raw = _detect_all(gray, FACE_MIN_NEIGHBORS)
    if not raw:
        return []
    confirmed = _detect_all(gray, CONFIRM_NEIGHBORS)
    if not confirmed:
        return []
    kept = [d for d in raw if any(_iou(d, c) > 0.35 for c in confirmed)]
    if scale != 1.0:
        kept = [(int(x * scale), int(y * scale), int(w * scale), int(h * scale))
                for x, y, w, h in kept]
    return kept


def blank_fill(image_bgr: np.ndarray, rect: tuple[int, int, int, int],
               ring: int = 14) -> np.ndarray:
    """Erase the whole person region: median-background fill plus matched
    per-pixel Gaussian noise (in place). Returns the image."""
    x, y, w, h = (int(v) for v in rect)
    hh, ww = image_bgr.shape[:2]
    x, y = max(0, min(x, ww - 1)), max(0, min(y, hh - 1))
    w, h = max(1, min(w, ww - x)), max(1, min(h, hh - y))
    x0, y0 = max(0, x - ring), max(0, y - ring)
    x1, y1 = min(ww, x + w + ring), min(hh, y + h + ring)

    crop = image_bgr[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[y - y0:y - y0 + h, x - x0:x - x0 + w] = 1
    border = crop[mask == 0]
    if border.size:
        med = np.median(border, axis=0)
        sigma = float(max(1.0, np.std(border) * 0.5))
    else:
        med = np.median(crop.reshape(-1, 3), axis=0)
        sigma = 2.0

    rng = np.random.default_rng(0x5EED)
    fill = np.clip(
        med[None, None, :] + rng.normal(0.0, sigma, (h, w, 3)), 0, 255
    ).astype(np.uint8)
    image_bgr[y:y + h, x:x + w] = fill
    return image_bgr


def check_image_file(path: str | Path) -> list[tuple[int, int, int, int]]:
    """Detections on an output file at the template-native reference scale;
    [] means the face gate passes."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return detect_persons(img, reference_width=REFERENCE_WIDTH)


def check_batch(paths) -> dict[str, list[tuple[int, int, int, int]]]:
    """Face gate over every output; any detection fails the run (T-FACE)."""
    results: dict[str, list] = {}
    for p in paths:
        results[str(p)] = check_image_file(p)
    return results
