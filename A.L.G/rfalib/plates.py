"""Clean-plate construction (plan 7.1 step 1, 4.2).

Builds the clean working plates from the user's template images:
  - split the ID template image into front / back at the calibrated boxes
  - upscale 3x (LANCZOS + mild unsharp)
  - erase every text box and the watermark with cv2.inpaint (TELEA, r=3)
  - fill every photo slot with median background + matched noise (D6)
Plates are cached in <workspace>/plates/ keyed by template hash + spec hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import faces

UNSARP_AMOUNT = 0.4
INPAINT_RADIUS = 3


# ---------------------------------------------------------------- primitives

def load_template(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read template image: {path}")
    return img


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = (int(v) for v in box)
    return img[y:y + h, x:x + w].copy()


def upscale(img: np.ndarray, scale: float) -> np.ndarray:
    """LANCZOS upscale + mild unsharp (amount 0.4); background graphics stay
    interpolated - never redrawn (we do not invent detail)."""
    if scale == 1:
        out = img.copy()
    else:
        out = cv2.resize(img, (max(1, round(img.shape[1] * scale)),
                               max(1, round(img.shape[0] * scale))),
                         interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(out, 1.0 + UNSARP_AMOUNT, blur, -UNSARP_AMOUNT, 0)


def erase_regions(img: np.ndarray, boxes) -> np.ndarray:
    """cv2.inpaint (TELEA, radius 3) over the given [x,y,w,h] boxes."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        x, y, w, h = (int(v) for v in box)
        x, y = max(0, x), max(0, y)
        w, h = max(1, min(w, img.shape[1] - x)), max(1, min(h, img.shape[0] - y))
        mask[y:y + h, x:x + w] = 255
    if mask.max() == 0:
        return img
    return cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def fill_photo_slots(img: np.ndarray, slots) -> np.ndarray:
    """Blank every portrait region to the plain backdrop (whole person gone)."""
    for box in slots:
        if not box:
            continue
        faces.blank_fill(img, box)
    return img


# ---------------------------------------------------------------- spec-driven

def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def plate_cache_path(plates_dir: Path, spec: dict, side: str) -> Path:
    return Path(plates_dir) / f"{spec['doc_type']}_{side}_{_spec_hash(spec)}.png"


def build_side_plate(template_img: np.ndarray, side_spec: dict,
                     scale: float, photo_slots: list | None = None,
                     watermark_box=None) -> np.ndarray:
    """Build one clean side plate at `scale` from the template (BGR)."""
    plate = crop(template_img, side_spec["box"])
    plate = upscale(plate, scale)
    boxes = []
    for fspec in (side_spec.get("fields") or {}).values():
        boxes.append(fspec["box"])
    if side_spec.get("mrz"):
        boxes.append(side_spec["mrz"]["box"])
    for b in side_spec.get("erase", []) or []:
        boxes.append(b)
    if watermark_box:
        boxes.append(watermark_box)
    plate = erase_regions(plate, boxes)
    slots = []
    for slot in (photo_slots or []):
        if slot:
            slots.append(slot)
    plate = fill_photo_slots(plate, slots)
    return plate


def ensure_plates(spec: dict, template_path: Path, plates_dir: Path,
                  force: bool = False) -> dict[str, Path]:
    """Build (or reuse cached) clean plates for every side of a spec.

    Returns {side_name: plate_path}.  The ID spec has front+back; TAX/LIC
    specs have a single 'front' side.
    """
    plates_dir = Path(plates_dir)
    plates_dir.mkdir(parents=True, exist_ok=True)
    template_img = load_template(template_path)
    scale = float(spec.get("plate_scale", 3))
    out: dict[str, Path] = {}
    for side_name, side_spec in spec["sides"].items():
        cache = plate_cache_path(plates_dir, spec, side_name)
        if cache.exists() and not force:
            out[side_name] = cache
            continue
        slots = []
        if side_name == "front":
            for key in ("front_main", "front_ghost"):
                slot = (spec.get("photo_slots") or {}).get(key)
                if slot:
                    slots.append(slot)
        plate = build_side_plate(
            template_img, side_spec, scale, slots,
            spec.get("watermark_box"))
        cv2.imwrite(str(cache), plate)
        out[side_name] = cache
    return out


def to_pil(plate_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
