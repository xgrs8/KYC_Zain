"""Image helpers for view_sample: oriented load, optional zoom region, size cap, JPEG bytes."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg")
MIN_SIDE, MAX_SIDE = 256, 2400


def load_oriented(path: str | Path) -> Image.Image:
    """Open an image, apply the EXIF rotation flag (phone photos) and convert to RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def parse_region(region) -> tuple[float, float, float, float] | None:
    """Validate a zoom region [x0, y0, x1, y1] given as fractions 0..1 of the image."""
    if region is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in region)
    except (TypeError, ValueError):
        raise ValueError("region must be four numbers [x0, y0, x1, y1] between 0 and 1")
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("region must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if (x1 - x0) < 0.02 or (y1 - y0) < 0.02:
        raise ValueError("region is too small (minimum 2% of the width and height)")
    return x0, y0, x1, y1


def render_view(path: str | Path, region=None, max_side: int = 1600,
                quality: int = 90) -> tuple[bytes, dict]:
    """Return (jpeg_bytes, info). The longest side of the output is at most ``max_side``;
    small crops are upscaled so that small print becomes easier to read."""
    max_side = max(MIN_SIDE, min(int(max_side), MAX_SIDE))
    box = parse_region(region)
    im = load_oriented(path)
    full_w, full_h = im.size
    if box is not None:
        x0, y0, x1, y1 = box
        im = im.crop((round(x0 * full_w), round(y0 * full_h), round(x1 * full_w), round(y1 * full_h)))
    w, h = im.size
    scale = max_side / max(w, h)
    if scale != 1.0:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    info = {"source_size": [full_w, full_h], "returned_size": list(im.size),
            "region": list(box) if box else None, "upscaled": bool(scale > 1.0)}
    return buf.getvalue(), info


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in ALLOWED_SUFFIXES
