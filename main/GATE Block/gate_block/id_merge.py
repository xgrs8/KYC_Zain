"""National ID: merge the front and back uploads into one image.

This mirrors A.L.G's own merge (``Software/A.L.G/rfalib/merge.py`` in the KYC_Zain repository) step for
step, so the Gate Block hands TeleOCR exactly the kind of image the dataset was built from:

    * common width = max(target width, front width, back width): the larger side is never downscaled;
    * a side whose width differs is resized to it with PIL LANCZOS, height = round(h * width / w);
    * the front goes on top and the back below, separated by a ``gap_px`` pure-white band.

``target_width`` must equal the ``merge_width`` setting A.L.G used for the dataset (1600, also A.L.G's
default). The reference pair front.jpg + back.jpg -> ID_23.png was made at 1600, and this module reproduces
that file pixel for pixel. Like A.L.G's ``merge_files``, the EXIF orientation of each upload is applied
first, so a phone photo stored sideways is merged upright. ``tests/test_imageio_merge.py`` checks all of it
against ``rfalib.merge`` itself when A.L.G can be imported: the function on many sizes and image modes,
whole files with and without an orientation tag, and the default width.

The merge does not crop or straighten anything. The Gate Block runs it right after Gate 1 (the upload
check) and before the quality gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .config import IdMergeCfg
from .imageio import ImageLoadError, ImageSource, array_to_bgr, open_pil

_RESAMPLE = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}


@dataclass
class MergedID:
    image: np.ndarray               # BGR uint8, (H, width, 3)
    front_rows: Tuple[int, int]     # [y0, y1) of the front side
    gap_rows: Tuple[int, int]       # [y0, y1) of the white band
    back_rows: Tuple[int, int]      # [y0, y1) of the back side

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def front(self) -> np.ndarray:
        return self.image[self.front_rows[0]:self.front_rows[1]]

    def back(self) -> np.ndarray:
        return self.image[self.back_rows[0]:self.back_rows[1]]


def merge_pil(front: Image.Image, back: Image.Image, cfg: IdMergeCfg) -> Image.Image:
    """A.L.G's ``merge_images``, line for line (with the width and gap taken from the config)."""
    width = max(int(cfg.target_width), front.width, back.width)       # never downscale the larger side
    gap = max(0, int(cfg.gap_px))
    resample = _RESAMPLE[cfg.resample]

    def fit(side: Image.Image) -> Image.Image:
        if side.width != width:
            side = side.resize((width, max(1, round(side.height * width / side.width))), resample)
        if side.mode not in ("RGB", "RGBA"):
            side = side.convert("RGB")
        return side

    f, b = fit(front), fit(back)
    out = Image.new("RGB", (width, f.height + gap + b.height), (cfg.background,) * 3)
    out.paste(f, (0, 0))
    out.paste(b, (0, f.height + gap))
    return out


def _as_pil(src: ImageSource, max_pixels: int = 0) -> Image.Image:
    if isinstance(src, np.ndarray):
        return Image.fromarray(np.ascontiguousarray(array_to_bgr(src)[:, :, ::-1]), "RGB")
    return open_pil(src, max_pixels)            # EXIF orientation applied


def merge_id(front: ImageSource, back: ImageSource, cfg: IdMergeCfg, max_pixels: int = 0) -> MergedID:
    """Merge the two uploads of a national ID into one image: front on top, white band, back below."""
    f, b = _as_pil(front, max_pixels), _as_pil(back, max_pixels)
    merged = merge_pil(f, b, cfg)
    width = merged.width
    fh = f.height if f.width == width else max(1, round(f.height * width / f.width))
    gap = max(0, int(cfg.gap_px))
    image = np.ascontiguousarray(np.asarray(merged.convert("RGB"))[:, :, ::-1])
    return MergedID(image=image, front_rows=(0, fh), gap_rows=(fh, fh + gap), back_rows=(fh + gap, merged.height))


def split_merged(image_bgr: np.ndarray, cfg: IdMergeCfg) -> MergedID:
    """Find the two sides of an *already merged* ID image by locating its white band.

    The band is a run of rows that are background-white across the whole width, at least a third of
    ``gap_px`` high, whose centre lies in the middle 40 % of the image (so a white margin at the top or
    bottom is never mistaken for it). JPEG compression frays the edges of the band; a run shorter than
    ``gap_px`` is therefore widened to ``gap_px`` around its centre, which restores the exact rows.
    Raises ``ImageLoadError('unreadable')`` when there is no such band."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ImageLoadError("unreadable", "a merged ID must be a colour image")
    h = image_bgr.shape[0]
    gap = max(1, int(cfg.gap_px))
    pure = image_bgr.min(axis=(1, 2)) >= int(cfg.background) - 3
    runs, start = [], None
    for y, v in enumerate(pure):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, h))
    need = max(1, min(gap, max(4, gap // 3)))
    best: Optional[Tuple[int, int]] = None
    for (a, b) in runs:
        centre = (a + b) / 2.0 / h
        if (b - a) >= need and 0.3 <= centre <= 0.7:
            if best is None or abs(centre - 0.5) < abs((best[0] + best[1]) / 2.0 / h - 0.5):
                best = (a, b)
    if best is None:
        raise ImageLoadError("unreadable", "no white band between two sides: this is not a merged ID image")
    a, b = best
    if (b - a) < gap:
        a = int(round((a + b) / 2.0 - gap / 2.0))
        a = min(max(1, a), h - gap - 1)
        b = a + gap
    if a < 1 or b > h - 1:
        raise ImageLoadError("unreadable", "the white band leaves no room for two sides")
    return MergedID(image=image_bgr, front_rows=(0, a), gap_rows=(a, b), back_rows=(b, h))
