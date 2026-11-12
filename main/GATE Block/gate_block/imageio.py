"""Robust image loading.

Phone photos arrive rotated by an EXIF flag, as RGBA / palette / 16-bit / CMYK files, truncated, huge or
simply blank. Each of those becomes either a clean BGR uint8 array or an :class:`ImageLoadError` with a
machine-readable ``code`` (the pipeline maps the code to an Iraqi-Arabic retake message). Nothing in the
Gate Block reads image files any other way.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import ImageCfg

try:  # iPhone HEIC photos open only when the optional plugin is installed
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

ImageSource = Union[str, "os.PathLike[str]", bytes, bytearray, np.ndarray]


class ImageLoadError(Exception):
    """The upload cannot be analysed. ``code``: unreadable | too_small | too_large | blank."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class LoadedImage:
    work: np.ndarray                 # BGR uint8, long side <= image.work_max_side
    small: np.ndarray                # BGR uint8, long side <= image.small_max_side
    original_size: Tuple[int, int]   # (width, height) after EXIF orientation
    work_scale: float                # work pixels per original pixel (<= 1.0)
    small_to_work: float             # work pixels per small pixel (>= 1.0)
    source: str                      # file name, "<bytes>" or "<array>"

    @property
    def work_size(self) -> Tuple[int, int]:
        h, w = self.work.shape[:2]
        return (w, h)


def pil_to_bgr(im: Image.Image) -> np.ndarray:
    """Any PIL image -> contiguous BGR uint8 (transparency flattened onto white)."""
    mode = im.mode
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, rgba).convert("RGB")
    elif mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.asarray(im).astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    elif mode != "RGB":
        im = im.convert("RGB")
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def open_pil(src: Union[str, "os.PathLike[str]", bytes, bytearray], max_pixels: int = 0) -> Image.Image:
    """Open a file / bytes as a fully loaded PIL image with the EXIF orientation applied."""
    if isinstance(src, (bytes, bytearray)):
        handle: Union[str, io.BytesIO] = io.BytesIO(bytes(src))
    else:
        handle = os.fspath(src)
        if not os.path.isfile(handle):
            raise ImageLoadError("unreadable", f"file not found: {handle}")
    try:
        with Image.open(handle) as im:
            w0, h0 = im.size
            if max_pixels and w0 * h0 > max_pixels:
                raise ImageLoadError("too_large", f"image has {w0 * h0} pixels (limit {max_pixels})")
            im = ImageOps.exif_transpose(im)
            im.load()
            return im.copy()
    except ImageLoadError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("too_large", str(exc)) from exc
    except Exception as exc:  # corrupt, truncated or unsupported file
        raise ImageLoadError("unreadable", f"cannot decode image: {exc}") from exc


def array_to_bgr(arr: np.ndarray) -> np.ndarray:
    """Grey / BGR / BGRA array with values 0..255 (any numeric dtype) -> contiguous BGR uint8."""
    if arr.size == 0 or arr.dtype.kind not in "uif":
        raise ImageLoadError("unreadable", f"unsupported array (shape {arr.shape}, dtype {arr.dtype})")
    if arr.ndim == 2:
        out = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        out = np.clip(arr[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        out = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ImageLoadError("unreadable", f"unsupported array shape {arr.shape}")
    return np.ascontiguousarray(out)


def resize_long_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img, 1.0
    s = max_side / float(long_side)
    out = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return out, s


def to_bgr(src: ImageSource, max_pixels: int = 0) -> Tuple[np.ndarray, str]:
    """Decode any supported source to a full-resolution BGR uint8 array. Returns (image, source name)."""
    if isinstance(src, np.ndarray):
        return array_to_bgr(src), "<array>"
    name = "<bytes>" if isinstance(src, (bytes, bytearray)) else os.fspath(src)
    return pil_to_bgr(open_pil(src, max_pixels)), name


def load_image(src: ImageSource, cfg: ImageCfg) -> LoadedImage:
    """Load ``src`` (path, bytes, or a BGR / grey / BGRA array) into a :class:`LoadedImage`."""
    bgr, name = to_bgr(src, cfg.max_pixels)
    h, w = bgr.shape[:2]
    if min(h, w) < cfg.min_side:
        raise ImageLoadError("too_small", f"image is {w}x{h}; the shorter side must be at least {cfg.min_side}px")
    probe = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
    if float(probe.std()) < 2.0:
        raise ImageLoadError("blank", "image has (almost) no content")
    work, work_scale = resize_long_side(bgr, cfg.work_max_side)
    small, _ = resize_long_side(work, cfg.small_max_side)
    return LoadedImage(work=work, small=small, original_size=(w, h), work_scale=work_scale,
                       small_to_work=work.shape[1] / float(small.shape[1]), source=name)
