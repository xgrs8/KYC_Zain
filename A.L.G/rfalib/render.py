"""Text rendering onto card plates (plan 7.3).

Default backend (always works, zero DLLs): arabic_reshaper + python-bidi,
drawn with Pillow ImageDraw.  Optional backend: Pillow Raqm
(direction='rtl', language='ar') when PIL.features.check('raqm') is true
AND the startup probe passes; otherwise silent fallback + log.

NEVER cv2.putText for Arabic (Hershey fonts have no Arabic).

A slight seeded jitter (plus/minus 1 px baseline, 2 % size, optional
0.3 px blur) keeps samples from looking pixel-identical (7.3).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, features

try:  # optional backend probes must never break import
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    arabic_reshaper = None
    get_display = None

FONTS_DIR = Path(__file__).parent / "fonts"

_ARABIC_RE = None


def _init_re():
    global _ARABIC_RE
    if _ARABIC_RE is None:
        import re
        _ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
    return _ARABIC_RE


def has_arabic(text: str) -> bool:
    return bool(_init_re().search(text or ""))


# ---------------------------------------------------------------- backend

def _probe_raqm() -> bool:
    """Startup probe: render an Arabic word with direction params; any
    exception or missing feature -> backend unavailable (silent fallback)."""
    try:
        if not features.check("raqm"):
            return False
        font = ImageFont.truetype(str(FONTS_DIR / "NotoNaskhArabic-Regular.ttf"), 32)
        img = Image.new("RGB", (200, 60), "white")
        d = ImageDraw.Draw(img)
        d.text((5, 5), "بغداد", font=font, fill="black",
               direction="rtl", language="ar")
        extrema = img.convert("L").getextrema()
        return extrema[0] < 200  # glyphs were actually drawn
    except Exception:
        return False


_RAQM_CACHED: bool | None = None


def render_backend() -> str:
    """'raqm' when the optional backend is usable, else 'reshaper'."""
    global _RAQM_CACHED
    if _RAQM_CACHED is None:
        _RAQM_CACHED = _probe_raqm()
    return "raqm" if _RAQM_CACHED else "reshaper"


_RESHAPER = None


def _get_reshaper():
    global _RESHAPER
    if _RESHAPER is None:
        _RESHAPER = arabic_reshaper.ArabicReshaper(
            configuration={"support_ligatures": True})
    return _RESHAPER


def shape(text: str) -> str:
    """Full shaping pipeline for display: reshape + bidi reorder."""
    if not has_arabic(text):
        return text
    if render_backend() == "raqm":
        return text  # raqm does shaping + bidi itself
    reshaped = _get_reshaper().reshape(text)
    return get_display(reshaped, base_dir="R")


# ---------------------------------------------------------------- fonts

def _font_candidates(name: str) -> list[Path]:
    """Font locations: repo package, frozen _MEIPASS, exe dir."""
    fname = Path(name).name
    cands = [FONTS_DIR / fname]
    import sys
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands += [Path(meipass) / "rfalib" / "fonts" / fname,
                  Path(meipass) / "fonts" / fname]
    exe = getattr(sys, "executable", "")
    if exe:
        cands.append(Path(exe).parent / "rfalib" / "fonts" / fname)
    if Path(name).is_absolute():
        cands.insert(0, Path(name))
    return cands


def font_path(name: str) -> Path:
    """Resolve a font by file name from rfalib/fonts (repo or frozen)."""
    for cand in _font_candidates(name):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"font {name!r} not found; looked in {[str(c) for c in _font_candidates(name)]}; "
        f"bundled: {[f.name for f in FONTS_DIR.glob('*.ttf')] if FONTS_DIR.exists() else []}")


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(name)), int(size))


# ---------------------------------------------------------------- measuring

def measure(text: str, font: ImageFont.FreeTypeFont, spacing: int = 0) -> tuple[int, int]:
    """(width, height) of the shaped text in pixels."""
    if spacing and len(text) > 1:
        w = sum(font.getlength(ch) + spacing for ch in text[:-1])
        w += font.getlength(text[-1])
        asc, desc = font.getmetrics()
        return int(w), asc + desc
    box = font.getbbox(text or " ")
    return (box[2] - box[0], box[3] - box[1])


# ---------------------------------------------------------------- drawing

def draw_text(img: Image.Image, box: tuple[int, int, int, int], text: str,
              font: str = "NotoNaskhArabic-Regular.ttf", size: int = 34,
              color: str | tuple = "#1a1a1a", align: str = "right",
              spacing: int = 0, valign: str = "center",
              rng: random.Random | None = None,
              blur: float = 0.0) -> Image.Image:
    """Draw `text` inside `box`=(x, y, w, h); returns the image (in place).

    align: 'left' | 'right' | 'center'   (Arabic fields are right-aligned)
    spacing: extra letter spacing in px (mono MRZ/digits use ~2)
    rng: when given, applies seeded jitter (+-1 px baseline, +-2 % size)
    """
    x, y, w, h = box
    size_eff = size
    dy = 0
    if rng is not None:
        dy = rng.randint(-1, 1)
        size_eff = max(8, int(round(size * (1 + rng.randint(-2, 2) / 100.0))))
    f = load_font(font, size_eff)
    shaped = shape(str(text))
    is_rtl = has_arabic(text)

    tw, th = measure(shaped, f, spacing)
    if valign == "center":
        ty = y + dy + max(0, (h - th) // 2)
    else:
        ty = y + dy
    ty = max(0, ty)

    if spacing and len(shaped) > 1:
        # manual letter-spaced drawing (used for MRZ / mono digits)
        from PIL import ImageFilter
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        cx = (x + w - tw) if align == "right" else (x if align == "left"
                                                    else x + (w - tw) // 2)
        cx = max(x, min(cx, x + max(0, w - tw)))
        for ch in shaped:
            ld.text((cx, ty), ch, font=f, fill=color)
            cx += f.getlength(ch) + spacing
        if blur and blur > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        _composite(img, layer)
        return img

    anchor_x = {"right": x + w - 1, "left": x + 1, "center": x + w // 2}[align]
    kw: dict[str, Any] = {}
    # both backends anchor at the ASCENDER line so the text's visual top
    # sits at ty regardless of backend (raqm adds direction/language only)
    anchor = {"right": "ra", "left": "la", "center": "ma"}[align]
    if render_backend() == "raqm" and is_rtl:
        kw.update(direction="rtl", language="ar")

    if blur and blur > 0:
        from PIL import ImageFilter
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((anchor_x, ty), shaped, font=f, fill=color, anchor=anchor, **kw)
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
        _composite(img, layer)
    else:
        d = ImageDraw.Draw(img)
        d.text((anchor_x, ty), shaped, font=f, fill=color, anchor=anchor, **kw)
    return img


def _composite(img: Image.Image, layer: Image.Image) -> None:
    """Composite an RGBA layer onto img whatever its mode."""
    if img.mode == "RGBA":
        img.alpha_composite(layer)
    else:
        base = img.convert("RGBA")
        base.alpha_composite(layer)
        img.paste(base.convert(img.mode), (0, 0))
