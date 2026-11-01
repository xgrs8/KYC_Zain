"""Front+back merge (plan 7.4 / 8.5, D11) - shared by R.F.G and A.L.G.

Front stacked ABOVE back; common width = the wider side scaled to the
target width (default 1140 px); 24 px white gap; never downscale the
larger side; PNG output (JPEG q95 optional).  The exact same function is
imported by both apps from the vendored byte-identical rfalib (T-SYNC /
T-MERGE), so a labeled R.F.G sample and its generated twin are
pixel-identical.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .constants import MERGE_DEFAULT_GAP, MERGE_DEFAULT_WIDTH

GAP_FILL = (255, 255, 255)


def merge_images(front: Image.Image, back: Image.Image,
                 width: int = MERGE_DEFAULT_WIDTH,
                 gap: int = MERGE_DEFAULT_GAP) -> Image.Image:
    """Return the merged image: front over `gap` px of white over back."""
    # honour "no downscaling of the larger side" (D11)
    width = max(int(width), front.width, back.width)
    gap = max(0, int(gap))

    def fit(side: Image.Image) -> Image.Image:
        if side.width != width:
            height = max(1, round(side.height * width / side.width))
            side = side.resize((width, height), Image.LANCZOS)
        if side.mode not in ("RGB", "RGBA"):
            side = side.convert("RGB")
        return side

    f, b = fit(front), fit(back)
    out = Image.new("RGB", (width, f.height + gap + b.height), GAP_FILL)
    out.paste(f, (0, 0))
    out.paste(b, (0, f.height + gap))
    return out


def merge_geometry(front: Image.Image, back: Image.Image,
                   width: int, gap: int) -> dict:
    """Geometry dict stored in both sidecars (D11)."""
    eff = max(int(width), front.width, back.width)
    fh = round(front.height * eff / front.width) if front.width != eff else front.height
    bh = round(back.height * eff / back.width) if back.width != eff else back.height
    return {
        "order": "front-over-back",
        "width": int(eff),
        "gap": int(gap),
        "front_size": [eff, int(fh)],
        "back_size": [eff, int(bh)],
        "merged_size": [int(eff), int(fh) + int(gap) + int(bh)],
        "format": "PNG",
    }


def merge_files(front_path: str | Path, back_path: str | Path,
                out_path: str | Path,
                width: int = MERGE_DEFAULT_WIDTH,
                gap: int = MERGE_DEFAULT_GAP,
                jpeg: str | Path | None = None) -> dict:
    """Merge two image files, save PNG (optional JPEG q95 twin).

    Returns the geometry dict for the sidecars.
    """
    front = Image.open(front_path)
    back = Image.open(back_path)
    merged = merge_images(front, back, width, gap)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, format="PNG")
    if jpeg is not None:
        merged.convert("RGB").save(jpeg, format="JPEG", quality=95)
    return merge_geometry(front, back, width, gap)
