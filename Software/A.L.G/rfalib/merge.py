"""Front+back image merge for A.L.G ID labels.

Front stacked ABOVE back; common width = the wider side scaled to the
target width (default 1600 px, the width of the project dataset); 24 px
white gap; never downscale the larger side; EXIF rotation applied; PNG
output (JPEG q95 optional). The same function is used for every generated
ID label so output geometry remains consistent, and the GATE Block
(main/GATE Block/gate_block/id_merge.py) repeats it step for step: an ID
merged here and one merged by the gate are the same pixels.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

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
    # Phone photos often store their rotation as an EXIF tag instead of in the pixels. Viewers (and the MCP
    # server's view_sample) show them upright, so the merged label image must be upright too; the PNG has no
    # EXIF tag that could rotate it back. Images without the tag are unchanged.
    front = ImageOps.exif_transpose(Image.open(front_path))
    back = ImageOps.exif_transpose(Image.open(back_path))
    merged = merge_images(front, back, width, gap)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path, format="PNG")
    if jpeg is not None:
        merged.convert("RGB").save(jpeg, format="JPEG", quality=95)
    return merge_geometry(front, back, width, gap)
