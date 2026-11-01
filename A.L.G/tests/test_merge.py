"""Merge geometry and a fixed golden pixel hash for the A.L.G support library."""
from __future__ import annotations

import hashlib

from PIL import Image

from rfalib import merge as M

GOLDEN_PIXEL_SHA = "43e013b0d4ae66887fc0ca06b449ece80437b27ee8b0fc90465b1d89bea741dc"


def _golden_pair() -> tuple[Image.Image, Image.Image]:
    front = Image.new("RGB", (400, 260), (205, 215, 225))
    back = Image.new("RGB", (380, 260), (215, 218, 226))
    return front, back


def test_merge_geometry():
    front, back = _golden_pair()
    merged = M.merge_images(front, back, width=1140, gap=24)
    assert merged.size == (1140, round(260 * 1140 / 400) + 24 + round(260 * 1140 / 380))
    h_front = round(260 * 1140 / 400)
    assert merged.getpixel((10, 10)) == (205, 215, 225)            # front on top
    assert merged.getpixel((10, h_front + 12)) == (255, 255, 255)  # white gap
    assert merged.getpixel((10, h_front + 34)) == (215, 218, 226)  # back below


def test_merge_golden_hash():
    """The merge result retains its expected pixel hash."""
    front, back = _golden_pair()
    merged = M.merge_images(front, back, width=1140, gap=24)
    assert hashlib.sha256(merged.tobytes()).hexdigest() == GOLDEN_PIXEL_SHA


def test_merge_files_roundtrip(tmp_path):
    front, back = _golden_pair()
    f, b = tmp_path / "ID_01_front.png", tmp_path / "ID_01_back.png"
    front.save(f)
    back.save(b)
    out = tmp_path / "ID_01.png"
    geom = M.merge_files(f, b, out, width=1140, gap=24)
    with Image.open(out) as im:
        assert im.size == tuple(geom["merged_size"])
        assert im.format == "PNG"
    assert geom["order"] == "front-over-back"
