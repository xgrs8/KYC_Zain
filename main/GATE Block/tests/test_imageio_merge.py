"""Image loading and the national-ID merge (front + back -> one image, the same rule as A.L.G)."""
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gate_block.config import GateConfig
from gate_block.id_merge import merge_id, merge_pil, split_merged
from gate_block.imageio import ImageLoadError, load_image, to_bgr

ROOT = Path(__file__).resolve().parent.parent
FRONT = ROOT / "templates" / "national_id" / "front.jpg"
BACK = ROOT / "templates" / "national_id" / "back.jpg"
REFERENCE = ROOT / "tests" / "data" / "id_merge_reference.png"
CFG = GateConfig.from_dict(None)


def _png_bytes(arr_rgb: np.ndarray, **save) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_rgb).save(buf, "PNG", **save)
    return buf.getvalue()


def _textured(w: int, h: int, seed: int = 0) -> np.ndarray:
    """A picture with large and small structure (coarse colour blocks plus fine noise)."""
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    fine = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    return (coarse // 2 + fine + 30).astype(np.uint8)


# ---------------------------------------------------------------------- loading
def test_loads_path_bytes_and_array_identically(tmp_path):
    arr = _textured(800, 600)
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), arr)
    from_path = load_image(str(path), CFG.image)
    from_pathlike = load_image(path, CFG.image)
    from_bytes = load_image(path.read_bytes(), CFG.image)
    from_array = load_image(arr, CFG.image)
    for loaded in (from_path, from_pathlike, from_bytes, from_array):
        assert loaded.original_size == (800, 600) and loaded.work.dtype == np.uint8 and loaded.work.shape == (600, 800, 3)
        assert np.array_equal(loaded.work, arr)
    assert from_path.source == str(path) and from_bytes.source == "<bytes>" and from_array.source == "<array>"


def test_work_and_small_copies_respect_the_configured_sizes():
    loaded = load_image(_textured(4000, 3000), CFG.image)
    assert max(loaded.work.shape[:2]) == CFG.image.work_max_side and max(loaded.small.shape[:2]) == CFG.image.small_max_side
    assert loaded.work_scale == pytest.approx(CFG.image.work_max_side / 4000)
    assert loaded.small_to_work == pytest.approx(CFG.image.work_max_side / CFG.image.small_max_side)


def test_exif_orientation_is_applied():
    arr = np.zeros((300, 600, 3), np.uint8)
    arr[:, :300] = (200, 0, 0)                                   # left half red (RGB)
    arr[:, 300:] = (0, 0, 200)
    arr += _textured(600, 300, 1) // 8                           # at most +31: no overflow
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6                                             # "rotate 90 degrees clockwise to display"
    Image.fromarray(arr).save(buf, "JPEG", exif=exif, quality=95)
    loaded = load_image(buf.getvalue(), CFG.image)
    assert loaded.original_size == (300, 600)                    # upright now
    top, bottom = loaded.work[:200].mean(axis=(0, 1)), loaded.work[400:].mean(axis=(0, 1))
    assert top[2] > 150 > top[0] and bottom[0] > 150 > bottom[2]  # BGR: red on top, blue below


@pytest.mark.parametrize("mode", ["L", "LA", "RGBA", "P", "CMYK", "I;16", "1"])
def test_unusual_image_modes_become_bgr(mode):
    base = Image.fromarray(_textured(400, 300))
    im = base.convert("I").point(lambda v: v * 200).convert("I;16") if mode == "I;16" else base.convert(mode)
    buf = io.BytesIO()
    im.save(buf, "TIFF" if mode in ("CMYK", "I;16") else "PNG")
    bgr, _ = to_bgr(buf.getvalue(), CFG.image.max_pixels)
    assert bgr.dtype == np.uint8 and bgr.shape == (300, 400, 3)


def test_transparency_is_flattened_onto_white():
    rgba = np.zeros((300, 400, 4), np.uint8)                     # fully transparent black
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    bgr, _ = to_bgr(buf.getvalue(), CFG.image.max_pixels)
    assert bgr.min() == 255


@pytest.mark.parametrize("src, code", [
    (b"this is not an image", "unreadable"),
    (b"", "unreadable"),
    ("/no/such/file.jpg", "unreadable"),
    (np.zeros((0, 10, 3), np.uint8), "unreadable"),
    (np.zeros((300, 400, 2), np.uint8), "unreadable"),
    (np.array([["a", "b"]]), "unreadable"),
    (np.full((100, 100, 3), 128, np.uint8), "too_small"),
    (np.full((600, 800, 3), 255, np.uint8), "blank"),
    (np.zeros((600, 800, 3), np.uint8), "blank"),
])
def test_bad_uploads_have_a_machine_readable_code(src, code):
    with pytest.raises(ImageLoadError) as err:
        load_image(src, CFG.image)
    assert err.value.code == code


def test_truncated_and_oversized_files_are_refused():
    data = FRONT.read_bytes()
    with pytest.raises(ImageLoadError) as err:
        load_image(data[: len(data) // 3], CFG.image)
    assert err.value.code == "unreadable"
    small_limit = GateConfig.from_dict({"image": {"max_pixels": 100_000}})
    with pytest.raises(ImageLoadError) as err:
        load_image(data, small_limit.image)
    assert err.value.code == "too_large"


# ---------------------------------------------------------------------- the merge
def test_merge_reproduces_the_reference_pixel_for_pixel():
    merged = merge_id(FRONT, BACK, CFG.id_merge)
    expected = cv2.imread(str(REFERENCE))
    assert merged.image.shape == expected.shape == (2048, 1600, 3)
    assert np.array_equal(merged.image, expected)
    assert merged.front_rows == (0, 1020) and merged.gap_rows == (1020, 1044) and merged.back_rows == (1044, 2048)
    assert merged.image[1020:1044].min() == 255                  # the white band
    assert merged.front().shape == (1020, 1600, 3) and merged.back().shape == (1004, 1600, 3)


def test_merge_accepts_paths_bytes_and_arrays():
    a = merge_id(FRONT, BACK, CFG.id_merge).image
    b = merge_id(FRONT.read_bytes(), BACK.read_bytes(), CFG.id_merge).image
    c = merge_id(cv2.imread(str(FRONT)), cv2.imread(str(BACK)), CFG.id_merge).image
    assert np.array_equal(a, b) and np.array_equal(a, c)


@pytest.mark.parametrize("fsize, bsize, width", [
    ((1444, 921), (1449, 909), 1600),                            # both enlarged to the target width
    ((800, 500), (640, 400), 1600),
    ((4032, 3024), (3000, 2000), 4032),                          # the larger upload is never made smaller
    ((1600, 1000), (1600, 1010), 1600),                          # already at the target width: untouched
    ((2000, 1200), (1000, 700), 2000),
])
def test_merge_size_rule(fsize, bsize, width):
    f, b = _textured(*fsize, seed=1), _textured(*bsize, seed=2)
    m = merge_id(f, b, CFG.id_merge)
    fh, bh = (round(size[1] * width / size[0]) for size in (fsize, bsize))
    assert m.width == width and m.image.shape[0] == fh + 24 + bh
    assert m.front_rows == (0, fh) and m.back_rows == (fh + 24, fh + 24 + bh)
    assert m.image[m.gap_rows[0]:m.gap_rows[1]].min() == 255
    if fsize[0] == width:
        assert np.array_equal(m.front(), f)                      # a side at the common width is copied, not resampled
    if bsize[0] == width:
        assert np.array_equal(m.back(), b)


def test_merge_follows_the_configured_gap_width_and_background():
    cfg = GateConfig.from_dict({"id_merge": {"target_width": 1000, "gap_px": 10, "background": 200}}).id_merge
    m = merge_id(_textured(500, 300, 1), _textured(500, 320, 2), cfg)
    assert m.width == 1000 and m.gap_rows == (600, 610) and m.back_rows == (610, 610 + 640)
    assert np.all(m.image[600:610] == 200)


def test_merge_applies_exif_orientation():
    upright = np.asarray(Image.open(FRONT).convert("RGB"))
    sideways = Image.fromarray(np.rot90(upright, 1))             # as a phone stores a portrait-held shot
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    sideways.save(buf, "PNG", exif=exif)
    m = merge_id(buf.getvalue(), BACK, CFG.id_merge)
    assert np.array_equal(m.image, merge_id(FRONT, BACK, CFG.id_merge).image)


def test_merge_refuses_unreadable_uploads():
    with pytest.raises(ImageLoadError):
        merge_id(b"junk", BACK, CFG.id_merge)


def _alg_merge():
    """A.L.G's own merge function, when the KYC_Zain repository is around (set ALG_PATH otherwise)."""
    candidates = [os.environ.get("ALG_PATH"), ROOT.parent.parent / "Software" / "A.L.G", ROOT.parent / "kyc_zain" / "Software" / "A.L.G"]
    for c in candidates:
        if c and (Path(c) / "rfalib" / "merge.py").is_file():
            sys.path.insert(0, str(c))
            try:
                from rfalib.merge import merge_images
                return merge_images
            except Exception:
                return None
            finally:
                sys.path.remove(str(c))
    return None


def test_merge_is_identical_to_alg():
    merge_images = _alg_merge()
    if merge_images is None:
        pytest.skip("A.L.G (rfalib.merge) not found next to this folder; set ALG_PATH to Software/A.L.G to run this check")
    for fsize, bsize in (((1444, 921), (1449, 909)), ((800, 500), (640, 400)), ((4032, 3024), (3000, 2000)),
                         ((1600, 1000), (1600, 1010)), ((1601, 900), (1599, 901)), ((333, 211), (1700, 1100))):
        f, b = Image.fromarray(_textured(*fsize, seed=3)), Image.fromarray(_textured(*bsize, seed=4))
        theirs = np.asarray(merge_images(f, b, width=CFG.id_merge.target_width, gap=CFG.id_merge.gap_px).convert("RGB"))
        ours = np.asarray(merge_pil(f, b, CFG.id_merge).convert("RGB"))
        assert np.array_equal(ours, theirs), (fsize, bsize)
    rgba = Image.fromarray(_textured(700, 400, 5)).convert("RGBA")
    grey = Image.fromarray(_textured(650, 420, 6)).convert("L")
    theirs = np.asarray(merge_images(rgba, grey, width=1600, gap=24).convert("RGB"))
    assert np.array_equal(np.asarray(merge_pil(rgba, grey, CFG.id_merge).convert("RGB")), theirs)


def _alg_module(name: str):
    """A module of A.L.G's rfalib (``merge``, ``constants``), when the KYC_Zain repository is around."""
    import importlib
    candidates = [os.environ.get("ALG_PATH"), ROOT.parent.parent / "Software" / "A.L.G", ROOT.parent / "kyc_zain" / "Software" / "A.L.G"]
    for c in candidates:
        if c and (Path(c) / "rfalib" / "merge.py").is_file():
            sys.path.insert(0, str(c))
            try:
                return importlib.import_module(f"rfalib.{name}")
            except Exception:
                return None
            finally:
                sys.path.remove(str(c))
    return None


def test_merging_files_is_identical_to_alg_also_for_rotated_phone_photos(tmp_path):
    """The dataset's IDs are merged by A.L.G's merge_files, the uploads by the gate: same files, same pixels."""
    alg = _alg_module("merge")
    if alg is None:
        pytest.skip("A.L.G (rfalib.merge) not found next to this folder; set ALG_PATH to Software/A.L.G to run this check")
    upright_f = np.asarray(Image.open(FRONT).convert("RGB"))
    upright_b = np.asarray(Image.open(BACK).convert("RGB"))
    cases = {"no tag": (FRONT, BACK)}
    for tag, turn in ((6, 1), (8, -1), (3, 2)):              # stored sideways / upside down, EXIF says how to show it
        exif = Image.Exif()
        exif[0x0112] = tag
        pair = []
        for name, arr, fmt in (("f", upright_f, "JPEG"), ("b", upright_b, "PNG")):
            path = tmp_path / f"{name}_{tag}.{fmt.lower()}"
            Image.fromarray(np.ascontiguousarray(np.rot90(arr, turn))).save(path, fmt, exif=exif, **({"quality": 95} if fmt == "JPEG" else {}))
            pair.append(path)
        cases[f"orientation {tag}"] = tuple(pair)
    for name, (f, b) in cases.items():
        out = tmp_path / f"alg_{name.replace(' ', '_')}.png"
        alg.merge_files(f, b, out, width=CFG.id_merge.target_width, gap=CFG.id_merge.gap_px)
        theirs = np.asarray(Image.open(out).convert("RGB"))
        ours = merge_id(f, b, CFG.id_merge).image[:, :, ::-1]
        assert ours.shape == theirs.shape and np.array_equal(ours, theirs), name
    assert np.array_equal(np.asarray(Image.open(tmp_path / "alg_no_tag.png").convert("RGB")),
                          np.asarray(Image.open(REFERENCE).convert("RGB")))     # and both equal ID_23.png


def test_the_merge_width_is_the_same_as_alg_s_default():
    constants = _alg_module("constants")
    if constants is None:
        pytest.skip("A.L.G (rfalib.constants) not found next to this folder; set ALG_PATH to Software/A.L.G")
    assert CFG.id_merge.target_width == constants.MERGE_DEFAULT_WIDTH
    assert CFG.id_merge.gap_px == constants.MERGE_DEFAULT_GAP


# ---------------------------------------------------------------------- splitting a merged image again
def test_split_finds_the_band_of_the_reference():
    m = split_merged(cv2.imread(str(REFERENCE)), CFG.id_merge)
    assert m.front_rows == (0, 1020) and m.back_rows == (1044, 2048)


@pytest.mark.parametrize("quality", [95, 75, 40])
def test_split_survives_jpeg_compression(quality):
    ok, buf = cv2.imencode(".jpg", cv2.imread(str(REFERENCE)), [cv2.IMWRITE_JPEG_QUALITY, quality])
    m = split_merged(cv2.imdecode(buf, cv2.IMREAD_COLOR), CFG.id_merge)
    assert m.front_rows == (0, 1020) and m.back_rows == (1044, 2048)


def test_split_ignores_white_margins_and_refuses_single_photos():
    img = cv2.imread(str(REFERENCE))
    padded = cv2.copyMakeBorder(img, 60, 60, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    m = split_merged(padded, CFG.id_merge)
    assert m.gap_rows == (1080, 1104)
    with pytest.raises(ImageLoadError):
        split_merged(cv2.imread(str(FRONT)), CFG.id_merge)
    with pytest.raises(ImageLoadError):
        split_merged(np.full((600, 800, 3), 255, np.uint8)[:, :, 0], CFG.id_merge)
