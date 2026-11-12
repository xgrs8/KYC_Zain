"""The measuring instruments on their own: synthetic inputs with a known answer."""
import cv2
import numpy as np
import pytest

from gate_block import geometry as G
from gate_block import measures as M
from gate_block.types import SEV_RANK, at_least, rank, worst


def text_like(w=600, h=400, seed=0):
    """Dark strokes on light paper."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 225, np.float32)
    for _ in range(120):
        x, y = int(rng.integers(10, w - 60)), int(rng.integers(10, h - 20))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(2, 6))), 40, -1)
    return img


# ---------------------------------------------------------------------- severities
def test_severity_ordering_helpers():
    assert [rank(s) for s in ("none", "mild", "moderate", "severe")] == [0, 1, 2, 3] and rank("unknown") < 0
    assert set(SEV_RANK) == {"none", "mild", "moderate", "severe"}
    assert at_least("severe", "moderate") and at_least("moderate", "moderate") and not at_least("mild", "moderate")
    assert not at_least("unknown", "mild")                        # unknown is never "at least" anything
    assert worst(["none", "severe", "mild"]) == "severe" and worst([]) == "none"


# ---------------------------------------------------------------------- sharpness
def test_edge_sharpness_falls_with_blur_and_ignores_exposure_and_contrast():
    img = text_like()
    values = [M.edge_sharpness(cv2.GaussianBlur(img, (0, 0), s) if s else img)[0] for s in (0, 1, 2, 3, 5)]
    assert all(a > b for a, b in zip(values, values[1:]))
    sharp, n = M.edge_sharpness(img)
    assert n > 500
    assert M.edge_sharpness(img * 0.5)[0] == pytest.approx(sharp, rel=0.05)                 # darker photo
    assert M.edge_sharpness((img - 128) * 0.5 + 128)[0] == pytest.approx(sharp, rel=0.05)   # flatter photo


def test_edge_sharpness_says_when_it_cannot_measure():
    flat = np.full((200, 300), 180, np.float32)
    assert M.edge_sharpness(flat) == (None, 0)
    img = text_like()
    mask = np.zeros(img.shape, bool)
    assert M.edge_sharpness(img, mask=mask)[0] is None


def test_blur_curve_is_invertible():
    img = text_like()
    sigmas, curve = M.blur_curve(img)
    assert np.all(np.diff(curve) < 0) and sigmas[0] == 0
    for s in (1.0, 2.0, 3.0, 4.0):
        measured, _ = M.edge_sharpness(cv2.GaussianBlur(img, (0, 0), s))
        assert M.sigma_equivalent(measured, sigmas, curve) == pytest.approx(s, abs=0.3)
    assert M.sigma_equivalent(curve[0] + 1.0, sigmas, curve) == 0.0
    assert M.sigma_equivalent(curve[-1] - 1.0, sigmas, curve) == sigmas[-1]


def test_motion_length_recovers_the_smear_and_its_direction():
    img = text_like()
    rulers = M.motion_rulers(img)
    assert M.motion_length(img, None, rulers)[0] == 0.0
    for angle in M.MOTION_ANGLES:
        for n in (5, 9, 15, 25):
            length, direction = M.motion_length(M.motion_smear(img, n, angle), None, rulers)
            assert direction == angle and length == pytest.approx(n, rel=0.1), (angle, n, length)
    # brightness, contrast and sensor noise have no direction; even blur shifts it far less than a smear
    assert M.motion_length(img * 0.5 + 60, None, rulers)[0] < 0.5
    rng = np.random.default_rng(1)
    assert M.motion_length(img + rng.normal(0, 6, img.shape).astype(np.float32), None, rulers)[0] < 1.0
    even, _ = M.motion_length(cv2.GaussianBlur(img, (0, 0), 3.0), None, rulers)
    assert even / np.sqrt(12) < 3.0                                  # as a Gaussian: less than the blur itself


def test_noise_estimate():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(text_like(), (0, 0), 1.0)
    assert M.noise_sigma(base) < 1.0
    for s in (4.0, 10.0):
        noisy = base + rng.normal(0, s, base.shape).astype(np.float32)
        assert M.noise_sigma(noisy) == pytest.approx(s, rel=0.25)


# ---------------------------------------------------------------------- lighting
def checker_template(w=960, h=600):
    """Light paper with a dark band and mid-grey panels: three brightness classes."""
    t = np.full((h, w), 220, np.float32)
    t[int(0.86 * h):] = 60
    t[int(0.2 * h):int(0.5 * h), int(0.1 * w):int(0.4 * w)] = 150
    return t


def test_even_light_gives_no_shadow_and_the_right_exposure():
    t = checker_template()
    for gain in (1.0, 0.6, 0.4):
        stats = M.lighting_stats(t * gain, t)
        assert stats["shadow_depth"] < 0.03 and stats["exposure_ratio"] == pytest.approx(gain, abs=0.03)


@pytest.mark.parametrize("gamma", [0.6, 0.8, 1.3, 1.8])
def test_a_tone_curve_is_not_a_shadow(gamma):
    t = checker_template()
    photo = 255.0 * (t / 255.0) ** (1.0 / gamma)
    assert M.lighting_stats(photo, t)["shadow_depth"] < 0.05
    with_flare = photo * 0.9 + 25.0
    assert M.lighting_stats(with_flare, t)["shadow_depth"] < 0.05


def test_a_shadow_is_measured_by_its_depth_also_through_a_tone_curve():
    t = checker_template()
    shade = np.ones_like(t)
    shade[:, : t.shape[1] // 3] = 0.6                              # the left third is 40 % darker
    assert M.lighting_stats(t * shade, t)["shadow_depth"] == pytest.approx(0.4, abs=0.04)
    toned = 255.0 * (t * shade / 255.0) ** (1.0 / 1.3)
    assert 0.2 < M.lighting_stats(toned, t)["shadow_depth"] < 0.5
    most = np.ones_like(t)
    most[:, : int(t.shape[1] * 0.7)] = 0.6                         # a shadow over most of the page is still a shadow
    assert M.lighting_stats(t * most, t)["shadow_depth"] == pytest.approx(0.4, abs=0.05)


def test_ignored_cells_do_not_count():
    t = checker_template()
    photo = t.copy()
    photo[: 200, : 320] *= 0.3                                     # something dark in the top-left corner
    assert M.lighting_stats(photo, t)["shadow_depth"] > 0.5
    ignore = M.boxes_to_cell_mask([(0.0, 0.0, 0.34, 0.34)], 24, 15)
    assert M.lighting_stats(photo, t, ignore=ignore)["shadow_depth"] < 0.05


def test_lit_curve_shapes():
    xs, ys = M.lit_curve(np.full(300, 200.0), np.full(300, 0.9))
    assert len(xs) == 1 and ys[0] == pytest.approx(0.9)            # one brightness class
    t = np.concatenate([np.full(150, 60.0), np.full(150, 220.0)])
    r = np.concatenate([np.full(150, 0.5), np.full(150, 1.0)])
    xs, ys = M.lit_curve(t, r)
    assert np.all(np.diff(xs) > 0) and np.interp(60, xs, ys) == pytest.approx(0.5) and np.interp(220, xs, ys) == pytest.approx(1.0)
    xs, ys = M.lit_curve(t, np.concatenate([np.full(150, 0.05), np.full(150, 1.0)]))
    assert np.interp(60, xs, ys) == pytest.approx(0.45)            # a deep shadow on one class is not explained away


# ---------------------------------------------------------------------- colour
def test_colour_correction_removes_the_colour_of_the_light():
    rng = np.random.default_rng(0)
    tpl = np.clip(rng.normal(200, 15, (300, 480, 3)), 0, 255).astype(np.uint8)
    warm = np.clip(tpl.astype(np.float32) * np.array([0.8, 1.0, 1.15], np.float32), 0, 255).astype(np.uint8)
    corrected = M.colour_correct(warm, M.colour_gains(warm, tpl))
    assert np.abs(corrected.astype(np.float32).mean(axis=(0, 1)) - tpl.mean(axis=(0, 1))).max() < 6.0


def test_skin_and_glare_masks():
    img = np.full((200, 300, 3), 235, np.uint8)
    img[50:150, 100:200] = (105, 150, 205)                         # BGR skin tone
    skin = M.skin_mask(img)
    assert skin[100, 150] and not skin[10, 10]
    finger = M.finger_mask(img, np.zeros((200, 300), bool))
    assert 0.1 < finger.mean() < 0.2
    assert M.finger_mask(img, np.ones((200, 300), bool)).sum() == 0          # the template has skin colour there itself
    img[0:40, 0:40] = 255
    sat = M.saturated_mask(img, 245, 45)
    assert sat[10, 10] and not sat[100, 150] and not sat[180, 280]


# ---------------------------------------------------------------------- geometry
def test_quad_ordering_and_measures():
    quad = G.order_quad([[300, 200], [10, 10], [300, 10], [10, 200]])
    assert quad.tolist() == [[10, 10], [300, 10], [300, 200], [10, 200]]
    assert G.quad_sides(quad) == pytest.approx((290, 190, 290, 190))
    assert G.polygon_area(quad) == pytest.approx(290 * 190)
    persp = M.quad_perspective(quad)
    assert persp["side_mismatch"] == pytest.approx(0.0) and persp["angle_dev_deg"] == pytest.approx(0.0, abs=1e-4)
    keystone = np.array([[40, 0], [260, 0], [300, 200], [0, 200]], np.float32)
    assert M.quad_perspective(keystone)["side_mismatch"] == pytest.approx(1 - 220 / 300, abs=1e-6)


def test_outside_fraction_and_clipping():
    inside = G.rect_quad(100, 80) + 10
    assert G.outside_fraction(inside, 200, 200) == pytest.approx(0.0)
    half = np.array([[150, 10], [250, 10], [250, 90], [150, 90]], np.float32)
    assert G.outside_fraction(half, 200, 200) == pytest.approx(0.5, abs=0.01)
    gone = half + 500
    assert G.outside_fraction(gone, 200, 200) == pytest.approx(1.0)
    assert G.polygon_area(G.clip_polygon(half, 200, 200)) == pytest.approx(50 * 80, rel=0.02)


def test_warp_round_trip_and_scales():
    img = cv2.cvtColor(np.clip(text_like(800, 600), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    quad = np.array([[100, 80], [700, 100], [680, 520], [120, 500]], np.float32)
    out, hm = G.warp_canonical(img, quad, (500, 350))
    assert out.shape == (350, 500, 3)
    corners = G.project_points(hm, quad)
    assert np.allclose(corners, [[0, 0], [499, 0], [499, 349], [0, 349]], atol=0.5)
    again = G.warp_homography(img, hm, (500, 350))
    assert np.abs(again.astype(np.float32) - out.astype(np.float32)).mean() < 2.0
    scales = M.local_scales(np.linalg.inv(hm), (500, 350))
    assert len(scales) == 5 and all(1.0 < s < 1.4 for s in scales)             # ~600 px across a 500 px canonical width


def test_document_outline_is_found_in_a_photo():
    from gate_block.config import GateConfig
    frame = np.full((900, 1200, 3), (60, 90, 120), np.uint8)
    quad = np.array([[250, 200], [950, 230], [930, 680], [240, 650]], np.int32)
    cv2.fillConvexPoly(frame, quad, (235, 235, 235))
    cv2.putText(frame, "DOCUMENT", (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (40, 40, 40), 4)
    cands = G.find_candidates(frame, [1.55, 0.72], GateConfig.from_dict(None).recognition)
    assert cands
    best = max(G.quad_iou(c.quad, quad.astype(np.float32), frame.shape[:2]) for c in cands)
    assert best > 0.9


def test_a_cut_outline_is_extended_to_the_known_shape():
    # a 1.5 : 1 card of 600 x 400 whose right 120 px are outside a 1000-wide frame
    visible = np.array([[520, 200], [999, 200], [999, 600], [520, 600]], np.float32)
    full = G.extend_cut_quad(visible, 1.5, (800, 1000))
    assert full is not None and full[:, 0].max() == pytest.approx(520 + 600, abs=12)
    assert G.outside_fraction(full, 999, 799) == pytest.approx(120 / 600, abs=0.03)
    whole = np.array([[100, 100], [700, 100], [700, 500], [100, 500]], np.float32)
    unchanged = G.extend_cut_quad(whole, 1.5, (800, 1000))
    assert unchanged is None or np.allclose(unchanged, whole, atol=1.0)        # nothing touches the frame: nothing to extend
