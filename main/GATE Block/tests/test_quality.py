"""The quality gate: it measures and reports, with one severity word per defect.

Photos are simulated from the real templates with a known defect, so the measured value can be compared
with what was put in. Expectations use wide margins: they must hold on any machine.
"""
import json
import math

import pytest

from gate_block.config import GateConfig
from gate_block.quality import info_score, sev_high, sev_low
from gate_block.report import DocumentAnalysis
from gate_block.types import DEFECTS, SEVERITIES, rank

from .conftest import TEMPLATE_KEYS

CFG = GateConfig.from_dict(None)


def sev(sq, defect):
    return sq.metrics[defect].severity


def value(sq, defect):
    return sq.metrics[defect].value


# ---------------------------------------------------------------------- severity words
def test_thresholds_are_inclusive_and_unknown_is_its_own_word():
    thr = (1.0, 2.0, 3.0)
    assert [sev_high(v, thr) for v in (0.99, 1.0, 1.99, 2.0, 2.99, 3.0, 99)] == ["none", "mild", "mild", "moderate", "moderate", "severe", "severe"]
    thr = (0.6, 0.4, 0.2)
    assert [sev_low(v, thr) for v in (0.61, 0.6, 0.41, 0.4, 0.21, 0.2, 0.0)] == ["none", "mild", "mild", "moderate", "moderate", "severe", "severe"]
    for bad in (None, float("nan"), float("inf")):
        assert sev_high(bad, thr) == "unknown" and sev_low(bad, thr) == "unknown" and info_score(bad, thr, True) is None
    assert info_score(0.0, (1.0, 2.0, 3.0), True) == 100.0 and info_score(99.0, (1.0, 2.0, 3.0), True) == 0.0
    assert 0.0 <= info_score(0.5, (0.6, 0.4, 0.2), False) <= 100.0


# ---------------------------------------------------------------------- a good photo measures clean
@pytest.mark.parametrize("key", TEMPLATE_KEYS)
@pytest.mark.parametrize("params", [{}, dict(rot_deg=180.0), dict(rot_deg=8.0, persp=0.04), dict(bg="dark"), dict(bg="plain")],
                         ids=["straight", "upside_down", "tilted", "dark_table", "plain_table"])
def test_good_photos_have_no_defect(measured, key, params):
    rec, sq = measured(key, **params)
    assert sq.reference == "template" and set(sq.metrics) == set(DEFECTS)
    for d in DEFECTS:
        allowed = ("none", "mild") if d == "perspective" else ("none",)
        assert sev(sq, d) in allowed, (d, value(sq, d), sev(sq, d))
    assert [n for n, f in sq.fields.items() if f.status != "good"] == []


PHONE_LOOKS = {
    "contrasty": dict(gamma=0.8, contrast=1.2, saturation=1.35, flare=0.04, tint=(0.92, 1.0, 1.06)),
    "flat": dict(gamma=1.25, contrast=0.85, flare=0.10),
    "lifted_midtones": dict(gamma=1.4),
    "darkened_midtones": dict(gamma=0.7),
    "vivid": dict(saturation=1.5, contrast=1.3),
}


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
@pytest.mark.parametrize("look", sorted(PHONE_LOOKS))
def test_a_cameras_tone_curve_is_not_a_defect(measured, key, look):
    """Phones render contrast, mid-tones and colour differently from the reference image. That must not be
    read as a shadow on the dark parts of the design, as darkness, or as anything else."""
    _, sq = measured(key, **PHONE_LOOKS[look])
    for d in ("shadow", "dark", "glare", "blur", "cut", "noise", "low_resolution"):
        assert sev(sq, d) == "none", (d, value(sq, d))
    assert all(f.status == "good" for f in sq.fields.values())


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_a_shadow_is_still_found_through_a_tone_curve(measured, key):
    _, sq = measured(key, shadow=0.45, gamma=0.8, contrast=1.2, saturation=1.35)
    assert rank(sev(sq, "shadow")) >= 2 and sev(sq, "dark") == "none"


def test_a_portrait_in_the_photo_slot_changes_nothing(gate):
    """The reference ID has an empty photo slot. A card with a portrait in it is the same document."""
    import cv2

    from gate_block import synth
    t = gate.pack.get("national_id", "front")
    card = synth.variant_from_template(t.bgr, {k: f.box for k, f in t.fields.items()}, 3)
    h, w = card.shape[:2]
    x0, y0, x1, y1 = int(0.05 * w), int(0.25 * h), int(0.33 * w), int(0.84 * h)
    card[y0:y1, x0:x1] = (205, 205, 210)                                                    # backdrop
    cx, cy = (x0 + x1) // 2, y0 + int(0.42 * (y1 - y0))
    cv2.ellipse(card, (cx, y1), (int(0.42 * (x1 - x0)), int(0.30 * (y1 - y0))), 0, 180, 360, (60, 50, 45), -1)      # jacket
    cv2.ellipse(card, (cx, cy - int(0.10 * (y1 - y0))), (int(0.25 * (x1 - x0)), int(0.24 * (y1 - y0))), 0, 0, 360, (30, 30, 35), -1)
    cv2.ellipse(card, (cx, cy), (int(0.19 * (x1 - x0)), int(0.22 * (y1 - y0))), 0, 0, 360, (120, 160, 215), -1)     # face
    for params, expected in (({}, "accept"), (dict(rot_deg=6.0, persp=0.03), "accept"), (dict(shadow=0.55), "enhance:deshadowing")):
        res = gate.process_image(synth.capture(card, synth.CaptureParams(seed=3, **params))[0], expected="national_id")
        assert res.status_code == expected, (params, res.status_code, res.report["quality_metrics"])
        assert res.report["quality_metrics"]["obstruction"]["severity"] == "none"


def test_light_colour_is_not_mistaken_for_a_finger_or_a_shadow(measured):
    for tint in ((0.85, 1.0, 1.12), (1.12, 1.0, 0.88)):                # warm lamp, cold daylight (B, G, R)
        _, sq = measured("tax_card/single", tint=tint)
        assert sev(sq, "obstruction") == "none" and sev(sq, "shadow") == "none" and sev(sq, "glare") == "none"
        assert all(f.status == "good" for f in sq.fields.values())


# ---------------------------------------------------------------------- each defect is measured, in its own metric
@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_blur_is_measured_and_judged_against_the_text_size(measured, pack, key):
    """The blur is measured in pixels (Gaussian sigma) and judged as a share of the height of the printed
    text: the same blur ruins small print and leaves large print readable."""
    text_h = pack.get(*key.split("/")).text_height
    readings = []
    for share, allowed in ((0.0, ("none",)), (0.08, ("none",)), (0.15, ("mild", "moderate")), (0.30, ("severe",))):
        applied = round(share * text_h, 2)
        rec, sq = measured(key, **({"blur_sigma": applied} if applied else {}))
        if rec.status != "recognized":
            assert share >= 0.3, (share, rec.status)        # only a very blurry photo may stop being recognised
            continue
        readings.append(value(sq, "blur"))
        assert sev(sq, "blur") in allowed, (applied, value(sq, "blur"), sev(sq, "blur"))
        if share == 0.15:
            assert sq.metrics["blur"].extra["sigma_px"] == pytest.approx(applied, rel=0.25)
            assert value(sq, "blur") == pytest.approx(share, abs=0.04)
        for other in ("dark", "shadow", "glare", "cut", "noise", "obstruction"):
            assert sev(sq, other) == "none", (applied, other)
    assert readings == sorted(readings)


def test_the_same_blur_is_worse_on_smaller_print(measured):
    _, licence = measured("business_license/single", blur_sigma=3.0)          # text ~15 px
    _, id_back = measured("national_id/back", blur_sigma=3.0)                  # text ~26 px
    assert value(licence, "blur") > 1.5 * value(id_back, "blur")
    assert rank(sev(licence, "blur")) > rank(sev(id_back, "blur"))


def test_a_small_but_sharp_photo_is_not_called_blurry(measured):
    """A distant photo is soft once enlarged; that is low resolution, not blur."""
    for key in TEMPLATE_KEYS:
        _, sq = measured(key, doc_frac=0.3)
        assert sev(sq, "blur") == "none", (key, value(sq, "blur"))


def test_motion_blur_counts_as_blur(measured):
    """A shake smears the strokes it crosses; the strokes along it stay crisp and must not hide it."""
    _, sharp = measured("tax_card/single")
    _, shaken = measured("tax_card/single", motion_len=15)
    assert rank(sev(shaken, "blur")) >= 2 and shaken.metrics["blur"].extra["motion_direction_deg"] == 0
    assert sharp.metrics["blur"].extra["motion_px"] < 6 and sev(sharp, "blur") == "none"
    _, id_back = measured("national_id/back", motion_len=21)                  # larger print, longer shake
    assert rank(sev(id_back, "blur")) >= 2
    _, slight = measured("national_id/back", motion_len=5)                    # a tremor the big print shrugs off
    assert rank(sev(slight, "blur")) <= 1


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_darkness_is_measured_relative_to_the_template(measured, key):
    _, dim = measured(key, exposure=0.6)
    _, dark = measured(key, exposure=0.3)
    assert value(dim, "dark") == pytest.approx(0.6, abs=0.06) and sev(dim, "dark") == "mild"
    assert value(dark, "dark") == pytest.approx(0.3, abs=0.05) and sev(dark, "dark") == "severe"
    assert sev(dim, "shadow") == "none" and sev(dim, "blur") == "none"          # evenly dim is not a shadow


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_a_shadow_is_a_shadow_not_darkness(measured, key):
    _, sq = measured(key, shadow=0.55)
    assert rank(sev(sq, "shadow")) >= 2 and value(sq, "shadow") == pytest.approx(0.55, abs=0.1)
    assert sev(sq, "dark") == "none" and sev(sq, "blur") == "none" and sev(sq, "glare") == "none"
    assert any("shadow" in f.causes for f in sq.fields.values())


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_glare_is_found_where_it_hides_content(measured, key):
    _, sq = measured(key, glare=0.9)
    assert rank(sev(sq, "glare")) >= 2
    assert any(f.causes.get("glare") in ("moderate", "severe") or f.causes.get("low_contrast") in ("moderate", "severe")
               for f in sq.fields.values())
    assert sev(sq, "blur") == "none" and sev(sq, "cut") == "none"


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_a_document_leaving_the_frame_is_cut(measured, key):
    _, sq = measured(key, cut=0.10)
    assert rank(sev(sq, "cut")) >= 2 and 0.05 < value(sq, "cut") < 0.15
    assert sq.geometry["frame"] == "cut"
    _, whole = measured(key)
    assert whole.geometry["frame"] == "complete" and value(whole, "cut") < 0.005


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_a_distant_document_has_too_few_pixels(measured, pack, key):
    """``low_resolution`` is the height of the smallest printed text in pixels of the photo."""
    _, near = measured(key)
    _, far = measured(key, doc_frac=0.2)
    assert sev(near, "low_resolution") == "none" and value(near, "low_resolution") >= 15
    ratio = value(far, "low_resolution") / value(near, "low_resolution")
    assert ratio == pytest.approx(far.metrics["low_resolution"].extra["min_scale"] / near.metrics["low_resolution"].extra["min_scale"], rel=0.02)
    assert 0.15 < ratio < 0.3                                                    # the photo shrank to about a quarter
    if pack.get(*key.split("/")).text_height < 22:                               # small print: too small to read now
        assert rank(sev(far, "low_resolution")) >= 2
    else:                                                                        # the ID's larger print is still legible
        assert rank(sev(far, "low_resolution")) <= 1


def test_noise_is_measured_above_the_documents_own_texture(measured):
    quiet = value(measured("tax_card/single")[1], "noise")
    some = value(measured("tax_card/single", noise=8.0)[1], "noise")
    heavy = measured("tax_card/single", noise=16.0)[1]
    assert quiet <= some <= value(heavy, "noise") and quiet < 1.0
    assert rank(sev(heavy, "noise")) >= 1


def test_camera_angle_is_reported_but_is_only_background(measured):
    _, sq = measured("tax_card/single", persp=0.12)
    assert rank(sev(sq, "perspective")) >= 1 and sev(sq, "blur") == "none"
    assert all(f.status == "good" for f in sq.fields.values())


def test_curl_is_measured_on_paper_only(measured):
    _, flat = measured("business_license/single")
    _, slight = measured("business_license/single", curl=0.05)       # bow of 5% of the height: reported, readable
    _, bowed = measured("business_license/single", curl=0.08)        # 8%: lines bend across fields, dewarp
    assert sev(flat, "curl") == "none" and rank(sev(slight, "curl")) >= 1 and rank(sev(bowed, "curl")) >= 2
    assert value(bowed, "curl") > value(slight, "curl") > value(flat, "curl")
    # a strong bend moves the print and shows the table: that is not a hand or an object on the paper
    _, strong = measured("business_license/single", curl=0.10)
    assert rank(sev(strong, "curl")) >= 2 and sev(strong, "obstruction") == "unknown"
    assert not [f for f in strong.fields.values() if f.cause == "obstructed"]
    _, card = measured("tax_card/single")
    assert sev(card, "curl") == "none"                                          # a rigid card cannot curl


# ---------------------------------------------------------------------- field level
def test_a_finger_is_reported_on_the_fields_it_covers(measured):
    _, sq = measured("tax_card/single", finger=True)
    covered = {n for n, f in sq.fields.items() if f.causes.get("obstructed") in ("moderate", "severe")}
    assert "expiry_date" in covered and "taxpayer_number" not in covered         # the finger is at the lower right
    assert sq.fields["taxpayer_number"].status == "good"
    assert rank(sev(sq, "obstruction")) >= 2
    assert sev(sq, "shadow") in ("none", "mild") and sev(sq, "dark") == "none"   # a finger is not a lighting problem


def test_a_finger_away_from_the_fields_harms_no_field(measured):
    _, sq = measured("tax_card/single", finger=True, finger_xy=(-0.40, 0.36))    # over the hologram corner
    assert rank(sev(sq, "obstruction")) >= 1
    assert all(f.causes.get("obstructed") in (None, "mild") for f in sq.fields.values())


def test_fields_outside_the_frame_are_reported_as_outside(measured):
    _, sq = measured("national_id/back", cut=0.10)
    outside = [n for n, f in sq.fields.items() if f.status == "outside"]
    assert outside and all(sq.fields[n].cause == "cut_off" for n in outside)


def test_glare_on_one_field_leaves_the_others_alone(measured):
    _, sq = measured("tax_card/single", glare=0.9)
    assert sq.fields["taxpayer_number"].causes.get("glare") in ("moderate", "severe")
    assert sq.fields["expiry_date"].status == "good"


def test_field_report_is_complete(measured):
    _, sq = measured("tax_card/single", blur_sigma=3.0)
    assert set(sq.fields) == {"taxpayer_number", "taxpayer_name", "economic_activity", "issue_date", "expiry_date"}
    for name, f in sq.fields.items():
        assert f.status in ("good", "fair", "poor", "outside", "unknown") and f.severity in SEVERITIES
        assert (f.cause is None) == (not f.causes)
        assert f.critical == (name in ("taxpayer_number", "taxpayer_name", "expiry_date"))


# ---------------------------------------------------------------------- no template alignment: unknown, never "fine"
def test_without_alignment_fields_are_unknown_not_missing(gate):
    from gate_block import synth
    from gate_block.imageio import load_image
    from gate_block.quality import analyze_side
    from gate_block.recognition import recognize
    loaded = load_image(synth.make_negative("other_card", 4), CFG.image)
    rec = recognize(loaded, gate.pack, CFG)
    assert rec.status != "recognized"
    sq = analyze_side(loaded, rec, gate.pack, CFG)
    assert sq.reference == "generic" and sq.fields == {} and sq.geometry["frame"] == "unknown"
    for d in ("cut", "glare", "shadow", "curl", "obstruction"):
        assert sev(sq, d) == "unknown" and value(sq, d) is None, d
    assert sq.warnings


# ---------------------------------------------------------------------- the JSON report
def test_report_keeps_values_scores_severities_and_unknowns_apart(measured):
    rec, sq = measured("tax_card/single", blur_sigma=3.0)
    report = DocumentAnalysis("tax_card", {"single": sq}, {"single": rec}).to_report()
    text = json.dumps(report, ensure_ascii=False)                                # plain JSON types only
    assert json.loads(text)["schema_version"] == 1
    assert set(report) == {"schema_version", "document_type", "document_recognition", "document_geometry", "quality_metrics",
                           "quality_by_side", "field_quality", "measurement_reference", "warnings"}
    recog = report["document_recognition"]["single"]
    assert recog["status"] == "recognized" and recog["template_used"] == "tax_card/single" and 0 <= recog["match_confidence"] <= 1
    blur = report["quality_metrics"]["blur"]
    assert set(blur) >= {"value", "severity", "quality_score", "reliable", "worst_side"}
    assert isinstance(blur["value"], float) and blur["severity"] in SEVERITIES and 0 <= blur["quality_score"] <= 100
    assert set(report["quality_metrics"]) == set(DEFECTS)
    geo = report["document_geometry"]["single"]
    assert geo["frame"] == "complete" and len(geo["corners_work_px"]) == 4 and math.isfinite(geo["rotation_deg"])
    field = report["field_quality"]["taxpayer_number"]
    assert set(field) == {"critical", "status", "severity", "cause", "causes", "measures", "side", "lost"}


def test_two_sided_report_names_the_side(gate):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "templates" / "national_id"
    res = gate.process_id(root / "front.jpg", root / "back.jpg")
    report = res.report
    assert set(report["quality_by_side"]) == {"front", "back"} == set(report["document_recognition"])
    assert "front.name" in report["field_quality"] and "back.mrz_line1" in report["field_quality"]
    assert len(report["field_quality"]) == 19
