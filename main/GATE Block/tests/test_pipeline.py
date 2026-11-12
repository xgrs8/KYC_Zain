"""The whole block, end to end: upload -> Gate 1 -> (ID merge) -> quality gate -> decision -> route."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gate_block import Gate, GateResult, StaticBackend, onehot, synth
from gate_block.config import GateConfig
from gate_block.messages import DEFAULT_MESSAGES_AR
from gate_block.types import DOCRES_TASKS, NEXT_BLOCKS

from .conftest import make_photo

ROOT = Path(__file__).resolve().parent.parent
ID_DIR = ROOT / "templates" / "national_id"
REFERENCE = ROOT / "tests" / "data" / "id_merge_reference.png"


def check_contract(res: GateResult) -> None:
    """Invariants every result must satisfy, whatever the route."""
    assert res.route in NEXT_BLOCKS and res.next_block == NEXT_BLOCKS[res.route]
    assert res.status_code.split(":")[0] == res.route
    assert res.stage in ("intake", "gate1", "quality")
    if res.route == "accept":
        assert res.teleocr_request and res.docres_request is None and res.message_ar is None and res.reason is None
        assert res.acceptable_as_is and res.stage == "quality"
    elif res.route == "enhance":
        assert res.operation in DOCRES_TASKS and res.docres_request["task"] == res.operation
        assert res.teleocr_request is None and res.message_ar is None
        assert res.docres_request["round"] == res.round_index + 1 and res.operation not in res.applied_ops
    elif res.route == "reject":
        assert res.reason in DEFAULT_MESSAGES_AR and res.message_ar == DEFAULT_MESSAGES_AR[res.reason]
        assert res.message_ar_with_side and res.teleocr_request is None and res.docres_request is None
    else:
        assert res.message_ar == DEFAULT_MESSAGES_AR["review"] and res.teleocr_request is None and res.docres_request is None
    text = json.dumps(res.to_dict(), ensure_ascii=False)                       # plain JSON, Arabic kept readable
    again = GateResult.from_dict(json.loads(text))
    assert again.to_dict() == res.to_dict()


# ---------------------------------------------------------------------- one photo
SCENARIOS = [
    ("tax_card/single", {}, "accept"),
    ("tax_card/single", dict(rot_deg=180.0), "accept"),
    ("tax_card/single", dict(persp=0.12), "accept"),
    ("tax_card/single", dict(gamma=0.8, contrast=1.2, saturation=1.35, flare=0.04), "accept"),
    ("tax_card/single", dict(blur_sigma=2.0), "accept"),                 # soft, but small next to the text
    ("tax_card/single", dict(blur_sigma=3.0), "enhance:deblurring"),
    ("tax_card/single", dict(blur_sigma=4.0), "enhance:deblurring"),
    ("tax_card/single", dict(blur_sigma=6.0), "reject:blur"),
    ("tax_card/single", dict(exposure=0.6), "enhance:appearance"),
    ("tax_card/single", dict(exposure=0.3), "reject:dark"),
    ("tax_card/single", dict(shadow=0.55), "enhance:deshadowing"),
    ("tax_card/single", dict(glare=0.9), "reject:glare"),
    ("tax_card/single", dict(cut=0.10), "reject:cut_off"),
    ("tax_card/single", dict(doc_frac=0.2), "reject:too_small"),          # print ~6 px tall
    ("tax_card/single", dict(finger=True), "reject:obstructed"),
    ("tax_card/single", dict(finger=True, finger_xy=(-0.40, 0.36)), "accept"),      # a thumb on the hologram corner
    ("business_license/single", {}, "accept"),
    ("business_license/single", dict(curl=0.05), "enhance:dewarping"),
    ("business_license/single", dict(curl=0.10), "enhance:dewarping"),   # not "obstructed": the bend moved the print
    ("business_license/single", dict(blur_sigma=6.0), "reject:blur"),
    ("national_id/front", {}, "accept"),
    ("national_id/front", dict(blur_sigma=5.0), "enhance:deblurring"),   # the ID's text is larger: more blur is needed
    ("national_id/front", dict(blur_sigma=9.0), "reject:blur"),
    ("national_id/back", dict(shadow=0.55), "enhance:deshadowing"),
    ("national_id/back", dict(cut=0.10), "reject:cut_off"),
]


@pytest.mark.parametrize("key, params, expected", SCENARIOS, ids=[f"{k.split('/')[0]}-{'-'.join(p) or 'clean'}" for k, p, _ in SCENARIOS])
def test_single_photo_scenarios(gate, key, params, expected):
    res = gate.process_image(make_photo(gate, key, **params), expected=key.split("/")[0])
    assert res.status_code == expected, (res.status_code, res.decision and res.decision["trace"])
    assert res.document_type == key.split("/")[0]
    check_contract(res)


def test_the_blur_message_reaches_the_user(gate):
    res = gate.process_image(make_photo(gate, "tax_card/single", blur_sigma=6.0), expected="tax_card")
    assert res.message_ar == "الصورة مغوشة من فضلك ارجع التقط الصورة بدون تغويش" and res.next_block == "retake"


@pytest.mark.parametrize("kind", ["natural", "screenshot", "text_page", "other_card", "shapes", "stripes"])
def test_pictures_of_something_else_stop_at_gate_one(gate, kind):
    res = gate.process_image(synth.make_negative(kind, 5), expected="tax_card")
    assert res.status_code == "reject:no_document" and res.stage == "gate1", res.status_code
    assert res.decision is None or res.decision["laya_eligible"] is False         # never shown to Laya
    assert res.teleocr_request is None and res.docres_request is None
    check_contract(res)


@pytest.mark.parametrize("src, code", [
    (b"not an image at all", "unreadable_file"),
    (np.full((600, 800, 3), 255, np.uint8), "blank_image"),
    (np.zeros((600, 800, 3), np.uint8), "blank_image"),
    (cv2.resize(cv2.imread(str(ID_DIR / "front.jpg")), (160, 100)), "image_too_small"),
    ("/no/such/file.jpg", "unreadable_file"),
])
def test_unusable_uploads_stop_at_intake(gate, src, code):
    res = gate.process_image(src)
    assert res.status_code == f"reject:{code}" and res.stage == "intake"
    check_contract(res)


def test_the_upload_slot_is_enforced(gate):
    license_photo = make_photo(gate, "business_license/single")
    res = gate.process_image(license_photo, expected="tax_card")
    assert res.status_code == "reject:wrong_document" and res.stage == "gate1"
    assert gate.process_image(license_photo).status_code == "accept"              # no slot given: any supported document
    with pytest.raises(ValueError):
        gate.process_image(license_photo, expected="passport")
    check_contract(res)


def test_path_bytes_and_array_inputs_agree(gate, tmp_path):
    img = make_photo(gate, "tax_card/single", shadow=0.55)
    path = tmp_path / "upload.png"
    cv2.imwrite(str(path), img)
    by_path = gate.process_image(path, expected="tax_card")
    by_bytes = gate.process_image(path.read_bytes(), expected="tax_card", workdir=tmp_path / "work")
    by_array = gate.process_image(img, expected="tax_card")
    assert by_path.status_code == by_bytes.status_code == by_array.status_code == "enhance:deshadowing"
    assert by_path.image_path == str(path) and by_path.docres_request["image"] == str(path)
    assert Path(by_bytes.image_path).is_file() and by_array.image_path is None    # bytes are saved only when a workdir is given
    assert by_array.image is not None and by_array.image.shape == img.shape
    assert set(by_path.timings_ms) == {"gate1", "quality", "decision", "total"}


def test_laya_state_matches_the_report(gate):
    res = gate.process_image(make_photo(gate, "tax_card/single", shadow=0.55), expected="tax_card")
    assert res.laya_state["doc"] == "tax_card" and res.laya_state["shadow"] == res.report["quality_metrics"]["shadow"]["severity"]
    assert json.loads(res.laya_state_json) == res.laya_state
    assert len(res.laya_state_json) <= gate.cfg.laya.max_state_chars


# ---------------------------------------------------------------------- national ID: two uploads, one merged image
def id_pair(gate, front=None, back=None):
    return make_photo(gate, "national_id/front", **(front or {})), make_photo(gate, "national_id/back", **(back or {}))


def test_id_reference_pair_is_merged_like_alg_and_accepted(gate, tmp_path):
    res = gate.process_id(ID_DIR / "front.jpg", ID_DIR / "back.jpg", workdir=tmp_path)
    assert res.status_code == "accept" and res.kind == "merged_id" and res.document_type == "national_id"
    assert res.regions == {"front": [0, 1020], "back": [1044, 2048]}
    assert np.array_equal(cv2.imread(res.image_path), cv2.imread(str(REFERENCE)))  # the file handed to TeleOCR
    assert res.teleocr_request == {"image": res.image_path, "kind": "merged_id", "document_type": "national_id"}
    assert res.gate1["front"]["template_used"] == "national_id/front" and res.gate1["back"]["template_used"] == "national_id/back"
    check_contract(res)


def test_id_photos_are_merged_then_judged_as_one_document(gate):
    front, back = id_pair(gate)
    res = gate.process_id(front, back)
    assert res.status_code == "accept" and res.image.shape[1] == 1600
    f0, f1 = res.regions["front"]
    b0, b1 = res.regions["back"]
    assert f0 == 0 and b0 == f1 + gate.cfg.id_merge.gap_px and b1 == res.image.shape[0]
    assert res.image[f1:b0].min() == 255                                           # the white band
    check_contract(res)


def test_id_sides_uploaded_the_wrong_way_round_are_swapped(gate):
    front, back = id_pair(gate)
    res = gate.process_id(back, front)
    assert res.status_code == "accept" and res.gate1["swapped"] is True and any("swapped" in w for w in res.warnings)
    straight = gate.process_id(front, back)
    assert np.array_equal(res.image, straight.image)                               # the merge is front-on-top either way
    no_swap = GateConfig.load_dir(GateConfig.default_dir())
    no_swap.routing.auto_swap_sides = False
    res = Gate(config=no_swap).process_id(back, front)
    assert res.status_code == "reject:wrong_side"


def test_id_gate_one_names_the_upload_that_is_wrong(gate):
    front, back = id_pair(gate)
    tax = make_photo(gate, "tax_card/single")
    res = gate.process_id(tax, back)
    assert res.status_code == "reject:wrong_document" and res.affected_sides == ["front"] and res.stage == "gate1"
    assert res.message_ar_with_side != res.message_ar and res.message_ar in res.message_ar_with_side
    res = gate.process_id(front, front)
    assert res.status_code == "reject:wrong_side" and res.affected_sides == ["back"]
    res = gate.process_id(front, synth.make_negative("natural", 2))
    assert res.status_code == "reject:no_document" and res.affected_sides == ["back"]
    res = gate.process_id(b"junk", back)
    assert res.status_code == "reject:unreadable_file" and res.affected_sides == ["front"] and res.stage == "intake"
    check_contract(res)


def test_id_defect_on_one_side_is_attributed_to_that_side(gate):
    front, back = id_pair(gate, back=dict(shadow=0.55))
    res = gate.process_id(front, back)
    assert res.status_code == "enhance:deshadowing" and res.affected_sides == ["back"]
    assert res.docres_request["regions"] == {"back": res.regions["back"]}          # DocRes works on that half only
    assert res.report["quality_by_side"]["front"]["shadow"]["severity"] == "none"
    check_contract(res)
    front, back = id_pair(gate, front=dict(cut=0.10))
    res = gate.process_id(front, back)
    assert res.status_code == "reject:cut_off" and res.affected_sides == ["front"]
    assert res.message_ar_with_side != res.message_ar
    front, back = id_pair(gate, front=dict(blur_sigma=6.0), back=dict(blur_sigma=6.0))
    res = gate.process_id(front, back)
    assert res.status_code == "reject:blur" and res.message_ar_with_side == res.message_ar   # both sides: no side named


def test_small_id_uploads_do_not_gain_detail_from_the_merge(gate):
    """The merge enlarges small uploads to the common width; that must not hide that they hold few pixels."""
    front, back = id_pair(gate, front=dict(doc_frac=0.2), back=dict(doc_frac=0.2))
    merged = gate.process_id(front, back)
    alone = gate.process_image(front, expected="national_id")
    text_px = merged.report["quality_by_side"]["front"]["low_resolution"]["value"]
    assert text_px == pytest.approx(alone.report["quality_metrics"]["low_resolution"]["value"], rel=0.05)
    assert text_px < 9                                                       # measured in upload pixels, not merged ones
    front, back = id_pair(gate, front=dict(doc_frac=0.17), back=dict(doc_frac=0.17))
    assert gate.process_id(front, back).status_code == "reject:too_small"   # print under 7 px: too small to read


def test_already_merged_ids(gate, tmp_path):
    res = gate.process_merged_id(REFERENCE)
    assert res.status_code == "accept" and res.image_path == str(REFERENCE) and res.regions == {"front": [0, 1020], "back": [1044, 2048]}
    check_contract(res)
    merged = cv2.imread(str(REFERENCE))
    upside = np.vstack([merged[1044:], merged[1020:1044], merged[:1020]])           # back on top
    assert gate.process_merged_id(upside).status_code == "reject:wrong_side"
    assert gate.process_merged_id(ID_DIR / "front.jpg").status_code == "reject:unreadable_file"   # not a merged image
    assert gate.process_merged_id(b"junk").status_code == "reject:unreadable_file"


# ---------------------------------------------------------------------- a whole submission
def test_submission(gate, tmp_path):
    front, back = id_pair(gate)
    out = gate.process_submission({"id_front": front, "id_back": back, "tax_card": make_photo(gate, "tax_card/single"),
                                   "business_license": make_photo(gate, "business_license/single")}, workdir=tmp_path)
    assert out["overall"] == "accept" and out["next_block"] == "teleocr"
    assert {k: v.status_code for k, v in out["documents"].items()} == {"national_id": "accept", "tax_card": "accept", "business_license": "accept"}
    out = gate.process_submission({"id_front": front, "id_back": back, "tax_card": make_photo(gate, "tax_card/single", shadow=0.55)})
    assert out["overall"] == "enhance" and out["documents"]["tax_card"].status_code == "enhance:deshadowing"
    out = gate.process_submission({"tax_card": make_photo(gate, "business_license/single"), "id_front": front})
    assert out["overall"] == "reject"
    assert out["documents"]["tax_card"].status_code == "reject:wrong_document"
    assert out["documents"]["national_id"].status_code == "reject:no_document" and out["documents"]["national_id"].affected_sides == ["back"]
    with pytest.raises(ValueError):
        gate.process_submission({"passport": front})
    assert gate.process_submission({})["overall"] == "reject"                      # nothing uploaded never passes


# ---------------------------------------------------------------------- after DocRes: verify before OCR
def test_recheck_accepts_a_successful_enhancement(gate):
    before = gate.process_image(make_photo(gate, "tax_card/single", shadow=0.55), expected="tax_card")
    assert before.status_code == "enhance:deshadowing" and not before.acceptable_as_is
    after = gate.recheck(make_photo(gate, "tax_card/single"), before)              # what a good deshadowing returns
    assert after.status_code == "accept" and after.applied_ops == ["deshadowing"] and after.round_index == 1
    assert after.recheck["used"] == "enhanced" and after.recheck["improved"] is True and after.recheck["worsened"] == []
    check_contract(after)


def test_recheck_never_passes_an_enhancement_that_made_things_worse(gate, tmp_path):
    original = tmp_path / "original.png"
    cv2.imwrite(str(original), make_photo(gate, "tax_card/single", shadow=0.55))
    before = gate.process_image(original, expected="tax_card")
    after = gate.recheck(make_photo(gate, "tax_card/single", blur_sigma=6.0), before)      # DocRes smeared the image
    assert after.status_code == "reject:shadow" and after.recheck["used"] == "original" and "blur" in after.recheck["worsened"]
    assert after.image_path == str(original) and after.decision["decided_by"] == "recheck"
    check_contract(after)
    unrecognisable = gate.recheck(synth.make_negative("natural", 3), before)               # DocRes returned garbage
    assert unrecognisable.status_code == "reject:shadow" and "recognition" in unrecognisable.recheck["worsened"]
    review = GateConfig.load_dir(GateConfig.default_dir())
    review.routing.on_enhancement_failure = "review"
    g2 = Gate(config=review)
    after = g2.recheck(make_photo(gate, "tax_card/single", blur_sigma=6.0), g2.process_image(original, expected="tax_card"))
    assert after.status_code == "review:enhancement_not_enough"
    check_contract(after)


def test_recheck_keeps_an_acceptable_original_when_the_enhancement_fails(gate, tmp_path):
    original = tmp_path / "original.png"
    cv2.imwrite(str(original), make_photo(gate, "tax_card/single", blur_sigma=3.0))        # only slightly blurry
    before = gate.process_image(original, expected="tax_card")
    assert before.status_code == "enhance:deblurring" and before.acceptable_as_is
    for bad in (make_photo(gate, "tax_card/single", blur_sigma=6.0), synth.make_negative("screenshot", 2), b"junk",
                make_photo(gate, "tax_card/single", cut=0.10)):
        after = gate.recheck(bad, before)
        assert after.status_code == "accept" and after.recheck["used"] == "original", after.recheck
        assert after.teleocr_request["image"] == str(original)
        check_contract(after)


def test_recheck_runs_another_task_while_rounds_remain(gate):
    both = make_photo(gate, "tax_card/single", blur_sigma=3.8, shadow=0.55)        # both moderate or worse
    first = gate.process_image(both, expected="tax_card")
    assert first.route == "enhance"
    other = {"deblurring": dict(shadow=0.55), "deshadowing": dict(blur_sigma=3.8)}[first.operation]   # the first task worked
    second = gate.recheck(make_photo(gate, "tax_card/single", **other), first)
    assert second.route == "enhance" and second.operation != first.operation and second.applied_ops == [first.operation]
    assert second.docres_request["round"] == 2 and second.docres_request["applied_before"] == [first.operation]
    third = gate.recheck(make_photo(gate, "tax_card/single"), second)
    assert third.status_code == "accept" and third.applied_ops == [first.operation, second.operation]
    # ... and when the second task does nothing, there is no third round
    stuck = gate.recheck(make_photo(gate, "tax_card/single", **other), second)
    assert stuck.route == "reject" and stuck.round_index == 2
    check_contract(stuck)


def test_recheck_when_docres_changes_nothing(gate):
    img = make_photo(gate, "tax_card/single", shadow=0.55)
    before = gate.process_image(img, expected="tax_card")
    after = gate.recheck(img, before)
    assert after.status_code == "reject:shadow" and after.recheck["improved"] is False and after.recheck["used"] == "enhanced"


def test_recheck_of_a_merged_id_keeps_the_original_file(gate, tmp_path):
    front, back = id_pair(gate, back=dict(shadow=0.55))
    before = gate.process_id(front, back, workdir=tmp_path)
    original_pixels = cv2.imread(before.image_path)
    good_front, good_back = id_pair(gate)
    enhanced = gate.process_id(good_front, good_back).image                        # same size, shadow gone
    assert enhanced.shape == original_pixels.shape
    stored = GateResult.from_dict(json.loads(json.dumps(before.to_dict())))        # as another process would reload it
    after = gate.recheck(enhanced, stored, workdir=tmp_path)
    assert after.status_code == "accept" and after.kind == "merged_id" and after.regions == before.regions
    assert after.image_path != before.image_path
    assert np.array_equal(cv2.imread(before.image_path), original_pixels)          # the original was not overwritten
    check_contract(after)


def test_recheck_needs_an_enhance_decision(gate):
    accepted = gate.process_image(make_photo(gate, "tax_card/single"), expected="tax_card")
    with pytest.raises(ValueError):
        gate.recheck(make_photo(gate, "tax_card/single"), accepted)


# ---------------------------------------------------------------------- the gate as a whole is not degenerate
def test_the_gate_uses_every_route_and_laya_cannot_collapse_it(gate):
    photos = [make_photo(gate, "tax_card/single", **p) for p in ({}, dict(blur_sigma=3.0), dict(blur_sigma=6.0), dict(shadow=0.55),
                                                                dict(cut=0.10), dict(exposure=0.6))]
    photos.append(synth.make_negative("screenshot", 1))
    expected = ["accept", "enhance", "reject", "enhance", "reject", "enhance", "reject"]
    assert [gate.process_image(p, expected="tax_card").route for p in photos] == expected
    for stuck in ("accept", "enhance", "reject"):
        g = Gate(backend=StaticBackend(onehot(stuck, reason="glare", task="binarization", confidence=0.99)))
        routes = [g.process_image(p, expected="tax_card").route for p in photos]
        assert routes[0] == "accept" and routes[2] == routes[4] == routes[6] == "reject", (stuck, routes)
        assert len(set(routes)) >= 2, (stuck, routes)


def test_config_directory_and_templates_directory_can_be_given(tmp_path):
    import shutil
    shutil.copytree(ROOT / "config", tmp_path / "cfg")
    (tmp_path / "cfg" / "rules.yaml").write_text((ROOT / "config" / "rules.yaml").read_text(encoding="utf-8")
                                                 .replace("enhance_at: mild", "enhance_at: moderate"), encoding="utf-8")
    g = Gate(config_dir=tmp_path / "cfg", templates_dir=ROOT / "templates")
    assert g.cfg.rules.enhance_at == "moderate"
    assert g.process_image(make_photo(g, "tax_card/single", blur_sigma=3.0), expected="tax_card").status_code == "accept"
    with pytest.raises(FileNotFoundError):
        Gate(config_dir=tmp_path / "missing")
