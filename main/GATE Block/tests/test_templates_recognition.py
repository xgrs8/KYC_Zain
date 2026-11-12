"""Templates and Gate 1 (is this photo one of the supported documents?)."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from gate_block import synth
from gate_block.config import GateConfig
from gate_block.imageio import load_image
from gate_block.recognition import recognize, transfer_recognition
from gate_block.templates import TemplateError, TemplatePack, read_layout

from .conftest import TEMPLATE_KEYS

ROOT = Path(__file__).resolve().parent.parent
CFG = GateConfig.from_dict(None)

# the field names A.L.G writes for each document (Software/A.L.G/rfalib/schema.py); TeleOCR is trained on them
ALG_FIELDS = {
    "national_id": {"national_id_number", "name", "father_name", "grandfather_name", "surname", "mother_name",
                    "maternal_grandfather_name", "sex", "blood_group", "card_serial", "issuing_authority", "issue_date",
                    "expiry_date", "place_of_birth", "birth_date", "family_number", "mrz_line1", "mrz_line2", "mrz_line3"},
    "tax_card": {"taxpayer_number", "taxpayer_name", "economic_activity", "issue_date", "expiry_date"},
    "business_license": {"license_number", "issue_date", "company_name", "company_type", "capital", "business_purposes",
                         "head_office", "valid_until"},
}


# ---------------------------------------------------------------------- templates
def test_the_pack_holds_the_four_reference_sides(pack):
    assert sorted(pack.keys()) == sorted(TEMPLATE_KEYS)
    assert pack.get("national_id", "front").rigid and pack.get("tax_card", "single").rigid
    assert not pack.get("business_license", "single").rigid                      # paper: may curl
    for t in pack.templates:
        assert max(t.size) == CFG.image.canonical_long_side
        assert len(t.kp_pts) >= 300, t.key                                       # enough static detail to match on
        assert t.fields and any(f.critical for f in t.fields.values()), t.key


def test_field_names_match_alg_schema(pack):
    for doc_type, expected in ALG_FIELDS.items():
        names = set()
        for t in pack.templates:
            if t.doc_type == doc_type:
                names |= set(t.fields)
        assert names == expected, doc_type


def test_field_boxes_are_inside_the_document_and_do_not_overlap(pack):
    for t in pack.templates:
        boxes = [(n, f.box) for n, f in t.fields.items()]
        for name, (x0, y0, x1, y1) in boxes:
            assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, (t.key, name)
        for i, (n1, a) in enumerate(boxes):
            for n2, b in boxes[i + 1:]:
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                assert ix <= 1e-9 or iy <= 1e-9, (t.key, n1, n2)


def test_every_field_box_holds_text_in_the_reference(pack):
    """A box drawn in the wrong place would sit on empty background: it must contain dark strokes."""
    for t in pack.templates:
        gray = np.clip(t.gray, 0, 255).astype(np.uint8)
        for name in t.fields:
            x0, y0, x1, y1 = t.field_box_px(name, 0.0)
            patch = gray[y0:y1, x0:x1]
            assert float(np.percentile(patch, 95)) - float(np.percentile(patch, 2)) > 40, (t.key, name)


def _copy_templates(tmp_path) -> Path:
    dst = tmp_path / "templates"
    shutil.copytree(ROOT / "templates", dst)
    return dst


def test_layout_mistakes_are_reported_with_their_place(tmp_path):
    def broken(edit):
        d = _copy_templates(tmp_path / str(len(list(tmp_path.iterdir()))))
        path = d / "tax_card" / "layout.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        return d

    def set_box(v):
        return lambda data: data["sides"]["single"]["fields"]["taxpayer_number"].__setitem__("box", v)

    for edit, needle in ((set_box([0.5, 0.5, 0.4, 0.6]), "taxpayer_number"), (set_box([0.1, 0.2, 0.3]), "taxpayer_number"),
                         (set_box([0.1, 0.2, 1.3, 0.4]), "taxpayer_number"), (set_box("left"), "taxpayer_number"),
                         (lambda d: d["sides"].__setitem__("rear", {}), "rear"),
                         (lambda d: d["sides"].__setitem__("back", {"fields": {}}), "back"),          # no back image
                         (lambda d: d.__setitem__("rigid", "yes"), "rigid"),
                         (lambda d: d["sides"]["single"]["fields"]["taxpayer_number"].__setitem__("critical", "yes"), "critical")):
        with pytest.raises(TemplateError, match=needle):
            TemplatePack.load(broken(edit), CFG)


def test_missing_or_ambiguous_templates_are_errors(tmp_path):
    with pytest.raises(TemplateError):
        TemplatePack.load(tmp_path / "nope", CFG)
    (tmp_path / "empty").mkdir()
    with pytest.raises(TemplateError):
        TemplatePack.load(tmp_path / "empty", CFG)
    d = _copy_templates(tmp_path)
    shutil.copy(d / "tax_card" / "single.jpg", d / "tax_card" / "single.png")       # two images for one side
    with pytest.raises(TemplateError, match="more than one"):
        TemplatePack.load(d, CFG)
    (d / "tax_card" / "single.png").unlink()
    (d / "tax_card" / "layout.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TemplateError):
        read_layout(d / "tax_card" / "layout.json")


def test_a_template_without_layout_still_recognises(tmp_path, gate):
    d = _copy_templates(tmp_path)
    (d / "tax_card" / "layout.json").unlink()
    pack = TemplatePack.load(d, CFG)
    t = pack.get("tax_card", "single")
    assert t.fields == {} and t.rigid
    rec = recognize(load_image(gate.pack.get("tax_card", "single").bgr, CFG.image), pack, CFG)
    assert rec.status == "recognized" and rec.key == "tax_card/single"


# ---------------------------------------------------------------------- Gate 1
@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_reference_images_are_recognised_as_themselves(gate, key):
    doc_type, side = key.split("/")
    rec = recognize(load_image(gate.pack.get(doc_type, side).bgr, CFG.image), gate.pack, CFG)
    assert rec.status == "recognized" and rec.key == key and rec.confidence >= 0.8


CONDITIONS = {
    "clean": {}, "tilted": dict(rot_deg=8.0, persp=0.04), "upside_down": dict(rot_deg=180.0), "blurry": dict(blur_sigma=3.0),
    "dim": dict(exposure=0.6), "far": dict(doc_frac=0.3), "shadow": dict(shadow=0.55), "angle": dict(persp=0.12),
}


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_photos_of_supported_documents_are_recognised(measured, key, condition):
    rec, sq = measured(key, **CONDITIONS[condition])
    assert rec.status == "recognized" and rec.key == key, (rec.status, rec.key, rec.evidence)
    assert sq.reference == "template"


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_other_backgrounds_and_other_people(gate, key):
    from .conftest import make_photo
    for seed, bg in ((11, "dark"), (12, "plain"), (13, "cloth"), (14, "clutter")):
        loaded = load_image(make_photo(gate, key, seed=seed, bg=bg, rot_deg=-6.0), CFG.image)
        rec = recognize(loaded, gate.pack, CFG)
        assert rec.status == "recognized" and rec.key == key, (bg, rec.status, rec.key)


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_a_very_blurry_document_is_not_called_unsupported(measured, key):
    """A failed match is not proof that the photo shows something else (the design brief, 3.2)."""
    rec, _ = measured(key, blur_sigma=5.0)
    assert rec.status != "unsupported"
    if rec.status == "recognized":
        assert rec.key == key


@pytest.mark.parametrize("key", TEMPLATE_KEYS)
def test_sideways_photos_are_never_mistaken_for_another_document(measured, key):
    rec, _ = measured(key, rot_deg=90.0)
    assert rec.status != "unsupported"
    if rec.status == "recognized":
        assert rec.key == key


@pytest.mark.parametrize("kind", synth.NEGATIVE_KINDS)
def test_other_pictures_are_never_recognised(gate, kind):
    for seed in (1, 2):
        img = synth.make_negative(kind, seed)
        try:
            loaded = load_image(img, CFG.image)
        except Exception:
            continue                                            # blank pictures are stopped at intake
        rec = recognize(loaded, gate.pack, CFG)
        assert rec.status != "recognized", (kind, seed, rec.key, rec.confidence)
        assert rec.confidence < CFG.recognition.recognized_score


def test_recognition_survives_the_merge_resize(gate):
    front = load_image(ROOT / "templates" / "national_id" / "front.jpg", CFG.image)
    rec = recognize(front, gate.pack, CFG)
    bigger = load_image(cv2.resize(front.work, (1600, 1020), interpolation=cv2.INTER_LANCZOS4), CFG.image)
    moved = transfer_recognition(rec, front, bigger)
    assert moved.status == rec.status and moved.key == rec.key == "national_id/front"
    ratio = bigger.small.shape[1] / front.small.shape[1]
    assert np.allclose(moved.best.quad_small, (rec.best.quad_small + 0.5) * ratio - 0.5, atol=1.0)
