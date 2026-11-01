"""T-LABEL (plan 11): the label gate - empty field blocks Generate;
[UNREADABLE] accepted; mixed junk rejected; truth policy (D10) and the
exact label JSON output format (D12)."""
from __future__ import annotations

import json

import pytest
from PIL import Image

from alg.label import LabelSession
from rfalib import sidecar as SC
from rfalib.constants import UNREADABLE
from rfalib.schema import label_keys


@pytest.fixture
def images(tmp_path):
    front = tmp_path / "ID_01_front.png"
    back = tmp_path / "ID_01_back.png"
    Image.new("RGB", (1140, 723), (210, 215, 222)).save(front)
    Image.new("RGB", (1140, 723), (215, 218, 224)).save(back)
    tax = tmp_path / "photo_tax.png"
    Image.new("RGB", (395, 512), (198, 214, 232)).save(tax)
    return {"front": front, "back": back, "tax": tax}


def _session(doc_type, images, workspace, name=None) -> LabelSession:
    if doc_type == "ID":
        return LabelSession("ID", name or "ID_01", str(images["front"]),
                            str(images["back"]), workspace, labeler="tester")
    return LabelSession("TAX", name or "photo_tax", str(images["tax"]),
                        None, workspace, labeler="tester")


# ------------------------------------------------------------ the gate

def test_empty_field_blocks_generate(tmp_path, images):
    session = _session("TAX", images, tmp_path)
    assert session.resolution() == (0, 5)
    problems = session.gate_problems()
    assert set(problems) == set(label_keys("TAX"))
    with pytest.raises(ValueError):
        session.generate()


def test_unreadable_accepted(tmp_path, images):
    session = _session("TAX", images, tmp_path)
    for key in label_keys("TAX"):
        session.set_value(key, "قيمة")
    session.mark_unreadable("economic_activity")
    resolved, total = session.resolution()
    assert (resolved, total) == (5, 5)
    result = session.generate()
    assert result["base"] == "TAX_01"
    label = json.loads(open(result["label"], encoding="utf-8").read())
    assert label["fields"]["economic_activity"] == UNREADABLE
    assert label["photo"] == "TAX_01.png"


def test_counter_reset_skips_existing_label_names_without_overwriting(
        tmp_path, images):
    from rfalib import naming

    labeled = tmp_path / "labeled"
    labeled.mkdir()
    old_label = labeled / "TAX_01.label.json"
    old_sidecar = labeled / "TAX_02.label.sidecar.json"
    old_label.write_text("keep label", encoding="utf-8")
    old_sidecar.write_text("keep sidecar", encoding="utf-8")
    naming.reset_counters(tmp_path)

    session = _session("TAX", images, tmp_path)
    for key in label_keys("TAX"):
        session.set_value(key, "قيمة")
    result = session.generate()

    assert result["base"] == "TAX_03"
    assert old_label.read_text(encoding="utf-8") == "keep label"
    assert old_sidecar.read_text(encoding="utf-8") == "keep sidecar"
    assert (labeled / "TAX_03.label.json").exists()
    assert naming.peek_next(tmp_path, "TAX") == 4


def test_mixed_junk_rejected(tmp_path, images):
    session = _session("TAX", images, tmp_path)
    for key in label_keys("TAX"):
        session.set_value(key, "قيمة")
    session.fields["taxpayer_number"] = f"123 {UNREADABLE}"
    problems = session.gate_problems()
    assert problems["taxpayer_number"].startswith("Mixed junk")
    with pytest.raises(ValueError):
        session.generate()


# ------------------------------------------------------------ outputs (D12)

def test_generate_writes_exact_format(tmp_path, images):
    session = _session("ID", images, tmp_path)
    values = {
        "national_id_number": "123456789012", "name": "أحمد",
        "father_name": "محمد", "grandfather_name": "علي",
        "surname": "الربيعي", "mother_name": "فاطمة",
        "maternal_grandfather_name": "حسين", "sex": "ذكر",
        "blood_group": "O+", "card_serial": "AA1234567",
        "issuing_authority": "بغداد / الكرخ", "issue_date": "2023/06/15",
        "expiry_date": "2033/06/14", "place_of_birth": "بغداد",
        "birth_date": "1985/02/14", "family_number": "1234L5M09876543210",
        "mrz_line1": "IDIRQAA12345671123456789012<<<"[:30].ljust(30, "<"),
        "mrz_line2": "8502145M3306144IRQ<<<<<<<<<<<7"[:30],
        "mrz_line3": "ALRBYAY<<AHMD<MHMD<ALY".ljust(30, "<"),
    }
    for key, value in values.items():
        session.set_value(key, value)
    result = session.generate(merge_width=1140, merge_gap=24)
    label = json.loads((tmp_path / "labeled" / "ID_01.label.json")
                       .read_text(encoding="utf-8"))
    assert list(label.keys()) == ["photo", "fields"]
    assert list(label["fields"].keys()) == list(label_keys("ID"))
    assert label["photo"] == "ID_01.png"
    assert label["fields"]["name"] == "أحمد"
    # merged image written, exact geometry (D11)
    with Image.open(result["image"]) as im:
        assert im.format == "PNG"
        assert im.size == (1140, 723 + 24 + 723)
    # sidecar
    sidecar = json.loads((tmp_path / "labeled" / "ID_01.label.sidecar.json")
                         .read_text(encoding="utf-8"))
    assert sidecar["labeler"] == "tester"
    assert sidecar["field_status"]["name"] == "typed"
    assert sidecar["truth_ref"] is None


def test_skip_requires_reason(tmp_path, images):
    session = _session("TAX", images, tmp_path)
    with pytest.raises(ValueError):
        session.skip("   ")
    result = session.skip("image too dark")
    sidecar = json.loads(open(result["sidecar"], encoding="utf-8").read())
    assert sidecar["skipped"] == "image too dark"


# ------------------------------------------------------------ truth policy (D10)

def _truth_file(tmp_path, images):
    truth = SC.truth_sidecar(
        doc_id="ID_01", doc_type="ID", case_id=None, batch="2026-10-30_01",
        seed="abc", doc_code="SYN-123456", split="train",
        printed={"national_id_number": "123456789012", "name": "أحمد",
                 "issue_date": "2023/06/15"})
    path = tmp_path / "ID_01.truth.json"
    SC.write_json(path, truth)
    return path


def test_use_truth_suggestion_must_be_accepted(tmp_path, images):
    _truth_file(tmp_path, images)
    session = _session("ID", images, tmp_path)
    assert session.truth is not None
    # nothing auto-filled
    assert session.fields["name"] == ""
    assert session.use_truth("name")
    assert session.fields["name"] == "أحمد"
    assert session.status["name"] == "accepted-truth"
    assert session.use_truth("national_id_number")
    # unknown key returns False
    assert not session.use_truth("not_a_field")


def test_truth_disagreement_logged(tmp_path, images):
    _truth_file(tmp_path, images)
    session = _session("ID", images, tmp_path)
    session.use_truth("name")
    session.set_value("name", "محمد")          # edit after acceptance
    session.use_truth("national_id_number")    # kept as accepted
    session.use_truth("issue_date")
    for key in label_keys("ID"):
        if session.fields[key] in ("",):
            session.set_value(key, "قيمة")
    session.set_value("mrz_line1", "x" * 30)
    session.set_value("mrz_line2", "y" * 30)
    session.set_value("mrz_line3", "z" * 30)
    session.set_value("expiry_date", "2033/06/14")
    session.set_value("birth_date", "1985/02/14")
    session.generate()
    sidecar = json.loads((tmp_path / "labeled" / "ID_01.label.sidecar.json")
                         .read_text(encoding="utf-8"))
    assert sidecar["field_status"]["name"] == "edited"
    assert sidecar["field_status"]["national_id_number"] == "accepted-truth"
    disagreements = {d["field"] for d in sidecar["truth_disagreements"]}
    assert "name" in disagreements
    assert "national_id_number" not in disagreements


def test_external_image_no_truth(tmp_path, images):
    session = LabelSession("TAX", "scan001", str(images["tax"]), None, tmp_path)
    assert session.truth is None
    assert session.truth_ref is None
    assert session.truth_values() == {}
