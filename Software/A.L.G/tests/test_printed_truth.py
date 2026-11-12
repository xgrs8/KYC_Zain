"""The photographed print run: printed truth, declared deviations, and preparing photos for labeling."""
import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from alg.label import LabelSession
from rfalib import printed_truth as PT
from rfalib import sidecar as SC
from rfalib.schema import field_warnings, label_keys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import prepare_photos as PP  # noqa: E402

# two rows of printables/ground_truth/identities.csv (fictional people of the print run)
ROW_001 = {
    "identity_id": "001", "first": "ديار", "father": "آرام", "grandfather": "بلال", "mother": "مريم",
    "surname": "الربيعي", "sex": "ذكر", "blood": "A+", "dob": "1973/01/31", "pob": "كركوك",
    "issue_office": "صلاح الدين", "issue_date": "2020/05/03", "expiry_date": "2030/05/03",
    "doc_number": "DV8752051", "national_number_front": "9836146454363",
    "national_number_back": "5200N1W26815209887", "mrz_line1": "IDIRQDV87520514<<<<<<<<<<<<<<<",
    "mrz_line2": "7301315M3005039IRQ<<<<<<<<<<<3", "mrz_line3": "ALRUBAIE<<DIYAR<ARAM<BILAL<<<<",
    "company_name": "شركة الامل لمواد البناء", "activity": "تجارة مواد البناء", "tax_number": "104618586000",
    "tax_issue": "2019/06/25", "tax_expiry": "2024/06/24", "biz_license_number": "907083",
    "biz_date": "2018/10/12", "biz_expiry": "2023/10/11", "biz_type": "شركة تضامنية",
    "capital": "567,000,000 ل.س", "hq": "حمص - شارع ستار جاد", "founders": "ديار الربيعي، هيمين الساعدي",
}
ROW_024 = {
    **ROW_001, "identity_id": "024", "first": "سارة", "father": "خالد", "grandfather": "حسين", "mother": "سندس",
    "surname": "الحسناوي", "sex": "انثى", "blood": "B+", "dob": "1976/01/07", "pob": "الرمادي",
    "issue_office": "واسط", "issue_date": "2019/08/14", "expiry_date": "2029/08/14", "doc_number": "FW0738366",
    "national_number_front": "1435996418931", "national_number_back": "2371X1Y70944335929",
    "mrz_line1": "IDIRQFW07383662<<<<<<<<<<<<<<<", "mrz_line2": "7601071F2908144IRQ<<<<<<<<<<<5",
    "mrz_line3": "ALHASNAWI<<SARAH<KHALID<HUSSEI",
}


# ------------------------------------------------------------------ printed truth
def test_id_is_labelled_by_printed_position():
    t = PT.truth_for("ID", ROW_001)
    p = t["printed"]
    assert p["grandfather_name"] == "أي بلال"                 # placeholder word before the grandfather
    assert p["surname"] == "أي نص"                            # placeholder instead of the surname
    assert p["maternal_grandfather_name"] == "الربيعي"        # the surname, one row lower
    assert p["national_id_number"] == "9836146454363" and p["card_serial"] == "DV8752051"
    assert p["vertical_serial"] == "000000"
    assert set(label_keys("ID")) <= set(p)                     # every label field has a printed value
    assert t["intended"] == {"grandfather_name": "بلال", "surname": "الربيعي"}
    text = "\n".join(t["defects"])
    assert "13 digits" in text and "composite check digit is 3" in text
    assert any(n.startswith("card_serial:") for n in t["notes"])
    assert not any(n.startswith("expiry_date:") for n in t["notes"])          # valid until 2030


def test_female_card_and_expired_documents_are_declared():
    t = PT.truth_for("ID", ROW_024)
    assert t["printed"]["sex"] == "انثى"
    assert any(d.startswith("sex:") for d in t["defects"])
    assert any("starts with 0" in d for d in PT.truth_for("ID", {**ROW_001, "national_number_front": "0" * 13})[
        "defects"])
    tax = PT.truth_for("TAX", ROW_001)
    assert tax["printed"]["taxpayer_name"] == "شركة الامل لمواد البناء" and tax["defects"] == []
    assert tax["notes"] == ["expiry_date: expired on 2024/06/24, before the print run (2026/10/31)"]


def test_licence_rows_are_shifted_and_the_currency_is_syrian():
    t = PT.truth_for("LIC", ROW_001)
    p = t["printed"]
    assert p["business_purposes"] == "حمص - شارع ستار جاد"     # the address, in the purposes row
    assert p["head_office"] == "ديار الربيعي، هيمين الساعدي"   # the founders, in the head-office row
    assert p["capital"] == "567,000,000 ل.س"
    assert t["intended"]["head_office"] == "حمص - شارع ستار جاد"
    text = "\n".join(t["defects"])
    assert "Syrian pound" in text and "(حمص)" in text and "تجارة مواد البناء" in text
    assert set(label_keys("LIC")) == set(p)


def test_missing_column_is_an_error():
    with pytest.raises(PT.PrintedTruthError, match="missing"):
        PT.truth_for("TAX", {"identity_id": "001", "tax_number": "104618586000"})


# ------------------------------------------------------------------ declared deviations
def test_printed_values_pass_the_format_check_only_with_the_declared_deviations():
    for doc_type in ("ID", "TAX", "LIC"):
        for row in (ROW_001, ROW_024):
            t = PT.truth_for(doc_type, row)
            assert field_warnings(doc_type, t["printed"], t["known_deviations"]) == {}, (doc_type, row["identity_id"])
    t = PT.truth_for("ID", ROW_001)
    plain = field_warnings("ID", t["printed"])
    assert {"national_id_number", "family_number"} <= set(plain)          # the schema alone flags them
    bad = field_warnings("ID", {"national_id_number": "98361464543"}, t["known_deviations"])
    assert "known to print" in bad["national_id_number"]                 # matches neither format


def test_label_session_uses_the_deviations_and_records_them(tmp_path):
    d = tmp_path / "inbox"
    d.mkdir()
    for side in ("front", "back"):
        Image.new("RGB", (900, 570), (200, 210, 220)).save(d / f"ID_101_glare_{side}.png")
    t = PT.truth_for("ID", ROW_001)
    SC.write_json(d / "ID_101.truth.json", SC.truth_sidecar(
        doc_id="ID_101", doc_type="ID", case_id="business_0001", batch="print_001", seed="2026",
        doc_code="ID_001", split="train", printed=t["printed"], intended=t["intended"], defects=t["defects"])
        | {"known_deviations": t["known_deviations"]})
    sess = LabelSession("ID", "ID_101_glare", str(d / "ID_101_glare_front.png"), str(d / "ID_101_glare_back.png"),
                        tmp_path / "ws")
    for key in sess.fields:
        assert sess.use_truth(key), key
    assert sess.warnings() == {}
    out = sess.generate()
    assert out["base"] == "ID_101"
    label = json.loads(Path(out["label"]).read_text("utf-8"))
    assert label["fields"]["surname"] == "أي نص" and "vertical_serial" not in label["fields"]
    side = json.loads(Path(out["sidecar"]).read_text("utf-8"))
    assert "national_id_number" in side["known_deviations"] and side["truth_disagreements"] == []


# ------------------------------------------------------------------ preparing the photos
def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    gt = root / "printables" / "ground_truth"
    gt.mkdir(parents=True)
    cols_id = ["identity_id", "first", "father", "grandfather", "mother", "surname", "sex", "blood",
               "national_number_front", "doc_number", "issue_office", "issue_date", "expiry_date", "pob", "dob",
               "national_number_back", "mrz_line1", "mrz_line2", "mrz_line3"]
    cols = {"national_id.csv": cols_id,
            "tax_card.csv": ["identity_id", "tax_number", "company_name", "activity", "tax_issue", "tax_expiry"],
            "business_license.csv": ["identity_id", "biz_license_number", "biz_date", "company_name", "biz_type",
                                     "capital", "hq", "founders", "biz_expiry"],
            "identities.csv": list(ROW_001)}
    for name, keys in cols.items():
        with open(gt / name, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows([ROW_001, ROW_024])
    ds = root / "Datasets"
    ds.mkdir()
    (ds / "split_manifest.csv").write_text(
        '"identity_id","identity_folder","split","split_seed"\n"001","business_0001","train","2026"\n'
        '"024","business_0024","validation","2026"\n', encoding="utf-8")

    def photo(rel: str, color=(200, 200, 200)) -> None:
        p = ds / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 400), color).save(p)

    photo("train/business_0001/id_card/print_001/glare/IMG_1_front.jpg")
    photo("train/business_0001/id_card/print_001/glare/IMG_2_back.jpg")
    photo("train/business_0001/tax_card/print_001/good/WhatsApp Image 1.jpeg")
    photo("train/business_0001/tax_card/print_001/blur/WhatsApp Image 2.jpeg")
    photo("validation/business_0024/business_license/print_001/keyboard/x.jpg")
    photo("validation/business_0024/id_card/print_001/table/a.jpg", (10, 10, 10))
    photo("validation/business_0024/id_card/print_001/table/b.jpg", (250, 250, 250))
    (ds / "train/business_0001/id_card/print_001/front.jpg").write_bytes(b"render, not a photo")
    return root


def test_prepare_names_links_and_truth(tmp_path):
    root = _dataset(tmp_path)
    inbox = tmp_path / "inbox"
    res = PP.prepare(root, inbox, detect=None, log=lambda s: None)
    assert sorted(res["added"]) == ["ID_101", "LIC_2401", "TAX_101", "TAX_102"]
    assert len(res["problems"]) == 2                       # business_0024's ID: sides unknown, no gate given
    assert all("cannot tell front from back" in p for p in res["problems"])
    names = sorted(p.name for p in inbox.iterdir())
    assert "ID_101_glare_front.jpg" in names and "ID_101_glare_back.jpg" in names
    assert "TAX_101_blur.jpeg" in names and "LIC_2401_keyboard.jpg" in names
    # A.L.G finds the truth of every photo through its name
    truth = SC.read_json(SC.find_truth_sidecar(inbox / "ID_101_glare_back.jpg"))
    assert truth["printed"]["surname"] == "أي نص" and truth["split"] == "train" and truth["doc_code"] == "ID_001"
    assert truth["capture"]["condition"] == "glare" and truth["capture"]["side_detection"] == "name"
    assert truth["capture"]["photos"]["front"].endswith("glare/IMG_1_front.jpg")
    lic = SC.read_json(inbox / "LIC_2401.truth.json")
    assert lic["split"] == "val" and lic["printed"]["capital"] == "567,000,000 ل.س"
    assert lic["source_of_truth"].startswith("printables/ground_truth/business_license.csv")
    # a rerun prepares nothing twice
    again = PP.prepare(root, inbox, detect=None, log=lambda s: None)
    assert again["added"] == []


def test_prepare_tells_id_sides_with_a_detector(tmp_path):
    root = _dataset(tmp_path)
    inbox = tmp_path / "inbox"

    def detect(path: Path):                                      # dark photo = front, here
        return ("front" if Image.open(path).getpixel((0, 0))[0] < 128 else None), 0.9

    res = PP.prepare(root, inbox, detect=detect, log=lambda s: None)
    assert "ID_2401" in res["added"] and res["problems"] == []
    truth = SC.read_json(inbox / "ID_2401.truth.json")
    assert truth["capture"]["photos"]["front"].endswith("table/a.jpg")
    assert truth["capture"]["photos"]["back"].endswith("table/b.jpg")
    assert "the other photo is the front" in truth["capture"]["side_detection"]


def test_prepare_reports_unpaired_sides():
    pairs, problems = PP.pair_id_photos([Path("x/one_front.jpg"), Path("x/two_front.jpg")], None)
    assert pairs == [] and "2 front and 0 back" in problems[0]


def test_prepare_converts_formats_alg_cannot_open(tmp_path):
    root = _dataset(tmp_path)
    folder = root / "Datasets/train/business_0001/tax_card/print_001/glare"
    folder.mkdir(parents=True)
    Image.new("RGB", (640, 400), (90, 90, 90)).save(folder / "photo.webp")
    (folder / "broken.heic").write_bytes(b"not really an image")
    inbox = tmp_path / "inbox"
    res = PP.prepare(root, inbox, detect=None, log=lambda s: None)
    names = sorted(p.name for p in inbox.iterdir())
    converted = [n for n in names if n.endswith("_glare.jpg")]
    assert len(converted) == 1 and res["placed"]["convert"] == 1
    with Image.open(inbox / converted[0]) as im:
        assert im.format == "JPEG"
    assert any("broken.heic" in p and "not placed" in p for p in res["problems"])
    assert not any(n.endswith(".heic") for n in names)
    # the failed photo left no truth sidecar behind and its number is free again
    bases = {n.split("_glare")[0] for n in converted}
    assert all((inbox / f"{b}.truth.json").exists() for b in bases)
