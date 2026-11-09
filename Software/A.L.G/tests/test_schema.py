"""T-SCHEMA / T-NORM for the A.L.G supporting library:
the exact 6.4 contract, [UNREADABLE] token, UTF-8 round-trip, and the
6.7 normalization vectors (incl. علي != على)."""
from __future__ import annotations

import json

import pytest

from rfalib import sidecar as SC
from rfalib.constants import UNREADABLE
from rfalib.normalize import normalize, same
from rfalib.schema import build_label, label_keys, validate_label

TAX_EXAMPLE = {
    "photo": "TAX_01.png",
    "fields": {
        "taxpayer_number": "101234567890",
        "taxpayer_name": "شركة الفرات للتجارة العامة",
        "economic_activity": "تجارة الجملة والتجزئة",
        "issue_date": "2022/04/12", "expiry_date": "2027/04/11",
    },
}


def test_tax_example_exact():
    assert tuple(TAX_EXAMPLE["fields"].keys()) == label_keys("TAX")
    assert validate_label(TAX_EXAMPLE, "TAX") == []


def test_key_counts():
    assert (len(label_keys("ID")), len(label_keys("TAX")),
            len(label_keys("LIC"))) == (19, 5, 8)


def test_build_label_order_and_photo():
    label = build_label("TAX", "TAX_02.png", TAX_EXAMPLE["fields"])
    assert list(label.keys()) == ["photo", "fields"]
    assert label["photo"] == "TAX_02.png"   # photo = the real saved file name
    assert list(label["fields"].keys()) == list(label_keys("TAX"))


def test_unreadable_and_junk():
    fields = dict(TAX_EXAMPLE["fields"])
    fields["taxpayer_name"] = UNREADABLE
    assert build_label("TAX", "TAX_01.png", fields)["fields"]["taxpayer_name"] == UNREADABLE
    fields["taxpayer_name"] = "شركة " + UNREADABLE
    with pytest.raises(ValueError):
        build_label("TAX", "TAX_01.png", fields)


def test_utf8_roundtrip(tmp_path):
    p = tmp_path / "TAX_01.label.json"
    SC.write_json(p, TAX_EXAMPLE)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")      # no BOM
    assert b"\xd8\xb4" in raw                        # unescaped Arabic (ش)
    assert json.loads(raw.decode("utf-8")) == TAX_EXAMPLE
    p2 = tmp_path / "again.json"
    SC.write_json(p2, json.loads(raw.decode("utf-8")))
    assert p2.read_bytes() == raw                    # byte-exact round trip


def test_normalize_vectors():
    assert normalize("مُحَمَّد") == "محمد"
    assert normalize("عـرض") == "عرض"
    assert normalize("علي\u200b") == "علي"
    assert normalize("  بغداد   الكرادة ") == "بغداد الكرادة"
    assert normalize("علي") != normalize("على")
    assert normalize("اسد") != normalize("أسد")
    assert same("علي ", "علي")
    assert not same("علي", "على")


def test_find_truth_sidecar(tmp_path):
    (tmp_path / "ID_07_front.png").write_bytes(b"")
    (tmp_path / "ID_07_back.png").write_bytes(b"")
    SC.write_json(tmp_path / "ID_07.truth.json", {"printed": {"name": "أحمد"}})
    assert SC.find_truth_sidecar(tmp_path / "ID_07_front.png") is not None
    assert SC.find_truth_sidecar(tmp_path / "ID_07_back.png").name == "ID_07.truth.json"
    assert SC.find_truth_sidecar(tmp_path / "photo.jpg") is None
