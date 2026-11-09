"""Shared fixtures for the MCP server tests (run from this folder: python -m pytest tests_mcp)."""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

MCP_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parents[2]
for root in (MCP_ROOT, APP_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from alg_mcp.config import Config          # noqa: E402
from alg_mcp.service import Service         # noqa: E402
from rfalib import mrz as RM                # noqa: E402

ID_VALUES = {
    "national_id_number": "198812345678", "name": "أحمد", "father_name": "علي", "grandfather_name": "حسين",
    "surname": "الجبوري", "mother_name": "فاطمة", "maternal_grandfather_name": "كاظم", "sex": "ذكر",
    "blood_group": "O+", "card_serial": "AB1234567", "issuing_authority": "مديرية الأحوال المدنية",
    "issue_date": "2021/05/17", "expiry_date": "2031/05/16", "place_of_birth": "بغداد",
    "birth_date": "1988/03/14", "family_number": "1234L5M12345678901",
}
(ID_VALUES["mrz_line1"], ID_VALUES["mrz_line2"], ID_VALUES["mrz_line3"]) = RM.build_td1(
    ID_VALUES["card_serial"], ID_VALUES["national_id_number"], ID_VALUES["birth_date"], ID_VALUES["sex"],
    ID_VALUES["expiry_date"], ID_VALUES["surname"], ID_VALUES["name"], ID_VALUES["father_name"],
    ID_VALUES["grandfather_name"])

TAX_VALUES = {"taxpayer_number": "101234567890", "taxpayer_name": "شركة الفرات للتجارة العامة",
              "economic_activity": "تجارة الجملة والتجزئة", "issue_date": "2022/04/12", "expiry_date": "2027/04/11"}
LIC_VALUES = {"license_number": "123456", "issue_date": "2023/06/15", "company_name": "شركة النور الحديثة للتجارة العامة",
              "company_type": "شركة ذات مسؤولية محدودة", "capital": "10,000,000 دينار عراقي",
              "business_purposes": "تجارة المواد الغذائية والمنظفات", "head_office": "بغداد - الكرادة - شارع فلسطين",
              "valid_until": "2028/06/14"}


def make_image(path: Path, size=(1000, 630), color=(200, 210, 230), label="doc") -> Path:
    im = Image.new("RGB", size, color)
    ImageDraw.Draw(im).text((20, 20), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


@pytest.fixture
def ws(tmp_path) -> Path:
    p = tmp_path / "ws"
    p.mkdir()
    return p


@pytest.fixture
def inbox(tmp_path) -> Path:
    d = tmp_path / "inbox"
    make_image(d / "ID_01_front.png", label="id front")
    make_image(d / "ID_01_back.png", label="id back")
    make_image(d / "ID_02_front.png", color=(220, 200, 200), label="id2 front")
    make_image(d / "ID_02_back.png", color=(220, 200, 200), label="id2 back")
    make_image(d / "TAX_03.png", size=(800, 560), label="tax")
    make_image(d / "LIC_04.png", size=(800, 1100), label="lic")
    # reference ("truth") values of the generator for ID_01, TAX_03 and LIC_04
    for stem, vals in (("ID_01", ID_VALUES), ("TAX_03", TAX_VALUES), ("LIC_04", LIC_VALUES)):
        (d / f"{stem}.truth.json").write_text(json.dumps(
            {"printed": vals, "intended": vals, "defects": [], "split": "train", "synthetic": True},
            ensure_ascii=False), encoding="utf-8")
    return d


def new_service(ws: Path, **kw) -> Service:
    return Service(Config.from_env(workspace=ws, **kw))


@pytest.fixture
def svc(ws) -> Service:
    return new_service(ws)


@pytest.fixture
def loaded(svc, inbox) -> Service:
    res = svc.add_paths([str(inbox)])
    assert res["added"] == 4 and not res["problems"], res
    return svc


def view_all(svc: Service, sid: str, agent: str, sides=("front",)) -> None:
    for side in sides:
        svc.view_sample(sid, side, agent=agent)


def sid_of(svc: Service, name: str) -> str:
    return svc._find(svc.store.snapshot(), name)["sid"]          # noqa: SLF001
