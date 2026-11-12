"""Printed truth for the photographed print run of the synthetic KYC dataset.

Where the documents come from
    The print sheets ``printables/{national_id,tax_card,business_license}_print_sheets.pdf`` were made by
    the "Z.ai Synthetic KYC Dataset Builder" (ReportLab, 31 Oct 2026) from the rows of
    ``printables/ground_truth/*.csv``: 90 fictional identities, each with a national ID (front and back,
    no portrait), a tax card and a company licence. Every document on the sheets is the render kept in
    ``Datasets/<split>/business_NNNN/<kind>/print_001/`` (same pixels, downscaled), the sheets were printed
    and cut out, and the photos of the cut-out documents are the labelling data.

Why a CSV row is not a label
    The generator did not print every value where the CSV says, left template placeholders on the cards
    and used values no Iraqi document carries. A label must be what is PRINTED (that is what OCR has to
    read), so this module turns one CSV row into

    printed           the text printed in each A.L.G field position: what the label must say
    intended          what the generator meant, for the fields whose print differs (kept for the later
                      cross-document checks; never used as a label)
    defects           "<field>: <what is wrong>" for every way the print departs from the CSV or from a
                      real Iraqi document
    notes             facts that are not generation errors but matter downstream (expired documents,
                      design elements next to a value)
    known_deviations  per field, the format this print run really uses, so A.L.G's format warnings and
                      MRZ cross-checks do not push a labeller to "correct" printed text

The mapping below was checked against all 90 printed documents of each kind
(``tools/verify_printed_truth.py`` reads every field of the print sheets and compares).
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Any

from .constants import SPLITS

# ---------------------------------------------------------------- the print run
GENERATOR = "Z.ai Synthetic KYC Dataset Builder (ReportLab print sheets, 2026-10-31)"
PRINT_RUN_DATE = _dt.date(2026, 10, 31)
CSV_FILES = {"ID": "national_id.csv", "TAX": "tax_card.csv", "LIC": "business_license.csv"}
SHEET_FILES = {"ID": "national_id_print_sheets.pdf", "TAX": "tax_card_print_sheets.pdf",
               "LIC": "business_license_print_sheets.pdf"}
# the generator's own name of each document file (also printed as a caption above it on the sheet)
CAPTION_PREFIX = {"ID": "ID", "TAX": "TAX", "LIC": "BIZ"}
# Datasets/<split>/business_NNNN/<kind>/
KIND_DIRS = {"ID": "id_card", "TAX": "tax_card", "LIC": "business_license"}
MANIFEST_SPLITS = {"train": "train", "validation": "val", "val": "val", "test": "test"}

PLACEHOLDER_SURNAME = "أي نص"          # "any text": the template placeholder left in the surname row
GRANDFATHER_PREFIX = "أي "              # the first word of that placeholder, left before the grandfather
VERTICAL_SERIAL = "000000"              # placeholder printed vertically on the right edge of every front
SYRIAN_POUND = "ل.س"

# Formats this print run really uses (pattern-level only: they never reveal a value of one card).
KNOWN_DEVIATIONS: dict[str, dict[str, dict[str, Any]]] = {
    "ID": {
        "national_id_number": {
            "pattern": r"^\d{13}$",
            "why": "the print run has 13-digit national numbers (some start with 0); Iraqi ones have 12"},
        "family_number": {
            "pattern": r"^\d{4}[A-Z]\d[A-Z]\d{11}$",
            "why": "the print run uses any two letters in the family number, not L and M"},
        "sex": {
            "pattern": r"^(ذكر|أنثى|انثى)$",
            "why": "female cards print انثى without the hamza"},
        "surname": {
            "skip_checks": ["names_vs_mrz"],
            "why": "the surname row holds a template placeholder, so it cannot match MRZ line 3"},
        "grandfather_name": {
            "skip_checks": ["names_vs_mrz"],
            "why": "a placeholder word is printed before the grandfather's name"},
        "mrz_line2": {
            "skip_checks": ["composite_check_digit"],
            "why": "the composite check digit of this print run does not follow ICAO 9303"},
    },
    "TAX": {},
    "LIC": {
        "capital": {
            "pattern": r"^\d{1,3}(,\d{3})* ل\.س$",
            "why": "the print run states the capital in Syrian pounds (ل.س)"},
        "business_purposes": {
            "pattern": r"^[؀-ۿ ]+ - [؀-ۿ0-9 ]+$",
            "why": "the purposes row (الغايات) holds the head-office address (city - street)"},
        "head_office": {
            "pattern": r"^[؀-ۿ ]{3,80}$",
            "why": "the head-office row (المركز الرئيسي) holds the founders' names"},
    },
}


class PrintedTruthError(ValueError):
    """A CSV row that does not fit the print run (missing column, unknown identity...)."""


# ---------------------------------------------------------------- loading
def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


def load_ground_truth(printables_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    """{doc_type: {identity_id: row}} from ``printables/ground_truth``; identities.csv adds the columns
    the per-document files lack (the company's activity for the licence notes)."""
    gt = Path(printables_dir) / "ground_truth"
    if not gt.is_dir():
        raise PrintedTruthError(f"ground_truth folder not found: {gt}")
    merged: dict[str, dict[str, str]] = {}
    ident = gt / "identities.csv"
    if ident.exists():
        merged = {r["identity_id"]: r for r in read_csv(ident)}
    out: dict[str, dict[str, dict[str, str]]] = {}
    for doc_type, name in CSV_FILES.items():
        rows = {}
        for r in read_csv(gt / name):
            rid = r.get("identity_id", "")
            if not rid:
                raise PrintedTruthError(f"{name}: a row without identity_id")
            rows[rid] = {**merged.get(rid, {}), **r}
        out[doc_type] = rows
    return out


def load_manifest(datasets_dir: Path) -> dict[str, dict[str, str]]:
    """{identity_id: {"folder", "split" (A.L.G spelling), "seed"}} from ``split_manifest.csv``."""
    out = {}
    for r in read_csv(Path(datasets_dir) / "split_manifest.csv"):
        split = MANIFEST_SPLITS.get(r.get("split", "").lower())
        if split not in SPLITS:
            raise PrintedTruthError(f"split_manifest.csv: unknown split {r.get('split')!r}")
        out[r["identity_id"]] = {"folder": r["identity_folder"], "split": split, "seed": r.get("split_seed", "")}
    return out


# ---------------------------------------------------------------- one row -> printed / intended / defects
def _need(row: dict[str, str], *keys: str) -> None:
    missing = [k for k in keys if not row.get(k)]
    if missing:
        raise PrintedTruthError(f"identity {row.get('identity_id')}: missing {missing}")


def _date(text: str) -> _dt.date | None:
    try:
        y, m, d = (int(p) for p in text.split("/"))
        return _dt.date(y, m, d)
    except (ValueError, TypeError):
        return None


def _expired_note(field: str, text: str) -> list[str]:
    d = _date(text)
    if d is not None and d < PRINT_RUN_DATE:
        return [f"{field}: expired on {text}, before the print run ({PRINT_RUN_DATE:%Y/%m/%d})"]
    return []


def _id_truth(row: dict[str, str]) -> dict[str, Any]:
    _need(row, "national_number_front", "first", "father", "grandfather", "mother", "surname", "sex", "blood",
          "doc_number", "issue_office", "issue_date", "expiry_date", "pob", "dob", "national_number_back",
          "mrz_line1", "mrz_line2", "mrz_line3")
    printed = {
        "national_id_number": row["national_number_front"],
        "name": row["first"],
        "father_name": row["father"],
        "grandfather_name": GRANDFATHER_PREFIX + row["grandfather"],
        "surname": PLACEHOLDER_SURNAME,
        "mother_name": row["mother"],
        "maternal_grandfather_name": row["surname"],
        "sex": row["sex"],
        "blood_group": row["blood"],
        "card_serial": row["doc_number"],
        "issuing_authority": row["issue_office"],
        "issue_date": row["issue_date"],
        "expiry_date": row["expiry_date"],
        "place_of_birth": row["pob"],
        "birth_date": row["dob"],
        "family_number": row["national_number_back"],
        "mrz_line1": row["mrz_line1"],
        "mrz_line2": row["mrz_line2"],
        "mrz_line3": row["mrz_line3"],
        # sidecar-only (not in the label JSON)
        "vertical_serial": VERTICAL_SERIAL,
    }
    intended = {"grandfather_name": row["grandfather"], "surname": row["surname"]}
    nn = row["national_number_front"]
    defects = [f"national_id_number: {len(nn)} digits are printed; Iraqi national ID numbers have 12"
               + (" (and this one starts with 0)" if nn.startswith("0") else "")]
    defects += [
        f"grandfather_name: the placeholder word '{GRANDFATHER_PREFIX.strip()}' is printed before the "
        f"grandfather's name ({row['grandfather']})",
        f"surname: the template placeholder '{PLACEHOLDER_SURNAME}' is printed instead of the surname "
        f"({row['surname']}), which is printed one row lower",
        "maternal_grandfather_name: the surname is printed in this row; the mother's father's name was "
        "never generated",
        "family_number: does not follow A.L.G's family-number format (4 digits, L, digit, M, 11 digits)",
        "mrz_line3: carries the intended surname and grandfather, so it differs from the printed name rows",
        f"vertical_serial: the placeholder '{VERTICAL_SERIAL}' is printed on every card",
    ]
    if row["sex"] == "انثى":
        defects.append("sex: 'انثى' is printed without the hamza (A.L.G expects 'أنثى')")
    from .mrz import check_digit  # local: keep import time low for the CLI
    l1, l2 = row["mrz_line1"], row["mrz_line2"]
    if len(l1) == 30 and len(l2) == 30:
        composite = check_digit(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29])
        if composite != l2[29]:
            defects.append(f"mrz_line2: the composite check digit is {l2[29]}; ICAO 9303 gives {composite} "
                           "(this print run does not compute it)")
    notes = ["card_serial: a letter 'A' of the card design is printed before the number on every card; it is "
             "not part of the document number (MRZ line 1 carries the number without it)"]
    notes += _expired_note("expiry_date", row["expiry_date"])
    return {"printed": printed, "intended": intended, "defects": defects, "notes": notes}


def _tax_truth(row: dict[str, str]) -> dict[str, Any]:
    _need(row, "tax_number", "company_name", "activity", "tax_issue", "tax_expiry")
    printed = {
        "taxpayer_number": row["tax_number"],
        "taxpayer_name": row["company_name"],
        "economic_activity": row["activity"],
        "issue_date": row["tax_issue"],
        "expiry_date": row["tax_expiry"],
    }
    return {"printed": printed, "intended": {}, "defects": [],
            "notes": _expired_note("expiry_date", row["tax_expiry"])}


def _lic_truth(row: dict[str, str]) -> dict[str, Any]:
    _need(row, "biz_license_number", "biz_date", "company_name", "biz_type", "capital", "hq", "founders",
          "biz_expiry")
    printed = {
        "license_number": row["biz_license_number"],
        "issue_date": row["biz_date"],
        "company_name": row["company_name"],
        "company_type": row["biz_type"],
        "capital": row["capital"],
        "business_purposes": row["hq"],
        "head_office": row["founders"],
        "valid_until": row["biz_expiry"],
    }
    intended = {"head_office": row["hq"], "founders": row["founders"]}
    city = row["hq"].split(" - ")[0].strip()
    activity = row.get("activity", "")
    defects = []
    if row["capital"].endswith(SYRIAN_POUND):
        defects.append(f"capital: the currency is printed as '{SYRIAN_POUND}' (Syrian pound); an Iraqi licence "
                       "states 'دينار عراقي'")
    defects += [
        "business_purposes: the head-office address is printed in the purposes row (الغايات); the purposes "
        "were never generated" + (f" (the company's activity on its tax card: {activity})" if activity else ""),
        "head_office: the founders' names are printed in the head-office row (المركز الرئيسي)",
        f"business_purposes: the printed address is in Syria ({city}), not in Iraq, and has two parts "
        "(city - street) where A.L.G expects three",
    ]
    return {"printed": printed, "intended": intended, "defects": defects,
            "notes": _expired_note("valid_until", row["biz_expiry"])}


_BUILDERS = {"ID": _id_truth, "TAX": _tax_truth, "LIC": _lic_truth}


def truth_for(doc_type: str, row: dict[str, str]) -> dict[str, Any]:
    """printed / intended / defects / notes / known_deviations for one CSV row of one document type."""
    if doc_type not in _BUILDERS:
        raise PrintedTruthError(f"unknown doc type {doc_type!r}")
    out = _BUILDERS[doc_type](row)
    out["known_deviations"] = {k: dict(v) for k, v in KNOWN_DEVIATIONS[doc_type].items()}
    return out


def caption(doc_type: str, identity_id: str) -> str:
    """The generator's file stem of a document, printed above it on the sheet (ID_001, TAX_001, BIZ_001)."""
    return f"{CAPTION_PREFIX[doc_type]}_{identity_id}"
