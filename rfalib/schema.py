"""The three label JSON schemas - the user's exact format (plan 6.4).

Keys and their order are FIXED (protocol rule 4): never add or rename keys.
Every field value is a non-empty string: a real transcription or exactly
"[UNREADABLE]".  The `photo` value always equals the real saved merged file
name and extension (e.g. "ID_01.png").

Field metadata (validators, RTL, Arabic card labels) is also used by both
UIs to build entry rows and by A.L.G for warn-level validation (8.4).
"""
from __future__ import annotations

import re
from typing import Any

from .constants import DOC_TYPES, UNREADABLE

# Arabic strings allowed: letters (incl.Presentation Forms-B after shaping
# input is raw, so base block), spaces, and the common punctuation used on
# the cards.
_AR = r"[\u0600-\u06FF\u0750-\u077F ][^<>{}\[\]]*"

SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "ID": [
        # key, arabic label, rtl, validator regex
        dict(key="national_id_number", ar="الرقم الوطني", rtl=False,
             pattern=r"^[1-9]\d{11}$"),
        dict(key="name", ar="الاسم", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,25}$"),
        dict(key="father_name", ar="اسم الأب", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,25}$"),
        dict(key="grandfather_name", ar="اسم الجد", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,25}$"),
        dict(key="surname", ar="اللقب / النسب", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,25}$"),
        dict(key="mother_name", ar="اسم الأم", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,25}$"),
        dict(key="maternal_grandfather_name", ar="اسم جد الأم", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,25}$"),
        dict(key="sex", ar="الجنس", rtl=True, pattern=r"^(ذكر|أنثى)$"),
        dict(key="blood_group", ar="زمرة الدم", rtl=False, pattern=r"^(O|A|B|AB)[+-]$"),
        dict(key="card_serial", ar="الرقم التسلسلي", rtl=False, pattern=r"^[A-Z]{2}\d{7}$"),
        dict(key="issuing_authority", ar="جهة الإصدار", rtl=True,
             pattern=r"^[\u0600-\u06FF /()\-]{3,40}$"),
        dict(key="issue_date", ar="تاريخ الإصدار", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
        dict(key="expiry_date", ar="تاريخ النفاذ", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
        dict(key="place_of_birth", ar="محل الولادة", rtl=True, pattern=r"^[\u0600-\u06FF ]{2,30}$"),
        dict(key="birth_date", ar="تاريخ الولادة", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
        dict(key="family_number", ar="الرقم العائلي", rtl=False, pattern=r"^\d{4}L\dM\d{11}$"),
        dict(key="mrz_line1", ar="منطقة القراءة الآلية ١", rtl=False, pattern=r"^[A-Z0-9<]{30}$"),
        dict(key="mrz_line2", ar="منطقة القراءة الآلية ٢", rtl=False, pattern=r"^[A-Z0-9<]{30}$"),
        dict(key="mrz_line3", ar="منطقة القراءة الآلية ٣", rtl=False, pattern=r"^[A-Z0-9<]{30}$"),
    ],
    "TAX": [
        dict(key="taxpayer_number", ar="رقم البطاقة الضريبية", rtl=False, pattern=r"^10\d{10}$"),
        dict(key="taxpayer_name", ar="اسم المكلف", rtl=True, pattern=r"^[\u0600-\u06FF ]{3,60}$"),
        dict(key="economic_activity", ar="النشاط الاقتصادي", rtl=True, pattern=r"^[\u0600-\u06FF ]{3,60}$"),
        dict(key="issue_date", ar="تاريخ الإصدار", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
        dict(key="expiry_date", ar="تاريخ الانتهاء", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
    ],
    "LIC": [
        dict(key="license_number", ar="رقم الرخصة", rtl=False, pattern=r"^\d{6}$"),
        dict(key="issue_date", ar="التاريخ", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
        dict(key="company_name", ar="اسم الشركة", rtl=True, pattern=r"^[\u0600-\u06FF ]{3,60}$"),
        dict(key="company_type", ar="نوع الشركة", rtl=True, pattern=r"^[\u0600-\u06FF ]{3,60}$"),
        dict(key="capital", ar="رأس المال", rtl=True,
             pattern=r"^\d{1,3}(,\d{3})* دينار عراقي$"),
        dict(key="business_purposes", ar="النشاط", rtl=True, pattern=r"^[\u0600-\u06FF ]{3,60}$"),
        dict(key="head_office", ar="المقر الرئيسي", rtl=True,
             pattern=r"^[\u0600-\u06FF0-9 ]+( - [\u0600-\u06FF0-9 ]+){2}$"),
        dict(key="valid_until", ar="تاريخ الانتهاء", rtl=False, pattern=r"^\d{4}/\d{2}/\d{2}$"),
    ],
}

_LABEL_KEYS = {t: tuple(f["key"] for f in SCHEMAS[t]) for t in SCHEMAS}

# Fields that exist on the card and in the truth sidecar but NOT in the
# label JSON (plan 4.4 / OQ-5).
SIDECAR_ONLY_FIELDS = {
    "ID": [dict(key="rfid_access_number", pattern=r"^[A-Z0-9]{9}$"),
           dict(key="vertical_serial", pattern=r"^\d{7}$")],
}


def label_keys(doc_type: str) -> tuple[str, ...]:
    """Ordered label keys for a doc type (exact 6.4 contract)."""
    return _LABEL_KEYS[doc_type]


def fields_meta(doc_type: str) -> list[dict[str, Any]]:
    return SCHEMAS[doc_type]


def is_unreadable(value: str) -> bool:
    return value == UNREADABLE


def field_warnings(doc_type: str, fields: dict[str, str]) -> dict[str, str]:
    """Warn-level regex validation (8.4). Returns {key: warning}.

    A field that is exactly [UNREADABLE] never warns.  Pattern mismatches
    produce a human-readable warning; the labeler can override (logged).
    """
    warnings: dict[str, str] = {}
    for meta in SCHEMAS[doc_type]:
        value = (fields.get(meta["key"]) or "").strip()
        if not value or value == UNREADABLE:
            continue
        if not re.match(meta["pattern"], value):
            warnings[meta["key"]] = (
                f"Value does not match the expected format for "
                f"{meta['key']} (e.g. pattern {meta['pattern']}).")
    return warnings


def build_label(doc_type: str, photo: str, fields: dict[str, str]) -> dict[str, Any]:
    """Build the label JSON object with the exact key order of 6.4."""
    if doc_type not in DOC_TYPES:
        raise ValueError(f"unknown doc type {doc_type!r}")
    out_fields: dict[str, str] = {}
    for key in _LABEL_KEYS[doc_type]:
        value = str(fields[key]).strip()
        if not value:
            raise ValueError(f"field {key!r} is empty - resolve it first")
        if UNREADABLE in value and value != UNREADABLE:
            raise ValueError(f"field {key!r} mixes text and {UNREADABLE}")
        out_fields[key] = value
    return {"photo": photo, "fields": out_fields}


def validate_label(obj: Any, doc_type: str | None = None) -> list[str]:
    """Structural validation of a parsed label JSON; [] = valid."""
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["label must be a JSON object"]
    if set(obj.keys()) != {"photo", "fields"}:
        problems.append("top-level keys must be exactly {photo, fields}")
        return problems
    photo = obj["photo"]
    if not isinstance(photo, str) or not photo:
        problems.append("photo must be a non-empty string")
    if not isinstance(obj["fields"], dict):
        problems.append("fields must be an object")
        return problems
    guessed = doc_type or _guess_type_from_photo(photo)
    if guessed is None:
        problems.append("cannot determine doc type from photo name")
        return problems
    keys = tuple(obj["fields"].keys())
    if keys != _LABEL_KEYS[guessed]:
        problems.append(f"field keys/order wrong for {guessed}")
        return problems
    for key, value in obj["fields"].items():
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key}: empty value")
        elif UNREADABLE in value and value != UNREADABLE:
            problems.append(f"{key}: mixed junk around {UNREADABLE}")
    return problems


def _guess_type_from_photo(photo: str) -> str | None:
    p = photo.upper()
    for t in DOC_TYPES:
        if p.startswith(t + "_"):
            return t
    return None


def doc_type_of_photo(photo: str) -> str | None:
    return _guess_type_from_photo(photo)
