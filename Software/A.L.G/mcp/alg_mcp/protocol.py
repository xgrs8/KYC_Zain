"""The labeling protocol the agents must follow, reason codes, and per-field format hints.

The same rules are enforced in code (service.py); this text only explains them.
"""
from __future__ import annotations

UNREADABLE = "[UNREADABLE]"

# reason codes for [UNREADABLE]; the agent must pick one (logged in the audit file)
REASONS: dict[str, str] = {
    "covered": "a finger, hand or object covers the field",
    "cut_off": "the field is partly or fully outside the photo",
    "blur": "out of focus or motion blur: at least one character is not certain",
    "glare": "a bright reflection hides at least one character",
    "low_light": "too dark or too noisy to be sure of every character",
    "too_small": "text too small / resolution too low to be sure of every character",
    "damaged": "fold, tear, stain, scratch or print defect hides a character",
    "other": "any other reason (a note is required)",
}
# the app's own condition tags are: glare, blur, low-res, other
REASON_TO_CONDITION = {"blur": "blur", "glare": "glare", "too_small": "low-res"}

# values agents use to dodge the rule; they are rejected with a pointer to mark_unreadable
EVASIONS = ("n/a", "na", "none", "null", "unknown", "unclear", "illegible", "unreadable",
            "not visible", "not readable", "missing", "-", "--", "?", "؟")

# (doc_type, key) -> (what to type, fictional example, side for ID)
HINTS: dict[str, dict[str, tuple[str, str, str | None]]] = {
    "ID": {
        "national_id_number": ("12 digits, first digit 1-9", "198812345678", "front"),
        "name": ("Arabic letters and spaces, as printed", "أحمد", "front"),
        "father_name": ("Arabic letters and spaces, as printed", "علي", "front"),
        "grandfather_name": ("Arabic letters and spaces, as printed", "حسين", "front"),
        "surname": ("Arabic letters and spaces, as printed", "الجبوري", "front"),
        "mother_name": ("Arabic letters and spaces, as printed", "فاطمة", "front"),
        "maternal_grandfather_name": ("Arabic letters and spaces, as printed", "كاظم", "front"),
        "sex": ("exactly ذكر or أنثى", "ذكر", "front"),
        "blood_group": ("O, A, B or AB followed by + or -", "O+", "front"),
        "card_serial": ("2 capital Latin letters + 7 digits (printed under the photo)", "AB1234567", "front"),
        "issuing_authority": ("Arabic text as printed on the back", "مديرية الأحوال المدنية", "back"),
        "issue_date": ("YYYY/MM/DD, Western digits, as printed", "2021/05/17", "back"),
        "expiry_date": ("YYYY/MM/DD, Western digits, as printed", "2031/05/16", "back"),
        "place_of_birth": ("Arabic place name, as printed", "بغداد", "back"),
        "birth_date": ("YYYY/MM/DD, Western digits, as printed", "1988/03/14", "back"),
        "family_number": ("4 digits, L, 1 digit, M, 11 digits", "1234L5M12345678901", "back"),
        "mrz_line1": ("30 characters from A-Z 0-9 and <  (no spaces)", "", "back"),
        "mrz_line2": ("30 characters from A-Z 0-9 and <  (no spaces)", "", "back"),
        "mrz_line3": ("30 characters from A-Z 0-9 and <  (no spaces)", "", "back"),
    },
    "TAX": {
        "taxpayer_number": ("12 digits starting with 10", "101234567890", None),
        "taxpayer_name": ("Arabic text, as printed", "شركة الفرات للتجارة العامة", None),
        "economic_activity": ("Arabic text, as printed", "تجارة الجملة والتجزئة", None),
        "issue_date": ("YYYY/MM/DD, Western digits, as printed", "2022/04/12", None),
        "expiry_date": ("YYYY/MM/DD, Western digits, as printed", "2027/04/11", None),
    },
    "LIC": {
        "license_number": ("6 digits", "123456", None),
        "issue_date": ("YYYY/MM/DD, Western digits, as printed", "2023/06/15", None),
        "company_name": ("Arabic text, as printed", "شركة النور الحديثة للتجارة العامة", None),
        "company_type": ("Arabic text, as printed", "شركة ذات مسؤولية محدودة", None),
        "capital": ("digits with thousands commas, a space, then the printed currency; "
                    "the app expects 'دينار عراقي'", "10,000,000 دينار عراقي", None),
        "business_purposes": ("Arabic text, as printed (the row labeled الغايات / النشاط)",
                              "تجارة المواد الغذائية والمنظفات", None),
        "head_office": ("three parts separated by ' - ' (city - area - street), as printed",
                        "بغداد - الكرادة - شارع فلسطين", None),
        "valid_until": ("YYYY/MM/DD, Western digits, as printed", "2028/06/14", None),
    },
}

def _fill_mrz_examples() -> None:
    """Fill the MRZ examples from the app's own builder so they always have valid check digits."""
    try:
        from rfalib.mrz import build_td1
        lines = build_td1("AB1234567", "198812345678", "1988/03/14", "ذكر", "2031/05/16",
                          "الجبوري", "أحمد", "علي", "حسين")
    except Exception:  # noqa: BLE001 - examples are cosmetic
        return
    for key, line in zip(("mrz_line1", "mrz_line2", "mrz_line3"), lines):
        desc, _example, side = HINTS["ID"][key]
        HINTS["ID"][key] = (desc, line, side)


_fill_mrz_examples()

INSTRUCTIONS = (
    "A.L.G labeling server. You label one synthetic Iraqi document image at a time (ID, TAX card or LIC "
    "business license). Call get_protocol FIRST and follow it exactly: every field is either the exact printed "
    "text or [UNREADABLE]; never guess; zoom in with view_sample(region=...) before giving up. Workflow: "
    "next_sample -> view_sample (every side) -> set_fields / mark_unreadable -> validate_sample -> submit_label."
)

PROTOCOL = """\
A.L.G LABELING PROTOCOL (short; read it fully once)

GOAL
Produce the label for ONE synthetic document image: one value for EVERY field of its schema.
The label records WHAT IS PRINTED ON THE IMAGE, never what you think was intended.

VALUE RULES
1. Every field is exactly one of: (a) the exact printed text, or (b) the token [UNREADABLE].
   Nothing in between: no empty value, no "N/A", no "?", no partial text, no "text [UNREADABLE]".
2. If you cannot read EVERY character of a field with certainty from the pixels, the WHOLE field is
   [UNREADABLE] (covered, cut off, blurred, glare, too dark, too small, damaged). Do not guess.
   Do not complete a pattern. Do not use a dictionary, the expected format, other fields, the MRZ or
   general knowledge to fill characters you cannot see.
3. Copy exactly what is printed: same spelling (even if it looks wrong), same Western digits 0-9, same
   punctuation, same spaces between words. Arabic stays Arabic: never transliterate, correct,
   normalize or translate. Do not add a unit or symbol that is not printed; do not drop one that is.
4. Read each field at its own place on the card. Other places (the MRZ, a repeated number) may only be
   used to double-check, never to fill a field you could not read at its own place.
5. If the printed value does not fit the expected format (for example a currency other than
   دينار عراقي), type what is printed and pass a format_exception that says why. Never "fix" it.

DOCUMENTS
- ID: ONE merged image, front on top and back below. You MUST view both sides. 19 fields incl. 3 MRZ lines.
- TAX: 5 fields. LIC: 8 fields. get_schema(doc_type) lists keys, Arabic labels and format hints.

MRZ (ID only)
- Each mrz_line is exactly 30 characters from A-Z, 0-9 and the filler "<". Count the "<" carefully:
  line 1 ends with 3, line 2 has a run of 11 just before its last character, line 3 is padded to 30.
- Before submitting call mrz_check: it verifies the ICAO check digits and compares the MRZ with the
  visual fields. A failure usually means one misread character or a miscounted "<" run: zoom in and
  re-read; if you still are not certain, mark that line [UNREADABLE].

WORKFLOW
1. next_sample(agent="<your model name>")  -> claims one sample.
2. view_sample for every side. Use region=[x0,y0,x1,y1] (fractions 0..1) to zoom into small text,
   and zoom BEFORE declaring anything unreadable.
3. set_fields for what you can read; mark_unreadable (with a reason code) for the rest.
4. validate_sample -> fix everything it reports.
5. submit_label. If it asks for confirmation, re-inspect the flagged fields first.
6. Repeat. If an image is not a usable document, skip_sample with a reason.

QUALITY
Your labels are audited against human labels and against the generator's reference values.
A confident wrong value is worse than [UNREADABLE]. Marking a clearly readable field unreadable is
also an error: zoom first. You never see the reference values; do not try to infer them.
"""


def reasons_text() -> str:
    return "; ".join(f"{k} = {v}" for k, v in REASONS.items())
