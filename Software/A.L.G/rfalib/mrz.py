"""MRZ (machine readable zone) - TD1 format, 3 lines x 30 chars (plan 6.3).

The MRZ is ALWAYS derived from the printed fields with valid ICAO check
digits - it is never randomized (D3).

Line 1 (30): "ID" + "IRQ" + doc_number(9) + cd + optional(12) + "<<<"
Line 2 (30): YYMMDD(birth) + cd + sex(M|F) + YYMMDD(expiry) + cd
             + "IRQ" + "<"*11 + composite
Line 3 (30): translit(surname) + "<<" + translit(name) + "<"
             + translit(father) + "<" + translit(grandfather), padded/truncated

composite = cd( L1[5:30] + L2[0:7] + L2[8:15] + L2[18:29] )

Check digit: weights 7,3,1 repeating; digits = face value; A-Z = 10-35;
'<' = 0; sum mod 10.  (ICAO 9303 / machine-readable passport standard.)
"""
from __future__ import annotations

import datetime as _dt

from .constants import MRZ_DOC_CODE, MRZ_LINE_LEN, MRZ_NATIONALITY, MRZ_STATE, TRANSLIT_TABLE

_WEIGHTS = (7, 3, 1)


def char_value(ch: str) -> int:
    """ICAO 9303 character value: 0-9 face value, A-Z 10-35, '<' 0."""
    if ch == "<":
        return 0
    if "0" <= ch <= "9":
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    raise ValueError(f"invalid MRZ character: {ch!r}")


def check_digit(data: str) -> str:
    """Compute the ICAO check digit for `data` (weights 7,3,1; mod 10)."""
    total = 0
    for i, ch in enumerate(data):
        total += char_value(ch) * _WEIGHTS[i % 3]
    return str(total % 10)


def transliterate(text: str) -> str:
    """Arabic -> Latin transliteration using the plan table.

    Non-Arabic characters (spaces, Latin, digits) pass through unchanged.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        two = text[i:i + 2]
        if two in TRANSLIT_TABLE:  # handles لا -> LA
            out.append(TRANSLIT_TABLE[two])
            i += 2
            continue
        ch = text[i]
        out.append(TRANSLIT_TABLE.get(ch, ch if not _is_arabic(ch) else ch))
        i += 1
    return "".join(out)


def _is_arabic(ch: str) -> bool:
    return "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"


# ---------------------------------------------------------------- helpers

def _iso(date_str: str) -> _dt.date:
    return _dt.date(*(int(p) for p in date_str.split("/")))


def _yymmdd(date_str: str) -> str:
    d = _iso(date_str)
    return f"{d.year % 100:02d}{d.month:02d}{d.day:02d}"


def _pad(data: str, length: int, fill: str = "<") -> str:
    data = data[:length]
    return data + fill * (length - len(data))


def _clean_latin(text: str, max_len: int) -> str:
    """Uppercase, keep A-Z only (spaces become '<'), truncate."""
    up = transliterate(text).upper()
    kept = "".join(ch if ("A" <= ch <= "Z" or "0" <= ch <= "9") else ""
                   for ch in up)
    return kept[:max_len]


def sex_initial(sex_arabic: str) -> str:
    """ذكر -> M, أنثى -> F (normalized comparison; anything else -> '<')."""
    from .normalize import normalize
    n = normalize(sex_arabic)
    if n in ("ذكر", "م"):
        return "M"
    if n in ("أنثى", "انثى", "f".upper()):
        return "F"
    # fall back: also accept already-latin M/F
    s = sex_arabic.strip().upper()
    if s in ("M", "F"):
        return s
    return "<"


# ---------------------------------------------------------------- TD1 builder

def build_td1(card_serial: str, national_id_number: str, birth_date: str,
              sex: str, expiry_date: str, surname: str, name: str,
              father_name: str, grandfather_name: str) -> tuple[str, str, str]:
    """Build the 3 TD1 lines from printed field values (all check digits valid).

    Dates are YYYY/MM/DD strings.  `card_serial` fills the L1 document-number
    slot (the template's own AF/AA novelty inconsistency is not copied - 4.3).
    Returns (line1, line2, line3).
    """
    doc_no = _pad(card_serial.upper().replace(" ", ""), 9)
    optional = _pad(national_id_number, 12)

    l1 = (MRZ_DOC_CODE + MRZ_STATE + doc_no + check_digit(doc_no)
          + optional + "<<<")
    assert len(l1) == MRZ_LINE_LEN

    dob = _yymmdd(birth_date)
    exp = _yymmdd(expiry_date)
    sx = sex_initial(sex)
    l2 = (dob + check_digit(dob) + sx + exp + check_digit(exp)
          + MRZ_NATIONALITY + "<" * 11)
    composite_input = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    l2 += check_digit(composite_input)
    assert len(l2) == MRZ_LINE_LEN

    name_part = "<<" + _clean_latin(name, 28)
    l3 = (_clean_latin(surname, 28) + name_part
          + "<" + _clean_latin(father_name, 28)
          + "<" + _clean_latin(grandfather_name, 28))
    l3 = _pad(l3, MRZ_LINE_LEN)
    assert len(l3) == MRZ_LINE_LEN

    return l1, l2, l3


# ---------------------------------------------------------------- verification

def verify_line_checks(lines: tuple[str, str, str]) -> bool:
    """Re-verify every check digit of a TD1 triple (for tests / selftest)."""
    l1, l2, l3 = lines
    if not all(len(x) == MRZ_LINE_LEN for x in (l1, l2, l3)):
        return False
    try:
        for line in lines:
            for c in line:
                char_value(c)
    except ValueError:
        return False
    ok = (
        l1[14] == check_digit(l1[5:14])
        and l2[6] == check_digit(l2[0:6])
        and l2[14] == check_digit(l2[8:14])
        and l2[29] == check_digit(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29])
    )
    return ok


def validate_td1(lines: tuple[str, str, str]) -> list[str]:
    """Full structural validation; returns list of problems (empty = valid)."""
    problems: list[str] = []
    l1, l2, l3 = lines
    if len(l1) != MRZ_LINE_LEN or len(l2) != MRZ_LINE_LEN or len(l3) != MRZ_LINE_LEN:
        problems.append("line lengths must be 30/30/30")
        return problems
    if not l1.startswith(MRZ_DOC_CODE + MRZ_STATE):
        problems.append("L1 must start with IDIRQ")
    if l1[17:30] != l1[17:30].replace(" ", ""):
        problems.append("L1 optional data contains spaces")
    for c in "".join(lines):
        if not ("A" <= c <= "Z" or "0" <= c <= "9" or c == "<"):
            problems.append(f"invalid character {c!r}")
            break
    if l2[7] not in ("M", "F", "<"):
        problems.append("L2 sex must be M/F/<")
    if l2[15:18] != MRZ_NATIONALITY:
        problems.append("L2 nationality must be IRQ")
    if not verify_line_checks(lines):
        problems.append("check digit mismatch")
    return problems
