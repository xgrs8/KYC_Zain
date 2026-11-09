"""MRZ helpers on top of the app's own ``rfalib.mrz`` (TD1, 3 x 30 characters).

Nothing here ever returns the "expected" MRZ text to an agent: agents must read the MRZ from the
image. We only say whether what they typed is valid and consistent.
"""
from __future__ import annotations

import re

from rfalib import mrz as RM

from .protocol import UNREADABLE

LINE_KEYS = ("mrz_line1", "mrz_line2", "mrz_line3")


def _yymmdd(date: str) -> str | None:
    m = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})", date or "")
    return f"{m.group(1)[2:]}{m.group(2)}{m.group(3)}" if m else None


def _fix_trailing(line: str) -> str | None:
    s = line.rstrip("<")
    return s.ljust(30, "<") if len(s) <= 30 else None


def _fix_line2(line: str) -> str | None:
    if len(line) == 30:
        return line
    if len(line) >= 19 and set(line[18:-1]) <= {"<"}:          # only the filler run has the wrong length
        return line[:18] + "<" * 11 + line[-1]
    return None


def suggest_repair(lines: tuple[str, str, str]) -> tuple[str, str, str] | None:
    """Fix ONLY the number of '<' fillers; accept the repair only if every check digit then verifies."""
    l1, l2, l3 = lines
    f1, f2, f3 = _fix_trailing(l1), _fix_line2(l2), _fix_trailing(l3)
    if None in (f1, f2, f3):
        return None
    cand = (f1, f2, f3)
    if cand == lines or RM.validate_td1(cand):
        return None
    return cand


def check_lines(lines: tuple[str, str, str]) -> dict:
    """Validate three typed lines. Returns {valid, problems, repair_available, repair_hint}."""
    problems: list[str] = []
    for i, line in enumerate(lines, 1):
        if len(line) != 30:
            problems.append(f"mrz_line{i} has {len(line)} characters, it must have exactly 30")
    if not problems:
        problems = list(RM.validate_td1(lines))
    repair = suggest_repair(lines) if problems else None
    out = {"valid": not problems, "problems": problems, "repair_available": repair is not None}
    if repair is not None:
        # We do not hand over the text; we tell the agent where the filler count is off.
        diffs = [f"mrz_line{i}: you typed {len(a)} chars" for i, (a, b) in enumerate(zip(lines, repair), 1)
                 if a != b]
        out["repair_hint"] = ("The data characters are consistent; only the number of '<' fillers is off ("
                              + "; ".join(diffs) + "). Zoom in, count the '<' run again and retype the line.")
    return out


def consistency(fields: dict[str, str]) -> list[dict]:
    """Cross-check the MRZ against the visual fields. Returns only the FAILED checks (soft flags)."""
    lines = [fields.get(k) for k in LINE_KEYS]
    if any(not v or v == UNREADABLE or len(v) != 30 for v in lines):
        return []
    l1, l2, l3 = lines

    def readable(key: str) -> str | None:
        v = fields.get(key)
        return None if not v or v == UNREADABLE else v

    failed: list[dict] = []

    def fail(check: str, visual: str, detail: str) -> None:
        failed.append({"check": check, "fields": [visual, "mrz"], "detail": detail})

    if (v := readable("card_serial")) is not None:
        if l1[5:14].rstrip("<") != v.upper().replace(" ", ""):
            fail("card_serial_vs_mrz", "card_serial", "card_serial differs from the document number in MRZ line 1")
    if (v := readable("national_id_number")) is not None:
        if l1[15:27] != v:
            fail("national_id_vs_mrz", "national_id_number", "national_id_number differs from MRZ line 1 optional data")
    if (v := readable("birth_date")) is not None:
        d = _yymmdd(v)
        if d is not None and l2[0:6] != d:
            fail("birth_date_vs_mrz", "birth_date", "birth_date differs from MRZ line 2")
    if (v := readable("expiry_date")) is not None:
        d = _yymmdd(v)
        if d is not None and l2[8:14] != d:
            fail("expiry_date_vs_mrz", "expiry_date", "expiry_date differs from MRZ line 2")
    if (v := readable("sex")) is not None:
        if RM.sex_initial(v) != l2[7]:
            fail("sex_vs_mrz", "sex", "sex differs from MRZ line 2")
    names = [readable(k) for k in ("surname", "name", "father_name", "grandfather_name")]
    if all(n is not None for n in names):
        try:
            expected = RM.build_td1("AB1234567", "000000000000", "2000/01/01", "M", "2030/01/01", *names)[2]
        except Exception:  # noqa: BLE001
            expected = None
        if expected is not None and expected != l3:
            fail("names_vs_mrz", "surname", "MRZ line 3 does not match the printed names "
                                           "(surname, name, father, grandfather)")
    return failed
