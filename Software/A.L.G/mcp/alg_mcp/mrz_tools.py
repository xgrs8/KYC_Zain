"""MRZ helpers on top of the app's own ``rfalib.mrz`` (TD1, 3 x 30 characters).

Nothing here ever returns the "expected" MRZ text to an agent: agents must read the MRZ from the
image. We only say whether what they typed is valid and consistent.

Two kinds of MRZ problem are kept apart on purpose:

* ``problems`` block the submission. They are the ones that almost always mean a misread: a line that
  is not 30 characters, a character outside ``A-Z 0-9 <``, a wrong fixed part (``IDIRQ``, ``IRQ``, the
  sex letter), or a wrong check digit for the document number, the birth date or the expiry date. Each
  of those check digits covers a few characters only, so the message says exactly where to look.
* ``warnings`` do not block. Today there is one: the composite check digit (last character of line 2).
  When every other check digit verifies, a composite mismatch means either that one character was
  misread or that the card itself is printed with a non-ICAO composite digit. Printed cards do that:
  in the project's ground truth (``printables/ground_truth/national_id.csv``) 81 of the 90 national
  IDs carry a composite digit that the ICAO formula does not give, while all 270 field check digits
  verify. Blocking on it pushed agents to change a correctly read character or to mark a readable line
  ``[UNREADABLE]``. Now the agent is asked to re-check that character and confirm, and the label goes
  to human review.

The agent must always type what is printed, never what the check digits would like to see.
"""
from __future__ import annotations

import difflib
import itertools
import re

from rfalib import mrz as RM

from .protocol import UNREADABLE

LINE_KEYS = ("mrz_line1", "mrz_line2", "mrz_line3")

# check digit name, (line index, digit position), data slice, where to look (1-based positions)
_FIELD_DIGITS = (
    ("document number", (0, 14), lambda l1, l2: l1[5:14],
     "mrz_line1: characters 6-14 (the document number) and character 15 (its check digit)"),
    ("birth date", (1, 6), lambda l1, l2: l2[0:6],
     "mrz_line2: characters 1-6 (the birth date) and character 7 (its check digit)"),
    ("expiry date", (1, 14), lambda l1, l2: l2[8:14],
     "mrz_line2: characters 9-14 (the expiry date) and character 15 (its check digit)"),
)
_COMPOSITE_WHERE = ("mrz_line2 character 30 (the composite check digit), and the characters only it covers: "
                    "mrz_line1 characters 16-30 and mrz_line2 characters 19-29")

# The printed names are transliterated by people, with vowels (الربيعي -> ALRUBAIE), while rfalib's
# table is letter by letter (ALRBYAY), so an exact comparison flags every real card. We compare consonant
# skeletons with a tolerance measured on the project's 90 ground-truth IDs: none of them is flagged, while
# 96% of the cases where one name is replaced by another person's (or name and grandfather are swapped) are.
NAME_SIMILARITY_MIN = 0.6


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


def _charset_ok(lines: tuple[str, str, str]) -> bool:
    return all("A" <= c <= "Z" or "0" <= c <= "9" or c == "<" for c in "".join(lines))


def _digit_ok(digit: str, data: str) -> bool:
    try:
        return digit == RM.check_digit(data)
    except ValueError:                                         # a character outside the MRZ alphabet
        return False


def _composite_ok(l1: str, l2: str) -> bool:
    return _digit_ok(l2[29], l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29])


def _hard_problems(lines: tuple[str, str, str]) -> list[str]:
    """Everything that blocks: lengths, alphabet, fixed parts, and the three field check digits."""
    problems: list[str] = []
    for i, line in enumerate(lines, 1):
        if len(line) != 30:
            problems.append(f"mrz_line{i} has {len(line)} characters, it must have exactly 30")
    if problems:
        return problems
    problems = [p for p in RM.validate_td1(lines) if p != "check digit mismatch"]
    if not _charset_ok(lines):
        return problems                                        # validate_td1 already named the bad character
    l1, l2, _ = lines
    for name, (li, pos), data, where in _FIELD_DIGITS:
        line = (l1, l2)[li]
        if not _digit_ok(line[pos], data(l1, l2)):
            problems.append(f"the {name} check digit does not match: re-read {where}")
    return problems


def suggest_repair(lines: tuple[str, str, str]) -> tuple[str, str, str] | None:
    """Fix ONLY the number of '<' fillers; accept the repair only if the field check digits then verify.

    The composite digit is not required: a card may be printed with a non-ICAO composite digit.
    """
    l1, l2, l3 = lines
    f1, f2, f3 = _fix_trailing(l1), _fix_line2(l2), _fix_trailing(l3)
    if None in (f1, f2, f3):
        return None
    cand = (f1, f2, f3)
    if cand == lines or _hard_problems(cand):
        return None
    return cand


def check_lines(lines: tuple[str, str, str]) -> dict:
    """Validate three typed lines.

    Returns {valid, problems, warnings, composite_ok, repair_available[, repair_hint]}: ``valid`` is True
    only when every ICAO check passes; ``problems`` block the submission, ``warnings`` need a re-check
    and a confirmation (see the module docstring).
    """
    problems = _hard_problems(lines)
    warnings: list[str] = []
    composite = None
    if all(len(x) == 30 for x in lines) and _charset_ok(lines):
        composite = _composite_ok(lines[0], lines[1])
        if not composite and not problems:
            warnings.append("the composite check digit does not match although every other check digit does. "
                            f"Re-read {_COMPOSITE_WHERE}. Some cards are printed with a composite digit that "
                            "does not follow ICAO 9303: if what you typed is what is printed, keep it "
                            "(do not change it to make the check pass) and confirm.")
    repair = suggest_repair(lines) if problems else None
    out = {"valid": not problems and composite is True, "problems": problems, "warnings": warnings,
           "composite_ok": composite, "repair_available": repair is not None}
    if repair is not None:
        # We do not hand over the text; we tell the agent where the filler count is off.
        diffs = [f"mrz_line{i}: you typed {len(a)} chars" for i, (a, b) in enumerate(zip(lines, repair), 1)
                 if a != b]
        out["repair_hint"] = ("The data characters are consistent; only the number of '<' fillers is off ("
                              + "; ".join(diffs) + "). Zoom in, count the '<' run again and retype the line.")
    return out


# ---------------------------------------------------------------------- cross-checks with the visual fields
def _skeleton(latin: str) -> str:
    """Consonant skeleton: A-Z only, no vowels or the letters that stand for long vowels, no doubles."""
    s = re.sub(r"[^A-Z]", "", latin.upper())
    s = re.sub(r"[AEIOUYW]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _latin(arabic: str) -> str:
    # taa marbuta is written as a vowel in the MRZ (آية -> AYA), not as the table's H
    return "".join(ch for ch in RM.transliterate(arabic.replace("ة", "ا")).upper() if "A" <= ch <= "Z")


def _close(printed: str, typed: str, cut: bool) -> bool:
    """One printed name (Arabic, maybe several words) against its MRZ spelling; ``cut``: the MRZ line ends
    inside this name, so only its beginning can be compared."""
    a, b = _skeleton(_latin(printed)), _skeleton(typed)
    if cut:
        a = a[:len(b)]
    if not a and not b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= NAME_SIMILARITY_MIN


def names_agree(printed: list[str], l3: str) -> bool:
    """True when the four printed names (surname, name, father, grandfather) plausibly match MRZ line 3.

    A name of several words (عبد الله) may be one MRZ word (ABDULLAH) or several (ABD<ALLAH), so every
    split is tried. A full line (no trailing '<') may cut the last names short or drop them."""
    surname, _, rest = l3.partition("<<")
    truncated = not l3.endswith("<")
    tokens = [t for t in rest.split("<") if t]
    if not _close(printed[0], surname.replace("<", ""), truncated and not tokens):
        return False
    given = printed[1:]
    options = [sorted({1, max(1, len(g.split()))}) for g in given]
    for counts in itertools.product(*options):
        need = sum(counts)
        if need != len(tokens) and not (truncated and need > len(tokens)):
            continue
        i, ok = 0, True
        for name, n in zip(given, counts):
            part = tokens[i:i + n]
            i += n
            if not part:                                       # dropped by the end of the line
                continue
            if not _close(name, "".join(part), truncated and i >= len(tokens)):
                ok = False
                break
        if ok:
            return True
    return False


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
    optional = l1[15:30].rstrip("<")
    if optional and (v := readable("national_id_number")) is not None:
        # some cards leave the optional data empty: then there is nothing to compare
        if optional != v.replace(" ", "")[:15]:
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
    if all(n is not None for n in names) and not names_agree(names, l3):    # type: ignore[arg-type]
        fail("names_vs_mrz", "surname", "MRZ line 3 does not match the printed names "
                                        "(surname, name, father, grandfather)")
    return failed
