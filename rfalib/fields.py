"""Field generators, validators and cross-field rules (plan 6.2, D3).

Seeding: `rng = random.Random(sha256(batch_seed + doc_id))` - decided by
the pipeline, which passes the rng in.  Uniqueness inside a batch is
enforced through the `uniques` set dict.  The MRZ is always DERIVED with
valid check digits - never randomized (D3).
"""
from __future__ import annotations

import datetime as _dt
import random
import re

from . import mrz
from .constants import AGE_MAX, AGE_MIN, DATE_MIN, VALIDITY_YEARS
from .pools import Pools

MAX_TRIES = 200


# ---------------------------------------------------------------- date utils

def add_years(d: _dt.date, years: int) -> _dt.date:
    """Add calendar years (Feb 29 falls back to Feb 28)."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def expiry_for(doc_type: str, issue: _dt.date) -> _dt.date:
    """expiry = issue + N years - 1 day (ID 10y-1d, TAX/LIC 5y-1d)."""
    return add_years(issue, VALIDITY_YEARS[doc_type]) - _dt.timedelta(days=1)


def fmt(d: _dt.date) -> str:
    return d.strftime("%Y/%m/%d")


def parse_date(s: str) -> _dt.date:
    return _dt.datetime.strptime(s, "%Y/%m/%d").date()


def _random_issue(rng: random.Random, doc_type: str) -> _dt.date:
    lo = _dt.date.fromisoformat(DATE_MIN[doc_type])
    hi = _dt.date.today()
    days = (hi - lo).days
    return lo + _dt.timedelta(days=rng.randint(0, max(0, days)))


# ---------------------------------------------------------------- uniques

class Uniques:
    """In-batch uniqueness store for all number fields."""

    def __init__(self) -> None:
        self._seen: dict[str, set] = {}

    def take(self, key: str, maker, rng: random.Random) -> str:
        seen = self._seen.setdefault(key, set())
        for _ in range(MAX_TRIES):
            value = maker(rng)
            if value not in seen:
                seen.add(value)
                return value
        raise RuntimeError(f"cannot generate unique value for {key!r}")


# ---------------------------------------------------------------- ID card

def generate_id(rng: random.Random, pools: Pools, uniques: Uniques) -> dict:
    """All 19 label fields + 2 sidecar-only fields, MRZ derived (6.2)."""
    sex = rng.choice(["ذكر", "أنثى"])
    given_pool = "given_m" if sex == "ذكر" else "given_f"
    issue = _random_issue(rng, "ID")
    age = rng.randint(AGE_MIN, AGE_MAX)
    birth = add_years(issue, -age)
    if birth >= issue:  # safety for extreme dates
        birth = issue - _dt.timedelta(days=366)

    name = rng.choice(pools.get(given_pool))
    father = rng.choice(pools.get("given_m"))
    grandfather = rng.choice(pools.get("given_m"))
    surname = rng.choice(pools.get("surnames"))
    mother = rng.choice(pools.get("given_f"))
    maternal_gf = rng.choice(pools.get("given_m"))

    fields = {
        "national_id_number": uniques.take(
            "national_id_number",
            lambda r: str(r.randint(1, 9)) + "".join(
                str(r.randint(0, 9)) for _ in range(11)), rng),
        "name": name,
        "father_name": father,
        "grandfather_name": grandfather,
        "surname": surname,
        "mother_name": mother,
        "maternal_grandfather_name": maternal_gf,
        "sex": sex,
        "blood_group": rng.choice(["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]),
        "card_serial": uniques.take(
            "card_serial",
            lambda r: "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
            + "".join(str(r.randint(0, 9)) for _ in range(7)), rng),
        "issuing_authority": pools.pick(rng, "authorities"),
        "issue_date": fmt(issue),
        "expiry_date": fmt(expiry_for("ID", issue)),
        "place_of_birth": pools.pick(rng, "cities"),
        "birth_date": fmt(birth),
        "family_number": uniques.take(
            "family_number",
            lambda r: ("".join(str(r.randint(0, 9)) for _ in range(4)) + "L"
                       + str(r.randint(0, 9)) + "M"
                       + "".join(str(r.randint(0, 9)) for _ in range(11))), rng),
    }
    fields["mrz_line1"], fields["mrz_line2"], fields["mrz_line3"] = mrz.build_td1(
        card_serial=fields["card_serial"],
        national_id_number=fields["national_id_number"],
        birth_date=fields["birth_date"], sex=fields["sex"],
        expiry_date=fields["expiry_date"], surname=fields["surname"],
        name=fields["name"], father_name=fields["father_name"],
        grandfather_name=fields["grandfather_name"])
    # sidecar-only (OQ-5): printed on the card, NOT in the label JSON
    fields["rfid_access_number"] = uniques.take(
        "rfid_access_number",
        lambda r: "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
                          for _ in range(9)), rng)
    fields["vertical_serial"] = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return fields


# ---------------------------------------------------------------- tax card

def generate_tax(rng: random.Random, pools: Pools, uniques: Uniques,
                 person: dict | None = None) -> dict:
    """Tax card fields; in case mode `person` is embedded in the name (D4)."""
    issue = _random_issue(rng, "TAX")
    if person is not None:
        taxpayer = (f"شركة {person['name']} {person['surname']} "
                    f"{rng.choice(['للتجارة العامة', 'للمقاولات العامة', 'للصناعات الحديثة', 'للخدمات المتكاملة'])}")
    else:
        taxpayer = pools.pick(rng, "companies")
    return {
        "taxpayer_number": uniques.take(
            "taxpayer_number",
            lambda r: "10" + "".join(str(r.randint(0, 9)) for _ in range(10)), rng),
        "taxpayer_name": taxpayer,
        "economic_activity": pools.pick(rng, "activities"),
        "issue_date": fmt(issue),
        "expiry_date": fmt(expiry_for("TAX", issue)),
    }


# ---------------------------------------------------------------- license

def _format_capital(amount: int) -> str:
    return f"{amount:,} دينار عراقي"


def generate_lic(rng: random.Random, pools: Pools, uniques: Uniques,
                 person: dict | None = None) -> dict:
    """Business license fields; in case mode `person` (an ID field dict)
    is embedded into the company name (D4)."""
    issue = _random_issue(rng, "LIC")
    if person is not None:
        company = (f"شركة {person['name']} {person['surname']} "
                   f"{rng.choice(['للتجارة العامة', 'للمقاولات العامة', 'للصناعات الحديثة', 'للخدمات المتكاملة'])}")
    else:
        company = pools.pick(rng, "companies")
    amount = rng.randrange(1, 100_001) * 1_000_000  # 1e6 .. 1e11 IQD
    head_office = " - ".join((pools.pick(rng, "cities"),
                              pools.pick(rng, "areas"),
                              pools.pick(rng, "streets")))
    return {
        "license_number": uniques.take(
            "license_number",
            lambda r: "".join(str(r.randint(0, 9)) for _ in range(6)), rng),
        "issue_date": fmt(issue),
        "company_name": company,
        "company_type": pools.pick(rng, "company_types"),
        "capital": _format_capital(amount),
        "business_purposes": pools.pick(rng, "purposes"),
        "head_office": head_office,
        "valid_until": fmt(expiry_for("LIC", issue)),
    }


GENERATORS = {"ID": generate_id, "TAX": generate_tax, "LIC": generate_lic}


def generate(doc_type: str, rng: random.Random, pools: Pools,
             uniques: Uniques, **kw) -> dict:
    return GENERATORS[doc_type](rng, pools, uniques, **kw)


# ---------------------------------------------------------------- cross-field

def check_crossfield(doc_type: str, f: dict) -> list[str]:
    """Cross-field rules enforced before render (6.2). [] = ok."""
    problems: list[str] = []
    try:
        if doc_type == "ID":
            birth = parse_date(f["birth_date"])
            issue = parse_date(f["issue_date"])
            expiry = parse_date(f["expiry_date"])
            if not (birth < issue < expiry):
                problems.append("birth < issue < expiry violated")
            if expiry != expiry_for("ID", issue):
                problems.append("ID expiry != issue + 10y - 1d")
            age = (issue - add_years(birth, 0)).days / 365.25
            if not (AGE_MIN - 1 <= age <= AGE_MAX + 1):
                problems.append(f"implausible age at issue: {age:.1f}")
            lines = (f["mrz_line1"], f["mrz_line2"], f["mrz_line3"])
            if mrz.validate_td1(lines):
                problems.append("MRZ structurally invalid")
            rebuilt = mrz.build_td1(
                f["card_serial"], f["national_id_number"], f["birth_date"],
                f["sex"], f["expiry_date"], f["surname"], f["name"],
                f["father_name"], f["grandfather_name"])
            if rebuilt != lines:
                problems.append("MRZ not derived from printed fields")
        elif doc_type == "TAX":
            issue = parse_date(f["issue_date"])
            expiry = parse_date(f["expiry_date"])
            if not expiry > issue:
                problems.append("expiry must be after issue")
            if expiry != expiry_for("TAX", issue):
                problems.append("TAX expiry != issue + 5y - 1d")
        elif doc_type == "LIC":
            issue = parse_date(f["issue_date"])
            valid = parse_date(f["valid_until"])
            if not valid > issue:
                problems.append("valid_until must be after issue_date")
            if valid != expiry_for("LIC", issue):
                problems.append("LIC valid_until != issue + 5y - 1d")
            if not re.match(r"^\d{1,3}(,\d{3})* دينار عراقي$", f["capital"]):
                problems.append("capital format wrong")
    except (KeyError, ValueError) as exc:
        problems.append(f"missing/invalid field: {exc}")
    return problems
