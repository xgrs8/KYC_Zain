"""MRZ checks: what blocks, what only needs a second look, and that an agent is never pushed to change or
drop a line it read correctly (the label must be what is printed, even when the card's check digit is wrong)."""
import json

import pytest
from alg_mcp import mrz_tools as MT
from conftest import ID_VALUES, view_all

from rfalib import mrz as RM

L1, L2, L3 = ID_VALUES["mrz_line1"], ID_VALUES["mrz_line2"], ID_VALUES["mrz_line3"]


def other_digit(d: str) -> str:
    return "0" if d != "0" else "1"


def with_char(line: str, pos: int, ch: str) -> str:
    return line[:pos] + ch + line[pos + 1:]


def bad_composite(l2: str) -> str:
    """Line 2 as printed on a card whose composite check digit does not follow ICAO 9303."""
    return with_char(l2, 29, other_digit(l2[29]))


def claim_id(svc, agent="agentA"):
    r = svc.next_sample(agent, name="ID_01")
    assert r["sample_id"], r
    return r["sample_id"]


# ---------------------------------------------------------------------- check_lines
def test_valid_lines_have_no_problem_and_no_warning():
    res = MT.check_lines((L1, L2, L3))
    assert res["valid"] and res["problems"] == [] and res["warnings"] == [] and res["composite_ok"] is True


def test_a_composite_only_mismatch_is_a_warning_not_a_problem():
    res = MT.check_lines((L1, bad_composite(L2), L3))
    assert not res["valid"] and res["problems"] == [] and res["composite_ok"] is False
    assert len(res["warnings"]) == 1
    assert "composite" in res["warnings"][0] and "do not change it" in res["warnings"][0]


@pytest.mark.parametrize("line, pos, name", [(0, 14, "document number"), (1, 6, "birth date"), (1, 14, "expiry date")])
def test_a_field_check_digit_mismatch_blocks_and_says_where_to_look(line, pos, name):
    lines = [L1, L2, L3]
    lines[line] = with_char(lines[line], pos, other_digit(lines[line][pos]))
    res = MT.check_lines(tuple(lines))
    assert [p for p in res["problems"] if name in p], res
    assert res["warnings"] == []                       # the composite is not reported on top of a field error


def test_structure_problems_still_block():
    assert any("IDIRQ" in p for p in MT.check_lines(("X" + L1[1:], L2, L3))["problems"])
    assert any("IRQ" in p for p in MT.check_lines((L1, L2[:15] + "IRX" + L2[18:], L3))["problems"])
    assert any("character" in p for p in MT.check_lines((L1, L2, L3[:5] + "a" + L3[6:]))["problems"])
    assert any("30" in p for p in MT.check_lines((L1, L2[:-1], L3))["problems"])


def test_filler_repair_hint_also_on_a_card_with_a_non_icao_composite():
    l2 = bad_composite(L2)
    short = l2[:18] + "<" * 10 + l2[-1]                                     # miscounted filler run
    res = MT.check_lines((L1, short, L3))
    assert res["problems"] and res["repair_available"] and "<" in res["repair_hint"]
    assert l2 not in json.dumps(res)                                       # the right text is never handed out


# ---------------------------------------------------------------------- consistency with the visual fields
def test_the_reference_card_is_consistent():
    assert MT.consistency(ID_VALUES) == []


def test_names_written_with_vowels_agree_with_the_printed_arabic():
    printed = ["الجبوري", "محمد", "علي", "حسن"]                        # surname, name, father, grandfather
    assert MT.names_agree(printed, "ALJUBOURI<<MOHAMMED<ALI<HASSAN")      # a full line: 30 characters
    assert MT.names_agree(["العبيدي", "آية", "كريم", "جاسم"], "ALOBAIDI<<AYA<KARIM<JASIM<<<<<")
    full = "ALJUBOURI<<MOHAMMED<ABDULLAH<H"                                 # the line end cuts the last name
    assert len(full) == 30 and MT.names_agree(["الجبوري", "محمد", "عبدالله", "حسن"], full)


def test_a_name_of_several_words_may_be_one_or_several_mrz_words():
    printed = ["الجبوري", "عبد الله", "علي", "حسن"]
    assert MT.names_agree(printed, "ALJUBOURI<<ABDULLAH<ALI<HASSAN<")
    assert MT.names_agree(printed, "ALJUBOURI<<ABD<ALLAH<ALI<HASSAN")
    assert not MT.names_agree(printed, "ALJUBOURI<<ALI<ABDULLAH<HASSAN")


def test_names_that_do_not_belong_to_the_card_are_flagged():
    printed = ["الجبوري", "محمد", "علي", "حسن"]
    assert not MT.names_agree(printed, "ALJUBOURI<<HASSAN<ALI<MOHAMMED")     # name and grandfather swapped
    assert not MT.names_agree(printed, "ALSAMARRAI<<MOHAMMED<ALI<HASSAN")    # another family
    assert not MT.names_agree(printed, "ALJUBOURI<<MOHAMMED<ALI<<<<<<<<")    # a name missing (line not full)
    fields = {**ID_VALUES, "mrz_line3": "ALSAMARRAI<<AHMAD<ALI<HUSSEIN<"}
    assert [f["check"] for f in MT.consistency(fields)] == ["names_vs_mrz"]


def test_empty_optional_data_is_not_a_national_id_mismatch():
    empty = L1[:15] + "<" * 15                                             # the card prints no optional data
    assert "national_id_vs_mrz" not in [f["check"] for f in MT.consistency({**ID_VALUES, "mrz_line1": empty})]
    other = L1[:15] + "123456789012<<<"
    assert "national_id_vs_mrz" in [f["check"] for f in MT.consistency({**ID_VALUES, "mrz_line1": other})]


# ---------------------------------------------------------------------- the service
def test_a_composite_only_mismatch_needs_confirmation_and_keeps_the_typed_line(loaded):
    sid = claim_id(loaded)
    printed = bad_composite(L2)
    loaded.set_fields(sid, {**ID_VALUES, "mrz_line2": printed}, agent="agentA")
    view_all(loaded, sid, "agentA", ("front", "back"))
    rep = loaded.validate_sample(sid, "agentA")
    assert rep["ready_to_submit"], rep["blocking"]
    assert [f["check"] for f in rep["flags"] if f["kind"] == "mrz_check_digit"] == ["composite_check_digit"]
    assert loaded.submit_label(sid, "agentA")["status"] == "needs_confirmation"
    done = loaded.submit_label(sid, "agentA", confirm=True)
    assert done["status"] == "done" and done["review_state"] == "needs_review"
    assert any(r.startswith("mrz_check_digit") for r in done["review_reasons"])
    label = json.loads((loaded.cfg.out_dir / f"{done['label_base']}.label.ai.json").read_text("utf-8"))
    assert label["fields"]["mrz_line2"]["value"] == printed                 # what is printed, not "fixed"


def test_a_blocking_mrz_problem_explains_the_way_out_and_never_suggests_unreadable(loaded):
    sid = claim_id(loaded)
    printed = with_char(L2, 6, other_digit(L2[6]))                         # the card's birth-date digit
    loaded.set_fields(sid, {**ID_VALUES, "mrz_line2": printed}, agent="agentA")
    view_all(loaded, sid, "agentA", ("front", "back"))
    sub = loaded.submit_label(sid, "agentA")
    assert sub["status"] == "blocked"
    msg = " ".join(sub["blocking"])
    assert "birth date" in msg and "exactly what is printed" in msg and "format_exception" in msg
    assert "or mark the bad line unreadable" not in msg
    # the agent is certain: it keeps the line and says why; nothing is changed, a human reviews it
    out = loaded.set_fields(sid, format_exceptions={"mrz_line2": "printed like this; checked at full zoom"},
                            agent="agentA")
    assert out["accepted_with_format_exception"] == ["mrz_line2"]
    rep = loaded.validate_sample(sid, "agentA")
    assert rep["ready_to_submit"], rep["blocking"]
    done = loaded.submit_label(sid, "agentA", confirm=True)
    assert done["status"] == "done" and done["review_state"] == "needs_review"
    label = json.loads((loaded.cfg.out_dir / f"{done['label_base']}.label.ai.json").read_text("utf-8"))
    assert label["fields"]["mrz_line2"]["value"] == printed


def test_a_format_exception_on_a_valid_mrz_is_not_recorded(loaded):
    sid = claim_id(loaded)
    out = loaded.set_fields(sid, dict(ID_VALUES), format_exceptions={"mrz_line2": "nothing is wrong here"},
                            agent="agentA")
    assert "mrz_line2" in out["accepted"] and out["accepted_with_format_exception"] == []


def test_retyping_the_line_drops_its_old_exception(loaded):
    sid = claim_id(loaded)
    loaded.set_fields(sid, {**ID_VALUES, "mrz_line2": with_char(L2, 6, other_digit(L2[6]))},
                      format_exceptions={"mrz_line2": "printed like this on the card"}, agent="agentA")
    view_all(loaded, sid, "agentA", ("front", "back"))
    assert loaded.validate_sample(sid, "agentA")["ready_to_submit"]
    loaded.set_fields(sid, {"mrz_line2": with_char(L2, 6, other_digit(L2[6]))}, agent="agentA")
    assert not loaded.validate_sample(sid, "agentA")["ready_to_submit"]    # a new reading needs a new decision
    loaded.set_fields(sid, {"mrz_line2": L2}, agent="agentA")
    assert loaded.validate_sample(sid, "agentA")["ready_to_submit"]


def test_the_builder_and_the_checker_agree():
    for serial, nid, dob, sex, exp in (("AB1234567", "198812345678", "1988/03/14", "ذكر", "2031/05/16"),
                                       ("ZX7654321", "200001010101", "2000/01/01", "أنثى", "2029/12/31")):
        lines = RM.build_td1(serial, nid, dob, sex, exp, "الجبوري", "أحمد", "علي", "حسين")
        assert MT.check_lines(lines)["valid"]
