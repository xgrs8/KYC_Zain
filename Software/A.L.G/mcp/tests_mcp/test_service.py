import json
import time

import pytest

from conftest import ID_VALUES, LIC_VALUES, TAX_VALUES, make_image, new_service, view_all
from alg_mcp import mrz_tools
from alg_mcp.service import ToolError
from rfalib import schema as S

UNREADABLE = "[UNREADABLE]"


def claim(svc, agent="agentA", **kw):
    r = svc.next_sample(agent, **kw)
    assert r["sample_id"], r
    return r


# ------------------------------------------------------------------ registration and claiming
def test_add_paths_infers_types_and_pairs(svc, inbox, tmp_path):
    make_image(tmp_path / "inbox" / "mystery.png")
    res = svc.add_paths([str(inbox)])
    assert res["added"] == 4                      # ID_01, ID_02 (pairs), TAX_03, LIC_04
    assert any("mystery.png" in p and "cannot tell" in p for p in res["problems"])
    again = svc.add_paths([str(inbox)])
    assert again["added"] == 0 and again["already_registered"] == 4        # idempotent
    st = svc.queue_status()
    assert st["total"] == 4 and st["by_doc_type"]["ID"]["waiting"] == 2


def test_agent_name_required_and_claim_is_idempotent(loaded):
    with pytest.raises(ToolError, match="Say who you are"):
        loaded.next_sample()
    a = loaded.next_sample("agentA")
    b = loaded.next_sample("agentA")
    assert a["sample_id"] == b["sample_id"] and b["resumed"] is True
    other = loaded.next_sample("agentB")                                    # redundancy 1: a different sample
    assert other["sample_id"] != a["sample_id"]
    named = loaded.next_sample("agentC", name="LIC_04")
    assert named["name"] == "LIC_04"


def test_no_path_is_ever_accepted_from_an_agent(loaded):
    with pytest.raises(ToolError, match="unknown sample"):
        loaded.view_sample("/etc/passwd", agent="agentA")
    with pytest.raises(ToolError, match="unknown sample"):
        loaded.set_fields("../../secret", {"name": "x"}, agent="agentA")


# ------------------------------------------------------------------ value rules
@pytest.mark.parametrize("bad", ["N/A", "n/a", "unknown", "?", "12?4", "١٢٣؟", "abc...", "[blurry]", "{x}",
                                 "اسم [UNREADABLE]", "[UNREADABLE]", "", "   ", "line\nbreak", "-"])
def test_value_rules_reject_dodges(loaded, bad):
    r = claim(loaded, name="TAX_03")
    out = loaded.set_fields(r["sample_id"], {"taxpayer_name": bad}, agent="agentA")
    assert "taxpayer_name" in out["rejected"], (bad, out)
    assert "taxpayer_name" not in out["accepted"]


def test_unknown_field_rejected_and_good_values_accepted(loaded):
    r = claim(loaded, name="TAX_03")
    out = loaded.set_fields(r["sample_id"], {"nonsense": "x", **TAX_VALUES}, agent="agentA")
    assert "nonsense" in out["rejected"]
    assert sorted(out["accepted"]) == sorted(TAX_VALUES)
    assert out["progress"]["resolved"] == 5 and not out["progress"]["missing"]


def test_format_warning_blocks_until_exception_is_given(loaded):
    r = claim(loaded, name="LIC_04")
    sid = r["sample_id"]
    vals = dict(LIC_VALUES, capital="567,000,000 ل.س")                     # printed currency differs from the schema
    out = loaded.set_fields(sid, vals, agent="agentA")
    assert "capital" in out["needs_format_decision"]
    view_all(loaded, sid, "agentA")
    sub = loaded.submit_label(sid, "agentA")
    assert sub["status"] == "blocked" and "capital" in json.dumps(sub, ensure_ascii=False)
    out = loaded.set_fields(sid, format_exceptions={"capital": "the card prints ل.س instead of دينار عراقي"}, agent="agentA")
    assert out["accepted_with_format_exception"] == ["capital"]
    sub = loaded.submit_label(sid, "agentA")
    assert sub["status"] == "needs_confirmation"                           # exceptions always need a conscious yes
    sub = loaded.submit_label(sid, "agentA", confirm=True)
    assert sub["status"] == "done" and sub["review_state"] == "needs_review"
    sidecar = json.loads((loaded.cfg.out_dir / f"{sub['label_base']}.label.sidecar.json").read_text("utf-8"))
    assert sidecar["warnings_overridden"] == ["capital"]                   # same trail as the desktop app


def test_unreadable_needs_valid_reason_and_adds_conditions(loaded):
    r = claim(loaded, name="TAX_03")
    sid = r["sample_id"]
    with pytest.raises(ToolError, match="reason must be one of"):
        loaded.mark_unreadable(sid, "taxpayer_number", "because", agent="agentA")
    with pytest.raises(ToolError, match="needs a note"):
        loaded.mark_unreadable(sid, "taxpayer_number", "other", agent="agentA")
    loaded.mark_unreadable(sid, "taxpayer_number", "blur", agent="agentA")
    loaded.mark_unreadable(sid, "issue_date", "glare", agent="agentA")
    loaded.mark_unreadable(sid, "expiry_date", "too_small", agent="agentA")
    assert loaded.validate_sample(sid, "agentA")["conditions"] == ["blur", "glare", "low-res"]
    out = loaded.set_fields(sid, unreadable={"taxpayer_name": "covered", "economic_activity": "bogus"}, agent="agentA")
    assert out["marked_unreadable"] == ["taxpayer_name"] and "economic_activity" in out["rejected"]
    with pytest.raises(ToolError, match="allowed"):
        loaded.set_conditions(sid, ["sunny"], "agentA")


# ------------------------------------------------------------------ submit gate and outputs
def test_submit_blocked_until_complete_and_viewed(loaded):
    r = claim(loaded, name="TAX_03")
    sid = r["sample_id"]
    sub = loaded.submit_label(sid, "agentA")
    assert sub["status"] == "blocked"
    text = json.dumps(sub["blocking"])
    assert "fields not set yet" in text and "have not viewed" in text
    loaded.set_fields(sid, dict(TAX_VALUES), agent="agentA")
    assert loaded.submit_label(sid, "agentA")["status"] == "blocked"       # still not viewed
    loaded.view_sample(sid, "front", agent="agentA")
    assert loaded.submit_label(sid, "agentA")["status"] == "done"


def test_require_view_can_be_switched_off_for_upload_mode(ws, inbox):
    svc = new_service(ws)
    svc.cfg.require_view = False
    svc.add_paths([str(inbox)])
    sid = claim(svc, name="TAX_03")["sample_id"]
    svc.set_fields(sid, dict(TAX_VALUES), agent="agentA")
    assert svc.submit_label(sid, "agentA")["status"] == "done"


def test_output_files_match_the_app_format(loaded):
    sid = claim(loaded, name="TAX_03")["sample_id"]
    loaded.set_fields(sid, dict(TAX_VALUES, expiry_date="2027/04/11"), agent="agentA")
    loaded.mark_unreadable(sid, "economic_activity", "covered", agent="agentA")
    loaded.view_sample(sid, "front", agent="agentA")
    res = loaded.submit_label(sid, "agentA")
    assert res["status"] == "done"
    d = loaded.cfg.out_dir
    base = res["label_base"]
    assert base == "TAX_03"
    for suffix in (".png", ".label.json", ".label.sidecar.json", ".label.ai.json"):
        assert (d / f"{base}{suffix}").exists(), suffix
    label = json.loads((d / f"{base}.label.json").read_text("utf-8"))
    assert list(label["fields"]) == list(S.label_keys("TAX"))              # fixed order, all keys
    assert label["fields"]["economic_activity"] == UNREADABLE
    assert S.validate_label(label, "TAX") == []                            # the app's own validator accepts it
    side = json.loads((d / f"{base}.label.sidecar.json").read_text("utf-8"))
    assert side["labeler"] == "ai:agentA" and side["field_status"]["economic_activity"] == "unreadable"
    assert side["synthetic"] is True                                       # taken from the truth sidecar
    ai = json.loads((d / f"{base}.label.ai.json").read_text("utf-8"))
    assert ai["agents"] == ["agentA"] and ai["fields"]["economic_activity"]["reason"] == "covered"
    # a finished sample cannot be claimed again
    assert loaded.next_sample("agentA", name="TAX_03")["sample_id"] is None


# ------------------------------------------------------------------ ID, MRZ and blind comparison
def label_id(svc, agent="agentA", overrides=None, unreadable=None, confirm=True):
    sid = claim(svc, agent, name="ID_01")["sample_id"]
    vals = {**ID_VALUES, **(overrides or {})}
    for k in (unreadable or {}):
        vals.pop(k, None)
    out = svc.set_fields(sid, vals, unreadable or {}, agent=agent)
    assert not out["rejected"], out
    view_all(svc, sid, agent, ("front", "back"))
    return sid, svc.submit_label(sid, agent, confirm=confirm)


def test_id_needs_both_sides_viewed(loaded):
    sid = claim(loaded, name="ID_01")["sample_id"]
    loaded.set_fields(sid, dict(ID_VALUES), agent="agentA")
    loaded.view_sample(sid, "front", agent="agentA")
    rep = loaded.validate_sample(sid, "agentA")
    assert not rep["ready_to_submit"] and "back" in json.dumps(rep["blocking"])
    loaded.view_sample(sid, "back", agent="agentA")
    assert loaded.validate_sample(sid, "agentA")["ready_to_submit"]


def test_id_matching_the_reference_is_auto_ok(loaded):
    sid, res = label_id(loaded)
    assert res["status"] == "done" and res["review_state"] == "auto_ok", res
    ai = json.loads((loaded.cfg.out_dir / "ID_01.label.ai.json").read_text("utf-8"))
    assert all(f["reference"] == "agrees" for f in ai["fields"].values())
    assert ai["reference_available"] is True


def test_reference_values_never_reach_the_agent(loaded):
    wrong = "محمد"
    sid, res = label_id(loaded, overrides={"name": wrong}, confirm=True)
    assert res["review_state"] == "needs_review"
    blob = json.dumps(res, ensure_ascii=False)
    assert ID_VALUES["name"] not in blob                                    # we flag it, we do not reveal it
    ai = json.loads((loaded.cfg.out_dir / "ID_01.label.ai.json").read_text("utf-8"))
    assert ai["fields"]["name"]["reference"] == "disagrees"
    with pytest.raises(ToolError, match="hidden from agents"):
        loaded.get_truth(sid, "agentA")


def test_unreadable_although_reference_exists_goes_to_review(loaded):
    sid, res = label_id(loaded, unreadable={"birth_date": "blur"})
    assert res["status"] in ("done", "needs_confirmation")
    if res["status"] == "needs_confirmation":
        res = loaded.submit_label(sid, "agentA", confirm=True)
    ai = json.loads((loaded.cfg.out_dir / "ID_01.label.ai.json").read_text("utf-8"))
    assert ai["fields"]["birth_date"]["reference"] == "agent_unreadable"
    assert ai["review"]["state"] == "needs_review"


def test_mrz_length_digit_and_repair_hint(loaded):
    l1, l2, l3 = ID_VALUES["mrz_line1"], ID_VALUES["mrz_line2"], ID_VALUES["mrz_line3"]
    assert mrz_tools.check_lines((l1, l2, l3))["valid"]
    short = l2[:18] + "<" * 10 + l2[-1]                                    # miscounted filler run
    res = mrz_tools.check_lines((l1, short, l3))
    assert not res["valid"] and res["repair_available"] and "<" in res["repair_hint"]
    assert l2 not in json.dumps(res)                                       # the right text is never handed out
    bad_digit = l2[:2] + ("9" if l2[2] != "9" else "8") + l2[3:]
    res = mrz_tools.check_lines((l1, bad_digit, l3))
    assert not res["valid"] and not res["repair_available"]


def test_invalid_mrz_blocks_submit_and_unreadable_unblocks(loaded):
    sid = claim(loaded, name="ID_01")["sample_id"]
    bad = ID_VALUES["mrz_line2"][:18] + "<" * 10 + ID_VALUES["mrz_line2"][-1]
    # the schema pattern (exactly 30 chars) rejects the short line as a format warning first
    out = loaded.set_fields(sid, {**ID_VALUES, "mrz_line2": bad}, agent="agentA")
    assert "mrz_line2" in out["needs_format_decision"]
    view_all(loaded, sid, "agentA", ("front", "back"))
    assert loaded.submit_label(sid, "agentA")["status"] == "blocked"
    loaded.mark_unreadable(sid, "mrz_line2", "blur", agent="agentA")
    res = loaded.submit_label(sid, "agentA")
    assert res["status"] in ("done", "needs_confirmation")


def test_wrong_digit_in_mrz_blocks_with_a_clear_message(loaded):
    sid = claim(loaded, name="ID_01")["sample_id"]
    l2 = ID_VALUES["mrz_line2"]
    bad = l2[:2] + ("9" if l2[2] != "9" else "8") + l2[3:]
    loaded.set_fields(sid, {**ID_VALUES, "mrz_line2": bad}, agent="agentA")
    view_all(loaded, sid, "agentA", ("front", "back"))
    sub = loaded.submit_label(sid, "agentA")
    assert sub["status"] == "blocked" and "MRZ is invalid" in json.dumps(sub)
    chk = loaded.mrz_check(sid, agent="agentA")
    assert chk["checked"] and not chk["valid"]


def test_mrz_visual_mismatch_needs_confirmation(loaded):
    sid, res = label_id(loaded, overrides={"birth_date": "1988/03/15"}, confirm=False)
    assert res["status"] == "needs_confirmation"
    kinds = [f["check"] for f in res["flags"] if f["kind"] == "mrz_consistency"]
    assert "birth_date_vs_mrz" in kinds
    res = loaded.submit_label(sid, "agentA", confirm=True)
    assert res["status"] == "done" and res["review_state"] == "needs_review"


# ------------------------------------------------------------------ redundancy and leases
def test_two_agents_agree_then_label_is_written(ws, inbox):
    svc = new_service(ws, redundancy=2)
    svc.add_paths([str(inbox)])
    sid, res = label_id(svc, "agentA")
    assert res["status"] == "submitted_waiting_for_second_agent"
    assert not (svc.cfg.out_dir / "ID_01.label.json").exists()
    again = svc.next_sample("agentA", name="ID_01")
    assert again["sample_id"] is None                                       # same agent never labels it twice
    sid_b = svc.next_sample("agentB", name="ID_01")["sample_id"]
    assert sid_b == sid
    svc.set_fields(sid, dict(ID_VALUES), agent="agentB")
    view_all(svc, sid, "agentB", ("front", "back"))
    res = svc.submit_label(sid, "agentB", confirm=True)
    assert res["status"] == "done"
    side = json.loads((svc.cfg.out_dir / "ID_01.label.sidecar.json").read_text("utf-8"))
    assert side["labeler"] == "ai:agentA+agentB"


def test_two_agents_disagree_then_human_resolves(ws, inbox):
    svc = new_service(ws, redundancy=2)
    svc.add_paths([str(inbox)])
    sid, _ = label_id(svc, "agentA")
    svc.next_sample("agentB", name="ID_01")
    svc.set_fields(sid, {**ID_VALUES, "mother_name": "زينب"}, agent="agentB")
    view_all(svc, sid, "agentB", ("front", "back"))
    res = svc.submit_label(sid, "agentB", confirm=True)
    assert res["status"] == "needs_human" and res["fields"] == ["mother_name"]
    assert not (svc.cfg.out_dir / "ID_01.label.json").exists()
    out = svc.resolve(sid, "agentA")
    assert out["status"] == "done"
    label = json.loads((svc.cfg.out_dir / "ID_01.label.json").read_text("utf-8"))
    assert label["fields"]["mother_name"] == ID_VALUES["mother_name"]


def test_two_agents_one_says_unreadable_is_a_disagreement(ws, inbox):
    svc = new_service(ws, redundancy=2)
    svc.add_paths([str(inbox)])
    sid, _ = label_id(svc, "agentA")
    svc.next_sample("agentB", name="ID_01")
    vals = dict(ID_VALUES)
    vals.pop("blood_group")
    svc.set_fields(sid, vals, {"blood_group": "glare"}, agent="agentB")
    view_all(svc, sid, "agentB", ("front", "back"))
    res = svc.submit_label(sid, "agentB", confirm=True)
    assert res["status"] == "needs_human" and "blood_group" in res["fields"]


def test_expired_lease_frees_the_sample(loaded):
    first = claim(loaded, "agentA", name="TAX_03")
    with loaded.store.tx() as st:
        for s in st["samples"].values():
            for c in s["claims"].values():
                c["expires"] = time.time() - 1
    taken = loaded.next_sample("agentB", name="TAX_03")
    assert taken["sample_id"] == first["sample_id"]
    with pytest.raises(ToolError, match="claimed by another agent"):
        loaded.set_fields(first["sample_id"], {"taxpayer_name": "x"}, agent="agentA")


def test_release_keeps_the_draft_and_skip_needs_a_reason(loaded):
    sid = claim(loaded, name="TAX_03")["sample_id"]
    loaded.set_fields(sid, {"taxpayer_number": "101234567890"}, agent="agentA")
    loaded.release_sample(sid, "agentA")
    back = loaded.next_sample("agentA", name="TAX_03")
    assert back["already_set"] == ["taxpayer_number"]
    with pytest.raises(ToolError, match="needs a reason"):
        loaded.skip_sample(sid, "no", "agentA")
    assert loaded.skip_sample(sid, "photo shows a different document", "agentA")["status"] == "skipped"
    assert loaded.queue_status()["by_status"]["skipped"] == 1


# ------------------------------------------------------------------ external photos get app-style names
def test_external_photo_gets_a_counter_name_in_the_app_style(ws, tmp_path):
    svc = new_service(ws)
    p = make_image(tmp_path / "IMG_0042.jpg", size=(900, 1200))
    svc.add_paths([str(p)], doc_type="LIC")
    sid = claim(svc, name="IMG_0042")["sample_id"]
    svc.set_fields(sid, dict(LIC_VALUES), agent="agentA")
    svc.view_sample(sid, "front", agent="agentA")
    res = svc.submit_label(sid, "agentA")
    assert res["status"] == "done" and res["label_base"] == "LIC_01"
    ai = json.loads((svc.cfg.out_dir / "LIC_01.label.ai.json").read_text("utf-8"))
    assert ai["review"]["state"] == "needs_review"                          # no reference values, single agent


# ------------------------------------------------------------------ a print run with declared deviations
PRINT_RUN_ROW = {
    "identity_id": "001", "first": "ديار", "father": "آرام", "grandfather": "بلال", "mother": "مريم",
    "surname": "الربيعي", "sex": "ذكر", "blood": "A+", "dob": "1973/01/31", "pob": "كركوك",
    "issue_office": "صلاح الدين", "issue_date": "2020/05/03", "expiry_date": "2030/05/03",
    "doc_number": "DV8752051", "national_number_front": "9836146454363",
    "national_number_back": "5200N1W26815209887", "mrz_line1": "IDIRQDV87520514<<<<<<<<<<<<<<<",
    "mrz_line2": "7301315M3005039IRQ<<<<<<<<<<<3", "mrz_line3": "ALRUBAIE<<DIYAR<ARAM<BILAL<<<<",
}


def _print_run(tmp_path, ws):
    from rfalib import printed_truth as PT
    d = tmp_path / "printrun"
    make_image(d / "ID_101_glare_front.png", label="front")
    make_image(d / "ID_101_glare_back.png", label="back")
    t = PT.truth_for("ID", PRINT_RUN_ROW)
    (d / "ID_101.truth.json").write_text(json.dumps(
        {"printed": t["printed"], "intended": t["intended"], "defects": t["defects"], "split": "train",
         "synthetic": True, "known_deviations": t["known_deviations"]}, ensure_ascii=False), encoding="utf-8")
    svc = new_service(ws)
    assert svc.add_paths([str(d)])["added"] == 1
    return svc, {k: v for k, v in t["printed"].items() if k in S.label_keys("ID")}


def test_declared_print_run_formats_need_no_exception(tmp_path, ws):
    svc, printed = _print_run(tmp_path, ws)
    sid = claim(svc, name="ID_101_glare")["sample_id"]
    out = svc.set_fields(sid, printed, agent="agentA")
    assert out["needs_format_decision"] == {} and not out["rejected"], out
    view_all(svc, sid, "agentA", ("front", "back"))
    rep = svc.validate_sample(sid, "agentA")
    assert rep["ready_to_submit"] and rep["flags"] == [], rep          # no composite / names_vs_mrz flag
    assert "names_vs_mrz" in svc.mrz_check(sid, agent="agentA")["checks_not_applied"]
    res = svc.submit_label(sid, "agentA")
    assert res["status"] == "done" and res["label_base"] == "ID_101" and res["review_state"] == "auto_ok", res
    ai = json.loads((svc.cfg.out_dir / "ID_101.label.ai.json").read_text("utf-8"))
    assert "national_id_number" in ai["known_deviations"]
    side = json.loads((svc.cfg.out_dir / "ID_101.label.sidecar.json").read_text("utf-8"))
    assert "family_number" in side["known_deviations"]


def test_a_value_that_fits_neither_format_still_needs_a_decision(tmp_path, ws):
    svc, printed = _print_run(tmp_path, ws)
    sid = claim(svc, name="ID_101_glare")["sample_id"]
    out = svc.set_fields(sid, {"national_id_number": "98361464543"}, agent="agentA")      # 11 digits
    assert "national_id_number" in out["needs_format_decision"]
    blob = json.dumps(out, ensure_ascii=False)
    assert printed["national_id_number"] not in blob and printed["surname"] not in blob   # still blind


def test_several_photos_of_one_card_get_free_names(ws, tmp_path):
    d = tmp_path / "photos"
    for cond in ("glare", "blur"):
        make_image(d / f"TAX_03_{cond}.png", size=(800, 560), label=cond)
    svc = new_service(ws)
    assert svc.add_paths([str(d)])["added"] == 2
    bases = []
    for cond in ("blur", "glare"):
        sid = claim(svc, name=f"TAX_03_{cond}")["sample_id"]
        svc.set_fields(sid, dict(TAX_VALUES), agent="agentA")
        svc.view_sample(sid, "front", agent="agentA")
        res = svc.submit_label(sid, "agentA", confirm=True)
        assert res["status"] == "done", res
        bases.append(res["label_base"])
    assert bases[0] == "TAX_03" and bases[1] != "TAX_03"            # the second photo is not overwritten
    assert (svc.cfg.out_dir / f"{bases[1]}.label.json").exists()
