import json

from alg.label import LabelSession
from alg_mcp import cli, review
from alg_mcp.config import Config
from conftest import ID_VALUES, TAX_VALUES, new_service, view_all

UNREADABLE = "[UNREADABLE]"


def run_cli(capsys, ws, *args):
    code = cli.main([args[0], "--workspace", str(ws), *args[1:]])
    out = capsys.readouterr()
    return code, out.out, out.err


def label_everything(svc):
    """Label ID_01 and TAX_03 exactly like the reference values -> both auto_ok."""
    sid = svc.next_sample("agentA", name="ID_01")["sample_id"]
    svc.set_fields(sid, dict(ID_VALUES), agent="agentA")
    view_all(svc, sid, "agentA", ("front", "back"))
    assert svc.submit_label(sid, "agentA", confirm=True)["review_state"] == "auto_ok"
    sid = svc.next_sample("agentA", name="TAX_03")["sample_id"]
    svc.set_fields(sid, dict(TAX_VALUES), agent="agentA")
    svc.view_sample(sid, "front", agent="agentA")
    assert svc.submit_label(sid, "agentA")["review_state"] == "auto_ok"


def test_add_status_and_doctor(capsys, ws, inbox):
    code, out, _ = run_cli(capsys, ws, "add", str(inbox))
    assert code == 0 and json.loads(out)["added"] == 4
    code, out, _ = run_cli(capsys, ws, "status")
    assert code == 0 and '"total": 4' in out and "AI labels by review state: none yet" in out
    code, out, _ = run_cli(capsys, ws, "doctor")
    assert code == 0 and "FAIL" not in out and "16 tools exposed" in out


def test_manifest_registration(capsys, ws, tmp_path, inbox):
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"doc_type,front,back,name\nLIC,{inbox / 'LIC_04.png'},,my_license\n", encoding="utf-8")
    code, out, _ = run_cli(capsys, ws, "add", "--manifest", str(manifest))
    assert code == 0 and "added LIC-" in out


def test_token_and_http_needs_a_token(capsys, ws):
    code, out, _ = run_cli(capsys, ws, "token", "--save")
    token = out.strip().splitlines()[-1]
    assert len(token) >= 40 and (ws / "alg_mcp_token.txt").read_text().strip() == token
    code, _, err = run_cli(capsys, ws.parent / "other", "serve", "--transport", "http")
    assert code == 2 and "needs a token" in err
    code, _, err = run_cli(capsys, ws, "serve", "--transport", "http", "--no-auth", "--host", "0.0.0.0")
    assert code == 2 and "127.0.0.1" in err


def test_review_sheet_approve_reject_and_promote(capsys, ws, inbox):
    svc = new_service(ws)
    svc.add_paths([str(inbox)])
    label_everything(svc)
    cfg = Config.from_env(workspace=ws)
    code, out, _ = run_cli(capsys, ws, "review", "--only", "auto_ok", "--n", "5")
    sheet = (ws / "ai_review.html").read_text("utf-8")
    assert code == 0 and "ID_01" in sheet and "data:image/jpeg;base64" in sheet and "TAX_03" in sheet
    code, out, _ = run_cli(capsys, ws, "promote", "--dry-run", "--auto-ok")
    res = json.loads(out)
    assert sorted(res["promoted"]) == ["ID_01", "TAX_03"] and not (cfg.human_dir / "ID_01.label.json").exists()
    run_cli(capsys, ws, "reject", "TAX_03")
    code, out, _ = run_cli(capsys, ws, "promote", "--auto-ok")
    assert json.loads(out)["promoted"] == ["ID_01"]                       # rejected labels never move
    assert (cfg.human_dir / "ID_01.label.json").exists() and (cfg.human_dir / "ID_01.label.ai.json").exists()
    assert not (cfg.human_dir / "TAX_03.label.json").exists()
    code, out, _ = run_cli(capsys, ws, "promote", "--auto-ok")
    assert json.loads(out)["conflicts_not_overwritten"] == ["ID_01"]       # the existing file wins
    run_cli(capsys, ws, "approve", "TAX_03")
    code, out, _ = run_cli(capsys, ws, "promote", "--approved")
    assert json.loads(out)["promoted"] == ["TAX_03"]
    code, _, err = run_cli(capsys, ws, "promote")
    assert code == 2 and "choose what to promote" in err


def test_compare_with_human_labels_matches_by_image(capsys, ws, inbox):
    svc = new_service(ws)
    svc.add_paths([str(inbox)])
    sid = svc.next_sample("agentA", name="ID_01")["sample_id"]
    ai_vals = {**ID_VALUES, "mother_name": "زينب"}                          # differs from the human
    ai_vals.pop("blood_group")
    svc.set_fields(sid, ai_vals, {"blood_group": "glare"}, agent="agentA")
    view_all(svc, sid, "agentA", ("front", "back"))
    svc.submit_label(sid, "agentA", confirm=True)
    # a human labels the same image with the desktop app's engine
    cfg = svc.cfg
    inbox_front, inbox_back = inbox / "ID_01_front.png", inbox / "ID_01_back.png"
    sess = LabelSession("ID", "ID_01", str(inbox_front), str(inbox_back), ws, labeler="human")
    for k, v in ID_VALUES.items():
        sess.set_value(k, v)
    sess.mark_unreadable("birth_date")                                      # human says unreadable, AI read it
    sess.generate(merge_width=cfg.merge_width, merge_gap=cfg.merge_gap, jpeg=cfg.jpeg, labeler="human")
    res = review.compare(cfg)
    assert res["images_matched"] == 1 and res["fields_compared"] == 19
    t = res["tally"]
    assert t["differ"] == 1 and t["ai_over_reads"] == 1 and t["ai_over_abstains"] == 1 and t["agree"] == 16
    assert res["dangerous_ai_over_reads"] == 1
    assert {e["field"] for e in res["examples"]} == {"mother_name", "birth_date"}
    code, out, _ = run_cli(capsys, ws, "compare")
    assert code == 0 and json.loads(out)["exact_agreement"] == round(16 / 19, 4)
