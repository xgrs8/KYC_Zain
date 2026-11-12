"""The command line and the tools around the gate (calibration, Laya dataset export, evaluation, overlay)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from gate_block import StaticBackend, onehot
from gate_block.cli import main
from gate_block.config import GateConfig
from gate_block.laya_client import QUESTIONS
from gate_block.tools import calibrate, evaluate, export_laya_dataset
from gate_block.tools.overlay import make_overlay
from gate_block.tools.samples import iter_samples

from .conftest import make_photo

ROOT = Path(__file__).resolve().parent.parent
ID_DIR = ROOT / "templates" / "national_id"


def run_cli(capsys, *argv):
    code = main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture(scope="module")
def photos_dir(gate, tmp_path_factory) -> Path:
    """A small labelled folder: good/ blur/ shadow/ cut/ plus one ID pair and one merged ID."""
    root = tmp_path_factory.mktemp("photos")
    for sub in ("good", "blur", "shadow", "cut"):
        (root / sub).mkdir()
    cv2.imwrite(str(root / "good" / "TAX_01.jpg"), make_photo(gate, "tax_card/single"))
    cv2.imwrite(str(root / "good" / "LIC_01.jpg"), make_photo(gate, "business_license/single"))
    cv2.imwrite(str(root / "good" / "omar_front.jpg"), make_photo(gate, "national_id/front"))
    cv2.imwrite(str(root / "good" / "omar_back.jpg"), make_photo(gate, "national_id/back"))
    cv2.imwrite(str(root / "good" / "ID_23.png"), cv2.imread(str(ROOT / "tests" / "data" / "id_merge_reference.png")))
    cv2.imwrite(str(root / "blur" / "TAX_02.jpg"), make_photo(gate, "tax_card/single", blur_sigma=3.0))
    cv2.imwrite(str(root / "blur" / "TAX_03.jpg"), make_photo(gate, "tax_card/single", blur_sigma=6.0))
    cv2.imwrite(str(root / "shadow" / "TAX_04.jpg"), make_photo(gate, "tax_card/single", shadow=0.55))
    cv2.imwrite(str(root / "cut" / "TAX_05.jpg"), make_photo(gate, "tax_card/single", cut=0.10))
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    return root


# ---------------------------------------------------------------------- samples
def test_sample_discovery(photos_dir):
    samples = {s.name: s for s in iter_samples(photos_dir)}
    assert set(samples) == {"good/TAX_01", "good/LIC_01", "good/omar", "good/ID_23", "blur/TAX_02", "blur/TAX_03", "shadow/TAX_04", "cut/TAX_05"}
    assert samples["good/omar"].kind == "id_pair" and samples["good/omar"].back.name == "omar_back.jpg"
    assert samples["good/ID_23"].kind == "merged_id" and samples["good/TAX_01"].expected == "tax_card"
    assert samples["good/LIC_01"].expected == "business_license" and samples["blur/TAX_02"].label == "blur"
    with pytest.raises(FileNotFoundError):
        list(iter_samples(photos_dir / "nope"))


# ---------------------------------------------------------------------- CLI
def test_cli_check_prints_json_with_arabic(capsys, photos_dir, tmp_path):
    out_file = tmp_path / "res" / "result.json"
    code, out, _ = run_cli(capsys, "check", photos_dir / "blur" / "TAX_03.jpg", "--expect", "tax_card", "--out", out_file)
    data = json.loads(out)
    assert code == 0 and data["status_code"] == "reject:blur" and "report" not in data
    assert data["message_ar"] == "الصورة مغوشة من فضلك ارجع التقط الصورة بدون تغويش" and "مغوشة" in out   # readable, not \\u escapes
    assert "report" in json.loads(out_file.read_text(encoding="utf-8"))            # the file is always complete
    code, out, _ = run_cli(capsys, "check", photos_dir / "good" / "TAX_01.jpg", "--full")
    assert code == 0 and json.loads(out)["report"]["schema_version"] == 1


def test_cli_id_merged_id_and_submission(capsys, photos_dir, tmp_path):
    code, out, _ = run_cli(capsys, "id", ID_DIR / "front.jpg", ID_DIR / "back.jpg", "--workdir", tmp_path, "--name", "omar")
    data = json.loads(out)
    assert code == 0 and data["status_code"] == "accept" and Path(data["image_path"]).name == "omar_merged.png"
    code, out, _ = run_cli(capsys, "merged-id", data["image_path"])
    assert code == 0 and json.loads(out)["status_code"] == "accept"
    code, out, _ = run_cli(capsys, "submission", "--id-front", ID_DIR / "front.jpg", "--id-back", ID_DIR / "back.jpg",
                           "--tax-card", photos_dir / "shadow" / "TAX_04.jpg", "--workdir", tmp_path)
    data = json.loads(out)
    assert code == 0 and data["overall"] == "enhance" and data["documents"]["tax_card"]["status_code"] == "enhance:deshadowing"
    code, _, err = run_cli(capsys, "submission")
    assert code == 2 and "at least one" in err


def test_cli_recheck_round_trip(capsys, photos_dir, tmp_path):
    first = tmp_path / "first.json"
    code, _, _ = run_cli(capsys, "check", photos_dir / "shadow" / "TAX_04.jpg", "--expect", "tax_card", "--out", first)
    assert code == 0
    code, out, _ = run_cli(capsys, "recheck", photos_dir / "good" / "TAX_01.jpg", "--previous", first)
    data = json.loads(out)
    assert code == 0 and data["status_code"] == "accept" and data["applied_ops"] == ["deshadowing"]
    code, _, err = run_cli(capsys, "recheck", photos_dir / "good" / "TAX_01.jpg", "--previous", tmp_path / "missing.json")
    assert code == 2 and "error" in err


def test_cli_tree_lists_the_tree_messages_and_questions(capsys):
    code, out, _ = run_cli(capsys, "tree")
    data = json.loads(out)
    assert code == 0 and data["laya_questions"] == QUESTIONS
    assert data["tree"]["docres_tasks_by_priority"][0] == "dewarping" and "blur" in data["messages_ar"]


def test_cli_doctor_passes_on_the_shipped_package(capsys):
    code, out, _ = run_cli(capsys, "doctor")
    assert code == 0 and "0 failed" in out and "[FAIL]" not in out
    assert "pixel for pixel" in out and "reject:no_document" in out


def test_cli_doctor_reports_a_broken_installation(capsys, tmp_path):
    (tmp_path / "rules.yaml").write_text("enhance_at: sometimes\n", encoding="utf-8")
    code, out, _ = run_cli(capsys, "doctor", "--config", tmp_path)
    assert code == 1 and "[FAIL]" in out


def test_cli_doctor_with_a_model_that_cannot_be_loaded(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "doctor", "--laya", tmp_path / "no_checkpoint")
    assert code == 1 and "[FAIL] Laya answers the gate's questions" in out


def test_cli_reports_errors_without_a_traceback(capsys, tmp_path):
    code, out, err = run_cli(capsys, "check", "photo.jpg", "--templates", tmp_path / "nope")
    assert code == 2 and "TemplateError" in err and "Traceback" not in err
    code, out, err = run_cli(capsys, "check", "photo.jpg", "--laya", tmp_path / "nope")
    assert code == 2 and "BackendError" in err


def test_module_entry_point_and_utf8_output():
    done = subprocess.run([sys.executable, "-m", "gate_block", "--version"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.startswith("gate_block ")
    done = subprocess.run([sys.executable, "-m", "gate_block", "tree"], cwd=ROOT, capture_output=True,
                          env={"PYTHONIOENCODING": "cp1252", "PATH": "", "SYSTEMROOT": ""})   # a legacy Windows console
    assert done.returncode == 0 and "مغوشة" in done.stdout.decode("utf-8")


def test_cli_overlay(capsys, photos_dir, tmp_path):
    out_file = tmp_path / "overlay.png"
    code, out, _ = run_cli(capsys, "overlay", photos_dir / "shadow" / "TAX_04.jpg", "--out", out_file)
    assert code == 0 and cv2.imread(str(out_file)) is not None


def test_overlay_for_an_id_pair(gate):
    img = make_overlay(gate, ID_DIR / "front.jpg", back=ID_DIR / "back.jpg")
    assert img.ndim == 3 and img.shape[0] > 1000


# ---------------------------------------------------------------------- calibration
def test_calibrate_writes_measurements_and_a_suggestion(gate, photos_dir, tmp_path):
    out = calibrate.run(gate, photos_dir, tmp_path, echo=lambda *_: None)
    rows = list(csv.DictReader(open(tmp_path / "measurements.csv", encoding="utf-8")))
    assert len(rows) == 10                                                          # 6 single photos + 2 two-sided IDs
    by_sample = {(r["sample"], r["side"]): r for r in rows}
    assert by_sample[("blur/TAX_03", "single")]["status_code"] == "reject:blur"
    assert float(by_sample[("blur/TAX_03", "single")]["blur"]) > float(by_sample[("good/TAX_01", "single")]["blur"])
    assert by_sample[("good/omar", "front")]["route"] == "accept" and by_sample[("good/ID_23", "back")]["reference"] == "template"
    assert (tmp_path / "suggested_thresholds.yaml").is_file()
    assert out["suggestion"]["n_good_sides"] == 6 and any("at least 20" in n for n in out["suggestion"]["notes"])


def test_calibration_moves_thresholds_only_outwards():
    cfg = GateConfig.from_dict(None)

    def rows(blur, dark):
        base = {"label": "good", "reference": "template", "low_contrast": 1.0, "shadow": 0.05, "glare": 0.0,
                "low_resolution": 14.0, "curl": 0.0, "noise": 0.5}
        return [dict(base, blur=blur * (1 + 0.005 * i), dark=dark) for i in range(30)]

    relaxed = calibrate.suggest(rows(blur=0.05, dark=1.0), cfg)
    assert relaxed["quality"] == {} and relaxed["notes"] == []                      # good photos already pass: nothing to change
    soft_camera = calibrate.suggest(rows(blur=0.115, dark=1.0), cfg)
    mild, moderate, severe = soft_camera["quality"]["blur_ratio"]
    assert mild > cfg.quality.blur_ratio[0] and mild < moderate < severe            # good photos stop reading as blurry
    dim_room = calibrate.suggest(rows(blur=0.05, dark=0.8), cfg)
    assert dim_room["quality"]["exposure_ref"] == pytest.approx(0.8)
    GateConfig.from_dict({"quality": dict(soft_camera["quality"])})                 # the suggestion is a valid config


# ---------------------------------------------------------------------- Laya dataset export
def test_export_rows_are_what_laya_train_reads(gate, photos_dir, tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("sample,decision,reason,task\nblur/TAX_02,reject,blur,\ngood/TAX_01,accept,,\ncut/TAX_05,accept,,\nghost,accept,,\n",
                      encoding="utf-8")
    summary = export_laya_dataset.run(gate, photos_dir, tmp_path / "data", labels_csv=labels, eval_fraction=0.3, echo=lambda *_: None)
    rows = [json.loads(line) for name in ("train.jsonl", "eval.jsonl")
            for line in (tmp_path / "data" / name).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and summary["train"]["rows"] + summary["eval"]["rows"] == 8
    by_sample = {r["meta"]["sample"]: r for r in rows}
    for row in rows:
        assert set(row) == {"state", "questions", "expected", "meta"} and row["questions"] == QUESTIONS
        assert isinstance(row["state"], str) and json.loads(row["state"])["doc"]
        for qid, label in row["expected"].items():
            assert label in QUESTIONS[qid]["criteria"]
        assert set(row["expected"]) <= {"decision", "reject_reason", "docres_task"}
    assert by_sample["blur/TAX_02"]["expected"] == {"decision": "reject", "reject_reason": "blur"}       # your label wins
    assert by_sample["blur/TAX_02"]["meta"]["label_source"] == "labels"
    assert by_sample["shadow/TAX_04"]["expected"] == {"decision": "enhance", "docres_task": "deshadowing"}   # the rules' label
    assert by_sample["blur/TAX_03"]["expected"] == {"decision": "reject", "reject_reason": "blur"}       # safety-rule rows are kept
    assert by_sample["good/omar"]["expected"] == {"decision": "accept"}
    assert summary["labels_not_matched"] == ["ghost"]
    assert summary["labels_contradicting_a_safety_rule"] == ["cut/TAX_05"]


def test_export_split_is_by_document_and_stable(gate, tmp_path):
    a = export_laya_dataset.run(gate, None, tmp_path / "a", simulate=12, seed=4, echo=lambda *_: None)
    b = export_laya_dataset.run(gate, None, tmp_path / "b", simulate=12, seed=4, echo=lambda *_: None)
    assert (tmp_path / "a" / "train.jsonl").read_text(encoding="utf-8") == (tmp_path / "b" / "train.jsonl").read_text(encoding="utf-8")
    assert a == b
    assert export_laya_dataset._group_of("omar__blur3", None) == "omar" == export_laya_dataset._group_of("omar__dark", None)
    assert export_laya_dataset._group_of("ID_23_blur", r"(ID_\d+)") == "ID_23"
    assert export_laya_dataset._is_eval("omar", 0.5, 0) == export_laya_dataset._is_eval("omar", 0.5, 0)


def test_export_refuses_a_gate_with_a_model_and_bad_labels(gate, photos_dir, tmp_path):
    from gate_block import Gate
    with pytest.raises(ValueError):
        export_laya_dataset.run(Gate(backend=StaticBackend(onehot("accept"))), photos_dir, tmp_path)
    for text in ("sample,decision,reason,task\nx,maybe,,\n", "sample,decision,reason,task\nx,reject,,\n",
                 "sample,decision,reason,task\nx,enhance,,sharpen\n", "sample,decision,reason,task\n,accept,,\n"):
        bad = tmp_path / "bad.csv"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            export_laya_dataset.read_labels(bad)


def test_unrecognised_photos_are_not_training_rows(gate):
    from gate_block import synth
    res = gate.process_image(synth.make_negative("other_card", 4))
    row, why = export_laya_dataset.row_from_result(res)
    assert row is None and why in ("laya_not_asked", "stopped_before_quality")


# ---------------------------------------------------------------------- evaluation
def test_evaluate_compares_rules_laya_and_both(photos_dir, tmp_path):
    from gate_block import Gate
    labels = tmp_path / "labels.csv"
    labels.write_text("sample,decision,reason,task\ngood/TAX_01,accept,,\ngood/LIC_01,accept,,\ngood/omar,accept,,\ngood/ID_23,accept,,\n"
                      "blur/TAX_02,enhance,,deblurring\nblur/TAX_03,reject,blur,\nshadow/TAX_04,enhance,,deshadowing\ncut/TAX_05,reject,cut_off,\n",
                      encoding="utf-8")
    always_accept = Gate(backend=StaticBackend(onehot("accept", confidence=0.99)))
    out = evaluate.run(always_accept, photos_dir, labels_csv=labels, out_path=tmp_path / "eval.json", echo=lambda *_: None)
    s = out["summary"]
    assert s["photos"] == 8 and s["labelled"] == 8 and s["errors"] == 0 and s["laya_connected"]
    assert s["rules"]["decision_accuracy"] == 1.0 and s["rules"]["false_acceptance_rate"] == 0.0
    assert s["laya_raw"]["decision_accuracy"] == 0.5 and s["laya_raw"]["false_acceptance_rate"] == 1.0   # the model alone is useless
    assert s["combined"]["false_acceptance_rate"] <= 0.25                           # ... the gate around it is not
    assert s["combined"]["confusion"]["reject->reject"] == 2
    assert s["combined"]["docres_task_accuracy"] == 1.0 and s["combined"]["review_rate"] == 0.0
    assert s["laya_agrees_with_rules"] == 0.5 and "accept_contradicted_by_measurements" in s["laya_overridden"]
    assert s["combined"]["accuracy_by_folder"]["good"] == {"photos": 4, "accuracy": 1.0}
    assert json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))["photos"][0]["sample"]


def test_evaluate_without_labels_or_model_says_so(gate, tmp_path):
    out = evaluate.run(gate, None, simulate=6, echo=lambda *_: None)
    s = out["summary"]
    assert s["label_sources"] == {"rules": 6} and not s["laya_connected"] and s["laya_raw"] == {"photos": 0}
    assert s["rules"]["decision_accuracy"] == 1.0 and any("by construction" in n for n in s["notes"])
    assert any("no Laya backend" in n for n in s["notes"])


def test_cli_tools_commands(capsys, photos_dir, tmp_path):
    code, out, _ = run_cli(capsys, "calibrate", photos_dir, "--out", tmp_path / "cal")
    assert code == 0 and (tmp_path / "cal" / "measurements.csv").is_file() and "How the current thresholds route" in out
    code, out, _ = run_cli(capsys, "export-laya", photos_dir, "--out", tmp_path / "data")
    assert code == 0 and (tmp_path / "data" / "train.jsonl").is_file()
    code, _, err = run_cli(capsys, "export-laya", "--out", tmp_path / "data")
    assert code == 2 and "photos folder" in err
    code, out, _ = run_cli(capsys, "evaluate", photos_dir, "--out", tmp_path / "ev.json")
    assert code == 0 and (tmp_path / "ev.json").is_file()
