"""Command line of the Gate Block.

    python -m gate_block check PHOTO [--expect tax_card]          one photo
    python -m gate_block id FRONT BACK --workdir out               national ID (merges the two sides)
    python -m gate_block merged-id ID_23.png                       an ID that is already merged
    python -m gate_block submission --id-front F --id-back B --tax-card T --business-license L
    python -m gate_block recheck ENHANCED --previous result.json   verify a DocRes output
    python -m gate_block doctor [--laya PATH]                      self-test of the installation
    python -m gate_block tree                                      the decision tree and the messages
    python -m gate_block overlay PHOTO --out overlay.png           what the gate saw
    python -m gate_block calibrate PHOTOS_DIR --out calibration    thresholds from your own photos
    python -m gate_block export-laya PHOTOS_DIR --out laya_data    Laya training rows from your photos
    python -m gate_block evaluate PHOTOS_DIR --labels labels.csv   accuracy of the rules, of Laya, and of both
    python -m gate_block laya-data --out laya_data                 training set that teaches Laya the gate's decisions
    python -m gate_block demo                                      simulated photos through the gate

Add ``--laya PATH_OR_HF_ID`` to let Laya decide; without it the deterministic rules decide.
Results are printed as JSON (UTF-8). ``--full`` adds the whole measurement report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):                    # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _build_gate(args):
    from .pipeline import Gate
    backend = None
    if getattr(args, "laya", None):
        from .laya_client import LayaBackend
        backend = LayaBackend(args.laya, device=args.device, calibration=args.laya_calibration)
    return Gate(templates_dir=args.templates, config_dir=args.config, backend=backend)


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _emit_result(res, args) -> None:
    """Print a gate result (short unless --full); the file written by --out is always complete."""
    if getattr(args, "out", None):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(_dump(res.to_dict()), encoding="utf-8")
    print(_dump(res.to_dict(include_report=bool(getattr(args, "full", False)))))


# ---------------------------------------------------------------------- commands
def cmd_check(args) -> int:
    gate = _build_gate(args)
    _emit_result(gate.process_image(args.image, expected=args.expect, workdir=args.workdir), args)
    return 0


def cmd_id(args) -> int:
    gate = _build_gate(args)
    _emit_result(gate.process_id(args.front, args.back, workdir=args.workdir, name=args.name), args)
    return 0


def cmd_merged_id(args) -> int:
    gate = _build_gate(args)
    _emit_result(gate.process_merged_id(args.image, workdir=args.workdir), args)
    return 0


def cmd_submission(args) -> int:
    uploads = {k: v for k, v in (("id_front", args.id_front), ("id_back", args.id_back), ("tax_card", args.tax_card),
                                 ("business_license", args.business_license)) if v}
    if not uploads:
        print("give at least one of --id-front/--id-back, --tax-card, --business-license", file=sys.stderr)
        return 2
    gate = _build_gate(args)
    out = gate.process_submission(uploads, workdir=args.workdir)
    docs = out["documents"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(_dump(dict(out, documents={k: v.to_dict() for k, v in docs.items()})), encoding="utf-8")
    print(_dump(dict(out, documents={k: v.to_dict(include_report=bool(args.full)) for k, v in docs.items()})))
    return 0


def cmd_recheck(args) -> int:
    from .pipeline import GateResult
    previous = GateResult.from_dict(json.loads(Path(args.previous).read_text(encoding="utf-8")))
    gate = _build_gate(args)
    _emit_result(gate.recheck(args.image, previous, workdir=args.workdir), args)
    return 0


def cmd_tree(args) -> int:
    from .config import GateConfig
    from .decision_tree import describe_tree, load_tree
    from .laya_client import QUESTIONS
    from .messages import load_messages
    cdir = Path(args.config) if args.config else GateConfig.default_dir()
    cdir = cdir if cdir.is_dir() else None
    text = _dump({"tree": describe_tree(load_tree(cdir)), "messages_ar": load_messages(cdir), "laya_questions": QUESTIONS})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_overlay(args) -> int:
    from .tools.overlay import write_overlay
    gate = _build_gate(args)
    print(write_overlay(gate, args.image, args.out, back=args.back))
    return 0


def cmd_calibrate(args) -> int:
    from .tools import calibrate
    calibrate.run(_build_gate(args), args.folder, args.out, good_label=args.good_label)
    return 0


def cmd_export_laya(args) -> int:
    from .tools import export_laya_dataset
    if args.laya:
        print("export-laya must run without --laya (the rows must not depend on the model being trained)", file=sys.stderr)
        return 2
    if not args.folder and not args.simulate:
        print("give a photos folder, --simulate N, or both", file=sys.stderr)
        return 2
    export_laya_dataset.run(_build_gate(args), args.folder, args.out, labels_csv=args.labels, eval_fraction=args.eval_fraction,
                            group_regex=args.group_regex, simulate=args.simulate, seed=args.seed)
    return 0


def cmd_evaluate(args) -> int:
    from .tools import evaluate
    if not args.folder and not args.simulate:
        print("give a photos folder, --simulate N, or both", file=sys.stderr)
        return 2
    evaluate.run(_build_gate(args), args.folder, labels_csv=args.labels, simulate=args.simulate, seed=args.seed, out_path=args.out)
    return 0


def cmd_laya_data(args) -> int:
    from .tools import laya_data
    if args.laya:
        print("laya-data must run without --laya (the labels must not depend on the model being trained)", file=sys.stderr)
        return 2
    laya_data.build_dataset(_build_gate(args), args.out, photos=args.photos, simulate=args.simulate,
                            synthetic=args.synthetic, eval_fraction=args.eval_fraction, seed=args.seed,
                            synthetic_enhance=args.synthetic_enhance)
    return 0


def _demo_cases(gate):
    from . import synth
    cases = [("clean photo", {}), ("slightly blurry", dict(blur_sigma=3.0)), ("very blurry", dict(blur_sigma=6.0)),
             ("dim room", dict(exposure=0.55)), ("shadow across", dict(shadow=0.55)), ("lamp reflection", dict(glare=0.9)),
             ("edge outside the frame", dict(cut=0.10)), ("finger on the document", dict(finger=True)), ("too far away", dict(doc_frac=0.2))]
    for t in gate.pack.templates:
        for name, kw in cases:
            params = dict(seed=7, rot_deg=4.0, persp=0.02, bg="wood", noise=1.0)
            if t.size[1] > t.size[0]:
                params["frame"] = (1200, 1600)
            params.update(kw)
            variant = synth.variant_from_template(t.bgr, {k: f.box for k, f in t.fields.items()}, 21)
            yield t, name, synth.capture(variant, synth.CaptureParams(**params))[0]


def cmd_demo(args) -> int:
    from . import synth
    gate = _build_gate(args)
    print(f"{'document':26s} {'photo':24s} {'outcome':24s} message")
    for t, name, photo in _demo_cases(gate):
        res = gate.process_image(photo, expected=t.doc_type)
        print(f"{t.key:26s} {name:24s} {res.status_code:24s} {res.message_ar or ''}")
    for kind in ("natural", "screenshot", "other_card"):
        res = gate.process_image(synth.make_negative(kind, 3))
        print(f"{'(not a document)':26s} {kind:24s} {res.status_code:24s} {res.message_ar or ''}")
    return 0


# ---------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    """Self-test: is the installation complete and does the gate behave?"""
    results: List[tuple] = []

    def check(name: str, fn) -> Any:
        try:
            detail = fn()
            results.append((True, name, "" if detail is None else str(detail)))
            return detail
        except Exception as exc:  # every check reports instead of stopping the run
            results.append((False, name, f"{type(exc).__name__}: {exc}"))
            return None

    import cv2
    import numpy as np
    import PIL
    import yaml
    print(f"gate_block {__version__} | python {sys.version.split()[0]} | opencv {cv2.__version__} | numpy {np.__version__} | "
          f"pillow {PIL.__version__} | pyyaml {yaml.__version__}")

    from .config import GateConfig
    from .decision_tree import load_tree
    from .laya_client import QUESTIONS, validate_questions
    from .messages import load_messages
    from .pipeline import Gate
    cdir = Path(args.config) if args.config else GateConfig.default_dir()
    def config_dir():
        if not cdir.is_dir():
            raise FileNotFoundError(f"{cdir} (the built-in defaults would be used)")
        return cdir
    check("config directory exists", config_dir)
    check("config files load", lambda: f"{len(GateConfig.load_dir(cdir).to_dict())} sections")
    check("Iraqi-Arabic messages are complete", lambda: f"{len(load_messages(cdir))} messages")
    check("decision tree is valid", lambda: ", ".join(load_tree(cdir).operation_order))
    check("Laya questions are valid", lambda: validate_questions(QUESTIONS) or f"{len(QUESTIONS)} questions")
    holder: Dict[str, Any] = {}

    def load_gate():
        holder["gate"] = Gate(templates_dir=args.templates, config_dir=args.config)
        return ", ".join(holder["gate"].pack.keys())
    check("templates load", load_gate)
    gate = holder.get("gate")
    if gate is None:
        return _doctor_report(results)
    for t in gate.pack.templates:
        def one(t=t):
            if len(t.kp_pts) < 100:
                raise ValueError(f"only {len(t.kp_pts)} static keypoints: use a sharper / larger reference image")
            if not t.fields:
                raise ValueError("no fields in layout.json: field-level checks are off for this template")
            return f"{len(t.kp_pts)} keypoints, {len(t.fields)} fields, {'rigid' if t.rigid else 'flexible'}"
        check(f"template {t.key}", one)

    def pristine():
        out = []
        for t in gate.pack.templates:
            res = gate.process_image(t.bgr, expected=t.doc_type)
            if res.route != "accept" or res.gate1["upload"]["template_used"] != t.key:
                raise AssertionError(f"{t.key} -> {res.status_code} (recognised as {res.gate1['upload']['template_used']})")
            out.append(t.key)
        return f"{len(out)} reference images recognised and accepted"
    check("each reference image is accepted as itself", pristine)

    def not_always_accept():
        from . import synth
        t = gate.pack.templates[0]
        variant = synth.variant_from_template(t.bgr, {k: f.box for k, f in t.fields.items()}, 5)
        frame = (1200, 1600) if t.size[1] > t.size[0] else (1600, 1200)
        blurry = gate.process_image(synth.capture(variant, synth.CaptureParams(frame=frame, blur_sigma=7.0, seed=1))[0])
        junk = gate.process_image(synth.make_negative("screenshot", 1))
        if blurry.route != "reject" or junk.status_code != "reject:no_document":
            raise AssertionError(f"very blurry -> {blurry.status_code}, screenshot -> {junk.status_code}")
        return f"very blurry -> {blurry.status_code}; screenshot -> {junk.status_code}"
    check("bad photos are rejected", not_always_accept)

    def merge_reference():
        ref = Path(gate.pack.directory).parent / "tests" / "data" / "id_merge_reference.png"
        front, back = gate.pack.directory / "national_id" / "front.jpg", gate.pack.directory / "national_id" / "back.jpg"
        if not (ref.is_file() and front.is_file() and back.is_file()):
            return "skipped (reference files not present)"
        from .id_merge import merge_id
        merged = merge_id(front, back, gate.cfg.id_merge).image
        expected = cv2.imread(str(ref))
        if merged.shape != expected.shape or not (merged == expected).all():
            raise AssertionError("the merge differs from the reference: check id_merge.yaml against A.L.G's merge_width")
        return "front.jpg + back.jpg = ID_23.png, pixel for pixel"
    check("ID merge matches the reference", merge_reference)

    if args.laya:
        def laya():
            from .laya_client import LayaBackend
            t0 = time.perf_counter()
            backend = LayaBackend(args.laya, device=args.device, calibration=args.laya_calibration)
            load_s = time.perf_counter() - t0
            # states as the gate builds them (with the permitted branches), and the rules' answer for each
            states = {
                "clean": ('{"doc":"tax_card","allowed":["accept"]}', "accept"),
                "severe blur": ('{"doc":"tax_card","blur":"severe","crit":{"taxpayer_number":"blur:severe"},'
                                '"oth":["blur:severe"],"allowed":["reject:blur"]}', "reject"),
                "moderate shadow": ('{"doc":"tax_card","shadow":"moderate",'
                                    '"allowed":["enhance:deshadowing","reject:shadow"]}', "enhance"),
            }
            lines, tops, agree = [], [], 0
            for name, (state, rules_answer) in states.items():
                t1 = time.perf_counter()
                ans = backend.decide(state, QUESTIONS)
                ms = (time.perf_counter() - t1) * 1000.0
                top = ans.top("decision")
                tops.append(top)
                agree += top == rules_answer
                lines.append(f"{name}: {top} ({ans.probs['decision'][top]:.2f}), {ms:.0f} ms")
            if len(set(tops)) == 1:
                verdict = ("SAME answer for every state: this checkpoint is not fine-tuned for the gate yet "
                           "(the router keeps it inside what the measurements allow)")
            else:
                verdict = f"agrees with the rules on {agree} of {len(states)} probe states"
            return f"loaded in {load_s:.1f}s; " + "; ".join(lines) + f" -> {verdict}"
        check("Laya answers the gate's questions", laya)
    else:
        results.append((True, "Laya", "not connected (no --laya): the deterministic rules decide"))
    return _doctor_report(results)


def _doctor_report(results) -> int:
    failed = 0
    for ok, name, detail in results:
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gate_block", description="GATE Block: document intake and quality gate.")
    p.add_argument("--version", action="version", version=f"gate_block {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config directory (default: config/ next to the package)")
    common.add_argument("--templates", help="templates directory (default: from config/paths.yaml)")
    common.add_argument("--laya", help="Laya checkpoint folder or Hugging Face id; without it the rules decide")
    common.add_argument("--laya-calibration", help="calibration file saved by Laya (optional)")
    common.add_argument("--device", help="device for Laya, e.g. cpu or cuda (default: Laya's choice)")
    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--workdir", help="folder for images the gate produces (the merged ID)")
    run.add_argument("--out", help="also write the complete JSON result (with the report) to this file")
    run.add_argument("--full", action="store_true", help="include the whole measurement report")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("check", parents=[common, run], help="gate one photo")
    s.add_argument("image")
    s.add_argument("--expect", choices=["national_id", "tax_card", "business_license"], help="the document this upload slot is for")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("id", parents=[common, run], help="gate a national ID (front and back uploads)")
    s.add_argument("front")
    s.add_argument("back")
    s.add_argument("--name", default="ID", help="base name of the merged image written to --workdir")
    s.set_defaults(func=cmd_id)

    s = sub.add_parser("merged-id", parents=[common, run], help="gate an ID that is already merged")
    s.add_argument("image")
    s.set_defaults(func=cmd_merged_id)

    s = sub.add_parser("submission", parents=[common, run], help="gate a whole submission")
    s.add_argument("--id-front")
    s.add_argument("--id-back")
    s.add_argument("--tax-card")
    s.add_argument("--business-license")
    s.set_defaults(func=cmd_submission)

    s = sub.add_parser("recheck", parents=[common, run], help="verify a DocRes output against the result that asked for it")
    s.add_argument("image", help="the enhanced image")
    s.add_argument("--previous", required=True, help="JSON result of the 'enhance' decision (the file written with --out)")
    s.set_defaults(func=cmd_recheck)

    s = sub.add_parser("doctor", parents=[common], help="self-test of the installation")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("tree", parents=[common], help="print the decision tree, the messages and Laya's questions")
    s.add_argument("--out")
    s.set_defaults(func=cmd_tree)

    s = sub.add_parser("overlay", parents=[common], help="draw the outline and field boxes the gate found")
    s.add_argument("image")
    s.add_argument("--back", help="the back upload, for a national ID")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_overlay)

    s = sub.add_parser("calibrate", parents=[common], help="suggest thresholds from your own photos")
    s.add_argument("folder")
    s.add_argument("--out", default="calibration")
    s.add_argument("--good-label", default="good", help="name of the folder that holds the good photos")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("export-laya", parents=[common], help="write Laya training rows from your photos")
    s.add_argument("folder", nargs="?", help="photos folder (optional when --simulate is used)")
    s.add_argument("--out", default="laya_data")
    s.add_argument("--labels", help="CSV with the columns sample, decision, reason, task")
    s.add_argument("--eval-fraction", type=float, default=0.15)
    s.add_argument("--group-regex", help="regex that extracts the document id from a file name (group 1)")
    s.add_argument("--simulate", type=int, default=0, help="add this many simulated photos labelled by the rules")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_export_laya)

    s = sub.add_parser("evaluate", parents=[common], help="compare the rules, Laya and both against labelled photos")
    s.add_argument("folder", nargs="?", help="photos folder (optional when --simulate is used)")
    s.add_argument("--labels", help="CSV with the columns sample, decision, reason, task")
    s.add_argument("--simulate", type=int, default=0, help="add this many simulated photos (labelled by the rules)")
    s.add_argument("--seed", type=int, default=1, help="seed of the simulated photos (use another one than for export-laya)")
    s.add_argument("--out", help="write the summary and the per-photo records to this JSON file")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("laya-data", parents=[common], help="write the training set that teaches Laya the gate's decisions")
    s.add_argument("--out", default="laya_data")
    s.add_argument("--photos", help="a folder of your photos (optional)")
    s.add_argument("--simulate", type=int, default=300, help="simulated photos of the templates")
    s.add_argument("--synthetic", type=int, default=4000, help="synthetic measurement states")
    s.add_argument("--synthetic-enhance", type=int, default=0,
                   help="extra synthetic states where two or more DocRes tasks are permitted (teaches the task choice)")
    s.add_argument("--eval-fraction", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_laya_data)

    s = sub.add_parser("demo", parents=[common], help="run simulated photos through the gate")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
