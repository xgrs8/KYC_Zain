"""Prepare the photos of the printed documents for labeling in A.L.G (the app or the MCP server).

    python tools/prepare_photos.py --dataset D:\\Desktop\\dataset --inbox D:\\Desktop\\dataset\\alg_inbox ^
        --gate "..\\..\\main\\GATE Block"

The photos are expected where the photographers put them:

    Datasets/<split>/business_NNNN/<kind>/print_NNN/<condition>/<photo files>
        kind       id_card | tax_card | business_license
        condition  good, blur, glare, handheld ... (any folder name; recorded as the capture condition)

For every photo (for an ID: one front and one back photo of the same condition folder) the inbox gets

    <TYPE>_<n>_<condition>[_front|_back].<ext>   the photo (a hard link when possible, else a copy)
    <TYPE>_<n>.truth.json                         what is printed on that document, how the print run
                                                  departs from a real document, and where the photo came from

with n = identity x 100 + k (k = 1..99 numbers the photos of one document), so the name tells which identity
a photo shows (ID_1203: identity 012, its third ID photo) and A.L.G finds the truth through the name.

The front and the back of an ID are told apart by the file name (front / back in it) or, when the name does
not say, by the GATE Block's Gate 1 (``--gate``). Photos already prepared are found in ``manifest.csv`` and
skipped, so the command can be rerun as photos come in. Nothing in the dataset folder is changed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfalib import naming as NM  # noqa: E402
from rfalib import printed_truth as PT  # noqa: E402
from rfalib import sidecar as SC  # noqa: E402
from rfalib.constants import RFALIB_VERSION  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
ALG_EXT = {".jpg", ".jpeg", ".png"}                     # what A.L.G opens; anything else is converted to JPEG
KIND_TO_TYPE = {v: k for k, v in PT.KIND_DIRS.items()}
MANIFEST = "manifest.csv"
MANIFEST_COLUMNS = ["base", "doc_type", "identity_id", "split", "condition", "print", "front", "back",
                    "side_detection", "prepared_at"]
SideFn = Callable[[Path], tuple[str | None, float]]          # photo -> (front|back|None, confidence)


# ---------------------------------------------------------------- photos and sides
def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def side_from_name(path: Path) -> str | None:
    words = set(re.split(r"[^a-z]+", path.stem.lower()))
    if "front" in words and "back" not in words:
        return "front"
    if "back" in words and "front" not in words:
        return "back"
    return None


def gate_side_detector(gate_dir: Path) -> SideFn:
    """Gate 1 of the GATE Block tells the two sides of the ID apart (and that it is an ID at all)."""
    sys.path.insert(0, str(gate_dir))
    from gate_block import Gate
    from gate_block.imageio import load_image
    from gate_block.recognition import recognize

    gate = Gate(config_dir=gate_dir / "config")

    def detect(path: Path) -> tuple[str | None, float]:
        try:
            rec = recognize(load_image(str(path), gate.cfg.image), gate.pack, gate.cfg)
        except Exception:  # noqa: BLE001 - an unreadable photo is simply "unknown"
            return None, 0.0
        if rec.doc_type != "national_id" or rec.status == "unsupported":
            return None, float(rec.confidence)
        return rec.side, float(rec.confidence)

    return detect


def pair_id_photos(photos: list[Path], detect: SideFn | None) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    """[(front, back, how)] for the photos of one condition folder, and the problems for the rest."""
    sides: dict[Path, tuple[str | None, str]] = {}
    for p in photos:
        s = side_from_name(p)
        sides[p] = (s, "name") if s else (None, "")
    unknown = [p for p, (s, _) in sides.items() if s is None]
    if unknown and detect is not None:
        for p in unknown:
            s, conf = detect(p)
            if s in ("front", "back"):
                sides[p] = (s, f"gate1 {conf:.2f}")
        unknown = [p for p, (s, _) in sides.items() if s is None]
        # two photos, one side known: the other one is the other side
        if len(photos) == 2 and len(unknown) == 1:
            known = next(p for p in photos if p not in unknown)
            other = "back" if sides[known][0] == "front" else "front"
            sides[unknown[0]] = (other, "the other photo is the " + sides[known][0])
            unknown = []
    fronts = [p for p in photos if sides[p][0] == "front"]
    backs = [p for p in photos if sides[p][0] == "back"]
    problems = [f"{p}: cannot tell front from back (put 'front' or 'back' in the file name"
                + ("" if detect else ", or pass --gate") + ")" for p in unknown]
    if len(fronts) != len(backs):
        problems.append(f"{photos[0].parent}: {len(fronts)} front and {len(backs)} back photo(s); "
                        "every front needs one back")
    pairs = [(f, b, " / ".join(sorted({sides[f][1], sides[b][1]}))) for f, b in zip(fronts, backs)]
    return pairs, problems


# ---------------------------------------------------------------- manifest
def read_manifest(inbox: Path) -> list[dict[str, str]]:
    path = inbox / MANIFEST
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(inbox: Path, rows: list[dict[str, str]]) -> None:
    tmp = inbox / (MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(inbox / MANIFEST)


def inbox_suffix(src: Path) -> str:
    return src.suffix.lower() if src.suffix.lower() in ALG_EXT else ".jpg"


def place(src: Path, dst: Path) -> str:
    """Hard link (no extra disk space) or copy; a format A.L.G cannot open (HEIC, WebP ...) is converted to
    JPEG, upright. Never overwrites."""
    if dst.exists():
        raise FileExistsError(dst)
    if src.suffix.lower() not in ALG_EXT:
        from PIL import Image, ImageOps
        try:
            import pillow_heif  # optional: iPhone photos
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(src) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(dst, format="JPEG", quality=95)
        return "convert"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def safe_condition(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", name).strip("-").lower() or "photo"


# ---------------------------------------------------------------- truth sidecar of one photo
def truth_sidecar_for(base: str, doc_type: str, identity_id: str, row: dict[str, str], manifest_row: dict,
                      case_folder: str, print_dir: str, condition: str, files: dict[str, str],
                      side_detection: str) -> dict:
    t = PT.truth_for(doc_type, row)
    out = SC.truth_sidecar(doc_id=base, doc_type=doc_type, case_id=case_folder, batch=print_dir,
                           seed=manifest_row.get("seed", ""), doc_code=PT.caption(doc_type, identity_id),
                           split=manifest_row["split"], printed=t["printed"], intended=t["intended"],
                           defects=t["defects"], files=sorted(Path(v).name for v in files.values()))
    out["generator"] = PT.GENERATOR
    out["truth_builder"] = f"A.L.G rfalib.printed_truth (rfalib {RFALIB_VERSION})"
    out["source_of_truth"] = (f"printables/ground_truth/{PT.CSV_FILES[doc_type]}, identity {identity_id}, "
                              "mapped to the printed positions by rfalib/printed_truth.py")
    out["notes"] = t["notes"]
    out["known_deviations"] = t["known_deviations"]
    out["capture"] = {"condition": condition, "photos": files, "side_detection": side_detection or None}
    return out


# ---------------------------------------------------------------- main
def prepare(dataset: Path, inbox: Path, detect: SideFn | None = None, dry_run: bool = False,
            log: Callable[[str], None] = print) -> dict:
    gt = PT.load_ground_truth(dataset / "printables")
    manifest = PT.load_manifest(dataset / "Datasets")
    by_folder = {v["folder"]: (rid, v) for rid, v in manifest.items()}
    inbox.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(inbox)
    done = {r["front"] for r in rows} | {r["back"] for r in rows if r["back"]}
    used = {r["base"] for r in rows}
    added, problems, placed = [], [], {"link": 0, "copy": 0, "convert": 0}

    def next_base(doc_type: str, identity_id: str) -> str:
        for k in range(1, 100):
            base = NM.doc_name(doc_type, int(identity_id) * 100 + k)
            if base not in used and not (inbox / f"{base}.truth.json").exists():
                used.add(base)
                return base
        raise RuntimeError(f"more than 99 photos of {doc_type} {identity_id}")

    for split_dir in sorted(p for p in (dataset / "Datasets").iterdir() if p.is_dir()):
        for case in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if case.name not in by_folder:
                problems.append(f"{case}: not in split_manifest.csv")
                continue
            identity_id, mrow = by_folder[case.name]
            if PT.MANIFEST_SPLITS.get(split_dir.name.lower()) != mrow["split"]:
                problems.append(f"{case}: the manifest puts {case.name} in {mrow['split']}, not {split_dir.name}")
            for kind_dir in sorted(p for p in case.iterdir() if p.is_dir() and p.name in KIND_TO_TYPE):
                doc_type = KIND_TO_TYPE[kind_dir.name]
                row = gt[doc_type].get(identity_id)
                if row is None:
                    problems.append(f"{kind_dir}: identity {identity_id} is not in {PT.CSV_FILES[doc_type]}")
                    continue
                for print_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
                    for cond_dir in sorted(p for p in print_dir.iterdir() if p.is_dir()):
                        photos = [p for p in photos_in(cond_dir) if p.relative_to(dataset).as_posix() not in done]
                        if not photos:
                            continue
                        cond = safe_condition(cond_dir.name)
                        if doc_type == "ID":
                            pairs, probs = pair_id_photos(photos, detect)
                            problems += probs
                            items = [({"front": f, "back": b}, how) for f, b, how in pairs]
                        else:
                            items = [({"image": p}, "") for p in photos]
                        for files, how in items:
                            base = next_base(doc_type, identity_id)
                            names = {side: (f"{base}_{cond}_{side}{inbox_suffix(p)}" if side != "image"
                                            else f"{base}_{cond}{inbox_suffix(p)}") for side, p in files.items()}
                            rel = {side: p.relative_to(dataset).as_posix() for side, p in files.items()}
                            truth = truth_sidecar_for(base, doc_type, identity_id, row, mrow, case.name,
                                                      print_dir.name, cond_dir.name, rel, how)
                            if not dry_run:
                                made: list[Path] = []
                                try:
                                    for side, p in files.items():
                                        placed[place(p, inbox / names[side])] += 1
                                        made.append(inbox / names[side])
                                except OSError as exc:              # e.g. a HEIC photo without pillow-heif
                                    problems.append(f"{', '.join(rel.values())}: not placed in the inbox ({exc})")
                                    for f in made:                  # only what this attempt created
                                        f.unlink(missing_ok=True)
                                    used.discard(base)
                                    continue
                                SC.write_json(inbox / f"{base}.truth.json", truth)
                            rows.append({"base": base, "doc_type": doc_type, "identity_id": identity_id,
                                         "split": mrow["split"], "condition": cond_dir.name, "print": print_dir.name,
                                         "front": rel.get("front") or rel["image"], "back": rel.get("back", ""),
                                         "side_detection": how,
                                         "prepared_at": _dt.datetime.now().replace(microsecond=0).isoformat()})
                            added.append(base)
    if not dry_run:
        write_manifest(inbox, rows)
    log(f"{len(added)} new document photo(s) prepared in {inbox} ({placed['link']} linked, {placed['copy']} copied, "
        f"{placed['convert']} converted to JPEG)"
        + (" [dry run: nothing written]" if dry_run else ""))
    for p in problems:
        log("  problem: " + p)
    return {"added": added, "problems": problems, "placed": placed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, type=Path, help="folder with printables/ and Datasets/")
    ap.add_argument("--inbox", required=True, type=Path, help="where A.L.G will load the photos from")
    ap.add_argument("--gate", type=Path, help="GATE Block folder: tells ID fronts from backs when names do not")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be prepared")
    a = ap.parse_args(argv)
    detect = gate_side_detector(a.gate.resolve()) if a.gate else None
    res = prepare(a.dataset, a.inbox, detect, a.dry_run)
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
