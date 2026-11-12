"""Check rfalib.printed_truth against what is really printed on the print sheets.

Every document on the sheets is read field by field with Tesseract (Arabic + Latin) and the reading is
compared with two candidates:

    printed   what rfalib.printed_truth says is printed in that field position
    naive     the CSV column with the same meaning (what one would label without knowing the print run)

For each field the report gives how often the reading is closer to one or the other and the mean
similarity, and it lists every document whose reading is far from the printed value, so a person can
look at exactly those. OCR is not perfect, so the numbers are evidence, not proof; the listed cases are
the ones to look at by eye.

    python tools/verify_printed_truth.py --dataset D:\\Desktop\\dataset --layouts "..\\..\\main\\GATE Block\\templates" --out verify

Needs poppler's ``pdfimages`` (keeps the drawing order of the sheets) and ``tesseract`` with the ``ara``
language on PATH.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfalib import printed_truth as PT  # noqa: E402
from rfalib.normalize import normalize  # noqa: E402

# layout.json side and template folder per document type and sheet image
SIDES = {"ID": [("national_id", "front"), ("national_id", "back")], "TAX": [("tax_card", "single")],
         "LIC": [("business_license", "single")]}
DIGITS = "0123456789/,"
MRZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
LATIN_FIELDS = {"national_id_number": DIGITS, "card_serial": MRZ, "issue_date": DIGITS, "expiry_date": DIGITS,
                "birth_date": DIGITS, "family_number": MRZ, "mrz_line1": MRZ, "mrz_line2": MRZ, "mrz_line3": MRZ,
                "taxpayer_number": DIGITS, "license_number": DIGITS, "valid_until": DIGITS,
                "blood_group": "ABO+-"}
# fields read from extra boxes (not in layout.json): (template side, box)
EXTRA_BOXES = {"ID": {"vertical_serial": ("front", [0.93, 0.78, 0.975, 0.99])}}
# regions that must be the same on every card (design, not data): (template side, box); the name box is
# the yardstick of a region that does change
STATIC_BOXES = {"ID": {"serial_prefix_A": ("front", [0.035, 0.888, 0.075, 0.952]),
                       "name (changes, for scale)": ("front", [0.52, 0.41, 0.775, 0.476])}}


def naive_values(doc_type: str, row: dict[str, str], printed: dict[str, str]) -> dict[str, str]:
    """The CSV column with the same meaning as each field (no knowledge of the print run)."""
    if doc_type == "ID":
        return {**printed, "grandfather_name": row["grandfather"], "surname": row["surname"],
                "maternal_grandfather_name": "", "vertical_serial": ""}
    if doc_type == "LIC":
        return {**printed, "business_purposes": row.get("activity", ""), "head_office": row["hq"]}
    return dict(printed)


def extract(pdf: Path, out: Path) -> list[Path]:
    if not shutil.which("pdfimages"):
        raise SystemExit("pdfimages (poppler) is needed: it keeps the order in which the sheets draw the images")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdfimages", "-j", str(pdf), str(out / "img")], check=True)
    return sorted(out.glob("img-*.jpg"))


def ocr(img: Image.Image, latin: str | None) -> str:
    big = ImageOps.grayscale(img).resize((img.width * 3, img.height * 3), Image.LANCZOS)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        big.save(fh.name)
        args = ["tesseract", fh.name, "stdout", "--psm", "7"]
        args += (["-l", "eng", "-c", f"tessedit_char_whitelist={latin}"] if latin else ["-l", "ara"])
        res = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    Path(fh.name).unlink(missing_ok=True)
    return " ".join(res.stdout.split())


def similarity(a: str, b: str) -> float:
    a, b = re.sub(r"\s+", "", normalize(a)), re.sub(r"\s+", "", normalize(b))
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def crop(img: Image.Image, box: list[float], pad: float = 0.004) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(max(0, x0 - pad) * w), int(max(0, y0 - pad) * h),
                     int(min(1, x1 + pad) * w), int(min(1, y1 + pad) * h)))


def _gray(im: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(im).resize(size), dtype=np.float32)


def group_consistency(crops: dict[str, list]) -> dict[str, dict]:
    """The same printed value must give the same pixels on every card (one renderer, one design), and a
    different value different pixels. Per field: the largest mean difference between two cards that
    should show the same text, and the smallest between two that should not."""
    out = {}
    for key, items in crops.items():
        size = items[0][1].size
        by_value: dict[str, list[np.ndarray]] = {}
        for _, im, value in items:
            by_value.setdefault(value, []).append(_gray(im, size))
        within = [float(np.abs(a - g[0]).mean()) for g in by_value.values() if len(g) > 1 for a in g[1:]]
        values = list(by_value)
        medians = [np.median(np.stack(by_value[v]), axis=0) for v in values]
        between = [float(np.abs(medians[i] - medians[j]).mean())
                   for i in range(len(medians)) for j in range(i + 1, len(medians))]
        # every card of a repeated value must look more like the other cards of its value than like any
        # card that carries a different value
        misplaced = []
        for i, v in enumerate(values):
            group = by_value[v]
            if len(group) < 2:
                continue
            for j, a in enumerate(group):
                rest = np.median(np.stack(group[:j] + group[j + 1:]), axis=0)
                own = float(np.abs(a - rest).mean())
                other = min((float(np.abs(a - m).mean()) for k, m in enumerate(medians) if k != i), default=1e9)
                if own >= other:
                    misplaced.append(v)
        mw = max(within) if within else 0.0
        mb = min(between) if between else None
        out[key] = {"distinct_values": len(by_value), "values_seen_twice": sum(len(g) > 1 for g in by_value.values()),
                    "max_within": round(mw, 2), "min_between": round(mb, 2) if mb is not None else None,
                    "misplaced_cards": len(misplaced), "ok": not misplaced}
    return out


def contact_sheets(crops: dict[str, list], readings: dict, folder: Path) -> None:
    """One HTML page per field: every card's crop next to the text the labels will carry, for the eye."""
    import base64
    import html
    import io
    folder.mkdir(parents=True, exist_ok=True)
    for key, items in crops.items():
        doc_type, field = key.split(".", 1)
        rows = []
        for rid, im, value in items:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            src = base64.b64encode(buf.getvalue()).decode()
            ocr_text = readings.get((doc_type, rid, field), readings.get((doc_type, rid, "capital_number"), ""))
            rows.append(f"<tr><td>{rid}</td><td><img src='data:image/png;base64,{src}' height='40'></td>"
                        f"<td dir='auto' class='v'>{html.escape(value)}</td><td dir='auto' class='o'>"
                        f"{html.escape(ocr_text)}</td></tr>")
        page = ("<html><head><meta charset='utf-8'><style>body{font-family:'Noto Naskh Arabic',sans-serif}"
                "td{padding:2px 8px;border-bottom:1px solid #ddd}.v{font-size:20px}.o{color:#888;font-size:12px}"
                f"</style></head><body><h3>{html.escape(key)}</h3><table>"
                "<tr><th>id</th><th>printed (sheet)</th><th>label text</th><th>ocr</th></tr>"
                + "".join(rows) + "</table></body></html>")
        (folder / f"{key}.html").write_text(page, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="folder with printables/ and Datasets/")
    ap.add_argument("--layouts", required=True, help="GATE Block templates folder (layout.json per document)")
    ap.add_argument("--out", required=True, help="report folder")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N identities (quick check)")
    a = ap.parse_args(argv)
    dataset, out = Path(a.dataset), Path(a.out)
    gt = PT.load_ground_truth(dataset / "printables")
    layouts = {d: json.loads((Path(a.layouts) / d / "layout.json").read_text(encoding="utf-8"))
               for d in ("national_id", "tax_card", "business_license")}
    jobs = []
    static: dict[str, list] = {}
    crops: dict[str, list] = {}                      # "DOC.field" -> [(identity, crop, printed)]
    for doc_type, rows in gt.items():
        images = extract(dataset / "printables" / PT.SHEET_FILES[doc_type], out / "sheets" / doc_type)
        per_doc = len(SIDES[doc_type])
        ids = sorted(rows)
        if len(images) != per_doc * len(ids):
            raise SystemExit(f"{doc_type}: {len(images)} images on the sheets for {len(ids)} identities")
        for n, rid in enumerate(ids[: a.limit or None]):
            truth = PT.truth_for(doc_type, rows[rid])
            naive = naive_values(doc_type, rows[rid], truth["printed"])
            for s, (tdir, side) in enumerate(SIDES[doc_type]):
                img = Image.open(images[n * per_doc + s]).convert("RGB")
                fields = layouts[tdir]["sides"][side]["fields"]
                boxes = {k: v["box"] for k, v in fields.items()}
                boxes.update({k: b for k, (sd, b) in EXTRA_BOXES.get(doc_type, {}).items() if sd == side})
                for key, (sd, box) in STATIC_BOXES.get(doc_type, {}).items():
                    if sd == side:
                        static.setdefault(key, []).append(crop(img, box, pad=0).convert("L"))
                for key, box in boxes.items():
                    expected = truth["printed"].get(key, "")
                    im = crop(img, box)
                    crops.setdefault(f"{doc_type}.{key}", []).append((rid, im, expected))
                    if key == "capital":            # digits and currency are read separately
                        number, _, currency = expected.partition(" ")
                        jobs.append((doc_type, rid, "capital_number", im, number, number))
                        jobs.append((doc_type, rid, "capital_currency", im, currency, currency))
                        continue
                    jobs.append((doc_type, rid, key, im, expected, naive.get(key, expected)))

    def run(job):
        doc_type, rid, key, im, printed, naive = job
        if key == "vertical_serial":
            im = im.rotate(90, expand=True)
        latin = LATIN_FIELDS.get(key) or (DIGITS if key in ("vertical_serial", "capital_number") else None)
        text = ocr(im, latin)
        if key == "capital_number":
            text = max(re.findall(r"[\d,]+", text) or [""], key=len)
        elif key == "capital_currency":
            text = "".join(re.findall(r"[\u0600-\u06FF.]", text))
        return {"doc_type": doc_type, "identity": rid, "field": key, "ocr": text, "printed": printed,
                "naive": naive, "sim_printed": round(similarity(text, printed), 3),
                "sim_naive": round(similarity(text, naive), 3)}

    with ThreadPoolExecutor(a.threads) as pool:
        results = list(pool.map(run, jobs))
    summary: dict[str, dict] = {}
    for r in results:
        s = summary.setdefault(f"{r['doc_type']}.{r['field']}", {"n": 0, "printed_closer": 0, "naive_closer": 0,
                                                               "no_reading": 0, "same_candidate": 0,
                                                               "mean_sim_printed": 0.0})
        s["n"] += 1
        s["mean_sim_printed"] += r["sim_printed"]
        if r["printed"] == r["naive"]:
            s["same_candidate"] += 1
        elif not r["ocr"]:
            s["no_reading"] += 1                    # OCR gave nothing: no evidence either way
        elif r["sim_printed"] > r["sim_naive"]:
            s["printed_closer"] += 1
        elif r["sim_printed"] < r["sim_naive"]:
            s["naive_closer"] += 1
        else:
            s["no_reading"] += 1
    for s in summary.values():
        s["mean_sim_printed"] = round(s["mean_sim_printed"] / s["n"], 3)
    static_report = {}
    for key, regions in static.items():
        arr = np.stack([np.asarray(c.resize(regions[0].size), dtype=np.float32) for c in regions])
        med = np.median(arr, axis=0)
        mad = np.abs(arr - med).mean(axis=(1, 2))
        static_report[key] = {"cards": len(regions), "max_mean_abs_diff": round(float(mad.max()), 2),
                              "median_mean_abs_diff": round(float(np.median(mad)), 2)}
    groups = group_consistency(crops)
    contact_sheets(crops, {(r["doc_type"], r["identity"], r["field"]): r["ocr"] for r in results}, out / "contact")
    worst = sorted((r for r in results if r["sim_printed"] < 0.75), key=lambda r: r["sim_printed"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "readings.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"fields": summary, "static_regions": static_report,
                                                  "same_value_same_print": groups,
                                                  "to_look_at": worst}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    for k, s in sorted(summary.items()):
        print(f"{k:34s} n={s['n']:3d} printed-closer={s['printed_closer']:3d} naive-closer={s['naive_closer']:3d}"
              f" no-reading={s['no_reading']:3d} same={s['same_candidate']:3d} mean-sim={s['mean_sim_printed']:.3f}")
    for k, g in sorted(groups.items()):
        if g["values_seen_twice"]:
            between = "n/a (one value)" if g["min_between"] is None else f"{g['min_between']:.2f}"
            print(f"{k:34s} {g['values_seen_twice']:2d} repeated value(s): same text differs by at most "
                  f"{g['max_within']:.2f}, different texts by at least {between}; cards closer to another "
                  f"value: {g['misplaced_cards']}: {'ok' if g['ok'] else 'CHECK'}")
    for k, v in static_report.items():
        print(f"static region {k}: {v}")
    print(f"{len(worst)} reading(s) far from the printed value: see {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
