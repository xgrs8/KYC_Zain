"""A.L.G entry point.

  python alg/main.py             -> GUI
  python alg/main.py --selftest  -> headless selftest (M10): renders the
      Arabic probe, computes the MRZ probe, verifies the merge gate on
      synthetic fixtures, writes NOTHING, exits 0 on success.
"""
from __future__ import annotations

import sys
from pathlib import Path

# bootstrap the app root so `python rfg/main.py` (or the frozen exe) finds
# both the app package and the vendored rfalib (self-contained folder, 5.1)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def selftest() -> int:
    """Headless selftest - nothing is written anywhere (plan 10.5)."""
    from PIL import Image

    from rfalib import merge, mrz, render, schema
    from rfalib.normalize import normalize, same

    print("[selftest] Arabic render probe...")
    img = Image.new("RGB", (600, 120), "white")
    render.draw_text(img, (20, 20, 560, 80), "بطاقة الهوية الوطنية",
                     font="NotoNaskhArabic-Regular.ttf", size=48,
                     color="#111111", align="right")
    assert img.convert("L").getextrema()[0] < 128, "Arabic glyphs did not render"
    print(f"        backend={render.render_backend()}")

    print("[selftest] MRZ probe...")
    lines = mrz.build_td1(
        card_serial="BB7654321", national_id_number="987654321098",
        birth_date="1990/08/21", sex="أنثى", expiry_date="2030/08/20",
        surname="العبيدي", name="زينب", father_name="حسين",
        grandfather_name="علي")
    assert all(len(x) == 30 for x in lines)
    assert mrz.verify_line_checks(lines)
    print(f"        L2={lines[1]}")

    print("[selftest] label gate probe...")
    fields = {k: "" for k in schema.label_keys("TAX")}
    problems = []
    try:
        schema.build_label("TAX", "TAX_01.png", fields)
    except ValueError as exc:
        problems.append(str(exc))
    assert problems, "empty fields must block the label"
    print(f"        empty gate blocks: {problems[0][:60]}...")

    print("[selftest] merge identity probe (ALG side)...")
    front = Image.new("RGB", (1140, 723), (200, 210, 220))
    back = Image.new("RGB", (1140, 723), (210, 215, 225))
    merged = merge.merge_images(front, back, width=1140, gap=24)
    assert merged.size == (1140, 723 + 24 + 723)
    print(f"        merged size={merged.size}")

    print("[selftest] truth comparison probe...")
    assert same("علي", "علي ") and not same("علي", "على")
    assert normalize("مُحَمَّد") == normalize("محمد")
    print("SELFTEST OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import ALGWindow
    from .ui.theme import apply_aero_style
    app = QApplication(sys.argv)
    apply_aero_style(app)
    win = ALGWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
