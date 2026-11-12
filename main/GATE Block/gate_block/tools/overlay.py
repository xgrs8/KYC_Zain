"""Draw what the gate saw: the document outline it found and every field box, coloured by status.

    python -m gate_block overlay PHOTO --out overlay.png

Use it to check a new template (do the boxes sit on the fields?) and to understand a decision.
Green = good, yellow = fair, red = poor or outside, grey = unknown. Critical fields have a thick border.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..imageio import load_image
from ..quality import analyze_side
from ..recognition import recognize

_COLOURS = {"good": (60, 170, 60), "fair": (0, 200, 230), "poor": (40, 40, 230), "outside": (40, 40, 230), "unknown": (150, 150, 150)}


def side_overlay(gate, image, title: str = "") -> np.ndarray:
    """Overlay for one photographed side (BGR)."""
    loaded = load_image(image, gate.cfg.image)
    rec = recognize(loaded, gate.pack, gate.cfg)
    sq = analyze_side(loaded, rec, gate.pack, gate.cfg, keep_rectified=True)
    photo = loaded.work.copy()
    quad = np.array(sq.geometry["corners_work_px"], np.float32)
    cv2.polylines(photo, [np.round(quad).astype(np.int32)], True, (255, 0, 255), max(2, photo.shape[1] // 400))
    rect = sq.rectified.copy()
    if sq.reference == "template" and rec.best is not None:
        t = rec.best.template
        for name, fq in sq.fields.items():
            x0, y0, x1, y1 = t.field_box_px(name, gate.cfg.quality.field_pad)
            colour = _COLOURS.get(fq.status, _COLOURS["unknown"])
            cv2.rectangle(rect, (x0, y0), (x1, y1), colour, 3 if fq.critical else 1)
            label = name if fq.status == "good" else f"{name}: {fq.cause} {fq.severity}"
            cv2.putText(rect, label, (x0 + 3, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    worst = [f"{k}={m.severity}" for k, m in sq.metrics.items() if m.severity not in ("none", "unknown")]
    header = f"{title}  {rec.status}: {rec.key or '-'} ({rec.confidence:.2f})  {' '.join(worst) or 'no defects'}"
    h = 640
    left = cv2.resize(photo, (int(photo.shape[1] * h / photo.shape[0]), h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(rect, (int(rect.shape[1] * h / rect.shape[0]), h), interpolation=cv2.INTER_AREA)
    canvas = np.full((h + 34, left.shape[1] + right.shape[1] + 10, 3), 255, np.uint8)
    canvas[34:, :left.shape[1]] = left
    canvas[34:, left.shape[1] + 10:] = right
    cv2.putText(canvas, header[:160], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def make_overlay(gate, image, back=None) -> np.ndarray:
    """Overlay for a photo, or for the two uploads of an ID stacked on top of each other."""
    panels: List[np.ndarray] = [side_overlay(gate, image, "front" if back is not None else "")]
    if back is not None:
        panels.append(side_overlay(gate, back, "back"))
    width = max(p.shape[1] for p in panels)
    out = np.full((sum(p.shape[0] for p in panels), width, 3), 255, np.uint8)
    y = 0
    for p in panels:
        out[y:y + p.shape[0], :p.shape[1]] = p
        y += p.shape[0]
    return out


def write_overlay(gate, image, out_path, back=None) -> Optional[str]:
    img = make_overlay(gate, image, back)
    if not cv2.imwrite(str(out_path), img):
        raise OSError(f"could not write {out_path}")
    return str(out_path)
