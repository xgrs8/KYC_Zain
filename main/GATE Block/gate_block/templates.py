"""Reference templates: "how the ID, the tax card and the licence look".

Folder layout::

    templates/
      national_id/        front.jpg  back.jpg  layout.json
      tax_card/           single.jpg           layout.json
      business_license/   single.jpg           layout.json

* ``front`` / ``back`` / ``single`` (or ``reference``) images are the recognition templates.
* ``layout.json`` describes the document: ``rigid`` (a plastic card cannot curl, paper can) and, per side,
  the ``card`` region inside the template image (to leave out a canvas margin or drop shadow), the
  ``fields`` (normalised boxes relative to the template image, each with a ``critical`` flag) and optional
  ``ignore`` boxes (areas whose look changes between documents or with the viewing angle: a portrait, a
  hologram). See ``docs/TEMPLATES.md``.

Everything the gate needs from a template is computed once, here, when the pack is loaded: the rectified
canonical image, ORB keypoints of the *static* content only, a low-frequency layout signature, the
template's own blur curve, its content mask and so on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import GateConfig
from .measures import (
    blur_curve,
    content_mask,
    contrast_range,
    motion_rulers,
    noise_sigma,
    skin_mask,
    text_band_height,
    to_gray,
    white_mask,
)
from .types import SIDES

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
SIDE_NAMES = {"front": "front", "back": "back", "single": "single", "reference": "single"}
THUMB_LONG = 96
Box = Tuple[float, float, float, float]


class TemplateError(Exception):
    """Template folder missing, empty or inconsistent."""


@dataclass
class FieldSpec:
    name: str
    box: Box                 # normalised, relative to the card (canonical) frame
    critical: bool = False


@dataclass
class Template:
    doc_type: str
    side: str
    rigid: bool
    size: Tuple[int, int]                    # canonical (width, height)
    bgr: np.ndarray                          # canonical card image
    gray: np.ndarray                         # float32 canonical grey
    fields: Dict[str, FieldSpec]
    ignore_boxes: List[Box]
    # recognition signature
    thumb_size: Tuple[int, int]
    thumb_gray: np.ndarray                   # locally normalised grey thumbnail
    thumb_edge: np.ndarray                   # blurred gradient magnitude thumbnail
    thumb_weight: np.ndarray                 # 1 on static content, 0 inside field / ignore boxes
    chroma_hist: np.ndarray                  # Lab a/b histogram of the static content
    has_chroma: bool
    kp_pts: np.ndarray                       # (N, 2) static ORB keypoints, canonical px
    des: np.ndarray                          # (N, 32) uint8 descriptors
    kp_hull_frac: float                      # share of the card spanned by the static keypoints
    # quality references
    text_mask: Optional[np.ndarray]          # bool mask of the field boxes (None: no fields, use everything)
    blur_sigmas: np.ndarray
    blur_curve: np.ndarray                   # the template's own blur "ruler", measured on the text regions
    content: np.ndarray                      # uint8 mask of printed content
    white: np.ndarray                        # uint8 mask of near-white pixels
    contrast: float
    noise: float                             # the template's own fine texture, read as "noise"
    skin: np.ndarray                         # bool mask of the template's own skin-coloured print (grown a little)
    field_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    motion_rulers: Dict[int, np.ndarray] = field(default_factory=dict)   # the motion-blur "ruler" (text regions)
    text_height: float = 20.0                # height of the smallest printed field text, canonical px
    field_text_height: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.doc_type}/{self.side}"

    @property
    def aspect(self) -> float:
        return self.size[0] / float(self.size[1])

    def variable_boxes(self) -> List[Box]:
        return [f.box for f in self.fields.values()] + list(self.ignore_boxes)

    def field_box_px(self, name: str, pad: float = 0.0) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.fields[name].box
        w, h = self.size
        return (max(0, int((x0 - pad) * w)), max(0, int((y0 - pad) * h)),
                min(w, int(np.ceil((x1 + pad) * w))), min(h, int(np.ceil((y1 + pad) * h))))


@dataclass
class TemplatePack:
    templates: List[Template]
    directory: Optional[Path] = None

    def get(self, doc_type: str, side: str) -> Optional[Template]:
        for t in self.templates:
            if t.doc_type == doc_type and t.side == side:
                return t
        return None

    def keys(self) -> List[str]:
        return [t.key for t in self.templates]

    def aspects(self) -> List[float]:
        return sorted({round(max(t.aspect, 1.0 / t.aspect), 4) for t in self.templates})

    def doc_types(self) -> List[str]:
        return sorted({t.doc_type for t in self.templates})

    @classmethod
    def load(cls, directory, cfg: GateConfig) -> "TemplatePack":
        d = Path(directory)
        if not d.is_dir():
            raise TemplateError(f"templates directory not found: {d} (see docs/TEMPLATES.md)")
        templates: List[Template] = []
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            layout_path = sub / "layout.json"
            layout = read_layout(layout_path) if layout_path.is_file() else {"sides": {}}
            seen = set()
            for img_path in sorted(sub.iterdir()):
                stem = img_path.stem.lower()
                if img_path.suffix.lower() not in IMAGE_EXT or stem not in SIDE_NAMES:
                    continue
                side = SIDE_NAMES[stem]
                if side in seen:
                    raise TemplateError(f"{sub}: more than one image for side {side!r}")
                seen.add(side)
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise TemplateError(f"cannot read template image {img_path}")
                side_layout = layout.get("sides", {}).get(side, {})
                templates.append(build_template(sub.name, side, bgr, side_layout, cfg, rigid=bool(layout.get("rigid", True))))
            unused = set(layout.get("sides", {})) - seen
            if unused:
                raise TemplateError(f"{layout_path}: layout for side(s) {sorted(unused)} but no such image in {sub}")
        if not templates:
            raise TemplateError(f"no template images in {d}; expected e.g. {d}/national_id/front.jpg (see docs/TEMPLATES.md)")
        return cls(templates=templates, directory=d)


# ---------------------------------------------------------------------- layout files
def _check_box(box, where: str) -> Box:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise TemplateError(f"{where}: a box is [x0, y0, x1, y1]")
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise TemplateError(f"{where}: box values must be numbers") from exc
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise TemplateError(f"{where}: need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {list(box)}")
    return (x0, y0, x1, y1)


def read_layout(path: Path) -> dict:
    """Read and validate a ``layout.json``; raises :class:`TemplateError` with the exact place of a mistake."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemplateError(f"cannot read layout file {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("sides", {}), dict):
        raise TemplateError(f"{path}: expected an object with a 'sides' object")
    if "rigid" in data and not isinstance(data["rigid"], bool):
        raise TemplateError(f"{path}: 'rigid' must be true or false")
    for side, sd in data.get("sides", {}).items():
        if side not in SIDES:
            raise TemplateError(f"{path}: unknown side {side!r} (use front, back or single)")
        if not isinstance(sd, dict):
            raise TemplateError(f"{path}:{side} must be an object")
        card = (0.0, 0.0, 1.0, 1.0)
        if "card" in sd:
            card = _check_box(sd["card"], f"{path}:{side}.card")
            if (card[2] - card[0]) < 0.5 or (card[3] - card[1]) < 0.5:
                raise TemplateError(f"{path}:{side}.card covers less than half of the image")
        fields = sd.get("fields", {})
        if not isinstance(fields, dict):
            raise TemplateError(f"{path}:{side}.fields must be an object of name -> {{box, critical}}")
        for name, spec in fields.items():
            if not isinstance(spec, dict):
                raise TemplateError(f"{path}:{side}.fields.{name} must be an object with a 'box'")
            box = _check_box(spec.get("box"), f"{path}:{side}.fields.{name}")
            if "critical" in spec and not isinstance(spec["critical"], bool):
                raise TemplateError(f"{path}:{side}.fields.{name}.critical must be true or false")
            if box[2] <= card[0] or box[0] >= card[2] or box[3] <= card[1] or box[1] >= card[3]:
                raise TemplateError(f"{path}:{side}.fields.{name} lies outside the card region")
        ignore = sd.get("ignore", [])
        if not isinstance(ignore, list):
            raise TemplateError(f"{path}:{side}.ignore must be a list of boxes")
        for i, b in enumerate(ignore):
            _check_box(b, f"{path}:{side}.ignore[{i}]")
    return data


# ---------------------------------------------------------------------- signatures
def _to_card_frame(box, card) -> Box:
    cx0, cy0, cx1, cy1 = card
    return (min(1.0, max(0.0, (box[0] - cx0) / (cx1 - cx0))), min(1.0, max(0.0, (box[1] - cy0) / (cy1 - cy0))),
            min(1.0, max(0.0, (box[2] - cx0) / (cx1 - cx0))), min(1.0, max(0.0, (box[3] - cy0) / (cy1 - cy0))))


def local_normalise(g: np.ndarray, sigma: float) -> np.ndarray:
    g = g.astype(np.float32)
    mu = cv2.GaussianBlur(g, (0, 0), sigma)
    var = cv2.GaussianBlur((g - mu) ** 2, (0, 0), sigma)
    return (g - mu) / np.sqrt(var + 25.0)


def thumbnail_signature(bgr: np.ndarray, thumb_size: Tuple[int, int]):
    """(locally normalised grey, blurred gradient magnitude, Lab a/b) of ``bgr`` at thumbnail size."""
    small = cv2.resize(bgr, thumb_size, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)
    return local_normalise(g, 3.0), edge, cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 1:3]


def chroma_histogram(ab: np.ndarray, weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """Normalised 12 x 12 histogram of Lab a/b, and whether the image is coloured at all."""
    a = ab[:, :, 0].astype(np.float32)
    b = ab[:, :, 1].astype(np.float32)
    w = np.ones(a.shape, np.float32) if weight is None else weight.astype(np.float32)
    hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=(12, 12), range=((0, 256), (0, 256)), weights=w.ravel())
    total = float(hist.sum())
    spread = float(np.sqrt(np.average((a - 128) ** 2 + (b - 128) ** 2, weights=w + 1e-9)))
    return (hist / total if total > 0 else hist), spread > 4.0


def preprocess_for_orb(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8)


def make_orb(n: int):
    return cv2.ORB_create(nfeatures=int(n), scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=10)


def _in_any(nx: float, ny: float, boxes: List[Box]) -> bool:
    return any(b[0] <= nx <= b[2] and b[1] <= ny <= b[3] for b in boxes)


def build_template(doc_type: str, side: str, bgr: np.ndarray, side_layout: dict, cfg: GateConfig,
                   rigid: bool = True) -> Template:
    h0, w0 = bgr.shape[:2]
    card = tuple(float(v) for v in side_layout.get("card", (0.0, 0.0, 1.0, 1.0)))
    cx0, cy0, cx1, cy1 = card
    crop = bgr[int(round(cy0 * h0)):int(round(cy1 * h0)), int(round(cx0 * w0)):int(round(cx1 * w0))]
    ch, cw = crop.shape[:2]
    scale = cfg.image.canonical_long_side / float(max(cw, ch))
    size = (max(8, int(round(cw * scale))), max(8, int(round(ch * scale))))
    canon = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    gray = to_gray(canon)

    fields = {name: FieldSpec(name=name, box=_to_card_frame(spec["box"], card), critical=bool(spec.get("critical", False)))
              for name, spec in side_layout.get("fields", {}).items()}
    ignore = [_to_card_frame(b, card) for b in side_layout.get("ignore", [])]
    variable = [f.box for f in fields.values()] + ignore

    # low-frequency layout signature; variable content is given zero weight
    ts = THUMB_LONG / float(max(size))
    thumb_size = (max(16, int(round(size[0] * ts))), max(16, int(round(size[1] * ts))))
    gn, edge, ab = thumbnail_signature(canon, thumb_size)
    weight = np.ones((thumb_size[1], thumb_size[0]), np.float32)
    for (x0, y0, x1, y1) in variable:
        weight[int(y0 * thumb_size[1]):int(np.ceil(y1 * thumb_size[1])), int(x0 * thumb_size[0]):int(np.ceil(x1 * thumb_size[0]))] = 0.0
    chroma, has_chroma = chroma_histogram(ab, weight)

    # ORB keypoints of the static content only, strongest first
    g8 = preprocess_for_orb(np.clip(gray, 0, 255).astype(np.uint8))
    kps, des = make_orb(cfg.recognition.orb_template_features).detectAndCompute(g8, None)
    pts, descs = [], []
    if des is not None:
        for i in sorted(range(len(kps)), key=lambda j: -kps[j].response):
            if _in_any(kps[i].pt[0] / size[0], kps[i].pt[1] / size[1], variable):
                continue
            pts.append(kps[i].pt)
            descs.append(des[i])
            if len(pts) >= cfg.recognition.orb_static_max:
                break
    kp_pts = np.asarray(pts, np.float32).reshape(-1, 2)
    des_arr = np.asarray(descs, np.uint8).reshape(-1, 32)
    hull_frac = 0.02
    if len(kp_pts) >= 3:
        hull = cv2.convexHull(kp_pts.reshape(-1, 1, 2))
        hull_frac = max(0.02, float(cv2.contourArea(hull)) / float(size[0] * size[1]))

    # Blur is judged where the text to be read is: the union of the field boxes. Printed labels and
    # backgrounds can be soft by design, which would make a whole-document curve flat and useless.
    text_mask: Optional[np.ndarray] = None
    if fields:
        text_mask = np.zeros((size[1], size[0]), bool)
        for (x0, y0, x1, y1) in (f.box for f in fields.values()):
            text_mask[int(y0 * size[1]):int(np.ceil(y1 * size[1])), int(x0 * size[0]):int(np.ceil(x1 * size[0]))] = True
    sigmas, curve = blur_curve(gray, mask=text_mask)
    tpl = Template(
        doc_type=doc_type, side=side, rigid=rigid, size=size, bgr=canon, gray=gray, fields=fields, ignore_boxes=ignore,
        thumb_size=thumb_size, thumb_gray=gn, thumb_edge=edge, thumb_weight=weight, chroma_hist=chroma,
        has_chroma=has_chroma, kp_pts=kp_pts, des=des_arr, kp_hull_frac=hull_frac, text_mask=text_mask, blur_sigmas=sigmas,
        blur_curve=curve, content=content_mask(gray), white=white_mask(canon), contrast=contrast_range(gray),
        noise=noise_sigma(gray),
        skin=cv2.dilate(skin_mask(canon).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool),
    )
    tpl.motion_rulers = motion_rulers(gray, text_mask)
    # each field is judged against its own softness, measured on exactly the box the gate measures later
    for name in fields:
        x0, y0, x1, y1 = tpl.field_box_px(name, cfg.quality.field_pad)
        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] >= 12 and patch.shape[1] >= 12:
            tpl.field_curves[name] = blur_curve(patch)
        band = text_band_height(patch)
        if band is not None and band >= 6.0:
            tpl.field_text_height[name] = band
    # How big the text to be read is: blur and resolution are judged against it. The smaller printed
    # fields set it (a low quantile rather than the minimum, so one odd box cannot decide it alone).
    if tpl.field_text_height:
        tpl.text_height = float(np.percentile(list(tpl.field_text_height.values()), 25))
    else:
        tpl.text_height = float(cfg.quality.default_text_height)
    return tpl
