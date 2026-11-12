"""A camera simulator and non-document images, for the test-suite and the demo.

``variant_from_template`` turns a reference template into "another person's document" (new random text in
every field box, static content untouched) and ``capture`` photographs it with controllable defects: blur,
motion, shadow, glare, darkness, perspective, curl, cut-off, a finger, noise, JPEG.

It exists so that the Gate Block can be exercised without a single real photo. It does NOT replace
calibration on real photos: simulated blur is cleaner than a real lens, and so on.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = {
    (False, False): ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf", "arial.ttf"],
    (True, False): ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
    (False, True): ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf", "/Library/Fonts/Courier New.ttf", "DejaVuSansMono.ttf", "consola.ttf"],
    (True, True): ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "C:/Windows/Fonts/consolab.ttf", "/Library/Fonts/Courier New Bold.ttf", "DejaVuSansMono-Bold.ttf", "consolab.ttf"],
}
_font_cache: Dict[Tuple[int, bool, bool], ImageFont.ImageFont] = {}
_NAMES = ["KARRAR", "HASSAN", "ALI", "ZAINAB", "MOHAMMED", "FATIMA", "OMAR", "NOOR", "HUSSEIN", "SARA",
          "AHMED", "LAYLA", "YOUSIF", "MARYAM", "IBRAHIM", "RAWAN", "SAJAD", "HIBA", "MUSTAFA", "DANA"]


def _font(size: int, bold: bool = False, mono: bool = False):
    key = (size, bold, mono)
    if key not in _font_cache:
        font = None
        for path in _FONT_CANDIDATES[(bold, mono)]:
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size)
            except TypeError:  # very old Pillow
                font = ImageFont.load_default()
        _font_cache[key] = font
    return _font_cache[key]


# ------------------------------------------------------------------------------ documents
def variant_from_template(canon_bgr: np.ndarray, fields: Dict[str, Tuple[float, float, float, float]], seed: int) -> np.ndarray:
    """Another instance of a reference template: the dark text inside every field box is painted out and
    new random text is drawn there. Emblems, labels, frames and patterns stay as they are."""
    r = random.Random(seed)
    h, w = canon_bgr.shape[:2]
    gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), np.uint8)
    for (x0, y0, x1, y1) in fields.values():
        xa, ya, xb, yb = int(x0 * w), int(y0 * h), int(np.ceil(x1 * w)), int(np.ceil(y1 * h))
        patch = gray[ya:yb, xa:xb]
        if patch.size == 0:
            continue
        ink = (patch < float(np.percentile(patch, 50)) - 28.0).astype(np.uint8)
        mask[ya:yb, xa:xb] = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    base = cv2.inpaint(canon_bgr, mask, 4, cv2.INPAINT_TELEA)
    pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for name, (x0, y0, x1, y1) in fields.items():
        xa, ya, xb, yb = int(x0 * w), int(y0 * h), int(np.ceil(x1 * w)), int(np.ceil(y1 * h))
        bh = yb - ya
        mono = name.startswith("mrz")
        size = max(12, int(bh * (0.7 if mono else 0.62)))
        kind = "mrz" if mono else r.choice(["digits", "latin", "arabic"])
        if kind == "digits":
            text = "".join(r.choice("0123456789") for _ in range(r.randint(6, 13)))
        elif kind == "latin":
            text = " ".join(r.choice(_NAMES) for _ in range(r.randint(1, 2)))
        elif kind == "arabic":
            text = "".join(r.choice("ابتثجحخدذرزسشصضطظعغفقكلمنهوي ") for _ in range(r.randint(5, 12)))
        else:
            text = "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<") for _ in range(30))
        font = _font(size, True, mono)
        while d.textlength(text, font=font) > (xb - xa) - 8 and size > 10:
            size -= 2
            font = _font(size, True, mono)
        d.text((xb - 6 - d.textlength(text, font=font), ya + (bh - size) * 0.35), text, font=font, fill=(55, 55, 58))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_other_card(seed: int) -> np.ndarray:
    """A card-shaped document with a different design: it must NOT be recognised as a supported one."""
    r = random.Random(seed)
    w, h = 1000, 630
    img = Image.new("RGB", (w, h), (r.randint(200, 250), r.randint(200, 250), r.randint(200, 250)))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = r.randint(0, 700), r.randint(0, 450)
        d.rectangle((x0, y0, x0 + r.randint(120, 380), y0 + r.randint(40, 160)),
                    fill=(r.randint(30, 200), r.randint(30, 200), r.randint(30, 200)))
    for i in range(9):
        text = "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789 ") for _ in range(r.randint(10, 24)))
        d.text((r.randint(20, 600), 40 + i * 62), text, font=_font(r.randint(22, 40), bool(r.getrandbits(1))), fill=(20, 20, 20))
    d.ellipse((r.randint(600, 800), r.randint(300, 450), 950, 600), outline=(160, 20, 20), width=6)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------------------ camera
@dataclass
class CaptureParams:
    frame: Tuple[int, int] = (1600, 1200)        # (width, height) of the photo
    doc_frac: float = 0.72                       # long side of the document as a share of the matching frame side
    rot_deg: float = 0.0
    persp: float = 0.0                           # corner jitter as a share of the document size
    blur_sigma: float = 0.0                      # extra Gaussian blur, frame pixels
    motion_len: int = 0                          # motion-blur length, frame pixels
    motion_angle: float = 0.0
    exposure: float = 1.0
    gamma: float = 1.0                           # tone curve: > 1 lifts the mid-tones, < 1 darkens them
    contrast: float = 1.0                        # > 1 a contrasty camera, < 1 a flat one (around mid-grey)
    flare: float = 0.0                           # 0..0.3 veiling light: blacks are lifted towards grey
    saturation: float = 1.0                      # > 1 the vivid colours phones like
    noise: float = 0.0                           # Gaussian noise sigma, grey levels
    shadow: float = 0.0                          # 0..1 darkness of a soft-edged band
    shadow_angle: float = 35.0
    shadow_width: float = 0.35                   # band width as a share of the frame diagonal
    glare: float = 0.0                           # 0..1
    glare_xy: Tuple[float, float] = (0.0, 0.0)   # offset from the document centre, share of the document size
    glare_size: float = 0.16                     # blob sigma as a share of the document long side
    finger: bool = False
    finger_xy: Tuple[float, float] = (0.30, 0.34)  # finger centre offset, share of the document size
    cut: float = 0.0                             # share of the document width pushed outside the frame
    jpeg: int = 92
    curl: float = 0.0                            # bow amplitude as a share of the document height
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # colour of the light as B, G, R gains
    bg: str = "wood"                             # wood | dark | plain | clutter | cloth
    base_softness: float = 0.7                   # softness every camera has
    seed: int = 0


def _background(kind: str, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "wood":
        stripes = 0.5 + 0.5 * np.sin(xx / 9.0 + 3.0 * np.sin(yy / 160.0) + rng.uniform(0, 6))
        bg = np.array([60, 95, 140], np.float32)[None, None, :] * (0.65 + 0.5 * stripes[:, :, None])
    elif kind == "dark":
        bg = np.full((h, w, 3), 38, np.float32) + 10 * np.sin(xx / 40.0)[:, :, None]
    elif kind == "plain":
        bg = np.full((h, w, 3), rng.uniform(110, 190), np.float32)
    elif kind == "clutter":
        bg = np.full((h, w, 3), 120, np.float32)
        for _ in range(60):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x1, y1 = x0 + int(rng.integers(30, 160)), y0 + int(rng.integers(20, 120))
            cv2.rectangle(bg, (x0, y0), (x1, y1), [int(c) for c in rng.integers(30, 230, 3)], -1)
            cv2.rectangle(bg, (x0, y0), (x1, y1), (20, 20, 20), 2)
    else:  # cloth
        n = cv2.resize(rng.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        bg = np.stack([150 + 25 * n, 140 + 25 * n, 170 + 25 * n], -1)
    return np.clip(bg + rng.normal(0, 3.0, (h, w, 1)).astype(np.float32), 0, 255)


def _bow(doc: np.ndarray, amp: float):
    """Bend the top and bottom edges like curled paper. Returns (image, alpha, source quad)."""
    h, w = doc.shape[:2]
    pad = int(math.ceil(abs(amp))) + 2
    yy, xx = np.mgrid[0:h + 2 * pad, 0:w].astype(np.float32)
    src_y = yy - pad - amp * np.sin(math.pi * xx / (w - 1))
    out = cv2.remap(doc, xx, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.remap(np.ones((h, w), np.float32), xx, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    quad = np.array([[0, pad], [w - 1, pad], [w - 1, h + pad - 1], [0, h + pad - 1]], np.float32)
    return out, alpha, quad


def capture(template_bgr: np.ndarray, p: CaptureParams):
    """Simulate a phone photo of ``template_bgr``. Returns (photo BGR uint8, meta dict)."""
    rng = np.random.default_rng(p.seed)
    fw, fh = p.frame
    th, tw = template_bgr.shape[:2]
    if abs(p.curl) > 1e-6:
        doc, alpha, src_quad = _bow(template_bgr, p.curl * th)
    else:
        doc, alpha = template_bgr, np.ones((th, tw), np.float32)
        src_quad = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    s = p.doc_frac * (fw if tw >= th else fh) / float(max(tw, th))
    dw, dh = tw * s, th * s
    c = np.array([[-dw / 2, -dh / 2], [dw / 2, -dh / 2], [dw / 2, dh / 2], [-dw / 2, dh / 2]], np.float32)
    ang = math.radians(p.rot_deg)
    c = c @ np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], np.float32).T
    if p.persp > 0:
        c = c + rng.uniform(-1, 1, c.shape).astype(np.float32) * np.array([dw, dh], np.float32) * p.persp
    dst = c + np.array([fw / 2.0, fh / 2.0], np.float32)
    if p.cut > 0:
        dst[:, 0] += (fw + p.cut * dw) - dst[:, 0].max()
    hm = cv2.getPerspectiveTransform(src_quad, dst.astype(np.float32))
    warped = cv2.warpPerspective(doc, hm, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = cv2.warpPerspective(alpha, hm, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = _background(p.bg, (fw, fh), rng) * (1 - wa[:, :, None]) + warped.astype(np.float32) * wa[:, :, None]

    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    gdir = rng.uniform(0, 2 * math.pi)                         # gentle lighting gradient and vignette
    grad = ((xx - fw / 2) * math.cos(gdir) + (yy - fh / 2) * math.sin(gdir)) / math.hypot(fw, fh)
    light = 1.0 + 0.24 * rng.uniform(-1, 1) * grad
    light *= 1.0 - 0.20 * (((xx - fw / 2) / fw) ** 2 + ((yy - fh / 2) / fh) ** 2)
    out *= light[:, :, None]
    center = dst.mean(axis=0)
    doc_long = max(dw, dh)
    if p.shadow > 0:
        a = math.radians(p.shadow_angle)
        proj = (xx - center[0]) * math.cos(a) + (yy - center[1]) * math.sin(a)
        t0 = rng.uniform(-0.15, 0.15) * math.hypot(dw, dh)
        band, edge = p.shadow_width * math.hypot(fw, fh), 0.04 * math.hypot(fw, fh)
        m = np.clip((proj - t0) / edge, 0, 1) * np.clip((t0 + band - proj) / edge, 0, 1)
        out *= (1 - p.shadow * m)[:, :, None]
    if p.glare > 0:
        gx, gy = center[0] + p.glare_xy[0] * dw, center[1] + p.glare_xy[1] * dh
        sg = p.glare_size * doc_long
        out += (p.glare * 420.0 * np.exp(-(((xx - gx) ** 2) + ((yy - gy) ** 2)) / (2 * sg * sg)))[:, :, None]
    if p.finger:
        fx, fy = int(center[0] + p.finger_xy[0] * dw), int(center[1] + p.finger_xy[1] * dh)
        axes = (int(0.16 * dw), int(0.30 * dh))
        layer = out.copy()
        cv2.ellipse(layer, (fx, fy), axes, 28, 0, 360, (105, 150, 205), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 2.0)
        mask = np.zeros((fh, fw), np.float32)
        cv2.ellipse(mask, (fx, fy), axes, 28, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)[:, :, None]
        out = out * (1 - mask) + layer * mask
    out = np.clip(out * np.asarray(p.tint, np.float32)[None, None, :], 0, 255)
    out = 255.0 * np.power(np.clip(out / 255.0 * p.exposure, 0, 1), 1.0 / max(p.gamma, 1e-3))
    if p.contrast != 1.0:
        out = np.clip((out - 127.5) * p.contrast + 127.5, 0, 255)
    if p.flare > 0:
        out = out * (1.0 - p.flare) + 255.0 * p.flare
    if p.saturation != 1.0:
        grey = out.mean(axis=2, keepdims=True)
        out = np.clip(grey + (out - grey) * p.saturation, 0, 255)
    sigma = math.sqrt(p.base_softness ** 2 + p.blur_sigma ** 2)
    if sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    if p.motion_len > 1:
        k = np.zeros((p.motion_len, p.motion_len), np.float32)
        cv2.line(k, (0, p.motion_len // 2), (p.motion_len - 1, p.motion_len // 2), 1.0, 1)
        rot = cv2.getRotationMatrix2D((p.motion_len / 2 - 0.5, p.motion_len / 2 - 0.5), p.motion_angle, 1.0)
        k = cv2.warpAffine(k, rot, (p.motion_len, p.motion_len))
        out = cv2.filter2D(out, -1, k / max(float(k.sum()), 1e-6))
    if p.noise > 0:
        out = out + rng.normal(0, p.noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(p.jpeg)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), {"dst_quad": dst.tolist(), "doc_px": (dw, dh), "frame": (fw, fh)}


# ------------------------------------------------------------------------------ non-documents
def _fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    acc = np.zeros((h, w, 3), np.float32)
    amp = total = 0.0
    amp = 1.0
    for o in range(octaves):
        gh, gw = max(2, h // (2 ** (6 - o))), max(2, w // (2 ** (6 - o)))
        acc += amp * cv2.resize(rng.normal(0, 1, (gh, gw, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.55
    acc /= total
    return (acc - acc.min()) / max(1e-6, float(acc.max() - acc.min())) * 255.0


NEGATIVE_KINDS = ("noise", "gradient", "shapes", "natural", "screenshot", "text_page", "blank_white", "blank_dark",
                  "other_card", "stripes")


def make_negative(kind: str, seed: int, size: Tuple[int, int] = (1600, 1200)) -> np.ndarray:
    """An image that is not a supported document."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "noise":
        return rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    if kind == "gradient":
        a = rng.uniform(0, 2 * math.pi)
        g = xx * math.cos(a) + yy * math.sin(a)
        g = (g - g.min()) / (g.max() - g.min())
        return np.stack([g * rng.uniform(60, 255), g * rng.uniform(60, 255), (1 - g) * rng.uniform(60, 255)], -1).astype(np.uint8)
    if kind == "shapes":
        img = _fractal_noise(h, w, rng, 3).astype(np.uint8)
        for _ in range(14):
            col = [int(c) for c in rng.integers(0, 255, 3)]
            if rng.random() < 0.5:
                cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(30, 250)), col, -1)
            else:
                x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
                cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(60, 500)), y0 + int(rng.integers(60, 400))), col, -1)
        return img
    if kind == "natural":
        img = _fractal_noise(h, w, rng, 6) * np.array([0.7, 1.0, 0.6], np.float32)
        horizon = int(h * rng.uniform(0.35, 0.6))
        img[:horizon] = img[:horizon] * 0.4 + np.array([200, 170, 120], np.float32) * 0.6
        img = np.clip(img + cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 1.1) * 38.0, 0, 255).astype(np.uint8)
        for _ in range(900):                                   # grass-like strokes: sharp natural texture
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(horizon, h))
            cv2.line(img, (x0, y0), (x0 + int(rng.integers(-12, 12)), y0 - int(rng.integers(10, 40))),
                     [int(c) for c in rng.integers(20, 200, 3)], 1, cv2.LINE_AA)
        return img
    if kind == "screenshot":
        img = np.full((h, w, 3), 245, np.uint8)
        cv2.rectangle(img, (0, 0), (w, 70), (60, 60, 60), -1)
        for i in range(12):
            y = 110 + i * 85
            cv2.rectangle(img, (60, y), (w - 60, y + 56), (220, 220, 225), -1)
            cv2.putText(img, f"Item {i} lorem ipsum", (90, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        return img
    if kind == "text_page":
        pil = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(pil)
        r = random.Random(seed)
        y = 60
        while y < h - 60:
            line = " ".join("".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(2, 9))) for _ in range(r.randint(6, 12)))
            d.text((80, y), line, font=_font(30), fill=(25, 25, 25))
            y += 52
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if kind in ("blank_white", "blank_dark"):
        img = np.full((h, w, 3), 255 if kind == "blank_white" else 8, np.uint8)
        img[0, 0] = (250, 250, 250) if kind == "blank_white" else (12, 12, 12)
        return img
    if kind == "stripes":
        s = 0.5 + 0.5 * np.sin(xx / rng.uniform(8, 40) + yy / rng.uniform(30, 90))
        return np.stack([s * 200 + 30, s * 120 + 60, (1 - s) * 180 + 20], -1).astype(np.uint8)
    if kind == "other_card":
        return capture(render_other_card(seed), CaptureParams(frame=size, seed=seed, rot_deg=float(rng.uniform(-8, 8)), bg="wood"))[0]
    raise ValueError(f"unknown negative kind {kind!r}")
