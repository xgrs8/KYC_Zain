"""Glossy colored 16x16 icons, drawn with QPainter (plan 9).

Classic Aero-era console icons: small, colorful, glossy - a vertical
gradient body, a light top highlight, a darker border and a simple white
glyph.  Drawn in code so the repo carries no binary icon assets; the
user-provided app icons (cube/ribbon .ico) are used for the window,
taskbar, exe and installer (10.6).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

GLYPH_WHITE = QColor(255, 255, 255, 235)

# base colors per icon family
COLORS = {
    "new": QColor(46, 141, 62),       # green
    "newdoc": QColor(66, 129, 196),   # blue document
    "stop": QColor(196, 66, 56),      # red
    "pdf": QColor(45, 111, 176),      # blue
    "folder": QColor(222, 174, 58),   # yellow folder
    "settings": QColor(120, 134, 152),  # gray gear
    "help": QColor(51, 113, 190),     # blue ?
    "calibrate": QColor(180, 92, 168),  # purple target
    "refresh": QColor(58, 130, 186),
    "image": QColor(88, 152, 116),
    "generate": QColor(46, 141, 62),
    "skip": QColor(204, 145, 46),
    "validate": QColor(58, 130, 186),
    "clear": QColor(180, 78, 78),
    "type": QColor(66, 129, 196),
}


def _glossy_pixmap(base: QColor, glyph_fn, size: int = 16) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    body = QRectF(1.0, 1.0, size - 2.0, size - 2.0)
    grad = QLinearGradient(body.topLeft(), body.bottomLeft())
    grad.setColorAt(0.0, base.lighter(135))
    grad.setColorAt(0.45, base)
    grad.setColorAt(1.0, base.darker(125))
    p.setPen(QPen(base.darker(165), 1.0))
    p.setBrush(QBrush(grad))
    p.drawRoundedRect(body, 3.0, 3.0)
    # top gloss highlight
    gloss = QLinearGradient(body.topLeft(), QPointF(body.center().x(), body.center().y()))
    gloss.setColorAt(0.0, QColor(255, 255, 255, 130))
    gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(gloss))
    p.drawRoundedRect(QRectF(body.left() + 1.0, body.top() + 1.0,
                             body.width() - 2.0, body.height() * 0.45), 2.5, 2.5)
    if glyph_fn:
        glyph_fn(p, size)
    p.end()
    return pm


# ------------------------------------------------------------------ glyphs

def _plus(p: QPainter, s: int) -> None:
    p.setPen(QPen(GLYPH_WHITE, 1.8))
    p.drawLine(s // 2, 4, s // 2, s - 4)
    p.drawLine(4, s // 2, s - 4, s // 2)


def _minus(p: QPainter, s: int) -> None:
    p.setPen(QPen(GLYPH_WHITE, 1.8))
    p.drawLine(4, s // 2, s - 4, s // 2)


def _square(p: QPainter, s: int) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(GLYPH_WHITE)
    p.drawRect(QRectF(s * 0.30, s * 0.30, s * 0.40, s * 0.40))


def _page(p: QPainter, s: int) -> None:
    path = QPainterPath(QPointF(s * 0.28, s * 0.18))
    path.lineTo(s * 0.62, s * 0.18)
    path.lineTo(s * 0.72, s * 0.30)
    path.lineTo(s * 0.72, s * 0.82)
    path.lineTo(s * 0.28, s * 0.82)
    path.closeSubpath()
    p.setPen(QPen(base_pen_color(), 0.8))
    p.setBrush(QBrush(QColor(252, 253, 255, 235)))
    p.drawPath(path)
    p.setPen(QPen(QColor(90, 110, 135), 1.0))
    for i in range(3):
        y = s * 0.42 + i * s * 0.13
        p.drawLine(QPointF(s * 0.34, y), QPointF(s * 0.66, y))


def _gear(p: QPainter, s: int) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(GLYPH_WHITE)
    c, r = QPointF(s / 2, s / 2), s * 0.30
    for k in range(8):
        ang = k * math.pi / 4
        tooth = QPointF(c.x() + (r + 2.4) * math.cos(ang),
                        c.y() + (r + 2.4) * math.sin(ang))
        p.drawEllipse(tooth, 1.1, 1.1)
    p.drawEllipse(c, r, r)
    p.setBrush(QColor(70, 84, 100, 220))
    p.drawEllipse(c, r * 0.42, r * 0.42)


def _question(p: QPainter, s: int) -> None:
    p.setPen(QPen(GLYPH_WHITE, 1.9))
    p.setFont(QFont("Segoe UI", 7, 700))
    p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "?")


def _target(p: QPainter, s: int) -> None:
    p.setPen(QPen(GLYPH_WHITE, 1.4))
    p.drawEllipse(QRectF(s * 0.26, s * 0.26, s * 0.48, s * 0.48))
    p.drawEllipse(QRectF(s * 0.40, s * 0.40, s * 0.20, s * 0.20))
    p.drawLine(s // 2, 2, s // 2, s * 0.28)
    p.drawLine(s // 2, s * 0.72, s // 2, s - 2)


def _folder(p: QPainter, s: int) -> None:
    p.setPen(QPen(QColor(150, 106, 20), 0.8))
    p.setBrush(QBrush(QColor(255, 244, 200, 235)))
    p.drawRect(QRectF(s * 0.14, s * 0.42, s * 0.72, s * 0.38))
    tab = QPainterPath(QPointF(s * 0.14, s * 0.42))
    tab.lineTo(s * 0.14, s * 0.30)
    tab.lineTo(s * 0.45, s * 0.30)
    tab.lineTo(s * 0.50, s * 0.42)
    tab.closeSubpath()
    p.drawPath(tab)


def _printer(p: QPainter, s: int) -> None:
    p.setPen(QPen(QColor(60, 80, 105), 0.8))
    p.setBrush(QBrush(QColor(250, 251, 253, 235)))
    p.drawRect(QRectF(s * 0.18, s * 0.40, s * 0.64, s * 0.30))
    p.drawRect(QRectF(s * 0.30, s * 0.18, s * 0.40, s * 0.22))
    p.drawRect(QRectF(s * 0.32, s * 0.62, s * 0.36, s * 0.20))


def _arrows(p: QPainter, s: int) -> None:
    p.setPen(QPen(GLYPH_WHITE, 1.6))
    p.drawArc(QRectF(s * 0.22, s * 0.22, s * 0.56, s * 0.56), 30 * 16, 120 * 16)
    p.drawArc(QRectF(s * 0.22, s * 0.22, s * 0.56, s * 0.56), 210 * 16, 120 * 16)
    p.setPen(Qt.NoPen)
    p.setBrush(GLYPH_WHITE)
    p.drawPolygon(QPolygonF([QPointF(s * 0.62, s * 0.16), QPointF(s * 0.78, s * 0.26),
                             QPointF(s * 0.60, s * 0.34)]))


def base_pen_color() -> QColor:
    return QColor(70, 90, 115)


_GLYPHS = {
    "new": _page, "newdoc": _page, "pdf": _printer, "folder": _folder,
    "settings": _gear, "help": _question, "calibrate": _target,
    "refresh": _arrows, "image": _page, "generate": _plus,
    "stop": _square, "skip": _minus, "validate": _page, "clear": _square,
    "type": _page,
}


def icon(name: str, size: int = 16) -> QIcon:
    """Glossy colored icon by name (created on demand, cached)."""
    cached = _CACHE.get((name, size))
    if cached is not None:
        return cached
    base = COLORS.get(name, QColor(90, 120, 160))
    glyph = _GLYPHS.get(name)
    ic = QIcon(_glossy_pixmap(base, glyph, size))
    _CACHE[(name, size)] = ic
    return ic


_CACHE: dict[tuple[str, int], QIcon] = {}


def QSize16():
    """16x16 QSize for toolbar/tree icon sizes."""
    from PySide6.QtCore import QSize
    return QSize(16, 16)
