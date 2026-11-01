"""The label form (plan 8.2 step 4): one row per field of the chosen
type's schema, in the exact 6.4 order.

Each row: label (English key + Arabic card label), value box (RTL +
right-aligned for Arabic fields, LTR for MRZ/serials/dates), an
[Unreadable] toggle, and - only when a truth sidecar was found next to
the source image - a "Use truth" button.  Plain entry rows otherwise.

Keyboard: Ctrl+U mark unreadable, Ctrl+T use truth, Enter next field.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rfalib.constants import CONDITION_TAGS, UNREADABLE
from rfalib.schema import fields_meta

from . import icons


class FieldRow(QWidget):
    changed = Signal()

    def __init__(self, meta: dict, has_truth: bool, parent=None):
        super().__init__(parent)
        self.key = meta["key"]
        self._has_truth = has_truth
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        if meta.get("rtl"):
            self.edit.setLayoutDirection(Qt.RightToLeft)
            self.edit.setAlignment(Qt.AlignRight)
        self._completion_glow = QGraphicsDropShadowEffect(self.edit)
        self._completion_glow.setBlurRadius(8)
        self._completion_glow.setOffset(0, 0)
        self._completion_glow.setColor(QColor(46, 145, 75, 68))
        self._completion_glow.setEnabled(False)
        self.edit.setGraphicsEffect(self._completion_glow)
        self.edit.setProperty("resolved", False)
        self.edit.textEdited.connect(self._text_edited)
        lay.addWidget(self.edit, 1)

        self.unreadable_btn = QPushButton("[Unreadable]")
        self.unreadable_btn.setCheckable(True)
        self.unreadable_btn.setFixedWidth(110)
        self.unreadable_btn.setCursor(Qt.PointingHandCursor)
        self.unreadable_btn.setToolTip("Mark this field unreadable (Ctrl+U)")
        self.unreadable_btn.toggled.connect(self._toggle_unreadable)
        lay.addWidget(self.unreadable_btn)

        self.truth_btn = QPushButton("Use truth")
        self.truth_btn.setFixedWidth(90)
        self.truth_btn.setCursor(Qt.PointingHandCursor)
        self.truth_btn.setToolTip("Accept the generator's printed value (Ctrl+T)")
        self.truth_btn.setVisible(has_truth)
        lay.addWidget(self.truth_btn)

    def _toggle_unreadable(self, on: bool) -> None:
        self.edit.setEnabled(not on)
        self.truth_btn.setEnabled(not on)
        self._refresh_completion_state()
        self.changed.emit()

    def _text_edited(self, text: str) -> None:
        self._refresh_completion_state(text.strip() != "")
        self.changed.emit()

    def _refresh_completion_state(self, filled: bool | None = None) -> None:
        resolved = (self.unreadable_btn.isChecked() or bool(self.edit.text().strip())
                    if filled is None else
                    self.unreadable_btn.isChecked() or filled)
        self.edit.setProperty("resolved", resolved)
        self._completion_glow.setEnabled(resolved)
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)
        self.edit.update()

    def set_unreadable(self) -> None:
        if not self.unreadable_btn.isChecked():
            self.unreadable_btn.setChecked(True)

    def clear_unreadable(self) -> None:
        if self.unreadable_btn.isChecked():
            self.unreadable_btn.setChecked(False)

    def value(self) -> str:
        return UNREADABLE if self.unreadable_btn.isChecked() else self.edit.text().strip()

    def set_value(self, value: str) -> None:
        if value == UNREADABLE:
            self.edit.clear()
            self.unreadable_btn.setChecked(True)
        else:
            self.unreadable_btn.setChecked(False)
            self.edit.setText(value)
        self._refresh_completion_state()
        self.changed.emit()

    def truth_clicked_connect(self, slot) -> None:
        self.truth_btn.clicked.connect(slot)


class LabelForm(QWidget):
    """Scrollable form for every field + condition tags + gate status."""

    resolution_changed = Signal(int, int)
    generate_requested = Signal()
    skip_requested = Signal()

    def __init__(self, doc_type: str, has_truth: bool, parent=None):
        super().__init__(parent)
        self.doc_type = doc_type
        self.rows: dict[str, FieldRow] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for meta in fields_meta(doc_type):
            row = FieldRow(meta, has_truth)
            label = QLabel(f"<b>{meta['key']}</b><br>{meta['ar']}")
            label.setLayoutDirection(Qt.RightToLeft)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow(label, row)
            row.changed.connect(lambda: self._emit_resolution())
            self.rows[meta["key"]] = row
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        cond_box = QGroupBox("Conditions (optional tags)")
        cl = QHBoxLayout(cond_box)
        self.condition_checks: dict[str, QCheckBox] = {}
        for tag in CONDITION_TAGS:
            cb = QCheckBox(tag)
            self.condition_checks[tag] = cb
            cl.addWidget(cb)
        outer.addWidget(cond_box)

        bottom = QHBoxLayout()
        self.status_label = QLabel("0/0 fields resolved")
        bottom.addWidget(self.status_label, 1)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setIcon(icons.icon("generate"))
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(lambda: self.generate_requested.emit())
        bottom.addWidget(self.generate_btn)
        skip_btn = QPushButton("Skip...")
        skip_btn.setIcon(icons.icon("skip"))
        skip_btn.clicked.connect(lambda: self.skip_requested.emit())
        bottom.addWidget(skip_btn)
        outer.addLayout(bottom)

        # keyboard (9): Ctrl+U / Ctrl+T / Enter / Ctrl+G
        QShortcut(QKeySequence("Ctrl+U"), self, activated=self._ctrl_u)
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self._ctrl_t)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self.generate_btn.click)
        QShortcut(QKeySequence("Return"), self, activated=self._enter_next)

    # ------------------------------------------------------------ helpers
    def _focused_row(self) -> FieldRow | None:
        w = self.focusWidget()
        while w is not None and not isinstance(w, FieldRow):
            w = w.parentWidget()
        return w if isinstance(w, FieldRow) else None

    def _ctrl_u(self) -> None:
        row = self._focused_row()
        if row is not None:
            row.set_unreadable() if not row.unreadable_btn.isChecked() \
                else row.clear_unreadable()

    def _ctrl_t(self) -> None:
        row = self._focused_row()
        if row is not None and row.truth_btn.isVisible() and row.truth_btn.isEnabled():
            row.truth_btn.click()

    def _enter_next(self) -> None:
        keys = [m["key"] for m in fields_meta(self.doc_type)]
        row = self._focused_row()
        if row is None:
            self.rows[keys[0]].edit.setFocus()
            return
        try:
            idx = keys.index(row.key)
        except ValueError:
            return
        if idx + 1 < len(keys):
            self.rows[keys[idx + 1]].edit.setFocus()
        else:
            self.generate_btn.setFocus()

    def _emit_resolution(self) -> None:
        resolved, total = self.resolution()
        self.status_label.setText(f"{resolved}/{total} fields resolved")
        self.generate_btn.setEnabled(resolved == total)
        self.resolution_changed.emit(resolved, total)

    # ------------------------------------------------------------ API
    def values(self) -> dict[str, str]:
        return {k: row.value() for k, row in self.rows.items()}

    def resolution(self) -> tuple[int, int]:
        total = len(self.rows)
        resolved = 0
        for row in self.rows.values():
            v = row.value()
            if v == UNREADABLE or v.strip():
                if not (UNREADABLE in v and v != UNREADABLE):
                    resolved += 1
        return resolved, total

    def set_values(self, values: dict[str, str]) -> None:
        for key, value in values.items():
            if key in self.rows:
                self.rows[key].set_value(value)
        self._emit_resolution()

    def conditions(self) -> list[str]:
        return [t for t, cb in self.condition_checks.items() if cb.isChecked()]

    def set_conditions(self, conditions: list[str]) -> None:
        for t, cb in self.condition_checks.items():
            cb.setChecked(t in conditions)
