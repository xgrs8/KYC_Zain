"""Add Samples dialog (plan 8.2 steps 1-2, R16/R17).

Step 1: pick the input type - ID / Tax card / Business license.
Step 2: upload - for the ID two slots (front, then back - both required);
        for TAX/LIC a single slot.  "Add another" queues more samples.
Accepted formats: PNG / JPEG.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from rfalib.constants import DOC_TYPE_ARABIC, DOC_TYPE_LABELS

from . import icons


class AddSamplesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Samples - pick type, then upload")
        self.setModal(True)
        self.setMinimumWidth(520)
        self._added: list[dict] = []
        layout = QVBoxLayout(self)

        type_box = QGroupBox("Step 1 - Pick the input type")
        tlay = QHBoxLayout()
        self.radios: dict[str, QRadioButton] = {}
        for t in ("ID", "TAX", "LIC"):
            rb = QRadioButton(f"{DOC_TYPE_LABELS[t]}\n{DOC_TYPE_ARABIC[t]}")
            rb.setChecked(t == "ID")
            tlay.setAlignment(rb, Qt.AlignHCenter)
            self.radios[t] = rb
            tlay.addWidget(rb)
        type_box.setLayout(tlay)
        layout.addWidget(type_box)

        up_box = QGroupBox("Step 2 - Upload image(s)")
        form = QFormLayout(up_box)
        self.front_edit = QLineEdit()
        self.front_edit.setPlaceholderText("front image (PNG/JPEG)")
        form.addRow("Front image:", self._picker_row(self.front_edit))
        self.back_edit = QLineEdit()
        self.back_edit.setPlaceholderText("back image (PNG/JPEG) - ID only")
        form.addRow("Back image:", self._picker_row(self.back_edit))
        self.back_label = QLabel("(ID needs both the front and the back)")
        form.addRow("", self.back_label)
        for t, rb in self.radios.items():
            rb.toggled.connect(self._type_changed)
        layout.addWidget(up_box)

        buttons = QDialogButtonBox()
        add_btn = QPushButton("Add to queue")
        add_btn.setIcon(icons.icon("new"))
        add_btn.clicked.connect(self._add)
        done_btn = QPushButton("Done")
        done_btn.setDefault(True)
        done_btn.clicked.connect(self.accept)
        buttons.addButton(add_btn, QDialogButtonBox.ActionRole)
        buttons.addButton(done_btn, QDialogButtonBox.AcceptRole)
        layout.addWidget(buttons)
        self._type_changed()

    def _picker_row(self, edit: QLineEdit) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(edit))
        lay.addWidget(edit)
        lay.addWidget(browse)
        return row

    def _browse(self, edit: QLineEdit) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, "Choose image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG)")
        if path:
            edit.setText(path)

    def _doc_type(self) -> str:
        for t, rb in self.radios.items():
            if rb.isChecked():
                return t
        return "ID"

    def _type_changed(self) -> None:
        is_id = self._doc_type() == "ID"
        self.back_edit.setEnabled(is_id)
        self.back_label.setVisible(is_id)

    def _add(self) -> None:
        doc_type = self._doc_type()
        front = self.front_edit.text().strip()
        back = self.back_edit.text().strip()
        if not front or not Path(front).exists():
            self.front_edit.setStyleSheet("border: 1px solid #C0392B;")
            return
        self.front_edit.setStyleSheet("")
        if doc_type == "ID" and (not back or not Path(back).exists()):
            self.back_edit.setStyleSheet("border: 1px solid #C0392B;")
            return
        self.back_edit.setStyleSheet("")
        self._added.append({
            "doc_type": doc_type,
            "name": Path(front).stem,
            "front": front,
            "back": back if doc_type == "ID" else None,
        })
        self.front_edit.clear()
        self.back_edit.clear()

    def added_samples(self) -> list[dict]:
        return self._added
