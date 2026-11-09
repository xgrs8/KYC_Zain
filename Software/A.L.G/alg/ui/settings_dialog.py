"""Settings dialog for A.L.G (labeler identity, merge, output options)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._settings = dict(settings)
        layout = QVBoxLayout(self)

        who_box = QGroupBox("Labeler")
        wform = QFormLayout(who_box)
        self.labeler_edit = QLineEdit(str(settings.get("labeler") or ""))
        self.labeler_edit.setPlaceholderText("name stored in every label sidecar")
        wform.addRow("Labeler:", self.labeler_edit)
        layout.addWidget(who_box)

        merge_box = QGroupBox("Merge (front + back)")
        mform = QFormLayout(merge_box)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1000, 1600)
        self.width_spin.setSingleStep(20)
        self.width_spin.setValue(int(settings.get("merge_width", 1140)))
        mform.addRow("Merged width (px, 1000-1600):", self.width_spin)
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 80)
        self.gap_spin.setValue(int(settings.get("merge_gap", 24)))
        mform.addRow("White gap (px):", self.gap_spin)
        layout.addWidget(merge_box)

        out_box = QGroupBox("Output options")
        oform = QFormLayout(out_box)
        self.jpeg_cb = QCheckBox("Also write JPEG (q95) next to every PNG")
        self.jpeg_cb.setChecked(bool(settings.get("jpeg_option", False)))
        oform.addRow(self.jpeg_cb)
        layout.addWidget(out_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result(self) -> dict:
        self._settings.update({
            "labeler": self.labeler_edit.text().strip(),
            "merge_width": self.width_spin.value(),
            "merge_gap": self.gap_spin.value(),
            "jpeg_option": self.jpeg_cb.isChecked(),
        })
        return self._settings
