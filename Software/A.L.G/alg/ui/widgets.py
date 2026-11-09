"""Shared MMC-style widgets: the Actions pane and small helpers (plan 9)."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import icons


class ActionsPane(QFrame):
    """Right-hand Actions pane: bold header, blue hyperlink items with
    16x16 leading icons, separators with a white edge (Hyper-V style)."""

    action_triggered = Signal(str)

    def __init__(self, title: str = "Actions", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("actionsPane")
        self._buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(1)
        header = QLabel(title)
        header.setObjectName("paneHeader")
        layout.addWidget(header)
        layout.addSpacing(4)

    def add_action(self, action_id: str, text: str, icon_name: str,
                   handler: Callable[[], None] | None = None,
                   enabled: bool = True) -> QPushButton:
        btn = QPushButton("  " + text)
        btn.setObjectName("actionLink")
        btn.setIcon(icons.icon(icon_name, 16))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setEnabled(enabled)
        if handler:
            btn.clicked.connect(handler)
        else:
            btn.clicked.connect(lambda: self.action_triggered.emit(action_id))
        self.layout().addWidget(btn)
        self._buttons[action_id] = btn
        return btn

    def add_separator(self) -> None:
        sep = QFrame()
        sep.setObjectName("actionsSeparator")
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        self.layout().addSpacing(3)
        self.layout().addWidget(sep)
        self.layout().addSpacing(3)

    def add_stretch(self) -> None:
        self.layout().addStretch(1)

    def set_action_enabled(self, action_id: str, enabled: bool) -> None:
        btn = self._buttons.get(action_id)
        if btn is not None:
            btn.setEnabled(enabled)

    def button(self, action_id: str) -> QPushButton | None:
        return self._buttons.get(action_id)


def status_progress_bar() -> tuple[QWidget, object]:
    """A small horizontal container for the status bar (label + bar)."""
    from PySide6.QtWidgets import QProgressBar
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setFixedWidth(220)
    return bar, bar
