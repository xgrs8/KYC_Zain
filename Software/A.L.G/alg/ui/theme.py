"""Aero-era console theme (plan 9 - v1.2 original concept look, binding).

Classic Windows-7/Server-2008R2 MMC chrome: soft blue-silver gradients,
glossy icons, #DCEBFC selection, #0066CC links, Segoe UI 9 pt.
Forbidden: dark theme, animations, emoji, oversized icons, web aesthetics.
"""
from __future__ import annotations

# ---- palette ------------------------------------------------------------
C = {
    "chrome_hi": "#FDFEFF", "chrome_mid": "#ECF1F8", "chrome_lo": "#E1E9F1",
    "menu_hi": "#F7FAFD", "menu_lo": "#EAF0F7",
    "border": "#C2D1E1", "border_soft": "#C0CFDE", "border_hard": "#6E8299",
    "pane": "#FDFEFF", "window": "#F0F5FA",
    "sel_hi": "#E0F0FC", "sel_lo": "#CDE5F9", "sel_flat": "#DCEBFC",
    "sel_border": "#A9CBEB", "hover_border": "#AECBE7",
    "link": "#0066CC", "link_alt": "#0A5FBE",
    "title_text": "#14385E", "text": "#22303F",
    "status_hi": "#F1F5FA", "status_lo": "#E3EAF2",
    "disabled": "#A6B2C0",
}

AERO_QSS = f"""
QMainWindow, QDialog {{
    background: {C['window']};
}}
QWidget {{
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 9pt;
    color: {C['text']};
}}
QMenuBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['menu_hi']}, stop:1 {C['menu_lo']});
    border-bottom: 1px solid {C['border']};
    padding: 1px;
}}
QMenuBar::item {{
    padding: 3px 8px 3px 8px;
    background: transparent;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:1 {C['sel_flat']});
    border: 1px solid {C['hover_border']};
}}
QMenu {{
    background: white;
    border: 1px solid {C['border']};
    padding: 3px;
}}
QMenu::item {{ padding: 4px 24px 4px 20px; }}
QMenu::item:selected {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['sel_hi']}, stop:1 {C['sel_lo']});
    border: 1px solid {C['sel_border']};
}}
QMenu::separator {{ height: 1px; background: {C['border']}; margin: 3px 6px; }}
QToolBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['chrome_hi']}, stop:1 {C['chrome_mid']});
    border-bottom: 1px solid {C['border']};
    padding: 2px 4px;
    spacing: 2px;
}}
QToolBar::separator {{
    width: 1px; margin: 3px 4px;
    background: {C['border']};
    border-right: 1px solid white;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 2px 4px;
}}
QToolButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:1 {C['sel_flat']});
    border: 1px solid {C['hover_border']};
}}
QToolButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['sel_flat']}, stop:1 {C['sel_hi']});
    border: 1px solid {C['sel_border']};
}}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['chrome_hi']}, stop:0.5 {C['chrome_mid']}, stop:1 {C['chrome_lo']});
    border: 1px solid {C['border_hard']};
    border-radius: 3px;
    padding: 4px 12px;
    min-height: 16px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:0.5 {C['sel_flat']}, stop:1 {C['sel_hi']});
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['sel_flat']}, stop:1 {C['sel_hi']});
    border: 1px solid {C['sel_border']};
}}
QPushButton:disabled {{ color: {C['disabled']}; border-color: {C['border']}; }}
QGroupBox {{
    background: {C['pane']};
    border: 1px solid {C['border_soft']};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 4px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    font-weight: 600;
}}
QTableWidget, QTableView, QTreeView, QTreeWidget, QListWidget {{
    background: white;
    alternate-background-color: white;
    border: 1px solid {C['border_soft']};
    selection-background-color: {C['sel_flat']};
    selection-color: {C['text']};
}}
QHeaderView::section {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FCFDFF, stop:1 #EFF3F8);
    border: 1px solid {C['border']};
    border-left-color: #E4EBF3;
    padding: 3px 6px;
    font-weight: 600;
}}
QTableWidget::item:selected, QTreeView::item:selected {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['sel_hi']}, stop:1 {C['sel_lo']});
    border: 1px solid {C['sel_border']};
}}
QTreeView::item:hover, QTableWidget::item:hover {{
    background: #E8F2FC;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
    background: white;
    border: 1px solid #7E90A4;
    border-radius: 2px;
    padding: 2px 4px;
    selection-background-color: {C['sel_flat']};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid #4A7BAD;
}}
QLineEdit[resolved="true"] {{
    background: #F8FCF8;
    border: 1px solid #79AD84;
}}
QLineEdit[resolved="true"]:focus {{
    border: 1px solid #5C9F6B;
}}
QLineEdit[resolved="true"]:disabled {{
    color: #526A57;
    background: #F4F9F4;
}}
QComboBox QAbstractItemView {{
    background: white;
    border: 1px solid {C['border']};
    selection-background-color: {C['sel_flat']};
}}
QProgressBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #E9EEF3, stop:1 #FBFCFD);
    border: 1px solid #9FB4CA;
    border-radius: 3px;
    text-align: center;
    min-height: 14px;
    max-height: 14px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #9FD0F5, stop:0.5 #63A9E4, stop:1 #3E86D2);
    border-radius: 2px;
    margin: 1px;
}}
QStatusBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C['status_hi']}, stop:1 {C['status_lo']});
    border-top: 1px solid {C['border']};
}}
QStatusBar::item {{ border: none; border-right: 1px solid {C['border']}; }}
QSplitter::handle {{ background: {C['border']}; width: 1px; height: 1px; }}
QCheckBox {{ spacing: 6px; }}
QScrollBar:vertical {{
    background: #F4F7FA; width: 12px; border-left: 1px solid {C['border']};
}}
QScrollBar::handle:vertical {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #FDFEFF, stop:1 #DCE6F0);
    border: 1px solid #A6B6C8; border-radius: 3px; min-height: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{
    background: #F4F7FA; height: 12px; border-top: 1px solid {C['border']};
}}
QScrollBar::handle:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FDFEFF, stop:1 #DCE6F0);
    border: 1px solid #A6B6C8; border-radius: 3px; min-width: 24px;
}}
QTabWidget::pane {{ border: 1px solid {C['border_soft']}; background: white; }}
QTabBar::tab {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #F7FAFD, stop:1 #E4EBF3);
    border: 1px solid {C['border']};
    padding: 4px 12px;
}}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #FFFFFF, stop:1 {C['sel_flat']});
    border-bottom: none;
}}
QLabel#paneHeader {{
    font-weight: 600;
    color: {C['title_text']};
}}
QPushButton#actionLink {{
    background: transparent;
    border: none;
    color: {C['link']};
    text-align: left;
    padding: 2px 4px 2px 2px;
}}
QPushButton#actionLink:hover {{
    background: #E8F2FC;
    border: 1px solid {C['hover_border']};
    border-radius: 3px;
    color: {C['link_alt']};
}}
QPushButton#actionLink:disabled {{ color: {C['disabled']}; }}
QFrame#actionsSeparator {{ max-height: 1px; background: {C['border']};
    border-bottom: 1px solid white; }}
QFrame#actionsPane {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #F7FAFC, stop:1 #EDF2F8);
    border: 1px solid {C['border_soft']};
    border-radius: 4px;
}}
QFrame#navPane, QFrame#centerPane {{
    background: {C['pane']};
    border: 1px solid {C['border_soft']};
    border-radius: 4px;
}}
"""

LINK_COLOR = C["link"]
TITLE_TEXT = C["title_text"]


def apply_aero_style(app) -> None:
    """Style + font for a QApplication; Fusion is the Linux/dev fallback
    (plan 9, OQ-9: windowsvista confirmed on Windows at M1)."""
    import sys

    from PySide6.QtGui import QFont, QFontDatabase
    if sys.platform == "win32":
        try:
            app.setStyle("windowsvista")
        except Exception:  # pragma: no cover
            app.setStyle("Fusion")
    else:
        app.setStyle("Fusion")
    available = set(QFontDatabase.families())
    for family in ("Segoe UI", "Noto Sans", "DejaVu Sans"):
        if family in available:
            font = QFont(family, 9)
            font.setStyleHint(QFont.SansSerif)
            app.setFont(font)
            break
    app.setStyleSheet(AERO_QSS)
