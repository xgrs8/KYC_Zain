"""A.L.G main window - the Hyper-V Manager-style console (plan 8.6, 9).

Tree (Workspace -> Labeling Steps 1-4 / Labels: Waiting, Done, Skipped),
center queue list + details (image preview + the label form), Actions
pane: New Label (pick type)..., Generate Label..., Add Images...,
Clear Current Session, Open Labeled Folder, Validate All Labels, About.
Generate auto-advances to the next Waiting sample (8.2 step 5).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rfalib.constants import (
    APP_TITLE_ALG,
    APP_VERSION,
    DOC_TYPE_ARABIC,
    DOC_TYPE_LABELS,
    QUEUE_STATES,
)
from rfalib.schema import label_keys, validate_label
from rfalib.sidecar import read_json

from .. import appsettings
from ..label import LabelSession
from ..paths import icon_file
from ..queue import QueueStore
from . import icons
from .label_form import LabelForm
from .upload_dialog import AddSamplesDialog
from .widgets import ActionsPane


class ALGWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE_ALG)
        self.setWindowIcon(QIcon(str(icon_file())))
        self.resize(1150, 760)
        self._settings = appsettings.load_settings()
        self._workspace = appsettings.ensure_workspace(self._settings)
        self.store = QueueStore(self._workspace)
        self._session: LabelSession | None = None

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._refresh_queue()
        self._select_next_waiting()

    # ------------------------------------------------------------ chrome
    def _build_menu(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        for text, slot in (("Add &Images...", self._add_samples),
                           ("&Open Labeled Folder", self._open_labeled),
                           ("E&xit", self.close)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            file_menu.addAction(act)

        actions_menu = mb.addMenu("&Actions")
        for text, slot, icon, seq in (
                ("&New Label (pick type)...", self._add_samples, "new", "Ctrl+N"),
                ("&Generate Label", self._generate, "generate", "Ctrl+G"),
                ("&Skip Sample...", self._skip, "skip", None),
                ("&Validate All Labels", self._validate_all, "validate", None),
                ("&Reset Sequence Counters...", self._reset_counters, "refresh", None),
                ("&Clear Current Session", self._clear_session, "clear", None)):
            act = QAction(text, self)
            act.setIcon(icons.icon(icon))
            if seq:
                act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            actions_menu.addAction(act)

        view_menu = mb.addMenu("&View")
        act = QAction("&Refresh", self)
        act.setShortcut("F5")
        act.triggered.connect(self._refresh_queue)
        view_menu.addAction(act)

        help_menu = mb.addMenu("&Help")
        act = QAction("&About A.L.G", self)
        act.triggered.connect(self._about)
        help_menu.addAction(act)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(icons.QSize16())
        for text, slot, icon in (
                ("Add Images...", self._add_samples, "image"),
                ("Generate Label", self._generate, "generate"),
                ("Skip...", self._skip, "skip"),
                ("Validate All", self._validate_all, "validate"),
                ("Open Labels", self._open_labeled, "folder"),
                ("Settings...", self._settings_dialog, "settings"),
                ("About", self._about, "help")):
            act = QAction(text, self)
            act.setIcon(icons.icon(icon))
            act.triggered.connect(slot)
            bar.addAction(act)
            if text in ("Skip...", "Open Labels"):
                bar.addSeparator()
        self.addToolBar(bar)

    # ------------------------------------------------------------ panes
    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)

        nav_frame = QFrame()
        nav_frame.setObjectName("navPane")
        nav = QVBoxLayout(nav_frame)
        nav.setContentsMargins(4, 4, 4, 4)
        nav.addWidget(QLabel("Navigation"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(icons.QSize16())
        root = QTreeWidgetItem(self.tree, [APP_TITLE_ALG])
        root.setIcon(0, self.windowIcon())
        steps = QTreeWidgetItem(["Labeling Steps 1-4"])
        steps.setIcon(0, icons.icon("type"))
        for label in ("1 Pick type", "2 Upload image(s)", "3 Type or mark fields",
                      "4 Generate label"):
            child = QTreeWidgetItem([label])
            child.setIcon(0, icons.icon("newdoc"))
            steps.addChild(child)
        root.addChild(steps)
        labels = QTreeWidgetItem(["Labels"])
        labels.setIcon(0, icons.icon("folder"))
        for state in QUEUE_STATES:
            child = QTreeWidgetItem([state])
            child.setIcon(0, icons.icon("validate" if state == "Done" else "newdoc"))
            labels.addChild(child)
        root.addChild(labels)
        self.tree.expandAll()
        nav.addWidget(self.tree)
        splitter.addWidget(nav_frame)

        center = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "State", "Fields"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._sample_selected)
        center.addWidget(self.table)

        self.tabs = QTabWidget()
        self.preview_stack = QStackedWidget()
        self.empty_state = QWidget()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.addStretch(1)
        empty_title = QLabel("Ready to label your first sample?")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_help = QLabel(
            "Add an ID front and back, or one tax card or business license image.")
        empty_help.setAlignment(Qt.AlignCenter)
        empty_help.setWordWrap(True)
        empty_layout.addWidget(empty_help)
        self.empty_add_button = QPushButton("Add Images to Start")
        self.empty_add_button.setMinimumHeight(38)
        self.empty_add_button.clicked.connect(self._add_samples)
        empty_layout.addWidget(self.empty_add_button, 0, Qt.AlignHCenter)
        empty_layout.addStretch(1)
        self.preview_stack.addWidget(self.empty_state)

        self.preview_label = QLabel("Select a sample to preview it.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_stack.addWidget(self.preview_label)
        self.tabs.addTab(self.preview_stack, "Preview")
        self.form_host = QWidget()
        self.form_layout = QVBoxLayout(self.form_host)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self._form: LabelForm | None = None
        self.tabs.addTab(self.form_host, "Label Form")
        center.addWidget(self.tabs)
        center.setSizes([240, 420])
        splitter.addWidget(center)

        self.actions = ActionsPane("Actions")
        self.actions.add_action("new", "New Label (pick type)...", "new",
                                self._add_samples)
        self.actions.add_action("generate", "Generate Label...", "generate",
                                self._generate)
        self.actions.add_action("add", "Add Images...", "image",
                                self._add_samples)
        self.actions.add_separator()
        self.actions.add_action("clear", "Clear Current Session", "clear",
                                self._clear_session)
        self.actions.add_action("open", "Open Labeled Folder", "folder",
                                self._open_labeled)
        self.actions.add_action("validate", "Validate All Labels", "validate",
                                self._validate_all)
        self.actions.add_action("reset", "Reset Sequence Counters...", "refresh",
                                self._reset_counters)
        self.actions.add_action("about", "About", "help", self._about)
        self.actions.add_stretch()
        ap = QWidget()
        apl = QVBoxLayout(ap)
        apl.setContentsMargins(0, 0, 0, 0)
        apl.addWidget(self.actions)
        splitter.addWidget(ap)
        splitter.setSizes([200, 610, 240])
        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        from PySide6.QtWidgets import QStatusBar
        sb = QStatusBar()
        self.status_label = QLabel("Ready - add images to start labeling")
        sb.addWidget(self.status_label, 1)
        self.progress_label = QLabel("")
        sb.addPermanentWidget(self.progress_label)
        self.setStatusBar(sb)

    # ------------------------------------------------------------ actions
    def _add_samples(self) -> None:
        try:
            dlg = AddSamplesDialog(self)
            result = dlg.exec()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Add Images failed", str(exc))
            self.status_label.setText(f"Could not open the image importer: {exc}")
            return
        if result != AddSamplesDialog.Accepted and not dlg.added_samples():
            return
        added = dlg.added_samples()
        if not added:
            return
        for item in added:
            self.store.add(item["doc_type"], item["name"], item["front"],
                           item["back"])
        self._refresh_queue()
        self.status_label.setText(f"{len(added)} sample(s) added to the queue")
        self._select_next_waiting()

    def _selected_sample(self):
        items = self.table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        sample_id = self.table.item(row, 0).data(Qt.UserRole)
        return self.store.get(sample_id)

    def _sample_selected(self) -> None:
        sample = self._selected_sample()
        if sample is None:
            return
        self._start_session(sample)

    def _start_session(self, sample) -> None:
        self.store.set_state(sample, "In progress") if sample.state == "Waiting" else None
        session = LabelSession(
            doc_type=sample.doc_type, name=sample.name, front=sample.front,
            back=sample.back, workspace=self._workspace,
            labeler=self._settings.get("labeler", "") or "")
        session.conditions = []
        self._session = session
        self._rebuild_form()
        front = Path(sample.front)
        if front.exists():
            pix = QPixmap(str(front))
            if not pix.isNull():
                self.preview_label.setPixmap(pix.scaledToHeight(
                    max(220, self.preview_label.height() or 300),
                    Qt.SmoothTransformation))
        self.status_label.setText(
            f"Labeling {sample.name} ({DOC_TYPE_LABELS[sample.doc_type]})")
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.tabs.setCurrentIndex(1)

    def _rebuild_form(self) -> None:
        if self._form is not None:
            self.form_layout.removeWidget(self._form)
            self._form.deleteLater()
            self._form = None
        if self._session is None:
            return
        has_truth = bool(self._session.truth)
        form = LabelForm(self._session.doc_type, has_truth)
        for key, row in form.rows.items():
            row.truth_btn.clicked.connect(
                lambda _c=False, k=key: self._use_truth(k))
        form.resolution_changed.connect(self._resolution_changed)
        form.generate_requested.connect(self._generate)
        form.skip_requested.connect(self._skip)
        # restore any previously typed values for this session
        form.set_values(self._session.fields)
        for tag in self._session.conditions:
            form.condition_checks[tag].setChecked(True)
        self.form_layout.addWidget(form)
        self._form = form
        form._emit_resolution()

    def _use_truth(self, key: str) -> None:
        if self._session is None or self._form is None:
            return
        if self._session.use_truth(key):
            self._form.rows[key].set_value(self._session.fields[key])

    def _resolution_changed(self, resolved: int, total: int) -> None:
        if self._session is None or self._form is None:
            return
        # pull widget values into the session (typed text)
        for key, value in self._form.values().items():
            if value != "[UNREADABLE]":
                if self._session.status.get(key) == "unreadable":
                    self._session.fields[key] = value
                else:
                    self._session.set_value(key, value)
            else:
                self._session.fields[key] = value
        self.progress_label.setText(f"{resolved}/{total} fields resolved")
        row = self.table.currentRow()
        if row >= 0:
            self.table.item(row, 3).setText(f"{resolved}/{total}")

    def _generate(self) -> None:
        if self._session is None or self._form is None:
            QMessageBox.information(self, "Generate Label",
                                    "Select a sample from the queue first.")
            return
        # sync current widget state into the session
        self._resolution_changed(0, 0)
        for key, value in self._form.values().items():
            if value == "[UNREADABLE]":
                self._session.mark_unreadable(key)
            else:
                self._session.set_value(key, value)
        self._session.conditions = self._form.conditions()
        problems = self._session.gate_problems()
        if problems:
            QMessageBox.warning(
                self, "Generate blocked",
                "Every field must be typed or marked [Unreadable]:\n"
                + "\n".join(f"{k}: {v}" for k, v in problems.items()))
            return
        warnings = self._session.warnings()
        if warnings:
            ret = QMessageBox.question(
                self, "Format warnings",
                "These values look unusual:\n"
                + "\n".join(f"{k}: {v}" for k, v in warnings.items())
                + "\n\nGenerate anyway? (the override is logged)")
            if ret != QMessageBox.Yes:
                return
            self._session.warnings_overridden = sorted(warnings)
        try:
            result = self._session.generate(
                merge_width=int(self._settings.get("merge_width", 1140)),
                merge_gap=int(self._settings.get("merge_gap", 24)),
                jpeg=bool(self._settings.get("jpeg_option", False)),
                labeler=self._settings.get("labeler", "") or None)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Generate failed", str(exc))
            return
        # mark the queue row Done and advance (8.2 step 5)
        sample = self._sample_for_session()
        if sample is not None:
            self.store.set_state(sample, "Done", output_name=result["base"])
        self.status_label.setText(
            f"Label written: {result['label']}  (photo: {Path(result['image']).name})")
        self._refresh_queue()
        self._select_next_waiting()

    def _sample_for_session(self):
        front = self._session.front if self._session else None
        for s in self.store.samples:
            if s.front == front:
                return s
        return None

    def _skip(self) -> None:
        if self._session is None:
            QMessageBox.information(self, "Skip", "Select a sample first.")
            return
        from PySide6.QtWidgets import QInputDialog
        reason, ok = QInputDialog.getText(
            self, "Skip Sample", "Skip reason (required):")
        if not ok or not reason.strip():
            return
        try:
            self._session.skip(reason.strip(),
                               labeler=self._settings.get("labeler") or None)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Skip failed", str(exc))
            return
        sample = self._sample_for_session()
        if sample is not None:
            self.store.set_state(sample, "Skipped", skip_reason=reason.strip())
        self.status_label.setText(f"Sample skipped: {reason.strip()}")
        self._refresh_queue()
        self._select_next_waiting()

    def _validate_all(self) -> None:
        labeled = sorted((self._workspace / "labeled").glob("*.label.json"))
        if not labeled:
            QMessageBox.information(self, "Validate All Labels",
                                    "No labels in "
                                    + str(self._workspace / "labeled"))
            return
        lines = []
        bad = 0
        for path in labeled:
            try:
                obj = read_json(path)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"{path.name}: unreadable ({exc})")
                bad += 1
                continue
            problems = validate_label(obj)
            if problems:
                lines.append(f"{path.name}: " + "; ".join(problems))
                bad += 1
            else:
                lines.append(f"{path.name}: OK")
        title = f"{len(labeled) - bad}/{len(labeled)} labels valid"
        QMessageBox.information(self, "Validate All Labels - " + title,
                                "\n".join(lines))

    def _reset_counters(self) -> None:
        from rfalib import naming

        ret = QMessageBox.question(
            self, "Reset Sequence Counters",
            "Reset all document counters to 1 (ID_01, TAX_01, LIC_01)?\n"
            "Existing files are kept. A.L.G will skip names already in the "
            "labeled folder so they are not overwritten.")
        if ret == QMessageBox.Yes:
            naming.reset_counters(self._workspace)
            self.status_label.setText("Sequence counters reset")

    def _clear_session(self) -> None:
        ret = QMessageBox.question(
            self, "Clear Current Session",
            "Remove every non-Done sample from the queue?\n"
            "(Done labels on disk are kept.)")
        if ret == QMessageBox.Yes:
            self.store.clear_session()
            self._session = None
            self._rebuild_form()
            self._refresh_queue()

    def _open_labeled(self) -> None:
        folder = str(self._workspace / "labeled")
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:  # noqa: BLE001
            QMessageBox.information(self, "Folder", folder)

    def _settings_dialog(self) -> None:
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec() == SettingsDialog.Accepted:
            self._settings = dlg.result()
            appsettings.save_settings(self._settings)
            self._workspace = appsettings.ensure_workspace(self._settings)
            self.status_label.setText("Settings saved")

    def _about(self) -> None:
        QMessageBox.about(
            self, "About A.L.G",
            f"<b>A.L.G - Automatic Label Generator</b><br>"
            f"Version {APP_VERSION}<br><br>"
            f"Human-driven labeling: pick the type, upload the image(s), "
            f"type every field or mark it [UNREADABLE], Generate.<br>"
            f"The ID front and back are merged exactly like R.F.G "
            f"(shared rfalib), and the label JSON matches the agreed "
            f"schemas byte-for-byte.")

    # ------------------------------------------------------------ data
    def _refresh_queue(self) -> None:
        self.table.setRowCount(0)
        for s in self.store.samples:
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(s.name)
            item.setData(Qt.UserRole, s.sample_id)
            self.table.setItem(row, 0, item)
            label = DOC_TYPE_LABELS.get(s.doc_type, "?")
            label_ar = DOC_TYPE_ARABIC.get(s.doc_type, "")
            self.table.setItem(row, 1, QTableWidgetItem(f"{label} / {label_ar}"))
            self.table.setItem(row, 2, QTableWidgetItem(s.state))
            fields = len(label_keys(s.doc_type))
            shown = f"{fields}/{fields}" if s.state == "Done" else f"-/{fields}"
            self.table.setItem(row, 3, QTableWidgetItem(shown))
        if not self.store.samples:
            self.preview_stack.setCurrentWidget(self.empty_state)

    def _select_next_waiting(self) -> None:
        sample = self.store.next_waiting()
        if sample is None:
            return
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) == sample.sample_id:
                self.table.selectRow(r)
                break
