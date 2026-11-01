"""Queue behavior (plan 8.2): states, persistence, next-waiting,
clear-session.  T-GUI for the A.L.G window structure."""
from __future__ import annotations

import pytest

from alg.queue import QueueStore


@pytest.fixture
def store(tmp_path):
    return QueueStore(tmp_path)


def test_add_and_states(store, tmp_path):
    s = store.add("ID", "ID_01", "/a/front.png", "/a/back.png")
    assert s.state == "Waiting"
    assert store.by_state("Waiting") == [s]
    store.set_state(s, "In progress")
    store.set_state(s, "Done", output_name="ID_01")
    assert s.state == "Done" and s.output_name == "ID_01"
    assert store.next_waiting() is None


def test_next_waiting_order(store):
    a = store.add("TAX", "one", "/a.png")
    b = store.add("TAX", "two", "/b.png")
    assert store.next_waiting() is a
    store.set_state(a, "Skipped", skip_reason="blur")
    assert store.next_waiting() is b


def test_persistence_roundtrip(tmp_path):
    store = QueueStore(tmp_path)
    store.add("ID", "ID_01", "/f.png", "/b.png")
    store.add("LIC", "lic", "/l.png")
    store.set_state(store.samples[0], "Done", output_name="ID_01")
    reloaded = QueueStore(tmp_path)
    assert [s.name for s in reloaded.samples] == ["ID_01", "lic"]
    assert reloaded.samples[0].state == "Done"
    assert reloaded.get(reloaded.samples[1].sample_id).doc_type == "LIC"


def test_clear_session_keeps_done(store):
    a = store.add("TAX", "one", "/a.png")
    b = store.add("TAX", "two", "/b.png")
    store.set_state(a, "Done", output_name="TAX_01")
    store.clear_session()
    assert [s.name for s in store.samples] == ["one"]
    assert store.next_waiting() is b or store.next_waiting() is None


def test_skip_reason_persisted(tmp_path):
    store = QueueStore(tmp_path)
    s = store.add("ID", "ID_02", "/f.png", "/b.png")
    store.set_state(s, "Skipped", skip_reason="glare everywhere")
    reloaded = QueueStore(tmp_path)
    assert reloaded.samples[0].skip_reason == "glare everywhere"


def test_corrupted_queue_starts_fresh(tmp_path):
    (tmp_path / "alg_queue.json").write_text("{not json", encoding="utf-8")
    store = QueueStore(tmp_path)
    assert store.samples == []


# ---------------------------------------------------------------- T-GUI

def test_alg_window_structure():
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QToolBar, QTreeWidget

    from alg.ui.main_window import ALGWindow
    QApplication.instance() or QApplication(["alg-test"])
    win = ALGWindow()
    menus = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert menus == ["File", "Actions", "View", "Help"]
    assert win.findChildren(QToolBar)
    trees = win.findChildren(QTreeWidget)
    assert trees
    root = trees[0].topLevelItem(0)
    children = [root.child(i).text(0) for i in range(root.childCount())]
    assert "Labeling Steps 1-4" in children and "Labels" in children
    assert win.table.columnCount() == 4
    headers = [win.table.horizontalHeaderItem(i).text()
               for i in range(win.table.columnCount())]
    assert headers == ["Name", "Type", "State", "Fields"]
    for key in ("new", "generate", "add", "clear", "open", "validate",
                "reset", "about"):
        assert key in win.actions._buttons, key
    assert win.statusBar() is not None
    assert not win.windowIcon().isNull()


def test_reset_counter_action_resets_all_document_types(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from alg.ui import main_window
    from rfalib import naming

    app = QApplication.instance() or QApplication(["alg-test"])
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        main_window.appsettings, "load_settings",
        lambda: {"workspace": str(workspace)})
    win = main_window.ALGWindow()
    for doc_type in ("ID", "TAX", "LIC"):
        naming.set_counter(workspace, doc_type, 25)
    monkeypatch.setattr(main_window.QMessageBox, "question",
                        lambda *_args, **_kwargs: QMessageBox.Yes)

    win.actions.button("reset").click()

    assert {t: naming.peek_next(workspace, t)
            for t in ("ID", "TAX", "LIC")} == {
                "ID": 1, "TAX": 1, "LIC": 1}
    assert win.status_label.text() == "Sequence counters reset"
    win.close()


def test_empty_state_add_button_opens_sample_dialog(tmp_path, monkeypatch):
    """The real import dialog opens and its queued image can be labeled."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton
    from PIL import Image

    from alg.ui import main_window
    from alg.ui.upload_dialog import AddSamplesDialog

    app = QApplication.instance() or QApplication(["alg-test"])
    workspace = tmp_path / "workspace"
    sample = tmp_path / "sample-tax.png"
    Image.new("RGB", (395, 512), (198, 214, 232)).save(sample)
    monkeypatch.setattr(
        main_window.appsettings, "load_settings",
        lambda: {"workspace": str(workspace)})

    win = main_window.ALGWindow()
    win.show()
    app.processEvents()
    assert win.empty_add_button.isVisible()
    assert win.empty_add_button.isEnabled()

    dialog_errors = []

    def complete_import_dialog():
        dialog = app.activeModalWidget()
        try:
            assert isinstance(dialog, AddSamplesDialog)
            QTest.mouseClick(dialog.radios["TAX"], Qt.LeftButton)
            dialog.front_edit.setText(str(sample))
            buttons = dialog.findChildren(QPushButton)
            add_button = next(button for button in buttons
                              if button.text() == "Add to queue")
            done_button = next(button for button in buttons
                               if button.text() == "Done")
            QTest.mouseClick(add_button, Qt.LeftButton)
            assert len(dialog.added_samples()) == 1
            QTest.mouseClick(done_button, Qt.LeftButton)
        except Exception as exc:  # noqa: BLE001
            dialog_errors.append(exc)
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(0, complete_import_dialog)
    QTest.mouseClick(win.empty_add_button, Qt.LeftButton)
    app.processEvents()
    assert not dialog_errors, str(dialog_errors)
    assert win.table.rowCount() == 1
    assert win._session is not None and win._session.doc_type == "TAX"
    assert win._form is not None and not win._form.generate_btn.isEnabled()
    for row in win._form.rows.values():
        QTest.mouseClick(row.unreadable_btn, Qt.LeftButton)
    app.processEvents()
    assert win._form.generate_btn.isEnabled()
    QTest.mouseClick(win._form.generate_btn, Qt.LeftButton)
    app.processEvents()
    assert win.store.samples[0].state == "Done"
    assert (workspace / "labeled" / "TAX_01.png").exists()
    assert (workspace / "labeled" / "TAX_01.label.json").exists()
    win.close()


def test_label_form_rtl_flags():
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from alg.ui.label_form import LabelForm
    from rfalib.schema import label_keys
    QApplication.instance() or QApplication(["alg-test"])
    form = LabelForm("ID", has_truth=True)
    assert set(form.rows) == set(label_keys("ID"))
    # Arabic fields are RTL + right-aligned; MRZ/serials are LTR
    assert form.rows["name"].edit.layoutDirection() == Qt.RightToLeft
    assert form.rows["name"].edit.alignment() == Qt.AlignRight
    assert form.rows["mrz_line1"].edit.layoutDirection() == Qt.LeftToRight
    # gate: Generate disabled while fields are empty
    assert form.generate_btn.isEnabled() is False
    # truth buttons exist only when a sidecar was found
    assert form.rows["name"].truth_btn.isVisibleTo(form)
    plain = LabelForm("ID", has_truth=False)
    assert plain.rows["name"].truth_btn.isHidden()


def test_filled_and_unreadable_fields_get_subtle_completion_feedback():
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from alg.ui.label_form import LabelForm

    app = QApplication.instance() or QApplication(["alg-test"])
    form = LabelForm("TAX", has_truth=False)
    form.show()
    app.processEvents()
    row = form.rows["taxpayer_number"]
    assert row.edit.property("resolved") is False
    assert not row._completion_glow.isEnabled()

    row.edit.setFocus()
    QTest.keyClicks(row.edit, "123456789")
    app.processEvents()
    assert row.edit.property("resolved") is True
    assert row._completion_glow.isEnabled()

    row.edit.clear()
    QTest.mouseClick(row.unreadable_btn, Qt.LeftButton)
    app.processEvents()
    assert row.edit.property("resolved") is True
    assert row._completion_glow.isEnabled()

    QTest.mouseClick(row.unreadable_btn, Qt.LeftButton)
    app.processEvents()
    assert row.edit.property("resolved") is False
    assert not row._completion_glow.isEnabled()
    form.close()


def test_form_resolution_gate():
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from alg.ui.label_form import LabelForm
    QApplication.instance() or QApplication(["alg-test"])
    form = LabelForm("TAX", has_truth=False)
    assert form.resolution() == (0, 5)
    form.rows["taxpayer_number"].edit.setText("101234567890")
    assert form.resolution() == (1, 5)
    form.rows["taxpayer_name"].set_value("[UNREADABLE]")
    assert form.resolution() == (2, 5)
    for row in form.rows.values():
        if not row.unreadable_btn.isChecked() and not row.edit.text().strip():
            row.set_value("[UNREADABLE]")
    assert form.resolution() == (5, 5)
    values = form.values()
    assert values["taxpayer_number"] == "101234567890"
    assert values["taxpayer_name"] == "[UNREADABLE]"
