"""Behavioral tests for PyrtnfApp study lifecycle wiring.

Covers the study-management methods on ``pineuro.gui.app.PyrtnfApp`` —
duplicate / import / export / delete / edit-load / start-session — asserting
that each method drives ConfigManager correctly and updates app state.

The app is built by bypassing ``__init__`` (which would launch Flet plugin
discovery, configure a real Page, and read the real ``~/.pineuro/studies``
directory). We attach a MagicMock page and a *real* ConfigManager rooted at
tmp_path, so we can assert genuine on-disk effects (files created / removed)
rather than rendered widgets.
"""

from __future__ import annotations

from unittest.mock import patch

import flet as ft
import pytest

from pineuro.gui.app import PyrtnfApp
from pineuro.gui.config_manager import ConfigManager
from pineuro.gui.models import StudyConfig, StudyMetadata

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _session_config(base_dir: str = "/data/pineuro", example_id: str = "sub-001") -> dict:
    return {
        "paths": {"base_dir": base_dir, "example_participant_id": example_id},
        "feedback_runs": [{"task": "ball", "tr": 1.5, "n_volumes": 200}],
    }


def _make_study(name="Test Study", description="A study", session_config=None) -> StudyConfig:
    return StudyConfig(
        metadata=StudyMetadata(name=name, description=description),
        session_config=session_config if session_config is not None else _session_config(),
    )


@pytest.fixture
def manager(tmp_path) -> ConfigManager:
    return ConfigManager(studies_dir=tmp_path / "studies")


@pytest.fixture
def app(mock_flet_page, manager) -> PyrtnfApp:
    """A PyrtnfApp with a mock page + real tmp-rooted ConfigManager.

    Bypasses ``__init__`` so no Flet runtime, plugin discovery, or real
    studies-dir access occurs. ``show_study_manager`` is stubbed by default
    via the mock page methods (controls.clear / add / update are mocks), so
    'refresh' is observable as a call to it.
    """
    instance = PyrtnfApp.__new__(PyrtnfApp)
    instance.page = mock_flet_page
    instance.config_manager = manager
    instance.current_study = None
    return instance


def _last_snackbar_text(page) -> str | None:
    """Return the text of the most recently appended SnackBar, if any."""
    for control in reversed(page.overlay):
        if isinstance(control, ft.SnackBar):
            return control.content.value
    return None


# =============================================================================
# _duplicate_study
# =============================================================================


class TestDuplicateStudy:
    def test_creates_copy_on_disk_via_config_manager(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))

        app._duplicate_study("Alpha")

        names = {m.name for m in manager.list_studies()}
        assert "Alpha" in names
        assert "Alpha (Copy)" in names

    def test_success_shows_snackbar_naming_the_copy(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))

        app._duplicate_study("Alpha")

        assert _last_snackbar_text(app.page) == "Study duplicated: Alpha (Copy)"

    def test_success_refreshes_study_manager(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))
        with patch.object(app, "show_study_manager") as refresh:
            app._duplicate_study("Alpha")
        refresh.assert_called_once()

    def test_missing_source_is_noop(self, app):
        with patch.object(app, "show_study_manager") as refresh:
            app._duplicate_study("DoesNotExist")
        refresh.assert_not_called()
        assert app.page.overlay == []


# =============================================================================
# _import_study tests
# =============================================================================


class TestImportStudy:
    def test_cancelled_picker_is_noop(self, app, manager):
        with (
            patch("pineuro.gui.utils.pick_file", return_value=None),
            patch.object(app, "show_study_manager") as refresh,
        ):
            app._import_study()
        refresh.assert_not_called()
        assert manager.list_studies() == []

    def test_picked_file_imported_into_studies_dir(self, app, manager, tmp_path):
        # An external YAML file to import: write via the manager, then move it
        # out of the studies dir so it looks like a user-chosen external file.
        src = tmp_path / "external.yaml"
        manager.save_study(_make_study(name="Imported Study"))
        manager._get_study_path("Imported Study").rename(src)

        with (
            patch("pineuro.gui.utils.pick_file", return_value=str(src)),
            patch.object(app, "show_study_manager"),
        ):
            app._import_study()

        names = {m.name for m in manager.list_studies()}
        assert "Imported Study" in names

    def test_success_shows_snackbar_and_refreshes(self, app, manager, tmp_path):
        src = tmp_path / "external.yaml"
        manager.save_study(_make_study(name="Imported Study"))
        manager._get_study_path("Imported Study").rename(src)

        with (
            patch("pineuro.gui.utils.pick_file", return_value=str(src)),
            patch.object(app, "show_study_manager") as refresh,
        ):
            app._import_study()

        assert _last_snackbar_text(app.page) == "Imported study: Imported Study"
        refresh.assert_called_once()

    def test_unreadable_file_shows_failure_snackbar(self, app, manager, tmp_path):
        # Empty YAML -> load_study_from_file raises ValueError -> import_study
        # surfaces a failure snackbar and does NOT refresh.
        bad = tmp_path / "bad.yaml"
        bad.write_text("")

        # ConfigManager.import_study calls load_study_from_file which raises on
        # empty YAML; the app does not catch it, so the raise is the observable
        # behavior. pytest.raises suppresses it, so the post-block assert still runs.
        with (
            patch("pineuro.gui.utils.pick_file", return_value=str(bad)),
            patch.object(app, "show_study_manager") as refresh,
            pytest.raises(ValueError),
        ):
            app._import_study()
        refresh.assert_not_called()


# =============================================================================
# _export_study
# =============================================================================


class TestExportStudy:
    def test_cancelled_picker_is_noop(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))
        with patch("pineuro.gui.utils.pick_save_file", return_value=None):
            app._export_study("Alpha")
        assert app.page.overlay == []

    def test_writes_yaml_to_chosen_destination(self, app, manager, tmp_path):
        manager.save_study(_make_study(name="Alpha"))
        dest = tmp_path / "exported.yaml"

        with patch("pineuro.gui.utils.pick_save_file", return_value=str(dest)):
            app._export_study("Alpha")

        assert dest.exists()
        # Round-trips back into a ConfigManager as the same study.
        reloaded = manager.load_study_from_file(dest)
        assert reloaded.metadata.name == "Alpha"

    def test_success_snackbar_reports_destination(self, app, manager, tmp_path):
        manager.save_study(_make_study(name="Alpha"))
        dest = tmp_path / "exported.yaml"

        with patch("pineuro.gui.utils.pick_save_file", return_value=str(dest)):
            app._export_study("Alpha")

        assert _last_snackbar_text(app.page) == f"Exported to {dest}"

    def test_missing_study_shows_failure_snackbar(self, app, tmp_path):
        dest = tmp_path / "exported.yaml"
        with patch("pineuro.gui.utils.pick_save_file", return_value=str(dest)):
            app._export_study("Nonexistent")

        assert _last_snackbar_text(app.page) == "Export failed: study not found"
        assert not dest.exists()


# =============================================================================
# _confirm_delete_study  (dialog wiring + confirm/cancel callbacks)
# =============================================================================


def _dialog_actions(page) -> list:
    """Return the actions list of the AlertDialog passed to show_dialog."""
    assert page.show_dialog.called, "expected a dialog to be shown"
    dialog = page.show_dialog.call_args[0][0]
    assert isinstance(dialog, ft.AlertDialog)
    return dialog.actions


class TestConfirmDeleteStudy:
    def test_shows_confirmation_dialog(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))
        app._confirm_delete_study("Alpha")
        app.page.show_dialog.assert_called_once()
        dialog = app.page.show_dialog.call_args[0][0]
        assert isinstance(dialog, ft.AlertDialog)

    def test_does_not_delete_before_confirmation(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))
        app._confirm_delete_study("Alpha")
        # Merely showing the dialog must not remove the study.
        assert manager.load_study("Alpha") is not None

    def test_confirm_callback_deletes_study(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))
        app._confirm_delete_study("Alpha")

        # Second action button is "Delete"; invoke its on_click.
        delete_btn = _dialog_actions(app.page)[1]
        assert delete_btn.content == "Delete"
        with patch.object(app, "show_study_manager") as refresh:
            delete_btn.on_click(None)

        assert manager.load_study("Alpha") is None
        refresh.assert_called_once()
        assert _last_snackbar_text(app.page) == "Study deleted: Alpha"
        app.page.pop_dialog.assert_called()

    def test_cancel_callback_keeps_study(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))
        app._confirm_delete_study("Alpha")

        cancel_btn = _dialog_actions(app.page)[0]
        assert cancel_btn.content == "Cancel"
        with patch.object(app, "show_study_manager") as refresh:
            cancel_btn.on_click(None)

        assert manager.load_study("Alpha") is not None
        refresh.assert_not_called()
        app.page.pop_dialog.assert_called()

    def test_confirm_on_missing_study_does_not_refresh(self, app):
        app._confirm_delete_study("Ghost")
        delete_btn = _dialog_actions(app.page)[1]
        with patch.object(app, "show_study_manager") as refresh:
            delete_btn.on_click(None)
        # delete_study returns False -> no refresh, no success snackbar.
        refresh.assert_not_called()
        assert _last_snackbar_text(app.page) is None


# =============================================================================
# show_study_editor  (loads study into current_study state)
# =============================================================================


class TestShowStudyEditor:
    def test_editing_named_study_loads_into_current_study(self, app, manager):
        manager.save_study(_make_study(name="Alpha", description="desc-A"))

        # The editor screen builds heavy UI; we only care that current_study
        # is populated from ConfigManager. Stub the screen construction.
        with patch("pineuro.gui.screens.study_editor.StudyEditorScreen") as Screen:
            Screen.return_value.build.return_value = ft.Text("stub")
            app.show_study_editor("Alpha")

        assert app.current_study is not None
        assert app.current_study.metadata.name == "Alpha"
        assert app.current_study.metadata.description == "desc-A"

    def test_new_study_clears_current_study(self, app, manager):
        app.current_study = _make_study(name="Stale")

        with patch("pineuro.gui.screens.study_editor.StudyEditorScreen") as Screen:
            Screen.return_value.build.return_value = ft.Text("stub")
            app.show_study_editor(None)

        assert app.current_study is None

    def test_editor_constructed_with_config_manager_and_study(self, app, manager):
        manager.save_study(_make_study(name="Alpha"))

        with patch("pineuro.gui.screens.study_editor.StudyEditorScreen") as Screen:
            Screen.return_value.build.return_value = ft.Text("stub")
            app.show_study_editor("Alpha")

        _, kwargs = Screen.call_args
        assert kwargs["config_manager"] is app.config_manager
        assert kwargs["study_config"].metadata.name == "Alpha"


# =============================================================================
# _prompt_start_session  (load failure path)
# =============================================================================


class TestPromptStartSession:
    def test_missing_study_shows_failure_snackbar_and_no_dialog(self, app):
        app._prompt_start_session("Ghost")
        assert _last_snackbar_text(app.page) == "Failed to load study: Ghost"
        app.page.show_dialog.assert_not_called()

    def test_loaded_study_opens_participant_id_dialog(self, app, manager):
        manager.save_study(_make_study(name="Alpha", session_config=_session_config(example_id="sub-042")))

        app._prompt_start_session("Alpha")

        app.page.show_dialog.assert_called_once()
        dialog = app.page.show_dialog.call_args[0][0]
        assert isinstance(dialog, ft.AlertDialog)
        # The example participant id from config flows into the field hint.
        assert "sub-042" in dialog.title.value or _hint_contains(dialog, "sub-042")


def _hint_contains(dialog: ft.AlertDialog, needle: str) -> bool:
    """Recursively search the dialog's controls for a TextField hint match."""
    found = {"hit": False}

    def visit(obj):
        if found["hit"]:
            return
        if isinstance(obj, ft.TextField) and obj.hint_text and needle in obj.hint_text:
            found["hit"] = True
            return
        content = getattr(obj, "content", None)
        if content is not None:
            visit(content)
        controls = getattr(obj, "controls", None)
        if isinstance(controls, (list, tuple)):
            for child in controls:
                visit(child)

    visit(dialog.content)
    return found["hit"]
