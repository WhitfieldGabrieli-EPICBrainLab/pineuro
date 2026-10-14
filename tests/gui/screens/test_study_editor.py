"""Tests for StudyEditorScreen master-detail navigation and session summaries."""

from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from pineuro.gui.models import StudyConfig, StudyMetadata
from pineuro.gui.screens.study_editor import StudyEditorScreen


def _session(localizers: list[dict], feedback_runs: list[dict], notes: str = "") -> dict:
    return {"notes": notes, "localizers": localizers, "feedback_runs": feedback_runs}


def _rest_loc() -> dict:
    return {"type": "rest", "tr": 1.2, "n_volumes": 250, "n_runs": 2, "networks": ["DefaultA", "ContA"]}


def _ball_fb(mask_sources: dict | None = None) -> dict:
    frc: dict = {"task": "ball", "tr": 1.2, "n_volumes": 150, "n_baseline": 25, "max_runs": 1}
    if mask_sources is not None:
        frc["mask_sources"] = mask_sources
    return frc


def _make_editor(sessions: list[dict]) -> StudyEditorScreen:
    """A StudyEditorScreen over the given sessions, with a lightweight (sessionless) page."""
    session_config = {
        "paths": {"base_dir": "/data/pineuro", "example_participant_id": "sub-001"},
        "simulator": {"enabled": True},
        "scanner": {},
        "reference": {},
        "display": {},
        "sessions": sessions,
    }
    cfg = StudyConfig(metadata=StudyMetadata(name="Test Study"), session_config=session_config)
    page = ft.Page.__new__(ft.Page)  # lightweight, no real session
    return StudyEditorScreen(page, MagicMock(), study_config=cfg)


# ----------------------------------------------------------------------
# _session_summary fingerprints
# ----------------------------------------------------------------------


def test_session_summary_localizer_and_feedback():
    editor = _make_editor([_session([_rest_loc()], [_ball_fb()])])
    summary = editor._session_summary(0)
    assert "Rest" in summary
    assert "Ball" in summary
    assert "<-" not in summary  # no cross-session reuse


def test_session_summary_marks_cross_session_reuse():
    sessions = [
        _session([_rest_loc()], [_ball_fb()]),
        _session([_rest_loc()], [_ball_fb(mask_sources={"DefaultA": "ses-01"})]),
    ]
    editor = _make_editor(sessions)
    assert "<- ses-01" in editor._session_summary(1)
    # The producing session itself shows no reuse marker.
    assert "<-" not in editor._session_summary(0)


def test_session_summary_empty_session():
    editor = _make_editor([_session([], [])])
    assert editor._session_summary(0) == "(empty)"


def test_session_summary_manual_and_unknown_types():
    sessions = [_session([{"type": "manual"}, {"type": "totally_unknown_type"}], [])]
    editor = _make_editor(sessions)
    summary = editor._session_summary(0)
    assert "Manual" in summary
    assert "totally_unknown_type" in summary  # falls back to the raw type id


def test_session_summary_out_of_range_is_empty_string():
    editor = _make_editor([_session([_rest_loc()], [_ball_fb()])])
    assert editor._session_summary(5) == ""


def test_loc_type_short():
    editor = _make_editor([_session([_rest_loc()], [])])
    assert editor._loc_type_short("rest") == "Rest"
    assert editor._loc_type_short("manual") == "Manual"
    assert editor._loc_type_short("nope") == "nope"


# ----------------------------------------------------------------------
# Scope transitions (master-detail navigation)
# ----------------------------------------------------------------------


def test_existing_study_opens_on_study_wide():
    # Every study — new or existing — opens on study-wide settings; the active
    # session still defaults to the first session for when the user drills in.
    editor = _make_editor([_session([_rest_loc()], [_ball_fb()])])
    assert editor._active_scope == "study"
    assert editor._active_session_index == 0


def test_new_study_opens_on_study_wide():
    page = ft.Page.__new__(ft.Page)
    editor = StudyEditorScreen(page, MagicMock(), study_config=None)
    assert editor.is_new
    assert editor._active_scope == "study"


# NOTE: these build() the editor. FeedbackTabBuilder touches the Flet event loop on
# construction, which the lightweight ``ft.Page.__new__`` test double lacks, so these
# use localizer-only sessions — the scope-navigation logic under test is independent of
# whether a session has feedback runs.


def test_select_study_wide_then_session():
    editor = _make_editor(
        [
            _session([_rest_loc()], []),
            _session([_rest_loc()], []),
        ]
    )
    editor.build()  # wires sidebar + detail pane

    editor._select_study_wide()
    assert editor._active_scope == "study"

    editor._on_select_session(1)
    assert editor._active_scope == "session"
    assert editor._active_session_index == 1

    # Selecting the active session from session scope is a no-op (stays put).
    editor._on_select_session(1)
    assert editor._active_session_index == 1


def test_inner_tab_index_is_clamped_to_tab_count():
    # 1 localizer + 0 feedback -> inner tabs = [Protocol, Localizer] = 2.
    editor = _make_editor([_session([_rest_loc()], [])])
    editor.build()
    editor._on_select_session(0)  # editor opens on study-wide; switch into the session scope
    editor._last_inner_index = 99
    editor._render_detail()
    assert editor._inner_tabs.selected_index == 1  # min(99, 2 - 1)


# ----------------------------------------------------------------------
# Localizer type change applies the task's default run count
# ----------------------------------------------------------------------


def _editor_for_type_change(loc: dict, new_type: str) -> StudyEditorScreen:
    """Editor primed to switch its single localizer to ``new_type`` via the dropdown.

    The UI-touching collect/rebuild steps are stubbed so the test isolates the
    type-change defaulting logic.
    """
    editor = _make_editor([_session([_rest_loc()], [])])
    editor._localizer_configs = [loc]
    dropdown = MagicMock()
    dropdown.value = new_type
    editor._loc_type_dropdowns = [dropdown]
    editor._collect_all_configs = MagicMock()
    editor._rebuild_tabs = MagicMock()
    return editor


def test_type_change_to_rit_defaults_to_one_run():
    # RIT declares default_n_runs=1, so selecting it should set n_runs to 1.
    editor = _editor_for_type_change({"type": "rest", "n_runs": 2}, "rit")
    editor._on_localizer_type_changed(0, None)
    assert editor._localizer_configs[0]["type"] == "rit"
    assert editor._localizer_configs[0]["n_runs"] == 1


def test_type_change_to_other_task_keeps_default_two_runs():
    # Tasks without a default_n_runs override fall back to the metadata default of 2.
    editor = _editor_for_type_change({"type": "rest", "n_runs": 5}, "selfother")
    editor._on_localizer_type_changed(0, None)
    assert editor._localizer_configs[0]["type"] == "selfother"
    assert editor._localizer_configs[0]["n_runs"] == 2


# ----------------------------------------------------------------------
# Backup settings
# ----------------------------------------------------------------------


def test_untouched_study_gains_no_backup_block():
    # Writing defaults would make every existing study look edited ("unsaved changes").
    editor = _make_editor([_session([_rest_loc()], [])])
    editor.build()  # creates the tab builders the form sync reads
    editor._update_study_from_form()
    assert "backup" not in editor.study_config.session_config


def test_enabling_backup_writes_the_block():
    editor = _make_editor([_session([_rest_loc()], [])])
    editor.build()
    editor.backup_enabled_checkbox.value = True
    editor.backup_destination_field.value = "  /Volumes/LabBackup  "
    editor._update_study_from_form()
    assert editor.study_config.session_config["backup"] == {
        "enabled": True,
        "destination": "/Volumes/LabBackup",
        "include_dicoms": True,
    }


def test_save_refuses_enabled_backup_without_a_folder():
    editor = _make_editor([_session([_rest_loc()], [])])
    editor.build()
    editor.page = MagicMock()  # snackbar plumbing only
    editor.backup_enabled_checkbox.value = True
    editor.backup_destination_field.value = ""

    assert editor._save_study() is False
    editor.config_manager.save_study.assert_not_called()
