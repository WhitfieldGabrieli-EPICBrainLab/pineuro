"""Behavioral tests for the mask selection dialog's pure selection logic.

The dialog is a large two-stage Flet UI. We do NOT stand up the real dialog or
test rendering. Instead we exercise the testable, observable logic:

* ``__init__`` bootstrap of ``self._selections`` (default MaskSelection per
  network, plus reuse of any selections already on disk).
* Stage-2 single-select handler (``_on_select_alternative_clicked``) building
  the right MaskSelection for the ICA-component and Yeo-atlas branches, and
  refusing to act when nothing is selected.
* Stage-1 accept handler restoring/keeping a selection.
* Combine-mode checkbox state helpers (selected-component readout and the
  max-2 enforcement that unchecks an over-selection).
* ``_apply_and_close`` modified-network detection (which masks get
  regenerated vs. left alone) and the save/callback contract.

Flet rendering (``_show_stage1`` / ``page.update``) and the heavy
mask-regeneration calls are mocked; the selection logic under test is not.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pineuro.gui.components.mask_selection_dialog import (
    MaskReviewInfo,
    MaskSelectionDialog,
)
from pineuro.mask_extraction import MaskSelection, save_mask_selections


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _info(
    name: str,
    *,
    current_component: int = 3,
    current_correlation: float = 0.20,
    alternatives: list[tuple[int, float]] | None = None,
    accepted: bool = True,
) -> MaskReviewInfo:
    """Build a MaskReviewInfo with no real overlay paths on disk."""
    return MaskReviewInfo(
        network_name=name,
        current_component=current_component,
        current_correlation=current_correlation,
        alternative_components=alternatives if alternatives is not None else [(current_component, current_correlation)],
        overlay_path=None,
        alternative_overlay_paths={},
        yeo_overlay_path=None,
        accepted=accepted,
    )


def _make_dialog(masks_info: list[MaskReviewInfo], masks_dir: Path, **overrides) -> MaskSelectionDialog:
    """Construct a real MaskSelectionDialog backed by a mock page.

    ``masks_dir`` is a real (possibly empty) directory; ``load_mask_selections``
    reads from it during ``__init__``.
    """
    page = MagicMock()
    page.overlay = []
    page.update = MagicMock()
    kwargs = dict(
        page=page,
        masks_info=masks_info,
        participant_id="sub-01",
        session_id="ses-01",
        masks_dir=masks_dir,
        components_file=masks_dir / "components.nii.gz",
        yeo_subject_space=masks_dir / "yeo2sub.nii.gz",
    )
    kwargs.update(overrides)
    return MaskSelectionDialog(**kwargs)


# --------------------------------------------------------------------------- #
# __init__ selection bootstrap
# --------------------------------------------------------------------------- #
class TestInitSelectionBootstrap:
    def test_creates_default_ica_selection_per_network(self, tmp_path):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        dlg = _make_dialog(
            [
                _info("DefaultA", current_component=7, current_correlation=0.42),
                _info("ContA", current_component=2, current_correlation=0.11),
            ],
            masks_dir,
        )

        assert set(dlg._selections) == {"DefaultA", "ContA"}
        default_a = dlg._selections["DefaultA"]
        assert default_a.source_type == "ica"
        assert default_a.component_index == 7
        assert default_a.correlation == pytest.approx(0.42)
        assert default_a.selection_reason == "auto"
        # Second network gets its own component/correlation.
        assert dlg._selections["ContA"].component_index == 2

    def test_existing_disk_selection_overrides_default(self, tmp_path):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        # A manually-chosen yeo_atlas fallback already saved on disk.
        save_mask_selections(
            masks_dir,
            {
                "DefaultA": MaskSelection(
                    network_name="DefaultA",
                    source_type="yeo_atlas",
                    component_index=None,
                    correlation=0.0,
                    selection_reason="yeo_fallback",
                )
            },
        )

        dlg = _make_dialog([_info("DefaultA", current_component=7, current_correlation=0.42)], masks_dir)

        sel = dlg._selections["DefaultA"]
        assert sel.source_type == "yeo_atlas"
        assert sel.selection_reason == "yeo_fallback"
        assert sel.component_index is None

    def test_masks_info_indexed_by_network_name(self, tmp_path):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        dlg = _make_dialog([_info("DefaultA"), _info("ContA")], masks_dir)
        assert set(dlg.masks_info) == {"DefaultA", "ContA"}
        assert dlg.masks_info["DefaultA"].network_name == "DefaultA"


# --------------------------------------------------------------------------- #
# Stage-2 single-select: _on_select_alternative_clicked
# --------------------------------------------------------------------------- #
class TestSelectAlternativeSingleSelect:
    def _dialog_in_stage2(self, tmp_path, network="DefaultA", alternatives=None):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir(exist_ok=True)
        alts = alternatives if alternatives is not None else [(3, 0.20), (5, 0.18), (9, 0.05)]
        dlg = _make_dialog([_info(network, current_component=3, alternatives=alts)], masks_dir)
        dlg._stage = 2
        dlg._current_reject_network = network
        dlg._combine_mode = False
        dlg._dialog = MagicMock()
        # Avoid rebuilding the real stage-1 Flet UI; the selection logic still runs.
        dlg._show_stage1 = MagicMock()
        return dlg

    def test_selecting_ica_component_builds_ica_selection_with_its_correlation(self, tmp_path):
        dlg = self._dialog_in_stage2(tmp_path)
        dlg._alternative_selection = MagicMock()
        dlg._alternative_selection.value = "ica_5"

        dlg._on_select_alternative_clicked(None)

        sel = dlg._selections["DefaultA"]
        assert sel.source_type == "ica"
        assert sel.component_index == 5
        assert sel.correlation == pytest.approx(0.18)
        assert sel.selection_reason == "manual"
        # Mask marked accepted and we returned to stage 1.
        assert dlg.masks_info["DefaultA"].accepted is True
        assert dlg._stage == 1
        dlg._show_stage1.assert_called_once()

    def test_selecting_yeo_atlas_builds_fallback_selection(self, tmp_path):
        dlg = self._dialog_in_stage2(tmp_path)
        dlg._alternative_selection = MagicMock()
        dlg._alternative_selection.value = "yeo_atlas"

        dlg._on_select_alternative_clicked(None)

        sel = dlg._selections["DefaultA"]
        assert sel.source_type == "yeo_atlas"
        assert sel.component_index is None
        assert sel.correlation == pytest.approx(0.0)
        assert sel.selection_reason == "yeo_fallback"

    def test_unknown_component_gets_zero_correlation(self, tmp_path):
        # Component index not present in alternative_components → corr defaults to 0.0.
        dlg = self._dialog_in_stage2(tmp_path, alternatives=[(3, 0.2)])
        dlg._alternative_selection = MagicMock()
        dlg._alternative_selection.value = "ica_99"

        dlg._on_select_alternative_clicked(None)

        sel = dlg._selections["DefaultA"]
        assert sel.component_index == 99
        assert sel.correlation == pytest.approx(0.0)

    def test_no_selection_shows_warning_and_does_not_change_selection(self, tmp_path):
        dlg = self._dialog_in_stage2(tmp_path)
        before = dlg._selections["DefaultA"]
        dlg._alternative_selection = MagicMock()
        dlg._alternative_selection.value = None  # nothing picked

        dlg._on_select_alternative_clicked(None)

        # Selection unchanged, no stage transition, a snackbar was surfaced.
        assert dlg._selections["DefaultA"] is before
        dlg._show_stage1.assert_not_called()
        assert len(dlg.page.overlay) == 1  # warning SnackBar appended

    def test_no_current_reject_network_is_a_noop(self, tmp_path):
        dlg = self._dialog_in_stage2(tmp_path)
        dlg._current_reject_network = None
        dlg._alternative_selection = MagicMock()
        dlg._alternative_selection.value = "ica_5"

        # Should return early without touching selections or rebuilding UI.
        dlg._on_select_alternative_clicked(None)
        dlg._show_stage1.assert_not_called()


# --------------------------------------------------------------------------- #
# Stage-1 accept: _on_accept_clicked
# --------------------------------------------------------------------------- #
class TestAcceptClicked:
    def test_accept_marks_info_accepted_and_keeps_existing_selection(self, tmp_path):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        dlg = _make_dialog([_info("DefaultA", current_component=3, accepted=False)], masks_dir)
        dlg._refresh_stage1 = MagicMock()  # don't rebuild Flet UI

        # Pre-set a manual selection; accepting must not clobber it.
        manual = MaskSelection(
            network_name="DefaultA",
            source_type="ica",
            component_index=5,
            correlation=0.18,
            selection_reason="manual",
        )
        dlg._selections["DefaultA"] = manual

        dlg._on_accept_clicked("DefaultA")

        assert dlg.masks_info["DefaultA"].accepted is True
        assert dlg._selections["DefaultA"] is manual  # untouched
        dlg._refresh_stage1.assert_called_once()

    def test_accept_recreates_selection_when_missing(self, tmp_path):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        dlg = _make_dialog([_info("DefaultA", current_component=4, current_correlation=0.33)], masks_dir)
        dlg._refresh_stage1 = MagicMock()
        del dlg._selections["DefaultA"]

        dlg._on_accept_clicked("DefaultA")

        sel = dlg._selections["DefaultA"]
        assert sel.source_type == "ica"
        assert sel.component_index == 4
        assert sel.correlation == pytest.approx(0.33)
        assert sel.selection_reason == "auto"


# --------------------------------------------------------------------------- #
# Combine-mode checkbox state
# --------------------------------------------------------------------------- #
class TestCombineCheckboxState:
    def _dialog(self, tmp_path):
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir(exist_ok=True)
        return _make_dialog([_info("DefaultA", alternatives=[(1, 0.2), (2, 0.15), (3, 0.1)])], masks_dir)

    def test_get_selected_combine_components_returns_checked_indices(self, tmp_path):
        dlg = self._dialog(tmp_path)
        dlg._combine_checkboxes = {
            1: MagicMock(value=True),
            2: MagicMock(value=False),
            3: MagicMock(value=True),
        }
        assert sorted(dlg._get_selected_combine_components()) == [1, 3]

    def test_get_selected_empty_when_none_checked(self, tmp_path):
        dlg = self._dialog(tmp_path)
        dlg._combine_checkboxes = {1: MagicMock(value=False), 2: MagicMock(value=False)}
        assert dlg._get_selected_combine_components() == []

    def test_third_selection_is_rejected_and_unchecked(self, tmp_path):
        dlg = self._dialog(tmp_path)
        cb1, cb2, cb3 = MagicMock(value=True), MagicMock(value=True), MagicMock(value=True)
        dlg._combine_checkboxes = {1: cb1, 2: cb2, 3: cb3}

        # User just checked the third box (idx=3) → over the limit.
        dlg._on_combine_checkbox_changed(MagicMock(), "DefaultA", 3)

        # The just-checked box is forced back off; a warning snackbar is surfaced.
        assert cb3.value is False
        assert len(dlg.page.overlay) == 1

    def test_second_selection_is_allowed(self, tmp_path):
        dlg = self._dialog(tmp_path)
        cb1, cb2 = MagicMock(value=True), MagicMock(value=True)
        dlg._combine_checkboxes = {1: cb1, 2: cb2}

        dlg._on_combine_checkbox_changed(MagicMock(), "DefaultA", 2)

        # Within the limit: nothing forced off, no warning snackbar appended.
        assert cb2.value is True
        assert dlg.page.overlay == []


# --------------------------------------------------------------------------- #
# Accept-all and apply: modified-network detection & save/callback contract
# --------------------------------------------------------------------------- #
class TestApplyAndClose:
    def _patch_regen(self, monkeypatch):
        """Mock the heavy regeneration call; record which networks were regenerated."""
        import pineuro.mask_extraction as me

        regenerated: list[str] = []

        def _fake_regen(*, selection, **kwargs):
            regenerated.append(selection.network_name)

        monkeypatch.setattr(me, "regenerate_mask_from_selection", _fake_regen)
        return regenerated

    def test_accept_all_marks_all_accepted_and_invokes_complete_callback(self, tmp_path, monkeypatch):
        self._patch_regen(monkeypatch)
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        completed: dict = {}

        dlg = _make_dialog(
            [_info("DefaultA", accepted=False), _info("ContA", accepted=False)],
            masks_dir,
            on_complete=lambda sels: completed.update(sels),
            ref_file=None,  # skip overlay regeneration branch
        )
        dlg._close_dialog = MagicMock()

        dlg._on_accept_all_clicked(None)

        # Both masks accepted.
        assert all(info.accepted for info in dlg.masks_info.values())
        # Completion callback fired with the selections dict, and it was persisted.
        assert set(completed) == {"DefaultA", "ContA"}
        assert (masks_dir / "mask_selections.json").exists()

    def test_only_changed_selections_are_regenerated(self, tmp_path, monkeypatch):
        regenerated = self._patch_regen(monkeypatch)
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        # Disk has an auto ICA selection for both networks.
        save_mask_selections(
            masks_dir,
            {
                "DefaultA": MaskSelection("DefaultA", "ica", 3, 0.20, "auto"),
                "ContA": MaskSelection("ContA", "ica", 2, 0.11, "auto"),
            },
        )
        dlg = _make_dialog(
            [_info("DefaultA", current_component=3), _info("ContA", current_component=2)],
            masks_dir,
            ref_file=None,
        )
        dlg._close_dialog = MagicMock()

        # Change only DefaultA (different component); leave ContA matching disk.
        dlg._selections["DefaultA"] = MaskSelection("DefaultA", "ica", 8, 0.30, "manual")
        # ContA selection still matches what __init__ loaded from disk.

        dlg._apply_and_close()

        assert regenerated == ["DefaultA"]

    def test_new_non_auto_selection_is_regenerated(self, tmp_path, monkeypatch):
        regenerated = self._patch_regen(monkeypatch)
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        # No file on disk → all selections are "new".
        dlg = _make_dialog([_info("DefaultA", current_component=3)], masks_dir, ref_file=None)
        dlg._close_dialog = MagicMock()

        # Replace the auto default with a manual yeo fallback (a "new" non-auto selection).
        dlg._selections["DefaultA"] = MaskSelection("DefaultA", "yeo_atlas", None, 0.0, "yeo_fallback")

        dlg._apply_and_close()

        assert regenerated == ["DefaultA"]

    def test_new_auto_selection_is_not_regenerated(self, tmp_path, monkeypatch):
        regenerated = self._patch_regen(monkeypatch)
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        # No file on disk; the default selection __init__ created is reason="auto".
        dlg = _make_dialog([_info("DefaultA", current_component=3)], masks_dir, ref_file=None)
        dlg._close_dialog = MagicMock()

        dlg._apply_and_close()

        assert regenerated == []
        # Selections still get persisted even when nothing was regenerated.
        assert (masks_dir / "mask_selections.json").exists()

    def test_combined_secondary_index_change_triggers_regeneration(self, tmp_path, monkeypatch):
        regenerated = self._patch_regen(monkeypatch)
        masks_dir = tmp_path / "03_masks"
        masks_dir.mkdir()
        save_mask_selections(
            masks_dir,
            {"DefaultA": MaskSelection("DefaultA", "ica", 3, 0.20, "auto")},
        )
        dlg = _make_dialog([_info("DefaultA", current_component=3)], masks_dir, ref_file=None)
        dlg._close_dialog = MagicMock()

        # Same primary component & source differs → ica_combined with a secondary index.
        dlg._selections["DefaultA"] = MaskSelection(
            "DefaultA",
            "ica_combined",
            3,
            0.25,
            "manual_combined",
            secondary_component_index=7,
            secondary_correlation=0.18,
        )

        dlg._apply_and_close()

        assert regenerated == ["DefaultA"]
