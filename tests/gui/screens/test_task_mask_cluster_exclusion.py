"""Leaving clusters out of a task mask, through the real review dialog.

The screen is a bare instance (see ``test_participant_workflow.py``) with a page
that runs background work inline; the masks, overrides, reproduce script and QC
report are the real ones, written from a synthetic cached GLM map.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import flet as ft
import nibabel as nib
import numpy as np
import pytest

from pineuro.gui.components.task_mask_tools import ClusterExcludePanel, provenance_label
from pineuro.gui.screens.participant_workflow import ParticipantWorkflowScreen, StepStatus

_PREFIX = "sub-01_ses-01"


class _Page:
    """Runs background work inline, so a click has finished when it returns."""

    def __init__(self):
        self.overlay: list = []

    def update(self):
        pass

    def run_thread(self, fn):
        fn()

    def run_task(self, fn):
        asyncio.run(fn())


def _walk(control):
    """Every control under ``control``."""
    yield control
    children = list(getattr(control, "controls", None) or [])
    if getattr(control, "content", None) is not None:
        children.append(control.content)
    for child in children:
        yield from _walk(child)


class _Review:
    """A participant with a three-cluster task mask, and helpers to drive its review dialog."""

    def __init__(self, tmp_path: Path, stat: np.ndarray | None = None, min_cluster_size: int = 5):
        from pineuro.gui.components.pipeline_runner import build_task_extraction_params
        from pineuro.mask_extraction import update_task_mask

        self.base_dir = tmp_path / "sub-01" / "ses-01"
        self.root = self.base_dir / "mask_extraction"
        self.thr_file = self.root / "03_masks" / f"{_PREFIX}_net_thr.nii.gz"
        (self.root / "01_glm").mkdir(parents=True)
        preprocessing = self.base_dir / "preprocessing"
        preprocessing.mkdir()

        if stat is None:
            stat = np.zeros((16, 16, 16))
            stat[1:5, 1:5, 1:5] = 5.0  # 64 voxels
            stat[8:11, 8:11, 8:11] = 4.0  # 27 voxels
            stat[13:15, 13:15, 13:15] = 3.0  # 8 voxels
        self.stat = stat
        nib.save(nib.Nifti1Image(self.stat, np.eye(4)), self.root / "01_glm" / f"{_PREFIX}_net_tmap.nii.gz")
        boldref = preprocessing / f"{_PREFIX}_task-rit_boldref.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(self.stat.shape), np.eye(4)), boldref)
        (preprocessing / f"{_PREFIX}_task-rit_run-01_attempt-01_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0\t10\ta\n10\t10\tb\n"
        )

        self.config = {
            "paths": {"base_dir": str(self.base_dir)},
            "localizers": [
                {
                    "type": "rit",
                    "contrasts": {"net": "a - b"},
                    "extraction": {"threshold_method": "cluster_extent", "min_cluster_size": min_cluster_size},
                }
            ],
        }
        update_task_mask(build_task_extraction_params(self.config, 0, "sub-01", "ses-01", overwrite=False), "net", None)

        screen = ParticipantWorkflowScreen.__new__(ParticipantWorkflowScreen)
        screen.page = _Page()
        screen.study_config = SimpleNamespace(session_config=self.config)
        screen.participant_id, screen.session_id = "sub-01", "ses-01"
        screen._bids_prefix = _PREFIX
        screen.step_statuses = {"coregistration": StepStatus.COMPLETE}
        screen._add_log = MagicMock()
        screen._refresh_step_list = MagicMock()
        screen._update_run_next_button = MagicMock()
        screen._set_qc_approved = MagicMock()
        screen._show_mask_review_dialog = lambda loc_idx=None: self.open()
        self.screen = screen

    def open(self) -> None:
        self.screen._show_task_mask_review_dialog(
            self.config, self.base_dir, self.root, self.base_dir / "qc", loc_idx=0
        )

    @property
    def dialog(self) -> ft.AlertDialog:
        return [c for c in self.screen.page.overlay if isinstance(c, ft.AlertDialog)][-1]

    @property
    def panel(self) -> ClusterExcludePanel:
        return next(c for c in _walk(self.dialog.content) if isinstance(c, ClusterExcludePanel))

    @property
    def table(self) -> ft.DataTable:
        return next(c for c in _walk(self.dialog.content) if isinstance(c, ft.DataTable))

    @property
    def approve(self) -> ft.Control:
        return self.dialog.actions[-1]

    def tick(self, row: int) -> None:
        box = self.table.rows[row].cells[0].content
        box.value = True
        box.on_change(SimpleNamespace(control=box))

    def voxels(self) -> int:
        return int(np.asarray(nib.load(self.thr_file).dataobj).sum())

    def selection(self) -> dict:
        return json.loads((self.root / "03_masks" / "mask_selections.json").read_text())["selections"]["net"]

    def overrides(self) -> dict:
        path = self.root / "mask_overrides.json"
        return json.loads(path.read_text()) if path.exists() else {}


@pytest.fixture
def review(tmp_path):
    state = _Review(tmp_path)
    state.open()
    return state


def test_table_lists_clusters_with_exclude_controls(review):
    assert review.voxels() == 99
    assert len(review.table.rows) == 3
    assert [c.label.value for c in review.table.columns][0] == "Exclude"
    assert review.panel.restore_button.visible is False


def test_dialog_encodes_for_flet(review):
    """The panel's plain callback attributes must not reach Flet's serializer."""
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.controls.object_patch import ObjectPatch
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    patch, _added, _removed = ObjectPatch.from_diff(None, review.dialog, control_cls=BaseControl)
    encoded = msgpack.packb(patch.to_message(), default=configure_encode_object_for_msgpack(BaseControl))
    assert b"Exclude selected" in encoded
    assert b"on_view" not in encoded


def test_view_shows_one_cluster_and_again_the_whole_mask(review):
    overlay_box = next(c for c in _walk(review.dialog.content) if isinstance(c, ft.Container) and c.clip_behavior)
    whole_mask = overlay_box.content

    review.table.rows[1].cells[-1].content.on_click(None)

    assert overlay_box.content is not whole_mask
    assert "Showing cluster 2 only" in review.panel.result_text.value
    assert review.approve.disabled is False  # viewing changes nothing, so it does not block approval
    assert review.voxels() == 99
    cluster_mask = review.base_dir / "qc" / "overlays" / "subject" / f"{_PREFIX}_net_cluster_mask.nii.gz"
    assert int(np.asarray(nib.load(cluster_mask).dataobj).sum()) == 27

    review.table.rows[1].cells[-1].content.on_click(None)
    assert overlay_box.content is whole_mask


def test_preview_changes_nothing_and_blocks_approval_until_discarded(review):
    before = review.thr_file.read_bytes()
    review.tick(0)

    review.panel.preview_button.on_click(None)

    assert review.thr_file.read_bytes() == before
    assert review.overrides() == {}
    assert "35 voxels" in review.panel.result_text.value
    assert review.approve.disabled is True
    assert review.panel.discard_button.visible is True

    review.panel.discard_button.on_click(None)
    assert review.approve.disabled is False
    assert review.panel.discard_button.visible is False


def test_exclude_then_restore(review):
    table_peak = review.selection()["cluster_table"][0]["peak_ijk"]
    review.tick(0)

    review.panel.apply_button.on_click(None)

    assert review.voxels() == 35
    assert np.all(review.stat[np.asarray(nib.load(review.thr_file).dataobj) > 0] != 5.0)
    assert review.overrides() == {"net": {"kind": "exclude_clusters", "peaks": [table_peak]}}
    assert "exclude_clusters" in (review.root / "reproduce_masks.py").read_text()
    sel = review.selection()
    assert sel["excluded_clusters"] == [{"n_voxels": 64, "peak_stat": 5.0, "peak_ijk": table_peak}]
    assert (
        b"Clusters Excluded at Review: net"
        in (review.base_dir / "qc" / f"{_PREFIX}_mask_extraction_qc.html").read_bytes()
    )
    review.screen._set_qc_approved.assert_called_with("initial", False, loc_idx=None)
    assert review.screen.step_statuses["coregistration"] == StepStatus.PENDING

    # The review reopens with what happened on show and the table open, ready to restore
    assert provenance_label(sel).endswith("1 cluster, 64 voxels")
    assert len(review.table.rows) == 2
    assert review.panel.restore_button.visible is True

    review.panel.restore_button.on_click(None)

    assert review.voxels() == 99
    assert review.overrides() == {}
    assert review.selection()["selection_reason"] == "auto"
    assert review.selection()["excluded_clusters"] is None


def test_exclusions_add_up_and_restore_keeps_an_adjusted_threshold(review):
    from pineuro.gui.components.pipeline_runner import build_task_extraction_params
    from pineuro.mask_extraction import save_mask_overrides, update_task_mask

    rethreshold = {"kind": "rethreshold", "cluster_start_threshold": 2.5, "min_cluster_size": 5}
    params = build_task_extraction_params(review.config, 0, "sub-01", "ses-01", overwrite=False)
    update_task_mask(params, "net", rethreshold)
    save_mask_overrides(review.root, {"net": rethreshold})
    review.open()

    review.tick(2)
    review.panel.apply_button.on_click(None)
    assert review.voxels() == 91
    first = review.overrides()["net"]
    assert first["cluster_start_threshold"] == 2.5

    review.tick(1)
    review.panel.apply_button.on_click(None)
    assert review.voxels() == 64
    assert len(review.overrides()["net"]["peaks"]) == 2
    assert review.selection()["threshold_params"]["cluster_start_threshold"] == 2.5
    assert len(review.table.rows) == 1

    review.panel.restore_button.on_click(None)
    assert review.voxels() == 99
    assert review.overrides() == {"net": rethreshold}
    assert review.selection()["selection_reason"] == "operator_rethreshold"


def _scattered_stat(n_clusters: int) -> np.ndarray:
    """A stat map of ``n_clusters`` separate single voxels."""
    stat = np.zeros((16, 16, 16))
    for ijk in list(np.ndindex(8, 8, 8))[:n_clusters]:
        stat[tuple(2 * v for v in ijk)] = 5.0
    return stat


def test_every_cluster_past_the_old_limit_of_15_can_be_excluded(tmp_path):
    state = _Review(tmp_path, stat=_scattered_stat(20), min_cluster_size=1)
    state.open()
    assert len(state.table.rows) == 20
    last_peak = state.selection()["cluster_table"][-1]["peak_ijk"]

    state.tick(19)
    state.panel.apply_button.on_click(None)

    assert state.voxels() == 19
    assert state.overrides()["net"]["peaks"] == [last_peak]

    # Every remaining cluster is in the QC report too
    report = (state.base_dir / "qc" / f"{_PREFIX}_mask_extraction_qc.html").read_bytes().decode("utf-8", "replace")
    for cluster in state.selection()["cluster_table"]:
        assert f"({', '.join(str(v) for v in cluster['peak_ijk'])})" in report


def test_table_says_what_it_leaves_out_past_the_saved_50(tmp_path):
    state = _Review(tmp_path, stat=_scattered_stat(60), min_cluster_size=1)
    state.open()

    rows = state.table.rows
    assert len(rows) == 51  # the 50 saved clusters and a last row for the rest
    assert rows[-1].cells[2].content.value == "+10"
    assert "not listed" in rows[-1].cells[4].content.value
    assert len(rows[-1].cells) == len(state.table.columns)


def test_controls_hidden_for_a_replaced_mask(tmp_path):
    state = _Review(tmp_path)
    selections_file = state.root / "03_masks" / "mask_selections.json"
    data = json.loads(selections_file.read_text())
    data["selections"]["net"]["source_type"] = "native_file"
    selections_file.write_text(json.dumps(data))

    state.open()

    assert not any(isinstance(c, ClusterExcludePanel) for c in _walk(state.dialog.content))
    assert len(state.table.columns) == 4
    assert len(state.table.rows[0].cells) == 4


def test_locked_after_feedback(tmp_path):
    state = _Review(tmp_path)
    state.screen.step_statuses["feedback_1"] = StepStatus.COMPLETE

    state.open()

    assert state.table.rows[0].cells[0].content.disabled is True
    state.tick(0)
    assert state.panel.apply_button.disabled is True
