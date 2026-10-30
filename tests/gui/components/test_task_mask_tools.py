"""Tests for pineuro.gui.components.task_mask_tools."""

import pytest

from pineuro.gui.components.task_mask_tools import (
    ClusterExcludePanel,
    TaskMaskAdjustPanel,
    build_replace_dialog,
    exclusion_override,
    fallback_label,
    mask_lock_reason,
    preview_summary,
    provenance_label,
    restored_override,
)


class TestMaskLockReason:
    def test_unlocked_before_feedback(self):
        assert mask_lock_reason(["preprocessing", "mask_extraction", "coregistration"]) is None

    @pytest.mark.parametrize("step_id", ["feedback_1", "transfer_2"])
    def test_locked_once_a_feedback_run_completed(self, step_id):
        assert "locked" in mask_lock_reason(["mask_extraction", step_id])


class TestLabels:
    def test_fallback_label_atlas_and_file(self):
        assert fallback_label({"atlas": "yeo_17", "network": "DefaultA"}) == "Yeo 17-network DefaultA"
        assert fallback_label({"file": "/masks/rumination_mni.nii.gz"}) == "rumination_mni.nii.gz"

    def test_provenance_none_for_study_mask(self):
        assert provenance_label({"selection_reason": "auto"}) is None

    def test_provenance_rethreshold_shows_parameters(self):
        sel = {
            "selection_reason": "operator_rethreshold",
            "stat_type": "t",
            "threshold_params": {
                "threshold_method": "cluster_extent",
                "cluster_start_threshold": 1.8,
                "min_cluster_size": 30,
                "max_total_voxels": None,
            },
        }
        assert provenance_label(sel) == "Re-thresholded at review (t ≥ 1.8, min 30 voxels)"

    def test_provenance_replacement_shows_source(self):
        sel = {"selection_reason": "fallback_configured", "source": "yeo_17:DefaultA"}
        assert provenance_label(sel) == "Study fallback mask: yeo_17:DefaultA"

    def test_provenance_exclusion_shows_parameters_and_what_was_removed(self):
        sel = {
            "selection_reason": "operator_exclude_clusters",
            "stat_type": "t",
            "threshold_params": {
                "threshold_method": "cluster_extent",
                "cluster_start_threshold": 1.5,
                "min_cluster_size": 50,
                "max_total_voxels": 2000,
            },
            "excluded_clusters": [{"n_voxels": 95, "peak_stat": 3.18, "peak_ijk": [48, 92, 62]}],
        }
        assert provenance_label(sel) == (
            "Clusters excluded at review (t ≥ 1.5, min 50 voxels, cap 2000): 1 cluster, 95 voxels"
        )


class TestExclusionOverride:
    def test_from_the_study_mask(self):
        assert exclusion_override(None, [[1, 2, 3]]) == {"kind": "exclude_clusters", "peaks": [[1, 2, 3]]}

    def test_keeps_threshold_values(self):
        current = {"kind": "rethreshold", "cluster_start_threshold": 1.5, "min_cluster_size": 50}
        assert exclusion_override(current, [[1, 2, 3]]) == {
            "kind": "exclude_clusters",
            "peaks": [[1, 2, 3]],
            "cluster_start_threshold": 1.5,
            "min_cluster_size": 50,
        }

    def test_adds_to_earlier_exclusions_without_duplicates(self):
        current = {"kind": "exclude_clusters", "peaks": [[1, 2, 3]], "min_cluster_size": 50}
        override = exclusion_override(current, [[1, 2, 3], [4, 5, 6]])
        assert override["peaks"] == [[1, 2, 3], [4, 5, 6]]
        assert override["min_cluster_size"] == 50

    def test_replacement_override_contributes_nothing(self):
        current = {"kind": "native_file", "file": "/masks/net.nii.gz"}
        assert exclusion_override(current, [[1, 2, 3]]) == {"kind": "exclude_clusters", "peaks": [[1, 2, 3]]}

    def test_restore_keeps_threshold_values(self):
        current = {"kind": "exclude_clusters", "peaks": [[1, 2, 3]], "min_cluster_size": 50}
        assert restored_override(current) == {"kind": "rethreshold", "min_cluster_size": 50}

    def test_restore_without_threshold_values_is_the_study_mask(self):
        assert restored_override({"kind": "exclude_clusters", "peaks": [[1, 2, 3]]}) is None
        assert restored_override(None) is None


class TestClusterExcludePanel:
    PEAKS = ([1, 2, 3], [4, 5, 6])

    def test_actions_need_a_ticked_cluster(self):
        panel = ClusterExcludePanel(2)
        assert panel.preview_button.disabled and panel.apply_button.disabled
        panel.set_ticked(self.PEAKS[0], True)
        assert not panel.preview_button.disabled and not panel.apply_button.disabled
        panel.set_ticked(self.PEAKS[0], False)
        assert panel.apply_button.disabled

    def test_every_cluster_cannot_be_excluded(self):
        panel = ClusterExcludePanel(2)
        panel.set_ticked(self.PEAKS[0], True)
        panel.set_ticked(self.PEAKS[1], True)
        assert panel.apply_button.disabled and panel.preview_button.disabled
        assert "must stay" in panel.apply_button.tooltip

    def test_locked_panel_disables_everything(self):
        panel = ClusterExcludePanel(2, has_exclusions=True, locked_reason="Masks are locked")
        panel.set_ticked(self.PEAKS[0], True)
        assert panel.apply_button.disabled and panel.preview_button.disabled
        assert panel.restore_button.disabled and panel.restore_button.visible
        assert panel.tick_box({"peak_ijk": self.PEAKS[0]}).disabled

    def test_callbacks_receive_the_ticked_peaks(self):
        applied, previewed, viewed = [], [], []
        panel = ClusterExcludePanel(2)
        panel.on_apply, panel.on_preview, panel.on_view = applied.append, previewed.append, viewed.append
        cluster = {"peak_ijk": self.PEAKS[0]}
        tick = panel.tick_box(cluster)
        tick.on_change(type("Event", (), {"control": type("Box", (), {"value": True})})())
        panel.apply_button.on_click(None)
        panel.preview_button.on_click(None)
        panel.view_button(cluster).on_click(None)
        assert applied == previewed == [[self.PEAKS[0]]]
        assert viewed == [cluster]

    def test_restore_is_shown_only_with_exclusions(self):
        assert ClusterExcludePanel(2).restore_button.visible is False
        assert ClusterExcludePanel(2, has_exclusions=True).restore_button.visible is True


class TestPreviewSummary:
    def test_surviving_clusters(self):
        sel = {
            "n_voxels_thresholded": 352,
            "n_clusters_found": 3,
            "cluster_table": [{"n_voxels": 200}, {"n_voxels": 100}, {"n_voxels": 52}],
            "cluster_outcome": "success",
        }
        assert preview_summary(sel) == "352 voxels · 3 clusters · largest 200 · success"

    def test_nothing_survives_reports_largest_input_cluster(self):
        sel = {
            "n_voxels_thresholded": 0,
            "n_clusters_input": 14,
            "largest_cluster_input": 31,
            "cluster_outcome": "no_clusters_survive",
        }
        assert preview_summary(sel) == (
            "0 voxels · 14 clusters below the minimum size (largest 31) · no clusters survive"
        )


def _panel(method="cluster_extent", values=None, **kw):
    return TaskMaskAdjustPanel(
        method,
        values or {"cluster_start_threshold": 2.0, "min_cluster_size": 50, "max_total_voxels": None, "threshold": 2000},
        on_preview=lambda o: None,
        on_apply=lambda o: None,
        on_reset=lambda: None,
        **kw,
    )


class TestTaskMaskAdjustPanel:
    def test_fields_follow_the_method(self):
        assert set(_panel("cluster_extent")._fields) == {
            "cluster_start_threshold",
            "min_cluster_size",
            "max_total_voxels",
        }
        assert set(_panel("top_n")._fields) == {"threshold"}

    def test_override_from_fields(self):
        panel = _panel()
        panel._fields["cluster_start_threshold"][0].value = "1.8"
        panel._fields["min_cluster_size"][0].value = "30"
        assert panel.build_override() == {
            "kind": "rethreshold",
            "cluster_start_threshold": 1.8,
            "min_cluster_size": 30,
            "max_total_voxels": None,
        }

    @pytest.mark.parametrize(("field", "value"), [("min_cluster_size", "3.5"), ("cluster_start_threshold", "-1")])
    def test_invalid_values_rejected(self, field, value):
        panel = _panel()
        panel._fields[field][0].value = value
        with pytest.raises(ValueError):
            panel.build_override()

    def test_invalid_value_shows_error_instead_of_calling_back(self):
        calls = []
        panel = TaskMaskAdjustPanel(
            "top_n", {"threshold": 2000}, on_preview=calls.append, on_apply=calls.append, on_reset=lambda: None
        )
        panel._fields["threshold"][0].value = "many"
        panel._emit(calls.append)
        assert calls == []
        assert panel.error_text.visible

    def test_locked_panel_disables_apply(self):
        panel = _panel(locked_reason="Masks are locked", override_active=True)
        assert panel.apply_button.disabled
        assert panel.reset_button.disabled and panel.reset_button.visible


def test_discard_preview_button_follows_pending_state():
    discarded = []
    panel = TaskMaskAdjustPanel(
        "top_n",
        {"threshold": 2000},
        on_preview=lambda o: None,
        on_apply=lambda o: None,
        on_reset=lambda: None,
        on_discard=lambda: discarded.append(True),
    )
    assert panel.discard_button.visible is False
    panel.set_preview_pending(True)
    assert panel.discard_button.visible is True
    panel.discard_button.on_click(None)
    assert discarded == [True]
    panel.set_preview_pending(False)
    assert panel.discard_button.visible is False


def _replace_dialog():
    chosen: list = []
    dialog = build_replace_dialog("rumination_network", on_choose=chosen.append, on_cancel=lambda: None)
    column = dialog.content.content
    _text, source, atlas_row, file_row, _error = column.controls
    return dialog, source, atlas_row, file_row, chosen


def test_replace_dialog_builds():
    dialog, *_ = _replace_dialog()
    assert "rumination_network" in dialog.title.value


def test_replace_dialog_shows_only_the_selected_source():
    _dialog, source, atlas_row, file_row, _chosen = _replace_dialog()
    assert (atlas_row.visible, file_row.visible) == (True, False)

    for file_kind in ("mni_file", "native_file"):
        source.value = file_kind
        source.on_change(None)
        assert (atlas_row.visible, file_row.visible) == (False, True)


@pytest.mark.parametrize("file_kind", ["mni_file", "native_file"])
def test_replace_dialog_returns_the_file_override(file_kind):
    dialog, source, _atlas_row, file_row, chosen = _replace_dialog()
    source.value = file_kind
    source.on_change(None)
    file_row.controls[0].value = "/masks/net.nii.gz"
    dialog.actions[1].on_click(None)
    assert chosen == [{"kind": file_kind, "file": "/masks/net.nii.gz", "selection_reason": "operator_replace"}]


def test_replace_dialog_returns_the_atlas_override():
    dialog, _source, atlas_row, _file_row, chosen = _replace_dialog()
    atlas_row.controls[1].value = "DefaultA"  # the network dropdown
    dialog.actions[1].on_click(None)
    assert chosen == [
        {"kind": "atlas_network", "atlas": "yeo_17", "network": "DefaultA", "selection_reason": "operator_replace"}
    ]
