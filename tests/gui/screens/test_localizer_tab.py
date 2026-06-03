"""Tests for pineuro.gui.screens._localizer_tab — auto-computed n_volumes display.

Covers the behaviour added by commit d4d964e: when the selected localizer task
exposes ``TaskMetadata.compute_max_run_seconds``, ``_refresh_n_volumes_info``
must update the displayed ceiling from the live TR + task params. Fixed-duration
tasks (e.g. fixation) must not touch the field.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _stub_builder(loc_type, *, tr="1.2", task_params=None, info_visible=True):
    """Construct a LocalizerTabBuilder bypassing __init__; only stubs what _refresh_n_volumes_info needs."""
    from pineuro.gui.screens._localizer_tab import LocalizerTabBuilder

    builder = LocalizerTabBuilder.__new__(LocalizerTabBuilder)
    builder._loc_type = loc_type
    builder.tr_field = SimpleNamespace(value=tr)
    builder.n_volumes_info_text = SimpleNamespace(visible=info_visible, value="")
    builder.config = {"task_params": task_params or {}}
    builder._collect_task_params = MagicMock(return_value=task_params or {})
    builder.page = MagicMock()
    return builder


class TestRefreshNVolumesInfoForVariableDurationTask:
    def test_writes_ceiling_message_for_rit(self):
        # RIT exposes compute_max_run_seconds (worst-case duration depends on params).
        builder = _stub_builder("rit", tr="1.5", task_params={"n_trials_per_run": 4})
        builder._refresh_n_volumes_info()
        # Default RIT 4-trial run ~670s @ TR=1.5 → ceil(447) volumes
        assert "Auto-computed:" in builder.n_volumes_info_text.value
        assert "volumes" in builder.n_volumes_info_text.value
        assert "TR=1.5" in builder.n_volumes_info_text.value
        builder.page.update.assert_called()

    def test_scales_inversely_with_tr(self):
        # Halving TR doubles the volume count for the same task duration.
        b1 = _stub_builder("rit", tr="2.0", task_params={})
        b2 = _stub_builder("rit", tr="1.0", task_params={})
        b1._refresh_n_volumes_info()
        b2._refresh_n_volumes_info()

        def _extract(msg):
            # The message reads like: Auto-computed 335 volumes worst-case run ...
            return int(msg.split(":")[1].strip().split(" ")[0])

        v_tr_2 = _extract(b1.n_volumes_info_text.value)
        v_tr_1 = _extract(b2.n_volumes_info_text.value)
        # math.ceil(D/1.0) is roughly 2× math.ceil(D/2.0); allow ±1 for ceil rounding.
        assert v_tr_1 == pytest.approx(2 * v_tr_2, abs=1)

    def test_invalid_tr_shows_helper_message(self):
        builder = _stub_builder("rit", tr="not-a-number")
        builder._refresh_n_volumes_info()
        assert "set a valid TR" in builder.n_volumes_info_text.value

    def test_zero_tr_shows_helper_message(self):
        builder = _stub_builder("rit", tr="0")
        builder._refresh_n_volumes_info()
        assert "set a valid TR" in builder.n_volumes_info_text.value

    def test_negative_tr_shows_helper_message(self):
        builder = _stub_builder("rit", tr="-1.0")
        builder._refresh_n_volumes_info()
        assert "set a valid TR" in builder.n_volumes_info_text.value


class TestRefreshNVolumesInfoForFixedDurationTask:
    def test_rest_leaves_value_untouched(self):
        # Rest does not declare compute_max_run_seconds → method should return without writing.
        builder = _stub_builder("rest", tr="1.5")
        builder.n_volumes_info_text.value = "(unchanged)"
        builder._refresh_n_volumes_info()
        assert builder.n_volumes_info_text.value == "(unchanged)"


class TestRefreshNVolumesInfoEarlyReturns:
    def test_invisible_info_text_skips_recompute(self):
        # When the info text is hidden, the method should short-circuit before computing.
        builder = _stub_builder("rit", tr="1.5", info_visible=False)
        builder.n_volumes_info_text.value = "(unchanged)"
        builder._refresh_n_volumes_info()
        assert builder.n_volumes_info_text.value == "(unchanged)"
        builder._collect_task_params.assert_not_called()

    def test_unknown_task_silent(self):
        builder = _stub_builder("definitely_not_a_real_task", tr="1.5")
        builder.n_volumes_info_text.value = "(unchanged)"
        builder._refresh_n_volumes_info()
        assert builder.n_volumes_info_text.value == "(unchanged)"


def _manual_builder(config):
    """Construct a real LocalizerTabBuilder for a manual-mask localizer config."""
    import flet as ft

    from pineuro.gui.screens._localizer_tab import LocalizerTabBuilder

    return LocalizerTabBuilder(
        index=0,
        config=config,
        page=MagicMock(),
        create_browse_button=lambda *a, **k: ft.IconButton(),
        create_file_browse_button=lambda *a, **k: ft.IconButton(),
    )


class TestManualMaskImport:
    def test_collect_config_emits_only_participant_agnostic_fields(self):
        # The study config carries only the mask space + network names; the
        # actual files are chosen per participant in the workflow.
        builder = _manual_builder(
            {
                "type": "manual",
                "id": "manual",
                "manual_mask_space": "anat",
                "manual_mask_names": ["DMN", "FPN"],
            }
        )
        out = builder.collect_config()
        assert out["type"] == "manual"
        assert out["manual_mask_space"] == "anat"
        assert out["manual_mask_names"] == ["DMN", "FPN"]
        # No per-participant file paths and no ICA/extraction block in the study config
        for key in ("manual_mask_paths", "manual_ref_file", "manual_anat_file", "extraction"):
            assert key not in out

    def test_get_mask_names_returns_row_names(self):
        builder = _manual_builder({"type": "manual", "manual_mask_names": ["DMN", "FPN"]})
        assert builder.get_mask_names() == ["DMN", "FPN"]

    def test_collect_config_skips_blank_rows(self):
        # An empty config still yields one (blank) row, which collect_config drops.
        builder = _manual_builder({"type": "manual", "manual_mask_names": []})
        out = builder.collect_config()
        assert out["manual_mask_names"] == []

    def test_build_returns_container(self):
        builder = _manual_builder({"type": "manual", "manual_mask_names": ["DMN"]})
        container = builder.build()
        assert container is not None


# ---------------------------------------------------------------------------
# collect_config() across localizer types
# ---------------------------------------------------------------------------


def _builder(config):
    """Construct a real LocalizerTabBuilder for the given localizer config."""
    import flet as ft

    from pineuro.gui.screens._localizer_tab import LocalizerTabBuilder

    return LocalizerTabBuilder(
        index=0,
        config=config,
        page=MagicMock(),
        create_browse_button=lambda *a, **k: ft.IconButton(),
        create_file_browse_button=lambda *a, **k: ft.IconButton(),
    )


class TestCollectConfigRest:
    def test_top_level_scan_fields_round_trip(self):
        builder = _builder(
            {
                "type": "rest",
                "id": "loc1",
                "tr": 1.5,
                "n_volumes": 300,
                "n_runs": 3,
                "motion_monitoring": "software",
                "networks": ["DefaultA", "ContA"],
            }
        )
        out = builder.collect_config()
        assert out["type"] == "rest"
        assert out["id"] == "loc1"
        assert out["tr"] == 1.5
        assert out["n_volumes"] == 300
        assert out["n_runs"] == 3
        assert out["motion_monitoring"] == "software"

    def test_networks_collected_from_visible_dropdowns(self):
        builder = _builder({"type": "rest", "id": "r", "networks": ["DefaultA", "ContA", "SalVentAttnA"]})
        out = builder.collect_config()
        assert out["networks"] == ["DefaultA", "ContA", "SalVentAttnA"]

    def test_networks_truncated_to_n_target(self):
        # Only the first _n_target_networks dropdowns contribute; extras are ignored.
        builder = _builder({"type": "rest", "id": "r", "networks": ["DefaultA", "ContA"]})
        # Stuff a value into a dropdown beyond the target count.
        builder._network_dropdowns[2].value = "SalVentAttnA"
        out = builder.collect_config()
        assert out["networks"] == ["DefaultA", "ContA"]

    def test_blank_network_dropdowns_dropped(self):
        builder = _builder({"type": "rest", "id": "r", "networks": ["DefaultA", "ContA"]})
        builder._network_dropdowns[1].value = None
        out = builder.collect_config()
        assert out["networks"] == ["DefaultA"]

    def test_extraction_has_ica_keys(self):
        builder = _builder(
            {
                "type": "rest",
                "id": "r",
                "extraction": {
                    "network_atlas": "yeo_17",
                    "n_components": 25,
                    "smooth_fwhm": 6.0,
                    "highpass": 0.02,
                },
            }
        )
        ex = builder.collect_config()["extraction"]
        assert ex["network_atlas"] == "yeo_17"
        assert ex["n_components"] == 25
        assert ex["smooth_fwhm"] == 6.0
        assert ex["highpass"] == 0.02
        assert ex["lowpass"] is None

    def test_rest_has_no_task_keys(self):
        out = _builder({"type": "rest", "id": "r"}).collect_config()
        for key in ("contrasts", "stat_type", "hrf_model", "drift_model", "task_params", "include_reverse_contrasts"):
            assert key not in out

    def test_defaults_when_config_minimal(self):
        # Empty scan fields → documented defaults; id falls back to loc_type.
        out = _builder({"type": "rest"}).collect_config()
        assert out["id"] == "rest"
        assert out["tr"] == 1.2
        assert out["n_volumes"] == 250
        assert out["n_runs"] == 2
        assert out["motion_monitoring"] == "siemens_moco"
        assert out["extraction"]["network_atlas"] == "yeo_17"
        assert out["extraction"]["n_components"] == 30
        assert out["extraction"]["smooth_fwhm"] == 5.0
        assert out["extraction"]["highpass"] == 0.01

    def test_threshold_defaults_to_top_n(self):
        ex = _builder({"type": "rest", "id": "r"}).collect_config()["extraction"]
        assert ex["threshold"] == 2000
        assert ex["threshold_method"] == "top_n"

    def test_source_dicoms_preserved(self):
        builder = _builder({"type": "rest", "id": "r", "source_dicoms": "/data/run1"})
        assert builder.collect_config()["source_dicoms"] == "/data/run1"


class TestCollectConfigClusterMethods:
    def test_cluster_method_emits_cluster_keys(self):
        builder = _builder(
            {
                "type": "rest",
                "id": "r",
                "extraction": {
                    "threshold_method": "cluster",
                    "atlas_source": "harvard_oxford",
                    "atlas_regions": ["Frontal Pole"],
                    "cluster_start_threshold": 3.0,
                    "cluster_step": 0.2,
                    "cluster_return_all": True,
                },
            }
        )
        ex = builder.collect_config()["extraction"]
        assert ex["threshold_method"] == "cluster"
        assert ex["atlas_source"] == "harvard_oxford"
        assert ex["atlas_regions"] == ["Frontal Pole"]
        assert ex["cluster_start_threshold"] == 3.0
        assert ex["cluster_step"] == 0.2
        assert ex["cluster_return_all"] is True

    def test_cluster_loading_sentinel_region_becomes_none(self):
        builder = _builder({"type": "rest", "id": "r", "extraction": {"threshold_method": "cluster"}})
        builder.atlas_region_dropdown.value = "__loading__"
        ex = builder.collect_config()["extraction"]
        assert ex["atlas_regions"] is None

    def test_cluster_extent_method_emits_extent_keys(self):
        builder = _builder(
            {
                "type": "rest",
                "id": "r",
                "extraction": {
                    "threshold_method": "cluster_extent",
                    "cluster_start_threshold": 2.5,
                    "min_cluster_size": 80,
                    "max_total_voxels": 5000,
                },
            }
        )
        ex = builder.collect_config()["extraction"]
        assert ex["threshold_method"] == "cluster_extent"
        assert ex["cluster_start_threshold"] == 2.5
        assert ex["min_cluster_size"] == 80
        assert ex["max_total_voxels"] == 5000

    def test_cluster_extent_empty_max_total_is_none(self):
        builder = _builder({"type": "rest", "id": "r", "extraction": {"threshold_method": "cluster_extent"}})
        builder.max_total_voxels_field.value = ""
        ex = builder.collect_config()["extraction"]
        assert ex["max_total_voxels"] is None

    def test_cluster_extent_grey_matter_sets_atlas_source(self):
        builder = _builder({"type": "rest", "id": "r", "extraction": {"threshold_method": "cluster_extent"}})
        builder.extent_grey_matter_checkbox.value = True
        ex = builder.collect_config()["extraction"]
        assert ex["atlas_source"] == "gray_matter"


class TestThresholdRowVisibility:
    """ "Voxels per mask" (threshold) drives top_n/cluster sizing only; cluster_extent
    ignores it, so the row is hidden for that mode and shown for the other two."""

    def test_hidden_initially_for_cluster_extent(self):
        builder = _builder({"type": "rit", "id": "r", "extraction": {"threshold_method": "cluster_extent"}})
        builder.build()  # build() creates self.threshold_row + sets initial visibility
        assert builder.threshold_row.visible is False

    @pytest.mark.parametrize("method", ["top_n", "cluster"])
    def test_visible_initially_for_count_based_methods(self, method):
        builder = _builder({"type": "rit", "id": "r", "extraction": {"threshold_method": method}})
        builder.build()
        assert builder.threshold_row.visible is True

    def test_toggle_follows_dropdown_changes(self):
        builder = _builder({"type": "rit", "id": "r", "extraction": {"threshold_method": "top_n"}})
        builder.build()
        assert builder.threshold_row.visible is True

        builder.threshold_method_dropdown.value = "cluster_extent"
        builder._on_threshold_method_changed(None)
        assert builder.threshold_row.visible is False
        assert builder.cluster_extent_settings_section.visible is True

        builder.threshold_method_dropdown.value = "cluster"
        builder._on_threshold_method_changed(None)
        assert builder.threshold_row.visible is True
        assert builder.cluster_extent_settings_section.visible is False


class TestCollectConfigTask:
    def test_task_emits_glm_fields(self):
        out = _builder({"type": "rit", "id": "rit1", "tr": 1.2, "n_volumes": 400, "n_runs": 1}).collect_config()
        assert out["type"] == "rit"
        assert out["stat_type"] == "t"
        assert out["hrf_model"] == "spm"
        assert out["drift_model"] == "cosine"

    def test_task_contrasts_from_metadata_when_config_empty(self):
        # RIT declares default_contrasts={"rumination_network": "rumination - distraction"}.
        out = _builder({"type": "rit", "id": "rit1"}).collect_config()
        assert out["contrasts"] == {"rumination_network": "rumination - distraction"}

    def test_task_contrasts_from_config_take_precedence(self):
        out = _builder({"type": "rit", "id": "rit1", "contrasts": {"custom": "a - b"}}).collect_config()
        assert out["contrasts"] == {"custom": "a - b"}

    def test_task_collects_task_params(self):
        out = _builder({"type": "rit", "id": "rit1", "task_params": {"n_trials_per_run": 6}}).collect_config()
        assert out["task_params"]["n_trials_per_run"] == 6
        # Other params come from metadata defaults.
        assert out["task_params"]["rating_timeout"] == 5.0

    def test_task_extraction_has_no_ica_keys(self):
        ex = _builder({"type": "rit", "id": "rit1"}).collect_config()["extraction"]
        for key in ("network_atlas", "smooth_fwhm", "highpass", "lowpass", "n_components"):
            assert key not in ex
        # But threshold settings are still present.
        assert ex["threshold_method"] == "top_n"

    def test_task_has_no_networks_key(self):
        out = _builder({"type": "rit", "id": "rit1"}).collect_config()
        assert "networks" not in out

    def test_reverse_contrasts_reflects_config(self):
        out = _builder({"type": "selfother", "id": "so", "include_reverse_contrasts": True}).collect_config()
        assert out["include_reverse_contrasts"] is True

    def test_reverse_contrasts_toggle_changes_output(self):
        builder = _builder({"type": "selfother", "id": "so"})
        assert builder.collect_config()["include_reverse_contrasts"] is False
        builder.include_reverse_contrasts_checkbox.value = True
        assert builder.collect_config()["include_reverse_contrasts"] is True


class TestCollectConfigManual:
    def test_manual_emits_only_four_keys(self):
        out = _builder(
            {"type": "manual", "id": "manual", "manual_mask_space": "mni", "manual_mask_names": ["DMN", "FPN"]}
        ).collect_config()
        assert set(out) == {"type", "id", "manual_mask_space", "manual_mask_names"}
        assert out["manual_mask_space"] == "mni"
        assert out["manual_mask_names"] == ["DMN", "FPN"]

    def test_manual_id_defaults_to_manual(self):
        out = _builder({"type": "manual", "manual_mask_names": ["DMN"]}).collect_config()
        assert out["id"] == "manual"
        assert out["manual_mask_space"] == "native"
