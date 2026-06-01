"""Tests for pineuro.gui.screens._feedback_tab.

Covers (1) study-wide sham gating: when sham is disabled study-wide (the Study
tab's "Enable Sham/Control Conditions" is off), the Sham Mask Construction
section is hidden and ``collect_config`` must not persist a per-run ``sham``
block; when enabled — or when no provider is supplied (back-compat default) —
the section is shown and the block is emitted. And (2) ``collect_config()``
assembling a complete per-run config dict (task / network assignment / GLM /
motion / connectivity / intermittent blocks) plus the session-qualified
mask-key encode/decode helpers as they round-trip through the builder.
"""

from unittest.mock import MagicMock

import flet as ft

from pineuro.gui.screens._feedback_tab import (
    _SRC_SEP,
    FeedbackTabBuilder,
    _mask_key,
    _split_mask_key,
)


def _feedback_builder(*, sham_enabled=None):
    """Construct a real ball-task FeedbackTabBuilder with a mock page."""
    return FeedbackTabBuilder(
        index=0,
        config={"task": "ball"},
        page=MagicMock(),
        create_browse_button=lambda *a, **k: ft.IconButton(),
        create_file_browse_button=lambda *a, **k: ft.IconButton(),
        sham_enabled=sham_enabled,
    )


def _builder(task="ball", config=None, masks=None, *, sham_enabled=lambda: False):
    """Construct a FeedbackTabBuilder for ``task`` with the given config + masks.

    ``masks`` is a list of ``(source_session, name, label)`` triples; defaults to
    two fresh current-session masks so multi-mask paths are exercisable.
    """
    cfg = {"task": task}
    if config:
        cfg.update(config)
    if masks is None:
        masks = [("", "NetA", "NetA"), ("", "NetB", "NetB")]
    return FeedbackTabBuilder(
        index=0,
        config=cfg,
        page=MagicMock(),
        available_mask_options=masks,
        sham_enabled=sham_enabled,
    )


class TestShamGatingCollectConfig:
    def test_omits_sham_when_disabled(self):
        out = _feedback_builder(sham_enabled=lambda: False).collect_config()
        assert "sham" not in out

    def test_includes_sham_when_enabled(self):
        out = _feedback_builder(sham_enabled=lambda: True).collect_config()
        assert "sham" in out
        assert out["sham"]["delivery_method"] in ("alternative_masks", "replay")

    def test_includes_sham_by_default(self):
        # No provider => treated as enabled (back-compat).
        out = _feedback_builder(sham_enabled=None).collect_config()
        assert "sham" in out


class TestShamSectionVisibility:
    def test_section_hidden_when_disabled(self):
        builder = _feedback_builder(sham_enabled=lambda: False)
        section = builder._build_sham_section()
        assert section is builder.sham_section
        assert section.visible is False

    def test_section_visible_when_enabled(self):
        builder = _feedback_builder(sham_enabled=lambda: True)
        section = builder._build_sham_section()
        assert section.visible is True


class TestCollectConfigStructure:
    """collect_config() assembles the full nested per-run schema."""

    def test_top_level_keys_and_task(self):
        cfg = _builder().collect_config()
        assert cfg["task"] == "ball"
        # The nested sub-block schema is always emitted.
        for key in (
            "qc",
            "glm",
            "connectivity",
            "intermittent",
            "network_assignment",
            "mask_sources",
            "feedback_mode",
            "roi_combine_method",
            "task_params",
        ):
            assert key in cfg, f"missing top-level key: {key}"

    def test_scan_params_coerced_from_field_strings(self):
        cfg = _builder(config={"max_runs": 4, "tr": 2.0, "n_volumes": 200, "n_baseline": 30}).collect_config()
        # Fields hold strings; collect_config coerces to the typed schema.
        assert cfg["max_runs"] == 4 and isinstance(cfg["max_runs"], int)
        assert cfg["tr"] == 2.0 and isinstance(cfg["tr"], float)
        assert cfg["n_volumes"] == 200
        assert cfg["n_baseline"] == 30

    def test_qc_block_values(self):
        cfg = _builder(
            config={"qc": {"fd_threshold": 0.3, "dvars_threshold": 2.0, "tsnr_threshold": 15}}
        ).collect_config()
        assert cfg["qc"] == {"fd_threshold": 0.3, "dvars_threshold": 2.0, "tsnr_threshold": 15.0}

    def test_glm_block_reflects_checkboxes_and_motion(self):
        cfg = _builder(
            config={
                "glm": {
                    "motion_correction": "software",
                    "include_drift": False,
                    "include_wm_regressor": True,
                    "include_csf_regressor": True,
                }
            }
        ).collect_config()
        glm = cfg["glm"]
        assert glm["motion_correction"] == "software"
        assert glm["include_drift"] is False
        assert glm["include_wm_regressor"] is True
        assert glm["include_csf_regressor"] is True
        # Smoothing disabled (no smooth_fwhm in config) => None.
        assert glm["smooth_fwhm"] is None

    def test_smoothing_emitted_when_enabled(self):
        cfg = _builder(config={"glm": {"smooth_fwhm": 4.5}}).collect_config()
        assert cfg["glm"]["smooth_fwhm"] == 4.5

    def test_intermittent_disabled_emits_none_interval(self):
        cfg = _builder().collect_config()
        assert cfg["intermittent"] == {"interval": None}

    def test_intermittent_enabled_emits_interval(self):
        cfg = _builder(config={"intermittent": {"interval": 7}}).collect_config()
        assert cfg["intermittent"] == {"interval": 7}

    def test_roi_combine_method_passthrough(self):
        cfg = _builder(config={"roi_combine_method": "mean"}).collect_config()
        assert cfg["roi_combine_method"] == "mean"

    def test_mask_source_and_sim_paths_preserved(self):
        cfg = _builder(
            config={
                "mask_source": "rest",
                "source_dicoms": "/data/dicoms",
                "transfer_source_dicoms": "/data/transfer",
            }
        ).collect_config()
        assert cfg["mask_source"] == "rest"
        assert cfg["source_dicoms"] == "/data/dicoms"
        assert cfg["transfer_source_dicoms"] == "/data/transfer"

    def test_use_transfer_runs_lifted_to_top_level(self):
        # Ball's use_transfer_runs param default is True.
        cfg = _builder().collect_config()
        assert cfg["use_transfer_runs"] is True
        assert cfg["task_params"]["use_transfer_runs"] is True


class TestCollectConfigNetworkAssignment:
    """network_assignment is the single source of truth for role->mask mapping."""

    def test_dual_network_differential_pair(self):
        cfg = _builder(
            config={"network_assignment": {"top_network": "NetA", "bottom_network": "NetB"}}
        ).collect_config()
        assert cfg["network_assignment"] == {"top_network": "NetA", "bottom_network": "NetB"}
        assert cfg["feedback_mode"] == "activation"

    def test_single_network_target_mask(self):
        # Thermometer is a single-network task -> single_mask_dropdown drives the
        # single role ("target_mask"). feedback_mode dropdown stays hidden.
        b = _builder(
            task="thermometer",
            config={"network_assignment": {"target_mask": "NetB"}},
        )
        assert b.feedback_mode_dropdown.visible is False
        cfg = b.collect_config()
        assert cfg["network_assignment"] == {"target_mask": "NetB"}

    def test_cross_session_source_routed_to_mask_sources(self):
        # NetB is sourced from an earlier session: the bare name goes to
        # network_assignment, the source session to mask_sources.
        masks = [("", "NetA", "NetA"), ("ses-01", "NetB", "ses-01 NetB")]
        cfg = _builder(
            config={
                "network_assignment": {"top_network": "NetA", "bottom_network": "NetB"},
                "mask_sources": {"NetB": "ses-01"},
            },
            masks=masks,
        ).collect_config()
        assert cfg["network_assignment"] == {"top_network": "NetA", "bottom_network": "NetB"}
        assert cfg["mask_sources"] == {"NetB": "ses-01"}

    def test_fresh_masks_have_empty_mask_sources(self):
        cfg = _builder(
            config={"network_assignment": {"top_network": "NetA", "bottom_network": "NetB"}}
        ).collect_config()
        assert cfg["mask_sources"] == {}

    def test_optional_role_persisted_only_when_visible_single_mode(self):
        # rittriggeredball: single movement_mode shows movement_network and hides
        # top/bottom. Only the visible, assigned optional role survives.
        masks = [("", "Trig", "Trig"), ("", "Top", "Top"), ("", "Bot", "Bot")]
        cfg = _builder(
            task="rittriggeredball",
            config={
                "task_params": {"movement_mode": "single"},
                "network_assignment": {
                    "trigger_network": "Trig",
                    "movement_network": "Top",
                    "top_network": "Bot",  # stale dual-mode value must be dropped
                },
            },
            masks=masks,
        ).collect_config()
        assert cfg["network_assignment"] == {"trigger_network": "Trig", "movement_network": "Top"}

    def test_optional_role_persisted_only_when_visible_dual_mode(self):
        masks = [("", "Trig", "Trig"), ("", "Top", "Top"), ("", "Bot", "Bot")]
        cfg = _builder(
            task="rittriggeredball",
            config={
                "task_params": {"movement_mode": "dual"},
                "network_assignment": {
                    "trigger_network": "Trig",
                    "movement_network": "Top",  # stale single-mode value must be dropped
                    "top_network": "Top",
                    "bottom_network": "Bot",
                },
            },
            masks=masks,
        ).collect_config()
        assert cfg["network_assignment"] == {
            "trigger_network": "Trig",
            "top_network": "Top",
            "bottom_network": "Bot",
        }
        assert "movement_network" not in cfg["network_assignment"]


class TestCollectConfigConnectivity:
    """The connectivity sub-block follows feedback_mode."""

    def test_activation_mode_emits_null_pairs(self):
        cfg = _builder(config={"feedback_mode": "activation"}).collect_config()
        assert cfg["feedback_mode"] == "activation"
        assert cfg["connectivity"]["pairs"] is None
        assert cfg["connectivity"]["window"] == 15

    def test_connectivity_mode_emits_pair_and_window(self):
        cfg = _builder(
            config={
                "feedback_mode": "connectivity",
                "connectivity": {"pairs": [["NetA", "NetB"]], "window": 22, "target": "decrease"},
            }
        ).collect_config()
        assert cfg["feedback_mode"] == "connectivity"
        conn = cfg["connectivity"]
        assert conn["pairs"] == [["NetA", "NetB"]]
        assert conn["window"] == 22
        assert conn["target"] == "decrease"

    def test_connectivity_identical_masks_drops_pair(self):
        # A and B must differ; identical selections produce a null pair.
        b = _builder(config={"feedback_mode": "connectivity"})
        b.connectivity_pair_b_dropdown.value = b.connectivity_pair_a_dropdown.value
        cfg = b.collect_config()
        assert cfg["connectivity"]["pairs"] is None

    def test_connectivity_cross_session_source_recorded(self):
        masks = [("", "NetA", "NetA"), ("ses-02", "NetB", "ses-02 NetB")]
        cfg = _builder(
            config={
                "feedback_mode": "connectivity",
                "connectivity": {"pairs": [["NetA", "NetB"]], "window": 10},
            },
            masks=masks,
        ).collect_config()
        assert cfg["connectivity"]["pairs"] == [["NetA", "NetB"]]
        assert cfg["mask_sources"].get("NetB") == "ses-02"


class TestCollectConfigShamBlock:
    """The per-run sham construction block is gated on study-wide sham state."""

    def test_sham_block_contents_when_enabled(self):
        cfg = _builder(sham_enabled=lambda: True).collect_config()
        sham = cfg["sham"]
        assert sham["delivery_method"] == "alternative_masks"
        assert sham["sham_mask_mode"] == "hemisphere_split"
        assert sham["hemisphere_network"] == "SomMotA"
        assert sham["target_hemisphere"] == "right"
        assert sham["alternative_mask_targets"] == ["SomMotA", "SomMotB"]

    def test_sham_targets_reflect_config(self):
        cfg = _builder(
            config={"sham": {"alternative_mask_targets": ["VisCent", "VisPeri"]}},
            sham_enabled=lambda: True,
        ).collect_config()
        assert cfg["sham"]["alternative_mask_targets"] == ["VisCent", "VisPeri"]


class TestCollectConfigIsolation:
    """Editing the builder must not mutate the caller's config dict."""

    def test_caller_config_not_mutated(self):
        orig = {"task": "ball", "network_assignment": {"top_network": "NetA"}}
        b = _builder(config=orig)
        # Mutate the builder's nested assignment via collect + reassign.
        b.differential_negative_dropdown.value = _mask_key("", "NetB")
        b.collect_config()
        # The original nested dict is untouched (deep-copied at construction).
        assert orig["network_assignment"] == {"top_network": "NetA"}


class TestMaskKeyHelpers:
    """Session-qualified mask key encode/decode and round-trip through collect."""

    def test_mask_key_joins_with_separator(self):
        assert _mask_key("ses-03", "Trigger") == f"ses-03{_SRC_SEP}Trigger"

    def test_round_trips_qualified(self):
        assert _split_mask_key(_mask_key("ses-03", "Trigger")) == ("ses-03", "Trigger")

    def test_round_trips_current_session(self):
        assert _split_mask_key(_mask_key("", "NetA")) == ("", "NetA")

    def test_bare_name_parses_as_current_session(self):
        assert _split_mask_key("NetA") == ("", "NetA")

    def test_name_may_contain_separator_only_first_split(self):
        # split(..., 1) means only the first separator delimits source from name.
        key = _mask_key("ses-01", f"odd{_SRC_SEP}name")
        assert _split_mask_key(key) == ("ses-01", f"odd{_SRC_SEP}name")

    def test_qualified_key_decodes_through_collect_config(self):
        # A qualified dropdown selection round-trips: name -> network_assignment,
        # source -> mask_sources, via _split_mask_key inside collect_config.
        masks = [("", "NetA", "NetA"), ("ses-04", "NetB", "ses-04 NetB")]
        b = _builder(masks=masks)
        b.differential_positive_dropdown.value = _mask_key("", "NetA")
        b.differential_negative_dropdown.value = _mask_key("ses-04", "NetB")
        cfg = b.collect_config()
        assert cfg["network_assignment"] == {"top_network": "NetA", "bottom_network": "NetB"}
        assert cfg["mask_sources"] == {"NetB": "ses-04"}
