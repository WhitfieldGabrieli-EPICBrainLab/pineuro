"""Tests for pineuro.session — configuration dataclasses and visit management."""

import pytest

from pineuro.session import (
    FeedbackConfig,
    MaskConfig,
    PathConfig,
    ShamConfig,
    TaskConfig,
    VisitConfig,
    detect_existing_visits,
    next_visit_id,
)

# =============================================================================
# PathConfig
# =============================================================================


class TestPathConfig:
    def test_required_fields(self, tmp_path):
        config = PathConfig(base_dir=tmp_path, participant_id="sub-01")
        assert config.base_dir == tmp_path
        assert config.participant_id == "sub-01"
        assert config.visit_id is None  # single-visit mode

    def test_with_visit(self, tmp_path):
        config = PathConfig(base_dir=tmp_path, participant_id="sub-01", visit_id="ses-01")
        assert config.visit_id == "ses-01"


# =============================================================================
# MaskConfig
# =============================================================================


class TestMaskConfig:
    def test_defaults(self):
        config = MaskConfig()
        assert config.n_yeo == 17
        assert config.network_atlas == "yeo_17"

    def test_custom_targets(self):
        config = MaskConfig(yeo_targets=["DefaultA", "ContA"])
        assert config.yeo_targets == ["DefaultA", "ContA"]


# =============================================================================
# FeedbackConfig
# =============================================================================


class TestFeedbackConfig:
    def test_defaults(self):
        config = FeedbackConfig()
        assert config.n_baseline == 25
        assert config.n_volumes == 150
        assert config.motion_correction == "siemens_moco"
        assert config.include_drift is True
        assert config.include_motion_regressors is True
        assert config.include_wm_regressor is False
        assert config.include_csf_regressor is False

    def test_custom_values(self):
        config = FeedbackConfig(
            n_baseline=10,
            n_volumes=100,
            motion_correction="siemens_moco",
            include_wm_regressor=True,
        )
        assert config.n_baseline == 10
        assert config.motion_correction == "siemens_moco"
        assert config.include_wm_regressor is True


# =============================================================================
# TaskConfig
# =============================================================================


class TestTaskConfig:
    def test_defaults(self):
        config = TaskConfig()
        assert config.task_type == "ball"
        assert config.max_feedback_runs == 2
        assert config.task_params == {}
        assert config.network_assignment == {}

    def test_custom_params(self):
        config = TaskConfig(
            task_type="thermometer",
            task_params={"orientation": "vertical"},
            network_assignment={"target": "DefaultA"},
        )
        assert config.task_type == "thermometer"
        assert config.task_params["orientation"] == "vertical"


# =============================================================================
# ShamConfig
# =============================================================================


class TestShamConfig:
    def test_defaults(self):
        config = ShamConfig()
        assert config.enabled is False
        assert config.delivery_method == "alternative_masks"

    def test_enabled_replay_requires_randomization_file(self, tmp_path):
        """ShamConfig with replay method requires a randomization_file."""
        with pytest.raises(ValueError, match="randomization_file is required"):
            ShamConfig(enabled=True, delivery_method="replay")

    def test_enabled_alternative_masks(self):
        config = ShamConfig(enabled=True, delivery_method="alternative_masks")
        assert config.enabled is True
        assert config.delivery_method == "alternative_masks"


# =============================================================================
# VisitConfig
# =============================================================================


class TestVisitConfig:
    def test_construction(self):
        config = VisitConfig(visit_id="ses-01", visit_number=1)
        assert config.visit_id == "ses-01"
        assert config.visit_number == 1
        assert config.mask_source == "compute"

    def test_reuse_masks(self):
        config = VisitConfig(
            visit_id="ses-02",
            visit_number=2,
            mask_source="reuse",
            mask_source_visit="ses-01",
        )
        assert config.mask_source == "reuse"
        assert config.mask_source_visit == "ses-01"


# =============================================================================
# detect_existing_visits / next_visit_id
# =============================================================================


class TestVisitDetection:
    def test_detect_existing_visits(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        (participant_dir / "ses-01").mkdir(parents=True)
        (participant_dir / "ses-02").mkdir()
        (participant_dir / "ses-03").mkdir()

        visits = detect_existing_visits(tmp_path, "sub-01")
        # Returns list of dicts with visit_id key
        visit_ids = sorted(v["visit_id"] for v in visits)
        assert visit_ids == ["ses-01", "ses-02", "ses-03"]

    def test_detect_empty(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        participant_dir.mkdir()

        visits = detect_existing_visits(tmp_path, "sub-01")
        assert visits == []

    def test_detect_no_participant_dir(self, tmp_path):
        visits = detect_existing_visits(tmp_path, "sub-nonexistent")
        assert visits == []

    def test_next_visit_id_first(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        participant_dir.mkdir()

        result = next_visit_id(tmp_path, "sub-01")
        assert result == "ses-01"

    def test_next_visit_id_increment(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        (participant_dir / "ses-01").mkdir(parents=True)
        (participant_dir / "ses-02").mkdir()

        result = next_visit_id(tmp_path, "sub-01")
        assert result == "ses-03"
