"""Tests for pineuro.session — configuration dataclasses and session management."""

import pytest

from pineuro.session import (
    ConnectivityConfig,
    FeedbackRunConfig,
    GLMConfig,
    IntermittentConfig,
    LocalizerConfig,
    LocalizerExtractionConfig,
    PathConfig,
    QCConfig,
    ReferenceConfig,
    ScannerConfig,
    SessionConfig,
    SessionMetadata,
    ShamConfig,
    SimulatorConfig,
    detect_existing_sessions,
    next_session_id,
)

# =============================================================================
# PathConfig
# =============================================================================


class TestPathConfig:
    def test_required_fields(self, tmp_path):
        config = PathConfig(base_dir=tmp_path, participant_id="sub-01")
        assert config.base_dir == tmp_path
        assert config.participant_id == "sub-01"
        assert config.session_id == "ses-01"  # default

    def test_with_session(self, tmp_path):
        config = PathConfig(base_dir=tmp_path, participant_id="sub-01", session_id="ses-02")
        assert config.session_id == "ses-02"
        assert config.preprocessing_dir == tmp_path / "sub-01" / "ses-02" / "preprocessing"

    def test_example_participant_id(self, tmp_path):
        config = PathConfig(base_dir=tmp_path, participant_id="sub-01", example_participant_id="sub-XXX")
        assert config.example_participant_id == "sub-XXX"

    def test_invalid_session_id_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="session_id"):
            PathConfig(base_dir=tmp_path, participant_id="sub-01", session_id="bogus")


# =============================================================================
# LocalizerConfig + LocalizerExtractionConfig
# =============================================================================


class TestLocalizerConfig:
    def test_defaults_rest(self):
        config = LocalizerConfig()
        assert config.type == "rest"
        assert config.id == "rest"  # defaults to type
        assert config.networks == ["DefaultA", "ContA"]
        assert config.extraction.n_yeo == 17
        assert config.extraction.network_atlas == "yeo_17"
        assert config.motion_correction == "none"

    def test_id_override(self):
        config = LocalizerConfig(id="my_loc")
        assert config.id == "my_loc"

    def test_mask_names_rest(self):
        config = LocalizerConfig(networks=["ContA", "DefaultA"])
        assert config.mask_names == ["ContA", "DefaultA"]

    def test_mask_names_task(self):
        config = LocalizerConfig(type="rnt", contrasts={"rumination": "rum - dist"})
        assert config.mask_names == ["rumination"]

    def test_mask_names_task_reverse(self):
        config = LocalizerConfig(type="rnt", contrasts={"rumination": "rum - dist"}, include_reverse_contrasts=True)
        assert config.mask_names == ["rumination", "rumination_reverse"]

    def test_manual_requires_names_when_no_paths(self):
        # GUI study config carries only network names — neither names nor paths is invalid.
        with pytest.raises(ValueError, match="manual_mask_names"):
            LocalizerConfig(type="manual")

    def test_manual_names_only_valid(self):
        # The GUI study config (names only; files chosen per participant) is valid.
        config = LocalizerConfig(type="manual", manual_mask_names=["DMN", "FPN"])
        assert config.is_manual
        assert config.mask_names == ["DMN", "FPN"]

    def test_manual_anat_space_names_only_ok(self):
        # Anatomical space with names only does NOT require a T1 at study-config
        # time — the T1 is a per-participant file chosen later.
        config = LocalizerConfig(type="manual", manual_mask_space="anat", manual_mask_names=["DMN"])
        assert config.manual_mask_space == "anat"

    def test_manual_mask_names(self, tmp_path):
        m1 = tmp_path / "roi_a.nii.gz"
        m2 = tmp_path / "roi_b.nii"
        for p in (m1, m2):
            p.touch()
        config = LocalizerConfig(type="manual", manual_mask_paths=[m1, m2])
        assert config.mask_names == ["roi_a", "roi_b"]

    def test_manual_mask_names_explicit_preferred(self, tmp_path):
        """Explicit manual_mask_names override the filename-derived defaults."""
        m1 = tmp_path / "roi_a.nii.gz"
        m2 = tmp_path / "roi_b.nii.gz"
        for p in (m1, m2):
            p.touch()
        config = LocalizerConfig(
            type="manual",
            manual_mask_paths=[m1, m2],
            manual_mask_names=["DMN", "FPN"],
        )
        assert config.mask_names == ["DMN", "FPN"]

    def test_manual_mask_names_length_mismatch_raises(self, tmp_path):
        m1 = tmp_path / "roi_a.nii.gz"
        m1.touch()
        with pytest.raises(ValueError, match="manual_mask_names"):
            LocalizerConfig(type="manual", manual_mask_paths=[m1], manual_mask_names=["DMN", "FPN"])

    def test_manual_anat_space_requires_anat_file(self, tmp_path):
        m1 = tmp_path / "roi_a.nii.gz"
        m1.touch()
        with pytest.raises(ValueError, match="manual_anat_file"):
            LocalizerConfig(type="manual", manual_mask_paths=[m1], manual_mask_space="anat")


class TestLocalizerExtractionConfig:
    def test_defaults(self):
        ex = LocalizerExtractionConfig()
        assert ex.n_yeo == 17
        assert ex.threshold == 2000
        assert ex.threshold_method == "top_n"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="threshold_method"):
            LocalizerExtractionConfig(threshold_method="bogus")


# =============================================================================
# FeedbackRunConfig + nested sub-blocks
# =============================================================================


class TestFeedbackRunConfig:
    def test_defaults(self):
        frc = FeedbackRunConfig()
        assert frc.task == "ball"
        assert frc.mask_source == ""
        assert frc.max_runs == 1
        assert frc.roi_combine_method == "inverse_sd"
        assert frc.qc.fd_threshold == 0.5
        assert frc.glm.motion_correction == "siemens_moco"
        assert frc.glm.include_drift is True
        assert frc.connectivity.window == 15
        assert frc.intermittent.interval is None

    def test_nested_sub_blocks(self):
        frc = FeedbackRunConfig(
            qc=QCConfig(fd_threshold=0.3),
            glm=GLMConfig(motion_correction="software", include_drift=False),
            connectivity=ConnectivityConfig(pairs=[["a", "b"]], target="decrease"),
            intermittent=IntermittentConfig(interval=5),
        )
        assert frc.qc.fd_threshold == 0.3
        assert frc.glm.motion_correction == "software"
        assert frc.connectivity.pairs == [["a", "b"]]
        assert frc.intermittent.interval == 5

    def test_invalid_roi_combine_method_rejected(self):
        with pytest.raises(ValueError, match="roi_combine_method"):
            FeedbackRunConfig(roi_combine_method="bogus")

    def test_legacy_use_efficiency_weights_migrates(self):
        # Old modern-schema YAMLs may still carry the boolean; _build_feedback_run
        # translates it to the roi_combine_method choice.
        true_run = SessionConfig._build_feedback_run({"use_efficiency_weights": True})
        false_run = SessionConfig._build_feedback_run({"use_efficiency_weights": False})
        assert true_run.roi_combine_method == "inverse_sd"
        assert false_run.roi_combine_method == "mean"

    def test_invalid_motion_correction_rejected(self):
        with pytest.raises(ValueError, match="motion_correction"):
            GLMConfig(motion_correction="bogus")


# =============================================================================
# Top-level block dataclasses
# =============================================================================


class TestNewTopLevelBlocks:
    def test_simulator(self):
        cfg = SimulatorConfig(enabled=False, feedback_speed=5.0)
        assert cfg.enabled is False
        assert cfg.feedback_speed == 5.0
        assert cfg.collection_speed == 1.0  # default unchanged

    def test_scanner(self):
        cfg = ScannerConfig(ip="192.168.2.1")
        assert cfg.ip == "192.168.2.1"
        assert cfg.dicom_dir == "/BOLD_DICOM"
        assert cfg.trigger_keys == ["t", "+", "5", "equal"]

    def test_scanner_trigger_keys_override(self):
        cfg = ScannerConfig(trigger_keys=["5"])
        assert cfg.trigger_keys == ["5"]

    def test_scanner_trigger_keys_are_independent_per_instance(self):
        a = ScannerConfig()
        b = ScannerConfig()
        a.trigger_keys.append("x")
        assert b.trigger_keys == ["t", "+", "5", "equal"]

    def test_reference(self):
        cfg = ReferenceConfig(initial_enabled=False)
        assert cfg.initial_enabled is False
        assert cfg.realtime_enabled is True  # default


# =============================================================================
# SessionConfig
# =============================================================================


class TestSessionConfig:
    def test_combined_mask_names(self, tmp_path):
        cfg = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01"),
            localizers=[
                LocalizerConfig(id="rest", type="rest", networks=["DefaultA", "ContA"]),
            ],
        )
        assert cfg.combined_mask_names == ["DefaultA", "ContA"]

    def test_combined_mask_names_deduplicates(self, tmp_path):
        cfg = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01"),
            localizers=[
                LocalizerConfig(id="r1", type="rest", networks=["DefaultA", "ContA"]),
                LocalizerConfig(id="r2", type="rest", networks=["ContA", "SalVentAttnA"]),
            ],
        )
        # First-occurrence wins, no duplicates
        assert cfg.combined_mask_names == ["DefaultA", "ContA", "SalVentAttnA"]

    def test_round_trip(self, tmp_path):
        cfg = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01"),
            localizers=[LocalizerConfig(id="rest", networks=["DefaultA", "ContA"])],
            feedback_runs=[
                FeedbackRunConfig(
                    task="ball",
                    mask_source="rest",
                    network_assignment={"top_network": "ContA", "bottom_network": "DefaultA"},
                )
            ],
        )
        d = cfg._to_dict()
        assert "mask" not in d and "feedback" not in d and "task" not in d and "scan" not in d
        assert d["localizers"][0]["id"] == "rest"
        assert d["feedback_runs"][0]["task"] == "ball"
        assert d["feedback_runs"][0]["network_assignment"]["top_network"] == "ContA"

        cfg2 = SessionConfig._from_dict(d)
        assert cfg2.feedback_runs[0].task == "ball"
        assert cfg2.feedback_runs[0].network_assignment == {
            "top_network": "ContA",
            "bottom_network": "DefaultA",
        }
        assert cfg2.combined_mask_names == ["DefaultA", "ContA"]

    def test_rejects_legacy_top_level_keys(self):
        for legacy_key in ("mask", "feedback", "task", "localizer_task", "scan"):
            with pytest.raises(ValueError, match="outdated schema"):
                SessionConfig._from_dict({"paths": {"base_dir": "/tmp", "participant_id": "sub-01"}, legacy_key: {}})

    def test_rejects_unknown_mask_source(self, tmp_path):
        data = {
            "paths": {"base_dir": str(tmp_path), "participant_id": "sub-01"},
            "localizers": [{"id": "rest", "type": "rest"}],
            "feedback_runs": [{"task": "ball", "mask_source": "no_such_localizer"}],
        }
        with pytest.raises(ValueError, match="mask_source"):
            SessionConfig._from_dict(data)


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
# SessionMetadata
# =============================================================================


class TestSessionMetadata:
    def test_construction(self):
        config = SessionMetadata(session_id="ses-01", session_number=1)
        assert config.session_id == "ses-01"
        assert config.session_number == 1
        assert config.mask_source == "compute"

    def test_reuse_masks(self):
        config = SessionMetadata(
            session_id="ses-02",
            session_number=2,
            mask_source="reuse",
            mask_source_session="ses-01",
        )
        assert config.mask_source == "reuse"
        assert config.mask_source_session == "ses-01"


# =============================================================================
# detect_existing_sessions / next_session_id
# =============================================================================


class TestSessionDetection:
    def test_detect_existing_sessions(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        (participant_dir / "ses-01").mkdir(parents=True)
        (participant_dir / "ses-02").mkdir()
        (participant_dir / "ses-03").mkdir()

        sessions = detect_existing_sessions(tmp_path, "sub-01")
        # Returns list of dicts with session_id key
        session_ids = sorted(v["session_id"] for v in sessions)
        assert session_ids == ["ses-01", "ses-02", "ses-03"]

    def test_detect_empty(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        participant_dir.mkdir()

        sessions = detect_existing_sessions(tmp_path, "sub-01")
        assert sessions == []

    def test_detect_no_participant_dir(self, tmp_path):
        sessions = detect_existing_sessions(tmp_path, "sub-nonexistent")
        assert sessions == []

    def test_next_session_id_first(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        participant_dir.mkdir()

        result = next_session_id(tmp_path, "sub-01")
        assert result == "ses-01"

    def test_next_session_id_increment(self, tmp_path):
        participant_dir = tmp_path / "sub-01"
        (participant_dir / "ses-01").mkdir(parents=True)
        (participant_dir / "ses-02").mkdir()

        result = next_session_id(tmp_path, "sub-01")
        assert result == "ses-03"
