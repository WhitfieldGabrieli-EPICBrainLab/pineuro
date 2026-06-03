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
    ShamRunConfig,
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
        assert config.extraction.network_atlas == "yeo_17"
        assert config.motion_correction == "none"

    def test_id_override(self):
        config = LocalizerConfig(id="my_loc")
        assert config.id == "my_loc"

    def test_mask_names_rest(self):
        config = LocalizerConfig(networks=["ContA", "DefaultA"])
        assert config.mask_names == ["ContA", "DefaultA"]

    def test_mask_names_task(self):
        config = LocalizerConfig(type="rit", contrasts={"rumination": "rum - dist"})
        assert config.mask_names == ["rumination"]

    def test_mask_names_task_reverse(self):
        config = LocalizerConfig(type="rit", contrasts={"rumination": "rum - dist"}, include_reverse_contrasts=True)
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
        assert ex.network_atlas == "yeo_17"
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
        assert frc.roi_combine_method == "voxel_efficiency"
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

    def test_simulator_speed_defaults_are_realtime(self):
        # Both simulator speeds default to 1.0× (real-time); no 10× feedback default.
        cfg = SimulatorConfig()
        assert cfg.collection_speed == 1.0
        assert cfg.feedback_speed == 1.0

    def test_scanner(self):
        cfg = ScannerConfig(ip="192.168.2.1")
        assert cfg.ip == "192.168.2.1"
        assert cfg.dicom_dir == "/BOLD_DICOM"
        assert cfg.trigger_keys == ["t", "+", "5", "="]

    def test_scanner_trigger_keys_override(self):
        cfg = ScannerConfig(trigger_keys=["5"])
        assert cfg.trigger_keys == ["5"]

    def test_scanner_trigger_keys_are_independent_per_instance(self):
        a = ScannerConfig()
        b = ScannerConfig()
        a.trigger_keys.append("x")
        assert b.trigger_keys == ["t", "+", "5", "="]

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


class TestResolveMaskNamesForRun:
    """_resolve_mask_names_for_run with optional roles and variable arity."""

    def _config(self, tmp_path, networks):
        return SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01"),
            localizers=[LocalizerConfig(networks=networks)],
        )

    def test_single_trigger_only(self, tmp_path):
        from pineuro.session import _resolve_mask_names_for_run
        from pineuro.tasks.task_registry import get_task_metadata

        cfg = self._config(tmp_path, ["RIT"])
        meta = get_task_metadata("rittriggeredball")
        names = _resolve_mask_names_for_run(cfg, meta, {"trigger_network": "RIT"})
        assert names == ["RIT"]

    def test_single_with_movement_override(self, tmp_path):
        from pineuro.session import _resolve_mask_names_for_run
        from pineuro.tasks.task_registry import get_task_metadata

        cfg = self._config(tmp_path, ["RIT", "ContA"])
        meta = get_task_metadata("rittriggeredball")
        names = _resolve_mask_names_for_run(cfg, meta, {"trigger_network": "RIT", "movement_network": "ContA"})
        assert names == ["RIT", "ContA"]

    def test_dual(self, tmp_path):
        from pineuro.session import _resolve_mask_names_for_run
        from pineuro.tasks.task_registry import get_task_metadata

        cfg = self._config(tmp_path, ["RIT", "ContA", "DefaultA"])
        meta = get_task_metadata("rittriggeredball")
        names = _resolve_mask_names_for_run(
            cfg, meta, {"trigger_network": "RIT", "top_network": "ContA", "bottom_network": "DefaultA"}
        )
        assert names == ["RIT", "ContA", "DefaultA"]

    def test_optional_role_not_backfilled(self, tmp_path):
        # Only the trigger is assigned; optional roles must NOT be positionally
        # backfilled from the combined mask list.
        from pineuro.session import _resolve_mask_names_for_run
        from pineuro.tasks.task_registry import get_task_metadata

        cfg = self._config(tmp_path, ["RIT", "ContA", "DefaultA"])
        meta = get_task_metadata("rittriggeredball")
        names = _resolve_mask_names_for_run(cfg, meta, {"trigger_network": "RIT"})
        assert names == ["RIT"]

    def test_two_network_task_unchanged(self, tmp_path):
        # Regression: a 2-network task (all required roles) still resolves to 2.
        from pineuro.session import _resolve_mask_names_for_run
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("ball")
        keys = [r.key for r in meta.network_roles]
        cfg = self._config(tmp_path, ["ContA", "DefaultA"])
        names = _resolve_mask_names_for_run(cfg, meta, {keys[0]: "ContA", keys[1]: "DefaultA"})
        assert names == ["ContA", "DefaultA"]


class TestShamConfig:
    """Study-wide sham gating (enabled / blind_mode / randomization_file)."""

    def test_defaults(self):
        config = ShamConfig()
        assert config.enabled is False
        assert config.blind_mode is False
        assert config.randomization_file is None

    def test_enabled_global(self, tmp_path):
        rand = tmp_path / "rand.tsv"
        rand.write_text("subject_id\tcondition\n")
        config = ShamConfig(enabled=True, blind_mode=True, randomization_file=rand)
        assert config.enabled is True
        assert config.blind_mode is True
        assert config.randomization_file == rand


class TestShamRunConfig:
    """Per-feedback-run sham mask construction."""

    def test_defaults(self):
        config = ShamRunConfig()
        assert config.delivery_method == "alternative_masks"
        assert config.sham_mask_mode == "hemisphere_split"
        assert config.hemisphere_network == "SomMotA"
        assert config.target_hemisphere == "right"
        assert config.alternative_mask_targets == ["SomMotA", "SomMotB"]

    def test_invalid_delivery_method(self):
        with pytest.raises(ValueError, match="delivery_method"):
            ShamRunConfig(delivery_method="bogus")

    def test_invalid_mask_mode(self):
        with pytest.raises(ValueError, match="sham_mask_mode"):
            ShamRunConfig(sham_mask_mode="bogus")

    def test_invalid_target_hemisphere(self):
        with pytest.raises(ValueError, match="target_hemisphere"):
            ShamRunConfig(target_hemisphere="up")


class TestShamReplayValidation:
    """Cross-block validation: replay delivery needs the study randomization file."""

    def _data(self, tmp_path, sham_global, run_sham):
        return {
            "paths": {"base_dir": str(tmp_path), "participant_id": "sub-01", "session_id": "ses-01"},
            "sham": sham_global,
            "localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA", "ContA"]}],
            "feedback_runs": [{"task": "ball", "mask_source": "rest", "sham": run_sham}],
        }

    def test_replay_requires_randomization_file(self, tmp_path):
        with pytest.raises(ValueError, match="randomization_file"):
            SessionConfig._from_dict(self._data(tmp_path, {"enabled": True}, {"delivery_method": "replay"}))

    def test_replay_with_randomization_file_ok(self, tmp_path):
        rand = tmp_path / "rand.tsv"
        rand.write_text("subject_id\tcondition\n")
        cfg = SessionConfig._from_dict(
            self._data(tmp_path, {"enabled": True, "randomization_file": str(rand)}, {"delivery_method": "replay"})
        )
        assert cfg.feedback_runs[0].sham.delivery_method == "replay"

    def test_per_run_sham_round_trips(self, tmp_path):
        cfg = SessionConfig._from_dict(self._data(tmp_path, {"enabled": True}, {"sham_mask_mode": "two_networks"}))
        assert isinstance(cfg.feedback_runs[0].sham, ShamRunConfig)
        assert cfg.feedback_runs[0].sham.sham_mask_mode == "two_networks"
        # asdict-based round-trip keeps the per-run sham block.
        cfg2 = SessionConfig._from_dict(cfg._to_dict())
        assert cfg2.feedback_runs[0].sham.sham_mask_mode == "two_networks"


# =============================================================================
# SessionMetadata
# =============================================================================


class TestSessionMetadata:
    def test_construction(self):
        config = SessionMetadata(session_id="ses-01", session_number=1, notes="baseline")
        assert config.session_id == "ses-01"
        assert config.session_number == 1
        assert config.notes == "baseline"


class TestLocalizerReuse:
    """Per-localizer mask reuse (reuse_from_session)."""

    def test_fresh_localizer_default(self):
        loc = LocalizerConfig(id="rest", networks=["DefaultA", "ContA"])
        assert loc.reuse_from_session is None
        assert loc.is_reused is False

    def test_reused_localizer(self):
        loc = LocalizerConfig(id="rest", networks=["DefaultA", "ContA"], reuse_from_session="ses-01")
        assert loc.reuse_from_session == "ses-01"
        assert loc.is_reused is True

    def test_invalid_reuse_session_rejected(self):
        with pytest.raises(ValueError, match="reuse_from_session"):
            LocalizerConfig(id="rest", networks=["DefaultA"], reuse_from_session="session-1")

    def test_reuse_round_trips(self, tmp_path):
        cfg = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01", session_id="ses-02"),
            localizers=[LocalizerConfig(id="rest", networks=["DefaultA", "ContA"], reuse_from_session="ses-01")],
            feedback_runs=[FeedbackRunConfig(task="ball", mask_source="rest")],
        )
        cfg2 = SessionConfig._from_dict(cfg._to_dict())
        assert cfg2.localizers[0].reuse_from_session == "ses-01"
        assert cfg2.localizers[0].is_reused is True

    def test_mask_sources_round_trips(self, tmp_path):
        cfg = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01", session_id="ses-02"),
            localizers=[LocalizerConfig(id="rest", networks=["DefaultA"], reuse_from_session="ses-01")],
            feedback_runs=[FeedbackRunConfig(task="ball", mask_source="rest", mask_sources={"DefaultA": "ses-01"})],
        )
        cfg2 = SessionConfig._from_dict(cfg._to_dict())
        assert cfg2.feedback_runs[0].mask_sources == {"DefaultA": "ses-01"}

    def test_tissue_required_round_trips(self, tmp_path):
        cfg = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01", session_id="ses-01"),
            localizers=[LocalizerConfig(id="rest", networks=["DefaultA"])],
            feedback_runs=[FeedbackRunConfig(task="ball", mask_source="rest")],
            tissue_required=True,
        )
        cfg2 = SessionConfig._from_dict(cfg._to_dict())
        assert cfg2.tissue_required is True
        # Default (None) is not serialized.
        cfg3 = SessionConfig(
            paths=PathConfig(base_dir=tmp_path, participant_id="sub-01", session_id="ses-01"),
            localizers=[LocalizerConfig(id="rest", networks=["DefaultA"])],
            feedback_runs=[FeedbackRunConfig(task="ball", mask_source="rest")],
        )
        assert "_tissue_required" not in cfg3._to_dict()


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

    def _status_for(self, tmp_path, build) -> str:
        """Build a single ``ses-01`` tree via ``build(ses_dir)`` and return its status."""
        ses_dir = tmp_path / "sub-01" / "ses-01"
        ses_dir.mkdir(parents=True)
        build(ses_dir)
        sessions = detect_existing_sessions(tmp_path, "sub-01")
        assert len(sessions) == 1
        return sessions[0]["status"]

    def test_status_not_started(self, tmp_path):
        assert self._status_for(tmp_path, lambda d: None) == "Not started"

    def test_status_collection_in_progress(self, tmp_path):
        # Collection data but no extracted masks (e.g. data/BBRF/sub-999/ses-02).
        def build(d):
            attempt = d / "collection" / "rittriggeredball_run-01_attempt-01"
            attempt.mkdir(parents=True)
            (attempt / "MR.0001").write_bytes(b"\x00")

        assert self._status_for(tmp_path, build) == "Collection in progress"

    def test_status_preprocessed(self, tmp_path):
        def build(d):
            preproc = d / "preprocessing"
            preproc.mkdir()
            (preproc / "sub-01_ses-01_task-rest_preproc.nii.gz").write_bytes(b"\x00")

        assert self._status_for(tmp_path, build) == "Preprocessed"

    def test_status_masks_extracted(self, tmp_path):
        def build(d):
            masks = d / "mask_extraction" / "03_masks"
            masks.mkdir(parents=True)
            (masks / "sub-01_ses-01_DMN_thr.nii.gz").write_bytes(b"\x00")

        assert self._status_for(tmp_path, build) == "Masks extracted"

    def test_status_masks_extracted_multilocalizer(self, tmp_path):
        def build(d):
            masks = d / "mask_extraction" / "loc_0_rest" / "03_masks"
            masks.mkdir(parents=True)
            (masks / "sub-01_ses-01_DMN_thr.nii.gz").write_bytes(b"\x00")

        assert self._status_for(tmp_path, build) == "Masks extracted"

    def test_status_masks_approved(self, tmp_path):
        def build(d):
            me = d / "mask_extraction"
            (me / "03_masks").mkdir(parents=True)
            (me / "03_masks" / "sub-01_ses-01_DMN_thr.nii.gz").write_bytes(b"\x00")
            (me / ".masks_approved").write_text("")

        assert self._status_for(tmp_path, build) == "Masks approved"

    def test_status_reference_coregistered(self, tmp_path):
        def build(d):
            masks = d / "mask_extraction" / "04_reference" / "masks"
            masks.mkdir(parents=True)
            (masks / "sub-01_ses-01_DMN_to_task-reference_run-01_thr.nii.gz").write_bytes(b"\x00")

        assert self._status_for(tmp_path, build) == "Reference coregistered"

    def test_status_realtime_reference_ready(self, tmp_path):
        def build(d):
            masks = d / "feedback" / "01_reference" / "masks"
            masks.mkdir(parents=True)
            (masks / "sub-01_ses-01_DMN_to_task-reference_run-02_thr.nii.gz").write_bytes(b"\x00")

        assert self._status_for(tmp_path, build) == "Realtime reference ready"

    def test_status_feedback_in_progress(self, tmp_path):
        # task_state.json lives under feedback/02_feedback_output (e.g.
        # data/rtCHR/sub-rtchr01/ses-01), not directly under feedback/.
        def build(d):
            output = d / "feedback" / "02_feedback_output"
            output.mkdir(parents=True)
            (output / "task_state.json").write_text("{}")

        status = self._status_for(tmp_path, build)
        assert status == "Feedback in progress"

    def test_has_feedback_uses_feedback_output_path(self, tmp_path):
        def build(d):
            output = d / "feedback" / "02_feedback_output"
            output.mkdir(parents=True)
            (output / "task_state.json").write_text("{}")

        ses_dir = tmp_path / "sub-01" / "ses-01"
        ses_dir.mkdir(parents=True)
        build(ses_dir)
        sessions = detect_existing_sessions(tmp_path, "sub-01")
        assert sessions[0]["has_feedback"] is True

    def test_status_reports_furthest_milestone(self, tmp_path):
        # A session with collection + masks + feedback reports only the furthest step.
        def build(d):
            attempt = d / "collection" / "rest_run-01_attempt-01"
            attempt.mkdir(parents=True)
            (attempt / "MR.0001").write_bytes(b"\x00")
            masks = d / "mask_extraction" / "03_masks"
            masks.mkdir(parents=True)
            (masks / "sub-01_ses-01_DMN_thr.nii.gz").write_bytes(b"\x00")
            output = d / "feedback" / "02_feedback_output"
            output.mkdir(parents=True)
            (output / "task_state.json").write_text("{}")

        assert self._status_for(tmp_path, build) == "Feedback in progress"

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
