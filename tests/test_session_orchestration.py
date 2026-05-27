"""Tests for pineuro.session — orchestration functions and run sequence building."""

from pineuro.session import (
    FeedbackRunConfig,
    LocalizerConfig,
    PathConfig,
    SessionConfig,
    _build_run_sequence,
    reset_session_state,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_config(tmp_path, task_params=None, use_transfer_runs=True):
    """Build a minimal SessionConfig for testing."""
    base_dir = tmp_path / "study"
    base_dir.mkdir(parents=True, exist_ok=True)
    session_dir = base_dir / "sub-01"
    session_dir.mkdir(exist_ok=True)

    paths = PathConfig(base_dir=base_dir, participant_id="sub-01")

    return SessionConfig(
        paths=paths,
        localizers=[LocalizerConfig(id="rest", type="rest", networks=["DefaultA", "ContA"])],
        feedback_runs=[
            FeedbackRunConfig(
                task="ball",
                mask_source="rest",
                task_params=task_params or {},
                use_transfer_runs=use_transfer_runs,
            )
        ],
    )


# =============================================================================
# _build_run_sequence
# =============================================================================


class TestBuildRunSequence:
    def test_with_transfer_runs(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=2, use_transfer_runs=True),
        ]
        seq = _build_run_sequence(config, n_feedback_runs=2)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["transfer", "feedback", "feedback", "transfer"]

    def test_without_transfer_runs(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=3, use_transfer_runs=False),
        ]
        seq = _build_run_sequence(config, n_feedback_runs=3)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["feedback", "feedback", "feedback"]

    def test_empty_feedback_runs(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = []
        seq = _build_run_sequence(config, n_feedback_runs=2)
        assert seq == []

    def test_single_feedback_run(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=1, use_transfer_runs=False),
        ]
        seq = _build_run_sequence(config, n_feedback_runs=1)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["feedback"]

    def test_multi_feedback_runs_with_transfers(self, tmp_path):
        """Multiple feedback task types with independent transfer settings."""
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="es", mask_source="rest", max_runs=2, use_transfer_runs=False),
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=2, use_transfer_runs=True),
        ]
        seq = _build_run_sequence(config, n_feedback_runs=0)
        run_types = [rt for rt, _ in seq]
        frc_indices = [idx for _, idx in seq]

        assert run_types == [
            "feedback",
            "feedback",  # es (no transfers)
            "transfer",
            "feedback",
            "feedback",
            "transfer",  # ball (with transfers)
        ]
        assert frc_indices == [0, 0, 1, 1, 1, 1]

    def test_multi_feedback_runs_all_no_transfer(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="a", mask_source="rest", max_runs=1, use_transfer_runs=False),
            FeedbackRunConfig(task="b", mask_source="rest", max_runs=3, use_transfer_runs=False),
        ]
        seq = _build_run_sequence(config, n_feedback_runs=0)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["feedback", "feedback", "feedback", "feedback"]

    def test_n_feedback_runs_arg_ignored(self, tmp_path):
        """n_feedback_runs arg is ignored when feedback_runs is populated."""
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=3, use_transfer_runs=False),
        ]
        seq = _build_run_sequence(config, n_feedback_runs=99)
        assert len(seq) == 3


# =============================================================================
# FeedbackRunConfig
# =============================================================================


class TestFeedbackRunConfig:
    def test_defaults(self):
        frc = FeedbackRunConfig()
        assert frc.task == "ball"
        assert frc.max_runs == 1
        assert frc.tr == 1.2
        assert frc.glm.motion_correction == "siemens_moco"

    def test_invalid_motion_correction(self):
        import pytest

        from pineuro.session import GLMConfig

        with pytest.raises(ValueError, match="motion_correction"):
            GLMConfig(motion_correction="invalid")

    def test_roundtrip_via_session_config(self, tmp_path):
        """FeedbackRunConfig survives SessionConfig serialization."""
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=2, tr=1.5),
            FeedbackRunConfig(
                task="thermo",
                mask_source="rest",
                max_runs=1,
                feedback_mode="connectivity",
            ),
        ]
        d = config._to_dict()
        assert "feedback_runs" in d
        assert len(d["feedback_runs"]) == 2
        assert d["feedback_runs"][0]["task"] == "ball"
        assert d["feedback_runs"][1]["feedback_mode"] == "connectivity"

        # Round-trip through _from_dict
        config2 = SessionConfig._from_dict(d)
        assert len(config2.feedback_runs) == 2
        assert config2.feedback_runs[0].task == "ball"
        assert config2.feedback_runs[0].max_runs == 2
        assert config2.feedback_runs[0].tr == 1.5
        assert config2.feedback_runs[1].task == "thermo"
        assert config2.feedback_runs[1].feedback_mode == "connectivity"


# =============================================================================
# reset_session_state
# =============================================================================


class TestResetSessionState:
    def test_creates_fresh_state(self, tmp_path):
        config = _make_config(tmp_path)
        # Ensure session_dir exists
        session_dir = config.paths.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)

        state = reset_session_state(config)
        assert state.completed_runs == []


# =============================================================================
# run_next_task — returns None when session complete
# =============================================================================


class TestRunNextTaskComplete:
    def test_returns_none_when_all_runs_done(self, tmp_path):
        from pineuro.session import run_next_task
        from pineuro.tasks.session import RunInfo, SessionState

        config = _make_config(tmp_path, task_params={"use_transfer_runs": False}, use_transfer_runs=False)
        session_dir = config.paths.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create session state with completed runs matching sequence length
        state = SessionState.reset(session_dir, task_id="ball")
        state.max_feedback_runs = 1
        # Manually add completed run
        state.completed_runs.append(RunInfo(run_number=1, run_type="feedback", volumes_processed=0, acq=1))
        state.save()

        mask_files = [tmp_path / "mask.nii.gz"]
        result = run_next_task(config, mask_files, n_feedback_runs=1)
        assert result is None
