"""Tests for pineuro.session — orchestration functions and run sequence building."""

from pineuro.session import (
    FeedbackConfig,
    MaskConfig,
    PathConfig,
    SessionConfig,
    TaskConfig,
    _build_run_sequence,
    reset_session_state,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_config(tmp_path, task_params=None):
    """Build a minimal SessionConfig for testing."""
    base_dir = tmp_path / "study"
    base_dir.mkdir(parents=True, exist_ok=True)
    session_dir = base_dir / "sub-01"
    session_dir.mkdir(exist_ok=True)

    paths = PathConfig(base_dir=base_dir, participant_id="sub-01")

    return SessionConfig(
        paths=paths,
        feedback=FeedbackConfig(),
        mask=MaskConfig(),
        task=TaskConfig(task_params=task_params or {}),
    )


# =============================================================================
# _build_run_sequence
# =============================================================================


class TestBuildRunSequence:
    def test_with_transfer_runs(self, tmp_path):
        config = _make_config(tmp_path, task_params={"use_transfer_runs": True})
        seq = _build_run_sequence(config, n_feedback_runs=2)
        assert seq == ["transfer", "feedback", "feedback", "transfer"]

    def test_without_transfer_runs(self, tmp_path):
        config = _make_config(tmp_path, task_params={"use_transfer_runs": False})
        seq = _build_run_sequence(config, n_feedback_runs=3)
        assert seq == ["feedback", "feedback", "feedback"]

    def test_default_includes_transfers(self, tmp_path):
        config = _make_config(tmp_path)
        seq = _build_run_sequence(config, n_feedback_runs=2)
        assert seq == ["transfer", "feedback", "feedback", "transfer"]

    def test_single_feedback_run(self, tmp_path):
        config = _make_config(tmp_path, task_params={"use_transfer_runs": False})
        seq = _build_run_sequence(config, n_feedback_runs=1)
        assert seq == ["feedback"]


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
        from pineuro.tasks.session import SessionState

        config = _make_config(tmp_path, task_params={"use_transfer_runs": False})
        session_dir = config.paths.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create session state with completed runs matching sequence length
        state = SessionState.reset(session_dir, task_id="ball")
        state.max_feedback_runs = 1
        # Manually add completed run
        from pineuro.tasks.session import RunInfo

        state.completed_runs.append(RunInfo(run_number=1, run_type="feedback", volumes_processed=0))
        state.save()

        mask_files = [tmp_path / "mask.nii.gz"]
        result = run_next_task(config, mask_files, n_feedback_runs=1)
        assert result is None
