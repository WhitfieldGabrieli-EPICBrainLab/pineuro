"""Tests for pineuro.session — orchestration functions and run sequence building."""

from pineuro.session import (
    FeedbackRunConfig,
    LocalizerConfig,
    PathConfig,
    SessionConfig,
    _build_run_sequence,
    _is_last_feedback_for_frc,
    reset_task_state,
    resolve_localizer_masks,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_config(tmp_path, task_params=None, use_transfer_runs=True):
    """Build a minimal SessionConfig for testing."""
    base_dir = tmp_path / "study"
    base_dir.mkdir(parents=True, exist_ok=True)
    participant_dir = base_dir / "sub-01"
    participant_dir.mkdir(exist_ok=True)

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
# resolve_localizer_masks (per-localizer; fresh + reused)
# =============================================================================


class TestResolveLocalizerMasks:
    def _config(self, base_dir, session_id, localizers):
        return SessionConfig(
            paths=PathConfig(base_dir=base_dir, participant_id="sub-01", session_id=session_id),
            localizers=localizers,
            feedback_runs=[],
        )

    def _write_masks(self, base_dir, session_id, names):
        md = base_dir / "sub-01" / session_id / "mask_extraction" / "03_masks"
        md.mkdir(parents=True, exist_ok=True)
        for n in names:
            (md / f"sub-01_{session_id}_{n}_thr.nii.gz").write_bytes(b"")

    def _write_boldref(self, base_dir, session_id, task):
        pp = base_dir / "sub-01" / session_id / "preprocessing"
        pp.mkdir(parents=True, exist_ok=True)
        (pp / f"sub-01_{session_id}_task-{task}_boldref.nii.gz").write_bytes(b"")

    def test_fresh_localizer(self, tmp_path):
        base = tmp_path / "study"
        self._write_masks(base, "ses-01", ["DefaultA", "ContA"])
        self._write_boldref(base, "ses-01", "rest")
        cfg = self._config(base, "ses-01", [LocalizerConfig(id="rest", type="rest", networks=["DefaultA", "ContA"])])
        groups = resolve_localizer_masks(cfg)
        assert len(groups) == 1
        _loc, masks, ref = groups[0]
        assert {m.name for m in masks} == {"sub-01_ses-01_DefaultA_thr.nii.gz", "sub-01_ses-01_ContA_thr.nii.gz"}
        assert ref.name == "sub-01_ses-01_task-rest_boldref.nii.gz"

    def test_mix_reused_and_fresh(self, tmp_path):
        base = tmp_path / "study"
        # ses-01 produced rest masks; ses-02 freshly extracts a rit localizer.
        self._write_masks(base, "ses-01", ["DefaultA", "ContA"])
        self._write_boldref(base, "ses-01", "rest")
        self._write_masks(base, "ses-02", ["self"])
        self._write_boldref(base, "ses-02", "rit")
        cfg = self._config(
            base,
            "ses-02",
            [
                LocalizerConfig(id="rest", type="rest", networks=["DefaultA", "ContA"], reuse_from_session="ses-01"),
                LocalizerConfig(id="rit", type="rit", contrasts={"self": "self>other"}),
            ],
        )
        groups = resolve_localizer_masks(cfg)
        assert len(groups) == 2

        rest_loc, rest_masks, rest_ref = groups[0]
        assert rest_loc.is_reused
        assert {m.name for m in rest_masks} == {"sub-01_ses-01_DefaultA_thr.nii.gz", "sub-01_ses-01_ContA_thr.nii.gz"}
        assert rest_ref.name == "sub-01_ses-01_task-rest_boldref.nii.gz"

        rit_loc, rit_masks, rit_ref = groups[1]
        assert not rit_loc.is_reused
        assert {m.name for m in rit_masks} == {"sub-01_ses-02_self_thr.nii.gz"}
        assert rit_ref.name == "sub-01_ses-02_task-rit_boldref.nii.gz"

    def test_all_reused_feedback_only(self, tmp_path):
        # A feedback-only session: its single (synthesized) localizer reuses ses-01.
        base = tmp_path / "study"
        self._write_masks(base, "ses-01", ["Trigger", "Movement"])
        self._write_boldref(base, "ses-01", "rit")
        cfg = self._config(
            base,
            "ses-02",
            [
                LocalizerConfig(
                    id="rit",
                    type="rit",
                    contrasts={"Trigger": "a>b", "Movement": "c>d"},
                    reuse_from_session="ses-01",
                )
            ],
        )
        groups = resolve_localizer_masks(cfg)
        assert len(groups) == 1
        loc, masks, ref = groups[0]
        assert loc.is_reused
        assert {m.name for m in masks} == {
            "sub-01_ses-01_Trigger_thr.nii.gz",
            "sub-01_ses-01_Movement_thr.nii.gz",
        }
        assert ref.name == "sub-01_ses-01_task-rit_boldref.nii.gz"


# =============================================================================
# _build_run_sequence
# =============================================================================


class TestBuildRunSequence:
    def test_with_transfer_runs(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=2, use_transfer_runs=True),
        ]
        seq = _build_run_sequence(config)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["transfer", "feedback", "feedback", "transfer"]

    def test_without_transfer_runs(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=3, use_transfer_runs=False),
        ]
        seq = _build_run_sequence(config)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["feedback", "feedback", "feedback"]

    def test_empty_feedback_runs(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = []
        seq = _build_run_sequence(config)
        assert seq == []

    def test_single_feedback_run(self, tmp_path):
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=1, use_transfer_runs=False),
        ]
        seq = _build_run_sequence(config)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["feedback"]

    def test_multi_feedback_runs_with_transfers(self, tmp_path):
        """Multiple feedback task types with independent transfer settings."""
        config = _make_config(tmp_path)
        config.feedback_runs = [
            FeedbackRunConfig(task="es", mask_source="rest", max_runs=2, use_transfer_runs=False),
            FeedbackRunConfig(task="ball", mask_source="rest", max_runs=2, use_transfer_runs=True),
        ]
        seq = _build_run_sequence(config)
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
        seq = _build_run_sequence(config)
        run_types = [rt for rt, _ in seq]
        assert run_types == ["feedback", "feedback", "feedback", "feedback"]


# =============================================================================
# _is_last_feedback_for_frc
# =============================================================================


class TestIsLastFeedbackForFrc:
    """The per-task "last feedback run" flag that gates ``last_feedback_only`` questions."""

    def test_two_task_blocks_fire_at_each_block_end(self):
        # ball x3 (frc 0) then thermometer x2 (frc 1): True only at the last
        # feedback of each block, not just the session's final feedback run.
        seq = [("feedback", 0)] * 3 + [("feedback", 1)] * 2
        flags = [_is_last_feedback_for_frc(seq, i, frc_idx) for i, (_, frc_idx) in enumerate(seq)]
        assert flags == [False, False, True, False, True]

    def test_single_block_fires_only_on_final_feedback(self):
        seq = [("feedback", 0)] * 3
        flags = [_is_last_feedback_for_frc(seq, i, 0) for i in range(len(seq))]
        assert flags == [False, False, True]

    def test_trailing_transfer_does_not_delay_the_flag(self):
        # transfer -> feedback x2 -> transfer (all frc 0): the last feedback is the
        # last *feedback* step, even though a transfer follows it.
        seq = [("transfer", 0), ("feedback", 0), ("feedback", 0), ("transfer", 0)]
        assert _is_last_feedback_for_frc(seq, 1, 0) is False
        assert _is_last_feedback_for_frc(seq, 2, 0) is True


# =============================================================================
# FeedbackRunConfig
# =============================================================================


class TestFeedbackRunConfig:
    # `test_defaults` lives in test_session_config.py (the fuller copy); this
    # class keeps only orchestration-specific FeedbackRunConfig behavior.
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
# _compute_regressors_for_task
# =============================================================================


class TestComputeRegressorsForTask:
    """The session runner's design dispatch.

    Shares ``designs.design_for_metadata`` with the GUI pipeline runner, so both
    produce the same design matrix for a given task. See
    tests/tasks/test_designs.py for the mapping itself.
    """

    def test_block_task_gets_one_task_column(self, tmp_path):
        from pineuro.session import _compute_regressors_for_task
        from pineuro.tasks.task_registry import get_task_metadata

        config = _make_config(tmp_path)
        run_config = config.feedback_runs[0]
        regressors = _compute_regressors_for_task(get_task_metadata("ball"), config, run_config)

        assert regressors.shape == (run_config.n_volumes, 1)
        # Baseline volumes carry no task signal.
        assert regressors[0, 0] == 0.0

    def test_nuisance_only_task_gets_no_task_columns(self, tmp_path):
        """Nuisance-only returns a (n_volumes, 0) array, not None.

        FeedbackEstimator treats both as zero task columns; the array keeps the
        return type uniform and matches what the GUI has always passed.
        """
        from pineuro.session import _compute_regressors_for_task
        from pineuro.tasks.task_registry import get_task_metadata

        config = _make_config(tmp_path)
        run_config = config.feedback_runs[0]
        regressors = _compute_regressors_for_task(get_task_metadata("gauge"), config, run_config)

        assert regressors.shape == (run_config.n_volumes, 0)

    def test_matches_the_gui_dispatch(self, tmp_path):
        """Both runners must build the same design for the same task."""
        from pineuro.session import _compute_regressors_for_task
        from pineuro.tasks.designs import design_for_metadata
        from pineuro.tasks.task_registry import get_task_metadata

        config = _make_config(tmp_path)
        run_config = config.feedback_runs[0]

        for task_id in ("ball", "gauge"):
            metadata = get_task_metadata(task_id)
            from_session = _compute_regressors_for_task(metadata, config, run_config)
            from_gui = design_for_metadata(metadata, run_config.n_baseline).compute_regressors(
                run_config.n_volumes, run_config.tr
            )
            assert from_session.shape == from_gui.shape, task_id
            assert (from_session == from_gui).all(), task_id


# =============================================================================
# reset_task_state
# =============================================================================


class TestResetTaskState:
    def test_creates_fresh_state(self, tmp_path):
        config = _make_config(tmp_path)
        # Ensure feedback_dir exists
        feedback_dir = config.paths.feedback_dir
        feedback_dir.mkdir(parents=True, exist_ok=True)

        state = reset_task_state(config)
        assert state.completed_runs == []


# =============================================================================
# run_next_task — returns None when session complete
# =============================================================================


class TestRunNextTaskComplete:
    def test_returns_none_when_all_runs_done(self, tmp_path):
        from pineuro.session import run_next_task
        from pineuro.tasks.state import RunInfo, TaskState

        config = _make_config(tmp_path, task_params={"use_transfer_runs": False}, use_transfer_runs=False)
        feedback_dir = config.paths.feedback_dir
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Create task state with completed runs matching sequence length
        state = TaskState.reset(feedback_dir, task_id="ball")
        state.max_feedback_runs = 1
        # Manually add completed run
        state.completed_runs.append(RunInfo(run_number=1, run_type="feedback", volumes_processed=0, attempt=1))
        state.save()

        mask_files = [tmp_path / "mask.nii.gz"]
        result = run_next_task(config, mask_files)
        assert result is None
