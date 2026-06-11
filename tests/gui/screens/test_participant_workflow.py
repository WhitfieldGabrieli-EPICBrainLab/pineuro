"""Behavioral tests for participant-workflow status detection.

The :class:`~pineuro.gui.screens.participant_workflow.ParticipantWorkflowScreen`
auto-detects each pipeline step's status from on-disk artifacts so an
interrupted session can be resumed. That filesystem-driven status computation is
pure logic and is what is exercised here.

The real screen ``__init__`` constructs a live Flet layout and pulls in heavy
pineuro modules, so we never build the full screen. Instead we drive the
status-check methods on a bare instance (``__new__``), wiring in only the
attributes each method actually reads (the established pattern for these GUI
screens — see ``test_session_qualified_masks.py``). The directory-scanning
helpers they delegate to (``PipelineRunner.find_available_collections``) are the
real ones; nothing under test is mocked.
"""

from pathlib import Path
from types import SimpleNamespace

from pineuro.gui.screens.participant_workflow import (
    ParticipantWorkflowScreen,
    StepStatus,
    _build_feedback_step_ids,
    _loc,
    _session_number_from_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_screen(session_config: dict | None = None) -> ParticipantWorkflowScreen:
    """A bare screen instance with only the attributes status checks read.

    ``__init__`` builds a live Flet layout, so bypass it. ``_check_feedback_slot_status``
    and ``_get_feedback_step_frc_index`` read ``self.study_config.session_config``;
    nothing else in the methods under test touches ``self``.
    """
    screen = ParticipantWorkflowScreen.__new__(ParticipantWorkflowScreen)
    screen.study_config = SimpleNamespace(session_config=session_config or {})
    return screen


def _make_collection_attempt(collection_dir: Path, task: str, run: int, attempt: int, n_files: int) -> Path:
    """Create a ``{task}_run-{run:02d}_attempt-{attempt}`` folder with ``n_files`` files."""
    folder = collection_dir / f"{task}_run-{run:02d}_attempt-{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    # Scanners often emit extension-less UID-named files; the counter counts any file.
    for i in range(n_files):
        (folder / f"MR.1.3.{i}").write_text("x")
    return folder


def _write_feedbacklog(path: Path, n_rows: int) -> Path:
    """Write a feedbacklog TSV with a header line plus ``n_rows`` data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["volume\tactivation"]
    lines += [f"{i}\t0.{i}" for i in range(n_rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
class TestSessionNumberFromId:
    def test_parses_two_digit_session(self):
        assert _session_number_from_id("ses-03") == 3

    def test_parses_multi_digit_session(self):
        assert _session_number_from_id("ses-12") == 12

    def test_unparseable_falls_back_to_one(self):
        # No participant has session 0; an unparseable id means "the only session".
        assert _session_number_from_id("baseline") == 1

    def test_non_digit_suffix_falls_back_to_one(self):
        assert _session_number_from_id("ses-baseline") == 1


class TestLoc:
    def test_loc_returns_indexed_localizer(self):
        config = {"localizers": [{"type": "rest"}, {"type": "rit"}]}
        assert _loc(config, 1) == {"type": "rit"}

    def test_loc_out_of_range_returns_empty(self):
        assert _loc({"localizers": [{"type": "rest"}]}, 5) == {}

    def test_loc_missing_key_returns_empty(self):
        assert _loc({}, 0) == {}

    def test_loc_none_localizers_returns_empty(self):
        # config.get("localizers") can be an explicit None.
        assert _loc({"localizers": None}, 0) == {}


class TestBuildFeedbackStepIds:
    def test_expands_feedback_runs_with_transfers(self):
        # Two ball feedback runs with transfers bracket the feedback runs:
        # pre-transfer, feedback x2, post-transfer (single source of truth in utils).
        config = {"feedback_runs": [{"task": "ball", "max_runs": 2, "use_transfer_runs": True}]}
        ids = [sid for sid, _label in _build_feedback_step_ids(config)]
        assert ids == ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]

    def test_labels_are_human_readable(self):
        config = {"feedback_runs": [{"task": "ball", "max_runs": 1, "use_transfer_runs": False}]}
        steps = _build_feedback_step_ids(config)
        # Single feedback run, no transfers: one step labelled from the task's display name.
        sid, label = steps[0]
        assert len(steps) == 1
        assert sid == "feedback_1"
        assert label.startswith("Feedback Run 1 (") and label.endswith(")")

    def test_no_feedback_runs_yields_no_steps(self):
        assert _build_feedback_step_ids({}) == []


# ---------------------------------------------------------------------------
# _check_attempt_collection_status — best status across attempt folders
# ---------------------------------------------------------------------------
class TestCheckAttemptCollectionStatus:
    def _status(self, collection_dir: Path, task: str, run: int, expected: int) -> StepStatus:
        screen = _make_screen()
        return screen._check_attempt_collection_status(collection_dir, task, run, expected)

    def test_no_attempts_is_pending(self, tmp_path):
        coll = tmp_path / "collection"
        coll.mkdir()
        assert self._status(coll, "task-rest", 1, 200) == StepStatus.PENDING

    def test_missing_directory_is_pending(self, tmp_path):
        assert self._status(tmp_path / "nope", "task-rest", 1, 200) == StepStatus.PENDING

    def test_single_complete_attempt_is_complete(self, tmp_path):
        coll = tmp_path / "collection"
        _make_collection_attempt(coll, "task-rest", run=1, attempt=1, n_files=200)
        assert self._status(coll, "task-rest", 1, 200) == StepStatus.COMPLETE

    def test_single_partial_attempt_is_interrupted(self, tmp_path):
        coll = tmp_path / "collection"
        _make_collection_attempt(coll, "task-rest", run=1, attempt=1, n_files=50)
        assert self._status(coll, "task-rest", 1, 200) == StepStatus.INTERRUPTED

    def test_any_complete_attempt_wins_over_partial(self, tmp_path):
        # A first partial attempt followed by a complete retry => COMPLETE.
        coll = tmp_path / "collection"
        _make_collection_attempt(coll, "task-rest", run=1, attempt=1, n_files=50)
        _make_collection_attempt(coll, "task-rest", run=1, attempt=2, n_files=200)
        assert self._status(coll, "task-rest", 1, 200) == StepStatus.COMPLETE

    def test_run_number_is_zero_padded_in_match(self, tmp_path):
        # run=2 must match the run-02 folder, not run-2/run-20.
        coll = tmp_path / "collection"
        _make_collection_attempt(coll, "task-reference", run=2, attempt=1, n_files=5)
        assert self._status(coll, "task-reference", 2, 5) == StepStatus.COMPLETE
        # And a query for a different run sees no attempts.
        assert self._status(coll, "task-reference", 1, 5) == StepStatus.PENDING

    def test_task_prefix_is_isolated(self, tmp_path):
        # Attempts for a different task must not be counted toward this one.
        coll = tmp_path / "collection"
        _make_collection_attempt(coll, "task-rest", run=1, attempt=1, n_files=200)
        assert self._status(coll, "task-reference", 1, 200) == StepStatus.PENDING


# ---------------------------------------------------------------------------
# _check_feedback_slot_status — attempt-based glob + best status
# ---------------------------------------------------------------------------
class TestCheckFeedbackSlotStatus:
    # A single ball feedback run with transfers => steps feedback_1 (task-ball
    # run-01) and transfer_1 (task-balltransfer run-01).
    CONFIG = {"feedback_runs": [{"task": "ball", "max_runs": 1, "use_transfer_runs": True}]}
    PID = "sub-01"
    SID = "ses-01"

    def _status(self, step_id: str, output_dir: Path, expected: int, config: dict | None = None) -> StepStatus:
        screen = _make_screen(config or self.CONFIG)
        return screen._check_feedback_slot_status(
            step_id=step_id,
            output_dir=output_dir,
            participant_id=self.PID,
            session_id=self.SID,
            expected_volumes=expected,
        )

    def _log_name(self, task_entity: str, run: int, attempt: int) -> str:
        return f"{self.PID}_{self.SID}_task-{task_entity}_run-{run:02d}_attempt-{attempt}_feedbacklog.tsv"

    def test_unknown_step_is_pending(self, tmp_path):
        # A step id that is not in this config's feedback steps yields PENDING.
        assert self._status("feedback_99", tmp_path, 150) == StepStatus.PENDING

    def test_missing_output_dir_is_pending(self, tmp_path):
        assert self._status("feedback_1", tmp_path / "nope", 150) == StepStatus.PENDING

    def test_no_matching_logs_is_pending(self, tmp_path):
        out = tmp_path / "02_feedback_output"
        out.mkdir()
        assert self._status("feedback_1", out, 150) == StepStatus.PENDING

    def test_single_complete_attempt_is_complete(self, tmp_path):
        out = tmp_path / "02_feedback_output"
        _write_feedbacklog(out / self._log_name("ball", run=1, attempt=1), n_rows=150)
        assert self._status("feedback_1", out, 150) == StepStatus.COMPLETE

    def test_single_partial_attempt_is_interrupted(self, tmp_path):
        out = tmp_path / "02_feedback_output"
        _write_feedbacklog(out / self._log_name("ball", run=1, attempt=1), n_rows=40)
        assert self._status("feedback_1", out, 150) == StepStatus.INTERRUPTED

    def test_complete_attempt_beats_partial(self, tmp_path):
        # COMPLETE > INTERRUPTED: a complete retry rescues a partial first attempt.
        out = tmp_path / "02_feedback_output"
        _write_feedbacklog(out / self._log_name("ball", run=1, attempt=1), n_rows=40)
        _write_feedbacklog(out / self._log_name("ball", run=1, attempt=2), n_rows=150)
        assert self._status("feedback_1", out, 150) == StepStatus.COMPLETE

    def test_transfer_step_uses_transfer_task_entity(self, tmp_path):
        # transfer_1 maps to the task-balltransfer entity, NOT task-ball.
        out = tmp_path / "02_feedback_output"
        _write_feedbacklog(out / self._log_name("balltransfer", run=1, attempt=1), n_rows=150)
        assert self._status("transfer_1", out, 150) == StepStatus.COMPLETE
        # The same files must not satisfy the feedback_1 (task-ball) slot.
        assert self._status("feedback_1", out, 150) == StepStatus.PENDING

    def test_other_participant_logs_are_ignored(self, tmp_path):
        # A different participant's log in the same dir must not count.
        out = tmp_path / "02_feedback_output"
        out.mkdir()
        other = f"sub-99_{self.SID}_task-ball_run-01_attempt-1_feedbacklog.tsv"
        _write_feedbacklog(out / other, n_rows=150)
        assert self._status("feedback_1", out, 150) == StepStatus.PENDING

    def test_other_run_number_is_ignored(self, tmp_path):
        # run-02 logs must not satisfy the run-01 slot.
        out = tmp_path / "02_feedback_output"
        _write_feedbacklog(out / self._log_name("ball", run=2, attempt=1), n_rows=150)
        assert self._status("feedback_1", out, 150) == StepStatus.PENDING


# ---------------------------------------------------------------------------
# _get_feedback_step_frc_index — step id -> feedback_runs index
# ---------------------------------------------------------------------------
class TestGetFeedbackStepFrcIndex:
    def _idx(self, step_id: str, config: dict):
        return _make_screen(config)._get_feedback_step_frc_index(step_id)

    def test_maps_feedback_and_transfer_to_first_frc(self):
        config = {"feedback_runs": [{"task": "ball", "max_runs": 2, "use_transfer_runs": True}]}
        # Both transfers and both feedback runs belong to feedback_runs[0].
        assert self._idx("transfer_1", config) == 0
        assert self._idx("feedback_1", config) == 0
        assert self._idx("feedback_2", config) == 0
        assert self._idx("transfer_2", config) == 0

    def test_maps_second_task_to_second_frc(self):
        # ball (no transfers) then gauge: gauge's step belongs to frc index 1.
        config = {
            "feedback_runs": [
                {"task": "ball", "max_runs": 1, "use_transfer_runs": False},
                {"task": "gauge", "max_runs": 1, "use_transfer_runs": False},
            ]
        }
        assert self._idx("feedback_1", config) == 0
        assert self._idx("feedback_2", config) == 1

    def test_unknown_step_returns_none(self):
        config = {"feedback_runs": [{"task": "ball", "max_runs": 1, "use_transfer_runs": False}]}
        assert self._idx("feedback_99", config) is None

    def test_no_feedback_runs_returns_none(self):
        assert self._idx("feedback_1", {}) is None


# ---------------------------------------------------------------------------
# _detect_completed_steps — localizer-only sessions still seed the mask pipeline
# ---------------------------------------------------------------------------
def _make_detect_screen(tmp_path: Path, session_config: dict, *, session_id: str = "ses-01"):
    """Bare screen wired with the attributes ``_detect_completed_steps`` reads.

    ``base_dir`` points at an empty ``tmp_path`` so every on-disk status check
    returns PENDING — what remains is the pure step-seeding logic.
    """
    screen = ParticipantWorkflowScreen.__new__(ParticipantWorkflowScreen)
    session_config.setdefault("paths", {})["base_dir"] = str(tmp_path)
    screen.study_config = SimpleNamespace(session_config=session_config)
    screen.participant_id = "sub-01"
    screen.session_id = session_id
    screen._bids_prefix = f"sub-01_{session_id}"
    screen.step_statuses = {}
    screen._initial_qc_approved = False
    screen._is_mni_atlas_mode = False
    screen._is_all_reused_mode = False
    return screen


class TestLocalizerOnlyStepSeeding:
    """Issue #3: a localizer-only session (no feedback) must still seed the
    preprocessing/extraction/approval pipeline so it produces masks a later
    feedback session reuses — without seeding any realtime/feedback steps."""

    def test_single_localizer_no_feedback_seeds_mask_pipeline(self, tmp_path):
        config = {
            "localizers": [{"type": "rest", "n_runs": 2, "networks": ["DefaultA"]}],
            "feedback_runs": [],
            # Drop the realtime tail so only the localizer pipeline remains to inspect.
            "reference": {"initial_enabled": False, "realtime_enabled": False},
        }
        screen = _make_detect_screen(tmp_path, config)
        screen._detect_completed_steps()

        # The masks-producing steps are seeded (PENDING — nothing on disk).
        assert screen.step_statuses.get("preprocessing") == StepStatus.PENDING
        assert screen.step_statuses.get("mask_extraction") == StepStatus.PENDING
        assert screen.step_statuses.get("mask_approval") == StepStatus.PENDING
        # The section is shown because this session produces masks.
        assert screen._has_mask_extraction is True
        # No feedback this session => no realtime/feedback steps seeded.
        assert "reference_realtime" not in screen.step_statuses
        assert "coregistration_realtime" not in screen.step_statuses
        assert not any(s.startswith("feedback_") for s in screen.step_statuses)

    def test_multi_localizer_no_feedback_seeds_per_localizer_steps(self, tmp_path):
        config = {
            "localizers": [
                {"type": "rest", "n_runs": 2, "networks": ["DefaultA"]},
                {"type": "rit", "n_runs": 1},
            ],
            "feedback_runs": [],
            "reference": {"initial_enabled": False, "realtime_enabled": False},
        }
        screen = _make_detect_screen(tmp_path, config)
        screen._detect_completed_steps()

        for loc_idx in (0, 1):
            assert screen.step_statuses.get(f"localizer_{loc_idx}_preprocessing") == StepStatus.PENDING
            assert screen.step_statuses.get(f"localizer_{loc_idx}_extraction") == StepStatus.PENDING
            assert screen.step_statuses.get(f"localizer_{loc_idx}_approval") == StepStatus.PENDING
        assert screen._has_mask_extraction is True
        assert not any(s.startswith("feedback_") for s in screen.step_statuses)

    def test_next_step_advances_into_preprocessing_after_collection(self, tmp_path):
        # The regression: previously _get_next_step skipped the unseeded
        # preprocessing step and reported the session complete after collection.
        config = {
            "localizers": [{"type": "rest", "n_runs": 2, "networks": ["DefaultA"]}],
            "feedback_runs": [],
            "reference": {"initial_enabled": False, "realtime_enabled": False},
        }
        screen = _make_detect_screen(tmp_path, config)
        screen._detect_completed_steps()
        # Mark collection done; the next pending step must be preprocessing, not None.
        screen.step_statuses["rest_run_1"] = StepStatus.COMPLETE
        screen.step_statuses["rest_run_2"] = StepStatus.COMPLETE

        nxt = screen._get_next_step()
        assert nxt is not None
        assert nxt[0] == "preprocessing"

    def test_all_reused_localizer_hides_mask_extraction(self, tmp_path):
        # Every localizer reused => no masks produced this session => section hidden.
        config = {
            "localizers": [{"type": "rest", "reuse_from_session": "ses-01", "networks": ["DefaultA"]}],
            "feedback_runs": [],
        }
        screen = _make_detect_screen(tmp_path, config, session_id="ses-02")
        screen._detect_completed_steps()
        assert screen._has_mask_extraction is False
