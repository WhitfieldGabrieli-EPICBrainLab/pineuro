"""Tests for pineuro.tasks.localizer_base — record_event, save_events.

The PsychoPy display methods (show_fixation, run_paradigm) are not unit-tested.
"""

import pandas as pd
import pytest


class _ConcreteLocalizer:
    """Holder for the lazily-built concrete subclass (avoids top-level PsychoPy import)."""


@pytest.fixture
def localizer(tmp_path):
    """Construct a real `BaseLocalizerTask` via a concrete subclass with stub abstracts."""
    from pineuro.tasks.localizer_base import BaseLocalizerTask

    class _StubLocalizer(BaseLocalizerTask):
        task_id = "stub_localizer"

        def setup(self):  # pragma: no cover
            pass

        def run_paradigm(self):  # pragma: no cover
            pass

    return _StubLocalizer(output_dir=tmp_path, participant_id="sub-01", session_id="ses-01", run=2, attempt=3)


class TestRecordEvent:
    def test_records_basic_event(self, localizer):
        localizer.record_event(onset=10.0, duration=2.0, trial_type="self")
        assert localizer._events == [{"onset": 10.0, "duration": 2.0, "trial_type": "self"}]

    def test_rounds_onset_and_duration(self, localizer):
        localizer.record_event(onset=10.123456, duration=2.987654, trial_type="x")
        ev = localizer._events[0]
        assert ev["onset"] == 10.123
        assert ev["duration"] == 2.988

    def test_extras_preserved(self, localizer):
        localizer.record_event(onset=0.0, duration=1.0, trial_type="q", response="left", rt=0.42)
        ev = localizer._events[0]
        assert ev["response"] == "left"
        assert ev["rt"] == 0.42


class TestSaveEvents:
    def test_default_path_uses_bids_prefix(self, localizer):
        localizer.record_event(onset=0.0, duration=1.0, trial_type="self")
        path = localizer.save_events()
        # Prefix uses task_id=stub_localizer, run=2, attempt=3
        assert path.name == "sub-01_ses-01_task-stub_localizer_run-02_attempt-03_events.tsv"
        assert path.exists()

    def test_custom_path_respected(self, localizer, tmp_path):
        localizer.record_event(onset=0.0, duration=1.0, trial_type="a")
        target = tmp_path / "custom.tsv"
        out = localizer.save_events(target)
        assert out == target
        assert target.exists()

    def test_tsv_columns_bids_first(self, localizer):
        localizer.record_event(onset=0.0, duration=1.0, trial_type="a", extra_col="foo")
        path = localizer.save_events()
        df = pd.read_csv(path, sep="\t")
        # BIDS-required columns come first, then extras
        assert list(df.columns[:3]) == ["onset", "duration", "trial_type"]
        assert "extra_col" in df.columns

    def test_empty_events_writes_header_only(self, localizer):
        path = localizer.save_events()
        df = pd.read_csv(path, sep="\t")
        assert list(df.columns) == ["onset", "duration", "trial_type"]
        assert len(df) == 0

    def test_creates_output_directory(self, localizer, tmp_path):
        # Point to a fresh subdir that doesn't exist yet
        localizer.output_dir = tmp_path / "nested" / "events_dir"
        localizer.record_event(onset=0.0, duration=1.0, trial_type="a")
        path = localizer.save_events()
        assert path.parent == localizer.output_dir
        assert path.exists()


class TestTaskStateRecording:
    """Localizers persist task state across runs when a TaskState is threaded in.

    The mechanism is inherited from TaskBase; a localizer that declares no
    task_state_fields simply gets no TaskState (None) and records nothing.
    """

    def test_run_record_fields_localizer_defaults(self, localizer):
        # localizer fixture is built with run=2, attempt=3
        localizer.record_event(onset=0.0, duration=1.0, trial_type="a")
        localizer.record_event(onset=1.0, duration=1.0, trial_type="b")
        fields = localizer._run_record_fields()
        assert fields["run_type"] == "localizer"
        assert fields["slot_run_number"] == 2  # from self.run_number
        assert fields["attempt"] == 3  # from self.attempt_number
        assert fields["volumes_processed"] == 2  # event count, not a constant 0
        assert fields["merge_into_task_state"] is False  # per-run data lives on the RunInfo

    def test_record_run_completion_is_noop_without_state(self, localizer, tmp_path):
        # No TaskState threaded in -> no file, no error (stateless localizer).
        assert localizer._task_state is None
        localizer._record_run_completion()
        assert not (tmp_path / "task_state.json").exists()

    def test_stateless_localizer_still_writes_run_ledger(self, localizer, tmp_path):
        # A localizer that declares no task_state_fields and records no snapshot
        # still gets a task_state.json with an empty task_state dict but a
        # populated completed_runs ledger (the rest/self_other case).
        from pineuro.tasks.state import TaskState

        state_dir = tmp_path / "task_state_stub_localizer"
        localizer._task_state = TaskState.create_for_task(state_dir, "stub_localizer")
        assert localizer._task_state_snapshot == {}  # nothing adaptive to record

        localizer._record_run_completion()

        assert (state_dir / "task_state.json").exists()
        assert localizer._task_state.task_state == {}  # no adaptive fields
        assert len(localizer._task_state.completed_runs) == 1

    def test_record_run_completion_writes_state(self, localizer, tmp_path):
        from pineuro.tasks.state import TaskState

        state_dir = tmp_path / "task_state_stub_localizer"
        localizer._task_state = TaskState.create_for_task(state_dir, "stub_localizer")
        localizer._task_state_snapshot = {"keywords_used": ["a", "b"]}

        localizer._record_run_completion()

        assert (state_dir / "task_state.json").exists()
        assert len(localizer._task_state.completed_runs) == 1
        run = localizer._task_state.completed_runs[0]
        assert run.run_type == "localizer"
        assert run.slot_run_number == 2
        assert run.attempt == 3
        assert run.task_state_snapshot == {"keywords_used": ["a", "b"]}
        # Localizer per-run data lives on the RunInfo only; the accumulating
        # task_state dict stays empty (merge_into_task_state=False).
        assert localizer._task_state.task_state == {}

    def test_cross_run_persistence_reloads_from_same_dir(self, tmp_path):
        """A localizer's per-run record survives between runs via the per-task dir.

        Localizers record with merge_into_task_state=False, so the data lives on
        the RunInfo (read back per-slot via get_run_for_slot), not the flat dict.
        """
        from pineuro.tasks.state import TaskState

        state_dir = tmp_path / "task_state_loc"

        # Run 1 (slot 1) records what it showed.
        s1 = TaskState.create_for_task(state_dir, "stub_localizer")
        s1.record_run(
            run_type="localizer",
            volumes_processed=0,
            attempt=1,
            slot_run_number=1,
            task_state_snapshot={"seen": ["x"]},
            merge_into_task_state=False,
        )
        assert s1.task_state == {}  # flat dict stays empty

        # Run 2 reloads from the SAME dir and can read run 1's record per-slot.
        s2 = TaskState.load(state_dir, task_id="stub_localizer")
        assert s2.get_run_for_slot(1).task_state_snapshot == {"seen": ["x"]}
        s2.record_run(
            run_type="localizer",
            volumes_processed=0,
            attempt=1,
            slot_run_number=2,
            task_state_snapshot={"seen": ["x", "y"]},
            merge_into_task_state=False,
        )
        assert len(s2.completed_runs) == 2
        assert s2.task_state == {}

    def test_retry_replaces_slot_entry(self, tmp_path):
        """Re-running a slot (higher attempt) replaces its entry, not appends."""
        from pineuro.tasks.state import TaskState

        state = TaskState.create_for_task(tmp_path / "task_state_retry", "stub_localizer")
        state.record_run(
            run_type="localizer", volumes_processed=0, attempt=1, slot_run_number=1, task_state_snapshot={}
        )
        state.record_run(
            run_type="localizer", volumes_processed=0, attempt=2, slot_run_number=1, task_state_snapshot={}
        )

        assert len(state.completed_runs) == 1
        assert state.completed_runs[0].attempt == 2
