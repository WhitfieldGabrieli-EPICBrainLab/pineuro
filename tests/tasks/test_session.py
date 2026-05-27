"""Tests for pineuro.tasks.session — SessionState (JSON-backed) and RunInfo."""


class TestSessionState:
    def test_create_for_task(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        assert state.task_id == "ball"
        assert state.next_run_number == 1
        assert state.feedback_run_count == 0

    def test_save_and_load(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        state.set("custom_key", 42)
        state.save()

        loaded = SessionState.load(tmp_path, "ball")
        assert loaded.get("custom_key") == 42

    def test_record_run(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        state.record_run(run_type="feedback", volumes_processed=100, acq=1)

        assert state.feedback_run_count == 1
        assert state.next_run_number == 2
        assert state.completed_runs[0].run_type == "feedback"
        assert state.completed_runs[0].volumes_processed == 100

    def test_reset(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        state.record_run(run_type="feedback", volumes_processed=50, acq=1)
        state.save()

        reset = SessionState.reset(tmp_path, "ball")
        assert reset.feedback_run_count == 0
        assert reset.next_run_number == 1


class TestRunInfo:
    def test_construction(self):
        from pineuro.tasks.session import RunInfo

        info = RunInfo(run_number=1, run_type="feedback", volumes_processed=200, acq=1)
        assert info.run_number == 1
        assert info.run_type == "feedback"
        assert info.acq == 1

    def test_to_dict_roundtrip(self):
        from pineuro.tasks.session import RunInfo

        info = RunInfo(
            run_number=2,
            run_type="transfer",
            volumes_processed=150,
            acq=1,
            task_state_snapshot={"score": 0.75},
        )
        d = info.to_dict()
        restored = RunInfo.from_dict(d)

        assert restored.run_number == 2
        assert restored.run_type == "transfer"
        assert restored.acq == 1
        assert restored.task_state_snapshot["score"] == 0.75
