"""Tests for pineuro.gui.components.pipeline_runner pure helpers."""

import queue
import threading
from unittest.mock import MagicMock

import pytest

from pineuro.gui.components.pipeline_runner import PipelineRunner, _select_sham_task_mask_names


class _FakeProcess:
    """Minimal stand-in for a multiprocessing.Process used by the watchdog."""

    def __init__(self, alive: bool = True):
        self._alive = alive
        self.terminated = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False


class TestSelectShamTaskMaskNames:
    def test_hemisphere_split_full_pair(self):
        # hemisphere_split produces both hemispheres; both roles must be
        # driven by the sham pair, not fall back to the real non-target.
        task = _select_sham_task_mask_names(
            sham_mask_names=["SomMotA_right", "SomMotA_left"],
            real_mask_names=["SomMotA", "DefaultA"],
        )
        assert task == ["SomMotA_right", "SomMotA_left"]

    def test_two_networks_full_pair(self):
        task = _select_sham_task_mask_names(
            sham_mask_names=["ContA", "SalA"],
            real_mask_names=["SomMotA", "DefaultA"],
        )
        assert task == ["ContA", "SalA"]

    def test_single_sham_mask_swaps_target_only(self):
        # Degraded case (only one sham mask resolved, e.g. a missing hemisphere
        # mask): the target role is swapped to sham, the real non-target is kept.
        task = _select_sham_task_mask_names(
            sham_mask_names=["SalA"],
            real_mask_names=["SomMotA", "DefaultA"],
        )
        assert task == ["SalA", "DefaultA"]

    def test_more_than_two_sham_masks_truncates(self):
        task = _select_sham_task_mask_names(
            sham_mask_names=["A", "B", "C"],
            real_mask_names=["X", "Y"],
        )
        assert task == ["A", "B"]

    def test_returns_copy_not_alias(self):
        # Mutating the result must not clobber the caller's sham list.
        sham = ["SomMotA_right", "SomMotA_left"]
        task = _select_sham_task_mask_names(sham, ["SomMotA", "DefaultA"])
        task.append("extra")
        assert sham == ["SomMotA_right", "SomMotA_left"]


class TestRunManualMaskImport:
    def test_raises_when_not_configured(self, tmp_path):
        # A manual localizer whose per-participant files were never chosen
        # must fail with a clear, actionable message — not a cryptic error.
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.participant_id = "sub-01"
        runner.overwrite = False
        runner._emit_progress = lambda *a, **k: None  # type: ignore[method-assign]

        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "native"}
        with pytest.raises(ValueError, match="not configured"):
            runner._run_manual_mask_import(
                loc,
                base_dir=tmp_path,
                root_dir=tmp_path / "mask_extraction",
                qc_dir=tmp_path / "qc",
            )


class TestEscalateChildStop:
    """The stop watchdog must only ever kill the process it was armed for."""

    def _bare_runner(self) -> PipelineRunner:
        runner = PipelineRunner.__new__(PipelineRunner)
        runner._task_lock = threading.Lock()
        return runner

    def test_terminates_only_the_armed_process(self):
        # Regression: a watchdog armed for run A (proc_a) must not terminate the
        # new run's process (proc_b), even though _current_process now points at B.
        # This is the orphaned-Timer bug that killed feedback re-runs early.
        runner = self._bare_runner()
        proc_a = _FakeProcess(alive=True)
        proc_b = _FakeProcess(alive=True)
        runner._current_process = proc_b

        runner._escalate_child_stop(proc_a)

        assert proc_a.terminated is True
        assert proc_b.terminated is False

    def test_noop_when_armed_process_already_exited(self):
        # A child that stopped cleanly within the grace period is not alive when
        # the watchdog fires; it must be left untouched.
        runner = self._bare_runner()
        proc = _FakeProcess(alive=False)

        runner._escalate_child_stop(proc)

        assert proc.terminated is False


class _NeverExitingProcess:
    """A child that ignores terminate/kill; every join() must carry a timeout."""

    pid = 4242
    exitcode = None

    def __init__(self):
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return True

    def join(self, timeout=None) -> None:
        assert timeout is not None, "unbounded join on a child that never exits"

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _ExitedProcess:
    exitcode = 0

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None) -> None:
        pass


def _finished_thread() -> threading.Thread:
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    return thread


def _teardown_runner() -> PipelineRunner:
    runner = PipelineRunner.__new__(PipelineRunner)
    runner._task_lock = threading.Lock()
    runner._child_done = threading.Event()
    runner._child_done_at = None
    runner._stop_watchdog = None
    runner._current_process = None
    runner._child_stop_event = None
    runner._child_show_questions = None
    runner._child_result = None
    runner._child_error = None
    runner._child_has_post_run_questions = False
    runner._task_mirror_callback = None
    return runner


class TestFinishTaskStepIsBounded:
    """A task subprocess that won't exit must never hang the step (or the GUI)."""

    def test_localizer_teardown_escalates_and_returns(self, caplog):
        runner = _teardown_runner()
        proc = _NeverExitingProcess()
        runner._current_process = proc

        runner._finish_task_step(proc, _finished_thread(), force_timeout=0.01)

        assert proc.terminated and proc.killed
        assert runner._current_process is None
        assert "abandoning it" in caplog.text

    def test_feedback_child_that_reported_done_gets_only_the_grace_period(self, monkeypatch):
        monkeypatch.setattr("pineuro.gui.components.pipeline_runner._GRACEFUL_STOP_TIMEOUT", 0.01)
        runner = _teardown_runner()
        runner._child_done.set()  # the reader saw DONE: the run itself is over
        proc = _NeverExitingProcess()

        runner._finish_task_step(proc, _finished_thread())

        assert proc.terminated and proc.killed

    def test_clean_exit_is_logged_relative_to_done(self, caplog):
        runner = _teardown_runner()
        runner._child_done_at = 0.0

        with caplog.at_level("INFO", logger="pineuro.gui.components.pipeline_runner"):
            runner._finish_task_step(_ExitedProcess(), _finished_thread(), force_timeout=0.01)

        assert "Task subprocess exited (exit code 0)" in caplog.text


class TestDrainChildMessages:
    def test_keeps_draining_after_done_without_acting_on_late_messages(self):
        from pineuro.tasks import subprocess_runner as sr

        runner = _teardown_runner()
        frames: list = []
        runner._task_mirror_callback = frames.append
        q: queue.Queue = queue.Queue()
        q.put((sr.MSG_RESULT, {"events_path": "/x"}))
        q.put((sr.MSG_DONE,))
        q.put((sr.MSG_MIRROR, {"type": "screenshot"}))  # still in flight when DONE arrived

        runner._drain_child_messages(_ExitedProcess(), q, runner._child_done)

        assert q.empty()  # read, so the child's queue feeder could flush and exit
        assert frames == []  # but not acted on
        assert runner._child_result == {"events_path": "/x"}
        assert runner._child_done.is_set() and runner._child_done_at is not None

    def test_failing_handler_is_logged_not_fatal(self, caplog):
        from pineuro.tasks import subprocess_runner as sr

        runner = _teardown_runner()
        runner._emit_progress = MagicMock(side_effect=RuntimeError("UI gone"))
        q: queue.Queue = queue.Queue()
        q.put((sr.MSG_PROGRESS, "hello"))
        q.put((sr.MSG_RESULT, {"ok": True}))
        q.put((sr.MSG_DONE,))

        runner._drain_child_messages(_ExitedProcess(), q, runner._child_done)

        assert runner._child_result == {"ok": True}
        assert "Error handling task subprocess message" in caplog.text


class TestBuildTaskExtractionParams:
    def _config(self, tmp_path, n_localizers=1, **loc_over):
        loc = {
            "type": "rit",
            "id": "rit",
            "tr": 1.2,
            "contrasts": {"rumination_network": "rumination - distraction"},
            "extraction": {"threshold_method": "cluster_extent", "min_cluster_size": 40},
        }
        loc.update(loc_over)
        rest = {"type": "rest", "id": "rest", "networks": ["DefaultA"]}
        return {"paths": {"base_dir": str(tmp_path)}, "localizers": [loc, rest][:n_localizers]}

    def _events(self, tmp_path):
        events = tmp_path / "preprocessing" / "sub-01_ses-01_task-rit_run-01_attempt-01_events.tsv"
        events.parent.mkdir(parents=True)
        events.write_text("onset\tduration\ttrial_type\n")
        return events

    def test_single_localizer_params(self, tmp_path):
        from pineuro.gui.components.pipeline_runner import build_task_extraction_params

        events = self._events(tmp_path)
        params = build_task_extraction_params(self._config(tmp_path), 0, "sub-01", "ses-01", overwrite=False)

        assert params["root_dir"] == tmp_path / "mask_extraction"
        assert params["events"] == str(events)
        assert params["threshold_method"] == "cluster_extent"
        assert params["min_cluster_size"] == 40
        assert params["ref_file"].endswith("sub-01_ses-01_task-rit_boldref.nii.gz")
        assert (params["drift_model"], params["hrf_model"]) == ("cosine", "spm")
        assert "mask_overrides" not in params

    def test_multi_localizer_root_carries_saved_overrides(self, tmp_path):
        from pineuro.gui.components.pipeline_runner import build_task_extraction_params
        from pineuro.mask_extraction import save_mask_overrides

        self._events(tmp_path)
        overrides = {"rumination_network": {"kind": "rethreshold", "min_cluster_size": 20}}
        save_mask_overrides(tmp_path / "mask_extraction" / "loc_0_rit", overrides)

        params = build_task_extraction_params(self._config(tmp_path, 2), 0, "sub-01", "ses-01", overwrite=True)

        assert params["root_dir"] == tmp_path / "mask_extraction" / "loc_0_rit"
        assert params["mask_overrides"] == overrides
        assert params["overwrite"] is True

    def test_default_contrasts_when_unset(self, tmp_path):
        from pineuro.gui.components.pipeline_runner import build_task_extraction_params
        from pineuro.tasks.task_registry import get_task_metadata

        self._events(tmp_path)
        params = build_task_extraction_params(
            self._config(tmp_path, contrasts={}), 0, "sub-01", "ses-01", overwrite=False
        )
        assert params["contrasts"] == get_task_metadata("rit").default_contrasts

    def test_missing_events_raises(self, tmp_path):
        from pineuro.gui.components.pipeline_runner import build_task_extraction_params

        with pytest.raises(FileNotFoundError, match="events"):
            build_task_extraction_params(self._config(tmp_path), 0, "sub-01", "ses-01", overwrite=False)


class TestBrainMaskParams:
    LOC = {"type": "rit", "preprocessing": {"brain_mask_threshold_fraction": 0.6}}

    def test_study_defaults_then_pineuro_defaults(self, tmp_path):
        from pineuro.gui.components.pipeline_runner import brain_mask_params

        assert brain_mask_params(self.LOC, tmp_path, "sub-01", "ses-01") == {
            "brain_mask_threshold_fraction": 0.6,
            "brain_mask_cleanup": 2,
        }
        assert brain_mask_params({"type": "rest"}, tmp_path, "sub-01", "ses-01") == {
            "brain_mask_threshold_fraction": 0.5,
            "brain_mask_cleanup": 2,
        }

    @pytest.mark.parametrize(("adjusted", "expected_fraction"), [(True, 0.8), (False, 0.6)])
    def test_operator_choice_wins(self, tmp_path, adjusted, expected_fraction):
        import json

        from pineuro.gui.components.pipeline_runner import brain_mask_params

        record = {"threshold_fraction": 0.8, "cleanup": 4, "adjusted_by_operator": adjusted}
        (tmp_path / "sub-01_ses-01_task-rit_preproc_mask.json").write_text(json.dumps(record))
        params = brain_mask_params(self.LOC, tmp_path, "sub-01", "ses-01")
        assert params["brain_mask_threshold_fraction"] == expected_fraction
        assert params["brain_mask_cleanup"] == (4 if adjusted else 2)
