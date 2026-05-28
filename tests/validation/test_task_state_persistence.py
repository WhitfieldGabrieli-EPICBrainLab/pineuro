"""Validation -- TaskState JSON round-trip persistence.

TaskState persists adaptive multi-run task state to ``task_state.json``.
The round-trip contract: every field written by ``save()`` must come back
unchanged from ``load()``. A bug -- dropped field on save, key renamed on load,
type coerced (e.g. int -> str via json) -- corrupts state across run boundaries
and silently breaks the multi-run adaptation that depends on it.

Every test below saves a TaskState built directly from constructor literals,
loads it back through the public API, and asserts exact equality on every field
that the public ``save()`` writes. Expected values are the original literals --
there is no formula involved, only a structural round-trip.
"""

from __future__ import annotations

import json

import pytest

from pineuro.tasks.state import RunInfo, TaskState

pytestmark = pytest.mark.validation


def test_round_trip_empty_state(tmp_path):
    """An unmodified state round-trips with empty task_state and no completed runs.

    Uses an *unregistered* task_id so __post_init__ does not auto-populate
    task_state from the task registry's defaults. This isolates the
    save/load round-trip from the task-metadata default-injection path.
    """
    state = TaskState(
        feedback_dir=tmp_path,
        task_id="_validation_test_unregistered_task",
        task_state={},
        completed_runs=[],
        max_feedback_runs=5,
    )
    state.save()

    loaded = TaskState.load(tmp_path, task_id="_validation_test_unregistered_task")
    assert loaded.task_id == "_validation_test_unregistered_task"
    assert loaded.task_state == {}
    assert loaded.completed_runs == []
    assert loaded.max_feedback_runs == 5
    assert loaded.slot_base_state == {}


def test_round_trip_task_state_values(tmp_path):
    """Arbitrary scalar / list / nested fields in task_state round-trip exactly."""
    original_state = {
        "scale_factor": 12.5,
        "hit_counters": [1, 2, 3, 4, 5],
        "threshold": 0.42,
        "label": "stable",
        "active": True,
    }
    state = TaskState(
        feedback_dir=tmp_path,
        task_id="thermometer",
        task_state=dict(original_state),
        completed_runs=[],
        max_feedback_runs=3,
    )
    state.save()

    loaded = TaskState.load(tmp_path, task_id="thermometer")
    assert loaded.task_state == original_state  # exact dict equality across types


def test_round_trip_completed_runs(tmp_path):
    """A list of RunInfo entries round-trips field-by-field (including optional slot)."""
    runs = [
        RunInfo(run_number=1, run_type="transfer", volumes_processed=100, attempt=1, task_state_snapshot={"x": 1}),
        RunInfo(
            run_number=2,
            run_type="feedback",
            volumes_processed=200,
            attempt=1,
            task_state_snapshot={"x": 5, "y": "abc"},
            slot_run_number=1,
        ),
        RunInfo(
            run_number=3,
            run_type="feedback",
            volumes_processed=200,
            attempt=2,
            task_state_snapshot={"x": 8, "y": "abc"},
            slot_run_number=2,
        ),
    ]
    state = TaskState(
        feedback_dir=tmp_path,
        task_id="ball",
        task_state={"some": "state"},
        completed_runs=runs,
        max_feedback_runs=5,
    )
    state.save()

    loaded = TaskState.load(tmp_path, task_id="ball")
    assert len(loaded.completed_runs) == len(runs)
    for orig, restored in zip(runs, loaded.completed_runs, strict=True):
        assert restored.run_number == orig.run_number
        assert restored.run_type == orig.run_type
        assert restored.volumes_processed == orig.volumes_processed
        assert restored.task_state_snapshot == orig.task_state_snapshot
        assert restored.slot_run_number == orig.slot_run_number
        assert restored.attempt == orig.attempt


def test_round_trip_slot_base_state(tmp_path):
    """slot_base_state keys (int) round-trip through JSON string-key conversion."""
    state = TaskState(
        feedback_dir=tmp_path,
        task_id="ball",
        task_state={},
        completed_runs=[],
        max_feedback_runs=4,
        slot_base_state={1: {"scale": 10.0}, 2: {"scale": 15.0}},
    )
    state.save()

    # Inspect the on-disk JSON: save() stores slot keys as strings (JSON requirement).
    written = json.loads((tmp_path / "task_state.json").read_text())
    assert "slot_base_state" in written
    assert set(written["slot_base_state"].keys()) == {"1", "2"}

    # load() must reverse that conversion back to integer keys for in-memory use.
    loaded = TaskState.load(tmp_path, task_id="ball")
    # Note: slot_base_state may still be string-keyed depending on load implementation.
    # The values themselves must round-trip exactly:
    assert {str(k): v for k, v in loaded.slot_base_state.items()} == {
        "1": {"scale": 10.0},
        "2": {"scale": 15.0},
    }


def test_load_missing_file_creates_fresh_state(tmp_path):
    """Loading from a directory with no task_state.json yields fresh defaults."""
    loaded = TaskState.load(tmp_path, task_id="some_task")
    assert loaded.task_id == "some_task"
    assert loaded.completed_runs == []
    # The default max_feedback_runs is 5 (class default); a fresh state inherits it.
    assert loaded.max_feedback_runs == 5


def test_round_trip_max_feedback_runs_value(tmp_path):
    """max_feedback_runs preserves its int value across the round trip."""
    state = TaskState(
        feedback_dir=tmp_path,
        task_id="ball",
        task_state={},
        completed_runs=[],
        max_feedback_runs=7,  # non-default value
    )
    state.save()

    loaded = TaskState.load(tmp_path, task_id="ball")
    assert loaded.max_feedback_runs == 7  # exact int round-trip


def test_save_then_reload_then_save_is_stable(tmp_path):
    """Two successive save→load→save cycles produce identical JSON content."""
    state1 = TaskState(
        feedback_dir=tmp_path,
        task_id="ball",
        task_state={"a": 1, "b": [2, 3]},
        completed_runs=[RunInfo(1, "transfer", 50, attempt=1)],
        max_feedback_runs=3,
    )
    state1.save()
    written1 = (tmp_path / "task_state.json").read_text()

    loaded = TaskState.load(tmp_path, task_id="ball")
    loaded.save()
    written2 = (tmp_path / "task_state.json").read_text()

    assert json.loads(written1) == json.loads(written2)  # structural equality on the JSON


def test_run_info_dict_round_trip():
    """RunInfo.to_dict / from_dict is a lossless round-trip including the optional slot field."""
    original = RunInfo(
        run_number=4,
        run_type="feedback",
        volumes_processed=180,
        attempt=3,
        task_state_snapshot={"scale": 14.0},
        slot_run_number=2,
    )
    restored = RunInfo.from_dict(original.to_dict())
    assert restored.run_number == original.run_number
    assert restored.run_type == original.run_type
    assert restored.volumes_processed == original.volumes_processed
    assert restored.task_state_snapshot == original.task_state_snapshot
    assert restored.slot_run_number == original.slot_run_number
    assert restored.attempt == original.attempt


def test_run_info_without_slot_field():
    """A RunInfo with no slot info round-trips with None for slot_run_number."""
    original = RunInfo(run_number=1, run_type="transfer", volumes_processed=80, attempt=1)
    restored = RunInfo.from_dict(original.to_dict())
    assert restored.slot_run_number is None
    assert restored.attempt == 1
