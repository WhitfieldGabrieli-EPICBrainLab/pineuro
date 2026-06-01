"""Tests for attempt discovery in the unified attempt dialog.

Covers ``scan_collection_attempts``: directory discovery, attempt-number
ranking, and the complete/incomplete determination based on ``n_volumes``
vs ``expected_volumes``. Dialog rendering is covered by a construction
smoke test elsewhere.
"""

from pineuro.gui.components.unified_attempt_dialog import (
    AttemptInfo,
    scan_collection_attempts,
)


def _make_attempt(collection_dir, task, run, attempt, n_files, *, with_subdir=False):
    """Create one attempt folder with ``n_files`` plain files inside.

    Optionally also drop a subdirectory to verify it is NOT counted as a volume.
    """
    folder = collection_dir / f"{task}_run-{run:02d}_attempt-{attempt:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (folder / f"vol-{i:04d}.dcm").write_bytes(b"")
    if with_subdir:
        (folder / "nested").mkdir()
    return folder


# ---------------------------------------------------------------------------
# Empty / missing inputs
# ---------------------------------------------------------------------------


def test_missing_collection_dir_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert scan_collection_attempts(missing, "task-rest", 1, 100) == []


def test_no_matching_attempts_returns_empty(tmp_path):
    # Folder exists but contains no attempt directories for this run.
    (tmp_path / "unrelated_dir").mkdir()
    assert scan_collection_attempts(tmp_path, "task-rest", 1, 100) == []


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovers_all_attempts_for_run(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 100)
    _make_attempt(tmp_path, "task-rest", 1, 2, 50)
    _make_attempt(tmp_path, "task-rest", 1, 3, 100)

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert len(attempts) == 3
    assert {a.attempt_num for a in attempts} == {1, 2, 3}
    assert all(isinstance(a, AttemptInfo) for a in attempts)


def test_directory_field_points_at_discovered_folder(tmp_path):
    folder = _make_attempt(tmp_path, "task-rest", 1, 7, 100)

    [attempt] = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert attempt.attempt_num == 7
    assert attempt.directory == folder


# ---------------------------------------------------------------------------
# Filtering: only the requested run/task, only directories
# ---------------------------------------------------------------------------


def test_excludes_other_runs(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 100)
    _make_attempt(tmp_path, "task-rest", 2, 1, 100)
    _make_attempt(tmp_path, "task-rest", 10, 1, 100)

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert len(attempts) == 1
    assert attempts[0].directory.name == "task-rest_run-01_attempt-01"


def test_excludes_other_tasks(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 100)
    _make_attempt(tmp_path, "task-selfother", 1, 1, 100)

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert len(attempts) == 1
    assert attempts[0].directory.name.startswith("task-rest_")


def test_run_number_is_zero_padded_in_pattern(tmp_path):
    # run=1 must match attempt-* folders named run-01, not run-1.
    folder = tmp_path / "task-rest_run-1_attempt-01"
    folder.mkdir()
    (folder / "vol-0000.dcm").write_bytes(b"")

    assert scan_collection_attempts(tmp_path, "task-rest", 1, 100) == []


def test_plain_file_matching_pattern_is_ignored(tmp_path):
    # A regular file whose name matches the glob must not be treated as an attempt.
    (tmp_path / "task-rest_run-01_attempt-01").write_bytes(b"not a directory")

    assert scan_collection_attempts(tmp_path, "task-rest", 1, 100) == []


# ---------------------------------------------------------------------------
# Volume counting
# ---------------------------------------------------------------------------


def test_n_volumes_counts_only_files_not_subdirs(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 40, with_subdir=True)

    [attempt] = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    # The nested subdirectory must not inflate the volume count.
    assert attempt.n_volumes == 40


def test_empty_attempt_has_zero_volumes(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 0)

    [attempt] = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert attempt.n_volumes == 0
    assert attempt.is_complete is False


# ---------------------------------------------------------------------------
# Complete / incomplete determination
# ---------------------------------------------------------------------------


def test_complete_when_volumes_meet_expected(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 100)

    [attempt] = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert attempt.n_volumes == 100
    assert attempt.expected_volumes == 100
    assert attempt.is_complete is True


def test_complete_when_volumes_exceed_expected(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 105)

    [attempt] = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert attempt.n_volumes == 105
    assert attempt.is_complete is True


def test_incomplete_when_one_volume_short(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 99)

    [attempt] = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert attempt.n_volumes == 99
    assert attempt.is_complete is False


def test_mixed_complete_and_incomplete(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 100)  # complete
    _make_attempt(tmp_path, "task-rest", 1, 2, 30)  # incomplete

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 100)
    by_num = {a.attempt_num: a for a in attempts}

    assert by_num[1].is_complete is True
    assert by_num[2].is_complete is False


# ---------------------------------------------------------------------------
# Ranking / ordering (highest attempt number first)
# ---------------------------------------------------------------------------


def test_attempts_sorted_descending_by_attempt_num(tmp_path):
    # Create out of order to prove sorting is not just insertion order.
    _make_attempt(tmp_path, "task-rest", 1, 2, 100)
    _make_attempt(tmp_path, "task-rest", 1, 10, 100)
    _make_attempt(tmp_path, "task-rest", 1, 1, 100)
    _make_attempt(tmp_path, "task-rest", 1, 3, 100)

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert [a.attempt_num for a in attempts] == [10, 3, 2, 1]


def test_attempt_num_is_numeric_not_lexical(tmp_path):
    # Lexical sort would put "9" after "10"; numeric sort puts 10 first.
    _make_attempt(tmp_path, "task-rest", 1, 9, 100)
    _make_attempt(tmp_path, "task-rest", 1, 10, 100)

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 100)

    assert [a.attempt_num for a in attempts] == [10, 9]


def test_expected_volumes_propagated_to_every_attempt(tmp_path):
    _make_attempt(tmp_path, "task-rest", 1, 1, 50)
    _make_attempt(tmp_path, "task-rest", 1, 2, 50)

    attempts = scan_collection_attempts(tmp_path, "task-rest", 1, 200)

    assert all(a.expected_volumes == 200 for a in attempts)
    assert all(a.is_complete is False for a in attempts)
