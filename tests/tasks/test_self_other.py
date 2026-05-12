"""Tests for pineuro.tasks.self_other — metadata, stim-file resolution + parsing."""

import csv

import pytest


class TestSelfOtherMetadata:
    def test_registered(self):
        from pineuro.tasks.task_registry import list_available_tasks

        assert "self_other" in list_available_tasks()

    def test_localizer_role(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("self_other")
        assert "localizer" in meta.roles


def _stub(stim_dir=None, stim_file=None, participant="sub-01", run=1, block_order=None):
    """Build a SelfOtherTask with just the attrs the load/resolve methods use."""
    from pineuro.tasks.self_other import SelfOtherTask

    task = SelfOtherTask.__new__(SelfOtherTask)
    task.stim_dir = stim_dir
    task._stim_file_override = stim_file
    task.participant_id = participant
    task.run_number = run
    task.block_order = block_order or ["voice", "rest", "voice", "rest"]
    task._voice_blocks = []
    task._voice_block_types = []
    task._questions = []
    return task


class TestResolveStimFile:
    def test_override_takes_priority(self, tmp_path):
        f = tmp_path / "explicit.tsv"
        f.write_text("a\tb\tc\td\n")
        task = _stub(stim_dir=tmp_path, stim_file=f)
        assert task._resolve_stim_file() == f

    def test_missing_override_raises(self, tmp_path):
        missing = tmp_path / "doesnt_exist.tsv"
        task = _stub(stim_dir=tmp_path, stim_file=missing)
        with pytest.raises(FileNotFoundError):
            task._resolve_stim_file()

    def test_missing_stim_dir_raises(self, tmp_path):
        task = _stub(stim_dir=tmp_path / "missing", stim_file=None)
        with pytest.raises(FileNotFoundError):
            task._resolve_stim_file()

    def test_glob_pattern_selfother(self, tmp_path):
        target = tmp_path / "sub-01_selfother_run-01.tsv"
        target.write_text("a\tb\tc\td\n")
        task = _stub(stim_dir=tmp_path)
        assert task._resolve_stim_file() == target

    def test_glob_no_match_raises(self, tmp_path):
        (tmp_path / "unrelated.tsv").write_text("a\n")
        task = _stub(stim_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            task._resolve_stim_file()


class TestLoadStimFile:
    @staticmethod
    def _write_stim(path, rows):
        with open(path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)

    def test_two_voice_blocks_infer_types(self, tmp_path):
        # Block order: voice, rest, voice, rest → 2 voice blocks (10 audio rows)
        stim = tmp_path / "sub-01_selfother_run-01.tsv"
        rows = []
        for i in range(5):  # block 1 — "self"
            rows.append([f"audio/self_{i}.wav", "2.0", "0.5", f"q_self_{i}.png" if i < 4 else ""])
        for i in range(5):  # block 2 — "other"
            rows.append([f"audio/other_{i}.wav", "2.0", "0.5", ""])
        self._write_stim(stim, rows)

        task = _stub(stim_dir=tmp_path)
        task._load_stim_file()

        assert task._voice_block_types == ["self", "other"]
        assert len(task._voice_blocks) == 2
        assert len(task._voice_blocks[0]) == 5
        # block_order length is 4 → 4 question entries collected
        assert len(task._questions) == 4

    def test_audio_row_durations_parsed_as_float(self, tmp_path):
        stim = tmp_path / "sub-01_selfother_run-01.tsv"
        rows = [[f"self_{i}.wav", "1.5", "0.25", ""] for i in range(5)]
        rows.extend([[f"other_{i}.wav", "1.5", "0.25", ""] for i in range(5)])
        self._write_stim(stim, rows)

        task = _stub(stim_dir=tmp_path)
        task._load_stim_file()
        assert task._voice_blocks[0][0]["duration"] == 1.5
        assert task._voice_blocks[0][0]["isi"] == 0.25

    def test_short_stim_file_raises(self, tmp_path):
        # block_order has 2 voice blocks → needs 10 rows; provide only 3
        stim = tmp_path / "sub-01_selfother_run-01.tsv"
        self._write_stim(stim, [["a", "1.0", "0.5", ""], ["b", "1.0", "0.5", ""], ["c", "1.0", "0.5", ""]])
        task = _stub(stim_dir=tmp_path)
        with pytest.raises(ValueError, match="rows"):
            task._load_stim_file()

    def test_missing_trailing_columns_padded(self, tmp_path):
        # CSV rows with only 1 or 2 cols (no isi/question) should be padded.
        # We need 5 audio rows for a single voice block, so use block_order=["voice"].
        stim = tmp_path / "sub-01_selfother_run-01.tsv"
        rows = [[f"self_{i}.wav", "1.0", "0.5"] for i in range(5)]  # missing column 3
        self._write_stim(stim, rows)
        task = _stub(stim_dir=tmp_path, block_order=["voice"])
        task._load_stim_file()
        # Question for the single block defaults to "" when column missing.
        assert task._questions == [""]
