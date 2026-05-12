"""Tests for pineuro.tasks.thermometer — metadata, condition file resolution, parsing."""

from unittest.mock import MagicMock

import pytest


class TestThermometerMetadata:
    def test_registered(self):
        from pineuro.tasks.task_registry import list_available_tasks

        assert "thermometer" in list_available_tasks()

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("thermometer")
        assert meta.task_id == "thermometer"
        assert meta.n_networks == 1  # single-network feedback


def _stub_thermometer(stim_dir, *, participant="sub-01", run=1, session="feedback"):
    """Build a ThermometerTask via __new__ with just enough state to drive lookups."""
    from pineuro.tasks.thermometer import ThermometerTask

    task = ThermometerTask.__new__(ThermometerTask)
    task.stim_dir = stim_dir
    task.coordinator = MagicMock(subject=participant, run_number=run, session=session)
    return task


class TestResolveConditionFile:
    def test_missing_stim_dir_raises(self, tmp_path):
        task = _stub_thermometer(stim_dir=None)
        with pytest.raises(FileNotFoundError):
            task._resolve_condition_file()

    def test_nonexistent_stim_dir_raises(self, tmp_path):
        task = _stub_thermometer(stim_dir=tmp_path / "missing")
        with pytest.raises(FileNotFoundError):
            task._resolve_condition_file()

    def test_finds_exact_bids_pattern(self, tmp_path):
        target = tmp_path / "sub-01_ses-feedback_task-stg_run-01.csv"
        target.write_text("instruction_text,voice_file1,voice_file2,voice_file3,voice_file4,question\n")
        task = _stub_thermometer(tmp_path)
        assert task._resolve_condition_file() == target

    def test_falls_back_to_thermometer_pattern(self, tmp_path):
        target = tmp_path / "sub-01_thermometer_run-01.csv"
        target.write_text("instruction_text,voice_file1,voice_file2,voice_file3,voice_file4,question\n")
        task = _stub_thermometer(tmp_path)
        assert task._resolve_condition_file() == target

    def test_no_match_raises(self, tmp_path):
        (tmp_path / "unrelated.csv").write_text("x\n")
        task = _stub_thermometer(tmp_path)
        with pytest.raises(FileNotFoundError):
            task._resolve_condition_file()


class TestLoadConditions:
    @staticmethod
    def _write_csv(path, rows):
        header = "instruction_text,voice_file1,voice_file2,voice_file3,voice_file4,question\n"
        body = "\n".join(",".join(r) for r in rows)
        path.write_text(header + body + "\n")

    def test_classifies_listen_vs_ignore(self, tmp_path):
        csv_path = tmp_path / "sub-01_thermometer_run-01.csv"
        self._write_csv(
            csv_path,
            [
                ["Listen carefully", "a1.wav", "a2.wav", "a3.wav", "a4.wav", "How clear?"],
                ["Ignore the voice", "b1.wav", "b2.wav", "b3.wav", "b4.wav", "How distracted?"],
            ],
        )
        task = _stub_thermometer(tmp_path)
        blocks = task._load_conditions()
        assert [b.block_type for b in blocks] == ["listen", "ignore"]
        assert blocks[0].instruction_text == "Listen carefully"
        assert blocks[0].question_text == "How clear?"

    def test_resolves_relative_audio_paths(self, tmp_path):
        csv_path = tmp_path / "sub-01_thermometer_run-01.csv"
        self._write_csv(
            csv_path,
            [["Listen", "voices/a1.wav", "voices/a2.wav", "voices/a3.wav", "voices/a4.wav", "q?"]],
        )
        task = _stub_thermometer(tmp_path)
        blocks = task._load_conditions()
        # Relative paths get joined with stim_dir
        assert blocks[0].audio_files[0] == str(tmp_path / "voices/a1.wav")

    def test_preserves_absolute_audio_paths(self, tmp_path):
        csv_path = tmp_path / "sub-01_thermometer_run-01.csv"
        abs_voice = tmp_path / "elsewhere" / "v1.wav"
        self._write_csv(
            csv_path,
            [["Listen", str(abs_voice), "v2.wav", "v3.wav", "v4.wav", "q?"]],
        )
        task = _stub_thermometer(tmp_path)
        blocks = task._load_conditions()
        # First file was absolute, so it stays absolute (not re-joined to stim_dir)
        assert blocks[0].audio_files[0] == str(abs_voice)
        # Second was relative
        assert blocks[0].audio_files[1] == str(tmp_path / "v2.wav")

    def test_strips_whitespace_in_fields(self, tmp_path):
        csv_path = tmp_path / "sub-01_thermometer_run-01.csv"
        self._write_csv(
            csv_path,
            [["  Listen now  ", " a1.wav ", "a2.wav", "a3.wav", "a4.wav", " trailing? "]],
        )
        task = _stub_thermometer(tmp_path)
        blocks = task._load_conditions()
        assert blocks[0].instruction_text == "Listen now"
        assert blocks[0].question_text == "trailing?"
        # File paths also stripped
        assert blocks[0].audio_files[0] == str(tmp_path / "a1.wav")
