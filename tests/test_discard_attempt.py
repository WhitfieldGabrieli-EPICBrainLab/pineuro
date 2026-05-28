"""Tests for pineuro.session.discard_attempt — discard-on-stop file cleanup."""

import shutil
from pathlib import Path

import pytest

from pineuro.feedback import get_slot_run_info
from pineuro.session import discard_attempt, get_localizer_run_info


def _touch(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestDiscardLocalizer:
    def test_removes_collection_dir(self, tmp_path):
        coll = tmp_path / "collection" / "task-rest_acq-01_run-01"
        _touch(coll / "a.dcm", "x")
        _touch(coll / "b.dcm", "y")

        removed = discard_attempt(
            tmp_path,
            subject="sub-01",
            session="ses-01",
            task="rest",
            acq=1,
            run=1,
            kind="localizer",
        )

        assert not coll.exists()
        assert removed == [coll]

    def test_missing_dir_returns_empty(self, tmp_path):
        # No collection dir exists yet — discard is a no-op, not an error.
        removed = discard_attempt(
            tmp_path,
            subject="sub-01",
            session="ses-01",
            task="rest",
            acq=1,
            run=1,
            kind="localizer",
        )
        assert removed == []

    def test_other_acqs_untouched(self, tmp_path):
        coll_dir = tmp_path / "collection"
        acq1 = coll_dir / "task-rest_acq-01_run-01"
        acq2 = coll_dir / "task-rest_acq-02_run-01"
        _touch(acq1 / "a.dcm")
        _touch(acq2 / "a.dcm")

        discard_attempt(tmp_path, subject="sub-01", session="ses-01", task="rest", acq=1, run=1, kind="localizer")

        assert not acq1.exists()
        assert acq2.exists()  # the other attempt is left alone

    def test_removes_preprocessing_events(self, tmp_path):
        """Localizer events TSV (and any per-attempt preprocessing file) is cleaned."""
        coll = tmp_path / "collection" / "task-rest_acq-01_run-01"
        _touch(coll / "a.dcm")
        preproc = tmp_path / "preprocessing"
        events = _touch(preproc / "sub-01_ses-01_task-rest_acq-01_run-01_events.tsv", "onset\tduration\n0\t10\n")
        # Sibling files from other attempts/tasks/runs must survive.
        other_acq = _touch(preproc / "sub-01_ses-01_task-rest_acq-02_run-01_events.tsv")
        other_run = _touch(preproc / "sub-01_ses-01_task-rest_acq-01_run-02_events.tsv")
        other_task = _touch(preproc / "sub-01_ses-01_task-rnt_acq-01_run-01_events.tsv")

        removed = discard_attempt(
            tmp_path, subject="sub-01", session="ses-01", task="rest", acq=1, run=1, kind="localizer"
        )

        assert not coll.exists()
        assert not events.exists()
        assert other_acq.exists()
        assert other_run.exists()
        assert other_task.exists()
        assert set(removed) == {coll, events}


class TestDiscardFeedback:
    def test_removes_all_artifacts(self, tmp_path):
        fb = tmp_path / "feedback" / "02_feedback_output"
        prefix = "sub-01_ses-01_task-feedback_acq-02_run-01"
        log = _touch(fb / f"{prefix}_feedbacklog.tsv", "header\nrow")
        design = _touch(fb / f"{prefix}_design.tsv")
        params = _touch(fb / f"{prefix}_parameters.json")
        nifti_dir = fb / "nifti_ses-01_task-feedback_acq-02_run-01"
        _touch(nifti_dir / "vol-0001.nii.gz")
        # Unrelated acq-03 file that must survive.
        survivor = _touch(fb / "sub-01_ses-01_task-feedback_acq-03_run-01_feedbacklog.tsv")

        removed = discard_attempt(
            tmp_path,
            subject="sub-01",
            session="ses-01",
            task="feedback",
            acq=2,
            run=1,
            kind="feedback",
        )

        for path in (log, design, params, nifti_dir):
            assert not path.exists(), f"{path} should be removed"
        assert survivor.exists()
        assert set(removed) == {log, design, params, nifti_dir}

    def test_missing_dir_returns_empty(self, tmp_path):
        removed = discard_attempt(
            tmp_path,
            subject="sub-01",
            session="ses-01",
            task="feedback",
            acq=1,
            run=1,
            kind="feedback",
        )
        assert removed == []

    def test_other_run_untouched(self, tmp_path):
        fb = tmp_path / "feedback" / "02_feedback_output"
        target = _touch(fb / "sub-01_ses-01_task-feedback_acq-01_run-01_feedbacklog.tsv")
        other_run = _touch(fb / "sub-01_ses-01_task-feedback_acq-01_run-02_feedbacklog.tsv")

        discard_attempt(tmp_path, subject="sub-01", session="ses-01", task="feedback", acq=1, run=1, kind="feedback")

        assert not target.exists()
        assert other_run.exists()


class TestDiscardIntegration:
    def test_localizer_discard_lets_next_attempt_reuse_acq(self, tmp_path):
        """After discard, get_localizer_run_info returns the same acq number."""
        coll = tmp_path / "collection"
        _touch(coll / "task-rest_acq-01_run-01" / "a.dcm")

        # Without discard, the next attempt would be acq-02.
        before = get_localizer_run_info(coll, "task-rest", run=1)
        assert before["acq"] == 2

        discard_attempt(tmp_path, subject="sub-01", session="ses-01", task="rest", acq=1, run=1, kind="localizer")

        after = get_localizer_run_info(coll, "task-rest", run=1)
        assert after["acq"] == 1  # acq slot freed up

    def test_feedback_discard_lets_next_attempt_reuse_acq(self, tmp_path):
        fb = tmp_path / "feedback" / "02_feedback_output"
        _touch(fb / "sub-01_ses-01_task-feedback_acq-01_run-01_feedbacklog.tsv", "header\nrow")

        before = get_slot_run_info(fb, "sub-01", "ses-01", "feedback", step_id="feedback_1")
        assert before["acq"] == 2

        discard_attempt(tmp_path, subject="sub-01", session="ses-01", task="feedback", acq=1, run=1, kind="feedback")

        after = get_slot_run_info(fb, "sub-01", "ses-01", "feedback", step_id="feedback_1")
        assert after["acq"] == 1


class TestDiscardFailures:
    def test_partial_failure_raises_aggregated_oserror(self, tmp_path, monkeypatch):
        fb = tmp_path / "feedback" / "02_feedback_output"
        log = _touch(fb / "sub-01_ses-01_task-feedback_acq-01_run-01_feedbacklog.tsv")
        params = _touch(fb / "sub-01_ses-01_task-feedback_acq-01_run-01_parameters.json")

        real_unlink = Path.unlink

        def flaky_unlink(self, *args, **kwargs):
            if self.name.endswith("_parameters.json"):
                raise PermissionError("locked by another process")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        with pytest.raises(OSError, match="Failed to discard 1 path"):
            discard_attempt(
                tmp_path,
                subject="sub-01",
                session="ses-01",
                task="feedback",
                acq=1,
                run=1,
                kind="feedback",
            )

        # Whatever could be removed was removed; the locked file remains.
        assert not log.exists()
        assert params.exists()

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown kind"):
            discard_attempt(
                tmp_path,
                subject="sub-01",
                session="ses-01",
                task="feedback",
                acq=1,
                run=1,
                kind="bogus",  # type: ignore[arg-type]
            )


def test_discard_uses_shutil_rmtree_for_directories(tmp_path, monkeypatch):
    """Sanity: directory removal goes through shutil.rmtree (not unlink)."""
    coll = tmp_path / "collection" / "task-rest_acq-01_run-01"
    _touch(coll / "a.dcm")

    seen = []
    real_rmtree = shutil.rmtree

    def spy(path, *args, **kwargs):
        seen.append(Path(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("pineuro.session.shutil.rmtree", spy)

    discard_attempt(tmp_path, subject="sub-01", session="ses-01", task="rest", acq=1, run=1, kind="localizer")

    assert coll in seen
