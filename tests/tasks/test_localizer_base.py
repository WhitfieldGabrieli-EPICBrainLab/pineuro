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

    return _StubLocalizer(output_dir=tmp_path, participant_id="sub-01", session_id="ses-01", run=2, acq=3)


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
        # Prefix uses task_id=stub_localizer, run=2, acq=3
        assert path.name == "sub-01_ses-01_task-stub_localizer_acq-03_run-02_events.tsv"
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
