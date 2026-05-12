"""Tests for pineuro.gui.components.feedback_summary — compute_feedback_summary.

Only the pure-data summary function is unit-tested; the Flet dialog (FeedbackSummaryDialog)
is excluded because its render-only branches would need extensive Flet mocking.
"""

from pathlib import Path

import pytest

from pineuro.tasks.task_registry import LiveMetric


def _write_feedbacklog(path: Path, *, n_volumes, fd_values=None, dvars_values=None, tsnr_values=None):
    """Write a minimal feedbacklog.tsv compatible with compute_feedback_summary."""
    fd = fd_values if fd_values is not None else [0.1] * n_volumes
    dvars = dvars_values if dvars_values is not None else [1.0] * n_volumes
    tsnr = tsnr_values if tsnr_values is not None else [40.0] * n_volumes
    lines = ["volume\telapsed\tfd\tdvars\ttsnr"]
    for i in range(n_volumes):
        lines.append(f"{i + 1}\t{i * 2.0}\t{fd[i]}\t{dvars[i]}\t{tsnr[i]}")
    path.write_text("\n".join(lines))


def _write_psychopy_log(path: Path, *, network_to_max_hits):
    """Write a psychopylog.tsv with cumulative *_hits columns."""
    networks = list(network_to_max_hits.keys())
    header = ["volume"] + [f"{n}_hits" for n in networks]
    lines = ["\t".join(header)]
    # Cumulative: max(0..n) → final equals network_to_max_hits[n]
    n_rows = max(network_to_max_hits.values()) + 1 if network_to_max_hits else 1
    for i in range(n_rows):
        row = [str(i + 1)]
        for n in networks:
            row.append(str(min(i, network_to_max_hits[n])))
        lines.append("\t".join(row))
    path.write_text("\n".join(lines))


_BALL_HITS_METRIC = LiveMetric(
    key="hits",
    label="Hits",
    column_pattern="*_hits",
    name_strip_suffix="_hits",
    remap_blind_names=True,
)


class TestCompleteFlag:
    def test_completed_when_volumes_meet_expected(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=100)
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=10, n_expected_volumes=100
        )
        assert summary.completed is True
        assert summary.total_volumes == 100

    def test_not_completed_when_short(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=50)
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=10, n_expected_volumes=100
        )
        assert summary.completed is False


class TestVolumeCounts:
    def test_baseline_clamped_to_total(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=5)  # fewer than baseline=10
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=10, n_expected_volumes=100
        )
        assert summary.baseline_volumes == 5  # clamped to total
        assert summary.feedback_volumes == 0  # max(0, 5 - 10)

    def test_feedback_volumes_after_baseline(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=50)
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=10, n_expected_volumes=100
        )
        assert summary.baseline_volumes == 10
        assert summary.feedback_volumes == 40


class TestFdStatistics:
    def test_threshold_count_and_percent(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        # 4 of 10 volumes above the 0.5 threshold
        _write_feedbacklog(fb, n_volumes=10, fd_values=[0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4])
        summary = compute_feedback_summary(
            feedbacklog_path=fb,
            psychopy_log_path=None,
            n_baseline=0,
            n_expected_volumes=10,
            fd_threshold=0.5,
        )
        assert summary.fd_above_threshold == 4
        assert summary.fd_pct_above_threshold == pytest.approx(40.0)
        assert summary.fd_threshold == 0.5

    def test_mean_and_max(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=5, fd_values=[0.0, 0.2, 0.4, 0.6, 0.8])
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=0, n_expected_volumes=5
        )
        assert summary.fd_mean == pytest.approx(0.4)
        assert summary.fd_max == pytest.approx(0.8)


class TestLiveMetricSnapshots:
    def test_no_psychopy_log(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=10)
        summary = compute_feedback_summary(
            feedbacklog_path=fb,
            psychopy_log_path=None,
            n_baseline=0,
            n_expected_volumes=10,
            live_metrics=[_BALL_HITS_METRIC],
        )
        assert summary.live_metric_snapshots == {}

    def test_missing_psychopy_log_silent(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        _write_feedbacklog(fb, n_volumes=10)
        summary = compute_feedback_summary(
            feedbacklog_path=fb,
            psychopy_log_path=tmp_path / "doesnt_exist.tsv",
            n_baseline=0,
            n_expected_volumes=10,
            live_metrics=[_BALL_HITS_METRIC],
        )
        assert summary.live_metric_snapshots == {}

    def test_no_metrics_declared_returns_empty(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        ppy = tmp_path / "psychopylog.tsv"
        _write_feedbacklog(fb, n_volumes=10)
        _write_psychopy_log(ppy, network_to_max_hits={"DefaultA": 3, "ContA": 5})
        summary = compute_feedback_summary(
            feedbacklog_path=fb,
            psychopy_log_path=ppy,
            n_baseline=0,
            n_expected_volumes=10,
        )
        # No live_metrics passed -> nothing surfaced.
        assert summary.live_metric_snapshots == {}

    def test_glob_pattern_extracts_per_network_hits(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        ppy = tmp_path / "psychopylog.tsv"
        _write_feedbacklog(fb, n_volumes=10)
        _write_psychopy_log(ppy, network_to_max_hits={"DefaultA": 3, "ContA": 5})

        summary = compute_feedback_summary(
            feedbacklog_path=fb,
            psychopy_log_path=ppy,
            n_baseline=0,
            n_expected_volumes=10,
            live_metrics=[_BALL_HITS_METRIC],
        )
        assert summary.live_metric_snapshots == {"hits": {"DefaultA": "3", "ContA": "5"}}


class TestTsnr:
    def test_uses_final_valid_value(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        # last value is the most stable estimate
        _write_feedbacklog(fb, n_volumes=5, tsnr_values=[10.0, 20.0, 30.0, 40.0, 50.0])
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=0, n_expected_volumes=5
        )
        assert summary.tsnr_final == 50.0

    def test_zero_when_no_valid_values(self, tmp_path):
        from pineuro.gui.components.feedback_summary import compute_feedback_summary

        fb = tmp_path / "feedbacklog.tsv"
        # All NaN: written as empty strings → pd.to_numeric → NaN
        _write_feedbacklog(fb, n_volumes=3, tsnr_values=["", "", ""])
        summary = compute_feedback_summary(
            feedbacklog_path=fb, psychopy_log_path=None, n_baseline=0, n_expected_volumes=3
        )
        assert summary.tsnr_final == 0.0
