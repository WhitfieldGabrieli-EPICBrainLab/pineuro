"""Tests for pineuro.gui.components.csv_poller.

Covers the whole module in one place (it was previously split between here and
tests/gui/test_csv_poller.py):
- ``read_live_metrics`` — glob/literal column extraction from a psychopy log.
- ``FeedbackLogPoller`` — row parsing + incremental tail-read + polling thread.
- ``FeedbackMonitor`` — progress tracking over a feedbacklog.

Thread/poller tests use the ``assert_eventually`` fixture (poll until the
condition holds) instead of a fixed ``time.sleep``, which is flaky on loaded CI.
"""

from unittest.mock import MagicMock

import pytest

from pineuro.gui.components.csv_poller import FeedbackLogPoller, FeedbackMonitor, read_live_metrics
from pineuro.tasks.task_registry import LiveMetric

# =============================================================================
# Helpers
# =============================================================================


def make_poller(log_path, callback=None):
    """Return a FeedbackLogPoller wired to a callback (default MagicMock)."""
    if callback is None:
        callback = MagicMock()
    return FeedbackLogPoller(log_path=log_path, on_new_data=callback, poll_interval=0.05)


# =============================================================================
# read_live_metrics
# =============================================================================


class TestReadLiveMetrics:
    def test_glob_pattern_matches_multiple_columns(self, tmp_path, write_tsv):
        log = write_tsv(
            tmp_path / "psychopylog.tsv",
            ["frame", "time", "volume", "ContA_hits", "DefaultA_hits"],
            [["1", "0.0", "1", "0", "0"], ["2", "1.0", "2", "3", "1"]],
        )
        metrics = [LiveMetric(key="hits", label="Hits", column_pattern="*_hits", name_strip_suffix="_hits")]
        assert read_live_metrics(log, metrics) == {"hits": {"ContA": "3", "DefaultA": "1"}}

    def test_literal_column_with_value_format(self, tmp_path, write_tsv):
        log = write_tsv(
            tmp_path / "psychopylog.tsv",
            ["frame", "time", "volume", "score"],
            [["1", "0.0", "1", "0.0"], ["2", "1.0", "2", "12.34"]],
        )
        metrics = [LiveMetric(key="score", label="Score", column_pattern="score", value_format="{value:.1f}")]
        assert read_live_metrics(log, metrics) == {"score": {"score": "12.3"}}

    def test_missing_file_returns_empty(self, tmp_path):
        metrics = [LiveMetric(key="hits", label="Hits", column_pattern="*_hits")]
        assert read_live_metrics(tmp_path / "missing.tsv", metrics) == {}

    def test_empty_metrics_returns_empty(self, tmp_path, write_tsv):
        log = write_tsv(tmp_path / "psychopylog.tsv", ["frame", "score"], [["1", "2.0"]])
        assert read_live_metrics(log, []) == {}

    def test_header_only_returns_empty(self, tmp_path, write_tsv):
        log = write_tsv(tmp_path / "psychopylog.tsv", ["frame", "score"], [])
        metrics = [LiveMetric(key="score", label="Score", column_pattern="score")]
        assert read_live_metrics(log, metrics) == {}

    def test_no_matching_columns_omits_metric(self, tmp_path, write_tsv):
        log = write_tsv(tmp_path / "psychopylog.tsv", ["frame", "score"], [["1", "0.0"], ["2", "5.5"]])
        metrics = [
            LiveMetric(key="hits", label="Hits", column_pattern="*_hits"),
            LiveMetric(key="score", label="Score", column_pattern="score"),
        ]
        out = read_live_metrics(log, metrics)
        assert "hits" not in out
        assert out["score"] == {"score": "5.5"}

    def test_multiple_metrics_independent(self, tmp_path, write_tsv):
        log = write_tsv(
            tmp_path / "psychopylog.tsv",
            ["frame", "triggered_prompts", "timeout_prompts"],
            [["1", "0", "0"], ["2", "2", "1"]],
        )
        metrics = [
            LiveMetric(key="triggered", label="Triggered", column_pattern="triggered_prompts"),
            LiveMetric(key="timeout", label="Timeout", column_pattern="timeout_prompts"),
        ]
        assert read_live_metrics(log, metrics) == {
            "triggered": {"triggered_prompts": "2"},
            "timeout": {"timeout_prompts": "1"},
        }


# =============================================================================
# FeedbackLogPoller._parse_row
# =============================================================================


class TestParseRow:
    @pytest.fixture
    def poller(self, tmp_path):
        p = make_poller(tmp_path / "log.tsv")
        p._headers = ["volume", "time", "ContA", "DefaultA", "is_baseline", "fd"]
        return p

    def test_volume_parsed_as_int(self, poller):
        data = poller._parse_row(["5", "10.0", "0.5", "-0.3", "False", "0.2"])
        assert data["volume"] == 5
        assert isinstance(data["volume"], int)

    def test_time_parsed_as_float(self, poller):
        data = poller._parse_row(["1", "2.5", "0.8", "-0.1", "True", "0.3"])
        assert data["time"] == pytest.approx(2.5)

    def test_activation_columns_parsed_as_float(self, poller):
        data = poller._parse_row(["1", "2.5", "0.8", "-0.1", "True", "0.3"])
        assert data["ContA"] == pytest.approx(0.8)
        assert data["DefaultA"] == pytest.approx(-0.1)

    def test_boolean_true_variants(self, poller):
        for true_val in ("True", "true", "1", "yes"):
            row = ["1", "0.0", "0.0", "0.0", true_val, "0.0"]
            assert poller._parse_row(row)["is_baseline"] is True, f"Expected True for {true_val!r}"

    def test_boolean_false_variants(self, poller):
        for false_val in ("False", "false", "0", "no"):
            row = ["1", "0.0", "0.0", "0.0", false_val, "0.0"]
            assert poller._parse_row(row)["is_baseline"] is False, f"Expected False for {false_val!r}"

    def test_fd_parsed_as_float(self, poller):
        data = poller._parse_row(["1", "0.0", "0.0", "0.0", "False", "0.35"])
        assert data["fd"] == pytest.approx(0.35)

    def test_length_mismatch_returns_none(self, poller):
        assert poller._parse_row(["1", "2.5"]) is None

    def test_all_fields_present(self, poller):
        data = poller._parse_row(["3", "6.0", "1.2", "-0.5", "False", "0.1"])
        assert set(data.keys()) == {"volume", "time", "ContA", "DefaultA", "is_baseline", "fd"}


class TestParseRowMaskPrefix:
    """_parse_row handles mask_ and motion_ column prefixes."""

    @pytest.fixture
    def poller(self, tmp_path):
        p = make_poller(tmp_path / "log.tsv")
        p._headers = ["volume", "mask_ContA", "motion_tx", "is_outlier"]
        return p

    def test_mask_prefix_parsed_as_float(self, poller):
        assert poller._parse_row(["1", "0.75", "0.02", "False"])["mask_ContA"] == pytest.approx(0.75)

    def test_motion_prefix_parsed_as_float(self, poller):
        assert poller._parse_row(["1", "0.75", "0.02", "False"])["motion_tx"] == pytest.approx(0.02)

    def test_is_outlier_boolean(self, poller):
        assert poller._parse_row(["1", "0.75", "0.02", "True"])["is_outlier"] is True


# =============================================================================
# FeedbackLogPoller._check_for_new_data (incremental read)
# =============================================================================


class TestIncrementalRead:
    def test_first_call_reads_header_only(self, tmp_path, write_tsv):
        log_path = write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], [["1", "0.5"]])
        callback = MagicMock()
        poller = make_poller(log_path, callback)

        poller._check_for_new_data()  # reads header, emits no rows
        callback.assert_not_called()
        assert poller._headers == ["volume", "ContA"]

    def test_second_call_emits_data_row(self, tmp_path, write_tsv):
        log_path = write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], [["1", "0.5"]])
        callback = MagicMock()
        poller = make_poller(log_path, callback)

        poller._check_for_new_data()  # header
        poller._check_for_new_data()  # row 1
        callback.assert_called_once()
        data = callback.call_args[0][0]
        assert data["volume"] == 1
        assert data["ContA"] == pytest.approx(0.5)

    def test_incremental_only_emits_new_rows(self, tmp_path, write_tsv):
        log_path = write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], [["1", "0.5"], ["2", "0.6"]])
        callback = MagicMock()
        poller = make_poller(log_path, callback)

        poller._check_for_new_data()  # header
        poller._check_for_new_data()  # rows 1 & 2
        assert callback.call_count == 2

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("3\t0.7\n")

        poller._check_for_new_data()  # only row 3
        assert callback.call_count == 3
        assert callback.call_args[0][0]["volume"] == 3

    def test_missing_file_does_not_raise(self, tmp_path):
        make_poller(tmp_path / "missing.tsv")._check_for_new_data()  # should silently return

    def test_row_count_tracked(self, tmp_path, write_tsv):
        log_path = write_tsv(
            tmp_path / "feedbacklog.tsv", ["volume", "ContA"], [["1", "0.1"], ["2", "0.2"], ["3", "0.3"]]
        )
        poller = make_poller(log_path)
        poller._check_for_new_data()  # header
        poller._check_for_new_data()  # 3 rows
        with poller._lock:
            assert poller._last_row_count == 3


# =============================================================================
# FeedbackLogPoller thread start / stop
# =============================================================================


class TestPollerThreading:
    def test_start_creates_daemon_thread(self, tmp_path, write_tsv):
        poller = make_poller(write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], []))
        poller.start()
        assert poller._thread is not None
        assert poller._thread.is_alive()
        assert poller._thread.daemon
        poller.stop()

    def test_stop_terminates_thread(self, tmp_path, write_tsv):
        poller = make_poller(write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], []))
        poller.start()
        poller.stop()
        assert poller._thread is None

    def test_duplicate_start_does_not_create_second_thread(self, tmp_path, write_tsv):
        poller = make_poller(write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], []))
        poller.start()
        first_thread = poller._thread
        poller.start()  # second call should be a no-op
        assert poller._thread is first_thread
        poller.stop()

    def test_poller_emits_data_while_running(self, tmp_path, write_tsv, assert_eventually):
        log_path = write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], [["1", "0.5"]])
        received = []
        poller = FeedbackLogPoller(log_path=log_path, on_new_data=received.append, poll_interval=0.02)
        poller.start()
        try:
            assert_eventually(lambda: len(received) >= 1, timeout=3.0)
        finally:
            poller.stop()
        assert received[0]["volume"] == 1


# =============================================================================
# FeedbackMonitor
# =============================================================================


class TestFeedbackMonitor:
    def test_progress_starts_at_zero(self, tmp_path):
        monitor = FeedbackMonitor(
            log_path=tmp_path / "log.tsv",
            mask_names=["ContA"],
            n_volumes=100,
            on_update=MagicMock(),
        )
        assert monitor.progress == pytest.approx(0.0)
        assert monitor.volumes_processed == 0

    def test_progress_after_data(self, tmp_path, write_tsv, assert_eventually):
        log_path = write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA", "fd"], [["50", "0.5", "0.1"]])
        updates = []

        def on_update(volume, mask_acts, motion, fd, dvars, tsnr):
            updates.append(volume)

        monitor = FeedbackMonitor(log_path=log_path, mask_names=["ContA"], n_volumes=100, on_update=on_update)
        monitor.start()
        try:
            # First poll cycle reads the header, the next reads the data row.
            assert_eventually(lambda: monitor.volumes_processed == 50, timeout=5.0)
        finally:
            monitor.stop()
        assert len(updates) >= 1
        assert monitor.progress == pytest.approx(0.5)

    def test_progress_clamped_to_one(self, tmp_path, write_tsv, assert_eventually):
        log_path = write_tsv(tmp_path / "feedbacklog.tsv", ["volume", "ContA"], [["200", "0.5"]])
        monitor = FeedbackMonitor(log_path=log_path, mask_names=["ContA"], n_volumes=100, on_update=MagicMock())
        monitor.start()
        try:
            assert_eventually(lambda: monitor.volumes_processed > 0, timeout=5.0)
        finally:
            monitor.stop()
        # volume 200 of 100 → progress must clamp to exactly 1.0
        assert monitor.progress == pytest.approx(1.0)
