"""Tests for pineuro.gui.components.csv_poller — FeedbackLogPoller and FeedbackMonitor."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pineuro.gui.components.csv_poller import FeedbackLogPoller, FeedbackMonitor

# =============================================================================
# Helpers
# =============================================================================


def make_poller(log_path, callback=None):
    """Return a FeedbackLogPoller with headers pre-populated for unit tests."""
    if callback is None:
        callback = MagicMock()
    poller = FeedbackLogPoller(log_path=log_path, on_new_data=callback, poll_interval=0.05)
    return poller


def write_tsv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(headers)]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# _parse_row
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
            data = poller._parse_row(row)
            assert data["is_baseline"] is True, f"Expected True for {true_val!r}"

    def test_boolean_false_variants(self, poller):
        for false_val in ("False", "false", "0", "no"):
            row = ["1", "0.0", "0.0", "0.0", false_val, "0.0"]
            data = poller._parse_row(row)
            assert data["is_baseline"] is False, f"Expected False for {false_val!r}"

    def test_fd_parsed_as_float(self, poller):
        data = poller._parse_row(["1", "0.0", "0.0", "0.0", "False", "0.35"])
        assert data["fd"] == pytest.approx(0.35)

    def test_length_mismatch_returns_none(self, poller):
        # Fewer values than headers
        data = poller._parse_row(["1", "2.5"])
        assert data is None

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
        data = poller._parse_row(["1", "0.75", "0.02", "False"])
        assert data["mask_ContA"] == pytest.approx(0.75)

    def test_motion_prefix_parsed_as_float(self, poller):
        data = poller._parse_row(["1", "0.75", "0.02", "False"])
        assert data["motion_tx"] == pytest.approx(0.02)

    def test_is_outlier_boolean(self, poller):
        data = poller._parse_row(["1", "0.75", "0.02", "True"])
        assert data["is_outlier"] is True


# =============================================================================
# _check_for_new_data (incremental read)
# =============================================================================


class TestIncrementalRead:
    def test_first_call_reads_header_only(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [["1", "0.5"]])

        callback = MagicMock()
        poller = make_poller(log_path, callback)

        # First call: reads header, does not emit rows
        poller._check_for_new_data()
        callback.assert_not_called()
        assert poller._headers == ["volume", "ContA"]

    def test_second_call_emits_data_row(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [["1", "0.5"]])

        callback = MagicMock()
        poller = make_poller(log_path, callback)

        poller._check_for_new_data()  # reads header
        poller._check_for_new_data()  # reads row 1
        callback.assert_called_once()
        data = callback.call_args[0][0]
        assert data["volume"] == 1
        assert data["ContA"] == pytest.approx(0.5)

    def test_incremental_only_emits_new_rows(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [["1", "0.5"], ["2", "0.6"]])

        callback = MagicMock()
        poller = make_poller(log_path, callback)

        poller._check_for_new_data()  # header
        poller._check_for_new_data()  # rows 1 & 2
        assert callback.call_count == 2

        # Append row 3
        with open(log_path, "a") as f:
            f.write("3\t0.7\n")

        poller._check_for_new_data()  # only row 3
        assert callback.call_count == 3
        data = callback.call_args[0][0]
        assert data["volume"] == 3

    def test_missing_file_does_not_raise(self, tmp_path):
        poller = make_poller(tmp_path / "missing.tsv")
        poller._check_for_new_data()  # should silently return

    def test_row_count_tracked(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [["1", "0.1"], ["2", "0.2"], ["3", "0.3"]])

        poller = make_poller(log_path)
        poller._check_for_new_data()  # header
        poller._check_for_new_data()  # 3 rows

        with poller._lock:
            assert poller._last_row_count == 3


# =============================================================================
# Thread start / stop
# =============================================================================


class TestPollerThreading:
    def test_start_creates_daemon_thread(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [])
        poller = make_poller(log_path)
        poller.start()
        assert poller._thread is not None
        assert poller._thread.is_alive()
        assert poller._thread.daemon
        poller.stop()

    def test_stop_terminates_thread(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [])
        poller = make_poller(log_path)
        poller.start()
        poller.stop()
        assert poller._thread is None

    def test_duplicate_start_does_not_create_second_thread(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [])
        poller = make_poller(log_path)
        poller.start()
        first_thread = poller._thread
        poller.start()  # second call should be a no-op
        assert poller._thread is first_thread
        poller.stop()

    def test_poller_emits_data_while_running(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [["1", "0.5"]])

        received = []

        def on_data(row):
            received.append(row)

        poller = FeedbackLogPoller(log_path=log_path, on_new_data=on_data, poll_interval=0.02)
        poller.start()
        time.sleep(0.2)  # give polling thread time to run
        poller.stop()

        assert len(received) >= 1
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

    def test_progress_after_data(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA", "fd"], [["50", "0.5", "0.1"]])

        updates = []

        def on_update(volume, mask_acts, motion, fd, dvars, tsnr):
            updates.append(volume)

        monitor = FeedbackMonitor(
            log_path=log_path,
            mask_names=["ContA"],
            n_volumes=100,
            on_update=on_update,
        )
        monitor.start()
        # FeedbackMonitor's poller uses 0.5s interval: first cycle reads header,
        # second cycle reads data row — need at least ~1.2s to receive one row.
        time.sleep(1.5)
        monitor.stop()

        assert len(updates) >= 1
        assert monitor.volumes_processed == 50
        assert monitor.progress == pytest.approx(0.5)

    def test_progress_clamped_to_one(self, tmp_path):
        log_path = tmp_path / "feedbacklog.tsv"
        write_tsv(log_path, ["volume", "ContA"], [["200", "0.5"]])

        monitor = FeedbackMonitor(
            log_path=log_path,
            mask_names=["ContA"],
            n_volumes=100,
            on_update=MagicMock(),
        )
        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        assert monitor.progress <= 1.0
