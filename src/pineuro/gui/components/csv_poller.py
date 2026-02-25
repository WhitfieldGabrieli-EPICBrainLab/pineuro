"""CSV/TSV file poller for real-time data monitoring.

This module provides a polling mechanism to watch feedback log files
and emit new data as it's written during neurofeedback sessions.
"""

import csv
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackLogPoller:
    """Polls a feedback log TSV file for new data rows.

    Monitors a feedbacklog.tsv file written by NeurofeedbackCoordinator
    and calls a callback whenever new rows are detected.

    The feedbacklog.tsv has columns:
    - volume: Volume number
    - timestamp: ISO timestamp
    - mask_*: Activation values for each mask (e.g., mask_DefaultA, mask_ContA)
    - motion_*: Motion parameters (optional)
    """

    def __init__(
        self,
        log_path: Path,
        on_new_data: Callable[[dict[str, Any]], None],
        poll_interval: float = 0.5,
    ):
        """Initialize the feedback log poller.

        Args:
            log_path: Path to the feedbacklog.tsv file
            on_new_data: Callback called with each new data row as a dict
            poll_interval: Polling interval in seconds (default 0.5s = 2 Hz)
        """
        self.log_path = Path(log_path)
        self.on_new_data = on_new_data
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_row_count = 0
        self._headers: list[str] = []
        self._file_offset = 0

    def start(self):
        """Start polling in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Poller already running")
            return

        self._stop_event.clear()
        with self._lock:
            self._last_row_count = 0
            self._headers = []
            self._file_offset = 0

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="FeedbackLogPoller",
        )
        self._thread.start()
        logger.info(f"Started polling: {self.log_path}")

    def stop(self):
        """Stop polling."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Stopped polling")

    def _poll_loop(self):
        """Main polling loop (runs in background thread)."""
        while not self._stop_event.is_set():
            try:
                self._check_for_new_data()
            except (OSError, csv.Error, ValueError) as e:
                logger.error(f"Error polling log file: {e}")

            # Wait for next poll interval
            self._stop_event.wait(self.poll_interval)

    def _check_for_new_data(self):
        """Check for new rows in the log file (incremental read from last offset)."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, newline="") as f:
                with self._lock:
                    has_headers = bool(self._headers)
                    seek_pos = self._file_offset

                if not has_headers:
                    # First call: read header line only
                    header_line = f.readline()
                    if not header_line:
                        return
                    reader = csv.reader([header_line], delimiter="\t")
                    headers = next(reader)
                    with self._lock:
                        self._headers = headers
                        self._file_offset = f.tell()
                    logger.debug(f"Log headers: {headers}")
                    return

                # Subsequent calls: read only new content from last position
                f.seek(seek_pos)
                new_lines = list(f)
                new_offset = f.tell()

            if not new_lines:
                return

            with self._lock:
                self._file_offset = new_offset
                self._last_row_count += len(new_lines)

            reader = csv.reader(new_lines, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                data = self._parse_row(row)
                if data:
                    self.on_new_data(data)

        except (OSError, csv.Error, ValueError) as e:
            logger.error(f"Error reading log file: {e}")

    def _parse_row(self, row: list[str]) -> dict[str, Any] | None:
        """Parse a TSV row into a data dictionary.

        Args:
            row: List of string values

        Returns:
            Dict with parsed values, or None if parsing fails
        """
        if len(row) != len(self._headers):
            logger.warning(f"Row length mismatch: {len(row)} vs {len(self._headers)}")
            return None

        data: dict[str, Any] = {}
        for i, header in enumerate(self._headers):
            value = row[i]

            # Try to convert to numeric types
            try:
                if header == "volume":
                    data[header] = int(value)
                elif header in ("time", "timestamp") or header.startswith("mask_") or header.startswith("motion_"):
                    data[header] = float(value)
                elif header in ("is_baseline", "is_outlier"):
                    # Handle boolean columns
                    data[header] = value.lower() in ("true", "1", "yes")
                else:
                    # Try to parse as float for numeric columns (e.g., DefaultA, ContA)
                    try:
                        data[header] = float(value)
                    except (ValueError, TypeError):
                        data[header] = value
            except (ValueError, TypeError):
                data[header] = value

        return data


class FeedbackMonitor:
    """High-level monitor combining file polling with data aggregation.

    Provides:
    - Automatic detection of mask columns
    - Data aggregation for plotting
    - Progress tracking
    """

    def __init__(
        self,
        log_path: Path,
        mask_names: list[str],
        n_volumes: int,
        on_update: Callable[[int, dict[str, float], dict[str, float], float | None, float | None, float | None], None],
        poll_interval: float = 0.5,
    ):
        """Initialize the feedback monitor.

        Args:
            log_path: Path to the feedbacklog.tsv file
            mask_names: List of expected mask names
            n_volumes: Expected total number of volumes
            on_update: Callback (volume, mask_activations, motion_params, fd, dvars, tsnr)
            poll_interval: Polling interval in seconds
        """
        self.log_path = Path(log_path)
        self.mask_names = mask_names
        self.n_volumes = n_volumes
        self.on_update = on_update

        self._poller: FeedbackLogPoller | None = None
        self._lock = threading.Lock()
        self._volumes_processed = 0

    def start(self):
        """Start monitoring."""
        with self._lock:
            self._volumes_processed = 0
        self._poller = FeedbackLogPoller(
            log_path=self.log_path,
            on_new_data=self._on_new_row,
        )
        self._poller.start()

    def stop(self):
        """Stop monitoring."""
        if self._poller:
            self._poller.stop()
            self._poller = None

    @property
    def volumes_processed(self) -> int:
        """Number of volumes processed so far."""
        with self._lock:
            return self._volumes_processed

    @property
    def progress(self) -> float:
        """Progress as a fraction (0.0 to 1.0)."""
        if self.n_volumes <= 0:
            return 0.0
        with self._lock:
            return min(1.0, self._volumes_processed / self.n_volumes)

    def _on_new_row(self, data: dict[str, Any]):
        """Handle new row from poller.

        Args:
            data: Parsed row data
        """
        volume = data.get("volume", self.volumes_processed + 1)
        with self._lock:
            self._volumes_processed = volume

        # Extract mask activations
        # New format: columns are just mask names (e.g., ContA, DefaultA, SomMotA)
        # Old format: columns had _delivered suffix (e.g., ContA_delivered)
        mask_activations = {}
        for mask_name in self.mask_names:
            # Try new format first (just mask name)
            if mask_name in data:
                mask_activations[mask_name] = data[mask_name]
            # Fall back to old format (_delivered suffix)
            elif f"{mask_name}_delivered" in data:
                mask_activations[mask_name] = data[f"{mask_name}_delivered"]

        # Extract motion parameters
        motion_params = {}
        for key, value in data.items():
            if key.startswith("motion_") and isinstance(value, (int, float)):
                motion_params[key] = value

        # Extract framewise displacement and QC metrics
        fd: float | None = data.get("fd")
        dvars: float | None = data.get("dvars")
        tsnr: float | None = data.get("tsnr")

        # Call update callback
        self.on_update(volume, mask_activations, motion_params, fd, dvars, tsnr)


def get_hits_from_psychopylog(log_path: Path) -> dict[str, int]:
    """Read hit counts from last row of psychopylog.tsv.

    The psychopylog.tsv is written per-frame by the ball task and includes
    columns ending in _hits (e.g., ContA_hits, DefaultA_hits).

    Args:
        log_path: Path to the psychopylog.tsv file

    Returns:
        Dict mapping mask name to hit count, or empty dict if file not found/empty.
    """
    if not log_path.exists():
        return {}
    try:
        with open(log_path) as f:
            lines = f.readlines()
        if len(lines) < 2:  # Need header + at least one data row
            return {}
        header = lines[0].strip().split("\t")
        last_row = lines[-1].strip().split("\t")
        hits = {}
        for i, col in enumerate(header):
            if col.endswith("_hits") and i < len(last_row):
                mask_name = col.replace("_hits", "")
                hits[mask_name] = int(last_row[i])
        return hits
    except Exception:
        return {}
