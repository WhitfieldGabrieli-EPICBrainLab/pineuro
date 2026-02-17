"""CSV/TSV file poller for real-time data monitoring.

This module provides a polling mechanism to watch feedback log files
and emit new data as it's written during neurofeedback sessions.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
import csv

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
        on_new_data: Callable[[Dict[str, Any]], None],
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
        self._thread: Optional[threading.Thread] = None
        self._last_row_count = 0
        self._headers: List[str] = []

    def start(self):
        """Start polling in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Poller already running")
            return

        self._stop_event.clear()
        self._last_row_count = 0
        self._headers = []

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
            except Exception as e:
                logger.error(f"Error polling log file: {e}")

            # Wait for next poll interval
            self._stop_event.wait(self.poll_interval)

    def _check_for_new_data(self):
        """Check for new rows in the log file."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter='\t')

                # Read all rows
                rows = list(reader)

                if not rows:
                    return

                # First row is header
                if not self._headers:
                    self._headers = rows[0]
                    logger.debug(f"Log headers: {self._headers}")

                # Check for new data rows
                current_row_count = len(rows) - 1  # Exclude header

                if current_row_count > self._last_row_count:
                    # Process new rows
                    for i in range(self._last_row_count + 1, len(rows)):
                        row = rows[i]
                        data = self._parse_row(row)
                        if data:
                            self.on_new_data(data)

                    self._last_row_count = current_row_count

        except Exception as e:
            logger.error(f"Error reading log file: {e}")

    def _parse_row(self, row: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a TSV row into a data dictionary.

        Args:
            row: List of string values

        Returns:
            Dict with parsed values, or None if parsing fails
        """
        if len(row) != len(self._headers):
            logger.warning(f"Row length mismatch: {len(row)} vs {len(self._headers)}")
            return None

        data = {}
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
        mask_names: List[str],
        n_volumes: int,
        on_update: Callable[[int, Dict[str, float], Dict[str, float], Optional[float], Optional[float], Optional[float]], None],
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

        self._poller: Optional[FeedbackLogPoller] = None
        self._volumes_processed = 0

    def start(self):
        """Start monitoring."""
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
        return self._volumes_processed

    @property
    def progress(self) -> float:
        """Progress as a fraction (0.0 to 1.0)."""
        if self.n_volumes <= 0:
            return 0.0
        return min(1.0, self._volumes_processed / self.n_volumes)

    def _on_new_row(self, data: Dict[str, Any]):
        """Handle new row from poller.

        Args:
            data: Parsed row data
        """
        volume = data.get("volume", self._volumes_processed + 1)
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
        fd: Optional[float] = data.get("fd")
        dvars: Optional[float] = data.get("dvars")
        tsnr: Optional[float] = data.get("tsnr")

        # Call update callback
        self.on_update(volume, mask_activations, motion_params, fd, dvars, tsnr)


def get_hits_from_psychopylog(log_path: Path) -> Dict[str, int]:
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
        with open(log_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:  # Need header + at least one data row
            return {}
        header = lines[0].strip().split('\t')
        last_row = lines[-1].strip().split('\t')
        hits = {}
        for i, col in enumerate(header):
            if col.endswith('_hits') and i < len(last_row):
                mask_name = col.replace('_hits', '')
                hits[mask_name] = int(last_row[i])
        return hits
    except Exception:
        return {}
