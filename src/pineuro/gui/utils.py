"""Utility functions for the pineuro GUI.

This module provides shared utilities used across GUI components:
validation helpers for form input, threading utilities for background
operations, and common UI helper functions.

Validation Functions
--------------------
validate_positive_number : Validate positive float values
validate_positive_int : Validate positive integer values
validate_path : Validate file/directory paths
validate_yeo_targets : Validate Yeo network target selection

Constants
---------
YEO_7_LABELS : list of str
    Network labels for the Yeo 7-network parcellation.
YEO_17_LABELS : list of str
    Network labels for the Yeo 17-network parcellation.

Formatting Functions
--------------------
format_timestamp : Format ISO timestamp for display
truncate_text : Truncate long text with ellipsis

Exceptions
----------
ValidationError : Raised when form validation fails

Threading
---------
BackgroundRunner : Run blocking operations in a thread
    Prevents GUI freezing during long-running operations.

Notes
-----
All validation functions raise ValidationError with descriptive
messages suitable for displaying to users in the GUI.

Examples
--------
>>> from pineuro.gui.utils import validate_positive_number, ValidationError
>>> try:
...     tr = validate_positive_number("1.2", "TR")
...     print(f"Valid TR: {tr}")
... except ValidationError as e:
...     print(f"Error: {e}")
Valid TR: 1.2
"""

import logging
import threading
from pathlib import Path
from typing import Any

import flet as ft

logger = logging.getLogger(__name__)


# Yeo network labels
YEO_7_LABELS = ["Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

YEO_17_LABELS = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicA",
    "LimbicB",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
]


class ValidationError(Exception):
    """Raised when a configuration parameter fails validation."""

    pass


def validate_positive_number(value: Any, name: str) -> float:
    """Validate that a value is a positive number.

    Args:
        value: Value to validate
        name: Parameter name (for error messages)

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got: {value}") from None

    if num <= 0:
        raise ValidationError(f"{name} must be positive, got: {num}")

    return num


def validate_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Parameter name (for error messages)

    Returns:
        Validated int value

    Raises:
        ValidationError: If validation fails
    """
    try:
        num = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be an integer, got: {value}") from None

    if num <= 0:
        raise ValidationError(f"{name} must be positive, got: {num}")

    return num


def validate_path(value: Any, name: str, must_exist: bool = False) -> Path:
    """Validate that a value is a valid path.

    Args:
        value: Value to validate
        name: Parameter name (for error messages)
        must_exist: If True, path must exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if not value:
        raise ValidationError(f"{name} cannot be empty")

    try:
        path = Path(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} is not a valid path: {e}") from e

    if must_exist and not path.exists():
        raise ValidationError(f"{name} does not exist: {path}")

    return path


def validate_yeo_networks(n_yeo: int, selected_networks: list) -> tuple[int, list]:
    """Validate Yeo network configuration.

    Args:
        n_yeo: Number of Yeo networks (7 or 17)
        selected_networks: List of selected network names

    Returns:
        Tuple of (validated n_yeo, validated network list)

    Raises:
        ValidationError: If validation fails
    """
    if n_yeo not in [7, 17]:
        raise ValidationError(f"n_yeo must be 7 or 17, got: {n_yeo}")

    if not selected_networks:
        raise ValidationError("At least one network must be selected")

    if len(selected_networks) < 2:
        raise ValidationError(f"At least 2 networks must be selected for feedback, got: {len(selected_networks)}")

    return n_yeo, selected_networks


def validate_participant_id(participant_id: str) -> tuple[bool, str | None]:
    """Validate a BIDS-compatible participant ID.

    Accepts IDs containing only letters, digits, and hyphens, starting with
    an alphanumeric character (e.g. ``'sub-01'``, ``'P001'``).

    Parameters
    ----------
    participant_id : str
        The participant ID to validate.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` if valid; ``(False, error_message)`` otherwise.
    """
    import re

    if not participant_id:
        return False, "Participant ID is required"
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$", participant_id):
        return False, "Use only letters, digits, and hyphens (e.g. sub-01)"
    return True, None


def validate_session_config(config_dict: dict) -> tuple[bool, str | None]:
    """Validate a complete session configuration.

    Args:
        config_dict: Dictionary representation of SessionConfig

    Returns:
        Tuple of (is_valid, error_message)
        If is_valid is True, error_message is None
        If is_valid is False, error_message contains the validation error
    """
    try:
        # Validate at least one localizer
        localizers = config_dict.get("localizers", [])
        if not localizers or not isinstance(localizers, list):
            raise ValidationError("at least one localizers entry is required")
        loc0 = localizers[0]
        validate_positive_number(loc0.get("tr", 0), "localizers[0].tr")
        validate_positive_int(loc0.get("n_volumes", 0), "localizers[0].n_volumes")
        validate_positive_int(loc0.get("n_runs", 0), "localizers[0].n_runs")
        if loc0.get("n_runs", 0) < 2:
            raise ValidationError("localizers[0].n_runs must be at least 2")

        # Validate extraction sub-block for ICA (rest) localizers
        if loc0.get("type", "rest") == "rest":
            extraction = loc0.get("extraction", {})
            n_yeo = extraction.get("n_yeo", 0)
            networks = loc0.get("networks", [])
            validate_yeo_networks(n_yeo, networks)
            validate_positive_number(extraction.get("smooth_fwhm", 0), "extraction.smooth_fwhm")
            validate_positive_number(extraction.get("highpass", 0), "extraction.highpass")

        # Validate at least one feedback run
        feedback_runs = config_dict.get("feedback_runs", [])
        if not feedback_runs:
            raise ValidationError("at least one feedback_runs entry is required")
        for i, frc in enumerate(feedback_runs):
            validate_positive_int(frc.get("n_baseline", 0), f"feedback_runs[{i}].n_baseline")
            validate_positive_int(frc.get("n_volumes", 0), f"feedback_runs[{i}].n_volumes")

        # Validate task parameters via metadata (if task type is registered)
        task_type = feedback_runs[0].get("task", "ball")
        try:
            from ..tasks.task_registry import get_task_metadata

            metadata = get_task_metadata(task_type)
            metadata.validate_config(feedback_runs[0])
        except (ValueError, ImportError):
            pass  # Task not registered, skip task-specific validation

        # All validations passed
        return True, None

    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        return False, f"Validation error: {e}"


class ThreadSafeCallback:
    """Wrapper for thread-safe GUI callbacks in Flet.

    Flet requires page.update() calls to happen from the main thread.
    This class helps manage callbacks from background threads.
    """

    def __init__(self, page):
        """Initialize with a Flet page object.

        Args:
            page: Flet Page object
        """
        self.page = page
        self._lock = threading.Lock()

    def update(self, callback):
        """Execute a callback and update the page thread-safely.

        Args:
            callback: Function to execute (should modify page controls)
        """
        with self._lock:
            callback()
            self.page.update()


def format_timestamp(iso_timestamp: str) -> str:
    """Format an ISO timestamp for display.

    Args:
        iso_timestamp: ISO format timestamp string

    Returns:
        Formatted date string (e.g., "2026-01-16 14:30")
    """
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return iso_timestamp


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis if too long.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with "..." if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def build_placeholder(text: str, height: float | None = None) -> ft.Container:
    """Build a centered placeholder container with greyed-out text.

    Used as a fallback display when a chart or component is unavailable.

    Args:
        text: Message to display in the placeholder.
        height: Optional fixed height for the container.

    Returns:
        A centred ``ft.Container`` with the placeholder text.
    """
    return ft.Container(
        content=ft.Text(text, size=16, color=ft.Colors.GREY_500),
        height=height,
        alignment=ft.Alignment(0, 0),
    )
