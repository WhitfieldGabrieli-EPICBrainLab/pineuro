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
        # Validate scanner parameters
        scan = config_dict.get("scan", {})
        validate_positive_number(scan.get("tr", 0), "TR")
        validate_positive_int(scan.get("n_volumes", 0), "n_volumes")
        validate_positive_int(scan.get("n_baseline", 0), "n_baseline")

        # Validate rest runs
        rest_n_volumes = scan.get("rest_n_volumes", [])
        if not isinstance(rest_n_volumes, list) or len(rest_n_volumes) < 2:
            raise ValidationError("rest_n_volumes must be a list with at least 2 values")

        # Validate mask parameters
        mask = config_dict.get("mask", {})
        n_yeo = mask.get("n_yeo", 0)
        yeo_targets = mask.get("yeo_targets", [])
        validate_yeo_networks(n_yeo, yeo_targets)

        validate_positive_number(mask.get("smooth_fwhm", 0), "smooth_fwhm")
        validate_positive_number(mask.get("highpass", 0), "highpass")

        # Validate feedback parameters
        feedback = config_dict.get("feedback", {})
        validate_positive_int(feedback.get("n_baseline", 0), "feedback.n_baseline")
        validate_positive_int(feedback.get("n_volumes", 0), "feedback.n_volumes")

        # Validate task parameters via metadata (if task type is registered)
        task = config_dict.get("task", {})
        task_type = task.get("task_type", "ball")
        try:
            from ..tasks.task_registry import get_task_metadata

            metadata = get_task_metadata(task_type)
            metadata.validate_config(task)
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


def open_mask_viewer_in_browser(
    ref_file: Path,
    mask_files: list,
    output_dir: Path,
    title: str = "Mask Review",
) -> Path:
    """Generate a combined nilearn HTML viewer for mask overlays and open in browser.

    Creates an HTML page with one interactive tri-planar viewer per mask,
    each overlaid on the BOLD reference image.

    Parameters
    ----------
    ref_file : Path
        Path to the BOLD reference NIfTI file (background image).
    mask_files : list of Path
        Paths to thresholded mask NIfTI files to display.
    output_dir : Path
        Directory to save the combined HTML file.
    title : str
        Title for the HTML page.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    from nilearn.plotting import view_img  # type: ignore[import-untyped]

    from pineuro.gui.components.qc_viewer import open_qc_report

    ref_str = str(ref_file)
    names = []
    viewers = []

    for mask_file in mask_files:
        # Derive human-readable name: strip participant prefix and _thr suffix
        stem = mask_file.stem  # e.g. "sub-01_DefaultA_thr"
        name = stem.split("_", 1)[1] if "_" in stem else stem
        if name.endswith("_thr"):
            name = name[:-4]
        names.append(name)

        viewer = view_img(
            stat_map_img=str(mask_file),
            bg_img=ref_str,
            title=name,
            threshold=0.5,
            symmetric_cmap=False,
            cmap="hot",
            colorbar=True,
            dim=-0.5,
            opacity=0.6,
            resampling_interpolation="nearest",
        )
        viewers.append(viewer)

    # Build combined HTML page
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>{title}</title>",
        "<style>",
        "  body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }",
        "  .mask-section { margin-bottom: 30px; background: white; padding: 15px;"
        " border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "  h1 { color: #333; }",
        "  h2 { color: #555; margin-top: 0; }",
        "</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Reference: {ref_file.name} &mdash; {len(mask_files)} mask(s)</p>",
    ]
    for name, viewer in zip(names, viewers, strict=False):
        html_parts.append('<div class="mask-section">')
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(viewer.get_iframe(width=900, height=500))
        html_parts.append("</div>")
    html_parts.append("</body></html>")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mask_viewer.html"
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info(f"Saved mask viewer to {output_path}")

    open_qc_report(output_path)
    return output_path
