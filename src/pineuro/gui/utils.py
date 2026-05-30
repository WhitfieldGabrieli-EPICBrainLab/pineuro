"""Utility functions for the pineuro GUI.

This module provides shared utilities used across GUI components:
validation helpers for form input, threading utilities for background
operations, and common UI helper functions.

Validation Functions
--------------------
validate_positive_number : Validate positive float values
validate_positive_int : Validate positive integer values
validate_path : Validate file/directory paths
validate_yeo_networks : Validate Yeo network selection

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

import copy
import logging
import threading
from pathlib import Path
from typing import Any, NamedTuple, cast

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


def validate_yeo_networks(n_networks: int, selected_networks: list) -> tuple[int, list]:
    """Validate Yeo network configuration.

    Args:
        n_networks: Number of Yeo networks (7 or 17)
        selected_networks: List of selected network names

    Returns:
        Tuple of (validated n_networks, validated network list)

    Raises:
        ValidationError: If validation fails
    """
    if n_networks not in [7, 17]:
        raise ValidationError(f"n_networks must be 7 or 17, got: {n_networks}")

    if not selected_networks:
        raise ValidationError("At least one network must be selected")

    if len(selected_networks) < 2:
        raise ValidationError(f"At least 2 networks must be selected for feedback, got: {len(selected_networks)}")

    return n_networks, selected_networks


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

        # Validate extraction sub-block for ICA (rest) localizers
        if loc0.get("type", "rest") == "rest":
            from pineuro.atlas_registry import ATLAS_SOURCES

            extraction = loc0.get("extraction", {})
            n_networks = ATLAS_SOURCES.get(extraction.get("network_atlas", "yeo_17"), {}).get("n_networks", 0)
            networks = loc0.get("networks", [])
            validate_yeo_networks(n_networks, networks)
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


class FeedbackStep(NamedTuple):
    """One feedback/transfer step in the participant pipeline.

    Attributes
    ----------
    step_id : str
        Globally unique slot key (``"feedback_1"``, ``"transfer_2"``, ...). Used as a
        dict key for step status tracking, so it stays a flat global counter.
    task_id : str
        Registered task id (e.g. ``"ball"``), as it appears in the task registry.
    kind : str
        ``"feedback"`` or ``"transfer"`` — whether real-time feedback is delivered.
    task_entity : str
        BIDS ``task-`` entity for this step's output files: ``task_id`` for feedback
        runs, ``f"{task_id}transfer"`` for transfer runs.
    run_number : int
        Run number for this step's filenames. Restarts at 1 per ``task_entity``, so
        each task's feedback and transfer outputs number independently.
    label : str
        Human-readable label (e.g. ``"Feedback Run 1 (Ball)"``).
    is_last_feedback : bool
        Whether this is the last feedback step of its feedback-run block. Gates
        "last_feedback_only" post-run questions per task. Always False for transfer steps.
    """

    step_id: str
    task_id: str
    kind: str
    task_entity: str
    run_number: int
    label: str
    is_last_feedback: bool = False


def _task_entity(task_id: str, kind: str) -> str:
    """BIDS ``task-`` entity for a feedback step: real task name, ``transfer`` suffix."""
    return task_id if kind == "feedback" else f"{task_id}transfer"


def build_feedback_steps(config: dict[str, Any]) -> list[FeedbackStep]:
    """Expand ``config["feedback_runs"]`` into ordered pipeline steps.

    Single source of truth for both step ordering (pre-transfer, feedback x N,
    post-transfer per feedback-run config) and per-task run numbering. ``step_id``
    stays a flat global counter (used as a status dict key); ``run_number`` restarts
    at 1 for each ``task_entity`` so that, e.g. ``feedback_runs = [ball x2 (with
    transfers), gauge x1]`` produces ``task-balltransfer_run-01``, ``task-ball_run-01``,
    ``task-ball_run-02``, ``task-balltransfer_run-02``, ``task-gauge_run-01``.

    Returns
    -------
    list[FeedbackStep]
        One entry per feedback/transfer step, in execution order.
    """
    from ..tasks.task_registry import get_task_metadata

    feedback_runs = config.get("feedback_runs", [])
    steps: list[FeedbackStep] = []
    fb_counter = 0
    tf_counter = 0
    run_counters: dict[str, int] = {}

    def _append(step_id: str, task_id: str, kind: str, label_suffix: str, is_last_feedback: bool = False) -> None:
        entity = _task_entity(task_id, kind)
        run_counters[entity] = run_counters.get(entity, 0) + 1
        slot = step_id.rsplit("_", 1)[1]
        label = f"{'Transfer' if kind == 'transfer' else 'Feedback'} Run {slot} ({label_suffix})"
        steps.append(FeedbackStep(step_id, task_id, kind, entity, run_counters[entity], label, is_last_feedback))

    for frc in feedback_runs:
        task_id = frc.get("task", "ball")
        try:
            label_suffix = get_task_metadata(task_id).display_name
        except (ValueError, ImportError):
            label_suffix = task_id.replace("_", " ").title()
        use_transfers = frc.get("use_transfer_runs", False)
        max_runs = frc.get("max_runs", 1)

        if use_transfers:
            tf_counter += 1
            _append(f"transfer_{tf_counter}", task_id, "transfer", label_suffix)
        for i in range(max_runs):
            fb_counter += 1
            _append(f"feedback_{fb_counter}", task_id, "feedback", label_suffix, is_last_feedback=(i == max_runs - 1))
        if use_transfers:
            tf_counter += 1
            _append(f"transfer_{tf_counter}", task_id, "transfer", label_suffix)

    return steps


def feedback_step_for(config: dict[str, Any], step_id: str) -> FeedbackStep | None:
    """Return the :class:`FeedbackStep` for ``step_id``, or ``None`` if not found."""
    for step in build_feedback_steps(config):
        if step.step_id == step_id:
            return step
    return None


def localizer_mask_names(loc: dict[str, Any]) -> list[str]:
    """Network/contrast mask names a localizer config (raw dict) produces.

    Mirrors :attr:`pineuro.session.LocalizerConfig.mask_names` for the GUI's raw
    dict configs: networks for rest, ``manual_mask_names`` (or names derived from
    ``manual_mask_paths``) for manual, contrast keys (plus ``_reverse`` variants
    when enabled) for task localizers.
    """
    loc_type = loc.get("type", "rest")
    if loc_type == "rest":
        return list(loc.get("networks", []))
    if loc_type == "manual":
        names = list(loc.get("manual_mask_names", []))
        if names:
            return names
        derived: list[str] = []
        for p in loc.get("manual_mask_paths", []):
            stem = Path(p).name
            for suffix in (".nii.gz", ".nii"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            derived.append(stem)
        return derived
    names = list(loc.get("contrasts", {}).keys())
    if loc.get("include_reverse_contrasts", False):
        names += [f"{k}_reverse" for k in loc.get("contrasts", {})]
    return names


def localizer_is_reused(loc: dict[str, Any]) -> bool:
    """Whether this localizer (raw dict) reuses masks from an earlier session."""
    return bool(loc.get("reuse_from_session"))


def reused_localizer_indices(config: dict[str, Any]) -> list[int]:
    """Indices of localizers whose masks are reused from an earlier session."""
    return [i for i, loc in enumerate(config.get("localizers", [])) if localizer_is_reused(loc)]


def has_fresh_localizer(config: dict[str, Any]) -> bool:
    """Whether any localizer is computed fresh this session (vs every one reused)."""
    return any(not localizer_is_reused(loc) for loc in config.get("localizers", []))


def ses_num(session_id: str) -> int | None:
    """Parse a BIDS session id ``'ses-NN'`` into its 1-based number, or None."""
    prefix = "ses-"
    if not isinstance(session_id, str) or not session_id.startswith(prefix):
        return None
    suffix = session_id[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def frc_needs_tissue(frc: dict[str, Any]) -> bool:
    """Whether a feedback run requests WM or CSF nuisance regressors."""
    glm = frc.get("glm", {}) or {}
    return bool(glm.get("include_wm_regressor", False) or glm.get("include_csf_regressor", False))


def frc_referenced_masks(frc: dict[str, Any]) -> set[str]:
    """Mask names a feedback run uses.

    ``network_assignment`` values for activation/differential modes, or connectivity
    pair names for connectivity mode. Mirrors
    ``participant_workflow._get_masks_for_frc``.
    """
    masks: set[str] = set()
    mode = frc.get("feedback_mode", "activation")
    if mode in ("activation", "differential"):
        for name in (frc.get("network_assignment", {}) or {}).values():
            if name:
                masks.add(name)
    elif mode == "connectivity":
        for pair in (frc.get("connectivity", {}) or {}).get("pairs") or []:
            for name in pair or []:
                if name:
                    masks.add(name)
    return masks


def localizer_producing(session: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The *fresh* localizer in ``session`` that produces mask ``name``, or None.

    Reused localizers are skipped so a cross-session lookup resolves to the session
    that originally computed the mask, never to a reuse pointer.
    """
    for loc in session.get("localizers", []):
        if localizer_is_reused(loc):
            continue
        if name in localizer_mask_names(loc):
            return cast("dict[str, Any]", loc)
    return None


def synthesize_reused_localizers(
    sessions: list[dict[str, Any]],
    feedback_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive reused-localizer entries from a session's cross-session mask selections.

    For each mask a feedback run sources from an earlier session (its ``mask_sources``
    map: name -> ``"ses-NN"``), find the localizer in that session producing it, group
    by ``(source_session, producer id/type)``, and emit one reused localizer per group
    restricted to the resolved names. The runtime per-localizer coregistration loop
    then pulls each group from its source session. Names with no resolvable producer
    are skipped (validation surfaces the error). Deterministic name ordering keeps the
    synthesized ``loc_idx`` reproducible.
    """
    reused: dict[str, str] = {}  # mask name -> source "ses-NN"
    for frc in feedback_runs:
        srcs = frc.get("mask_sources", {}) or {}
        for name in frc_referenced_masks(frc):
            src = srcs.get(name, "")
            if src:
                reused[name] = src

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for name in sorted(reused):
        src = reused[name]
        num = ses_num(src)
        if num is None or not (0 <= num - 1 < len(sessions)):
            continue
        producer = localizer_producing(sessions[num - 1], name)
        if producer is None:
            continue
        pid = producer.get("id") or producer.get("type") or "localizer"
        key = (src, pid)
        if key not in groups:
            groups[key] = {"producer": producer, "src": src, "pid": pid, "names": []}
            order.append(key)
        groups[key]["names"].append(name)

    synthesized: list[dict[str, Any]] = []
    for key in order:
        g = groups[key]
        src_loc: dict[str, Any] = g["producer"]
        names: list[str] = g["names"]
        ptype = src_loc.get("type", "rest")
        entry: dict[str, Any] = {"id": g["pid"], "type": ptype, "reuse_from_session": g["src"]}
        if src_loc.get("extraction"):
            entry["extraction"] = copy.deepcopy(src_loc["extraction"])
        if ptype == "rest":
            entry["networks"] = names
        elif ptype == "manual":
            entry["manual_mask_names"] = names
        else:  # task localizer
            base_contrasts = src_loc.get("contrasts", {}) or {}
            contrasts: dict[str, str] = {}
            include_reverse = False
            for nm in names:
                if nm.endswith("_reverse"):
                    include_reverse = True
                    base = nm[: -len("_reverse")]
                    if base in base_contrasts:
                        contrasts[base] = base_contrasts[base]
                elif nm in base_contrasts:
                    contrasts[nm] = base_contrasts[nm]
            entry["contrasts"] = contrasts
            if include_reverse:
                entry["include_reverse_contrasts"] = True
        synthesized.append(entry)
    return synthesized


def session_reuse_sources(sessions: list[dict[str, Any]], session_index: int) -> set[str]:
    """The set of earlier session ids that session ``session_index`` pulls masks from."""
    if not (0 <= session_index < len(sessions)):
        return set()
    sources: set[str] = set()
    for frc in sessions[session_index].get("feedback_runs", []):
        for src in (frc.get("mask_sources", {}) or {}).values():
            if src:
                sources.add(src)
    return sources


def tissue_required_in_session(sessions: list[dict[str, Any]], session_index: int) -> bool:
    """Whether session ``session_index`` must extract WM/CSF tissue masks.

    True if the session's own feedback needs nuisance regressors, or any *later*
    session that reuses this session's masks does — so a fresh-localizer session
    produces tissue masks a downstream feedback-only session can coregister.
    """
    if not (0 <= session_index < len(sessions)):
        return False
    if any(frc_needs_tissue(frc) for frc in sessions[session_index].get("feedback_runs", [])):
        return True
    this_id = f"ses-{session_index + 1:02d}"
    for later_idx in range(session_index + 1, len(sessions)):
        if this_id in session_reuse_sources(sessions, later_idx) and any(
            frc_needs_tissue(frc) for frc in sessions[later_idx].get("feedback_runs", [])
        ):
            return True
    return False


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


# ---------------------------------------------------------------------------
# Native file/folder dialogs
#
# All pickers below branch on platform.  On Windows/Linux a ``tkinter``
# dialog runs its own internal modal loop, so it works when dispatched to the
# main thread via :class:`~pineuro.gui.dispatch.MainThreadDispatcher`.  On
# macOS, however, the app runs ``ft.app()`` in a background thread and reserves
# the main thread for a bare queue loop with *no* Tk/Cocoa event loop — calling
# a tkinter native dialog in that state hangs the GUI (Python issue #38974), so
# on macOS we shell out to a native AppleScript dialog via ``osascript`` (a
# separate process with its own event loop, no main-thread requirement).
# ---------------------------------------------------------------------------

# AppleScript dialogs.  Dynamic strings are passed as ``argv`` (never
# interpolated into the source) via the ``on run argv`` handler, so titles /
# names / paths need no escaping.  An empty-string argument means "unset".
_OSA_CHOOSE_FOLDER = """\
on run argv
    set p to item 1 of argv
    set d to item 2 of argv
    if d is "" then
        set f to (choose folder with prompt p)
    else
        set f to (choose folder with prompt p default location (POSIX file d as alias))
    end if
    return POSIX path of f
end run
"""

_OSA_CHOOSE_FILE = """\
on run argv
    set p to item 1 of argv
    set d to item 2 of argv
    if d is "" then
        set f to (choose file with prompt p)
    else
        set f to (choose file with prompt p default location (POSIX file d as alias))
    end if
    return POSIX path of f
end run
"""

_OSA_CHOOSE_FILE_NAME = """\
on run argv
    set p to item 1 of argv
    set n to item 2 of argv
    set d to item 3 of argv
    if d is "" then
        if n is "" then
            set f to (choose file name with prompt p)
        else
            set f to (choose file name with prompt p default name n)
        end if
    else
        if n is "" then
            set f to (choose file name with prompt p default location (POSIX file d as alias))
        else
            set f to (choose file name with prompt p default name n default location (POSIX file d as alias))
        end if
    end if
    return POSIX path of f
end run
"""


def _osascript_path() -> str | None:
    """Return the path to ``osascript``, or ``None`` if it is not available."""
    import shutil

    return shutil.which("osascript")


def _run_osascript(script: str, *args: str) -> str | None:
    """Run an AppleScript via ``osascript``, passing dynamic strings as ``argv``.

    The *script* reads its inputs through ``on run argv`` (``item 1 of argv`` …)
    so user-supplied values are never interpolated into the source — no escaping
    or injection risk.

    Returns the trimmed stdout (a POSIX path) on success, or ``None`` when the
    user cancels (osascript exits non-zero with ``User canceled. (-128)``) or the
    dialog cannot be shown.
    """
    import subprocess

    osascript = _osascript_path()
    if osascript is None:
        logger.error("osascript not found — cannot open native macOS dialog")
        return None

    try:
        proc = subprocess.run(
            [osascript, "-e", script, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - subprocess launch failure
        logger.error(f"Error invoking osascript: {exc}")
        return None

    if proc.returncode != 0:
        # Non-zero is the normal "cancelled" path (-128); log anything else.
        logger.debug("osascript returned %s: %s", proc.returncode, proc.stderr.strip())
        return None

    return proc.stdout.strip() or None


def _seed_dir(initial_path: str | None) -> str:
    """Return an existing directory derived from *initial_path*, or ``""``.

    A file path yields its parent; a directory path is used directly.  Returns
    the empty string (the AppleScript "unset" sentinel) when nothing exists.
    """
    import os

    if initial_path:
        if os.path.isfile(initial_path):
            return os.path.dirname(initial_path)
        if os.path.isdir(initial_path):
            return initial_path
    return ""


def pick_file(
    title: str = "Select file",
    *,
    filetypes: list[tuple[str, str]] | None = None,
    initial_path: str | None = None,
) -> str | None:
    """Open a native file-open dialog and return the chosen path.

    On macOS uses a native AppleScript (``osascript``) dialog; on Windows/Linux
    uses ``tkinter.filedialog.askopenfilename`` dispatched to the main thread via
    :class:`~pineuro.gui.dispatch.MainThreadDispatcher`.

    Args:
        title: Dialog window title.
        filetypes: tkinter file-type filters as ``(label, pattern)`` tuples.
            **Ignored on macOS** (AppleScript filters by UTI, not glob).
        initial_path: A file or directory used to seed the dialog's starting folder.

    Returns:
        The selected absolute path, or ``None`` if the dialog was cancelled or
        could not be opened.
    """
    import sys

    if sys.platform == "darwin":
        return _run_osascript(_OSA_CHOOSE_FILE, title, _seed_dir(initial_path))

    if filetypes is None:
        filetypes = [("All Files", "*.*")]

    try:
        import os
        import tkinter as tk
        from tkinter import filedialog

        from pineuro.gui.dispatch import get_dispatcher
    except ImportError:
        logger.error("tkinter is not available — cannot open file picker")
        return None

    def _show_dialog() -> str:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial_dir = None
        if initial_path:
            if os.path.isfile(initial_path):
                initial_dir = os.path.dirname(initial_path)
            elif os.path.isdir(initial_path):
                initial_dir = initial_path
        result = filedialog.askopenfilename(title=title, initialdir=initial_dir, filetypes=filetypes)
        root.destroy()
        return result

    try:
        path = get_dispatcher().submit(_show_dialog).result()
    except Exception as exc:
        logger.error(f"Error opening file picker: {exc}")
        return None

    return path or None


def pick_directory(
    title: str = "Select folder",
    *,
    initial_path: str | None = None,
) -> str | None:
    """Open a native folder-selection dialog and return the chosen path.

    On macOS uses a native AppleScript (``osascript``) dialog; on Windows/Linux
    uses ``tkinter.filedialog.askdirectory`` dispatched to the main thread via
    :class:`~pineuro.gui.dispatch.MainThreadDispatcher`.

    Args:
        title: Dialog window title.
        initial_path: An existing directory used to seed the starting folder.

    Returns:
        The selected directory path, or ``None`` if cancelled / unavailable.
    """
    import sys

    if sys.platform == "darwin":
        # ``POSIX path of`` a folder yields a trailing slash; strip it so the
        # result matches tkinter's askdirectory (and the Windows/Linux) convention.
        path = _run_osascript(_OSA_CHOOSE_FOLDER, title, _seed_dir(initial_path))
        return (path.rstrip("/") or path) if path else path

    try:
        import os
        import tkinter as tk
        from tkinter import filedialog

        from pineuro.gui.dispatch import get_dispatcher
    except ImportError:
        logger.error("tkinter is not available — cannot open folder picker")
        return None

    def _show_dialog() -> str:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial_dir = initial_path if initial_path and os.path.isdir(initial_path) else None
        result = filedialog.askdirectory(title=title, initialdir=initial_dir)
        root.destroy()
        return result

    try:
        path = get_dispatcher().submit(_show_dialog).result()
    except Exception as exc:
        logger.error(f"Error opening folder picker: {exc}")
        return None

    return path or None


def pick_save_file(
    title: str = "Save file",
    *,
    default_name: str | None = None,
    default_extension: str | None = None,
    filetypes: list[tuple[str, str]] | None = None,
    initial_path: str | None = None,
) -> str | None:
    """Open a native save-file dialog and return the chosen path.

    On macOS uses a native AppleScript (``osascript``) dialog; on Windows/Linux
    uses ``tkinter.filedialog.asksaveasfilename`` dispatched to the main thread
    via :class:`~pineuro.gui.dispatch.MainThreadDispatcher`.

    Args:
        title: Dialog window title.
        default_name: Suggested file name (``initialfile`` on tkinter,
            ``default name`` on macOS).
        default_extension: Extension appended when the user omits one
            (``defaultextension`` on tkinter; ``choose file name`` does not
            enforce one, so it is appended manually on macOS).
        filetypes: tkinter file-type filters; **ignored on macOS**.
        initial_path: A file or directory used to seed the starting folder.

    Returns:
        The selected path, or ``None`` if cancelled / unavailable.
    """
    import os
    import sys

    if sys.platform == "darwin":
        path = _run_osascript(_OSA_CHOOSE_FILE_NAME, title, default_name or "", _seed_dir(initial_path))
        if path and default_extension and not os.path.splitext(path)[1]:
            ext = default_extension if default_extension.startswith(".") else f".{default_extension}"
            path = path + ext
        return path

    if filetypes is None:
        filetypes = [("All Files", "*.*")]

    try:
        import tkinter as tk
        from tkinter import filedialog

        from pineuro.gui.dispatch import get_dispatcher
    except ImportError:
        logger.error("tkinter is not available — cannot open save dialog")
        return None

    def _show_dialog() -> str:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        kwargs: dict[str, Any] = {"title": title, "filetypes": filetypes}
        if default_extension:
            kwargs["defaultextension"] = default_extension
        if default_name:
            kwargs["initialfile"] = default_name
        seed = _seed_dir(initial_path)
        if seed:
            kwargs["initialdir"] = seed
        result = filedialog.asksaveasfilename(**kwargs)
        root.destroy()
        return result

    try:
        path = get_dispatcher().submit(_show_dialog).result()
    except Exception as exc:
        logger.error(f"Error opening save dialog: {exc}")
        return None

    return path or None
