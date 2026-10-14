"""Reusable GUI components for the pineuro participant workflow.

This package contains Flet-based UI components used primarily in the
participant workflow screen. These components provide real-time
visualization, dialog interactions, and background task management.

Pipeline Execution
------------------
PipelineRunner : Background thread execution of pipeline steps
    Runs preprocessing, mask extraction, and feedback in a thread
    while providing progress updates to the GUI.

Real-Time Visualization
-----------------------
ActivationPlot : Network activation timeseries chart
    Displays real-time activation values for each network mask.
    Uses flet-charts LineChart for smooth updates.

FDPlot : Framewise displacement timeseries chart
    Shows motion (FD) over time with threshold highlighting.
    Helps identify volumes with excessive motion.

TaskMirror : PsychoPy task screen mirroring
    Displays a scaled mirror of the PsychoPy stimulus window
    for experimenter monitoring.

Data Polling
------------
FeedbackLogPoller : Background TSV file watcher
    Polls feedbacklog.tsv for new rows and triggers callbacks.

FeedbackMonitor : High-level feedback task monitor
    Combines log polling with activation/FD plotting.

QC Integration
--------------
QCReportManager : QC report discovery and management
    Finds and catalogs QC reports across pipeline stages.

open_qc_report() : Open QC report in browser
    Platform-aware browser launch for HTML reports.

Collection Progress
-------------------
CollectionProgressPanel : DICOM/NIfTI collection display
    Shows progress during volume collection with file counts.

Disk Space
----------
DiskSpaceBar : Free-disk-space readout at the bottom of the window
    Free space on the volumes the current screen writes to, flagged when
    below the projected size of one more session.

Dialogs
-------
RunSelectionDialog : Feedback run selection
    Configure which runs to execute with acquisition parameters.

MaskSelectionDialog : ICA component review
    Review and optionally modify network mask assignments.

SessionSelectionDialog : Multi-session session management
    Select existing sessions or create new ones for longitudinal studies.

FeedbackSummaryDialog : Post-run summary
    Display hit counts, motion statistics, and run completion status.

Usage
-----
Components are typically instantiated by the ParticipantWorkflowScreen
and added to the page controls. Example:

>>> from pineuro.gui.components import ActivationPlot
>>> plot = ActivationPlot(
...     mask_names=['ContA', 'DefaultA'],
...     n_volumes=200,
...     height=250,
... )
>>> container = plot.build()  # Returns Flet Container
"""

import importlib

# Components are resolved on demand (PEP 562) so that importing one component
# submodule does not eagerly import its siblings -- several of which pull numpy,
# pandas, pydicom, or the flet_charts/matplotlib stack at import time.  Mirrors
# the lazy-import pattern in ``pineuro/__init__.py``.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ActivationData": (".activation_plot", "ActivationData"),
    "ActivationPlot": (".activation_plot", "ActivationPlot"),
    "BaseDialog": (".base_dialog", "BaseDialog"),
    "CollectionProgressPanel": (".collection_progress", "CollectionProgressPanel"),
    "DiskSpaceBar": (".disk_space_bar", "DiskSpaceBar"),
    "FeedbackLogPoller": (".csv_poller", "FeedbackLogPoller"),
    "FeedbackMonitor": (".csv_poller", "FeedbackMonitor"),
    "FeedbackSummaryData": (".feedback_summary", "FeedbackSummaryData"),
    "FeedbackSummaryDialog": (".feedback_summary", "FeedbackSummaryDialog"),
    "compute_feedback_summary": (".feedback_summary", "compute_feedback_summary"),
    "MaskReviewInfo": (".mask_selection_dialog", "MaskReviewInfo"),
    "MaskSelectionDialog": (".mask_selection_dialog", "MaskSelectionDialog"),
    "PipelineRunner": (".pipeline_runner", "PipelineRunner"),
    "FDPlot": (".qc_plots", "FDPlot"),
    "QCReportManager": (".qc_viewer", "QCReportManager"),
    "open_qc_report": (".qc_viewer", "open_qc_report"),
    "SessionSelectionDialog": (".session_selection_dialog", "SessionSelectionDialog"),
    "TaskMirror": (".task_mirror", "TaskMirror"),
    "RunGroup": (".unified_attempt_dialog", "RunGroup"),
    "UnifiedAttemptDialog": (".unified_attempt_dialog", "UnifiedAttemptDialog"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        val = getattr(importlib.import_module(module_path, __name__), attr)
        globals()[name] = val  # cache; subsequent access skips __getattr__
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDialog",
    "PipelineRunner",
    "ActivationPlot",
    "ActivationData",
    "FDPlot",
    "TaskMirror",
    "FeedbackLogPoller",
    "FeedbackMonitor",
    "QCReportManager",
    "open_qc_report",
    "CollectionProgressPanel",
    "DiskSpaceBar",
    "UnifiedAttemptDialog",
    "RunGroup",
    "MaskSelectionDialog",
    "MaskReviewInfo",
    "SessionSelectionDialog",
    "FeedbackSummaryDialog",
    "FeedbackSummaryData",
    "compute_feedback_summary",
]
