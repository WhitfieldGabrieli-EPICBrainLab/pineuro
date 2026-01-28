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

FeedbackMonitor : High-level feedback session monitor
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

Dialogs
-------
RunSelectionDialog : Feedback run selection
    Configure which runs to execute with acquisition parameters.

MaskSelectionDialog : ICA component review
    Review and optionally modify network mask assignments.

VisitSelectionDialog : Multi-visit session management
    Select existing visits or create new ones for longitudinal studies.

FeedbackSummaryDialog : Post-run summary
    Display hit counts, motion statistics, and run completion status.

Usage
-----
Components are typically instantiated by the ParticipantWorkflowScreen
and added to the page controls. Example:

>>> from pineuro.gui.components import ActivationPlot
>>> plot = ActivationPlot(
...     mask_names=['DefaultA', 'ContA'],
...     n_volumes=200,
...     height=250,
... )
>>> container = plot.build()  # Returns Flet Container
"""

from .pipeline_runner import PipelineRunner
from .activation_plot import ActivationPlot, ActivationData
from .fd_plot import FDPlot, FDData
from .task_mirror import TaskMirror
from .csv_poller import FeedbackLogPoller, FeedbackMonitor
from .qc_viewer import QCReportManager, open_qc_report
from .collection_progress import CollectionProgressPanel
from .run_selection_dialog import RunSelectionDialog, RunInfo
from .mask_selection_dialog import MaskSelectionDialog, MaskReviewInfo
from .visit_selection_dialog import VisitSelectionDialog
from .feedback_summary import FeedbackSummaryDialog, FeedbackSummaryData, compute_feedback_summary

__all__ = [
    "PipelineRunner",
    "ActivationPlot",
    "ActivationData",
    "FDPlot",
    "FDData",
    "TaskMirror",
    "FeedbackLogPoller",
    "FeedbackMonitor",
    "QCReportManager",
    "open_qc_report",
    "CollectionProgressPanel",
    "RunSelectionDialog",
    "RunInfo",
    "MaskSelectionDialog",
    "MaskReviewInfo",
    "VisitSelectionDialog",
    "FeedbackSummaryDialog",
    "FeedbackSummaryData",
    "compute_feedback_summary",
]
