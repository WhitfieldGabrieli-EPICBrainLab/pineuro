"""Reusable GUI components for the pineuro application.

This package contains reusable components and utilities for the GUI:
- PipelineRunner: Background thread execution of pipeline steps
- ActivationPlot: Real-time activation timeseries chart
- FDPlot: Real-time framewise displacement timeseries chart
- TaskMirror: Generic task visualization mirror
- FeedbackLogPoller: Polls TSV files for new data
- FeedbackMonitor: High-level monitor for feedback sessions
- QCReportManager: Manages QC report discovery and viewing
- CollectionProgressPanel: Progress display for DICOM collection
- RunSelectionDialog: Dialog for selecting runs for preprocessing
- MaskSelectionDialog: Dialog for reviewing and customizing network masks
- VisitSelectionDialog: Dialog for selecting/creating visits in multi-visit studies
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
]
