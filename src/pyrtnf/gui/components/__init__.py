"""Reusable GUI components for the pyrtnf application.

This package contains reusable components and utilities for the GUI:
- PipelineRunner: Background thread execution of pipeline steps
- ActivationPlot: Real-time activation timeseries chart
- FDPlot: Real-time framewise displacement timeseries chart
- TaskMirror: Generic task visualization mirror
- FeedbackLogPoller: Polls TSV files for new data
- FeedbackMonitor: High-level monitor for feedback sessions
- QCReportManager: Manages QC report discovery and viewing
"""

from .pipeline_runner import PipelineRunner
from .activation_plot import ActivationPlot, ActivationData
from .fd_plot import FDPlot, FDData
from .task_mirror import TaskMirror
from .csv_poller import FeedbackLogPoller, FeedbackMonitor
from .qc_viewer import QCReportManager, open_qc_report

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
]
