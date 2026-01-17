"""QC report viewer component for displaying HTML reports.

This module provides functionality to view QC HTML reports generated
by pyrtnf's QC module. Since Flet doesn't have native HTML embedding,
reports are opened in an external browser.
"""

import logging
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional, List
import platform

logger = logging.getLogger(__name__)


def open_qc_report(report_path: Path, screen: Optional[int] = None) -> bool:
    """Open a QC report HTML file in the default browser.

    Args:
        report_path: Path to the HTML report file
        screen: Optional screen number to open on (Linux only with xdg-open)

    Returns:
        True if report was opened successfully, False otherwise
    """
    if not report_path.exists():
        logger.warning(f"QC report not found: {report_path}")
        return False

    try:
        # Convert to absolute path
        abs_path = report_path.resolve()
        url = f"file://{abs_path}"

        # Try to use xdg-open on Linux for better screen control
        system = platform.system()
        if system == "Linux" and screen is not None:
            # Use screen-aware browser opening (from pyrtnf.qc)
            try:
                from pyrtnf.qc import _open_html_in_browser
                _open_html_in_browser(abs_path)
                return True
            except ImportError:
                pass

        # Fall back to webbrowser module
        webbrowser.open(url)
        logger.info(f"Opened QC report: {report_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to open QC report: {e}")
        return False


class QCReportManager:
    """Manages QC reports for a participant session.

    Detects available QC reports and provides methods to open them.
    """

    def __init__(self, base_dir: Path, participant_id: str):
        """Initialize the QC report manager.

        Args:
            base_dir: Base directory for participant data
            participant_id: Participant ID
        """
        self.base_dir = Path(base_dir)
        self.participant_id = participant_id

    def get_preprocessing_qc(self) -> Optional[Path]:
        """Get path to preprocessing QC report.

        Returns:
            Path to report if exists, None otherwise
        """
        # Preprocessing QC is in preprocessing/qc/
        qc_dir = self.base_dir / "preprocessing" / "qc"
        report_path = qc_dir / f"{self.participant_id}_task-rest_preprocessing_qc.html"

        if report_path.exists():
            return report_path

        # Try alternative naming
        for pattern in ["*preprocessing_qc.html", "*preproc_qc.html"]:
            matches = list(qc_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def get_mask_extraction_qc(self) -> Optional[Path]:
        """Get path to mask extraction QC report.

        Returns:
            Path to report if exists, None otherwise
        """
        qc_dir = self.base_dir / "mask_extraction" / "qc"
        report_path = qc_dir / f"{self.participant_id}_mask_extraction_qc.html"

        if report_path.exists():
            return report_path

        # Try alternative naming
        for pattern in ["*mask_extraction_qc.html", "*masks_qc.html"]:
            matches = list(qc_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def get_coregistration_qc(self, run_id: int = 1) -> Optional[Path]:
        """Get path to 2vol coregistration QC report.

        Args:
            run_id: Run number (1 for initial, 2 for realtime)

        Returns:
            Path to report if exists, None otherwise
        """
        # Coregistration QC location depends on run
        if run_id == 1:
            qc_dir = self.base_dir / "mask_extraction" / "04_2vol" / "qc"
        else:
            qc_dir = self.base_dir / "feedback" / "01_2vol" / "qc"

        # Try standard naming
        report_path = qc_dir / f"{self.participant_id}_2vol_run-{run_id:02d}_qc.html"
        if report_path.exists():
            return report_path

        # Try alternative patterns
        for pattern in ["*2vol*qc.html", "*coreg*qc.html"]:
            matches = list(qc_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def get_feedback_qc(self, task: str, run: int) -> Optional[Path]:
        """Get path to feedback/transfer run QC report.

        Args:
            task: Task type ("feedback" or "transfer")
            run: Run number

        Returns:
            Path to report if exists, None otherwise
        """
        qc_dir = self.base_dir / "feedback" / "qc"
        report_path = qc_dir / f"{self.participant_id}_task-{task}_run-{run:02d}_feedback_qc.html"

        if report_path.exists():
            return report_path

        # Try alternative pattern
        pattern = f"*task-{task}_run-{run:02d}*qc.html"
        matches = list(qc_dir.glob(pattern))
        if matches:
            return matches[0]

        return None

    def get_combined_qc(self) -> Optional[Path]:
        """Get path to combined session QC report.

        Returns:
            Path to report if exists, None otherwise
        """
        qc_dir = self.base_dir / "qc"
        report_path = qc_dir / f"{self.participant_id}_qc.html"

        if report_path.exists():
            return report_path

        # Try alternative patterns
        for pattern in ["*session_qc.html", "*combined_qc.html"]:
            matches = list(qc_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def list_available_reports(self) -> List[dict]:
        """List all available QC reports.

        Returns:
            List of dicts with report info (name, path, exists)
        """
        reports = [
            {
                "name": "Preprocessing",
                "type": "preprocessing",
                "path": self.get_preprocessing_qc(),
            },
            {
                "name": "Mask Extraction",
                "type": "mask_extraction",
                "path": self.get_mask_extraction_qc(),
            },
            {
                "name": "Initial 2vol Coregistration",
                "type": "coregistration_initial",
                "path": self.get_coregistration_qc(run_id=1),
            },
            {
                "name": "Realtime 2vol Coregistration",
                "type": "coregistration_realtime",
                "path": self.get_coregistration_qc(run_id=2),
            },
            {
                "name": "Combined Session QC",
                "type": "combined",
                "path": self.get_combined_qc(),
            },
        ]

        # Add feedback run QC reports
        for task in ["transfer", "feedback"]:
            for run in range(1, 10):  # Check up to 9 runs
                path = self.get_feedback_qc(task, run)
                if path:
                    reports.append({
                        "name": f"{task.capitalize()} Run {run}",
                        "type": f"{task}_run_{run}",
                        "path": path,
                    })

        # Filter to only existing reports
        return [r for r in reports if r["path"] is not None]

    def open_report(self, report_type: str) -> bool:
        """Open a QC report by type.

        Args:
            report_type: Type of report (preprocessing, mask_extraction, etc.)

        Returns:
            True if opened successfully
        """
        path_getters = {
            "preprocessing": lambda: self.get_preprocessing_qc(),
            "mask_extraction": lambda: self.get_mask_extraction_qc(),
            "coregistration_initial": lambda: self.get_coregistration_qc(1),
            "coregistration_realtime": lambda: self.get_coregistration_qc(2),
            "combined": lambda: self.get_combined_qc(),
        }

        # Handle feedback/transfer runs
        if report_type.startswith(("transfer_run_", "feedback_run_")):
            parts = report_type.split("_")
            task = parts[0]
            run = int(parts[2])
            path = self.get_feedback_qc(task, run)
        elif report_type in path_getters:
            path = path_getters[report_type]()
        else:
            logger.warning(f"Unknown report type: {report_type}")
            return False

        if path:
            return open_qc_report(path)
        else:
            logger.warning(f"Report not found: {report_type}")
            return False
