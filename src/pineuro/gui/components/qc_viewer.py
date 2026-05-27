"""QC report viewer component for displaying HTML reports.

This module provides functionality to view QC HTML reports generated
by pineuro's QC module. Since Flet doesn't have native HTML embedding,
reports are opened in an external browser.
"""

import logging
import platform
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


def open_qc_report(report_path: Path, screen: int | None = None) -> bool:
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
        url = abs_path.as_uri()

        # Try to use xdg-open on Linux for better screen control
        system = platform.system()
        if system == "Linux" and screen is not None:
            # Use screen-aware browser opening (from pineuro.qc)
            try:
                from pineuro.qc import _open_html_in_browser

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
        self.network_order: list[str] | None = None

    def _sort_overlays(self, plots: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
        """Sort overlay plots by network_order if set, otherwise keep as-is."""
        if not self.network_order:
            return plots

        order = self.network_order

        def sort_key(item: tuple[str, Path]) -> int:
            label = item[0]
            # Extract network name from label like "DefaultA Mask (Subject Space)"
            network = label.split(" Mask")[0] if " Mask" in label else label
            try:
                return order.index(network)
            except ValueError:
                return len(order)

        return sorted(plots, key=sort_key)

    def get_preprocessing_qc(self) -> Path | None:
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

    def get_mask_extraction_qc(self) -> Path | None:
        """Get path to mask extraction QC report.

        Returns:
            Path to report if exists, None otherwise
        """
        # Primary location: participant-level qc/ directory
        # (This is where qc.generate_mask_extraction_report() outputs by default)
        qc_dir = self.base_dir / "qc"
        report_path = qc_dir / f"{self.participant_id}_mask_extraction_qc.html"

        if report_path.exists():
            return report_path

        # Try alternative patterns in participant-level qc/
        if qc_dir.exists():
            for pattern in ["*mask_extraction_qc.html", "*masks_qc.html"]:
                matches = list(qc_dir.glob(pattern))
                if matches:
                    return matches[0]

        # Fallback: check step-level directory (mask_extraction/qc/)
        step_qc_dir = self.base_dir / "mask_extraction" / "qc"
        if step_qc_dir.exists():
            step_report = step_qc_dir / f"{self.participant_id}_mask_extraction_qc.html"
            if step_report.exists():
                return step_report

            for pattern in ["*mask_extraction_qc.html", "*masks_qc.html"]:
                matches = list(step_qc_dir.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def get_coregistration_qc(self, run_id: int = 1) -> Path | None:
        """Get path to reference coregistration QC report.

        Args:
            run_id: Run number (1 for initial, 2 for realtime)

        Returns:
            Path to report if exists, None otherwise
        """
        # Coregistration QC location depends on run
        if run_id == 1:
            qc_dir = self.base_dir / "mask_extraction" / "04_reference" / "qc"
        else:
            qc_dir = self.base_dir / "feedback" / "01_reference" / "qc"

        # Try standard naming
        report_path = qc_dir / f"{self.participant_id}_reference_run-{run_id:02d}_qc.html"
        if report_path.exists():
            return report_path

        # Try alternative patterns
        for pattern in ["*reference*qc.html", "*coreg*qc.html"]:
            matches = list(qc_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def get_feedback_qc(self, task: str, run: int, acq: int | None = None) -> Path | None:
        """Get path to feedback/transfer run QC report.

        Args:
            task: Task type ("feedback" or "transfer")
            run: Run number
            acq: Acquisition number (optional)

        Returns:
            Path to report if exists, None otherwise
        """
        qc_dir = self.base_dir / "feedback" / "02_feedback_output" / "qc"

        # Try acq-specific path first if acq provided
        if acq is not None:
            report_path = qc_dir / f"{self.participant_id}_task-{task}_acq-{acq:02d}_run-{run:02d}_feedback_qc.html"
            if report_path.exists():
                return report_path

        # Glob to find the matching report when acq isn't specified
        pattern = f"*task-{task}*run-{run:02d}*qc.html"
        matches = list(qc_dir.glob(pattern))
        if matches:
            # Return most recent if multiple
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        return None

    def get_combined_qc(self) -> Path | None:
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

    def list_available_reports(self) -> list[dict]:
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
                "name": "Initial Reference Coregistration",
                "type": "coregistration_initial",
                "path": self.get_coregistration_qc(run_id=1),
            },
            {
                "name": "Realtime Reference Coregistration",
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
                    reports.append(
                        {
                            "name": f"{task.capitalize()} Run {run}",
                            "type": f"{task}_run_{run}",
                            "path": path,
                        }
                    )

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

    # =========================================================================
    # Plot Discovery Methods (for inline QC viewing)
    # =========================================================================

    def get_preprocessing_plots(self) -> list[tuple[str, Path]]:
        """Get key preprocessing QC plots.

        Returns:
            List of (label, path) tuples for available plots
        """
        plots: list[tuple[str, Path]] = []
        qc_dir = self.base_dir / "preprocessing" / "qc"

        if not qc_dir.exists():
            return plots

        # Brain mask coverage (most important)
        mask_plot = qc_dir / "brain_mask_coverage.png"
        if mask_plot.exists():
            plots.append(("Brain Mask Coverage", mask_plot))

        # Motion timeseries plots (show first run only for brevity in modal)
        motion_plots = sorted(qc_dir.glob("motion_timeseries_*.png"))
        if motion_plots:
            plots.append(("Motion Parameters", motion_plots[0]))

        # Framewise displacement plots (show first run only)
        fd_plots = sorted(qc_dir.glob("framewise_displacement_*.png"))
        if fd_plots:
            plots.append(("Framewise Displacement", fd_plots[0]))

        # Registration overlay (if multi-run)
        reg_plot = qc_dir / "registration_overlay.png"
        if reg_plot.exists():
            plots.append(("Multi-Run Registration", reg_plot))

        return plots

    def get_mask_extraction_plots(self) -> list[tuple[str, Path]]:
        """Get key mask extraction QC plots.

        Returns:
            List of (label, path) tuples for available plots
        """
        plots: list[tuple[str, Path]] = []
        qc_dir = self.base_dir / "qc"

        if not qc_dir.exists():
            return plots

        # ICA component grid
        ica_grid = qc_dir / "ica_component_grid.png"
        if ica_grid.exists():
            plots.append(("ICA Components", ica_grid))

        # Subject space overlays (all networks)
        subject_overlays: list[tuple[str, Path]] = []
        overlay_dir = qc_dir / "overlays" / "subject"
        if overlay_dir.exists():
            for overlay in sorted(overlay_dir.glob("*_overlay.png")):
                # Extract network name from filename
                # e.g., "sub-01_DefaultA_overlay.png" -> "DefaultA"
                network = overlay.stem.replace(f"{self.participant_id}_", "").replace("_overlay", "")
                subject_overlays.append((f"{network} Mask (Subject Space)", overlay))
        plots.extend(self._sort_overlays(subject_overlays))

        # MNI space overlays (registration quality check)
        mni_overlays: list[tuple[str, Path]] = []
        mni_dir = qc_dir / "overlays" / "mni"
        if mni_dir.exists():
            for overlay in sorted(mni_dir.glob("*_overlay.png")):
                network = overlay.stem.replace(f"{self.participant_id}_", "").replace("_overlay", "")
                mni_overlays.append((f"{network} Mask (MNI Space)", overlay))
        plots.extend(self._sort_overlays(mni_overlays))

        # Reference overlays (coregistration) - only run-01 for initial QC
        ref_overlays: list[tuple[str, Path]] = []
        reference_dir = qc_dir / "overlays" / "reference"
        if reference_dir.exists():
            for overlay in sorted(reference_dir.glob("*_overlay.png")):
                # Only include run-01 overlays for initial mask extraction QC
                if "run-01" not in overlay.stem:
                    continue
                # Extract network name (e.g., "DefaultA" from "sub-01_DefaultA_thr_to_task-reference_run-01_mean_overlay")
                network = overlay.stem.replace(f"{self.participant_id}_", "").replace("_overlay", "")
                # Clean up the network name to remove the thr_to_task-reference_run-XX_mean suffix
                if "_thr_to_task-reference" in network:
                    network = network.split("_thr_to_task-reference")[0]
                ref_overlays.append((f"{network} Mask (Reference Space)", overlay))
        plots.extend(self._sort_overlays(ref_overlays))

        return plots

    def get_reference_qc_plots(self, run_id: int = 2) -> list[tuple[str, Path]]:
        """Get ONLY reference coregistration QC plots for a specific run.

        Use this for realtime coregistration approval where only
        the reference overlays are relevant (ICA and subject space didn't change).

        Args:
            run_id: Run number to filter overlays (default 2 for realtime).
                    Only overlays containing "run-{run_id:02d}" in filename are included.

        Returns:
            List of (label, path) tuples for reference overlay images
        """
        plots: list[tuple[str, Path]] = []
        run_pattern = f"run-{run_id:02d}"

        # QC overlays are in participant-level qc/ directory
        qc_dir = self.base_dir / "qc"

        if not qc_dir.exists():
            logger.debug(f"QC directory not found: {qc_dir}")
            return plots

        # Reference overlays only - filter by run number
        reference_dir = qc_dir / "overlays" / "reference"
        if reference_dir.exists():
            for overlay in sorted(reference_dir.glob("*_overlay.png")):
                # Only include overlays for the specified run
                if run_pattern not in overlay.stem:
                    continue
                # Extract network name (e.g., "DefaultA" from "sub-01_DefaultA_thr_to_task-reference_run-02_mean_overlay")
                network = overlay.stem.replace(f"{self.participant_id}_", "").replace("_overlay", "")
                # Clean up the network name to remove the thr_to_task-reference_run-XX_mean suffix
                if "_thr_to_task-reference" in network:
                    network = network.split("_thr_to_task-reference")[0]
                plots.append((f"{network} Mask (Reference Space)", overlay))

        plots = self._sort_overlays(plots)

        if not plots:
            logger.debug(f"No reference overlay plots found in {reference_dir} for {run_pattern}")

        return plots

    def get_feedback_plots(self, task: str, run: int) -> list[tuple[str, Path]]:
        """Get key feedback QC plots.

        Args:
            task: Task type ("feedback" or "transfer")
            run: Run number

        Returns:
            List of (label, path) tuples for available plots
        """
        plots: list[tuple[str, Path]] = []
        qc_dir = self.base_dir / "feedback" / "qc"

        if not qc_dir.exists():
            return plots

        # Activation timeseries
        ts_plot = qc_dir / "activation_timeseries.png"
        if ts_plot.exists():
            plots.append(("Activation Timeseries", ts_plot))

        # Baseline distributions
        baseline_plot = qc_dir / "baseline_distributions.png"
        if baseline_plot.exists():
            plots.append(("Baseline Distributions", baseline_plot))

        # Design matrix
        design_plot = qc_dir / "design_matrix.png"
        if design_plot.exists():
            plots.append(("GLM Design Matrix", design_plot))

        return plots
