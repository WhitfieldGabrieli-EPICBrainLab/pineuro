"""Participant Workflow screen for executing pipeline steps.

This module provides the main dashboard for executing pineuro pipeline steps for a
single participant:
- Step-by-step workflow with status tracking
- Parameter preview before execution
- Real-time monitoring during feedback runs
- Log console for pipeline output
- QC report viewing

The workflow is divided into four main sections:
1. Localizer Session: Initial 2vol collection and resting-state collection
2. Mask Extraction: Preprocessing, ICA-based network mask extraction, and coregistration to 2vol
3. Feedback Session: Real-time 2vol collection and feedback runs
4. Quality Control: View QC reports for each phase

Status is automatically detected based on output file existence, allowing sessions
to be resumed after interruption.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum

import flet as ft

from ..config_manager import ConfigManager
from ..models import StudyConfig
from ..components.pipeline_runner import PipelineRunner
from ..components.activation_plot import ActivationPlot
from ..components.fd_plot import FDPlot
from ..components.task_mirror import TaskMirror
from ..components.csv_poller import FeedbackMonitor
from ..components.qc_viewer import QCReportManager
from ..components.collection_progress import CollectionProgressPanel
from ..components.run_selection_dialog import RunSelectionDialog, RunInfo
from ..components.mask_selection_dialog import MaskSelectionDialog, MaskReviewInfo
from pineuro.feedback import get_next_run_number, get_slot_run_info, STEP_TO_RUN_NUMBER
from pineuro.utils import get_dicom_files

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"  # Partial completion (user stopped early)
    FAILED = "failed"


class ParticipantWorkflowScreen:
    """Participant workflow dashboard for pipeline execution."""

    def __init__(
        self,
        page: ft.Page,
        config_manager: ConfigManager,
        study_config: StudyConfig,
        participant_id: str,
        visit_id: Optional[str] = None,
        visit_config: Optional["VisitConfig"] = None,
        on_back=None,
    ):
        """Initialize the participant workflow screen.

        Args:
            page: Flet Page object
            config_manager: ConfigManager instance
            study_config: Study configuration to use
            participant_id: Participant ID (e.g., "sub-01")
            visit_id: Visit identifier (e.g., "ses-01"). None for single-visit mode.
            visit_config: VisitConfig for new visits. None when continuing existing visit.
            on_back: Callback when back button is clicked
        """
        self.page = page
        self.config_manager = config_manager
        self.participant_id = participant_id
        self.visit_id = visit_id
        self.visit_config = visit_config
        self.on_back_callback = on_back

        # Create participant-specific base directory
        # Make a deep copy of study config to avoid mutating the original
        import copy
        self.study_config = copy.deepcopy(study_config)
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])

        # The base_dir might already contain participant/visit paths from a previous session
        # (if the study_config object was reused). We need to get the study-level base_dir.
        # Look for participant ID in the path and strip it and everything after.
        base_dir_str = str(base_dir)
        if f"/{participant_id}" in base_dir_str:
            # Strip participant and everything after it
            base_dir = Path(base_dir_str.split(f"/{participant_id}")[0])
            logger.debug(f"Stripped participant path, using base_dir: {base_dir}")

        if visit_id:
            # Multi-visit mode: base_dir/participant_id/visit_id/
            participant_base_dir = base_dir / participant_id / visit_id
        else:
            # Single-visit mode: base_dir/participant_id/
            participant_base_dir = base_dir / participant_id

        # Update the session config with participant-specific base directory
        self.study_config.session_config["paths"]["base_dir"] = str(participant_base_dir)

        # Create the participant directory if it doesn't exist
        participant_base_dir.mkdir(parents=True, exist_ok=True)

        # Visit config file path for persistence
        visit_config_file = participant_base_dir / "visit_config.json"

        # Store visit config info in session config if provided (new visit)
        if visit_config:
            visit_config_dict = {
                "visit_id": visit_config.visit_id,
                "visit_number": visit_config.visit_number,
                "mask_source": visit_config.mask_source,
                "mask_source_visit": visit_config.mask_source_visit,
                "localizer_enabled": visit_config.localizer_enabled,
                "notes": visit_config.notes,
            }
            self.study_config.session_config["visit"] = visit_config_dict

            # Persist visit config to file
            import json
            with open(visit_config_file, "w") as f:
                json.dump(visit_config_dict, f, indent=2)
            logger.info(f"Saved visit config to {visit_config_file}: mask_source={visit_config.mask_source}")

        elif visit_config_file.exists():
            # Load existing visit config from file (continuing existing visit)
            import json
            with open(visit_config_file, "r") as f:
                visit_config_dict = json.load(f)
            self.study_config.session_config["visit"] = visit_config_dict
            logger.info(f"Loaded visit config from {visit_config_file}: mask_source={visit_config_dict.get('mask_source')}")
        else:
            logger.info(f"No visit config found (visit_config={visit_config}, file_exists={visit_config_file.exists()})")

        # DEBUG: Log what we have in session config
        logger.info(f"DEBUG: session_config['visit'] = {self.study_config.session_config.get('visit', 'NOT SET')}")
        visit_str = f" ({visit_id})" if visit_id else ""
        logger.info(f"Participant data directory{visit_str}: {participant_base_dir}")

        # Mask approval state (must be initialized before _detect_completed_steps)
        self._initial_qc_approved = self._check_qc_approval("initial")

        # Initialize step statuses
        self.step_statuses: Dict[str, StepStatus] = {}
        self._detect_completed_steps()

        # Current active panel (preview or monitoring)
        self.active_panel: Optional[ft.Control] = None

        # Run Next button (will be created in build)
        self.run_next_button: Optional[ft.FilledButton] = None

        # Right panel container (for dynamic content updates)
        self.right_panel_container: Optional[ft.Container] = None

        # Left panel container (for step list updates)
        self.left_panel_container: Optional[ft.Container] = None
        self.step_panel_list: Optional[ft.ExpansionPanelList] = None

        # Pipeline runner for background execution
        self.pipeline_runner: Optional[PipelineRunner] = None

        # Log messages (for log console)
        self.log_messages: List[str] = []
        self.log_listview: Optional[ft.ListView] = None  # For auto-scroll
        self.log_console_container: Optional[ft.Container] = None  # For resizing
        self.log_console_height: float = 150  # Initial height
        self._dragging: bool = False  # Track if currently dragging
        self._drag_start_y: float = 0  # Starting Y position of drag
        self._is_running: bool = False  # Track if a pipeline step is running
        self._current_step_id: Optional[str] = None  # Track current running step

        # File logging for export functionality
        self._log_file_handler: Optional[logging.FileHandler] = None

        # Cache mask reuse mode flag for easy access
        visit_config = self.study_config.session_config.get("visit", {})
        self._is_mask_reuse_mode = visit_config.get("mask_source") == "reuse"
        self._mask_source_visit = visit_config.get("mask_source_visit")
        self._log_file_path: Optional[Path] = None
        self._setup_file_logging()

        # QC report manager for detecting and opening reports
        self.qc_manager = QCReportManager(
            base_dir=participant_base_dir,
            participant_id=participant_id,
        )

        # Selected rest runs for preprocessing (None = use all available)
        self._selected_rest_runs: Optional[List[int]] = None

    def _check_qc_approval(self, approval_type: str) -> bool:
        """Check if masks have been approved.

        Args:
            approval_type: Currently only "initial" is used (after mask extraction)

        Returns:
            True if the approval marker file exists
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        marker_path = base_dir / "mask_extraction" / ".masks_approved"
        return marker_path.exists()

    def _set_qc_approved(self, approval_type: str, approved: bool):
        """Set mask approval state.

        Args:
            approval_type: Currently only "initial" is used
            approved: True to approve, False to revoke
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        marker_path = base_dir / "mask_extraction" / ".masks_approved"

        if approved:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
            logger.info(f"Mask approval marker created: {marker_path}")
        elif marker_path.exists():
            marker_path.unlink()
            logger.info(f"Mask approval marker removed: {marker_path}")

        self._initial_qc_approved = approved

        # Update mask_approval step status
        self.step_statuses["mask_approval"] = StepStatus.COMPLETE if approved else StepStatus.PENDING

    def _has_custom_mask_selections(self) -> bool:
        """Check if user has customized mask selections.

        Returns:
            True if any mask was manually modified from the auto-selected ICA component
        """
        from pineuro.mask_extraction import load_mask_selections

        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        masks_dir = base_dir / "mask_extraction" / "03_masks"

        selections = load_mask_selections(masks_dir)
        if not selections:
            return False

        return any(
            sel.selection_reason in ("manual", "yeo_fallback")
            for sel in selections.values()
        )

    def _show_mask_review_dialog(self):
        """Show the mask review/selection dialog."""
        from glob import glob
        import pandas as pd
        from pineuro.mask_extraction import load_mask_selections, get_alternative_components
        from pineuro import qc

        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"
        masks_dir = mask_extraction_dir / "03_masks"
        ica_dir = mask_extraction_dir / "01_ica"
        qc_dir = base_dir / "qc"

        # Get network names from config (include sham alternatives if enabled)
        config = self.study_config.session_config
        network_names = list(config["mask"]["yeo_targets"])  # Copy to avoid modifying original
        n_yeo = config["mask"]["n_yeo"]
        threshold = config["mask"].get("threshold", 2000)

        # Include sham alternative networks if sham is enabled with alternative masks
        sham_config = config.get("sham", {})
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            alternative_targets = sham_config.get("alternative_mask_targets", [])
            for target in alternative_targets:
                if target and target not in network_names:
                    network_names.append(target)

        # Load correlation matrix
        corr_file = masks_dir / "Yeo_IC_spatial_correlation.csv"
        if not corr_file.exists():
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Correlation matrix not found. Run mask extraction first."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        corr_df = pd.read_csv(corr_file)

        # Load existing selections
        selections = load_mask_selections(masks_dir)

        # Find required files
        components_files = sorted(glob(str(ica_dir / "*_components.nii.gz")))
        if not components_files:
            self.page.snack_bar = ft.SnackBar(
                ft.Text("ICA components not found. Run mask extraction first."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        components_file = Path(components_files[0])

        yeo2sub_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "Yeo_to_*.nii.gz")))
        if not yeo2sub_files:
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Yeo atlas in subject space not found."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return
        yeo_subject_space = Path(yeo2sub_files[0])

        # Generate overlays only for RECOMMENDED (best) components initially
        # Alternative overlays will be generated on-demand when user rejects a mask
        n_components = config["mask"].get("n_components", 20)
        self._add_log("[INFO] Generating overlays for recommended ICA components...")
        self.page.update()

        alternative_overlays = qc.generate_alternative_overlays(
            mask_extraction_dir=mask_extraction_dir,
            participant_id=self.participant_id,
            network_names=network_names,
            n_alternatives=1,  # Only generate overlay for top component per network
            output_dir=qc_dir,
            n_yeo=n_yeo,
        )

        # Yeo overlays will be generated on-demand when user rejects a mask
        yeo_overlays = {}

        # Build MaskReviewInfo for each network
        masks_info = []
        overlays_dir = qc_dir / "overlays" / "subject"

        for network in network_names:
            # Get current selection or defaults
            if network in selections:
                sel = selections[network]
                current_component = sel.component_index if sel.component_index is not None else 0
                current_correlation = sel.correlation
            else:
                current_component = int(corr_df[network].idxmax())
                current_correlation = float(corr_df[network].iloc[current_component])

            # Get all ICA components as alternatives (sorted by correlation)
            n_components = config["mask"].get("n_components", 20)
            alternatives = get_alternative_components(corr_df, network, top_n=n_components)

            # Find overlay paths
            overlay_path = overlays_dir / f"{self.participant_id}_{network}_overlay.png"
            if not overlay_path.exists():
                overlay_path = None

            info = MaskReviewInfo(
                network_name=network,
                current_component=current_component,
                current_correlation=current_correlation,
                alternative_components=alternatives,
                overlay_path=overlay_path,
                alternative_overlay_paths=alternative_overlays.get(network, {}),
                yeo_overlay_path=yeo_overlays.get(network),
            )
            masks_info.append(info)

        def on_complete(new_selections):
            # Set mask approval when user completes review
            self._set_qc_approved("initial", True)
            self._add_log("[SUCCESS] Masks reviewed and approved")
            self._refresh_step_list()
            self._update_run_next_button()
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Masks approved! You can now run coregistration."),
                bgcolor=ft.Colors.GREEN,
            )
            self.page.snack_bar.open = True
            self.page.update()

        # Find preprocessing outputs for script generation
        preprocessing_dir = base_dir / "preprocessing"
        input_file = preprocessing_dir / f"{self.participant_id}_task-rest_preproc.nii.gz"
        brain_mask_file = preprocessing_dir / f"{self.participant_id}_task-rest_preproc_mask.nii.gz"
        ref_file = preprocessing_dir / f"{self.participant_id}_task-rest_boldref.nii.gz"

        # Get ICA parameters from config
        n_components = config["mask"].get("n_components", 20)
        smooth_fwhm = config["mask"].get("smooth_fwhm")
        highpass = config["mask"].get("highpass")
        lowpass = config["mask"].get("lowpass")

        dialog = MaskSelectionDialog(
            page=self.page,
            masks_info=masks_info,
            participant_id=self.participant_id,
            masks_dir=masks_dir,
            components_file=components_file,
            yeo_subject_space=yeo_subject_space,
            n_yeo=n_yeo,
            threshold=threshold,
            on_complete=on_complete,
            on_log=self._add_log,
            # Script generation context
            mask_extraction_dir=mask_extraction_dir,
            input_file=input_file if input_file.exists() else None,
            brain_mask_file=brain_mask_file if brain_mask_file.exists() else None,
            ref_file=ref_file if ref_file.exists() else None,
            yeo_targets=network_names,
            n_components=n_components,
            smooth_fwhm=smooth_fwhm,
            highpass=highpass,
            lowpass=lowpass,
        )
        dialog.show()

    def _check_collection_status(self, collection_dir: Path, expected_volumes: int) -> StepStatus:
        """Check status of a DICOM collection step.

        Args:
            collection_dir: Directory containing collected DICOMs
            expected_volumes: Expected number of volumes

        Returns:
            StepStatus based on collection completeness
        """
        if not collection_dir.exists():
            return StepStatus.PENDING

        dicom_count = len(list(collection_dir.glob("*.dcm")))
        if dicom_count >= expected_volumes:
            return StepStatus.COMPLETE
        elif dicom_count > 0:
            return StepStatus.INTERRUPTED
        else:
            return StepStatus.PENDING

    def _get_rest_run_info(self) -> List[RunInfo]:
        """Get information about available resting-state runs.

        Returns:
            List of RunInfo objects for all configured localizer runs
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        n_localizer_runs = config["scan"].get("localizer_n_runs", 2)

        runs = []

        # Get expected volumes (use new localizer_n_volumes, fall back to legacy)
        expected_vols = config["scan"].get("localizer_n_volumes")
        if expected_vols is None:
            rest_n_volumes = config["scan"].get("rest_n_volumes", [200])
            expected_vols = (
                rest_n_volumes[0]
                if isinstance(rest_n_volumes, list)
                else rest_n_volumes
            )

        # Build run info for each configured localizer run
        for run_num in range(1, n_localizer_runs + 1):
            run_dir = base_dir / "collection" / f"task-rest_run-{run_num:02d}"
            actual_vols = len(get_dicom_files(run_dir)) if run_dir.exists() else 0

            runs.append(
                RunInfo(
                    run_id=run_num,
                    run_name=f"Localizer Run {run_num}",
                    directory=run_dir,
                    expected_volumes=expected_vols,
                    actual_volumes=actual_vols,
                    is_complete=actual_vols >= expected_vols,
                    selected=actual_vols > 0,  # Pre-select if has data
                )
            )

        return runs

    def _needs_run_selection(self) -> bool:
        """Check if run selection dialog should be shown for preprocessing.

        Returns True if:
        - Not all configured runs have data
        - Any run is incomplete
        """
        runs = self._get_rest_run_info()
        runs_with_data = [r for r in runs if r.actual_volumes > 0]

        # No data at all - don't show dialog
        if not runs_with_data:
            return False

        # All configured runs complete - no need for selection
        if len(runs_with_data) == len(runs) and all(r.is_complete for r in runs_with_data):
            return False

        # Show selection if:
        # - Not all runs have data
        # - Any run is incomplete
        return True

    def _show_run_selection_dialog(self, on_selected):
        """Show the run selection dialog.

        Args:
            on_selected: Callback with list of selected run IDs
        """
        runs = self._get_rest_run_info()

        def on_confirm(selected_run_ids: List[int]):
            self._selected_rest_runs = selected_run_ids
            on_selected(selected_run_ids)

        def on_cancel():
            # Return to preview panel
            self._on_preview_cancel()

        dialog = RunSelectionDialog(
            page=self.page,
            runs=runs,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
        )
        dialog.show()

    def _check_feedback_step_status(self, log_path: Path, expected_volumes: int) -> StepStatus:
        """Check status of a feedback/transfer run based on feedbacklog.

        Args:
            log_path: Path to the feedbacklog.tsv file
            expected_volumes: Expected number of volumes

        Returns:
            StepStatus based on run completeness
        """
        if not log_path.exists():
            return StepStatus.PENDING

        # Count data rows (excluding header)
        try:
            with open(log_path, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
        except Exception:
            return StepStatus.PENDING

        if row_count >= expected_volumes:
            return StepStatus.COMPLETE
        elif row_count > 0:
            return StepStatus.INTERRUPTED
        else:
            return StepStatus.PENDING

    def _check_feedback_slot_status(
        self, step_id: str, output_dir: Path, participant_id: str, expected_volumes: int
    ) -> StepStatus:
        """Check status of a feedback/transfer slot using acq-based naming.

        Looks for any acq-* files for the given slot's run number and determines
        if any attempt is complete or if there are interrupted attempts.

        Args:
            step_id: Workflow step ID (e.g., 'transfer_1', 'feedback_2')
            output_dir: Feedback output directory
            participant_id: Participant ID
            expected_volumes: Expected number of volumes

        Returns:
            StepStatus based on best attempt status (COMPLETE if any complete,
            INTERRUPTED if any partial, PENDING if none found)
        """
        if step_id not in STEP_TO_RUN_NUMBER:
            return StepStatus.PENDING

        run_num = STEP_TO_RUN_NUMBER[step_id]
        task_type = "transfer" if "transfer" in step_id else "feedback"

        if not output_dir.exists():
            return StepStatus.PENDING

        # Search for any acq-* files for this slot's run number
        pattern = f"{participant_id}_task-{task_type}_acq-*_run-{run_num:02d}_feedbacklog.tsv"
        log_files = list(output_dir.glob(pattern))

        # Also check legacy naming (without acq) for backward compatibility
        legacy_pattern = f"{participant_id}_task-{task_type}_run-{run_num:02d}_feedbacklog.tsv"
        legacy_file = output_dir / legacy_pattern
        if legacy_file.exists() and legacy_file not in log_files:
            log_files.append(legacy_file)

        if not log_files:
            return StepStatus.PENDING

        # Check each log file and return the best status
        has_complete = False
        has_interrupted = False

        for log_path in log_files:
            try:
                with open(log_path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # subtract header
                if row_count >= expected_volumes:
                    has_complete = True
                elif row_count > 0:
                    has_interrupted = True
            except Exception:
                continue

        # Return best status: complete > interrupted > pending
        if has_complete:
            return StepStatus.COMPLETE
        elif has_interrupted:
            return StepStatus.INTERRUPTED
        else:
            return StepStatus.PENDING

    def _detect_completed_steps(self):
        """Detect which steps are already completed based on output files."""
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        participant_id = self.participant_id

        # Get dynamic configuration values
        n_localizer_runs = config["scan"].get("localizer_n_runs", 2)
        twovol_initial_enabled = config["scan"].get("twovol_initial_enabled", True)
        twovol_realtime_enabled = config["scan"].get("twovol_realtime_enabled", True)

        # Check if this is a mask-reuse visit
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        logger.info(f"DEBUG _detect_completed_steps: visit_config={visit_config}, is_mask_reuse={is_mask_reuse}")

        # Build dynamic steps list
        steps = []

        if is_mask_reuse:
            # Mask reuse mode: only need 2vol + coregistration + feedback
            # Skip localizer, preprocessing, mask extraction
            if twovol_realtime_enabled:
                steps.extend(["2vol_realtime", "coregistration_realtime"])
            steps.extend([
                "transfer_1",
                "feedback_1",
                "feedback_2",
                "transfer_2",
            ])
        else:
            # Standard mode: full pipeline
            if twovol_initial_enabled:
                steps.append("2vol_initial")
            # Add rest runs dynamically
            for i in range(1, n_localizer_runs + 1):
                steps.append(f"rest_run_{i}")
            steps.extend([
                "preprocessing",
                "mask_extraction",
            ])
            # Add coregistration only if initial 2vol is enabled
            if twovol_initial_enabled:
                steps.append("coregistration")
            if twovol_realtime_enabled:
                steps.extend(["2vol_realtime", "coregistration_realtime"])
            steps.extend([
                "transfer_1",
                "feedback_1",
                "feedback_2",
                "transfer_2",
            ])

        for step in steps:
            self.step_statuses[step] = StepStatus.PENDING

        # Only check localizer/preprocessing/mask extraction steps if NOT in mask reuse mode
        if not is_mask_reuse:
            # Check 2vol initial collection (only if enabled)
            if twovol_initial_enabled:
                twovol_initial_dir = base_dir / "collection" / "task-2vol_run-01"
                n_vols = config["scan"].get("twovol_initial_n_volumes", config["scan"].get("twovol_n_volumes", 2))
                status = self._check_collection_status(twovol_initial_dir, n_vols)
                self.step_statuses["2vol_initial"] = status
                if status != StepStatus.PENDING:
                    logger.info(f"Detected {status.value}: 2vol_initial")

            # Check localizer run collections dynamically
            expected_vols = config["scan"].get("localizer_n_volumes")
            if expected_vols is None:
                rest_n_volumes = config["scan"].get("rest_n_volumes", [200])
                expected_vols = rest_n_volumes[0] if isinstance(rest_n_volumes, list) else rest_n_volumes

            for run_num in range(1, n_localizer_runs + 1):
                step_id = f"rest_run_{run_num}"
                run_dir = base_dir / "collection" / f"task-rest_run-{run_num:02d}"
                status = self._check_collection_status(run_dir, expected_vols)
                self.step_statuses[step_id] = status
                if status != StepStatus.PENDING:
                    logger.info(f"Detected {status.value}: {step_id}")

            # Check preprocessing outputs
            preprocessing_dir = base_dir / "preprocessing"
            preproc_file = preprocessing_dir / f"{participant_id}_task-rest_preproc.nii.gz"
            mask_file = preprocessing_dir / f"{participant_id}_task-rest_preproc_mask.nii.gz"
            ref_file = preprocessing_dir / f"{participant_id}_task-rest_boldref.nii.gz"

            if preproc_file.exists() and mask_file.exists() and ref_file.exists():
                self.step_statuses["preprocessing"] = StepStatus.COMPLETE
                logger.info("Detected complete: preprocessing")

            # Check mask extraction outputs
            mask_extraction_dir = base_dir / "mask_extraction"
            masks_dir = mask_extraction_dir / "03_masks"

            # Check for any network mask files
            if masks_dir.exists() and len(list(masks_dir.glob(f"{participant_id}_*.nii.gz"))) > 0:
                self.step_statuses["mask_extraction"] = StepStatus.COMPLETE
                logger.info("Detected complete: mask_extraction")

            # Check mask approval status (based on approval marker file)
            if self._initial_qc_approved:
                self.step_statuses["mask_approval"] = StepStatus.COMPLETE
                logger.info("Detected complete: mask_approval")
            else:
                self.step_statuses["mask_approval"] = StepStatus.PENDING

            # Check coregistration (2vol) outputs - initial (run-01) is in mask_extraction/04_2vol
            initial_twovol_masks_dir = mask_extraction_dir / "04_2vol" / "masks"

            if initial_twovol_masks_dir.exists() and len(list(initial_twovol_masks_dir.glob("*_to_task-2vol_run-01_*.nii.gz"))) > 0:
                self.step_statuses["coregistration"] = StepStatus.COMPLETE
                logger.info("Detected complete: coregistration")

        # Define directories needed for remaining checks
        feedback_dir = base_dir / "feedback"
        if is_mask_reuse:
            mask_extraction_dir = base_dir / "mask_extraction"  # Still needed for path references

        # Check 2vol realtime collection (only if enabled)
        if twovol_realtime_enabled:
            # In mask reuse mode, 2vol is run-01 (no initial 2vol)
            # In standard mode, 2vol is run-02 (after initial 2vol)
            realtime_2vol_run = 1 if is_mask_reuse else 2
            twovol_realtime_dir = base_dir / "collection" / f"task-2vol_run-{realtime_2vol_run:02d}"
            n_vols = config["scan"].get("twovol_realtime_n_volumes", config["scan"].get("twovol_n_volumes", 2))
            status = self._check_collection_status(twovol_realtime_dir, n_vols)
            self.step_statuses["2vol_realtime"] = status
            if status != StepStatus.PENDING:
                logger.info(f"Detected {status.value}: 2vol_realtime")

            # Check coregistration_realtime outputs - realtime is in feedback/01_2vol
            realtime_twovol_masks_dir = feedback_dir / "01_2vol" / "masks"
            if realtime_twovol_masks_dir.exists() and len(list(realtime_twovol_masks_dir.glob(f"*_to_task-2vol_run-{realtime_2vol_run:02d}_*.nii.gz"))) > 0:
                self.step_statuses["coregistration_realtime"] = StepStatus.COMPLETE
                logger.info("Detected complete: coregistration_realtime")
        # Note: When 2vol realtime is disabled, those steps are not in the steps list,
        # so we don't need to mark them as complete

        # Check feedback runs (transfer and feedback) using slot-based naming
        feedback_output_dir = feedback_dir / "02_feedback_output"
        expected_feedback_vols = config["feedback"]["n_volumes"]

        # Check each feedback/transfer step using slot-based detection
        for step_id in ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]:
            status = self._check_feedback_slot_status(
                step_id=step_id,
                output_dir=feedback_output_dir,
                participant_id=participant_id,
                expected_volumes=expected_feedback_vols,
            )
            self.step_statuses[step_id] = status
            if status != StepStatus.PENDING:
                logger.info(f"Detected {status.value}: {step_id}")

        logger.info(f"Session loaded: {self.participant_id}")

    def _get_session_for_next_step(self) -> Optional[str]:
        """Determine which session contains the earliest incomplete step.

        Returns:
            Session name ("localizer", "mask_extraction", "feedback") or None if all complete
        """
        next_step = self._get_next_step()
        if next_step is None:
            return None

        step_id, _ = next_step
        return self._get_session_for_step(step_id)

    def _get_session_for_step(self, step_id: str) -> str:
        """Get the session name for a given step ID.

        Args:
            step_id: Step identifier

        Returns:
            Session name ("localizer", "mask_extraction", "feedback")
        """
        # Handle dynamic rest_run_* steps
        if step_id.startswith("rest_run_"):
            return "localizer"

        step_to_session = {
            "2vol_initial": "localizer",
            "preprocessing": "mask_extraction",
            "mask_extraction": "mask_extraction",
            "mask_approval": "mask_extraction",
            "coregistration": "mask_extraction",
            "2vol_realtime": "feedback",
            "coregistration_realtime": "feedback",
            "transfer_1": "feedback",
            "feedback_1": "feedback",
            "feedback_2": "feedback",
            "transfer_2": "feedback",
        }

        return step_to_session.get(step_id, "feedback")

    def _get_status_icon(self, status: StepStatus) -> ft.Icon:
        """Get icon for a step status.

        Args:
            status: Step status

        Returns:
            Flet Icon control
        """
        icon_map = {
            StepStatus.PENDING: ft.icons.Icons.PAUSE_CIRCLE_OUTLINE,
            StepStatus.RUNNING: ft.icons.Icons.SYNC,
            StepStatus.COMPLETE: ft.icons.Icons.CHECK_CIRCLE,
            StepStatus.INTERRUPTED: ft.icons.Icons.WARNING_AMBER,
            StepStatus.FAILED: ft.icons.Icons.ERROR,
        }

        color_map = {
            StepStatus.PENDING: ft.Colors.GREY_400,
            StepStatus.RUNNING: ft.Colors.BLUE_500,
            StepStatus.COMPLETE: ft.Colors.GREEN_500,
            StepStatus.INTERRUPTED: ft.Colors.ORANGE_500,
            StepStatus.FAILED: ft.Colors.RED_500,
        }

        return ft.Icon(
            icon_map[status],
            color=color_map[status],
            size=20,
        )

    def _create_step_button(
        self,
        step_id: str,
        label: str,
        on_click,
    ) -> ft.Container:
        """Create a button for a pipeline step.

        Args:
            step_id: Step identifier
            label: Button label
            on_click: Click handler

        Returns:
            Container with button and status icon
        """
        status = self.step_statuses.get(step_id, StepStatus.PENDING)

        # In mask reuse mode, preprocessing and mask approval are inherited from source visit
        # so these gates should not apply
        if self._is_mask_reuse_mode:
            requires_preprocessing = False
            requires_approval = False
        else:
            # Gate 1: Preprocessing must be complete before mask extraction and subsequent steps
            preprocessing_complete = self.step_statuses.get("preprocessing") == StepStatus.COMPLETE
            steps_requiring_preprocessing = [
                "mask_extraction", "coregistration", "2vol_realtime", "coregistration_realtime",
                "transfer_1", "feedback_1", "feedback_2", "transfer_2"
            ]
            requires_preprocessing = step_id in steps_requiring_preprocessing and not preprocessing_complete

            # Gate 2: Mask approval required for coregistration and all subsequent steps
            gated_steps = [
                "coregistration", "2vol_realtime", "coregistration_realtime",
                "transfer_1", "feedback_1", "feedback_2", "transfer_2"
            ]
            requires_approval = step_id in gated_steps and not self._initial_qc_approved

        button = ft.ElevatedButton(
            label,
            on_click=on_click,
            disabled=self._is_running or status == StepStatus.RUNNING or requires_preprocessing or requires_approval,
        )

        # Show lock icon if blocked, otherwise show status icon
        if requires_preprocessing or requires_approval:
            icon = ft.Icon(ft.Icons.LOCK, color=ft.Colors.ORANGE_500, size=20)
        else:
            icon = self._get_status_icon(status)

        return ft.Container(
            content=ft.Row(
                [
                    button,
                    icon,
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=5,
        )

    def _build_localizer_session_section(self) -> ft.ExpansionPanel:
        """Build the localizer session section with collapsible steps.

        Returns:
            ExpansionPanel control
        """
        config = self.study_config.session_config
        controls = []

        # Add Initial 2vol section only if enabled
        if config["scan"].get("twovol_initial_enabled", True):
            controls.extend([
                ft.Text("Initial 2vol", size=14, weight=ft.FontWeight.W_500),
                self._create_step_button(
                    "2vol_initial",
                    "Collect",
                    lambda e: self._on_step_clicked("2vol_initial"),
                ),
                ft.Divider(),
            ])

        # Add localizer/rest runs dynamically based on localizer_n_runs
        n_localizer_runs = config["scan"].get("localizer_n_runs", 2)
        controls.append(ft.Text("Resting State", size=14, weight=ft.FontWeight.W_500))

        for run_num in range(1, n_localizer_runs + 1):
            step_id = f"rest_run_{run_num}"
            controls.append(
                self._create_step_button(
                    step_id,
                    f"Collect Run {run_num}",
                    lambda e, sid=step_id: self._on_step_clicked(sid),
                )
            )

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Localizer Session", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(controls, spacing=10),
            expanded=self._get_session_for_next_step() == "localizer",
        )

    def _build_mask_extraction_section(self) -> ft.ExpansionPanel:
        """Build the mask extraction section with collapsible steps.

        Returns:
            ExpansionPanel control
        """
        # Build controls list
        controls = [
            ft.Text("Preprocessing", size=14, weight=ft.FontWeight.W_500),
            self._create_step_button(
                "preprocessing",
                "Run Preprocessing",
                lambda e: self._on_step_clicked("preprocessing"),
            ),
            ft.Divider(),
            ft.Text("Network Masks", size=14, weight=ft.FontWeight.W_500),
            self._create_step_button(
                "mask_extraction",
                "Extract Masks",
                lambda e: self._on_step_clicked("mask_extraction"),
            ),
        ]

        # Add mask review/approval gate after mask extraction completes
        if self.step_statuses.get("mask_extraction") == StepStatus.COMPLETE:
            if self._initial_qc_approved:
                # Approved - show "Review Masks" button with checkmark
                masks_customized = self._has_custom_mask_selections()
                status_text = "Customized" if masks_customized else "Default ICA"
                status_color = ft.Colors.BLUE if masks_customized else ft.Colors.GREY_600

                controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.TextButton(
                                    "Review Masks",
                                    icon=ft.Icons.TUNE,
                                    on_click=lambda e: self._show_mask_review_dialog(),
                                ),
                                ft.Text(f"({status_text})", size=12, color=status_color),
                                ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
                            ],
                            spacing=5,
                        ),
                        padding=ft.padding.only(left=10),
                    )
                )
            else:
                # Not approved - show prominent approval button
                controls.append(
                    ft.Container(
                        content=ft.FilledButton(
                            "Review & Approve Masks",
                            icon=ft.Icons.APPROVAL,
                            bgcolor=ft.Colors.ORANGE,
                            on_click=lambda e: self._show_mask_review_dialog(),
                        ),
                        padding=ft.padding.only(left=10, top=5),
                    )
                )

        # Add coregistration section only if initial 2vol is enabled
        config = self.study_config.session_config
        if config["scan"].get("twovol_initial_enabled", True):
            controls.extend([
                ft.Divider(),
                ft.Text("Coregistration", size=14, weight=ft.FontWeight.W_500),
                self._create_step_button(
                    "coregistration",
                    "Coregister to 2vol",
                    lambda e: self._on_step_clicked("coregistration"),
                ),
            ])

            # Add optional QC view after coregistration if complete (non-essential)
            if self.step_statuses.get("coregistration") == StepStatus.COMPLETE:
                controls.append(
                    ft.Container(
                        content=ft.TextButton(
                            "View Coregistration QC",
                            icon=ft.Icons.ASSESSMENT,
                            on_click=lambda e: self._show_coregistration_qc("initial"),
                        ),
                        padding=ft.padding.only(left=10),
                    )
                )

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Mask Extraction", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(
                controls,
                spacing=10,
            ),
            expanded=self._get_session_for_next_step() == "mask_extraction",
        )

    def _build_feedback_section(self) -> ft.ExpansionPanel:
        """Build the feedback session section.

        Returns:
            ExpansionPanel control
        """
        config = self.study_config.session_config

        # Build controls list
        controls = []

        # Add Real-time 2vol section only if enabled
        if config["scan"].get("twovol_realtime_enabled", True):
            controls.extend([
                ft.Text("Real-time 2vol", size=14, weight=ft.FontWeight.W_500),
                self._create_step_button(
                    "2vol_realtime",
                    "Collect",
                    lambda e: self._on_step_clicked("2vol_realtime"),
                ),
                ft.Divider(),
                ft.Text("Coregistration", size=14, weight=ft.FontWeight.W_500),
                self._create_step_button(
                    "coregistration_realtime",
                    "Coregister",
                    lambda e: self._on_step_clicked("coregistration_realtime"),
                ),
            ])

            # Add optional QC view after coregistration_realtime if complete (non-essential)
            if self.step_statuses.get("coregistration_realtime") == StepStatus.COMPLETE:
                controls.append(
                    ft.Container(
                        content=ft.TextButton(
                            "View Coregistration QC",
                            icon=ft.Icons.ASSESSMENT,
                            on_click=lambda e: self._show_coregistration_qc("realtime"),
                        ),
                        padding=ft.padding.only(left=10),
                    )
                )

        # Add feedback runs section
        controls.extend([
            ft.Divider(),
            ft.Text("Feedback Runs", size=14, weight=ft.FontWeight.W_500),
            self._create_step_button(
                "transfer_1",
                "Transfer Run 1",
                lambda e: self._on_step_clicked("transfer_1"),
            ),
            self._create_step_button(
                "feedback_1",
                "Feedback Run 1",
                lambda e: self._on_step_clicked("feedback_1"),
            ),
            self._create_step_button(
                "feedback_2",
                "Feedback Run 2",
                lambda e: self._on_step_clicked("feedback_2"),
            ),
            self._create_step_button(
                "transfer_2",
                "Transfer Run 2",
                lambda e: self._on_step_clicked("transfer_2"),
            ),
        ])

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Feedback Session", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(
                controls,
                spacing=10,
            ),
            expanded=self._get_session_for_next_step() == "feedback",
        )

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes.

        Args:
            path: Directory path to measure

        Returns:
            Total size in bytes, or 0 if directory doesn't exist
        """
        if not path.exists():
            return 0
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except (PermissionError, OSError) as e:
            logger.warning(f"Error calculating size for {path}: {e}")
        return total

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human-readable string.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string (e.g., "2.5 GB", "500 MB")
        """
        if size_bytes >= 1024 ** 3:
            return f"{size_bytes / (1024 ** 3):.1f} GB"
        elif size_bytes >= 1024 ** 2:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes} B"

    def _check_cleanup_available(self) -> dict:
        """Check what can be cleaned up.

        Returns:
            dict with keys:
            - collection_exists: bool
            - collection_size: int (bytes)
            - collection_subdirs: list of subdir names
            - total_size: int (bytes)
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        result = {
            "collection_exists": False,
            "collection_size": 0,
            "collection_subdirs": [],
            "total_size": 0,
        }

        if collection_dir.exists():
            subdirs = [d.name for d in collection_dir.iterdir() if d.is_dir()]
            if subdirs:
                result["collection_exists"] = True
                result["collection_subdirs"] = sorted(subdirs)
                result["collection_size"] = self._get_dir_size(collection_dir)

        result["total_size"] = result["collection_size"]
        return result

    def _build_utilities_section(self) -> ft.ExpansionPanel:
        """Build the session utilities section with cleanup step.

        Returns:
            ExpansionPanel control
        """
        cleanup_info = self._check_cleanup_available()

        if cleanup_info["total_size"] > 0:
            status_text = f"Available ({self._format_size(cleanup_info['total_size'])})"
            status_color = ft.Colors.ORANGE
            icon = ft.Icons.DELETE_SWEEP
        else:
            status_text = "Nothing to clean"
            status_color = ft.Colors.GREY
            icon = ft.Icons.CHECK_CIRCLE

        cleanup_button = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(icon, color=status_color, size=20),
                    ft.TextButton(
                        "Cleanup Session",
                        on_click=lambda e: self._on_cleanup_clicked(),
                        disabled=self._is_running,
                    ),
                    ft.Text(status_text, size=12, color=status_color),
                ],
                spacing=5,
            ),
            padding=ft.padding.only(left=10),
        )

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Session Utilities", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(
                [cleanup_button],
                spacing=10,
            ),
            expanded=False,
        )

    def _on_cleanup_clicked(self):
        """Handle click on cleanup step - show preview panel."""
        self._show_cleanup_preview_panel()

    def _show_cleanup_preview_panel(self):
        """Show the cleanup preview in the right panel."""
        cleanup_info = self._check_cleanup_available()

        if cleanup_info["total_size"] == 0:
            # Nothing to clean
            content = ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(ft.Icons.CHECK_CIRCLE, size=64, color=ft.Colors.GREEN),
                        ft.Text("Nothing to Clean", size=20, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            "No temporary files found. The collection directory is empty or doesn't exist.",
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
        else:
            # Build list of directories to delete
            dir_items = []
            if cleanup_info["collection_exists"]:
                dir_items.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Icon(ft.Icons.FOLDER, color=ft.Colors.AMBER),
                                        ft.Text("collection/", weight=ft.FontWeight.BOLD),
                                        ft.Text(
                                            f"({self._format_size(cleanup_info['collection_size'])})",
                                            color=ft.Colors.GREY_600,
                                        ),
                                    ],
                                    spacing=10,
                                ),
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text(f"  - {subdir}", size=12, color=ft.Colors.GREY_700)
                                            for subdir in cleanup_info["collection_subdirs"][:6]
                                        ] + (
                                            [ft.Text(f"  ... and {len(cleanup_info['collection_subdirs']) - 6} more", size=12, color=ft.Colors.GREY_500)]
                                            if len(cleanup_info["collection_subdirs"]) > 6 else []
                                        ),
                                        spacing=2,
                                    ),
                                    padding=ft.padding.only(left=30),
                                ),
                            ],
                            spacing=5,
                        ),
                        padding=10,
                        bgcolor=ft.Colors.GREY_100,
                        border_radius=5,
                    )
                )

            content = ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Session Cleanup", size=20, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text("The following directories will be permanently deleted:"),
                        ft.Container(height=10),
                        *dir_items,
                        ft.Container(height=10),
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Text("Total:", weight=ft.FontWeight.BOLD),
                                    ft.Text(self._format_size(cleanup_info["total_size"])),
                                ],
                                spacing=10,
                            ),
                            padding=10,
                            bgcolor=ft.Colors.BLUE_50,
                            border_radius=5,
                        ),
                        ft.Container(height=20),
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Icon(ft.Icons.WARNING, color=ft.Colors.ORANGE),
                                    ft.Text(
                                        "Warning: This action cannot be undone.",
                                        color=ft.Colors.ORANGE_900,
                                        weight=ft.FontWeight.W_500,
                                    ),
                                ],
                                spacing=10,
                            ),
                            padding=10,
                            bgcolor=ft.Colors.ORANGE_50,
                            border_radius=5,
                        ),
                        ft.Container(height=20),
                        ft.Row(
                            [
                                ft.OutlinedButton(
                                    "Cancel",
                                    on_click=lambda e: self._show_default_preview(),
                                ),
                                ft.FilledButton(
                                    "Run Cleanup",
                                    icon=ft.Icons.DELETE_SWEEP,
                                    bgcolor=ft.Colors.RED,
                                    on_click=lambda e: self._show_cleanup_confirmation(),
                                    disabled=self._is_running,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                            spacing=10,
                        ),
                    ],
                    spacing=5,
                ),
                padding=20,
            )

        if self.right_panel_container:
            self.right_panel_container.content = content
            self.right_panel_container.update()

    def _show_default_preview(self):
        """Show the default preview panel."""
        if self.right_panel_container:
            self.right_panel_container.content = self._build_preview_panel()
            self.right_panel_container.update()

    def _show_cleanup_confirmation(self):
        """Show confirmation dialog before cleanup."""
        cleanup_info = self._check_cleanup_available()

        def on_confirm(e):
            dialog.open = False
            self.page.update()
            self._run_cleanup()

        def on_cancel(e):
            dialog.open = False
            self.page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Confirm Cleanup"),
            content=ft.Text(
                f"Are you sure you want to delete {self._format_size(cleanup_info['total_size'])} "
                f"of temporary files?\n\nThis action cannot be undone."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=on_cancel),
                ft.FilledButton(
                    "Delete",
                    bgcolor=ft.Colors.RED,
                    on_click=on_confirm,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _run_cleanup(self):
        """Execute cleanup - delete collection directory."""
        import shutil

        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        self._add_log("[INFO] Starting cleanup...")

        try:
            if collection_dir.exists():
                shutil.rmtree(collection_dir)
                self._add_log(f"[SUCCESS] Deleted: {collection_dir}")
                logger.info(f"Cleanup: deleted {collection_dir}")

            self._add_log("[SUCCESS] Cleanup complete")

            # Show success snackbar
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Cleanup complete. Temporary files have been deleted."),
                bgcolor=ft.Colors.GREEN,
                open=True,
            )
            self.page.update()

            # Refresh the UI - step list first (to update utilities section status)
            self._refresh_step_list()
            # Show default preview panel (cleanup complete state)
            self._show_default_preview()

        except Exception as e:
            self._add_log(f"[ERROR] Cleanup failed: {e}")
            logger.error(f"Cleanup failed: {e}")

            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Cleanup failed: {e}"),
                bgcolor=ft.Colors.RED,
                open=True,
            )
            self.page.update()

    def _show_qc_approval_dialog(self, approval_type: str):
        """Show mandatory QC approval dialog after coregistration.

        Args:
            approval_type: "initial" (after coregistration) or "realtime" (after coregistration_realtime)
        """
        from ..components.qc_plot_viewer import QCPlotViewer

        logger.info(f"Opening QC approval dialog: {approval_type}")

        # Get appropriate plots based on approval type
        # Initial: show ICA, subject space, and 2vol overlays (full mask extraction review)
        # Realtime: show ONLY 2vol overlays (since ICA and subject space didn't change)
        if approval_type == "initial":
            plots = self.qc_manager.get_mask_extraction_plots()
            title = "Initial 2vol QC - Review Required"
            success_msg = "Initial coregistration approved! You can now collect real-time 2vol."
        else:
            # In mask reuse mode, realtime 2vol is run-01 (no initial 2vol collection)
            # In standard mode, realtime 2vol is run-02
            run_id = 1 if self._is_mask_reuse_mode else 2
            plots = self.qc_manager.get_2vol_qc_plots(run_id=run_id)
            title = "Real-time 2vol QC - Review Required"
            success_msg = "Real-time coregistration approved! You can now run feedback."

        html_path = self.qc_manager.get_mask_extraction_qc()

        def on_approve():
            self._set_qc_approved(approval_type, True)
            self._add_log(f"[SUCCESS] {title.split(' -')[0]} approved")
            self._refresh_step_list()
            self.page.snack_bar = ft.SnackBar(
                ft.Text(success_msg),
                bgcolor=ft.Colors.GREEN,
                open=True,
            )
            self.page.update()

        def on_reject():
            self._add_log(f"[WARNING] {title.split(' -')[0]} rejected")
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Masks rejected. Please review and re-run coregistration."),
                bgcolor=ft.Colors.ORANGE,
                open=True,
            )
            self.page.update()

        viewer = QCPlotViewer(
            page=self.page,
            title=title,
            plots=plots,
            html_report_path=html_path,
            require_approval=True,
            on_approve=on_approve,
            on_reject=on_reject,
        )
        viewer.show()

    def _show_coregistration_qc(self, qc_type: str):
        """Show coregistration QC viewer (non-essential, view-only).

        Args:
            qc_type: "initial" (after coregistration) or "realtime" (after coregistration_realtime)
        """
        from ..components.qc_plot_viewer import QCPlotViewer

        logger.info(f"Opening coregistration QC viewer: {qc_type}")

        # Get 2vol overlay plots - use run_id=1 for initial, run_id=2 for realtime
        # In mask reuse mode, realtime 2vol is run-01 (no initial 2vol collection)
        if qc_type == "initial":
            run_id = 1
            title = "Initial 2vol Coregistration QC"
        else:
            # In mask reuse mode, realtime 2vol is run-01
            run_id = 1 if self._is_mask_reuse_mode else 2
            title = "Real-time 2vol Coregistration QC"

        plots = self.qc_manager.get_2vol_qc_plots(run_id=run_id)
        html_path = self.qc_manager.get_mask_extraction_qc()

        viewer = QCPlotViewer(
            page=self.page,
            title=title,
            plots=plots,
            html_report_path=html_path,
            require_approval=False,  # View-only, no approval needed
        )
        viewer.show()

    def _build_log_console(self) -> ft.Column:
        """Build the log console component with draggable divider.

        Returns:
            Column with divider and log viewer
        """
        # Create ListView for log entries with auto_scroll
        initial_log = ft.Text(
            "[INFO] Session initialized",
            size=12,
            font_family="monospace",
            selectable=True,
            color=ft.Colors.WHITE,
        )
        self.log_listview = ft.ListView(
            controls=[initial_log],
            spacing=2,
            auto_scroll=True,
        )

        # Store reference to the scrollable container for resizing
        self.log_console_container = ft.Container(
            content=self.log_listview,
            height=self.log_console_height,
            bgcolor=ft.Colors.GREY_900,
            border_radius=5,
            padding=10,
        )

        # Create draggable divider using Container with drag events
        drag_divider = ft.Container(
            content=ft.Row(
                [ft.Icon(ft.icons.Icons.DRAG_HANDLE, size=20, color=ft.Colors.GREY_600)],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            height=8,
            bgcolor=ft.Colors.GREY_300,
            on_hover=lambda e: None,  # Enable mouse tracking
        )

        # Wrap in GestureDetector for drag functionality
        drag_divider_detector = ft.GestureDetector(
            content=drag_divider,
            drag_interval=10,  # Update every 10ms during drag
            on_vertical_drag_start=self._on_divider_drag_start,
            on_vertical_drag_update=self._on_divider_drag_update,
            on_vertical_drag_end=self._on_divider_drag_end,
            mouse_cursor=ft.MouseCursor.RESIZE_UP_DOWN,
        )

        # Log console with header
        log_panel = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("Log Console", weight=ft.FontWeight.BOLD),
                            ft.TextButton(
                                "Clear",
                                icon=ft.icons.Icons.CLEAR,
                                on_click=self._clear_log,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.log_console_container,
                ],
                spacing=5,
            ),
            padding=10,
            bgcolor=ft.Colors.GREY_100,
        )

        return ft.Column(
            [
                drag_divider_detector,
                log_panel,
            ],
            spacing=0,
        )

    def _build_preview_panel(self) -> ft.Container:
        """Build the empty parameter preview panel.

        Returns:
            Container with placeholder message
        """
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "Parameter Preview / Monitoring",
                        size=16,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Container(height=20),
                    ft.Text(
                        "Click a step button to preview parameters",
                        size=14,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=20,
            alignment=ft.alignment.Alignment.CENTER,
            expand=True,
        )

    def _rebuild_header(self):
        """Rebuild the header to update the Run Next button."""
        logger.info("=== REBUILDING HEADER ===")

        # Determine button state based on next step and running state
        next_step = self._get_next_step()
        if next_step is None:
            button_text = "Session Complete"
            button_disabled = True
            button_icon = ft.icons.Icons.CHECK_CIRCLE
            logger.info("Run Next: Session Complete")
        else:
            step_id, step_label = next_step
            button_text = f"Run Next: {step_label}"
            button_disabled = self._is_running  # Disable when running
            button_icon = ft.icons.Icons.PLAY_ARROW
            logger.info(f"Run Next: {step_label}")

        # Recreate the button with new properties
        self.run_next_button = ft.FilledButton(
            button_text,
            icon=button_icon,
            on_click=lambda e: self._on_run_next_clicked(),
            disabled=button_disabled,
        )

        # Rebuild the header container's content
        self.header_container.content = ft.Row(
            [
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.icons.Icons.ARROW_BACK,
                            on_click=self._on_back,
                            tooltip="Back to Studies",
                            disabled=self._is_running,  # Disable when running
                        ),
                        ft.Text(
                            f"pineuro: {self.study_config.metadata.name}",
                            size=20,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Text(
                            f"| Participant: {self.participant_id}{self._simulator_text}",
                            size=16,
                            color=ft.Colors.GREY_700,
                        ),
                    ],
                    spacing=10,
                ),
                self.run_next_button,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        # Update the header container
        self.header_container.update()
        logger.info("Header container updated")

    def _build_step_preview(self, step_id: str) -> ft.Container:
        """Build parameter preview panel for a specific step.

        Args:
            step_id: Step identifier

        Returns:
            Container with step details and start button
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        participant_id = self.participant_id

        # Step metadata
        step_info = self._get_step_info(step_id, config, base_dir, participant_id)

        # Build parameter list
        param_rows = []
        for key, value in step_info["parameters"].items():
            param_rows.append(
                ft.Row(
                    [
                        ft.Text(f"{key}:", size=13, weight=ft.FontWeight.W_500, width=150),
                        ft.Text(str(value), size=13, color=ft.Colors.GREY_700),
                    ],
                    spacing=10,
                )
            )

        # Check if outputs exist (warning about existing data)
        outputs_exist = step_info.get("outputs_exist", False)
        overwrite_warning = None
        feedback_steps = ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]
        if outputs_exist:
            # For feedback steps, data is archived not overwritten
            if step_id in feedback_steps:
                warning_text = "Previous run data exists and will be archived before starting fresh."
                warning_icon = ft.icons.Icons.ARCHIVE
            else:
                warning_text = "Output files already exist. Running this step will overwrite them."
                warning_icon = ft.icons.Icons.WARNING

            overwrite_warning = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(warning_icon, color=ft.Colors.ORANGE_700, size=20),
                        ft.Text(
                            warning_text,
                            size=12,
                            color=ft.Colors.ORANGE_700,
                        ),
                    ],
                    spacing=10,
                ),
                padding=10,
                bgcolor=ft.Colors.ORANGE_50,
                border_radius=5,
            )

        # Build preview content
        content_items = [
            ft.Text(
                step_info["label"],
                size=18,
                weight=ft.FontWeight.BOLD,
            ),
            ft.Divider(),
            ft.Text(
                step_info["description"],
                size=13,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            ft.Container(height=10),
            ft.Text("Parameters:", size=14, weight=ft.FontWeight.W_500),
            ft.Container(
                content=ft.Column(param_rows, spacing=5),
                padding=ft.padding.only(left=10),
            ),
        ]

        if overwrite_warning:
            content_items.extend([
                ft.Container(height=10),
                overwrite_warning,
            ])

        content_items.extend([
            ft.Container(height=20),
            ft.Row(
                [
                    ft.TextButton("Cancel", on_click=lambda e: self._on_preview_cancel()),
                    ft.FilledButton(
                        "Start",
                        icon=ft.icons.Icons.PLAY_ARROW,
                        on_click=lambda e: self._on_preview_start(step_id),
                    ),
                ],
                alignment=ft.MainAxisAlignment.END,
                spacing=10,
            ),
        ])

        return ft.Container(
            content=ft.Column(
                content_items,
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
            expand=True,
        )

    def _get_all_network_targets(self, config: dict) -> list:
        """Get combined list of network targets including sham alternatives.

        Args:
            config: Session configuration

        Returns:
            List of all network names to extract
        """
        network_names = list(config["mask"]["yeo_targets"])

        # Include sham alternative networks if sham is enabled with alternative masks
        sham_config = config.get("sham", {})
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            alternative_targets = sham_config.get("alternative_mask_targets", [])
            for target in alternative_targets:
                if target and target not in network_names:
                    network_names.append(target)

        return network_names

    def _get_mask_extraction_step_info(self, config: dict, base_dir: Path) -> dict:
        """Get step info for mask extraction, including sham networks if enabled.

        Args:
            config: Session configuration
            base_dir: Base directory path

        Returns:
            Dictionary with step information
        """
        all_networks = self._get_all_network_targets(config)

        return {
            "label": "Extract Network Masks",
            "description": f"ICA-based extraction of {len(all_networks)} network masks using {config['mask']['n_yeo']}-network Yeo atlas.",
            "parameters": {
                "Networks": ", ".join(all_networks),
                "Yeo Atlas": f"{config['mask']['n_yeo']} networks",
                "Smoothing": f"{config['mask']['smooth_fwhm']} mm FWHM",
                "Highpass": f"{config['mask']['highpass']} Hz",
                "Output": base_dir / "mask_extraction",
            },
            "output_path": base_dir / "mask_extraction" / "03_masks",
        }

    def _get_coregistration_realtime_step_info(self, config: dict, base_dir: Path) -> dict:
        """Get step info for realtime coregistration, handling mask reuse mode.

        Args:
            config: Session configuration
            base_dir: Base directory path

        Returns:
            Dictionary with step information
        """
        # Check for mask reuse mode
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        mask_source_visit = visit_config.get("mask_source_visit")

        # In mask reuse mode, 2vol is run-01 (no initial 2vol)
        run_num = 1 if is_mask_reuse else 2

        if is_mask_reuse and mask_source_visit:
            # Mask reuse mode: masks come from source visit
            study_base_dir = base_dir.parent.parent  # Go up from participant/visit to study
            source_visit_dir = study_base_dir / self.participant_id / mask_source_visit
            input_masks_path = source_visit_dir / "mask_extraction" / "03_masks"
            description = f"Register network masks from {mask_source_visit} to real-time 2vol space for feedback runs."
        else:
            # Standard mode: masks come from current visit
            input_masks_path = base_dir / "mask_extraction" / "03_masks"
            description = "Register network masks to real-time 2vol space for feedback runs."

        return {
            "label": "Coregister to Real-time 2vol",
            "description": description,
            "parameters": {
                "Input Masks": input_masks_path,
                "Target": f"2vol run-{run_num:02d}",
                "Output": base_dir / "feedback" / "01_2vol" / "masks",
            },
            "output_path": base_dir / "feedback" / "01_2vol" / "masks",
            "output_pattern": f"*_to_task-2vol_run-{run_num:02d}_*.nii.gz",
        }

    def _get_2vol_realtime_step_info(self, config: dict, base_dir: Path) -> dict:
        """Get step info for realtime 2vol collection, handling mask reuse mode.

        Args:
            config: Session configuration
            base_dir: Base directory path

        Returns:
            Dictionary with step information
        """
        twovol_enabled = config["scan"].get("twovol_realtime_enabled", True)

        # Check for mask reuse mode
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"

        # In mask reuse mode, 2vol is run-01 (no initial 2vol)
        run_num = 1 if is_mask_reuse else 2

        if not twovol_enabled:
            return {
                "label": "Collect Real-time 2vol (Disabled)",
                "description": "Real-time 2vol collection is disabled. Feedback will use masks from subject boldref space.",
                "parameters": {"Status": "Disabled - using boldref masks"},
                "output_path": base_dir / "collection" / f"task-2vol_run-{run_num:02d}",
            }

        return {
            "label": "Collect Real-time 2vol",
            "description": "Collect 2 volumes immediately before feedback session for real-time registration verification.",
            "parameters": {
                "Task": f"2vol (run-{run_num:02d})",
                "Volumes": config["scan"].get("twovol_realtime_n_volumes", config["scan"].get("twovol_n_volumes", 2)),
                "Source": config["scan"]["twovol_realtime_dicoms"],
                "Output": base_dir / "collection" / f"task-2vol_run-{run_num:02d}",
                "Enabled": "Yes",
            },
            "output_path": base_dir / "collection" / f"task-2vol_run-{run_num:02d}",
        }

    def _get_step_info(self, step_id: str, config: dict, base_dir: Path, participant_id: str) -> dict:
        """Get information about a pipeline step for preview.

        Args:
            step_id: Step identifier
            config: Session configuration
            base_dir: Base directory path
            participant_id: Participant ID

        Returns:
            Dictionary with step label, description, parameters, and outputs_exist flag
        """
        # Map step IDs to their information
        step_info_map = {
            "2vol_initial": {
                "label": "Collect Initial 2vol" + ("" if config["scan"].get("twovol_initial_enabled", True) else " (Disabled)"),
                "description": "Collect 2 volumes for initial positioning verification before localizer acquisition." if config["scan"].get("twovol_initial_enabled", True) else "Initial 2vol collection is disabled in study configuration.",
                "parameters": {
                    "Task": "2vol (run-01)",
                    "Volumes": config["scan"].get("twovol_initial_n_volumes", config["scan"].get("twovol_n_volumes", 2)),
                    "Source": config["scan"]["twovol_initial_dicoms"],
                    "Output": base_dir / "collection" / "task-2vol_run-01",
                    "Enabled": "Yes" if config["scan"].get("twovol_initial_enabled", True) else "No",
                } if config["scan"].get("twovol_initial_enabled", True) else {"Status": "Disabled"},
                "output_path": base_dir / "collection" / "task-2vol_run-01",
            },
            "preprocessing": {
                "label": "Run Preprocessing",
                "description": "Motion correction, brain extraction, and coregistration of resting-state runs.",
                "parameters": {
                    "Input": base_dir / "collection" / "task-rest_*",
                    "Output": base_dir / "preprocessing",
                    "Expected Outputs": "preproc.nii.gz, mask.nii.gz, boldref.nii.gz",
                },
                "output_path": base_dir / "preprocessing" / f"{participant_id}_task-rest_preproc.nii.gz",
            },
            "mask_extraction": self._get_mask_extraction_step_info(config, base_dir),
            "coregistration": {
                "label": "Coregister to 2vol",
                "description": "Register network masks from rest space to 2vol feedback space.",
                "parameters": {
                    "Input Masks": base_dir / "mask_extraction" / "03_masks",
                    "Target": "2vol run-01",
                    "Output": base_dir / "mask_extraction" / "04_2vol" / "masks",
                },
                "output_path": base_dir / "mask_extraction" / "04_2vol" / "masks",
                "output_pattern": "*_to_task-2vol_run-01_*.nii.gz",
            },
            "2vol_realtime": self._get_2vol_realtime_step_info(config, base_dir),
            "coregistration_realtime": self._get_coregistration_realtime_step_info(config, base_dir),
            "transfer_1": {
                "label": "Transfer Run 1",
                "description": "Practice run without neurofeedback to establish baseline performance.",
                "parameters": {
                    "Task": "transfer (run-01)",
                    "Volumes": config["feedback"]["n_volumes"],
                    "Baseline": config["feedback"]["n_baseline"],
                    "Feedback": "OFF",
                    "Output": base_dir / "feedback" / "02_feedback_output",
                    "Simulator Speed": f"{config['scan'].get('feedback_simulator_speed', config['scan'].get('collection_speed', 10.0))}x",
                },
                "output_path": base_dir / "feedback" / "02_feedback_output" / f"{participant_id}_task-transfer_run-01_feedbacklog.tsv",
            },
            "feedback_1": {
                "label": "Feedback Run 1",
                "description": "First neurofeedback run with real-time activation feedback.",
                "parameters": {
                    "Task": "feedback (run-01)",
                    "Volumes": config["feedback"]["n_volumes"],
                    "Baseline": config["feedback"]["n_baseline"],
                    "Feedback": "ON",
                    "Networks": ", ".join(config['mask']['mask_names']),
                    "Output": base_dir / "feedback" / "02_feedback_output",
                    "Simulator Speed": f"{config['scan'].get('feedback_simulator_speed', config['scan'].get('collection_speed', 10.0))}x",
                },
                "output_path": base_dir / "feedback" / "02_feedback_output" / f"{participant_id}_task-feedback_run-01_feedbacklog.tsv",
            },
            "feedback_2": {
                "label": "Feedback Run 2",
                "description": "Second neurofeedback run with real-time activation feedback.",
                "parameters": {
                    "Task": "feedback (run-02)",
                    "Volumes": config["feedback"]["n_volumes"],
                    "Baseline": config["feedback"]["n_baseline"],
                    "Feedback": "ON",
                    "Networks": ", ".join(config['mask']['mask_names']),
                    "Output": base_dir / "feedback" / "02_feedback_output",
                    "Simulator Speed": f"{config['scan'].get('feedback_simulator_speed', config['scan'].get('collection_speed', 10.0))}x",
                },
                "output_path": base_dir / "feedback" / "02_feedback_output" / f"{participant_id}_task-feedback_run-02_feedbacklog.tsv",
            },
            "transfer_2": {
                "label": "Transfer Run 2",
                "description": "Final practice run without feedback to assess learning transfer.",
                "parameters": {
                    "Task": "transfer (run-02)",
                    "Volumes": config["feedback"]["n_volumes"],
                    "Baseline": config["feedback"]["n_baseline"],
                    "Feedback": "OFF",
                    "Output": base_dir / "feedback" / "02_feedback_output",
                    "Simulator Speed": f"{config['scan'].get('feedback_simulator_speed', config['scan'].get('collection_speed', 10.0))}x",
                },
                "output_path": base_dir / "feedback" / "02_feedback_output" / f"{participant_id}_task-transfer_run-02_feedbacklog.tsv",
            },
        }

        # Handle dynamic rest_run_* steps
        if step_id.startswith("rest_run_"):
            run_num = int(step_id.split("_")[-1])
            run_idx = run_num - 1  # 0-indexed for accessing lists

            localizer_dicoms = config["scan"].get("localizer_source_dicoms") or config["scan"].get("rest_source_dicoms", [])
            source_dicom = localizer_dicoms[run_idx] if run_idx < len(localizer_dicoms) else ""

            info = {
                "label": f"Collect Localizer Run {run_num}",
                "description": f"Collect localizer fMRI data (run {run_num}) for network mask extraction.",
                "parameters": {
                    "Task": f"{config['scan'].get('localizer_task_type', 'rest')} (run-{run_num:02d})",
                    "Volumes": config["scan"].get("localizer_n_volumes") or 200,
                    "Source": source_dicom,
                    "Output": base_dir / "collection" / f"task-rest_run-{run_num:02d}",
                    "Simulator Speed": f"{config['scan'].get('collection_speed', config['scan'].get('simulator_speed', 10.0))}x",
                },
                "output_path": base_dir / "collection" / f"task-rest_run-{run_num:02d}",
            }
        else:
            info = step_info_map.get(step_id, {
                "label": step_id,
                "description": "Unknown step",
                "parameters": {},
                "output_path": None,
            })

        # For feedback/transfer steps, show slot-based run info with acq entity
        feedback_steps = ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]
        if step_id in feedback_steps:
            task_type = "transfer" if "transfer" in step_id else "feedback"
            output_dir = base_dir / "feedback" / "02_feedback_output"

            # Get slot-based run info with acquisition letter
            run_info = get_slot_run_info(
                output_dir=output_dir,
                subject=participant_id,
                task=task_type,
                step_id=step_id,
            )

            # Update parameters to show run number and acquisition attempt
            info["parameters"]["Task"] = f"{task_type} (run-{run_info['run']:02d})"
            info["parameters"]["Attempt"] = f"acq-{run_info['acq']:02d}"

            # If there are previous attempts, show them
            if run_info["existing_acqs"]:
                attempts_str = ", ".join(f"acq-{a:02d}" for a in run_info["existing_acqs"])
                info["parameters"]["Previous Attempts"] = attempts_str
                info["description"] = info["description"] + f" Previous attempts ({attempts_str}) will be preserved."

        # Check if outputs exist
        output_path = info.get("output_path")
        output_pattern = info.get("output_pattern")

        if output_pattern and output_path:
            # Use pattern-based check (for steps with multiple output files)
            output_dir = Path(output_path)
            if output_dir.exists() and len(list(output_dir.glob(output_pattern))) > 0:
                info["outputs_exist"] = True
            else:
                info["outputs_exist"] = False
        elif output_path and Path(output_path).exists():
            info["outputs_exist"] = True
        else:
            info["outputs_exist"] = False

        return info

    def _on_preview_cancel(self):
        """Handle cancel button in preview panel."""
        logger.info("Preview cancelled")

        # Restore empty preview panel
        if self.right_panel_container is not None:
            self.right_panel_container.content = self._build_preview_panel()
            self.right_panel_container.update()

    def _on_preview_start(self, step_id: str):
        """Handle start button in preview panel.

        Args:
            step_id: Step identifier to execute
        """
        # For preprocessing, check if run selection is needed
        if step_id == "preprocessing" and self._needs_run_selection():
            # Show run selection dialog first
            def on_runs_selected(selected_runs: List[int]):
                self._add_log(f"Selected runs for preprocessing: {selected_runs}")
                self._start_step(step_id)

            self._show_run_selection_dialog(on_runs_selected)
            return

        # Clear any previous selection for non-preprocessing steps
        if step_id != "preprocessing":
            self._selected_rest_runs = None

        self._start_step(step_id)

    def _start_step(self, step_id: str):
        """Actually start executing a pipeline step.

        This is called after any pre-step dialogs (like run selection) are completed.

        Args:
            step_id: Step identifier to execute
        """
        logger.info(f"Starting step: {step_id}")

        # Set running state and lock UI
        self._is_running = True
        self._update_ui_lock_state()

        # Update step status to running
        self.step_statuses[step_id] = StepStatus.RUNNING
        self._add_log(f"Starting {step_id}...")

        # Create pipeline runner if not exists
        if self.pipeline_runner is None:
            self.pipeline_runner = PipelineRunner(
                config=self.study_config.session_config,
                participant_id=self.participant_id,
                on_progress=self._on_pipeline_progress,
                on_complete=self._on_pipeline_complete,
                on_error=self._on_pipeline_error,
                on_collection_progress=self._on_collection_progress,
            )

        # Pass selected runs to runner for preprocessing step
        if step_id == "preprocessing":
            self.pipeline_runner.set_selected_rest_runs(self._selected_rest_runs)
        else:
            self.pipeline_runner.set_selected_rest_runs(None)

        # Show monitoring panel (creates task_mirror for feedback steps)
        self._show_monitoring_panel(step_id)

        # Connect task mirror callback for feedback/transfer steps
        feedback_steps = ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]
        if step_id in feedback_steps:
            self.pipeline_runner.set_task_mirror_callback(self._on_task_mirror_update)
        else:
            self.pipeline_runner.set_task_mirror_callback(None)

        # Execute the step in background thread
        self.pipeline_runner.run_step(step_id)

    def _on_pipeline_progress(self, message: str):
        """Handle progress updates from pipeline runner.

        Args:
            message: Progress message
        """
        # This callback runs in the background thread
        async def update_ui():
            self._add_log(f"[PROGRESS] {message}")
            self.page.update()

        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Failed to schedule progress update: {e}")

    def _on_pipeline_complete(self, success: bool, result_message: Optional[str]):
        """Handle pipeline step completion.

        Args:
            success: Whether step completed successfully
            result_message: Result or error message
        """
        # Capture current step ID before it gets cleared
        completed_step_id = self._current_step_id

        # This callback runs in the background thread
        # We need to schedule the UI update on the main thread
        async def update_ui():
            # Clear running state and unlock UI
            self._is_running = False

            if success:
                self._add_log(f"[SUCCESS] {result_message}")
            else:
                self._add_log(f"[ERROR] {result_message}")

            # Always detect step status after completion (handles both success and interruption)
            self._detect_completed_steps()

            # Update UI lock state (re-enables navigation controls)
            self._update_ui_lock_state()

            # Restore empty preview panel
            if self.right_panel_container is not None:
                self.right_panel_container.content = self._build_preview_panel()
                self.right_panel_container.update()

            # Show completion snackbar
            self.page.snack_bar = ft.SnackBar(
                ft.Text(result_message if result_message else "Step completed"),
                open=True,
                bgcolor=ft.Colors.GREEN if success else ft.Colors.RED,
            )
            self.page.update()

            # Auto-show QC approval dialog after coregistration steps
            if success and completed_step_id == "coregistration":
                self._show_qc_approval_dialog("initial")
            elif success and completed_step_id == "coregistration_realtime":
                self._show_qc_approval_dialog("realtime")

        # Schedule the UI update on the main thread
        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Failed to schedule UI update: {e}")

    def _on_pipeline_error(self, error: Exception):
        """Handle pipeline step error.

        Args:
            error: Exception that occurred
        """
        # This callback runs in the background thread
        async def update_ui():
            self._add_log(f"[ERROR] {str(error)}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Error: {str(error)}"),
                open=True,
                bgcolor=ft.Colors.RED,
            )
            self.page.update()

        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Failed to schedule error update: {e}")

    def _on_divider_drag_start(self, e):
        """Handle start of divider drag.

        Args:
            e: Vertical drag start event (DragStartEvent)
        """
        self._dragging = True
        self._drag_start_y = self.log_console_height
        logger.debug(f"Drag start - initial height: {self.log_console_height}")

    def _on_divider_drag_update(self, e):
        """Handle divider drag movement.

        Args:
            e: Vertical drag update event (DragUpdateEvent)
        """
        if not self._dragging or self.log_console_container is None:
            return

        # DragUpdateEvent.primary_delta is a float representing the primary axis movement
        # For on_vertical_drag_update, this is the vertical (Y) movement
        # Positive means dragging down, negative means dragging up
        delta_y = e.primary_delta

        # Update height (subtract delta_y because dragging up should increase height,
        # dragging down should decrease height - divider is above the console)
        new_height = self.log_console_height - delta_y

        # Constrain height between 50 and 500 pixels
        new_height = max(50, min(500, new_height))

        self.log_console_height = new_height
        self.log_console_container.height = new_height
        self.log_console_container.update()

    def _on_divider_drag_end(self, e):
        """Handle end of divider drag.

        Args:
            e: Vertical drag end event (DragEndEvent)
        """
        self._dragging = False
        logger.debug(f"Drag end - final height: {self.log_console_height}")

    def _add_log(self, message: str):
        """Add a log message to the console.

        Args:
            message: Log message
        """
        self.log_messages.append(message)
        # Keep only last 100 messages
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)

        # Add new log entry to ListView
        if self.log_listview is not None:
            log_entry = ft.Text(
                message,
                size=12,
                font_family="monospace",
                selectable=True,
                color=ft.Colors.WHITE,
            )
            self.log_listview.controls.append(log_entry)
            # Keep only last 100 entries in ListView too
            if len(self.log_listview.controls) > 100:
                self.log_listview.controls.pop(0)
            self.log_listview.update()

        logger.info(message)

    def _setup_file_logging(self):
        """Set up file-based logging to capture full debug output for export."""
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = logs_dir / f"session_{timestamp}.log"

        # Create file handler for the root logger to capture all pineuro logs
        self._log_file_handler = logging.FileHandler(self._log_file_path)
        self._log_file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._log_file_handler.setFormatter(formatter)

        # Add handler to pineuro root logger to capture all module logs
        pineuro_logger = logging.getLogger("pineuro")
        pineuro_logger.addHandler(self._log_file_handler)

        logger.info(f"Session log file: {self._log_file_path}")

    def _clear_log(self, e=None):
        """Clear the log console display."""
        self.log_messages.clear()
        if self.log_listview is not None:
            self.log_listview.controls.clear()
            self.log_listview.update()
        logger.debug("Log console cleared by user")

    def _show_monitoring_panel(self, step_id: str):
        """Show monitoring panel for a running step.

        Args:
            step_id: Step identifier
        """
        self._current_step_id = step_id
        config = self.study_config.session_config

        # Check if this is a feedback/transfer step
        feedback_steps = ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]
        is_feedback_step = step_id in feedback_steps

        if is_feedback_step:
            # Create activation plot for feedback/transfer runs
            # Include all masks: real + sham alternatives (if enabled)
            mask_names = list(config["mask"]["mask_names"])  # Copy to avoid modifying config
            sham_config = config.get("sham", {})
            if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
                alternative_targets = sham_config.get("alternative_mask_targets", [])
                for alt_name in alternative_targets:
                    if alt_name and alt_name not in mask_names:
                        mask_names.append(alt_name)

            n_volumes = config["feedback"]["n_volumes"]

            self.activation_plot = ActivationPlot(
                mask_names=mask_names,
                n_volumes=n_volumes,
                height=180,
                y_min=-2.0,
                y_max=2.0,
            )

            # Create FD plot for motion monitoring
            self.fd_plot = FDPlot(
                n_volumes=n_volumes,
                height=120,
            )

            # Create task mirror for visual feedback display
            self.task_mirror = TaskMirror(width=180, height=250)

            # Create progress and status UI elements
            self._progress_bar = ft.ProgressBar(value=0, width=400)
            self._progress_text = ft.Text("0 / 0 volumes", size=14)
            self._status_text = ft.Text("Waiting for data...", size=14, color=ft.Colors.GREY_600)
            self._hits_text = ft.Text("Hits: --", size=14)

            # Build monitoring panel with plots and task mirror side by side
            monitoring_panel = ft.Container(
                content=ft.Column(
                    [
                        # Header
                        ft.Text(
                            self._get_step_label(step_id),
                            size=18,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Divider(),

                        # Side-by-side: Plots column and Task mirror
                        ft.Row(
                            [
                                # Plots column (expandable)
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text("Network Activation", size=14, weight=ft.FontWeight.W_500),
                                            self.activation_plot.build(),
                                            ft.Text("Framewise Displacement", size=14, weight=ft.FontWeight.W_500),
                                            self.fd_plot.build(),
                                        ],
                                        spacing=5,
                                    ),
                                    expand=True,
                                ),
                                # Task mirror (fixed width)
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text("Task Display", size=14, weight=ft.FontWeight.W_500),
                                            self.task_mirror.build(),
                                        ],
                                        spacing=5,
                                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    ),
                                    width=200,
                                ),
                            ],
                            spacing=20,
                            vertical_alignment=ft.CrossAxisAlignment.START,
                        ),

                        # Progress section
                        ft.Container(height=10),
                        ft.Row(
                            [
                                ft.Text("Progress:", size=14, weight=ft.FontWeight.W_500),
                                self._progress_bar,
                                self._progress_text,
                            ],
                            spacing=10,
                            alignment=ft.MainAxisAlignment.START,
                        ),

                        # Status section
                        ft.Row(
                            [
                                ft.Text("Status:", size=14, weight=ft.FontWeight.W_500),
                                self._status_text,
                            ],
                            spacing=10,
                        ),
                        ft.Row(
                            [
                                self._hits_text,
                            ],
                            spacing=10,
                        ),

                        # Stop button
                        ft.Container(height=10),
                        ft.Row(
                            [
                                ft.FilledButton(
                                    "Stop",
                                    icon=ft.icons.Icons.STOP,
                                    on_click=lambda e: self._on_stop_clicked(),
                                    bgcolor=ft.Colors.RED,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ],
                    spacing=10,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=20,
                expand=True,
            )

            # Start monitoring the feedback log file
            self._start_feedback_monitoring(step_id)

        else:
            # Check if this is a collection step (including dynamic rest_run_* steps)
            is_collection_step = (
                step_id in ["2vol_initial", "2vol_realtime"] or
                step_id.startswith("rest_run_")
            )

            if is_collection_step:
                # Collection progress panel with DICOM preview
                n_volumes = self._get_collection_n_volumes(step_id)
                tr = config["scan"].get("tr")

                self.collection_progress = CollectionProgressPanel(
                    n_volumes=n_volumes,
                    tr=tr,
                )

                monitoring_panel = ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                self._get_step_label(step_id),
                                size=18,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Divider(),
                            self.collection_progress.build(),
                            ft.Row(
                                [
                                    ft.FilledButton(
                                        "Stop",
                                        icon=ft.Icons.STOP,
                                        on_click=lambda e: self._on_stop_clicked(),
                                        bgcolor=ft.Colors.RED,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.END,
                            ),
                        ],
                        spacing=10,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=20,
                    expand=True,
                )
            else:
                # Simple monitoring panel for other non-feedback steps (preprocessing, etc.)
                monitoring_panel = ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                f"Running: {self._get_step_label(step_id)}",
                                size=18,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Divider(),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.ProgressRing(),
                                        ft.Text("Processing...", size=14),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    spacing=10,
                                ),
                                alignment=ft.Alignment(0, 0),  # Center alignment
                                expand=True,
                            ),
                            ft.Row(
                                [
                                    ft.FilledButton(
                                        "Stop",
                                        icon=ft.Icons.STOP,
                                        on_click=lambda e: self._on_stop_clicked(),
                                        bgcolor=ft.Colors.RED,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.END,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=20,
                    expand=True,
                )

        # Replace right panel content
        if self.right_panel_container is not None:
            self.right_panel_container.content = monitoring_panel
            self.right_panel_container.update()

    def _get_step_label(self, step_id: str) -> str:
        """Get human-readable label for a step.

        Args:
            step_id: Step identifier

        Returns:
            Human-readable label
        """
        # Handle dynamic rest_run_* steps
        if step_id.startswith("rest_run_"):
            run_num = step_id.split("_")[-1]
            return f"Collect Rest Run {run_num}"

        labels = {
            "2vol_initial": "Collect Initial 2vol",
            "preprocessing": "Run Preprocessing",
            "mask_extraction": "Extract Masks",
            "coregistration": "Coregister to 2vol",
            "2vol_realtime": "Collect Real-time 2vol",
            "coregistration_realtime": "Coregister to Real-time 2vol",
            "transfer_1": "Transfer Run 1",
            "feedback_1": "Feedback Run 1",
            "feedback_2": "Feedback Run 2",
            "transfer_2": "Transfer Run 2",
        }
        return labels.get(step_id, step_id)

    def _get_collection_n_volumes(self, step_id: str) -> int:
        """Get expected number of volumes for a collection step.

        Args:
            step_id: Step identifier

        Returns:
            Expected number of volumes
        """
        config = self.study_config.session_config

        if step_id == "2vol_initial":
            return config["scan"].get("twovol_initial_n_volumes", config["scan"].get("twovol_n_volumes", 2))

        elif step_id == "2vol_realtime":
            return config["scan"].get("twovol_realtime_n_volumes", config["scan"].get("twovol_n_volumes", 2))

        elif step_id.startswith("rest_run_"):
            # Use new localizer_n_volumes, fall back to legacy
            n_volumes = config["scan"].get("localizer_n_volumes")
            if n_volumes is not None:
                return n_volumes
            # Legacy: try to get from rest_n_volumes list
            run_num = int(step_id.split("_")[-1])
            run_idx = run_num - 1
            rest_volumes = config["scan"].get("rest_n_volumes", [200])
            if isinstance(rest_volumes, list) and run_idx < len(rest_volumes):
                return rest_volumes[run_idx]
            return rest_volumes[0] if isinstance(rest_volumes, list) else rest_volumes

        return 0

    def _on_collection_progress(
        self,
        volumes_collected: int,
        n_volumes: int,
        elapsed_time: float,
        dicom_path: Path,
    ):
        """Handle collection progress updates from pipeline runner.

        Args:
            volumes_collected: Number of volumes collected so far
            n_volumes: Total expected volumes
            elapsed_time: Time elapsed since start (seconds)
            dicom_path: Path to the latest collected DICOM file
        """
        # This callback runs in the background thread
        # Schedule UI update on the main thread
        async def update_ui():
            if hasattr(self, 'collection_progress') and self.collection_progress is not None:
                self.collection_progress.update(volumes_collected, elapsed_time, dicom_path)
                self.page.update()

        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Failed to schedule collection progress update: {e}")

    def _start_feedback_monitoring(self, step_id: str):
        """Start monitoring feedback log file for real-time updates.

        Args:
            step_id: Step identifier (transfer_1, feedback_1, etc.)
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])

        # Include all masks: real + sham alternatives (if enabled)
        mask_names = list(config["mask"]["mask_names"])  # Copy to avoid modifying config
        sham_config = config.get("sham", {})
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            alternative_targets = sham_config.get("alternative_mask_targets", [])
            for alt_name in alternative_targets:
                if alt_name and alt_name not in mask_names:
                    mask_names.append(alt_name)

        n_volumes = config["feedback"]["n_volumes"]

        # Determine task type and get slot-based run info
        task_type = "transfer" if "transfer" in step_id else "feedback"
        output_dir = base_dir / "feedback" / "02_feedback_output"

        # Get the slot-based run info (run number and acq letter)
        run_info = get_slot_run_info(
            output_dir=output_dir,
            subject=self.participant_id,
            task=task_type,
            step_id=step_id,
        )

        # Construct log file path with acq entity (format as 2-digit number)
        log_path = output_dir / f"{self.participant_id}_task-{task_type}_acq-{run_info['acq']:02d}_run-{run_info['run']:02d}_feedbacklog.tsv"
        logger.info(f"Will monitor feedback log: {log_path}")

        # Create and start feedback monitor
        self.feedback_monitor = FeedbackMonitor(
            log_path=log_path,
            mask_names=mask_names,
            n_volumes=n_volumes,
            on_update=self._on_feedback_update,
            poll_interval=0.5,
        )
        self.feedback_monitor.start()

    def _stop_feedback_monitoring(self):
        """Stop the feedback log monitor."""
        if hasattr(self, 'feedback_monitor') and self.feedback_monitor:
            self.feedback_monitor.stop()
            self.feedback_monitor = None

    def _on_task_mirror_update(self, state: Dict) -> None:
        """Handle task visual state update for GUI mirroring.

        Called from BallTask (or any task) with its current visual state.
        This runs in a background thread, so we schedule UI update on main thread.

        Args:
            state: Visual state dictionary from task.get_mirror_state()
        """
        # Capture state in closure
        state_copy = state.copy()

        async def update_mirror():
            try:
                if hasattr(self, 'task_mirror') and self.task_mirror is not None:
                    self.task_mirror.update_state(state_copy)
            except Exception as e:
                logger.debug(f"Task mirror update failed: {e}")

        try:
            self.page.run_task(update_mirror)
        except Exception as e:
            logger.debug(f"Failed to schedule task mirror update: {e}")

    def _on_feedback_update(self, volume: int, mask_activations: Dict[str, float], motion_params: Dict[str, float], fd: Optional[float]):
        """Handle real-time feedback data update.

        Args:
            volume: Volume number
            mask_activations: Dict mapping mask name to activation value
            motion_params: Dict of motion parameters
            fd: Framewise displacement in mm (None if not available)
        """
        logger.debug(f"Feedback update: vol={volume}, activations={mask_activations}, fd={fd}")

        # Schedule UI update on main thread
        async def update_ui():
            try:
                # Update activation plot
                if self.activation_plot and mask_activations:
                    self.activation_plot.add_data(volume, mask_activations)

                # Update FD plot
                if hasattr(self, 'fd_plot') and self.fd_plot and fd is not None:
                    self.fd_plot.add_data(volume, fd)

                # Update progress
                n_volumes = self.study_config.session_config["feedback"]["n_volumes"]
                progress = min(1.0, volume / n_volumes) if n_volumes > 0 else 0

                if self._progress_bar:
                    self._progress_bar.value = progress
                    self._progress_bar.update()

                if self._progress_text:
                    self._progress_text.value = f"{volume} / {n_volumes} volumes"
                    self._progress_text.update()

                if self._status_text:
                    n_baseline = self.study_config.session_config["feedback"]["n_baseline"]
                    if volume < n_baseline:
                        self._status_text.value = f"Baseline ({volume}/{n_baseline})"
                    else:
                        self._status_text.value = "Feedback Active"
                    self._status_text.color = ft.Colors.GREEN_600 if volume >= n_baseline else ft.Colors.ORANGE_600
                    self._status_text.update()

                # Update hits/activation display
                if self._hits_text and mask_activations:
                    # Show current activation values
                    act_strs = [f"{k}: {v:.2f}" for k, v in mask_activations.items()]
                    self._hits_text.value = "Current: " + ", ".join(act_strs)
                    self._hits_text.update()

            except Exception as e:
                logger.error(f"Error updating feedback UI: {e}")

        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.debug(f"Failed to schedule feedback UI update: {e}")

    def _on_stop_clicked(self):
        """Handle stop button click during execution - shows confirmation dialog."""
        logger.info("Stop button clicked - showing confirmation dialog")

        # Check if this is a feedback/transfer step (which has post-run questions)
        is_feedback_step = self._current_step_id in (
            "transfer_1", "feedback_1", "feedback_2", "transfer_2"
        )

        # Checkbox for showing questions (only relevant for feedback steps)
        show_questions_checkbox = ft.Checkbox(
            label="Show post-run questions to participant",
            value=False,  # Default to not showing questions on stop
        )

        def confirm_stop(e):
            self.page.pop_dialog()
            show_questions = show_questions_checkbox.value if is_feedback_step else False
            logger.info(f"Stop confirmed (show_questions={show_questions})")

            # Stop feedback monitoring
            self._stop_feedback_monitoring()

            if self.pipeline_runner is not None:
                self.pipeline_runner.stop(show_questions=show_questions)
                if show_questions:
                    self._add_log("[WARNING] Stop requested - showing questions first")
                else:
                    self._add_log("[WARNING] Stop requested by user")

        def cancel_stop(e):
            self.page.pop_dialog()
            logger.info("Stop cancelled")

        # Build dialog content
        content_items = [
            ft.Text(
                "Are you sure you want to stop the current run?",
                size=14,
            ),
        ]

        if is_feedback_step:
            content_items.extend([
                ft.Divider(),
                show_questions_checkbox,
                ft.Text(
                    "If checked, the participant will still see the post-run questionnaire.",
                    size=12,
                    color=ft.Colors.GREY_600,
                    italic=True,
                ),
            ])

        dialog = ft.AlertDialog(
            title=ft.Text("Stop Run"),
            content=ft.Container(
                content=ft.Column(content_items, tight=True, spacing=10),
                width=400,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=cancel_stop),
                ft.FilledButton(
                    "Stop",
                    icon=ft.Icons.STOP,
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.RED_700,
                    on_click=confirm_stop,
                ),
            ],
        )

        self.page.show_dialog(dialog)

    def _on_step_clicked(self, step_id: str):
        """Handle step button click.

        Args:
            step_id: Step identifier
        """
        # Ignore clicks while running
        if self._is_running:
            return

        logger.info(f"Step clicked: {step_id}")

        # Build parameter preview panel for this step
        preview_panel = self._build_step_preview(step_id)

        # Replace the right panel content with the preview
        if self.right_panel_container is not None:
            self.right_panel_container.content = preview_panel
            self.right_panel_container.update()
        else:
            logger.warning("right_panel_container not initialized")

    def _get_next_step(self) -> Optional[tuple[str, str]]:
        """Determine the next step to run in the workflow.

        Considers both pending and interrupted steps. Interrupted steps
        (force-stopped runs) are treated as needing to be re-run.

        Returns:
            Tuple of (step_id, step_label) or None if all complete
        """
        # Build dynamic workflow order based on configuration
        config = self.study_config.session_config
        n_localizer_runs = config["scan"].get("localizer_n_runs", 2)
        twovol_initial_enabled = config["scan"].get("twovol_initial_enabled", True)
        twovol_realtime_enabled = config["scan"].get("twovol_realtime_enabled", True)

        workflow = []

        # In mask reuse mode, skip localizer and mask extraction steps
        if not self._is_mask_reuse_mode:
            # Add Initial 2vol if enabled
            if twovol_initial_enabled:
                workflow.append(("2vol_initial", "Collect Initial 2vol"))

            # Add localizer runs dynamically
            for run_num in range(1, n_localizer_runs + 1):
                workflow.append((f"rest_run_{run_num}", f"Collect Rest Run {run_num}"))

            # Add mask extraction steps
            workflow.extend([
                ("preprocessing", "Run Preprocessing"),
                ("mask_extraction", "Extract Masks"),
                ("mask_approval", "Review & Approve Masks"),
            ])

            # Add coregistration only if initial 2vol is enabled
            if twovol_initial_enabled:
                workflow.append(("coregistration", "Coregister to 2vol"))

        # Add realtime 2vol if enabled
        if twovol_realtime_enabled:
            workflow.extend([
                ("2vol_realtime", "Collect Real-time 2vol"),
                ("coregistration_realtime", "Coregister to Real-time 2vol"),
            ])

        # Add feedback runs
        workflow.extend([
            ("transfer_1", "Transfer Run 1"),
            ("feedback_1", "Feedback Run 1"),
            ("feedback_2", "Feedback Run 2"),
            ("transfer_2", "Transfer Run 2"),
        ])

        # Debug: log current statuses
        logger.info("Current step statuses:")
        for step_id, _ in workflow:
            status = self.step_statuses.get(step_id, StepStatus.PENDING)
            logger.info(f"  {step_id}: {status.value}")

        # Find first pending or interrupted step (interrupted takes priority as it needs re-run)
        for step_id, step_label in workflow:
            status = self.step_statuses.get(step_id)
            if status in (StepStatus.PENDING, StepStatus.INTERRUPTED):
                status_desc = "interrupted" if status == StepStatus.INTERRUPTED else "pending"
                logger.info(f"Next {status_desc} step: {step_id} ({step_label})")
                return (step_id, step_label)

        logger.info("No pending/interrupted steps found - all complete")
        return None

    def _on_run_next_clicked(self):
        """Handle Run Next button click."""
        # Ignore clicks while running
        if self._is_running:
            return

        next_step = self._get_next_step()

        if next_step is None:
            logger.info("No pending steps - session complete")
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Session complete! All steps finished."),
                open=True,
            )
            self.page.update()
            return

        step_id, step_label = next_step
        logger.info(f"Run Next clicked: {step_id} ({step_label})")

        # Handle mask_approval step specially - open the review dialog
        if step_id == "mask_approval":
            self._show_mask_review_dialog()
            return

        # Trigger the step (same as clicking step button)
        self._on_step_clicked(step_id)

    def _on_back(self, e):
        """Handle back button click."""
        # Ignore clicks while running
        if self._is_running:
            return

        logger.info("Back to Study Manager")
        if self.on_back_callback:
            self.on_back_callback()

    def _update_run_next_button(self):
        """Update the Run Next button text based on next step."""
        if self.run_next_button is None:
            logger.warning("run_next_button not initialized, cannot update")
            return

        next_step = self._get_next_step()
        if next_step is None:
            self.run_next_button.text = "Session Complete"
            self.run_next_button.disabled = True
            self.run_next_button.icon = ft.icons.CHECK_CIRCLE
            logger.info("Updated Run Next button: Session Complete")
        else:
            step_id, step_label = next_step
            self.run_next_button.text = f"Run Next: {step_label}"
            self.run_next_button.disabled = False
            self.run_next_button.icon = ft.icons.Icons.PLAY_ARROW
            logger.info(f"Updated Run Next button: {step_label}")

        # Force immediate visual update
        try:
            self.run_next_button.update()
        except Exception as e:
            logger.warning(f"Button update failed: {e}")

    def _update_ui_lock_state(self):
        """Enable/disable navigation controls based on running state."""
        # Disable/enable Run Next button
        if self.run_next_button:
            self.run_next_button.disabled = self._is_running or self._get_next_step() is None
            try:
                self.run_next_button.update()
            except Exception as e:
                logger.warning(f"Run Next button update failed: {e}")

        # Refresh step list to update step button states
        self._refresh_step_list()

        # Rebuild header to update back button state
        self._rebuild_header()

    def _refresh_step_list(self):
        """Rebuild the step list to reflect updated statuses."""
        logger.info("=== REFRESH STEP LIST CALLED ===")

        if self.step_panel_list is None:
            logger.warning("step_panel_list not initialized, cannot refresh")
            return

        # Rebuild the expansion panel list with updated statuses
        logger.info("Rebuilding step panel controls...")
        panels = []
        # Only show localizer and mask extraction sections if NOT in mask reuse mode
        if not self._is_mask_reuse_mode:
            panels.append(self._build_localizer_session_section())
            panels.append(self._build_mask_extraction_section())
        panels.append(self._build_feedback_section())
        panels.append(self._build_utilities_section())
        self.step_panel_list.controls = panels

        # Force updates in the correct order
        try:
            self.step_panel_list.update()
            logger.info("Step panel controls updated")
        except Exception as e:
            logger.warning(f"Step panel update failed: {e}")

        # Update the Run Next button by rebuilding the header
        self._rebuild_header()

        # Force a complete page refresh
        try:
            self.page.update()
            logger.info("Page updated")
        except Exception as e:
            logger.warning(f"Page update failed: {e}")

        logger.info("=== REFRESH STEP LIST COMPLETE ===")

    def build(self) -> ft.Control:
        """Build the participant workflow UI.

        Returns:
            Flet control for the workflow screen
        """
        # Header
        use_simulator = self.study_config.session_config.get("scan", {}).get("collection_speed", 1.0) > 1.0
        simulator_text = " | Simulator Mode" if use_simulator else ""

        # Store simulator text for header rebuilding
        self._simulator_text = simulator_text

        # Create Run Next button
        next_step = self._get_next_step()
        if next_step is None:
            button_text = "Session Complete"
            button_disabled = True
            button_icon = ft.icons.Icons.CHECK_CIRCLE
        else:
            step_id, step_label = next_step
            button_text = f"Run Next: {step_label}"
            button_disabled = False
            button_icon = ft.icons.Icons.PLAY_ARROW

        self.run_next_button = ft.FilledButton(
            button_text,
            icon=button_icon,
            on_click=lambda e: self._on_run_next_clicked(),
            disabled=button_disabled,
        )

        # Store reference to header container for dynamic updates
        self.header_container = ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.IconButton(
                                icon=ft.icons.Icons.ARROW_BACK,
                                on_click=self._on_back,
                                tooltip="Back to Studies",
                                disabled=self._is_running,  # Disable when running
                            ),
                            ft.Text(
                                f"pineuro: {self.study_config.metadata.name}",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Text(
                                f"| Participant: {self.participant_id}{self._simulator_text}",
                                size=16,
                                color=ft.Colors.GREY_700,
                            ),
                        ],
                        spacing=10,
                    ),
                    self.run_next_button,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=20,
            bgcolor=ft.Colors.BLUE_50,
        )

        # Left panel: Step dashboard
        # Store reference for dynamic updates
        # Only show localizer and mask extraction sections if NOT in mask reuse mode
        panels = []
        if not self._is_mask_reuse_mode:
            panels.append(self._build_localizer_session_section())
            panels.append(self._build_mask_extraction_section())
        panels.append(self._build_feedback_section())
        panels.append(self._build_utilities_section())

        self.step_panel_list = ft.ExpansionPanelList(
            expand_icon_color=ft.Colors.BLUE_500,
            elevation=2,
            controls=panels,
        )

        left_panel = ft.Container(
            content=ft.Column(
                [self.step_panel_list],
                scroll=ft.ScrollMode.AUTO,
            ),
            width=300,
            padding=10,
        )

        # Right panel: Preview/Monitoring
        self.right_panel_container = ft.Container(
            content=self._build_preview_panel(),
            expand=True,
            bgcolor=ft.Colors.WHITE,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=5,
        )

        # Main content area
        content_area = ft.Row(
            [
                left_panel,
                ft.VerticalDivider(),
                self.right_panel_container,
            ],
            spacing=0,
            expand=True,
        )

        # Log console at bottom
        log_console = self._build_log_console()

        return ft.Column(
            [
                self.header_container,
                ft.Container(
                    content=content_area,
                    expand=True,
                ),
                log_console,
            ],
            spacing=0,
            expand=True,
        )
