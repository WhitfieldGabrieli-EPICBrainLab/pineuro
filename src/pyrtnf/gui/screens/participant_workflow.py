"""Participant Workflow screen for executing pipeline steps.

This module provides the main dashboard for executing pyrtnf pipeline steps for a
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
from pyrtnf.feedback import get_next_run_number, get_slot_run_info, STEP_TO_RUN_NUMBER

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
        on_back=None,
    ):
        """Initialize the participant workflow screen.

        Args:
            page: Flet Page object
            config_manager: ConfigManager instance
            study_config: Study configuration to use
            participant_id: Participant ID (e.g., "sub-01")
            on_back: Callback when back button is clicked
        """
        self.page = page
        self.config_manager = config_manager
        self.participant_id = participant_id
        self.on_back_callback = on_back

        # Create participant-specific base directory
        # Modify study config to include participant ID in base_dir path
        self.study_config = study_config
        base_dir = Path(study_config.session_config["paths"]["base_dir"])
        participant_base_dir = base_dir / participant_id

        # Update the session config with participant-specific base directory
        self.study_config.session_config["paths"]["base_dir"] = str(participant_base_dir)

        # Create the participant directory if it doesn't exist
        participant_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Participant data directory: {participant_base_dir}")

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

        # QC report manager for detecting and opening reports
        self.qc_manager = QCReportManager(
            base_dir=participant_base_dir,
            participant_id=participant_id,
        )

        # QC approval state (two gates)
        self._initial_qc_approved = self._check_qc_approval("initial")
        self._realtime_qc_approved = self._check_qc_approval("realtime")

    def _check_qc_approval(self, approval_type: str) -> bool:
        """Check if QC has been approved for a checkpoint.

        Args:
            approval_type: "initial" (after coregistration) or "realtime" (after coregistration_realtime)

        Returns:
            True if the approval marker file exists
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        if approval_type == "initial":
            marker_path = base_dir / "mask_extraction" / ".qc_approved_initial"
        else:  # realtime
            marker_path = base_dir / "feedback" / ".qc_approved_realtime"
        return marker_path.exists()

    def _set_qc_approved(self, approval_type: str, approved: bool):
        """Set QC approval state for a checkpoint.

        Args:
            approval_type: "initial" or "realtime"
            approved: True to approve, False to revoke
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        if approval_type == "initial":
            marker_path = base_dir / "mask_extraction" / ".qc_approved_initial"
        else:  # realtime
            marker_path = base_dir / "feedback" / ".qc_approved_realtime"

        if approved:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
            logger.info(f"QC approval marker created: {marker_path}")
        elif marker_path.exists():
            marker_path.unlink()
            logger.info(f"QC approval marker removed: {marker_path}")

        if approval_type == "initial":
            self._initial_qc_approved = approved
        else:
            self._realtime_qc_approved = approved

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

        # Initialize all as pending
        steps = [
            "2vol_initial",
            "rest_run_1",
            "rest_run_2",
            "preprocessing",
            "mask_extraction",
            "coregistration",
            "2vol_realtime",
            "coregistration_realtime",
            "transfer_1",
            "feedback_1",
            "feedback_2",
            "transfer_2",
        ]

        for step in steps:
            self.step_statuses[step] = StepStatus.PENDING

        # Check 2vol initial collection
        twovol_initial_dir = base_dir / "collection" / "task-2vol_run-01"
        status = self._check_collection_status(twovol_initial_dir, config["scan"]["twovol_n_volumes"])
        self.step_statuses["2vol_initial"] = status
        if status != StepStatus.PENDING:
            logger.info(f"Detected {status.value}: 2vol_initial")

        # Check rest run collections
        rest_run_1_dir = base_dir / "collection" / "task-rest_run-01"
        expected_vols = config["scan"]["rest_n_volumes"][0] if isinstance(config["scan"]["rest_n_volumes"], list) else config["scan"]["rest_n_volumes"]
        status = self._check_collection_status(rest_run_1_dir, expected_vols)
        self.step_statuses["rest_run_1"] = status
        if status != StepStatus.PENDING:
            logger.info(f"Detected {status.value}: rest_run_1")

        rest_run_2_dir = base_dir / "collection" / "task-rest_run-02"
        expected_vols = config["scan"]["rest_n_volumes"][1] if isinstance(config["scan"]["rest_n_volumes"], list) and len(config["scan"]["rest_n_volumes"]) > 1 else config["scan"]["rest_n_volumes"]
        status = self._check_collection_status(rest_run_2_dir, expected_vols)
        self.step_statuses["rest_run_2"] = status
        if status != StepStatus.PENDING:
            logger.info(f"Detected {status.value}: rest_run_2")

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

        # Check coregistration (2vol) outputs - initial (run-01) is in mask_extraction/04_2vol
        feedback_dir = base_dir / "feedback"
        initial_twovol_masks_dir = mask_extraction_dir / "04_2vol" / "masks"

        if initial_twovol_masks_dir.exists() and len(list(initial_twovol_masks_dir.glob("*_to_task-2vol_run-01_*.nii.gz"))) > 0:
            self.step_statuses["coregistration"] = StepStatus.COMPLETE
            logger.info("Detected complete: coregistration")

        # Check 2vol realtime collection
        twovol_realtime_dir = base_dir / "collection" / "task-2vol_run-02"
        status = self._check_collection_status(twovol_realtime_dir, config["scan"]["twovol_n_volumes"])
        self.step_statuses["2vol_realtime"] = status
        if status != StepStatus.PENDING:
            logger.info(f"Detected {status.value}: 2vol_realtime")

        # Check coregistration_realtime (2vol run-02) outputs - realtime is in feedback/01_2vol
        realtime_twovol_masks_dir = feedback_dir / "01_2vol" / "masks"
        if realtime_twovol_masks_dir.exists() and len(list(realtime_twovol_masks_dir.glob("*_to_task-2vol_run-02_*.nii.gz"))) > 0:
            self.step_statuses["coregistration_realtime"] = StepStatus.COMPLETE
            logger.info("Detected complete: coregistration_realtime")

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
        step_to_session = {
            "2vol_initial": "localizer",
            "rest_run_1": "localizer",
            "rest_run_2": "localizer",
            "preprocessing": "mask_extraction",
            "mask_extraction": "mask_extraction",
            "coregistration": "mask_extraction",
            "2vol_realtime": "feedback",
            "coregistration_realtime": "feedback",
            "transfer_1": "feedback",
            "feedback_1": "feedback",
            "feedback_2": "feedback",
            "transfer_2": "feedback",
        }

        next_step = self._get_next_step()
        if next_step is None:
            return None

        step_id, _ = next_step
        return step_to_session.get(step_id)

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

        # Gate 1: Initial QC approval required for feedback session start
        gate1_steps = ["2vol_realtime", "coregistration_realtime"]
        requires_initial_approval = step_id in gate1_steps and not self._initial_qc_approved

        # Gate 2: Realtime QC approval required for feedback/transfer runs
        gate2_steps = ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]
        requires_realtime_approval = step_id in gate2_steps and not self._realtime_qc_approved

        requires_approval = requires_initial_approval or requires_realtime_approval

        button = ft.ElevatedButton(
            label,
            on_click=on_click,
            disabled=self._is_running or status == StepStatus.RUNNING or requires_approval,
        )

        # Show lock icon if blocked by QC approval, otherwise show status icon
        if requires_approval:
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
        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Localizer Session", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(
                [
                    ft.Text("Initial 2vol", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "2vol_initial",
                        "Collect",
                        lambda e: self._on_step_clicked("2vol_initial"),
                    ),
                    ft.Divider(),
                    ft.Text("Resting State", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "rest_run_1",
                        "Collect Run 1",
                        lambda e: self._on_step_clicked("rest_run_1"),
                    ),
                    self._create_step_button(
                        "rest_run_2",
                        "Collect Run 2",
                        lambda e: self._on_step_clicked("rest_run_2"),
                    ),
                ],
                spacing=10,
            ),
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
            ft.Divider(),
            ft.Text("Coregistration", size=14, weight=ft.FontWeight.W_500),
            self._create_step_button(
                "coregistration",
                "Coregister to 2vol",
                lambda e: self._on_step_clicked("coregistration"),
            ),
        ]

        # Add QC review/approval after coregistration if complete
        if self.step_statuses.get("coregistration") == StepStatus.COMPLETE:
            if self._initial_qc_approved:
                # Approved - show "View QC" button with checkmark
                controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.TextButton(
                                    "View QC",
                                    icon=ft.Icons.ASSESSMENT,
                                    on_click=lambda e: self._show_qc_approval_dialog("initial"),
                                ),
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
                            on_click=lambda e: self._show_qc_approval_dialog("initial"),
                        ),
                        padding=ft.padding.only(left=10, top=5),
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
        # Build controls list
        controls = [
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
        ]

        # Add QC review/approval after coregistration_realtime if complete
        if self.step_statuses.get("coregistration_realtime") == StepStatus.COMPLETE:
            if self._realtime_qc_approved:
                # Approved - show "View QC" button with checkmark
                controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.TextButton(
                                    "View QC",
                                    icon=ft.Icons.ASSESSMENT,
                                    on_click=lambda e: self._show_qc_approval_dialog("realtime"),
                                ),
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
                            on_click=lambda e: self._show_qc_approval_dialog("realtime"),
                        ),
                        padding=ft.padding.only(left=10, top=5),
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
            plots = self.qc_manager.get_2vol_qc_plots()
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
                            ft.Row(
                                [
                                    ft.TextButton("Clear", icon=ft.icons.Icons.CLEAR),
                                    ft.TextButton("Export", icon=ft.icons.Icons.SAVE),
                                ],
                                spacing=5,
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
                            f"pyrtnf: {self.study_config.metadata.name}",
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
                "label": "Collect Initial 2vol",
                "description": "Collect 2 volumes for initial positioning verification before resting-state acquisition.",
                "parameters": {
                    "Task": "2vol (run-01)",
                    "Volumes": config["scan"]["twovol_n_volumes"],
                    "Source": config["scan"]["twovol_initial_dicoms"],
                    "Output": base_dir / "collection" / "task-2vol_run-01",
                    "Simulator Speed": f"{config['scan']['collection_speed']}x",
                },
                "output_path": base_dir / "collection" / "task-2vol_run-01",
            },
            "rest_run_1": {
                "label": "Collect Rest Run 1",
                "description": "Collect resting-state fMRI data (run 1) for network mask extraction.",
                "parameters": {
                    "Task": "rest (run-01)",
                    "Volumes": config["scan"]["rest_n_volumes"][0] if isinstance(config["scan"]["rest_n_volumes"], list) else config["scan"]["rest_n_volumes"],
                    "Source": config["scan"]["rest_source_dicoms"][0] if isinstance(config["scan"]["rest_source_dicoms"], list) else config["scan"]["rest_source_dicoms"],
                    "Output": base_dir / "collection" / "task-rest_run-01",
                    "Simulator Speed": f"{config['scan']['collection_speed']}x",
                },
                "output_path": base_dir / "collection" / "task-rest_run-01",
            },
            "rest_run_2": {
                "label": "Collect Rest Run 2",
                "description": "Collect resting-state fMRI data (run 2) for network mask extraction.",
                "parameters": {
                    "Task": "rest (run-02)",
                    "Volumes": config["scan"]["rest_n_volumes"][1] if isinstance(config["scan"]["rest_n_volumes"], list) and len(config["scan"]["rest_n_volumes"]) > 1 else config["scan"]["rest_n_volumes"],
                    "Source": config["scan"]["rest_source_dicoms"][1] if isinstance(config["scan"]["rest_source_dicoms"], list) and len(config["scan"]["rest_source_dicoms"]) > 1 else config["scan"]["rest_source_dicoms"],
                    "Output": base_dir / "collection" / "task-rest_run-02",
                    "Simulator Speed": f"{config['scan']['collection_speed']}x",
                },
                "output_path": base_dir / "collection" / "task-rest_run-02",
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
            "mask_extraction": {
                "label": "Extract Network Masks",
                "description": f"ICA-based extraction of {len(config['mask']['yeo_targets'])} network masks using {config['mask']['n_yeo']}-network Yeo atlas.",
                "parameters": {
                    "Networks": ", ".join(config['mask']['yeo_targets']),
                    "Yeo Atlas": f"{config['mask']['n_yeo']} networks",
                    "Smoothing": f"{config['mask']['smooth_fwhm']} mm FWHM",
                    "Highpass": f"{config['mask']['highpass']} Hz",
                    "Output": base_dir / "mask_extraction",
                },
                "output_path": base_dir / "mask_extraction" / "03_masks",
            },
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
            "2vol_realtime": {
                "label": "Collect Real-time 2vol",
                "description": "Collect 2 volumes immediately before feedback session for real-time registration verification.",
                "parameters": {
                    "Task": "2vol (run-02)",
                    "Volumes": config["scan"]["twovol_n_volumes"],
                    "Source": config["scan"]["twovol_realtime_dicoms"],
                    "Output": base_dir / "collection" / "task-2vol_run-02",
                    "Simulator Speed": f"{config['scan']['collection_speed']}x",
                },
                "output_path": base_dir / "collection" / "task-2vol_run-02",
            },
            "coregistration_realtime": {
                "label": "Coregister to Real-time 2vol",
                "description": "Register network masks to real-time 2vol space for feedback runs.",
                "parameters": {
                    "Input Masks": base_dir / "mask_extraction" / "03_masks",
                    "Target": "2vol run-02",
                    "Output": base_dir / "feedback" / "01_2vol" / "masks",
                },
                "output_path": base_dir / "feedback" / "01_2vol" / "masks",
                "output_pattern": "*_to_task-2vol_run-02_*.nii.gz",
            },
            "transfer_1": {
                "label": "Transfer Run 1",
                "description": "Practice run without neurofeedback to establish baseline performance.",
                "parameters": {
                    "Task": "transfer (run-01)",
                    "Volumes": config["feedback"]["n_volumes"],
                    "Baseline": config["feedback"]["n_baseline"],
                    "Feedback": "OFF",
                    "Output": base_dir / "feedback" / "02_feedback_output",
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
                },
                "output_path": base_dir / "feedback" / "02_feedback_output" / f"{participant_id}_task-transfer_run-02_feedbacklog.tsv",
            },
        }

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
            )

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
            mask_names = config["mask"]["mask_names"]
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
            # Simple monitoring panel for non-feedback steps
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
                                    icon=ft.icons.Icons.STOP,
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
        labels = {
            "2vol_initial": "Collect Initial 2vol",
            "rest_run_1": "Collect Rest Run 1",
            "rest_run_2": "Collect Rest Run 2",
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

    def _start_feedback_monitoring(self, step_id: str):
        """Start monitoring feedback log file for real-time updates.

        Args:
            step_id: Step identifier (transfer_1, feedback_1, etc.)
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        mask_names = config["mask"]["mask_names"]
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
        if self.feedback_monitor:
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
        # Define workflow order
        workflow = [
            ("2vol_initial", "Collect Initial 2vol"),
            ("rest_run_1", "Collect Rest Run 1"),
            ("rest_run_2", "Collect Rest Run 2"),
            ("preprocessing", "Run Preprocessing"),
            ("mask_extraction", "Extract Masks"),
            ("coregistration", "Coregister to 2vol"),
            ("2vol_realtime", "Collect Real-time 2vol"),
            ("coregistration_realtime", "Coregister to Real-time 2vol"),
            ("transfer_1", "Transfer Run 1"),
            ("feedback_1", "Feedback Run 1"),
            ("feedback_2", "Feedback Run 2"),
            ("transfer_2", "Transfer Run 2"),
        ]

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
        self.step_panel_list.controls = [
            self._build_localizer_session_section(),
            self._build_mask_extraction_section(),
            self._build_feedback_section(),
        ]

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
                                f"pyrtnf: {self.study_config.metadata.name}",
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
        self.step_panel_list = ft.ExpansionPanelList(
            expand_icon_color=ft.Colors.BLUE_500,
            elevation=2,
            controls=[
                self._build_localizer_session_section(),
                self._build_mask_extraction_section(),
                self._build_feedback_section(),
            ],
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
