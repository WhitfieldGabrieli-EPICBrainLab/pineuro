"""Participant Workflow screen for executing pipeline steps.

This module provides the main dashboard for executing pineuro pipeline steps for a
single participant:
- Step-by-step workflow with status tracking
- Parameter preview before execution
- Real-time monitoring during feedback runs
- Log console for pipeline output
- QC report viewing

The workflow is divided into four main sections:
1. Localizer Tasks: Initial Reference collection and localizer task/rest collection
2. Mask Extraction: Preprocessing, network mask extraction (ICA or GLM), and coregistration to reference
3. Feedback Tasks: Real-time Reference collection and feedback runs
4. Quality Control: View QC reports for each phase

Status is automatically detected based on output file existence, allowing sessions
to be resumed after interruption.
"""

import contextlib
import logging
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from pineuro.session import SessionMetadata

# Heavy pineuro imports (feedback, preprocessing, session, sham, tasks.state)
# are deferred to method bodies to avoid loading numpy/nibabel/scipy at import time.
import flet as ft

from ..components.activation_plot import ActivationPlot
from ..components.collection_progress import CollectionProgressPanel
from ..components.csv_poller import FeedbackMonitor
from ..components.mask_selection_dialog import MaskReviewInfo, MaskSelectionDialog
from ..components.pipeline_runner import PipelineRunner
from ..components.qc_plots import DVARSPlot, FDPlot, TSNRPlot
from ..components.qc_viewer import QCReportManager
from ..components.task_mirror import TaskMirror
from ..components.unified_attempt_dialog import RunGroup, UnifiedAttemptDialog, scan_collection_attempts
from ..config_manager import ConfigManager
from ..config_snapshot import (
    archive_config_snapshot,
    compare_protocol_configs,
    format_differences,
    load_config_snapshot,
    save_config_snapshot,
)
from ..models import StudyConfig
from ..utils import truncate_text

logger = logging.getLogger(__name__)

# File-type filter for NIfTI file pickers (manual mask configuration).
NIFTI_FILETYPES = [("NIfTI files", "*.nii *.nii.gz"), ("All Files", "*.*")]


def _loc(config: dict, idx: int = 0) -> dict:
    """Return ``config["localizers"][idx]`` as a dict, or ``{}`` if missing."""
    locs = config.get("localizers") or []
    return locs[idx] if idx < len(locs) else {}


def _frc(config: dict, idx: int = 0) -> dict:
    """Return ``config["feedback_runs"][idx]`` as a dict, or ``{}`` if missing."""
    frcs = config.get("feedback_runs") or []
    return frcs[idx] if idx < len(frcs) else {}


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"  # Partial completion (user stopped early)
    FAILED = "failed"
    SKIPPED = "skipped"  # Bypassed: downstream step already complete


def _build_feedback_step_ids(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Build ``(step_id, label)`` pairs for feedback/transfer pipeline steps.

    Returns
    -------
    list[tuple[str, str]]
        E.g. ``[("transfer_1", "Transfer Run 1 (Ball)"), ("feedback_1", "Feedback Run 1 (Ball)"), ...]``
    """
    feedback_runs = config.get("feedback_runs", [])
    steps: list[tuple[str, str]] = []
    fb_counter = 0
    tf_counter = 0
    for frc in feedback_runs:
        task_type = frc.get("task", "?")
        label_suffix = task_type.replace("_", " ").title()
        use_transfers = frc.get("use_transfer_runs", False)
        max_runs = frc.get("max_runs", 1)

        if use_transfers:
            tf_counter += 1
            steps.append((f"transfer_{tf_counter}", f"Transfer Run {tf_counter} ({label_suffix})"))
        for _ in range(max_runs):
            fb_counter += 1
            steps.append((f"feedback_{fb_counter}", f"Feedback Run {fb_counter} ({label_suffix})"))
        if use_transfers:
            tf_counter += 1
            steps.append((f"transfer_{tf_counter}", f"Transfer Run {tf_counter} ({label_suffix})"))
    return steps


class ParticipantWorkflowScreen:
    """Participant workflow dashboard for pipeline execution."""

    def __init__(
        self,
        page: ft.Page,
        config_manager: ConfigManager,
        study_config: StudyConfig,
        participant_id: str,
        session_id: str,
        session_metadata: Optional["SessionMetadata"] = None,
        on_back=None,
    ):
        """Initialize the participant workflow screen.

        Args:
            page: Flet Page object
            config_manager: ConfigManager instance
            study_config: Study configuration to use
            participant_id: Participant ID (e.g., "sub-01")
            session_id: Session identifier (e.g., "ses-01").
            session_metadata: SessionMetadata for new sessions. None when continuing existing session.
            on_back: Callback when back button is clicked
        """
        self.page = page
        self.config_manager = config_manager
        self.participant_id = participant_id
        self.session_id = session_id
        self._bids_prefix = f"{participant_id}_{session_id}"
        self.session_metadata = session_metadata
        self.on_back_callback = on_back

        # Create participant-specific base directory
        # Make a deep copy of study config to avoid mutating the original
        import copy

        self.study_config = copy.deepcopy(study_config)
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])

        # The base_dir might already contain participant/session paths from a previous session
        # (if the study_config object was reused). We need to get the study-level base_dir.
        # Look for participant ID in the path parts and strip it and everything after.
        parts = base_dir.parts
        if participant_id in parts:
            idx = parts.index(participant_id)
            base_dir = Path(*parts[:idx])

        participant_base_dir = base_dir / participant_id / session_id

        # Update the session config with participant-specific base directory
        self.study_config.session_config["paths"]["base_dir"] = str(participant_base_dir)

        # Create the participant directory if it doesn't exist
        participant_base_dir.mkdir(parents=True, exist_ok=True)

        # Session config file path for persistence
        session_metadata_file = participant_base_dir / "session_metadata.json"

        # Store session config info in session config if provided (new session)
        if session_metadata:
            session_metadata_dict = {
                "session_id": session_metadata.session_id,
                "session_number": session_metadata.session_number,
                "mask_source": session_metadata.mask_source,
                "mask_source_session": session_metadata.mask_source_session,
                "localizer_enabled": session_metadata.localizer_enabled,
                "notes": session_metadata.notes,
            }
            self.study_config.session_config["session"] = session_metadata_dict

            # Persist session config to file
            import json

            with open(session_metadata_file, "w") as f:
                json.dump(session_metadata_dict, f, indent=2)

        elif session_metadata_file.exists():
            # Load existing session config from file (continuing existing session)
            import json

            with open(session_metadata_file) as f:
                session_metadata_dict = json.load(f)
            self.study_config.session_config["session"] = session_metadata_dict

        # Study config snapshot: save on first run, compare on re-run
        self._config_differences = []
        saved_snapshot = load_config_snapshot(participant_base_dir)
        if saved_snapshot is None:
            save_config_snapshot(self.study_config, participant_base_dir, participant_id, session_id)
        else:
            self._config_differences = compare_protocol_configs(saved_snapshot, self.study_config.to_dict())
            if self._config_differences:
                logger.warning(
                    "Study config changed since last run for %s (%s):\n%s",
                    participant_id,
                    session_id,
                    format_differences(self._config_differences),
                )
        self._participant_base_dir = participant_base_dir

        # Mask approval state (must be initialized before _detect_completed_steps)
        self._initial_qc_approved = self._check_qc_approval("initial")

        # Initialize step statuses
        self.step_statuses: dict[str, StepStatus] = {}
        self._detect_completed_steps()

        # Current active panel (preview or monitoring)
        self.active_panel: ft.Control | None = None

        # Run Next button (will be created in build)
        self.run_next_button: ft.FilledButton | None = None

        # Right panel container (for dynamic content updates)
        self.right_panel_container: ft.Container | None = None

        # Left panel container (for step list updates)
        self.left_panel_container: ft.Container | None = None
        self.step_panel_list: ft.ExpansionPanelList | None = None

        # Pipeline runner for background execution
        self.pipeline_runner: PipelineRunner | None = None

        # Log messages (for log console)
        self.log_messages: list[str] = []
        self.log_listview: ft.ListView | None = None  # For auto-scroll
        self.log_console_container: ft.Container | None = None  # For resizing
        self.log_console_height: float = 150  # Initial height
        self._dragging: bool = False  # Track if currently dragging
        self._drag_start_y: float = 0  # Starting Y position of drag
        self._is_running: bool = False  # Track if a pipeline step is running
        self._current_step_id: str | None = None  # Track current running step

        # File logging for export functionality
        self._log_file_handler: logging.FileHandler | None = None

        # Cache mask reuse mode flag for easy access
        session_conf_dict: dict = self.study_config.session_config.get("session", {})
        self._is_mask_reuse_mode = session_conf_dict.get("mask_source") == "reuse"
        self._mask_source_session = session_conf_dict.get("mask_source_session")
        # Track whether user switched to MNI atlas masks mid-session
        self._is_mni_atlas_mode: bool = False
        self._log_file_path: Path | None = None
        self._setup_file_logging()

        # QC report manager for detecting and opening reports
        self.qc_manager = QCReportManager(
            base_dir=participant_base_dir,
            participant_id=participant_id,
            session_id=session_id,
        )

        # Selected rest runs for preprocessing (None = use all available)
        self._selected_rest_runs: list[int] | None = None
        # Selected attempts per run for preprocessing (None = auto-select best)
        self._selected_attempts: dict[int, list[int]] | None = None

        # Overwrite flag — set to True to re-run already-complete steps
        self._pending_overwrite: bool = False

        # In-flight attempt captured at stop time so on_complete can offer
        # Keep vs Discard once the worker thread actually finishes.
        self._pending_discard_attempt: dict[str, Any] | None = None

        # Last failed step (for error recovery panel)
        self._last_failed_step: str | None = None

    def _check_qc_approval(self, approval_type: str) -> bool:
        """Check if masks have been approved.

        For multi-localizer, checks all per-localizer approval markers.
        For single-localizer, checks the single mask_extraction marker.

        Args:
            approval_type: Currently only "initial" is used (after mask extraction)

        Returns:
            True if the approval marker file(s) exist
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        config = self.study_config.session_config
        localizers = config.get("localizers", [])

        if localizers and len(localizers) > 1:
            # Multi-localizer: all per-localizer markers must exist
            for loc_idx, loc in enumerate(localizers):
                loc_type = loc.get("type", "rest")
                loc_task_name = "rest" if loc_type == "rest" else loc_type
                loc_dir = base_dir / "mask_extraction" / f"loc_{loc_idx}_{loc_task_name}"
                if not (loc_dir / ".masks_approved").exists():
                    return False
            return True
        else:
            marker_path = base_dir / "mask_extraction" / ".masks_approved"
            return marker_path.exists()

    def _set_qc_approved(self, approval_type: str, approved: bool, loc_idx: int | None = None):
        """Set mask approval state.

        Args:
            approval_type: Currently only "initial" is used
            approved: True to approve, False to revoke
            loc_idx: For multi-localizer, the localizer index to approve
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        config = self.study_config.session_config
        localizers = config.get("localizers", [])
        is_multi = localizers and len(localizers) > 1

        if loc_idx is not None and is_multi:
            # Per-localizer approval
            loc_type = localizers[loc_idx].get("type", "rest")
            loc_task_name = "rest" if loc_type == "rest" else loc_type
            loc_dir = base_dir / "mask_extraction" / f"loc_{loc_idx}_{loc_task_name}"
            marker_path = loc_dir / ".masks_approved"

            if approved:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.touch()
            elif marker_path.exists():
                marker_path.unlink()

            self.step_statuses[f"localizer_{loc_idx}_approval"] = (
                StepStatus.COMPLETE if approved else StepStatus.PENDING
            )

            # Update _initial_qc_approved based on whether ALL localizers are approved
            self._initial_qc_approved = all(
                self.step_statuses.get(f"localizer_{i}_approval") == StepStatus.COMPLETE for i in range(len(localizers))
            )
        else:
            # Single-localizer approval
            marker_path = base_dir / "mask_extraction" / ".masks_approved"

            if approved:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.touch()
            elif marker_path.exists():
                marker_path.unlink()

            self._initial_qc_approved = approved
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

        return any(sel.selection_reason in ("manual", "yeo_fallback") for sel in selections.values())

    def _check_reference_available(self) -> tuple[bool, str]:
        """Check if a reference image is available for MNI registration.

        Returns:
            (available, description) tuple
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        task_name = _loc(config).get("type", "rest")

        # Check boldref from preprocessing
        preprocessing_dir = base_dir / "preprocessing"
        boldref = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_boldref.nii.gz"
        if boldref.exists():
            return True, "boldref from preprocessing"

        # Check initial reference DICOMs
        collection_dir = base_dir / "collection"
        ref_dirs = sorted(collection_dir.glob("task-reference_run-01_attempt-*"))
        for ref_dir in reversed(ref_dirs):
            if any(ref_dir.iterdir()):
                return True, f"initial reference scan ({ref_dir.name})"

        return False, ""

    def _confirm_mni_fallback(self, from_stage: str):
        """Show confirmation dialog before switching to MNI atlas masks.

        Args:
            from_stage: Which stage triggered the fallback ("localizer" or "mask_extraction")
        """
        if self._is_running:
            return

        has_ref, ref_source = self._check_reference_available()

        if not has_ref:
            # No reference available
            dialog = ft.AlertDialog(
                title=ft.Text("Reference Scan Required"),
                content=ft.Text(
                    "MNI atlas mask extraction requires a reference scan for "
                    "registration to subject space.\n\n"
                    "Please collect an initial reference scan first, or run "
                    "preprocessing to generate a BOLD reference."
                ),
                actions=[ft.TextButton("OK", on_click=lambda e: self._close_mni_dialog())],
            )
            self._mni_dialog = dialog
            self.page.overlay.append(dialog)
            dialog.open = True
            self.page.update()
            return

        config = self.study_config.session_config
        atlas_name = _loc(config).get("extraction", {}).get("network_atlas", "yeo_17")
        targets = (
            config["localizers"][0].get("networks", ["DefaultA", "ContA"])
            if config.get("localizers")
            else ["DefaultA", "ContA"]
        )
        targets_str = ", ".join(targets) if targets else "(none configured)"

        dialog = ft.AlertDialog(
            title=ft.Text("Use MNI Atlas Masks?"),
            content=ft.Column(
                [
                    ft.Text(
                        "This will skip the localizer/ICA pipeline and create masks "
                        "directly from the MNI atlas registered to subject space."
                    ),
                    ft.Container(height=10),
                    ft.Text(f"Reference: {ref_source}", size=12, color=ft.Colors.GREY_600),
                    ft.Text(f"Atlas: {atlas_name}", size=12, color=ft.Colors.GREY_600),
                    ft.Text(f"Networks: {targets_str}", size=12, color=ft.Colors.GREY_600),
                    ft.Container(height=10),
                    ft.Text(
                        "Note: Atlas masks reflect population-average network boundaries, "
                        "not individual functional topography. They are less specific than "
                        "ICA-derived masks.",
                        size=11,
                        italic=True,
                        color=ft.Colors.ORANGE_700,
                    ),
                ],
                tight=True,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_mni_dialog()),
                ft.FilledButton(
                    "Use MNI Masks",
                    on_click=lambda e: self._execute_mni_fallback(from_stage),
                ),
            ],
        )
        self._mni_dialog = dialog
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _close_mni_dialog(self):
        """Close the MNI fallback dialog."""
        if hasattr(self, "_mni_dialog") and self._mni_dialog:
            self._mni_dialog.open = False
            self.page.update()

    def _execute_mni_fallback(self, from_stage: str):
        """Execute the MNI atlas mask extraction after user confirmation.

        Args:
            from_stage: Which stage triggered it ("localizer" or "mask_extraction")
        """
        self._close_mni_dialog()

        # Switch to MNI atlas mode
        self._is_mni_atlas_mode = True

        # Mark skipped steps as complete
        config = self.study_config.session_config
        localizers = config.get("localizers", [])

        if from_stage == "localizer":
            # Skipping from localizer: mark all localizer runs as complete
            if localizers and len(localizers) > 1:
                for loc_idx, loc in enumerate(localizers):
                    for run_num in range(1, loc.get("n_runs", 2) + 1):
                        step_id = f"localizer_{loc_idx}_run_{run_num}"
                        if self.step_statuses.get(step_id) != StepStatus.COMPLETE:
                            self.step_statuses[step_id] = StepStatus.COMPLETE
            else:
                n_localizer_runs = config["localizers"][0].get("n_runs", 2) if config.get("localizers") else 2
                for run_num in range(1, n_localizer_runs + 1):
                    step_id = f"rest_run_{run_num}"
                    if self.step_statuses.get(step_id) != StepStatus.COMPLETE:
                        self.step_statuses[step_id] = StepStatus.COMPLETE

        # Mark preprocessing and standard mask extraction as complete (skipped)
        if localizers and len(localizers) > 1:
            for loc_idx in range(len(localizers)):
                for suffix in ("_preprocessing", "_extraction"):
                    step_id = f"localizer_{loc_idx}{suffix}"
                    if self.step_statuses.get(step_id) != StepStatus.COMPLETE:
                        self.step_statuses[step_id] = StepStatus.COMPLETE
        else:
            if self.step_statuses.get("preprocessing") != StepStatus.COMPLETE:
                self.step_statuses["preprocessing"] = StepStatus.COMPLETE
            if self.step_statuses.get("mask_extraction") != StepStatus.COMPLETE:
                self.step_statuses["mask_extraction"] = StepStatus.COMPLETE

        # Revoke any prior mask approval (new masks need fresh review)
        self._set_qc_approved("initial", False)

        # Register the MNI step as pending
        self.step_statuses["mni_mask_extraction"] = StepStatus.PENDING

        # Rebuild the step list UI and start the MNI extraction immediately
        # Force overwrite — we're deliberately replacing existing masks
        self._refresh_step_list()
        self._pending_overwrite = True
        self._start_step("mni_mask_extraction")

    def _revert_from_mni(self):
        """Revert from MNI atlas mode back to standard mask extraction.

        Re-runs the original mask extraction pipeline with overwrite=True
        so ICA/GLM masks replace the MNI atlas masks.
        """
        if self._is_running:
            return

        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"

        # Check that ICA/GLM outputs exist to revert to
        loc_type = _loc(config).get("type", "rest")
        if loc_type == "rest":
            ica_dir = mask_extraction_dir / "01_ica"
            has_original = ica_dir.exists() and any(ica_dir.glob("*_components.nii.gz"))
        else:
            glm_dir = mask_extraction_dir / "01_glm"
            has_original = glm_dir.exists() and any(glm_dir.glob("*.nii.gz"))

        if not has_original:
            _snack = ft.SnackBar(
                ft.Text("No original ICA/GLM outputs found. Run the full localizer pipeline first."),
                bgcolor=ft.Colors.ORANGE,
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Switch back to standard mode
        self._is_mni_atlas_mode = False
        self._set_qc_approved("initial", False)

        # Reset mask extraction to pending so user can re-run it
        self.step_statuses["mask_extraction"] = StepStatus.PENDING
        self.step_statuses.pop("mni_mask_extraction", None)

        self._add_log("[INFO] Reverted to standard mask extraction mode")
        self._refresh_step_list()
        self._update_run_next_button()

        # Start mask extraction immediately with overwrite
        self._pending_overwrite = True
        self._start_step("mask_extraction")

    def _show_mask_review_dialog(self, loc_idx: int | None = None):
        """Show the mask review/selection dialog.

        Args:
            loc_idx: For multi-localizer, the localizer index to review.
                     None for single-localizer mode.
        """
        from glob import glob

        import pandas as pd  # type: ignore[import-untyped]

        from pineuro import qc
        from pineuro.mask_extraction import get_alternative_components, load_mask_selections

        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"
        qc_dir = base_dir / "qc"
        config = self.study_config.session_config

        # For multi-localizer, use per-localizer subdirectory
        localizers = config.get("localizers", [])
        if loc_idx is not None and localizers and len(localizers) > 1:
            loc = localizers[loc_idx]
            loc_type = loc.get("type", "rest")
            loc_task_name = "rest" if loc_type == "rest" else loc_type
            mask_extraction_dir = base_dir / "mask_extraction" / f"loc_{loc_idx}_{loc_task_name}"
            qc_dir = base_dir / "qc" / f"loc_{loc_idx}_{loc_task_name}"

        # Detect mask source from mask_selections.json (authoritative, not just instance var)
        masks_dir = mask_extraction_dir / "03_masks"
        selections_file = masks_dir / "mask_selections.json"
        if selections_file.exists():
            import json as _json

            try:
                _sel_data = _json.loads(selections_file.read_text())
                _sel_dicts = [v for v in _sel_data.values() if isinstance(v, dict)]
                if any(v.get("source_type") == "mni_atlas" for v in _sel_dicts):
                    # Ensure instance state is consistent
                    self._is_mni_atlas_mode = True
                    self._show_mni_mask_review_dialog(config, base_dir, mask_extraction_dir, qc_dir)
                    return
                if any(v.get("source_type") == "manual_import" for v in _sel_dicts):
                    # Manual import reuses the simple card/overlay review dialog
                    self._show_mni_mask_review_dialog(
                        config,
                        base_dir,
                        mask_extraction_dir,
                        qc_dir,
                        loc_idx=loc_idx,
                        source_label="Manual import",
                    )
                    return
            except (ValueError, AttributeError):
                pass

        # MNI atlas mode flag (set during this session before selections exist on disk)
        if self._is_mni_atlas_mode:
            self._show_mni_mask_review_dialog(config, base_dir, mask_extraction_dir, qc_dir, loc_idx=loc_idx)
            return

        # Task-based localizers use a simpler review flow (no ICA component selection)
        idx = loc_idx if (loc_idx is not None and len(localizers) > 1) else 0
        loc_type = _loc(config, idx).get("type", "rest")
        is_task_localizer = loc_type not in ("rest", "manual")
        if is_task_localizer:
            self._show_task_mask_review_dialog(config, base_dir, mask_extraction_dir, qc_dir, loc_idx=loc_idx)
            return

        masks_dir = mask_extraction_dir / "03_masks"
        ica_dir = mask_extraction_dir / "01_ica"

        # Get network names — for multi-localizer, use the specific localizer's targets
        if loc_idx is not None and localizers and len(localizers) > 1:
            loc = localizers[loc_idx]
            network_names = list(loc.get("networks", loc.get("networks", loc.get("yeo_targets", []))))
            n_yeo = loc.get(
                "n_yeo",
                (config["localizers"][0].get("extraction", {}).get("n_yeo", 17) if config.get("localizers") else 17),
            )
            threshold = loc.get(
                "threshold",
                (
                    config["localizers"][0].get("extraction", {}).get("threshold", 2000)
                    if config.get("localizers")
                    else 2000
                ),
            )
        else:
            network_names = list(config["localizers"][0]["networks"])  # Copy to avoid modifying original
            n_yeo = config["localizers"][0]["extraction"]["n_yeo"]
            threshold = (
                config["localizers"][0].get("extraction", {}).get("threshold", 2000)
                if config.get("localizers")
                else 2000
            )

        # Include sham alternative networks if sham is enabled with alternative masks
        sham_config = config.get("sham", {})
        sham_hemisphere_network = None
        sham_target_hemisphere = "right"  # Default
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            sham_mask_mode = sham_config.get("sham_mask_mode", "two_networks")
            if sham_mask_mode == "hemisphere_split":
                # Hemisphere split mode - record the network for overlay generation
                sham_hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                sham_target_hemisphere = sham_config.get("target_hemisphere", "right")
                # Make sure the network is in the list for ICA selection
                if sham_hemisphere_network and sham_hemisphere_network not in network_names:
                    network_names.append(sham_hemisphere_network)
            else:
                # Two networks mode (legacy)
                alternative_targets = sham_config.get("alternative_mask_targets", [])
                for target in alternative_targets:
                    if target and target not in network_names:
                        network_names.append(target)

        # Set network order on QC manager so overlay plots match config order
        self.qc_manager.network_order = network_names

        # Load correlation matrix
        corr_file = masks_dir / "network_IC_spatial_correlation.csv"
        if not corr_file.exists():
            corr_file = masks_dir / "Yeo_IC_spatial_correlation.csv"
        if not corr_file.exists():
            _snack = ft.SnackBar(
                ft.Text("Correlation matrix not found. Run mask extraction first."), bgcolor=ft.Colors.ORANGE
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        corr_df = pd.read_csv(corr_file)

        # Load existing selections
        selections = load_mask_selections(masks_dir)

        # Find required files
        components_files = sorted(glob(str(ica_dir / "*_components.nii.gz")))
        if not components_files:
            _snack = ft.SnackBar(
                ft.Text("ICA components not found. Run mask extraction first."), bgcolor=ft.Colors.ORANGE
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return
        components_file = Path(components_files[0])

        yeo2sub_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "Yeo_to_*.nii.gz")))
        if not yeo2sub_files:
            _snack = ft.SnackBar(ft.Text("Yeo atlas in subject space not found."), bgcolor=ft.Colors.ORANGE)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return
        yeo_subject_space = Path(yeo2sub_files[0])

        # Resolve ref_file early (needed for overlay generation and script context)
        preprocessing_dir = base_dir / "preprocessing"
        idx = loc_idx if (loc_idx is not None and len(localizers) > 1) else 0
        loc_task = _loc(config, idx).get("type", "rest")
        ref_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task}_boldref.nii.gz"

        # Generate overlays only for RECOMMENDED (best) components initially
        # Alternative overlays will be generated on-demand when user rejects a mask
        n_components = (
            config["localizers"][0].get("extraction", {}).get("n_components", 30) if config.get("localizers") else 30
        )
        self._add_log("[INFO] Generating overlays for recommended ICA components...")
        self.page.update()

        alternative_overlays = qc.generate_alternative_overlays(
            mask_extraction_dir=mask_extraction_dir,
            participant_id=self.participant_id,
            session_id=self.session_id,
            network_names=network_names,
            n_alternatives=1,  # Only generate overlay for top component per network
            output_dir=qc_dir,
            n_yeo=n_yeo,
            ref_file=ref_file if ref_file.exists() else None,
        )

        # Yeo overlays will be generated on-demand when user rejects a mask
        yeo_overlays: dict[str, Path] = {}

        # Generate hemisphere overlays for sham verification (if hemisphere_split mode)
        hemisphere_overlays = {}
        if sham_hemisphere_network:
            self._add_log(f"[INFO] Generating hemisphere overlays for {sham_hemisphere_network}...")
            self.page.update()
            hemisphere_overlays = qc.generate_hemisphere_overlays(
                mask_extraction_dir=mask_extraction_dir,
                participant_id=self.participant_id,
                session_id=self.session_id,
                hemisphere_network=sham_hemisphere_network,
                output_dir=qc_dir,
                n_yeo=n_yeo,
                ref_file=ref_file if ref_file.exists() else None,
            )

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
            n_components = (
                config["localizers"][0].get("extraction", {}).get("n_components", 30)
                if config.get("localizers")
                else 30
            )
            alternatives = get_alternative_components(corr_df, network, top_n=n_components)

            # Find overlay paths
            overlay_candidate = overlays_dir / f"{self._bids_prefix}_{network}_overlay.png"
            overlay_path: Path | None = overlay_candidate if overlay_candidate.exists() else None

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

        def on_complete(new_selections, _loc_idx=loc_idx):
            # Set mask approval when user completes review
            self._set_qc_approved("initial", True, loc_idx=_loc_idx)
            loc_label = f" (Localizer {_loc_idx + 1})" if _loc_idx is not None else ""
            self._add_log(f"[SUCCESS] Masks reviewed and approved{loc_label}")
            self._refresh_step_list()
            self._update_run_next_button()
            _snack = ft.SnackBar(ft.Text(f"Masks approved{loc_label}!"), bgcolor=ft.Colors.GREEN)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        # Find preprocessing outputs for script generation
        # (preprocessing_dir, loc_task, ref_file already resolved above)
        input_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task}_preproc.nii.gz"
        brain_mask_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task}_preproc_mask.nii.gz"

        # Get ICA parameters from config
        n_components = (
            config["localizers"][0].get("extraction", {}).get("n_components", 30) if config.get("localizers") else 30
        )
        smooth_fwhm = (
            config["localizers"][0].get("extraction", {}).get("smooth_fwhm") if config.get("localizers") else None
        )
        highpass = config["localizers"][0].get("extraction", {}).get("highpass") if config.get("localizers") else None
        lowpass = config["localizers"][0].get("extraction", {}).get("lowpass") if config.get("localizers") else None

        # Generate tissue mask overlays (WM/CSF) if they exist
        tissue_mask_overlays: dict[str, Path] = {}
        tissue_dir = mask_extraction_dir / "05_tissue"
        if tissue_dir.exists() and ref_file.exists():
            overlays_dir = qc_dir / "overlays" / "subject"
            overlays_dir.mkdir(parents=True, exist_ok=True)
            for tissue_type in ["wm", "csf"]:
                tissue_files = list(tissue_dir.glob(f"*_tissue_{tissue_type}_mask.nii.gz"))
                if not tissue_files:
                    continue
                overlay_path = overlays_dir / f"{self._bids_prefix}_tissue_{tissue_type}_overlay.png"
                if not overlay_path.exists():
                    try:
                        from pineuro.qc import plot_mask_overlay_tripanel, select_slices_from_atlas

                        slices = select_slices_from_atlas(
                            atlas_path=ref_file,
                            labels=[],
                            target_label="",
                            top_n_per_plane=1,
                            mask_path=tissue_files[0],
                        )
                        label = "White Matter" if tissue_type == "wm" else "CSF"
                        plot_mask_overlay_tripanel(
                            mask_path=tissue_files[0],
                            ref_path=ref_file,
                            out_path=overlay_path,
                            slices=slices,
                            title=f"{label} mask (MNI template prior)",
                        )
                    except Exception as ex:
                        logger.warning(f"Failed to generate tissue overlay for {tissue_type}: {ex}")
                if overlay_path.exists():
                    tissue_mask_overlays[tissue_type] = overlay_path

        # Get full HTML QC report path (if available)
        html_report_path = self.qc_manager.get_mask_extraction_qc()

        dialog = MaskSelectionDialog(
            page=self.page,
            masks_info=masks_info,
            participant_id=self.participant_id,
            session_id=self.session_id,
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
            # Hemisphere split overlay info (for sham verification)
            hemisphere_network=sham_hemisphere_network,
            hemisphere_overlays=hemisphere_overlays,
            target_hemisphere=sham_target_hemisphere,
            # Tissue mask overlays (WM/CSF nuisance regression)
            tissue_mask_overlays=tissue_mask_overlays,
            # Full HTML QC report
            html_report_path=html_report_path,
        )
        dialog.show()

    def _show_task_mask_review_dialog(
        self, config: dict, base_dir: Path, mask_extraction_dir: Path, qc_dir: Path, loc_idx: int | None = None
    ):
        """Show mask review dialog for task-based (GLM) mask extraction.

        Task-based masks don't have ICA components to select — the user just
        reviews the thresholded GLM contrast masks and approves or rejects.
        """
        import base64

        from pineuro.gui.components.qc_viewer import open_qc_report

        masks_dir = mask_extraction_dir / "03_masks"
        if not masks_dir.exists():
            _snack = ft.SnackBar(ft.Text("Task masks not found. Run mask extraction first."), bgcolor=ft.Colors.ORANGE)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        mask_files = sorted(
            masks_dir.glob(f"{self._bids_prefix}_*_thr.nii.gz"),
            key=lambda p: ("_reverse" in p.name, p.name),  # Forward first, reverse after
        )
        if not mask_files:
            _snack = ft.SnackBar(
                ft.Text("No thresholded masks found. Run mask extraction first."), bgcolor=ft.Colors.ORANGE
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Get reference file for mask viewer
        localizers = config.get("localizers", [])
        idx = loc_idx if (loc_idx is not None and len(localizers) > 1) else 0
        loc_task = _loc(config, idx).get("type", "rest")
        preprocessing_dir = base_dir / "preprocessing"
        ref_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task}_boldref.nii.gz"

        # Find QC report
        qc_report_path = qc_dir / f"{self._bids_prefix}_mask_extraction_qc.html"
        qc_report: Path | None = qc_report_path
        if not qc_report_path.exists():
            # Try glob fallback
            matches = list(qc_dir.glob("*mask_extraction_qc.html"))
            qc_report = matches[0] if matches else None

        # Load mask info from selections JSON
        import json

        selections_data = {}
        selections_file = masks_dir / "mask_selections.json"
        if selections_file.exists():
            with open(selections_file) as f:
                raw = json.load(f)
            # Task selections have a "selections" wrapper
            selections_data = raw.get("selections", raw)

        # Build mask cards with overlay images
        overlays_dir = qc_dir / "overlays" / "subject"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        mask_cards: list[ft.Control] = []
        added_reverse_divider = False
        for mask_file in mask_files:
            # Strip full .nii.gz suffix then participant prefix and _thr suffix
            bare = mask_file.name.replace(".nii.gz", "").replace(".nii", "")
            mask_name = bare.replace(f"{self._bids_prefix}_", "").replace("_thr", "")

            # Add divider before first reverse contrast mask
            if "_reverse" in mask_name and not added_reverse_divider:
                added_reverse_divider = True
                mask_cards.append(ft.Divider(height=2, color=ft.Colors.ORANGE_200))
                mask_cards.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.SWAP_VERT, color=ft.Colors.ORANGE, size=16),
                                ft.Text(
                                    "Reverse Contrasts",
                                    size=12,
                                    color=ft.Colors.ORANGE_700,
                                    weight=ft.FontWeight.W_500,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=ft.padding.only(left=5, bottom=5),
                    )
                )

            # Get stats from selections
            sel = selections_data.get(mask_name, {})
            parts = []
            contrast = sel.get("contrast_definition", "")
            if contrast:
                parts.append(f"Contrast: {contrast}")
            n_vox = sel.get("n_voxels_thresholded")
            if n_vox is not None:
                parts.append(f"{n_vox} voxels")
            peak = sel.get("peak_stat")
            if peak is not None:
                parts.append(f"peak t = {peak:.2f}")
            stat_text = "  |  ".join(parts)

            # Find overlay image — generate on-demand if missing or stale (mask newer than overlay)
            overlay_path = overlays_dir / f"{self._bids_prefix}_{mask_name}_overlay.png"
            needs_overlay = not overlay_path.exists() or (
                overlay_path.exists()
                and mask_file.exists()
                and mask_file.stat().st_mtime > overlay_path.stat().st_mtime
            )
            if needs_overlay and ref_file.exists() and mask_file.exists():
                try:
                    from pineuro.qc import plot_mask_overlay_tripanel, select_slices_from_atlas, summarize_mask

                    summary = summarize_mask(mask_file, name=mask_name)
                    atlas_slices = select_slices_from_atlas(
                        atlas_path=ref_file,
                        labels=[],
                        target_label="",
                        top_n_per_plane=1,
                        mask_path=mask_file,
                    )
                    plot_mask_overlay_tripanel(
                        mask_path=mask_file,
                        ref_path=ref_file,
                        out_path=overlay_path,
                        slices=atlas_slices,
                        title=f"{mask_name}: {sel.get('contrast_definition', '')} ({summary.voxels} voxels)",
                    )
                except Exception as ex:
                    logger.warning(f"Failed to generate overlay for {mask_name}: {ex}")

            if overlay_path.exists():
                b64 = base64.b64encode(overlay_path.read_bytes()).decode("utf-8")
                overlay_content: ft.Control = ft.Image(
                    src=f"data:image/png;base64,{b64}",
                    fit=ft.BoxFit.CONTAIN,
                    height=150,
                )
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=ft.Colors.GREY_500),
                    height=150,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=ft.Colors.GREY_200,
                )

            # Build collapsible cluster table widget if available
            cluster_table_widget: ft.Control | None = None
            cluster_table_data = sel.get("cluster_table")
            if cluster_table_data:
                max_display = 15
                table_rows = []
                for c in cluster_table_data[:max_display]:
                    region = c.get("anatomical_label", "")
                    table_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text(str(c.get("cluster_id", "")), size=11)),
                                ft.DataCell(ft.Text(str(c.get("n_voxels", "")), size=11)),
                                ft.DataCell(ft.Text(f"{c.get('peak_stat', 0):.2f}", size=11)),
                                ft.DataCell(ft.Text(region if region else "—", size=11)),
                            ],
                        )
                    )
                if len(cluster_table_data) > max_display:
                    table_rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text("...", size=11, italic=True)),
                                ft.DataCell(
                                    ft.Text(f"+{len(cluster_table_data) - max_display} more", size=11, italic=True)
                                ),
                                ft.DataCell(ft.Text("")),
                                ft.DataCell(ft.Text("")),
                            ],
                        )
                    )
                table_content = ft.Container(
                    content=ft.DataTable(
                        columns=[
                            ft.DataColumn(ft.Text("#", size=11, weight=ft.FontWeight.BOLD)),  # type: ignore[arg-type, call-arg]
                            ft.DataColumn(ft.Text("Voxels", size=11, weight=ft.FontWeight.BOLD), numeric=True),  # type: ignore[arg-type, call-arg]
                            ft.DataColumn(ft.Text("Peak", size=11, weight=ft.FontWeight.BOLD), numeric=True),  # type: ignore[arg-type, call-arg]
                            ft.DataColumn(ft.Text("Region", size=11, weight=ft.FontWeight.BOLD)),  # type: ignore[arg-type, call-arg]
                        ],
                        rows=table_rows,
                        column_spacing=15,
                        data_row_min_height=28,
                        heading_row_height=30,
                    ),
                    visible=False,
                    padding=ft.padding.symmetric(horizontal=5),
                    animate_opacity=ft.Animation(200),
                )

                def _toggle_table(e, tc=table_content):
                    tc.visible = not tc.visible
                    # Update the chevron icon
                    e.control.icon = ft.Icons.EXPAND_LESS if tc.visible else ft.Icons.EXPAND_MORE
                    tc.update()
                    e.control.update()

                cluster_table_widget = ft.Column(
                    [
                        ft.TextButton(
                            f"Clusters ({len(cluster_table_data)})",
                            icon=ft.Icons.EXPAND_MORE,
                            style=ft.ButtonStyle(color=ft.Colors.BLUE_700),
                            on_click=_toggle_table,
                        ),
                        table_content,
                    ],
                    spacing=0,
                )

            card_children: list[ft.Control] = [
                ft.Row(
                    [
                        ft.Text(mask_name, size=16, weight=ft.FontWeight.W_500),
                        ft.Container(expand=True),
                        ft.Text(stat_text, size=13, color=ft.Colors.GREY_700),
                    ],
                ),
                ft.Container(
                    content=overlay_content,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=5,
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                ),
            ]
            if cluster_table_widget is not None:
                card_children.append(cluster_table_widget)

            card = ft.Container(
                content=ft.Column(card_children, spacing=10),
                padding=15,
                bgcolor=ft.Colors.GREY_100,
                border_radius=8,
                border=ft.border.all(2, ft.Colors.GREY_300),
            )
            mask_cards.append(card)

        # Add tissue mask overlays (WM/CSF) as view-only cards if they exist
        tissue_dir = mask_extraction_dir / "05_tissue"
        if tissue_dir.exists():
            tissue_labels = {"wm": "White Matter", "csf": "CSF"}
            tissue_files = list(tissue_dir.glob(f"{self._bids_prefix}_tissue_*_mask.nii.gz"))
            if tissue_files:
                mask_cards.append(ft.Divider(height=2, color=ft.Colors.PURPLE_200))
                mask_cards.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.PURPLE, size=16),
                                ft.Text(
                                    "Tissue Masks (nuisance regression, view only)",
                                    size=12,
                                    color=ft.Colors.PURPLE_700,
                                    weight=ft.FontWeight.W_500,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=ft.padding.only(left=5, bottom=5),
                    )
                )
                for tissue_file in tissue_files:
                    # Determine tissue type from filename
                    tissue_type = "wm" if "_wm_" in tissue_file.name else "csf"
                    tissue_label = tissue_labels.get(tissue_type, tissue_type.upper())

                    # Generate overlay on-demand
                    tissue_overlay = overlays_dir / f"{self._bids_prefix}_tissue_{tissue_type}_overlay.png"
                    if not tissue_overlay.exists() and ref_file.exists():
                        try:
                            from pineuro.qc import plot_mask_overlay_tripanel, select_slices_from_atlas

                            atlas_slices = select_slices_from_atlas(
                                atlas_path=ref_file,
                                labels=[],
                                target_label="",
                                top_n_per_plane=1,
                                mask_path=tissue_file,
                            )
                            plot_mask_overlay_tripanel(
                                mask_path=tissue_file,
                                ref_path=ref_file,
                                out_path=tissue_overlay,
                                slices=atlas_slices,
                                title=f"{tissue_label} mask (MNI template prior)",
                            )
                        except Exception as ex:
                            logger.warning(f"Failed to generate tissue overlay for {tissue_type}: {ex}")

                    if tissue_overlay.exists():
                        t_b64 = base64.b64encode(tissue_overlay.read_bytes()).decode("utf-8")
                        t_overlay_content: ft.Control = ft.Image(
                            src=f"data:image/png;base64,{t_b64}",
                            fit=ft.BoxFit.CONTAIN,
                            height=120,
                        )
                    else:
                        t_overlay_content = ft.Container(
                            content=ft.Text("No preview available", color=ft.Colors.GREY_500),
                            height=120,
                            alignment=ft.Alignment(0, 0),
                            bgcolor=ft.Colors.GREY_200,
                        )

                    tissue_card = ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Text(tissue_label, size=14, weight=ft.FontWeight.W_500),
                                        ft.Container(expand=True),
                                        ft.Text("MNI template prior", size=11, color=ft.Colors.PURPLE_400),
                                    ]
                                ),
                                ft.Container(
                                    content=t_overlay_content,
                                    border=ft.border.all(1, ft.Colors.GREY_300),
                                    border_radius=5,
                                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=12,
                        bgcolor=ft.Colors.PURPLE_50,
                        border_radius=6,
                        border=ft.border.all(1, ft.Colors.PURPLE_200),
                    )
                    mask_cards.append(tissue_card)

        def _close_dialog(dlg):
            dlg.open = False
            self.page.update()

        def on_approve(e, _loc_idx=loc_idx):
            self._set_qc_approved("initial", True, loc_idx=_loc_idx)
            loc_label = f" (Localizer {_loc_idx + 1})" if _loc_idx is not None else ""
            self._add_log(f"[SUCCESS] Task masks reviewed and approved{loc_label}")
            # Close dialog using the same pattern as ICA's _close_dialog:
            # set open=False then page.update() immediately — do NOT remove from overlay
            dialog.open = False
            self.page.update()
            self._refresh_step_list()
            self._update_run_next_button()
            _snack = ft.SnackBar(ft.Text(f"Masks approved{loc_label}!"), bgcolor=ft.Colors.GREEN)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        def on_reject(e):
            """Stage 2: show atlas region overlay as alternative (like Yeo in ICA)."""
            _close_dialog(dialog)

            extraction = _loc(config).get("extraction", {})
            atlas_name = extraction.get("atlas_name")
            threshold_method = extraction.get("threshold_method", "top_n")
            if threshold_method != "cluster" or not atlas_name:
                self._add_log("[INFO] Masks rejected — re-run mask extraction with different parameters")
                return

            # Find atlas in subject space
            atlas_subject_file = masks_dir / f"{self._bids_prefix}_atlas_{atlas_name}.nii.gz"
            if not atlas_subject_file.exists():
                atlas_subject_file = mask_extraction_dir / "02_mnireg" / f"{atlas_name}_to_{self._bids_prefix}.nii.gz"

            if not atlas_subject_file.exists() or not ref_file.exists():
                self._add_log("[INFO] Masks rejected — atlas in subject space not available")
                return

            # Generate atlas overlay on demand
            from pineuro.qc import generate_atlas_region_overlay

            atlas_overlay_png = generate_atlas_region_overlay(
                atlas_subject_path=atlas_subject_file,
                ref_path=ref_file,
                output_dir=qc_dir,
                participant_id=self.participant_id,
                session_id=self.session_id,
                atlas_name=atlas_name,
            )

            if not atlas_overlay_png or not atlas_overlay_png.exists():
                self._add_log("[INFO] Masks rejected — could not generate atlas overlay")
                return

            # Build Stage 2 dialog: show atlas overlay, offer to use it
            b64_atlas = base64.b64encode(atlas_overlay_png.read_bytes()).decode("utf-8")

            def on_use_atlas(e, _loc_idx=loc_idx):
                """Replace cluster mask with the full atlas region mask."""
                import shutil as _shutil

                _close_dialog(stage2_dialog)

                for mask_file in mask_files:
                    _shutil.copy2(atlas_subject_file, mask_file)
                    self._add_log(f"[INFO] Replaced {mask_file.name} with atlas region mask")

                self._set_qc_approved("initial", True, loc_idx=_loc_idx)
                self._add_log("[SUCCESS] Using atlas region mask — approved")
                self._refresh_step_list()
                self._update_run_next_button()
                _snack = ft.SnackBar(ft.Text("Atlas mask applied and approved."), bgcolor=ft.Colors.GREEN)
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()

            def on_back(e):
                _close_dialog(stage2_dialog)
                # Re-show Stage 1 (avoid appending a duplicate if still present)
                if dialog not in self.page.overlay:
                    self.page.overlay.append(dialog)
                dialog.open = True
                self.page.update()

            def on_close_stage2(e):
                _close_dialog(stage2_dialog)

            stage2_dialog = ft.AlertDialog(
                modal=True,
                title=ft.Row(
                    [
                        ft.Icon(ft.Icons.MAP, color=ft.Colors.TEAL),
                        ft.Text("Alternative: Atlas Region Mask", expand=True),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            on_click=on_close_stage2,
                            tooltip="Close",
                        ),
                    ],
                ),
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "The cluster mask was rejected. You can use the full "
                                "atlas region (warped to subject space) as an alternative, "
                                "or go back to re-examine the cluster mask.",
                                size=13,
                                color=ft.Colors.GREY_700,
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Row(
                                            [
                                                ft.Text(
                                                    f"Atlas: {atlas_name}",
                                                    size=14,
                                                    weight=ft.FontWeight.W_500,
                                                ),
                                                ft.Container(expand=True),
                                                ft.Text(
                                                    "In subject space",
                                                    size=12,
                                                    color=ft.Colors.GREY_600,
                                                ),
                                            ],
                                        ),
                                        ft.Container(
                                            content=ft.Image(
                                                src=f"data:image/png;base64,{b64_atlas}",
                                                fit=ft.BoxFit.CONTAIN,
                                                height=180,
                                            ),
                                            border=ft.border.all(1, ft.Colors.TEAL_200),
                                            border_radius=5,
                                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                                        ),
                                    ],
                                    spacing=8,
                                ),
                                padding=15,
                                bgcolor=ft.Colors.TEAL_50,
                                border_radius=8,
                                border=ft.border.all(2, ft.Colors.TEAL_200),
                            ),
                        ],
                        spacing=15,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    width=650,
                    height=400,
                ),
                actions=[
                    ft.OutlinedButton(
                        "Back",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=on_back,
                    ),
                    ft.FilledButton(
                        "Use Atlas Mask",
                        icon=ft.Icons.CHECK,
                        on_click=on_use_atlas,
                    ),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
            )

            self.page.overlay.append(stage2_dialog)
            stage2_dialog.open = True
            self.page.update()

        def on_open_qc(e):
            if qc_report and qc_report.exists():
                open_qc_report(qc_report)
            else:
                _snack = ft.SnackBar(ft.Text("QC report not found."), bgcolor=ft.Colors.ORANGE)
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()

        def on_close(e):
            _close_dialog(dialog)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.BIOTECH, color=ft.Colors.BLUE),
                    ft.Text("Review Task Masks", expand=True),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        on_click=on_close,
                        tooltip="Close",
                    ),
                ],
            ),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            f"Extracted {len(mask_files)} mask(s) from GLM contrasts. "
                            "Review the overlays below, then approve or reject.",
                            size=13,
                            color=ft.Colors.GREY_700,
                        ),
                        *mask_cards,
                    ],
                    spacing=15,
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=650,
                height=500,
            ),
            actions=[
                ft.OutlinedButton(
                    "View QC Report",
                    icon=ft.Icons.ASSESSMENT,
                    on_click=on_open_qc,
                ),
                ft.OutlinedButton(
                    "Reject",
                    icon=ft.Icons.CLOSE,
                    on_click=on_reject,
                ),
                ft.FilledButton(
                    "Approve",
                    icon=ft.Icons.CHECK,
                    on_click=on_approve,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _show_mni_mask_review_dialog(
        self,
        config: dict,
        base_dir: Path,
        mask_extraction_dir: Path,
        qc_dir: Path,
        loc_idx: int | None = None,
        source_label: str = "MNI Atlas",
    ):
        """Show mask review dialog for atlas-derived or imported masks.

        MNI atlas masks and manually imported masks don't have ICA components
        or GLM contrasts — the user reviews the mask overlays on the reference
        and approves. ``source_label`` distinguishes the two in the UI.
        """
        import base64

        from pineuro.gui.components.qc_viewer import open_qc_report

        _step_hint = "mask import" if source_label == "Manual import" else "MNI mask extraction"

        masks_dir = mask_extraction_dir / "03_masks"
        if not masks_dir.exists():
            _snack = ft.SnackBar(ft.Text(f"Masks not found. Run {_step_hint} first."), bgcolor=ft.Colors.ORANGE)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        mask_files = sorted(masks_dir.glob(f"{self._bids_prefix}_*_thr.nii.gz"))
        if not mask_files:
            _snack = ft.SnackBar(
                ft.Text(f"No thresholded masks found. Run {_step_hint} first."), bgcolor=ft.Colors.ORANGE
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Load mask selections for stats (also records the manual reference path)
        import json

        selections_data = {}
        selections_file = masks_dir / "mask_selections.json"
        if selections_file.exists():
            with open(selections_file) as f:
                selections_data = json.load(f)

        # Find reference file (boldref, MNI ref, or the manual import's reference)
        from glob import glob

        preprocessing_dir = base_dir / "preprocessing"
        ref_files = sorted(glob(str(preprocessing_dir / "*boldref.nii.gz"))) if preprocessing_dir.exists() else []
        if not ref_files:
            mni_ref_dir = mask_extraction_dir / "00_mni_ref"
            if mni_ref_dir.exists():
                ref_files = sorted(glob(str(mni_ref_dir / "*.nii.gz")))
        ref_file = Path(ref_files[0]) if ref_files else None
        if ref_file is None:
            # Manual imports record the reference EPI in mask_selections.json
            manual_ref = selections_data.get("ref_file")
            if manual_ref and Path(manual_ref).exists():
                ref_file = Path(manual_ref)

        # Find QC report
        qc_report_path = qc_dir / f"{self._bids_prefix}_mask_extraction_qc.html"
        qc_report: Path | None = qc_report_path
        if not qc_report_path.exists():
            matches = list(qc_dir.glob("*mask_extraction_qc.html"))
            qc_report = matches[0] if matches else None

        atlas_name = _loc(config).get("extraction", {}).get("network_atlas", "yeo_17")

        # Build mask cards with overlay images
        overlays_dir = qc_dir / "overlays" / "subject"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        mask_cards: list[ft.Control] = []

        for mask_file in mask_files:
            bare = mask_file.name.replace(".nii.gz", "").replace(".nii", "")
            mask_name = bare.replace(f"{self._bids_prefix}_", "").replace("_thr", "")

            # Get stats from selections
            sel = selections_data.get(mask_name, {})
            n_voxels = sel.get("n_voxels_thresholded", "?")
            if source_label == "Manual import":
                space = sel.get("mask_space", selections_data.get("mask_space", "native"))
                stat_text = f"{n_voxels} voxels  |  imported ({space} space)"
            else:
                stat_text = f"{n_voxels} voxels  |  Atlas: {atlas_name}"

            # Find or generate overlay (regenerate if mask is newer than existing overlay)
            overlay_path = overlays_dir / f"{self._bids_prefix}_{mask_name}_overlay.png"
            needs_overlay = not overlay_path.exists() or (
                overlay_path.exists()
                and mask_file.exists()
                and mask_file.stat().st_mtime > overlay_path.stat().st_mtime
            )
            if needs_overlay and ref_file and ref_file.exists() and mask_file.exists():
                try:
                    from pineuro.qc import plot_mask_overlay_tripanel, select_slices_from_atlas, summarize_mask

                    summary = summarize_mask(mask_file, name=mask_name)
                    atlas_slices = select_slices_from_atlas(
                        atlas_path=ref_file,
                        labels=[],
                        target_label="",
                        top_n_per_plane=1,
                        mask_path=mask_file,
                    )
                    plot_mask_overlay_tripanel(
                        mask_path=mask_file,
                        ref_path=ref_file,
                        out_path=overlay_path,
                        slices=atlas_slices,
                        title=f"{mask_name} ({source_label}, {summary.voxels} voxels)",
                    )
                except Exception as ex:
                    logger.warning(f"Failed to generate overlay for {mask_name}: {ex}")

            if overlay_path.exists():
                b64 = base64.b64encode(overlay_path.read_bytes()).decode("utf-8")
                overlay_content: ft.Control = ft.Image(
                    src=f"data:image/png;base64,{b64}",
                    fit=ft.BoxFit.CONTAIN,
                    height=150,
                )
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=ft.Colors.GREY_500),
                    height=150,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=ft.Colors.GREY_200,
                )

            card = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text(mask_name, size=16, weight=ft.FontWeight.W_500),
                                ft.Container(expand=True),
                                ft.Text(stat_text, size=13, color=ft.Colors.GREY_700),
                            ],
                        ),
                        ft.Container(
                            content=overlay_content,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=5,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        ),
                    ],
                    spacing=10,
                ),
                padding=15,
                bgcolor=ft.Colors.GREY_100,
                border_radius=8,
                border=ft.border.all(2, ft.Colors.GREY_300),
            )
            mask_cards.append(card)

        def _close_dialog(dlg):
            dlg.open = False
            self.page.update()

        def on_approve(e, _loc_idx=loc_idx):
            self._set_qc_approved("initial", True, loc_idx=_loc_idx)
            self._add_log(f"[SUCCESS] {source_label} masks reviewed and approved")
            dialog.open = False
            self.page.update()
            self._refresh_step_list()
            self._update_run_next_button()
            _snack = ft.SnackBar(
                ft.Text(f"{source_label} masks approved! You can now proceed."), bgcolor=ft.Colors.GREEN
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        def on_reject(e):
            _close_dialog(dialog)
            self._add_log(f"[INFO] {source_label} masks rejected — re-run the mask step with different settings")

        def on_open_qc(e):
            if qc_report and qc_report.exists():
                open_qc_report(qc_report)
            else:
                _snack = ft.SnackBar(ft.Text("QC report not found."), bgcolor=ft.Colors.ORANGE)
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()

        def on_close(e):
            _close_dialog(dialog)

        is_manual_source = source_label == "Manual import"
        if is_manual_source:
            intro_text = (
                f"Imported {len(mask_files)} user-supplied mask(s), registered to the "
                "participant's functional space. Review the overlays below, then approve or reject."
            )
        else:
            intro_text = (
                f"Extracted {len(mask_files)} mask(s) from {atlas_name} atlas "
                "registered to subject space. Review the overlays below, then approve or reject."
            )
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.UPLOAD_FILE if is_manual_source else ft.Icons.MAP, color=ft.Colors.TEAL),
                    ft.Text(
                        "Review Imported Masks" if is_manual_source else "Review MNI Atlas Masks",
                        expand=True,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        on_click=on_close,
                        tooltip="Close",
                    ),
                ],
            ),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            intro_text,
                            size=13,
                            color=ft.Colors.GREY_700,
                        ),
                        *mask_cards,
                    ],
                    spacing=15,
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=650,
                height=500,
            ),
            actions=[
                ft.OutlinedButton(
                    "View QC Report",
                    icon=ft.Icons.ASSESSMENT,
                    on_click=on_open_qc,
                ),
                ft.OutlinedButton(
                    "Reject",
                    icon=ft.Icons.CLOSE,
                    on_click=on_reject,
                ),
                ft.FilledButton(
                    "Approve",
                    icon=ft.Icons.CHECK,
                    on_click=on_approve,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    # ------------------------------------------------------------------
    # Manual mask import — per-participant file configuration
    # ------------------------------------------------------------------

    def _manual_localizer_for_step(self, step_id: str) -> int | None:
        """Return the localizer index if step_id is a manual-mask import step.

        Handles the single-localizer ``mask_extraction`` step and the
        multi-localizer ``localizer_{idx}_extraction`` steps; returns None
        for any other step or non-manual localizer.
        """
        localizers = self.study_config.session_config.get("localizers", [])
        if not localizers:
            return None
        if step_id == "mask_extraction":
            return 0 if localizers[0].get("type") == "manual" else None
        if step_id.startswith("localizer_") and step_id.endswith("_extraction"):
            try:
                idx = int(step_id.split("_")[1])
            except (ValueError, IndexError):
                return None
            if 0 <= idx < len(localizers) and localizers[idx].get("type") == "manual":
                return idx
        return None

    def _show_manual_mask_config_dialog(self, loc_idx: int = 0, on_saved: Callable[[], None] | None = None):
        """Collect THIS participant's manual mask / reference / T1 files.

        The study config only carries the mask space and network names; the
        actual files differ per participant and are persisted here to
        ``base_dir/manual_masks.json`` (keyed by localizer id).

        Args:
            loc_idx: Index of the manual localizer being configured.
            on_saved: Called after a successful save (e.g. to start the import
                step). When None, the dialog only persists the selections.
        """
        from pineuro.gui.components.manual_masks import read_manual_masks, write_manual_masks
        from pineuro.gui.utils import pick_file

        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        localizers = config.get("localizers", [])
        if loc_idx >= len(localizers):
            return
        loc = localizers[loc_idx]
        loc_id = loc.get("id", "manual")
        mask_space = loc.get("manual_mask_space", "native")
        names = list(loc.get("manual_mask_names", []) or [])

        record = read_manual_masks(base_dir, loc_id) or {}
        existing_masks = record.get("masks", {}) or {}

        def _file_row(field: ft.TextField, title: str) -> ft.Row:
            def _pick(_e, _f: ft.TextField = field, _t: str = title) -> None:
                picked = pick_file(title=_t, filetypes=NIFTI_FILETYPES, initial_path=_f.value or None)
                if picked:
                    _f.value = picked
                    _f.update()

            return ft.Row([field, ft.IconButton(icon=ft.Icons.FILE_OPEN, tooltip="Browse", on_click=_pick)])

        # One file picker per network name
        mask_fields: dict[str, ft.TextField] = {}
        mask_rows: list[ft.Control] = []
        for name in names:
            field = ft.TextField(
                label=name,
                width=470,
                value=existing_masks.get(name, "") or "",
                read_only=True,
                hint_text="NIfTI mask file for this network",
            )
            mask_fields[name] = field
            mask_rows.append(_file_row(field, f"Select mask for {name}"))

        ref_field = ft.TextField(
            label="Reference EPI volume",
            width=470,
            value=record.get("reference", "") or "",
            read_only=True,
            hint_text="Native-space BOLD/EPI reference in this participant's scan space",
        )
        anat_field = ft.TextField(
            label="Anatomical (T1) image",
            width=470,
            value=record.get("anat", "") or "",
            read_only=True,
            hint_text="T1 the masks were drawn on",
        )
        error_text = ft.Text("", color=ft.Colors.RED, size=12, visible=False)

        content_items: list[ft.Control] = [
            ft.Text(
                f"Select this participant's mask files ({mask_space} space). They are saved "
                "with this participant and not shared across the study.",
                size=12,
                color=ft.Colors.GREY_700,
            ),
            ft.Divider(),
            ft.Text("Network masks", size=13, weight=ft.FontWeight.BOLD),
            *mask_rows,
            ft.Divider(),
            ft.Text("Reference EPI", size=13, weight=ft.FontWeight.BOLD),
            _file_row(ref_field, "Select reference EPI"),
        ]
        if mask_space == "anat":
            content_items += [
                ft.Divider(),
                ft.Text("Anatomical (T1)", size=13, weight=ft.FontWeight.BOLD),
                _file_row(anat_field, "Select T1 image"),
            ]
        content_items.append(error_text)

        def _close() -> None:
            dialog.open = False
            self.page.update()

        def _on_save(_e) -> None:
            missing: list[str] = []
            masks_out: dict[str, str] = {}
            for name in names:
                value = (mask_fields[name].value or "").strip()
                if not value or not Path(value).is_file():
                    missing.append(f"mask '{name}'")
                else:
                    masks_out[name] = value
            ref_value = (ref_field.value or "").strip()
            if not ref_value or not Path(ref_value).is_file():
                missing.append("reference EPI")
            anat_value = (anat_field.value or "").strip()
            if mask_space == "anat" and (not anat_value or not Path(anat_value).is_file()):
                missing.append("T1 image")

            if missing:
                error_text.value = "Select a valid existing file for: " + ", ".join(missing)
                error_text.visible = True
                error_text.update()
                return

            write_manual_masks(
                base_dir,
                loc_id,
                {
                    "mask_space": mask_space,
                    "masks": masks_out,
                    "reference": ref_value,
                    "anat": anat_value if mask_space == "anat" else None,
                },
            )
            self._add_log(f"[INFO] Manual mask files configured for localizer '{loc_id}'")
            _close()
            if on_saved is not None:
                on_saved()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.UPLOAD_FILE, color=ft.Colors.TEAL),
                    ft.Text(f"Manual Mask Files — {self.participant_id}", expand=True),
                ]
            ),
            content=ft.Container(
                content=ft.Column(content_items, spacing=10, scroll=ft.ScrollMode.AUTO),
                width=660,
                height=480,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: _close()),
                ft.FilledButton(
                    "Save & Import" if on_saved is not None else "Save",
                    icon=ft.Icons.SAVE,
                    on_click=_on_save,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

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

        # Count all files in directory - DICOM files from scanners often lack .dcm extension
        # (e.g., Siemens names them with UIDs like "MR.1.3.12.2...")
        dicom_count = len([f for f in collection_dir.iterdir() if f.is_file()])
        if dicom_count >= expected_volumes:
            return StepStatus.COMPLETE
        elif dicom_count > 0:
            return StepStatus.INTERRUPTED
        else:
            return StepStatus.PENDING

    def _check_attempt_collection_status(
        self, collection_dir: Path, task: str, run: int, expected_volumes: int
    ) -> StepStatus:
        """Check collection status across all attempts for a task+run.

        Scans for task-{task}_run-{run:02d}_attempt-* folders and returns COMPLETE
        if any attempt has enough volumes.

        Args:
            collection_dir: Parent collection directory
            task: Task prefix (e.g., 'task-rest', 'task-reference')
            run: Run number
            expected_volumes: Expected number of volumes

        Returns:
            StepStatus based on best available attempt
        """
        collections = PipelineRunner.find_available_collections(
            collection_dir=collection_dir,
            task=task,
            run=run,
            expected_volumes=expected_volumes,
        )

        if not collections:
            return StepStatus.PENDING

        # If any attempt is complete, the step is complete
        if any(c["is_complete"] for c in collections):
            return StepStatus.COMPLETE

        # Has data but none complete
        return StepStatus.INTERRUPTED

    def _get_unified_attempt_info(self) -> list[RunGroup]:
        """Get all attempts for all configured localizer runs.

        Returns:
            List of RunGroup objects with all attempts per run
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"
        n_localizer_runs = _loc(config).get("n_runs", 2)
        task_name = _loc(config).get("type", "rest")

        expected_vols = config["localizers"][0].get("n_volumes") if config.get("localizers") else None
        if expected_vols is None:
            rest_n_volumes = (
                [config["localizers"][0].get("n_volumes", 200)] * config["localizers"][0].get("n_runs", 1)
                if config.get("localizers")
                else [200]
            )
            expected_vols = rest_n_volumes[0] if isinstance(rest_n_volumes, list) else rest_n_volumes

        run_groups = []
        for run_num in range(1, n_localizer_runs + 1):
            attempts = scan_collection_attempts(
                collection_dir=collection_dir,
                task=f"task-{task_name}",
                run=run_num,
                expected_volumes=expected_vols,
            )
            run_groups.append(
                RunGroup(
                    run_id=run_num,
                    run_name=f"Localizer Run {run_num}",
                    attempts=attempts,
                )
            )

        return run_groups

    def _needs_unified_dialog(self) -> bool:
        """Check if the unified attempt selection dialog should be shown.

        Returns True if any run is incomplete/missing or has multiple attempts.
        """
        run_groups = self._get_unified_attempt_info()

        # No data at all — don't show
        if not any(rg.attempts for rg in run_groups):
            return False

        for rg in run_groups:
            # Run has no data
            if not rg.attempts:
                return True
            # Run has multiple attempts
            if len(rg.attempts) > 1:
                return True
            # Single attempt but incomplete
            if not rg.attempts[0].is_complete:
                return True

        return False

    def _show_unified_attempt_dialog(self, on_selected):
        """Show the unified attempt selection dialog.

        Args:
            on_selected: Callback invoked after user confirms selection
        """
        run_groups = self._get_unified_attempt_info()

        def on_confirm(selected_attempts: dict[int, list[int]]):
            self._selected_attempts = selected_attempts
            self._selected_rest_runs = list(selected_attempts.keys())
            self._add_log(f"Selected attempts: {selected_attempts}")
            on_selected(selected_attempts)

        def on_cancel():
            self._on_preview_cancel()

        dialog = UnifiedAttemptDialog(
            page=self.page,
            run_groups=run_groups,
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
            with open(log_path) as f:
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
        self, step_id: str, output_dir: Path, participant_id: str, session_id: str, expected_volumes: int
    ) -> StepStatus:
        """Check status of a feedback/transfer slot using attempt-based naming.

        Looks for any attempt-* files for the given slot's run number and determines
        if any attempt is complete or if there are interrupted attempts.

        Args:
            step_id: Workflow step ID (e.g., 'transfer_1', 'feedback_2')
            output_dir: Feedback output directory
            participant_id: Participant ID
            session_id: Session identifier (e.g., 'ses-01')
            expected_volumes: Expected number of volumes

        Returns:
            StepStatus based on best attempt status (COMPLETE if any complete,
            INTERRUPTED if any partial, PENDING if none found)
        """
        from pineuro.feedback import parse_step_run_number

        try:
            run_num = parse_step_run_number(step_id)
        except ValueError:
            return StepStatus.PENDING
        task_type = "transfer" if "transfer" in step_id else "feedback"

        if not output_dir.exists():
            return StepStatus.PENDING

        # Search for any attempt-* files for this slot's run number
        pattern = f"{participant_id}_{session_id}_task-{task_type}_run-{run_num:02d}_attempt-*_feedbacklog.tsv"
        log_files = list(output_dir.glob(pattern))

        if not log_files:
            return StepStatus.PENDING

        # Check each log file and return the best status
        has_complete = False
        has_interrupted = False

        for log_path in log_files:
            try:
                with open(log_path) as f:
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
        n_localizer_runs = config["localizers"][0].get("n_runs", 2) if config.get("localizers") else 2
        reference_initial_enabled = config.get("reference", {}).get("initial_enabled", True)
        reference_realtime_enabled = config.get("reference", {}).get("realtime_enabled", True)

        # Check if this is a mask-reuse session
        session_metadata = config.get("session", {})
        is_mask_reuse = session_metadata.get("mask_source") == "reuse"

        # Build dynamic steps list
        steps = []

        # Build feedback/transfer steps from feedback_runs or legacy config
        fb_steps = _build_feedback_step_ids(config)

        has_feedback = len(fb_steps) > 0
        localizers = config.get("localizers", [])
        has_localizers = len(localizers) > 1 or n_localizer_runs > 0

        # Store for section visibility in _refresh_step_list
        self._has_feedback = has_feedback
        self._has_localizers = has_localizers

        if is_mask_reuse:
            # Mask reuse mode: only need reference + coregistration + feedback
            # Skip localizer, preprocessing, mask extraction
            if reference_realtime_enabled:
                steps.extend(["reference_realtime", "coregistration_realtime"])
            # Add transfer and feedback steps
            steps.extend(sid for sid, _label in fb_steps)
        elif not has_localizers and has_feedback:
            # No localizers: MNI atlas mode
            self._is_mni_atlas_mode = True
            # Use reference_realtime as registration target (masks in realtime space)
            if reference_realtime_enabled:
                steps.append("reference_realtime")
            steps.extend(["mni_mask_extraction", "mask_approval"])
            # Add transfer and feedback steps (no coregistration needed —
            # masks are already in realtime reference space)
            steps.extend(sid for sid, _label in fb_steps)
        elif has_localizers and not has_feedback:
            # Localizer-only mode: collect localizers with QC monitoring, no feedback
            if reference_initial_enabled:
                steps.append("reference_initial")
            if localizers and len(localizers) > 1:
                for loc_idx, loc in enumerate(localizers):
                    if loc.get("type") == "manual":
                        continue
                    loc_n_runs = loc.get("n_runs", 2)
                    for run in range(1, loc_n_runs + 1):
                        steps.append(f"localizer_{loc_idx}_run_{run}")
            else:
                for i in range(1, n_localizer_runs + 1):
                    steps.append(f"rest_run_{i}")
        else:
            # Standard mode: full pipeline (localizers + feedback)
            if reference_initial_enabled:
                steps.append("reference_initial")

            if localizers and len(localizers) > 1:
                # Multi-localizer: collect ALL runs first, then per-localizer preprocess→extract→approve.
                # Manual localizers import pre-built masks — no collection or preprocessing.
                for loc_idx, loc in enumerate(localizers):
                    if loc.get("type") == "manual":
                        continue
                    loc_n_runs = loc.get("n_runs", 2)
                    for run in range(1, loc_n_runs + 1):
                        steps.append(f"localizer_{loc_idx}_run_{run}")
                for loc_idx, loc in enumerate(localizers):
                    if loc.get("type") != "manual":
                        steps.append(f"localizer_{loc_idx}_preprocessing")
                    steps.append(f"localizer_{loc_idx}_extraction")
                    steps.append(f"localizer_{loc_idx}_approval")
            elif config.get("localizers") and config["localizers"][0].get("type") == "manual":
                # Single manual localizer: import masks only, no collection/preprocessing
                steps.append("mask_extraction")
            else:
                # Single localizer (legacy or 1-element list)
                for i in range(1, n_localizer_runs + 1):
                    steps.append(f"rest_run_{i}")
                steps.extend(["preprocessing", "mask_extraction"])

            # Add coregistration only if initial reference is enabled
            if reference_initial_enabled:
                steps.append("coregistration")
            if reference_realtime_enabled:
                steps.extend(["reference_realtime", "coregistration_realtime"])
            # Add transfer and feedback steps
            steps.extend(sid for sid, _label in fb_steps)

        for step in steps:
            self.step_statuses[step] = StepStatus.PENDING

        # Only check localizer/preprocessing/mask extraction steps if NOT in mask reuse mode
        if not is_mask_reuse:
            # Check reference initial collection (only if enabled)
            collection_dir = base_dir / "collection"
            if reference_initial_enabled:
                n_vols = config.get("reference", {}).get("initial_n_volumes", 2)
                status = self._check_attempt_collection_status(collection_dir, "task-reference", 1, n_vols)
                self.step_statuses["reference_initial"] = status

            preprocessing_dir = base_dir / "preprocessing"
            mask_extraction_dir = base_dir / "mask_extraction"
            masks_dir = mask_extraction_dir / "03_masks"
            localizers = config.get("localizers", [])

            if localizers and len(localizers) > 1:
                # Multi-localizer: check per-localizer steps
                for loc_idx, loc in enumerate(localizers):
                    loc_type = loc.get("type", "rest")
                    loc_task_name = "rest" if loc_type == "rest" else loc_type
                    loc_dir = mask_extraction_dir / f"loc_{loc_idx}_{loc_task_name}"

                    # Manual localizers have no collection or preprocessing steps.
                    if loc_type != "manual":
                        loc_n_runs = loc.get("n_runs", 2)
                        loc_n_vols = loc.get("n_volumes", 250)

                        # Check collection steps
                        for run_num in range(1, loc_n_runs + 1):
                            step_id = f"localizer_{loc_idx}_run_{run_num}"
                            status = self._check_attempt_collection_status(
                                collection_dir,
                                f"task-{loc_task_name}",
                                run_num,
                                loc_n_vols,
                            )
                            self.step_statuses[step_id] = status

                        # Check preprocessing
                        step_id = f"localizer_{loc_idx}_preprocessing"
                        preproc_f = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task_name}_preproc.nii.gz"
                        mask_f = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task_name}_preproc_mask.nii.gz"
                        ref_f = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task_name}_boldref.nii.gz"
                        if preproc_f.exists() and mask_f.exists() and ref_f.exists():
                            self.step_statuses[step_id] = StepStatus.COMPLETE

                    # Check extraction (per-localizer subdirectory)
                    step_id = f"localizer_{loc_idx}_extraction"
                    loc_masks = loc_dir / "03_masks"
                    if loc_masks.exists() and len(list(loc_masks.glob(f"{self._bids_prefix}_*.nii.gz"))) > 0:
                        self.step_statuses[step_id] = StepStatus.COMPLETE

                    # Check per-localizer approval marker
                    approval_marker = loc_dir / ".masks_approved"
                    if approval_marker.exists():
                        self.step_statuses[f"localizer_{loc_idx}_approval"] = StepStatus.COMPLETE
            elif config.get("localizers") and config["localizers"][0].get("type") == "manual":
                # Single manual localizer: only the import (mask_extraction) step.
                if masks_dir.exists() and len(list(masks_dir.glob(f"{self._bids_prefix}_*.nii.gz"))) > 0:
                    self.step_statuses["mask_extraction"] = StepStatus.COMPLETE
            else:
                # Single localizer (legacy)
                expected_vols = config["localizers"][0].get("n_volumes") if config.get("localizers") else None
                if expected_vols is None:
                    rest_n_volumes = (
                        [config["localizers"][0].get("n_volumes", 200)] * config["localizers"][0].get("n_runs", 1)
                        if config.get("localizers")
                        else [200]
                    )
                    expected_vols = rest_n_volumes[0] if isinstance(rest_n_volumes, list) else rest_n_volumes

                loc_task_name = _loc(config).get("type", "rest")

                for run_num in range(1, n_localizer_runs + 1):
                    step_id = f"rest_run_{run_num}"
                    status = self._check_attempt_collection_status(
                        collection_dir,
                        f"task-{loc_task_name}",
                        run_num,
                        expected_vols,
                    )
                    self.step_statuses[step_id] = status

                preproc_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task_name}_preproc.nii.gz"
                mask_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task_name}_preproc_mask.nii.gz"
                ref_file = preprocessing_dir / f"{self._bids_prefix}_task-{loc_task_name}_boldref.nii.gz"
                if preproc_file.exists() and mask_file.exists() and ref_file.exists():
                    self.step_statuses["preprocessing"] = StepStatus.COMPLETE

                if masks_dir.exists() and len(list(masks_dir.glob(f"{self._bids_prefix}_*.nii.gz"))) > 0:
                    self.step_statuses["mask_extraction"] = StepStatus.COMPLETE

            # Check if masks were created via MNI atlas fallback
            selections_file = masks_dir / "mask_selections.json"
            if selections_file.exists():
                try:
                    import json

                    with open(selections_file) as f:
                        sel_data = json.load(f)
                    if any(s.get("source_type") == "mni_atlas" for s in sel_data.values() if isinstance(s, dict)):
                        self._is_mni_atlas_mode = True
                        self.step_statuses["mni_mask_extraction"] = StepStatus.COMPLETE
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Check mask approval status (single-localizer only; multi-localizer uses per-localizer approval)
            if not (localizers and len(localizers) > 1):
                if self._initial_qc_approved:
                    self.step_statuses["mask_approval"] = StepStatus.COMPLETE
                else:
                    self.step_statuses["mask_approval"] = StepStatus.PENDING

            # Check coregistration (reference) outputs - initial (run-01) is in mask_extraction/04_reference
            initial_reference_masks_dir = mask_extraction_dir / "04_reference" / "masks"

            if (
                initial_reference_masks_dir.exists()
                and len(list(initial_reference_masks_dir.glob("*_to_task-reference_run-01_*.nii.gz"))) > 0
            ):
                self.step_statuses["coregistration"] = StepStatus.COMPLETE

        # Define directories needed for remaining checks
        feedback_dir = base_dir / "feedback"
        if is_mask_reuse:
            mask_extraction_dir = base_dir / "mask_extraction"  # Still needed for path references

        # Check reference realtime collection (only if enabled)
        collection_dir = base_dir / "collection"
        if reference_realtime_enabled:
            # In mask reuse mode, reference is run-01 (no initial reference)
            # In standard mode, reference is run-02 (after initial reference)
            realtime_reference_run = 1 if is_mask_reuse else 2
            n_vols = config.get("reference", {}).get("realtime_n_volumes", 2)
            status = self._check_attempt_collection_status(
                collection_dir, "task-reference", realtime_reference_run, n_vols
            )
            self.step_statuses["reference_realtime"] = status

            # Check coregistration_realtime outputs - realtime is in feedback/01_reference
            realtime_reference_masks_dir = feedback_dir / "01_reference" / "masks"
            if (
                realtime_reference_masks_dir.exists()
                and len(
                    list(
                        realtime_reference_masks_dir.glob(
                            f"*_to_task-reference_run-{realtime_reference_run:02d}_*.nii.gz"
                        )
                    )
                )
                > 0
            ):
                self.step_statuses["coregistration_realtime"] = StepStatus.COMPLETE
        # Note: When reference realtime is disabled, those steps are not in the steps list,
        # so we don't need to mark them as complete

        # Check feedback runs (transfer and feedback) using slot-based naming
        feedback_output_dir = feedback_dir / "02_feedback_output"
        feedback_runs = config.get("feedback_runs", [])

        # Check each feedback/transfer step using the shared step builder
        for step_id, _label in fb_steps:
            # Resolve per-FRC n_volumes for this step
            frc_idx = self._get_feedback_step_frc_index(step_id)
            if frc_idx is not None and frc_idx < len(feedback_runs):
                expected_vols = feedback_runs[frc_idx].get("n_volumes", 150)
            else:
                expected_vols = 150

            status = self._check_feedback_slot_status(
                step_id=step_id,
                output_dir=feedback_output_dir,
                participant_id=participant_id,
                session_id=self.session_id,
                expected_volumes=expected_vols,
            )
            self.step_statuses[step_id] = status

        # --- Skip detection ---
        # If preprocessing is complete, any uncollected localizer-session steps
        # were intentionally bypassed (user proceeded with fewer runs).
        localizers = config.get("localizers", [])
        any_preproc_complete = False
        if localizers and len(localizers) > 1:
            any_preproc_complete = any(
                self.step_statuses.get(f"localizer_{i}_preprocessing") == StepStatus.COMPLETE
                for i in range(len(localizers))
            )
        else:
            any_preproc_complete = self.step_statuses.get("preprocessing") == StepStatus.COMPLETE

        if not is_mask_reuse and any_preproc_complete:
            if reference_initial_enabled and self.step_statuses.get("reference_initial") in (
                StepStatus.PENDING,
                StepStatus.INTERRUPTED,
            ):
                self.step_statuses["reference_initial"] = StepStatus.SKIPPED

            if localizers and len(localizers) > 1:
                for loc_idx, loc in enumerate(localizers):
                    loc_n_runs = loc.get("n_runs", 2)
                    for run_num in range(1, loc_n_runs + 1):
                        step_id = f"localizer_{loc_idx}_run_{run_num}"
                        if self.step_statuses.get(step_id) in (StepStatus.PENDING, StepStatus.INTERRUPTED):
                            self.step_statuses[step_id] = StepStatus.SKIPPED
            else:
                for run_num in range(1, n_localizer_runs + 1):
                    step_id = f"rest_run_{run_num}"
                    if self.step_statuses.get(step_id) in (StepStatus.PENDING, StepStatus.INTERRUPTED):
                        self.step_statuses[step_id] = StepStatus.SKIPPED

            # Coregistration requires the initial reference — skip if reference was skipped
            if (
                reference_initial_enabled
                and self.step_statuses.get("reference_initial") == StepStatus.SKIPPED
                and self.step_statuses.get("coregistration") in (StepStatus.PENDING, StepStatus.INTERRUPTED)
            ):
                self.step_statuses["coregistration"] = StepStatus.SKIPPED

    def _get_section_for_next_step(self) -> str | None:
        """Determine which section contains the earliest incomplete step.

        Returns:
            Section name ("localizer", "mask_extraction", "feedback") or None if all complete
        """
        next_step = self._get_next_step()
        if next_step is None:
            return None

        step_id, _ = next_step
        return self._get_section_for_step(step_id)

    def _get_section_for_step(self, step_id: str) -> str:
        """Get the section name for a given step ID.

        Args:
            step_id: Step identifier

        Returns:
            Section name ("localizer", "mask_extraction", "feedback")
        """
        # Handle dynamic step IDs
        if step_id.startswith("rest_run_"):
            return "localizer"
        if step_id.startswith("localizer_") and "_run_" in step_id:
            return "localizer"
        if step_id.startswith("localizer_") and (
            step_id.endswith("_preprocessing") or step_id.endswith("_extraction") or step_id.endswith("_approval")
        ):
            return "mask_extraction"

        step_to_section = {
            "reference_initial": "localizer",
            "preprocessing": "mask_extraction",
            "mask_extraction": "mask_extraction",
            "mni_mask_extraction": "mask_extraction",
            "mask_approval": "mask_extraction",
            "coregistration": "mask_extraction",
            "reference_realtime": "feedback",
            "coregistration_realtime": "feedback",
        }

        return step_to_section.get(step_id, "feedback")

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
            StepStatus.SKIPPED: ft.Icons.FAST_FORWARD,
        }

        color_map = {
            StepStatus.PENDING: ft.Colors.GREY_400,
            StepStatus.RUNNING: ft.Colors.BLUE_500,
            StepStatus.COMPLETE: ft.Colors.GREEN_500,
            StepStatus.INTERRUPTED: ft.Colors.ORANGE_500,
            StepStatus.FAILED: ft.Colors.RED_500,
            StepStatus.SKIPPED: ft.Colors.BLUE_GREY_300,
        }

        return ft.Icon(
            icon_map[status],
            color=color_map[status],
            size=20,
        )

    def _build_mask_to_localizer_map(self) -> dict[str, int]:
        """Build a mapping from mask name to the localizer index that produces it.

        For task localizers, mask names come from the ``contrasts`` dict keys.
        For rest localizers, mask names come from the ``yeo_targets`` list.

        Returns:
            Dict mapping mask name to localizer index.
        """
        config = self.study_config.session_config
        localizers = config.get("localizers", [])
        mask_to_loc: dict[str, int] = {}
        for loc_idx, loc in enumerate(localizers):
            loc_type = loc.get("type", "rest")
            if loc_type == "rest":
                for target in loc.get("networks", loc.get("yeo_targets", [])):
                    mask_to_loc[target] = loc_idx
            else:
                # Task localizer — contrasts dict keys are the mask names
                for mask_name in loc.get("contrasts", {}):
                    mask_to_loc[mask_name] = loc_idx
        return mask_to_loc

    def _get_feedback_step_frc_index(self, step_id: str) -> int | None:
        """Map a feedback/transfer step ID to its ``feedback_runs`` list index.

        Returns:
            Index into ``feedback_runs``, or None if not a feedback/transfer step.
        """
        config = self.study_config.session_config
        feedback_runs = config.get("feedback_runs", [])
        fb_counter = 0
        tf_counter = 0
        for frc_idx, frc in enumerate(feedback_runs):
            use_transfers = frc.get("use_transfer_runs", False)
            max_runs = frc.get("max_runs", 1)
            if use_transfers:
                tf_counter += 1
                if step_id == f"transfer_{tf_counter}":
                    return frc_idx
            for _ in range(max_runs):
                fb_counter += 1
                if step_id == f"feedback_{fb_counter}":
                    return frc_idx
            if use_transfers:
                tf_counter += 1
                if step_id == f"transfer_{tf_counter}":
                    return frc_idx
        return None

    def _get_masks_for_frc(self, frc: dict) -> set[str]:
        """Extract the set of mask names a feedback run config actually uses.

        Reads from ``network_assignment`` (single source of truth) for activation
        mode, or from the connectivity sub-block for connectivity mode.
        """
        masks: set[str] = set()
        mode = frc.get("feedback_mode", "activation")
        net_assign = frc.get("network_assignment", {}) or {}

        if mode in ("activation", "differential"):
            masks.update(name for name in net_assign.values() if name)
        elif mode == "connectivity":
            connectivity_pairs = (frc.get("connectivity", {}) or {}).get("pairs") or []
            for pair in connectivity_pairs:
                for name in pair or []:
                    if name:
                        masks.add(name)

        return masks

    def _fb(self, key: str, default=None):
        """Read a feedback parameter from the active FRC (new nested schema).

        Maps legacy flat keys to their new sub-block locations
        (qc / glm / connectivity / intermittent).
        """
        _NESTED_KEY_MAP = {
            "fd_threshold": ("qc", "fd_threshold"),
            "dvars_threshold": ("qc", "dvars_threshold"),
            "tsnr_threshold": ("qc", "tsnr_threshold"),
            "motion_correction": ("glm", "motion_correction"),
            "include_drift": ("glm", "include_drift"),
            "include_motion_regressors": ("glm", "include_motion_regressors"),
            "include_wm_regressor": ("glm", "include_wm_regressor"),
            "include_csf_regressor": ("glm", "include_csf_regressor"),
            "smooth_fwhm": ("glm", "smooth_fwhm"),
            "connectivity_pairs": ("connectivity", "pairs"),
            "connectivity_window": ("connectivity", "window"),
            "connectivity_target": ("connectivity", "target"),
            "feedback_interval": ("intermittent", "interval"),
        }
        frc = getattr(self, "_active_frc", None)
        if not frc:
            return default
        if key in frc:
            return frc[key]
        sub_key = _NESTED_KEY_MAP.get(key)
        if sub_key is not None:
            block, inner = sub_key
            return frc.get(block, {}).get(inner, default)
        return default

    def _resolve_frc_mask_names(self, frc: dict) -> list[str]:
        """Return an ordered list of mask names for a feedback run config.

        Similar to ``_get_masks_for_frc`` but returns a stable list suitable
        for the activation plot and feedback monitoring.
        """
        # Use explicit network_assignment if available (covers 3+ network tasks)
        net_assign = frc.get("network_assignment", {})
        if net_assign:
            task_type = frc.get("task", "ball")
            try:
                from pineuro.tasks.task_registry import get_task_metadata

                meta = get_task_metadata(task_type)
                names = [net_assign[role.key] for role in meta.network_roles if role.key in net_assign]
                if names:
                    return names
            except (ValueError, KeyError):
                pass

        mode = frc.get("feedback_mode", "activation")
        if mode in ("activation", "differential"):
            # If we got here, network_assignment didn't cover all roles — return whatever it has.
            return [v for v in (frc.get("network_assignment", {}) or {}).values() if v]
        elif mode == "connectivity":
            names_list: list[str] = []
            for pair in (frc.get("connectivity", {}) or {}).get("pairs") or []:
                for name in pair or []:
                    if name and name not in names_list:
                        names_list.append(name)
            return names_list
        return []

    def _is_step_approval_unlocked(self, step_id: str) -> bool:
        """Check whether a step's mask-approval prerequisite is satisfied.

        For single-localizer, this is equivalent to ``_initial_qc_approved``.
        For multi-localizer:
        - Reference realtime / coregistration: ANY localizer approved
        - Feedback/transfer steps: all localizers that produce the needed masks are approved
        """
        config = self.study_config.session_config
        localizers = config.get("localizers", [])

        if not (localizers and len(localizers) > 1):
            # Single-localizer: use blanket approval flag
            return self._initial_qc_approved

        # Multi-localizer: per-step gating
        is_feedback_transfer = step_id.startswith("feedback_") or step_id.startswith("transfer_")

        if is_feedback_transfer:
            # Feedback/transfer: need all localizers that produce required masks to be approved
            frc_idx = self._get_feedback_step_frc_index(step_id)
            if frc_idx is None:
                return self._initial_qc_approved
            frc = config["feedback_runs"][frc_idx]
            needed_masks = self._get_masks_for_frc(frc)
            if not needed_masks:
                return self._initial_qc_approved
            mask_to_loc = self._build_mask_to_localizer_map()
            required_loc_indices = {mask_to_loc[m] for m in needed_masks if m in mask_to_loc}
            if not required_loc_indices:
                # Masks not mapped to any localizer — fall back to all approved
                return self._initial_qc_approved
            return all(
                self.step_statuses.get(f"localizer_{i}_approval") == StepStatus.COMPLETE for i in required_loc_indices
            )
        else:
            # Reference realtime, coregistration, coregistration_realtime:
            # ANY localizer approved unlocks these
            return any(
                self.step_statuses.get(f"localizer_{i}_approval") == StepStatus.COMPLETE for i in range(len(localizers))
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

        # In mask reuse mode, preprocessing and mask approval are inherited from source session
        # so these gates should not apply
        if self._is_mask_reuse_mode:
            requires_preprocessing = False
            requires_approval = False
        else:
            # Gate 1: Preprocessing must be complete before mask extraction and subsequent steps
            config = self.study_config.session_config
            localizers = config.get("localizers", [])
            # Manual localizers import pre-built masks — they have no preprocessing
            # step, so the preprocessing gate must not lock their pipeline.
            non_manual = [loc for loc in localizers if loc.get("type") != "manual"]
            if not non_manual:
                # All-manual study: nothing to preprocess.
                preprocessing_complete = True
            elif len(localizers) > 1:
                preprocessing_complete = any(
                    self.step_statuses.get(f"localizer_{i}_preprocessing") == StepStatus.COMPLETE
                    for i in range(len(localizers))
                )
            else:
                preprocessing_complete = self.step_statuses.get("preprocessing") == StepStatus.COMPLETE
            steps_requiring_preprocessing = [
                "mask_extraction",
                "coregistration",
                "reference_realtime",
                "coregistration_realtime",
            ]
            # Also check for feedback/transfer steps (dynamic)
            is_feedback_transfer = step_id.startswith("transfer_") or step_id.startswith("feedback_")
            requires_preprocessing = (
                step_id in steps_requiring_preprocessing or is_feedback_transfer
            ) and not preprocessing_complete

            # Gate 2: Mask approval required for coregistration and all subsequent steps
            # Uses per-feedback gating for multi-localizer studies
            gated_steps = [
                "coregistration",
                "reference_realtime",
                "coregistration_realtime",
            ]
            needs_approval = step_id in gated_steps or is_feedback_transfer
            requires_approval = needs_approval and not self._is_step_approval_unlocked(step_id)

        # Completed preprocessing / mask-extraction steps stay clickable so the user
        # can re-run them (e.g. after editing inputs outside pineuro). Give them an
        # explicit "Re-run" label + refresh icon so it's clear the action is available;
        # the preview panel then prompts for skip/overwrite before anything re-runs.
        is_import = label == "Import Masks"
        is_rerunnable = not is_import and (
            step_id in ("preprocessing", "mask_extraction")
            or (
                step_id.startswith("localizer_")
                and (step_id.endswith("_preprocessing") or step_id.endswith("_extraction"))
            )
        )
        if status == StepStatus.COMPLETE and is_rerunnable:
            display_label = "Re-run Preprocessing" if "preprocessing" in step_id else "Re-run Mask Extraction"
            display_icon = ft.Icons.REFRESH
        else:
            display_label = label
            display_icon = None

        button = ft.ElevatedButton(
            display_label,
            icon=display_icon,
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

    def _build_localizer_tasks_section(self) -> ft.ExpansionPanel:
        """Build the localizer tasks section with collapsible steps.

        Returns:
            ExpansionPanel control
        """
        config = self.study_config.session_config
        controls = []

        # Add Initial Reference section only if enabled
        if config.get("reference", {}).get("initial_enabled", True):
            controls.extend(
                [
                    ft.Text("Initial Reference", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "reference_initial",
                        "Collect",
                        lambda e: self._on_step_clicked("reference_initial"),
                    ),
                    ft.Divider(),
                ]
            )

        # Add localizer runs dynamically
        localizers = config.get("localizers", [])

        if localizers and len(localizers) > 1:
            # Multi-localizer: show each localizer with its runs
            for loc_idx, loc in enumerate(localizers):
                loc_type = loc.get("type", "rest")
                # Manual localizers import pre-built masks — no collection runs.
                if loc_type == "manual":
                    continue
                loc_n_runs = loc.get("n_runs", 2)

                # Get display name for this localizer
                if loc_type == "rest":
                    loc_label = "Resting State"
                else:
                    try:
                        from pineuro.tasks import get_task_metadata

                        task_meta = get_task_metadata(loc_type)
                        loc_label = task_meta.display_name
                    except (ValueError, ImportError):
                        loc_label = loc_type.replace("_", " ").title()

                if loc_idx > 0:
                    controls.append(ft.Divider())
                controls.append(ft.Text(f"Localizer {loc_idx + 1}: {loc_label}", size=14, weight=ft.FontWeight.W_500))

                for run_num in range(1, loc_n_runs + 1):
                    step_id = f"localizer_{loc_idx}_run_{run_num}"
                    controls.append(
                        self._create_step_button(
                            step_id,
                            f"Collect Run {run_num}",
                            lambda e, sid=step_id: self._on_step_clicked(sid),
                        )
                    )
        elif config.get("localizers") and config["localizers"][0].get("type") == "manual":
            # Single manual localizer: no collection runs to show.
            controls.append(
                ft.Text(
                    "Manual mask import — no localizer scan is collected.",
                    size=12,
                    color=ft.Colors.GREY_600,
                    italic=True,
                )
            )
        else:
            # Single localizer
            n_localizer_runs = _loc(config).get("n_runs", 2)
            loc_type = _loc(config).get("type", "rest")
            if loc_type == "rest":
                localizer_label = "Resting State"
            elif loc_type == "manual":
                localizer_label = "Manual Masks"
            else:
                try:
                    from pineuro.tasks import get_task_metadata

                    task_meta = get_task_metadata(loc_type)
                    localizer_label = task_meta.display_name
                except (ValueError, ImportError):
                    localizer_label = f"Localizer Task ({loc_type})"

            controls.append(ft.Text(localizer_label, size=14, weight=ft.FontWeight.W_500))

            for run_num in range(1, n_localizer_runs + 1):
                step_id = f"rest_run_{run_num}"
                controls.append(
                    self._create_step_button(
                        step_id,
                        f"Collect Run {run_num}",
                        lambda e, sid=step_id: self._on_step_clicked(sid),
                    )
                )

        # MNI atlas fallback button (skip localizer entirely)
        if not self._is_mask_reuse_mode and not self._is_mni_atlas_mode:
            controls.extend(
                [
                    ft.Divider(),
                    ft.Container(
                        content=ft.OutlinedButton(
                            "Skip Localizer: Use MNI Atlas Masks",
                            icon=ft.Icons.MAP,
                            on_click=lambda e: self._confirm_mni_fallback("localizer"),
                        ),
                        padding=ft.padding.only(top=5),
                    ),
                ]
            )

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Localizer Tasks", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(controls, spacing=10),
            expanded=self._get_section_for_next_step() == "localizer",
        )

    def _build_mask_extraction_section(self) -> ft.ExpansionPanel:
        """Build the mask extraction section with collapsible steps.

        Returns:
            ExpansionPanel control
        """
        config = self.study_config.session_config

        # In MNI atlas mode, show simplified section
        if self._is_mni_atlas_mode:
            controls: list[ft.Control] = [
                ft.Text("MNI Atlas Masks", size=14, weight=ft.FontWeight.W_500),
            ]
            mni_status = self.step_statuses.get("mni_mask_extraction", StepStatus.PENDING)
            if mni_status == StepStatus.COMPLETE:
                atlas_name = _loc(config).get("extraction", {}).get("network_atlas", "yeo_17")
                controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Text(f"MNI Atlas ({atlas_name})", size=12, color=ft.Colors.GREY_600),
                                ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
                            ],
                            spacing=5,
                        ),
                        padding=ft.padding.only(left=10),
                    )
                )
                # Mask review/approval gate
                if self._initial_qc_approved:
                    controls.append(
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.TextButton(
                                        "Review Masks",
                                        icon=ft.Icons.TUNE,
                                        on_click=lambda e: self._show_mask_review_dialog(),
                                    ),
                                    ft.Text("(MNI Atlas)", size=12, color=ft.Colors.GREY_600),
                                    ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
                                ],
                                spacing=5,
                            ),
                            padding=ft.padding.only(left=10),
                        )
                    )
                else:
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
            else:
                controls.append(
                    self._create_step_button(
                        "mni_mask_extraction",
                        "Extract MNI Atlas Masks",
                        lambda e: self._on_step_clicked("mni_mask_extraction"),
                    ),
                )

            # Still show coregistration if initial reference is enabled
            if config.get("reference", {}).get("initial_enabled", True):
                controls.extend(
                    [
                        ft.Divider(),
                        ft.Text("Coregistration", size=14, weight=ft.FontWeight.W_500),
                        self._create_step_button(
                            "coregistration",
                            "Coregister to Reference",
                            lambda e: self._on_step_clicked("coregistration"),
                        ),
                    ]
                )
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

            # Revert to original masks option
            controls.extend(
                [
                    ft.Divider(),
                    ft.Container(
                        content=ft.OutlinedButton(
                            "Revert to Original Masks",
                            icon=ft.Icons.UNDO,
                            on_click=lambda e: self._revert_from_mni(),
                        ),
                        padding=ft.padding.only(top=5),
                    ),
                ]
            )

            return ft.ExpansionPanel(
                header=ft.ListTile(
                    title=ft.Text("Mask Extraction", weight=ft.FontWeight.BOLD),
                ),
                content=ft.Column(controls, spacing=10),
                expanded=self._get_section_for_next_step() == "mask_extraction",
            )

        # Standard mode: preprocessing + ICA/GLM extraction
        localizers = config.get("localizers", [])
        controls: list[ft.Control] = []  # type: ignore[no-redef]

        if localizers and len(localizers) > 1:
            # Multi-localizer: per-localizer preprocessing + extraction + approval
            for loc_idx, loc in enumerate(localizers):
                loc_type = loc.get("type", "rest")
                is_manual = loc_type == "manual"
                if loc_type == "rest":
                    loc_label = "Rest"
                elif is_manual:
                    loc_label = "Manual"
                else:
                    try:
                        from pineuro.tasks import get_task_metadata

                        task_meta = get_task_metadata(loc_type)
                        loc_label = task_meta.display_name.replace(" Localizer", "").replace(" Task", "")
                    except (ValueError, ImportError):
                        loc_label = loc_type.replace("_", " ").title()

                if loc_idx > 0:
                    controls.append(ft.Divider())

                preproc_id = f"localizer_{loc_idx}_preprocessing"
                extract_id = f"localizer_{loc_idx}_extraction"
                approval_id = f"localizer_{loc_idx}_approval"
                controls.append(ft.Text(f"Localizer {loc_idx + 1}: {loc_label}", size=14, weight=ft.FontWeight.W_500))
                # Manual localizers skip preprocessing — they import pre-built masks.
                if not is_manual:
                    controls.append(
                        self._create_step_button(
                            preproc_id,
                            "Preprocess",
                            lambda e, sid=preproc_id: self._on_step_clicked(sid),
                        )
                    )
                controls.append(
                    self._create_step_button(
                        extract_id,
                        "Import Masks" if is_manual else "Extract Masks",
                        lambda e, sid=extract_id: self._on_step_clicked(sid),
                    )
                )
                if is_manual:
                    controls.append(
                        ft.Container(
                            content=ft.TextButton(
                                "Configure / edit mask files…",
                                icon=ft.Icons.FOLDER_OPEN,
                                on_click=lambda e, idx=loc_idx: self._show_manual_mask_config_dialog(loc_idx=idx),
                            ),
                            padding=ft.padding.only(left=10),
                        )
                    )

                # Per-localizer approval button (shown after extraction completes)
                extraction_complete = self.step_statuses.get(extract_id) == StepStatus.COMPLETE
                if extraction_complete:
                    approval_complete = self.step_statuses.get(approval_id) == StepStatus.COMPLETE
                    if approval_complete:
                        controls.append(
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.TextButton(
                                            "Review Masks",
                                            icon=ft.Icons.TUNE,
                                            on_click=lambda e, idx=loc_idx: self._show_mask_review_dialog(loc_idx=idx),
                                        ),
                                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
                                    ],
                                    spacing=5,
                                ),
                                padding=ft.padding.only(left=10),
                            )
                        )
                    else:
                        controls.append(
                            ft.Container(
                                content=ft.FilledButton(
                                    "Review & Approve Masks",
                                    icon=ft.Icons.APPROVAL,
                                    bgcolor=ft.Colors.ORANGE,
                                    on_click=lambda e, idx=loc_idx: self._show_mask_review_dialog(loc_idx=idx),
                                ),
                                padding=ft.padding.only(left=10, top=5),
                            )
                        )
        elif config.get("localizers") and config["localizers"][0].get("type") == "manual":
            # Single manual localizer: import masks only — no preprocessing.
            controls.extend(
                [
                    ft.Text("Network Masks", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "mask_extraction",
                        "Import Masks",
                        lambda e: self._on_step_clicked("mask_extraction"),
                    ),
                    ft.Container(
                        content=ft.TextButton(
                            "Configure / edit mask files…",
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=lambda e: self._show_manual_mask_config_dialog(loc_idx=0),
                        ),
                        padding=ft.padding.only(left=10),
                    ),
                ]
            )

            # Review / approve the imported masks (shown once import completes)
            if self.step_statuses.get("mask_extraction") == StepStatus.COMPLETE:
                if self._initial_qc_approved:
                    controls.append(
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.TextButton(
                                        "Review Masks",
                                        icon=ft.Icons.TUNE,
                                        on_click=lambda e: self._show_mask_review_dialog(),
                                    ),
                                    ft.Text("(Manual import)", size=12, color=ft.Colors.GREY_600),
                                    ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=20),
                                ],
                                spacing=5,
                            ),
                            padding=ft.padding.only(left=10),
                        )
                    )
                else:
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
        else:
            # Single localizer
            controls.extend(
                [
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
            )

            # Single-localizer approval
            all_extraction_complete = self.step_statuses.get("mask_extraction") == StepStatus.COMPLETE
            if all_extraction_complete:
                if self._initial_qc_approved:
                    # Approved - show "Review Masks" button with checkmark
                    masks_customized = self._has_custom_mask_selections()
                    _cfg = self.study_config.session_config
                    localizer_type = _cfg.get("mask", {}).get("localizer_type", "rest")
                    _loc0 = _cfg.get("localizers", [{}])[0] if _cfg.get("localizers") else {}
                    if _loc0.get("type") == "manual":
                        method_label = "Manual"
                    elif localizer_type == "task":
                        method_label = "GLM"
                    else:
                        method_label = "ICA"
                    status_text = "Customized" if masks_customized else method_label
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

        # MNI fallback button (always available unless in mask-reuse or already MNI mode)
        if not self._is_mask_reuse_mode and not self._is_mni_atlas_mode:
            controls.extend(
                [
                    ft.Divider(),
                    ft.Container(
                        content=ft.OutlinedButton(
                            "Use MNI Atlas Masks Instead",
                            icon=ft.Icons.MAP,
                            on_click=lambda e: self._confirm_mni_fallback("mask_extraction"),
                        ),
                        padding=ft.padding.only(top=5),
                    ),
                ]
            )

        # Add coregistration section only if initial reference is enabled
        config = self.study_config.session_config
        if config.get("reference", {}).get("initial_enabled", True):
            controls.extend(
                [
                    ft.Divider(),
                    ft.Text("Coregistration", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "coregistration",
                        "Coregister to Reference",
                        lambda e: self._on_step_clicked("coregistration"),
                    ),
                ]
            )

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
            expanded=self._get_section_for_next_step() == "mask_extraction",
        )

    def _build_feedback_section(self) -> ft.ExpansionPanel:
        """Build the feedback tasks section.

        Returns:
            ExpansionPanel control
        """
        config = self.study_config.session_config

        # Build controls list
        controls = []

        # Add Real-time Reference section only if enabled
        if config.get("reference", {}).get("realtime_enabled", True):
            controls.extend(
                [
                    ft.Text("Real-time Reference", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "reference_realtime",
                        "Collect",
                        lambda e: self._on_step_clicked("reference_realtime"),
                    ),
                    ft.Divider(),
                    ft.Text("Coregistration", size=14, weight=ft.FontWeight.W_500),
                    self._create_step_button(
                        "coregistration_realtime",
                        "Coregister",
                        lambda e: self._on_step_clicked("coregistration_realtime"),
                    ),
                ]
            )

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
        controls.extend(
            [
                ft.Divider(),
                ft.Text("Feedback Runs", size=14, weight=ft.FontWeight.W_500),
            ]
        )

        # Add each feedback/transfer step with optional "View Summary" button
        feedback_steps = _build_feedback_step_ids(config)
        for step_id, step_label in feedback_steps:
            controls.append(
                self._create_step_button(
                    step_id,
                    step_label,
                    lambda e, sid=step_id: self._on_step_clicked(sid),
                )
            )
            # Add "View Summary" button if step is complete or interrupted
            step_status = self.step_statuses.get(step_id)
            if step_status in (StepStatus.COMPLETE, StepStatus.INTERRUPTED):
                controls.append(
                    ft.Container(
                        content=ft.TextButton(
                            "View Summary",
                            icon=ft.Icons.ASSESSMENT,
                            on_click=lambda e, sid=step_id: self._show_feedback_summary(sid),
                        ),
                        padding=ft.padding.only(left=10),
                    )
                )

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Feedback Tasks", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(
                controls,
                spacing=10,
            ),
            expanded=self._get_section_for_next_step() == "feedback",
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
        if size_bytes >= 1024**3:
            return f"{size_bytes / (1024**3):.1f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / (1024**2):.1f} MB"
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
            - tmp_exists: bool
            - tmp_size: int (bytes)
            - tmp_subdirs: list of subdir names
            - total_size: int (bytes)
        """
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"
        tmp_dir = base_dir / "tmp"

        result: dict[str, Any] = {
            "collection_exists": False,
            "collection_size": 0,
            "collection_subdirs": [],
            "tmp_exists": False,
            "tmp_size": 0,
            "tmp_subdirs": [],
            "total_size": 0,
        }

        if collection_dir.exists():
            subdirs = [d.name for d in collection_dir.iterdir() if d.is_dir()]
            if subdirs:
                result["collection_exists"] = True
                result["collection_subdirs"] = sorted(subdirs)
                result["collection_size"] = self._get_dir_size(collection_dir)

        if tmp_dir.exists():
            subdirs = [d.name for d in tmp_dir.iterdir() if d.is_dir()]
            if subdirs:
                result["tmp_exists"] = True
                result["tmp_subdirs"] = sorted(subdirs)
                result["tmp_size"] = self._get_dir_size(tmp_dir)

        result["total_size"] = result["collection_size"] + result["tmp_size"]
        return result

    def _build_scanner_connection_section(self) -> ft.ExpansionPanel:
        """Build the scanner connection test section.

        Only shown when simulator mode is disabled. Provides a test button
        and setup guide link for verifying real-time scanner connectivity.
        """
        config = self.study_config.session_config
        scanner_ip = config.get("scanner", {}).get("ip", "")
        watch_dir = config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM")

        self._scanner_status_icon = ft.Icon(ft.Icons.HELP_OUTLINE, color=ft.Colors.GREY, size=20)
        self._scanner_status_text = ft.Text(
            "Not tested yet",
            size=12,
            color=ft.Colors.GREY_600,
            italic=True,
        )

        test_button = ft.ElevatedButton(
            "Test Connection",
            icon=ft.Icons.SETTINGS_ETHERNET,
            on_click=lambda e: self._on_test_scanner_connection(),
            disabled=self._is_running,
        )

        setup_guide_button = ft.TextButton(
            "Setup Guide",
            icon=ft.Icons.HELP_OUTLINE,
            on_click=lambda e: self._on_show_setup_guide(),
        )

        ip_text = scanner_ip if scanner_ip else "not set"

        return ft.ExpansionPanel(
            header=ft.ListTile(
                title=ft.Text("Scanner Connection", weight=ft.FontWeight.BOLD),
            ),
            content=ft.Column(
                [
                    ft.Text(
                        f"IP: {ip_text} | Dir: {watch_dir}",
                        size=11,
                        color=ft.Colors.GREY_600,
                    ),
                    ft.Container(
                        content=ft.Row(
                            [
                                test_button,
                                self._scanner_status_icon,
                                self._scanner_status_text,
                            ],
                            spacing=10,
                        ),
                        padding=ft.padding.only(left=10),
                    ),
                    ft.Container(
                        content=setup_guide_button,
                        padding=ft.padding.only(left=10),
                    ),
                ],
                spacing=5,
            ),
            expanded=True,
        )

    def _on_test_scanner_connection(self):
        """Handle scanner connection test button click."""
        from ..components.scanner_connection_dialog import ScannerConnectionDialog

        config = self.study_config.session_config
        scanner_ip = config.get("scanner", {}).get("ip", "")
        watch_dir = config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM")

        def on_result(result):
            if result is None:
                return
            try:
                if result.overall_ok:
                    self._scanner_status_icon.name = ft.Icons.CHECK_CIRCLE
                    self._scanner_status_icon.color = ft.Colors.GREEN
                    self._scanner_status_text.value = "Connection OK"
                    self._scanner_status_text.color = ft.Colors.GREEN
                    self._scanner_status_text.italic = False
                else:
                    self._scanner_status_icon.name = ft.Icons.ERROR
                    self._scanner_status_icon.color = ft.Colors.RED
                    self._scanner_status_text.value = "Issues detected"
                    self._scanner_status_text.color = ft.Colors.RED
                    self._scanner_status_text.italic = False
                self.page.update()
            except Exception:
                pass

        dialog = ScannerConnectionDialog(
            page=self.page,
            scanner_ip=scanner_ip,
            watch_dir=watch_dir,
            on_close=on_result,
        )
        dialog.show()

    def _on_show_setup_guide(self):
        """Show scanner setup guide dialog from participant workflow."""
        from pineuro.scanner_setup import get_current_platform_key, get_setup_instructions

        config = self.study_config.session_config
        watch_dir = config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM")
        scanner_ip = config.get("scanner", {}).get("ip", "")

        instructions = get_setup_instructions(watch_dir, scanner_ip)
        platform_key = get_current_platform_key()
        initial_index = {"windows": 0, "macos": 1, "linux": 2}.get(platform_key, 0)

        def _make_content(text: str) -> ft.Container:
            return ft.Container(
                content=ft.Column(
                    [ft.Text(text, font_family="monospace", selectable=True, size=12)],
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=10,
            )

        tabs = ft.Tabs(
            length=3,
            selected_index=initial_index,
            content=ft.Column(
                [
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label="Windows"),
                            ft.Tab(label="macOS"),
                            ft.Tab(label="Linux"),
                        ],
                    ),
                    ft.TabBarView(
                        controls=[
                            _make_content(instructions["windows"]),
                            _make_content(instructions["macos"]),
                            _make_content(instructions["linux"]),
                        ],
                        expand=True,
                    ),
                ],
                expand=True,
            ),
            expand=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Scanner Setup Guide"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("SMB Share Setup", size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(content=tabs, height=300),
                        ft.Divider(),
                        ft.Text("Scanner Console Configuration", size=14, weight=ft.FontWeight.BOLD),
                        ft.Text(instructions["scanner_notes"], size=12),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=700,
                height=550,
            ),
            actions=[ft.TextButton("Close", on_click=lambda e: _close())],
        )

        def _close():
            dialog.open = False
            self.page.update()

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

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
                            "No temporary files found.",
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.Alignment(0, 0),
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
                                        cast(
                                            list[ft.Control],
                                            [
                                                ft.Text(f"  - {subdir}", size=12, color=ft.Colors.GREY_700)
                                                for subdir in cleanup_info["collection_subdirs"][:6]
                                            ]
                                            + (
                                                [
                                                    ft.Text(
                                                        f"  ... and {len(cleanup_info['collection_subdirs']) - 6} more",
                                                        size=12,
                                                        color=ft.Colors.GREY_500,
                                                    )
                                                ]
                                                if len(cleanup_info["collection_subdirs"]) > 6
                                                else []
                                            ),
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

            if cleanup_info["tmp_exists"]:
                dir_items.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Icon(ft.Icons.FOLDER, color=ft.Colors.AMBER),
                                        ft.Text("tmp/", weight=ft.FontWeight.BOLD),
                                        ft.Text(
                                            f"({self._format_size(cleanup_info['tmp_size'])})",
                                            color=ft.Colors.GREY_600,
                                        ),
                                    ],
                                    spacing=10,
                                ),
                                ft.Container(
                                    content=ft.Column(
                                        cast(
                                            list[ft.Control],
                                            [
                                                ft.Text(f"  - {subdir}", size=12, color=ft.Colors.GREY_700)
                                                for subdir in cleanup_info["tmp_subdirs"][:6]
                                            ]
                                            + (
                                                [
                                                    ft.Text(
                                                        f"  ... and {len(cleanup_info['tmp_subdirs']) - 6} more",
                                                        size=12,
                                                        color=ft.Colors.GREY_500,
                                                    )
                                                ]
                                                if len(cleanup_info["tmp_subdirs"]) > 6
                                                else []
                                            ),
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
        """Execute cleanup - delete collection directory and tmp subdirectories."""
        import shutil

        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"
        tmp_dir = base_dir / "tmp"

        self._add_log("[INFO] Starting cleanup...")

        try:
            if collection_dir.exists():
                shutil.rmtree(collection_dir)
                self._add_log(f"[SUCCESS] Deleted: {collection_dir}")

            if tmp_dir.exists():
                for subdir in tmp_dir.iterdir():
                    if subdir.is_dir():
                        shutil.rmtree(subdir)
                        self._add_log(f"[SUCCESS] Deleted: {subdir}")

            self._add_log("[SUCCESS] Cleanup complete")

            # Show success snackbar
            _snack = ft.SnackBar(
                ft.Text("Cleanup complete. Temporary files have been deleted."), bgcolor=ft.Colors.GREEN
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

            # Refresh the UI - step list first (to update utilities section status)
            self._refresh_step_list()
            # Show default preview panel (cleanup complete state)
            self._show_default_preview()

        except Exception as e:
            self._add_log(f"[ERROR] Cleanup failed: {e}")
            logger.error(f"Cleanup failed: {e}")

            _snack = ft.SnackBar(ft.Text(f"Cleanup failed: {e}"), bgcolor=ft.Colors.RED)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

    def _show_qc_approval_dialog(self, approval_type: str):
        """Show mandatory QC approval dialog after coregistration.

        Args:
            approval_type: "initial" (after coregistration) or "realtime" (after coregistration_realtime)
        """
        from ..components.qc_plot_viewer import QCPlotViewer

        # Get appropriate plots based on approval type
        # Initial: show ICA, subject space, and reference overlays (full mask extraction review)
        # Realtime: show ONLY reference overlays (since ICA and subject space didn't change)
        if approval_type == "initial":
            plots = self.qc_manager.get_mask_extraction_plots()
            title = "Initial Reference QC - Review Required"
        else:
            # In mask reuse mode, realtime reference is run-01 (no initial reference collection)
            # In standard mode, realtime reference is run-02
            run_id = 1 if self._is_mask_reuse_mode else 2
            plots = self.qc_manager.get_reference_qc_plots(run_id=run_id)
            title = "Real-time Reference QC"

        # Filter out combined hemisphere mask from plots when using hemisphere_split sham mode
        # (we only want to show left/right splits, not the combined one)
        sham_config = self.study_config.session_config.get("sham", {})
        if sham_config.get("enabled", False):
            delivery_method = sham_config.get("delivery_method", "alternative_masks")
            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
            if delivery_method == "alternative_masks" and mask_mode == "hemisphere_split":
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                # Filter out plots with the combined mask name (e.g., "SomMotA Mask")
                # Keep plots with left/right (e.g., "SomMotA_left Mask", "SomMotA_right Mask")
                plots = [
                    (label, path)
                    for label, path in plots
                    if not (hemisphere_network in label and "_left" not in label and "_right" not in label)
                ]

        html_path = self.qc_manager.get_mask_extraction_qc()

        viewer = QCPlotViewer(
            page=self.page,
            title=title,
            plots=plots,
            html_report_path=html_path,
            require_approval=False,
        )
        viewer.show()

    def _show_coregistration_qc(self, qc_type: str):
        """Show coregistration QC viewer (non-essential, view-only).

        Args:
            qc_type: "initial" (after coregistration) or "realtime" (after coregistration_realtime)
        """
        from ..components.qc_plot_viewer import QCPlotViewer

        # Get reference overlay plots - use run_id=1 for initial, run_id=2 for realtime
        # In mask reuse mode, realtime reference is run-01 (no initial reference collection)
        if qc_type == "initial":
            run_id = 1
            title = "Initial Reference Coregistration QC"
        else:
            # In mask reuse mode, realtime reference is run-01
            run_id = 1 if self._is_mask_reuse_mode else 2
            title = "Real-time Reference Coregistration QC"

        plots = self.qc_manager.get_reference_qc_plots(run_id=run_id)

        # Filter out combined hemisphere mask from plots when using hemisphere_split sham mode
        sham_config = self.study_config.session_config.get("sham", {})
        if sham_config.get("enabled", False):
            delivery_method = sham_config.get("delivery_method", "alternative_masks")
            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
            if delivery_method == "alternative_masks" and mask_mode == "hemisphere_split":
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                plots = [
                    (label, path)
                    for label, path in plots
                    if not (hemisphere_network in label and "_left" not in label and "_right" not in label)
                ]

        html_path = self.qc_manager.get_mask_extraction_qc()

        viewer = QCPlotViewer(
            page=self.page,
            title=title,
            plots=plots,
            html_report_path=html_path,
            require_approval=False,  # View-only, no approval needed
        )
        viewer.show()

    def _show_feedback_summary(self, step_id: str):
        """Show feedback run summary dialog after completion.

        Args:
            step_id: The completed step ID (transfer_1, feedback_1, feedback_2, transfer_2)
        """
        from pineuro import qc
        from pineuro.tasks.task_registry import get_task_metadata

        from ..components.feedback_summary import (
            FeedbackSummaryDialog,
            compute_feedback_summary,
        )

        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])

        # Determine task type and run number from step_id
        task = "transfer" if "transfer" in step_id else "feedback"
        run = 1 if "_1" in step_id else 2

        # Get config values — prefer per-FRC values
        frc_idx = self._get_feedback_step_frc_index(step_id)
        feedback_runs = config.get("feedback_runs", [])
        frc = feedback_runs[frc_idx] if frc_idx is not None and frc_idx < len(feedback_runs) else None

        def _fv(key, default=None):
            if frc is None:
                return default
            if key in ("fd_threshold", "dvars_threshold", "tsnr_threshold"):
                return frc.get("qc", {}).get(key, default)
            return frc.get(key, default)

        n_baseline = _fv("n_baseline", 25)
        n_volumes = _fv("n_volumes", 150)
        fd_threshold = _fv("fd_threshold", 0.5)
        dvars_threshold = _fv("dvars_threshold", 1.5)

        # Resolve declarative live metrics for this run's task.
        summary_task_type = (
            frc.get("task", "ball")
            if frc is not None
            else (config["feedback_runs"][0].get("task", "ball") if config.get("feedback_runs") else "ball")
        )
        try:
            summary_live_metrics = list(get_task_metadata(summary_task_type).live_metrics)
        except (KeyError, ValueError):
            summary_live_metrics = []

        # Find the feedbacklog and psychopylog files
        output_dir = base_dir / "feedback" / "02_feedback_output"
        feedbacklog_path = self._find_latest_feedback_file(output_dir, task, run, "feedbacklog.tsv")
        psychopylog_path = self._find_latest_feedback_file(output_dir, task, run, "psychopylog.tsv")

        if not feedbacklog_path or not feedbacklog_path.exists():
            logger.warning(f"No feedbacklog found for {step_id}, skipping summary")
            return

        # Extract attempt from the discovered filename (e.g., "..._run-01_attempt-01_feedbacklog.tsv")
        import re

        attempt_match = re.search(r"_attempt-(\d+)", feedbacklog_path.name)
        if attempt_match is None:
            logger.error(f"Feedbacklog {feedbacklog_path.name} has no attempt entity; cannot generate summary")
            return
        attempt: int = int(attempt_match.group(1))

        # Compute summary data
        try:
            summary_data = compute_feedback_summary(
                feedbacklog_path=feedbacklog_path,
                psychopy_log_path=psychopylog_path,
                n_baseline=n_baseline,
                n_expected_volumes=n_volumes,
                fd_threshold=fd_threshold,
                dvars_threshold=dvars_threshold,
                live_metrics=summary_live_metrics,
            )
        except Exception as e:
            logger.error(f"Failed to compute feedback summary: {e}")
            return

        # Get blind display names if blind mode is enabled
        # Include all masks (real + sham alternatives) for proper blinding
        frc_idx = self._get_feedback_step_frc_index(step_id)
        feedback_runs = config.get("feedback_runs", [])
        if frc_idx is not None and frc_idx < len(feedback_runs):
            mask_names = self._resolve_frc_mask_names(feedback_runs[frc_idx])
        else:
            mask_names = self._resolve_mask_names(config)
        sham_config = config.get("sham", {})
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            sham_mask_mode = sham_config.get("sham_mask_mode", "two_networks")
            if sham_mask_mode == "hemisphere_split":
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                _target_hemi = sham_config.get("target_hemisphere", "right")
                for hemi in ["left", "right"]:
                    hemi_name = f"{hemisphere_network}_{hemi}"
                    if hemi_name not in mask_names:
                        mask_names.append(hemi_name)
            else:
                alternative_targets = sham_config.get("alternative_mask_targets", [])
                for alt_name in alternative_targets:
                    if alt_name and alt_name not in mask_names:
                        mask_names.append(alt_name)
        display_names_dict = self._get_blind_display_names_dict(mask_names)

        # Generate summary plots (regenerate if blind mode to apply display names)
        qc_dir = output_dir / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)

        activation_plot_path = qc_dir / f"{self._bids_prefix}_task-{task}_run-{run:02d}_activation.png"
        connectivity_plot_path = qc_dir / f"{self._bids_prefix}_task-{task}_run-{run:02d}_connectivity.png"

        try:
            # In blind mode, always regenerate plots to ensure display names are applied
            needs_regen = display_names_dict is not None
            if needs_regen or not activation_plot_path.exists():
                qc.plot_activation_timeseries(
                    feedback_log_path=feedbacklog_path,
                    out_path=activation_plot_path,
                    n_baseline=n_baseline,
                    title=f"Activation - {self.participant_id} ({task} run {run})",
                    display_names=display_names_dict,
                )
            if needs_regen or not connectivity_plot_path.exists():
                qc.plot_connectivity_timeseries(
                    feedback_log_path=feedbacklog_path,
                    out_path=connectivity_plot_path,
                    n_baseline=n_baseline,
                    title=f"Connectivity - {self.participant_id} ({task} run {run})",
                    display_names=display_names_dict,
                )
        except Exception as e:
            logger.warning(f"Failed to generate summary plots: {e}")

        def on_view_full_qc():
            # Regenerate QC report to ensure correct attempt, expected_volumes, and display names
            try:
                qc.generate_feedback_report(
                    feedback_dir=output_dir,
                    subject=self.participant_id,
                    session=self.session_id,
                    task=task,
                    run=run,
                    attempt=attempt,
                    output_dir=qc_dir,
                    overwrite=True,
                    expected_volumes=n_volumes,
                    display_names=display_names_dict,
                    fd_threshold=fd_threshold,
                    live_metrics=summary_live_metrics,
                )
            except Exception as e:
                logger.warning(f"Failed to regenerate QC report: {e}")
            self.qc_manager.open_report(f"{task}_run_{run}")

        dialog = FeedbackSummaryDialog(
            page=self.page,
            summary_data=summary_data,
            subject=self.participant_id,
            task=task,
            run=run,
            activation_plot_path=activation_plot_path if activation_plot_path.exists() else None,
            connectivity_plot_path=connectivity_plot_path if connectivity_plot_path.exists() else None,
            on_view_full_qc=on_view_full_qc,
            display_names=display_names_dict,
            live_metrics=summary_live_metrics,
        )
        dialog.show()

    def _find_latest_feedback_file(self, output_dir: Path, task: str, run: int, suffix: str) -> Path | None:
        """Find the most recent run-NN_attempt-NN feedback file for this task/run.

        Args:
            output_dir: Directory containing feedback output files
            task: Task type (feedback/transfer)
            run: Run number
            suffix: File suffix (e.g., "feedbacklog.tsv", "psychopylog.tsv")

        Returns:
            Path to the most recent matching file, or None if not found
        """
        pattern = f"{self._bids_prefix}_task-{task}_run-{run:02d}_attempt-*_{suffix}"
        matches = list(output_dir.glob(pattern))

        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        return None

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

    def _build_resumption_banner(self) -> ft.Container | None:
        """Build a banner showing already-complete steps when resuming a session.

        Returns:
            A styled container with step summary and overwrite checkbox, or None
            if no steps are complete (fresh session).
        """
        completed = [step for step, status in self.step_statuses.items() if status == StepStatus.COMPLETE]
        if not completed:
            return None

        # Friendly labels for step IDs
        def _label(step_id: str) -> str:
            labels = {
                "reference_initial": "Initial Ref",
                "preprocessing": "Preprocessing",
                "mask_extraction": "Mask Extraction",
                "coregistration": "Coregistration",
                "reference_realtime": "RT Ref",
                "coregistration_realtime": "RT Coregistration",
            }
            if step_id in labels:
                return labels[step_id]
            if step_id.startswith("rest_run_"):
                return f"Rest Run {step_id.split('_')[-1]}"
            if step_id.startswith("localizer_") and "_run_" in step_id:
                parts = step_id.split("_")
                return f"Loc {int(parts[1]) + 1} Run {parts[-1]}"
            if step_id.startswith("localizer_") and step_id.endswith("_preprocessing"):
                return f"Loc {int(step_id.split('_')[1]) + 1} Preproc"
            if step_id.startswith("localizer_") and step_id.endswith("_extraction"):
                return f"Loc {int(step_id.split('_')[1]) + 1} Extraction"
            if step_id.startswith("localizer_") and step_id.endswith("_approval"):
                return f"Loc {int(step_id.split('_')[1]) + 1} Approval"
            if step_id.startswith("feedback_"):
                return f"Feedback {step_id.split('_')[-1]}"
            if step_id.startswith("transfer_"):
                return f"Transfer {step_id.split('_')[-1]}"
            return step_id

        step_names = ", ".join(_label(s) for s in completed)

        # Last-modified time of the session directory
        import datetime

        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        last_modified = ""
        try:
            mtime = base_dir.stat().st_mtime
            last_modified = f"  ·  Last modified: {datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')}"
        except OSError:
            pass

        banner = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.RESTORE, color=ft.Colors.AMBER_700, size=20),
                    ft.Column(
                        [
                            ft.Text(
                                f"Resuming session — {len(completed)} step(s) already complete{last_modified}",
                                size=13,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.AMBER_900,
                            ),
                            ft.Text(
                                step_names,
                                size=12,
                                color=ft.Colors.AMBER_800,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    ft.IconButton(
                        ft.Icons.CLOSE,
                        icon_color=ft.Colors.AMBER_700,
                        icon_size=18,
                        tooltip="Dismiss",
                        on_click=lambda e: self._dismiss_resumption_banner(),
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=8),
            bgcolor=ft.Colors.AMBER_50,
            border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.AMBER_200)),
        )
        self._resumption_banner = banner
        return banner

    def _dismiss_resumption_banner(self):
        """Hide the resumption banner."""
        if self._resumption_banner:
            self._resumption_banner.visible = False
            self.page.update()

    # ------------------------------------------------------------------
    # Config change banner / dialog
    # ------------------------------------------------------------------

    def _build_config_change_banner(self) -> ft.Container | None:
        """Build a warning banner when the study config has changed since last run.

        Returns:
            A styled container with a summary and action buttons, or None
            if there are no config differences.
        """
        if not self._config_differences:
            return None

        n = len(self._config_differences)

        banner = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.ORANGE_700, size=20),
                    ft.Text(
                        f"Study configuration has {n} change{'s' if n != 1 else ''} since last run",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ORANGE_900,
                        expand=True,
                    ),
                    ft.TextButton(
                        "Review",
                        on_click=lambda e: self._show_config_change_dialog(),
                    ),
                    ft.FilledButton(
                        "Update",
                        on_click=lambda e: self._accept_config_changes(),
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=8),
            bgcolor=ft.Colors.ORANGE_50,
            border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.ORANGE_200)),
        )
        self._config_change_banner = banner
        return banner

    def _build_simulator_banner(self) -> ft.Container | None:
        """Build a persistent banner shown when simulator mode is enabled.

        Returns:
            A styled container, or None if simulator mode is disabled.
        """
        simulator_cfg = self.study_config.session_config.get("simulator", {})
        if not simulator_cfg.get("enabled", True):
            return None

        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FAST_FORWARD, color=ft.Colors.RED_700, size=20),
                    ft.Text(
                        "Simulator mode — replaying recorded DICOMs (no live scanner)",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.RED_900,
                        expand=True,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=8),
            bgcolor=ft.Colors.RED_50,
            border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.RED_200)),
        )

    def _show_config_change_dialog(self):
        """Show a dialog listing all config differences."""
        rows = []
        for diff in self._config_differences:
            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(diff.field_path, size=12)),
                        ft.DataCell(ft.Text(str(diff.saved_value), size=12)),
                        ft.DataCell(ft.Text(str(diff.current_value), size=12)),
                    ]
                )
            )

        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Field", weight=ft.FontWeight.BOLD, size=12)),
                ft.DataColumn(ft.Text("Previous", weight=ft.FontWeight.BOLD, size=12)),
                ft.DataColumn(ft.Text("Current", weight=ft.FontWeight.BOLD, size=12)),
            ],
            rows=rows,
            column_spacing=20,
        )

        def _close(e):
            dialog.open = False
            self.page.update()

        def _update(e):
            dialog.open = False
            self.page.update()
            self._accept_config_changes()

        dialog = ft.AlertDialog(
            modal=False,
            title=ft.Text("Study Configuration Changed"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "The following protocol settings differ from the "
                            "snapshot saved when this participant was first run:",
                            size=13,
                        ),
                        ft.Container(height=8),
                        table,
                    ],
                    scroll=ft.ScrollMode.AUTO,
                    tight=True,
                ),
                width=600,
                height=400,
            ),
            actions=[
                ft.TextButton("Close", on_click=_close),
                ft.FilledButton("Update Snapshot", on_click=_update),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _accept_config_changes(self):
        """Archive the old snapshot and save the current config."""
        archive_config_snapshot(self._participant_base_dir)
        save_config_snapshot(
            self.study_config,
            self._participant_base_dir,
            self.participant_id,
            self.session_id,
        )
        self._config_differences = []
        if hasattr(self, "_config_change_banner") and self._config_change_banner:
            self._config_change_banner.visible = False
            self.page.update()
        logger.info("Config snapshot updated for %s", self.participant_id)

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
        # Determine button state based on next step and running state
        next_step = self._get_next_step()
        if next_step is None:
            button_text = "Session Complete"
            button_disabled = True
            button_icon = ft.icons.Icons.CHECK_CIRCLE
        else:
            step_id, step_label = next_step
            button_text = f"Run Next: {step_label}"
            button_disabled = self._is_running  # Disable when running
            button_icon = ft.icons.Icons.PLAY_ARROW

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
        param_rows: list[ft.Control] = []
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
        is_feedback_transfer = step_id.startswith("transfer_") or step_id.startswith("feedback_")
        is_collection = (
            step_id.startswith("rest_run_")
            or (step_id.startswith("localizer_") and "_run_" in step_id)
            or step_id in ("reference_initial", "reference_realtime")
        )
        if outputs_exist:
            if is_collection or is_feedback_transfer:
                # Collection and feedback steps use attempt numbering - previous data is archived
                warning_text = "Previous run data exists and will be archived before starting fresh."
                warning_icon = ft.icons.Icons.ARCHIVE
            else:
                warning_text = (
                    "Output files from a previous run exist. After you click Start you'll "
                    "choose Skip Existing (keep them) or Overwrite All (re-run from scratch)."
                )
                warning_icon = ft.icons.Icons.REPLAY

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
            content_items.extend(
                [
                    ft.Container(height=10),
                    overwrite_warning,
                ]
            )

        content_items.extend(
            [
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
            ]
        )

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
        network_names = list(config["localizers"][0]["networks"])

        # Include sham alternative networks if sham is enabled with alternative masks
        sham_config = config.get("sham", {})
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            sham_mask_mode = sham_config.get("sham_mask_mode", "two_networks")
            if sham_mask_mode == "hemisphere_split":
                # Hemisphere split mode - add the network that will be split
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                if hemisphere_network and hemisphere_network not in network_names:
                    network_names.append(hemisphere_network)
            else:
                # Two networks mode (legacy)
                alternative_targets = sham_config.get("alternative_mask_targets", [])
                for target in alternative_targets:
                    if target and target not in network_names:
                        network_names.append(target)

        return network_names

    def _resolve_mask_names(self, config: dict) -> list[str]:
        """Resolve mask names from the first localizer.

        Mirrors :pyattr:`LocalizerConfig.mask_names`: rest uses ``networks``,
        manual uses ``manual_mask_names``, task uses ``contrasts`` keys (with
        optional ``_reverse`` variants).
        """
        loc = _loc(config)
        if not loc:
            return []
        loc_type = loc.get("type", "rest")
        if loc_type == "manual":
            return list(loc.get("manual_mask_names", []))
        if loc_type == "rest":
            return list(loc.get("networks", []))
        contrasts = loc.get("contrasts", {})
        names = list(contrasts.keys())
        if loc.get("include_reverse_contrasts", False):
            names += [f"{n}_reverse" for n in names]
        return names

    def _get_blind_display_names(self, mask_names: list[str]) -> list[str] | None:
        """Get blinded display names if blind mode is enabled.

        When blind mode is enabled, returns generic labels instead of actual
        network names to maintain operator blinding.

        Args:
            mask_names: List of actual network mask names

        Returns:
            List of display names if blind mode enabled, None otherwise
        """
        sham_config = self.study_config.session_config.get("sham", {})
        if sham_config.get("blind_mode", False):
            # First mask = Target (top ring), Second = Non-Target (bottom ring)
            # Additional masks get generic labels to maintain blinding
            blind_labels = ["Target", "Non-Target"]
            # Extend with generic labels for any additional masks (sham alternatives)
            for i in range(2, len(mask_names)):
                blind_labels.append(f"Network {i + 1}")
            return blind_labels[: len(mask_names)]
        return None

    def _get_blind_display_names_dict(self, mask_names: list[str]) -> dict[str, str] | None:
        """Get blinded display names as a dict mapping for FeedbackSummaryDialog.

        Args:
            mask_names: List of actual network mask names

        Returns:
            Dict mapping network names to display names if blind mode enabled, None otherwise
        """
        display_names = self._get_blind_display_names(mask_names)
        if display_names:
            return {mask_names[i]: display_names[i] for i in range(min(len(mask_names), len(display_names)))}
        return None

    def _get_mask_extraction_step_info(self, config: dict, base_dir: Path) -> dict:
        """Get step info for mask extraction, including sham networks if enabled.

        Args:
            config: Session configuration
            base_dir: Base directory path

        Returns:
            Dictionary with step information
        """
        loc0 = _loc(config)
        loc_type = loc0.get("type", "rest")

        if loc_type == "manual":
            from pineuro.gui.components.manual_masks import read_manual_masks

            names = loc0.get("manual_mask_names", []) or []
            record = read_manual_masks(base_dir, loc0.get("id", "manual"))
            reference = (record or {}).get("reference") or "(select per participant)"
            return {
                "label": "Import Network Masks",
                "description": f"Import {len(names)} user-supplied mask(s) into the feedback pipeline.",
                "parameters": {
                    "Masks": ", ".join(names) or "(none)",
                    "Mask space": loc0.get("manual_mask_space", "native"),
                    "Reference": str(reference),
                    "Output": base_dir / "mask_extraction",
                },
                "output_path": base_dir / "mask_extraction",
                "output_pattern": "0*",
            }

        if loc_type != "rest":
            contrasts = loc0.get("contrasts", {})
            extraction = loc0.get("extraction", {})
            return {
                "label": "Extract Network Masks (GLM)",
                "description": f"GLM contrast-based extraction of {len(contrasts)} masks from task localizer.",
                "parameters": {
                    "Contrasts": ", ".join(f"{k}: {v}" for k, v in contrasts.items()),
                    "Threshold": f"{extraction.get('threshold', 2000)} voxels",
                    "Smoothing": f"{extraction.get('smooth_fwhm', 5.0)} mm FWHM",
                    "Output": base_dir / "mask_extraction",
                },
                "output_path": base_dir / "mask_extraction",
                "output_pattern": "0*",
            }
        else:
            all_networks = self._get_all_network_targets(config)
            return {
                "label": "Extract Network Masks (ICA)",
                "description": f"ICA-based extraction of {len(all_networks)} network masks using {config['localizers'][0]['extraction']['n_yeo']}-network Yeo atlas.",
                "parameters": {
                    "Networks": ", ".join(all_networks),
                    "Yeo Atlas": f"{config['localizers'][0]['extraction']['n_yeo']} networks",
                    "Smoothing": f"{config['localizers'][0]['extraction']['smooth_fwhm']} mm FWHM",
                    "Highpass": f"{config['localizers'][0]['extraction']['highpass']} Hz",
                    "Output": base_dir / "mask_extraction",
                },
                "output_path": base_dir / "mask_extraction",
                "output_pattern": "0*",
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
        session_metadata = config.get("session", {})
        is_mask_reuse = session_metadata.get("mask_source") == "reuse"
        mask_source_session = session_metadata.get("mask_source_session")

        # In mask reuse mode, reference is run-01 (no initial reference)
        run_num = 1 if is_mask_reuse else 2

        if is_mask_reuse and mask_source_session:
            # Mask reuse mode: masks come from source session
            study_base_dir = base_dir.parent.parent  # Go up from participant/session to study
            source_session_dir = study_base_dir / self.participant_id / mask_source_session
            me_dir = source_session_dir / "mask_extraction"
            input_masks_path = me_dir / "03_masks"
            description = (
                f"Register network masks from {mask_source_session} to real-time reference space for feedback runs."
            )
        else:
            # Standard mode: masks come from current session
            me_dir = base_dir / "mask_extraction"
            input_masks_path = me_dir / "03_masks"
            description = "Register network masks to real-time reference space for feedback runs."

        return {
            "label": "Coregister to Real-time Reference",
            "description": description,
            "parameters": {
                "Input Masks": input_masks_path,
                "Target": f"reference run-{run_num:02d}",
                "Output": base_dir / "feedback" / "01_reference" / "masks",
            },
            "output_path": base_dir / "feedback" / "01_reference" / "masks",
            "output_pattern": f"*_to_task-reference_run-{run_num:02d}_*.nii.gz",
        }

    def _get_reference_initial_step_info(self, config: dict, base_dir: Path) -> dict:
        """Get step info for initial reference collection with attempt tracking.

        Args:
            config: Session configuration
            base_dir: Base directory path

        Returns:
            Dictionary with step information
        """
        enabled = config.get("reference", {}).get("initial_enabled", True)

        if not enabled:
            return {
                "label": "Collect Initial Reference (Disabled)",
                "description": "Initial Reference collection is disabled in study configuration.",
                "parameters": {"Status": "Disabled"},
                "output_path": base_dir / "collection",
                "output_pattern": "task-reference_run-01_attempt-*",
            }

        from pineuro.session import get_localizer_run_info

        n_vols = config.get("reference", {}).get("initial_n_volumes", 2)
        collection_dir = base_dir / "collection"

        run_info = get_localizer_run_info(
            collection_dir=collection_dir,
            task="task-reference",
            run=1,
        )

        description = "Collect 2 volumes for initial positioning verification before localizer acquisition."
        if run_info["existing_attempts"]:
            attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
            description += f" Previous attempts ({attempts_str}) will be preserved."

        info: dict[str, Any] = {
            "label": "Collect Initial Reference",
            "description": description,
            "parameters": {
                "Task": "reference (run-01)",
                "Attempt": f"attempt-{run_info['attempt']:02d}",
                "Volumes": n_vols,
                "Source": config.get("reference", {}).get("initial_dicoms", ""),
                "Output": base_dir / "collection" / f"task-reference_run-01_attempt-{run_info['attempt']:02d}",
                "Enabled": "Yes",
            },
            "output_path": base_dir / "collection",
            "output_pattern": "task-reference_run-01_attempt-*",
        }
        if run_info["existing_attempts"]:
            attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
            info["parameters"]["Previous Attempts"] = attempts_str

        return info

    def _get_reference_realtime_step_info(self, config: dict, base_dir: Path) -> dict:
        """Get step info for realtime reference collection, handling mask reuse mode.

        Args:
            config: Session configuration
            base_dir: Base directory path

        Returns:
            Dictionary with step information
        """
        reference_enabled = config.get("reference", {}).get("realtime_enabled", True)

        # Check for mask reuse mode
        session_metadata = config.get("session", {})
        is_mask_reuse = session_metadata.get("mask_source") == "reuse"

        # In mask reuse mode, reference is run-01 (no initial reference)
        run_num = 1 if is_mask_reuse else 2

        if not reference_enabled:
            return {
                "label": "Collect Real-time Reference (Disabled)",
                "description": "Real-time Reference collection is disabled. Feedback will use masks from subject boldref space.",
                "parameters": {"Status": "Disabled - using boldref masks"},
                "output_path": base_dir / "collection",
                "output_pattern": f"task-reference_run-{run_num:02d}_attempt-*",
            }

        from pineuro.session import get_localizer_run_info

        n_vols = config.get("reference", {}).get("realtime_n_volumes", 2)
        collection_dir = base_dir / "collection"

        run_info = get_localizer_run_info(
            collection_dir=collection_dir,
            task="task-reference",
            run=run_num,
        )

        description = "Collect 2 volumes immediately before feedback tasks for real-time registration verification."
        if run_info["existing_attempts"]:
            attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
            description += f" Previous attempts ({attempts_str}) will be preserved."

        info: dict[str, Any] = {
            "label": "Collect Real-time Reference",
            "description": description,
            "parameters": {
                "Task": f"reference (run-{run_num:02d})",
                "Attempt": f"attempt-{run_info['attempt']:02d}",
                "Volumes": n_vols,
                "Source": config.get("reference", {}).get("realtime_dicoms", ""),
                "Output": base_dir
                / "collection"
                / f"task-reference_run-{run_num:02d}_attempt-{run_info['attempt']:02d}",
                "Enabled": "Yes",
            },
            "output_path": base_dir / "collection",
            "output_pattern": f"task-reference_run-{run_num:02d}_attempt-*",
        }
        if run_info["existing_attempts"]:
            attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
            info["parameters"]["Previous Attempts"] = attempts_str

        return info

    def _get_transfer_step_info(self, config: dict, base_dir: Path, participant_id: str, transfer_num: int) -> dict:
        """Build step info for a transfer run, using per-FRC values when available."""
        step_id = f"transfer_{transfer_num}"
        frc_idx = self._get_feedback_step_frc_index(step_id)
        feedback_runs = config.get("feedback_runs", [])
        frc = feedback_runs[frc_idx] if frc_idx is not None and frc_idx < len(feedback_runs) else None

        def _val(key, default=None):
            if frc is None:
                return default
            if key in ("fd_threshold", "dvars_threshold", "tsnr_threshold"):
                return frc.get("qc", {}).get(key, default)
            if key in (
                "motion_correction",
                "include_drift",
                "include_motion_regressors",
                "include_wm_regressor",
                "include_csf_regressor",
                "smooth_fwhm",
            ):
                return frc.get("glm", {}).get(key, default)
            return frc.get(key, default)

        description = (
            "Practice run without neurofeedback to establish baseline performance."
            if transfer_num == 1
            else "Final practice run without feedback to assess learning transfer."
        )

        return {
            "label": f"Transfer Run {transfer_num}",
            "description": description,
            "parameters": {
                "Task": f"transfer (run-{transfer_num:02d})",
                "Volumes": _val("n_volumes", 150),
                "Baseline": _val("n_baseline", 25),
                "Feedback": "OFF",
                "MoCo": {
                    "siemens_moco": "Siemens (scanner)",
                    "software": "ANTsPy (software)",
                    "none": "Disabled",
                }.get(_val("motion_correction", "siemens_moco"), "Unknown"),
                "Output": base_dir / "feedback" / "02_feedback_output",
                "Simulator Speed": f"{config.get('simulator', {}).get('feedback_speed', 1.0)}x",
            },
            "output_path": base_dir / "feedback" / "02_feedback_output",
            "output_pattern": f"{self._bids_prefix}_task-transfer_run-{transfer_num:02d}_attempt-*_feedbacklog.tsv",
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
        from pineuro.feedback import get_slot_run_info
        from pineuro.session import get_localizer_run_info
        from pineuro.tasks.state import TaskState

        # Load task state for adaptive state display
        feedback_dir = base_dir / "feedback" / "02_feedback_output"
        # Cache task states per task type for multi-FRC support
        feedback_runs = config.get("feedback_runs", [])
        _task_states: dict[str, TaskState | None] = {}
        default_task_type = config["feedback_runs"][0].get("task", "ball") if config.get("feedback_runs") else "ball"

        def _get_task_state(task_id: str) -> TaskState | None:
            if task_id not in _task_states:
                try:
                    _task_states[task_id] = TaskState.load(feedback_dir, task_id=task_id)
                except Exception:
                    _task_states[task_id] = None
            return _task_states[task_id]

        def get_task_state_summary(step_id: str) -> str:
            """Get task state summary string for a specific step."""
            frc_idx = self._get_feedback_step_frc_index(step_id)
            if frc_idx is not None and frc_idx < len(feedback_runs):
                task_id = feedback_runs[frc_idx].get("task", default_task_type)
            else:
                task_id = default_task_type
            task_state = _get_task_state(task_id)
            if not task_state or not task_state.task_state:
                return "initial"

            if step_id in ["transfer_1", "feedback_1"]:
                return "initial"

            if not task_state.completed_runs:
                return "initial"

            # Show key task state values
            state = task_state.task_state
            parts = []
            for key, value in state.items():
                if isinstance(value, float):
                    parts.append(f"{key}={value:.1f}")
                else:
                    parts.append(f"{key}={value}")
            return ", ".join(parts) if parts else "initial"

        # Map step IDs to their information
        step_info_map = {
            "reference_initial": self._get_reference_initial_step_info(config, base_dir),
            "preprocessing": {
                "label": "Run Preprocessing",
                "description": "Motion correction, brain extraction, and coregistration of localizer runs.",
                "parameters": {
                    "Input": base_dir / "collection" / "task-*_run-*_attempt-*",
                    "Output": base_dir / "preprocessing",
                    "Expected Outputs": "preproc.nii.gz, mask.nii.gz, boldref.nii.gz",
                },
                "output_path": base_dir / "preprocessing",
                "output_pattern": "0*",
            },
            "mask_extraction": self._get_mask_extraction_step_info(config, base_dir),
            "mni_mask_extraction": {
                "label": "Extract MNI Atlas Masks",
                "description": "Register MNI atlas to subject space and extract network masks directly from atlas parcellation.",
                "parameters": {
                    "Atlas": _loc(config).get("extraction", {}).get("network_atlas", "yeo_17"),
                    "Networks": ", ".join(
                        config["localizers"][0].get("networks", ["DefaultA", "ContA"])
                        if config.get("localizers")
                        else ["DefaultA", "ContA"]
                    ),
                    "Output": base_dir / "mask_extraction" / "03_masks",
                },
                "output_path": base_dir / "mask_extraction" / "03_masks",
                "output_pattern": f"{self._bids_prefix}_*_thr.nii.gz",
            },
            "coregistration": {
                "label": "Coregister to Reference",
                "description": "Register network masks from rest space to reference feedback space.",
                "parameters": {
                    "Input Masks": base_dir / "mask_extraction" / "03_masks",
                    "Target": "reference run-01",
                    "Output": base_dir / "mask_extraction" / "04_reference" / "masks",
                },
                "output_path": base_dir / "mask_extraction" / "04_reference" / "masks",
                "output_pattern": "*_to_task-reference_run-01_*.nii.gz",
            },
            "reference_realtime": self._get_reference_realtime_step_info(config, base_dir),
            "coregistration_realtime": self._get_coregistration_realtime_step_info(config, base_dir),
            "transfer_1": self._get_transfer_step_info(config, base_dir, participant_id, transfer_num=1),
            # Note: feedback_N entries are generated dynamically below
            "transfer_2": self._get_transfer_step_info(config, base_dir, participant_id, transfer_num=2),
        }

        # Handle multi-localizer step IDs
        if step_id.startswith("localizer_") and "_run_" in step_id:
            parts = step_id.split("_")
            loc_idx = int(parts[1])
            run_num = int(parts[-1])

            localizers = config.get("localizers", [])
            loc = localizers[loc_idx] if loc_idx < len(localizers) else {}
            loc_type = loc.get("type", "rest")
            loc_n_vols = loc.get("n_volumes", 250)
            loc_tr = loc.get("tr", 1.2)
            loc_source_dicoms = loc.get("source_dicoms", [])
            source_dicom = loc_source_dicoms[run_num - 1] if run_num <= len(loc_source_dicoms) else ""

            # Get display name
            if loc_type == "rest":
                loc_display = "Resting State"
            else:
                try:
                    from pineuro.tasks import get_task_metadata

                    task_meta = get_task_metadata(loc_type)
                    loc_display = task_meta.display_name
                except (ValueError, ImportError):
                    loc_display = loc_type.replace("_", " ").title()

            loc_task_name = "rest" if loc_type == "rest" else loc_type
            collection_dir = base_dir / "collection"
            run_info = get_localizer_run_info(
                collection_dir=collection_dir,
                task=f"task-{loc_task_name}",
                run=run_num,
            )

            description = f"Collect {loc_display} data (run {run_num}) for network mask extraction."
            if run_info["existing_attempts"]:
                attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
                description += f" Previous attempts ({attempts_str}) will be preserved."

            params: dict[str, Any] = {
                "Task": f"{loc_display} (run-{run_num:02d})",
                "Attempt": f"attempt-{run_info['attempt']:02d}",
                "Volumes": loc_n_vols,
                "TR": f"{loc_tr}s",
                "Source": source_dicom or "(not configured)",
                "Output": base_dir
                / "collection"
                / f"task-{loc_task_name}_run-{run_num:02d}_attempt-{run_info['attempt']:02d}",
                "Simulator Speed": f"{config.get('simulator', {}).get('collection_speed', 1.0)}x",
            }

            # Add task-specific parameters for non-rest localizers
            if loc_type != "rest":
                task_params = loc.get("task_params", {})
                for key, value in task_params.items():
                    display_key = key.replace("_", " ").title()
                    params[display_key] = value
            else:
                # Rest-specific params
                if loc.get("n_components"):
                    params["ICA Components"] = loc["n_components"]
                if loc.get("networks", loc.get("yeo_targets")):
                    params["Target Networks"] = ", ".join(loc.get("networks", loc.get("yeo_targets", [])))
                if loc.get("network_atlas"):
                    params["Atlas"] = loc["network_atlas"]

            if run_info["existing_attempts"]:
                params["Previous Attempts"] = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])

            info: dict[str, Any] = {
                "label": f"Collect {loc_display} Run {run_num}",
                "description": description,
                "parameters": params,
                "output_path": base_dir / "collection",
                "output_pattern": f"task-{loc_task_name}_run-{run_num:02d}_attempt-*",
            }

        elif step_id.startswith("localizer_") and step_id.endswith("_preprocessing"):
            loc_idx = int(step_id.split("_")[1])
            localizers = config.get("localizers", [])
            loc = localizers[loc_idx] if loc_idx < len(localizers) else {}
            loc_type = loc.get("type", "rest")
            loc_task_name = "rest" if loc_type == "rest" else loc_type

            if loc_type == "rest":
                loc_display = "Resting State"
            else:
                try:
                    from pineuro.tasks import get_task_metadata

                    task_meta = get_task_metadata(loc_type)
                    loc_display = task_meta.display_name
                except (ValueError, ImportError):
                    loc_display = loc_type.replace("_", " ").title()

            info = {
                "label": f"Preprocess {loc_display}",
                "description": f"Motion correction, brain extraction, and coregistration of {loc_display.lower()} localizer data.",
                "parameters": {
                    "Task": loc_display,
                    "Runs": loc.get("n_runs", 2),
                    "Input": base_dir / "collection" / f"task-{loc_task_name}_*",
                    "Output": base_dir / "preprocessing",
                    "Expected Outputs": f"{self._bids_prefix}_task-{loc_task_name}_preproc.nii.gz, _mask.nii.gz, _boldref.nii.gz",
                },
                "output_path": base_dir / "preprocessing",
                "output_pattern": f"{self._bids_prefix}_task-{loc_task_name}_preproc.nii.gz",
            }

        elif step_id.startswith("localizer_") and step_id.endswith("_extraction"):
            loc_idx = int(step_id.split("_")[1])
            localizers = config.get("localizers", [])
            loc = localizers[loc_idx] if loc_idx < len(localizers) else {}
            loc_type = loc.get("type", "rest")
            loc_task_name = "rest" if loc_type == "rest" else loc_type

            if loc_type == "rest":
                loc_display = "Resting State"
                method = "ICA"
                params = {
                    "Method": "ICA (Independent Component Analysis)",
                    "Atlas": loc.get("network_atlas", "yeo_17"),
                    "Components": loc.get("n_components", 20),
                    "Target Networks": ", ".join(loc.get("networks", loc.get("yeo_targets", []))),
                    "Smoothing": f"{loc.get('smooth_fwhm', 5.0)} mm FWHM",
                    "Highpass": f"{loc.get('highpass', 0.01)} Hz",
                    "Threshold": f"{loc.get('threshold', 2000)} ({loc.get('threshold_method', 'top_n')})",
                }
            elif loc_type == "manual":
                from pineuro.gui.components.manual_masks import read_manual_masks

                loc_display = "Manual Import"
                method = "Manual"
                _record = read_manual_masks(base_dir, loc.get("id", "manual"))
                params = {
                    "Method": "Manual import (user-supplied masks)",
                    "Mask space": loc.get("manual_mask_space", "native"),
                    "Masks": ", ".join(loc.get("manual_mask_names", []) or []) or "(none)",
                    "Reference": str((_record or {}).get("reference") or "(select per participant)"),
                }
            else:
                try:
                    from pineuro.tasks import get_task_metadata

                    task_meta = get_task_metadata(loc_type)
                    loc_display = task_meta.display_name
                except (ValueError, ImportError):
                    loc_display = loc_type.replace("_", " ").title()
                method = "GLM"
                contrasts = loc.get("contrasts", {})
                params = {
                    "Method": "GLM (General Linear Model)",
                    "Contrasts": ", ".join(f"{k}: {v}" for k, v in contrasts.items()) if contrasts else "(none)",
                    "Stat Type": loc.get("stat_type", "t"),
                    "HRF Model": loc.get("hrf_model", "spm"),
                    "Threshold": f"{loc.get('threshold', 2000)} ({loc.get('threshold_method', 'cluster')})",
                }

            loc_dir_name = f"loc_{loc_idx}_{loc_task_name}"
            params["Output"] = base_dir / "mask_extraction" / loc_dir_name / "03_masks"

            info = {
                "label": f"Extract {loc_display} Masks ({method})",
                "description": f"Extract network masks from {loc_display.lower()} localizer using {method}.",
                "parameters": params,
                "output_path": base_dir / "mask_extraction" / loc_dir_name / "03_masks",
                "output_pattern": f"{self._bids_prefix}_*.nii.gz",
            }

        # Handle dynamic rest_run_* steps (single-localizer)
        elif step_id.startswith("rest_run_"):
            run_num = int(step_id.split("_")[-1])
            run_idx = run_num - 1  # 0-indexed for accessing lists

            localizer_dicoms = config["localizers"][0].get("source_dicoms", []) if config.get("localizers") else []
            source_dicom = localizer_dicoms[run_idx] if run_idx < len(localizer_dicoms) else ""

            loc_task = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
            loc_name = loc_task if loc_task != "rest" else "rest"
            expected_vols = (config["localizers"][0].get("n_volumes") if config.get("localizers") else None) or 250

            # Get attempt info for this run (same pattern as feedback steps)
            collection_dir = base_dir / "collection"
            run_info = get_localizer_run_info(
                collection_dir=collection_dir,
                task=f"task-{loc_name}",
                run=run_num,
            )

            description = f"Collect localizer fMRI data (run {run_num}) for network mask extraction."
            if run_info["existing_attempts"]:
                attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
                description += f" Previous attempts ({attempts_str}) will be preserved."

            info = {
                "label": f"Collect Localizer Run {run_num}",
                "description": description,
                "parameters": {
                    "Task": f"{loc_task} (run-{run_num:02d})",
                    "Attempt": f"attempt-{run_info['attempt']:02d}",
                    "Volumes": expected_vols,
                    "Source": source_dicom,
                    "Output": base_dir
                    / "collection"
                    / f"task-{loc_name}_run-{run_num:02d}_attempt-{run_info['attempt']:02d}",
                    "Simulator Speed": f"{config.get('simulator', {}).get('collection_speed', 1.0)}x",
                },
                "output_path": base_dir / "collection",
                "output_pattern": f"task-{loc_name}_run-{run_num:02d}_attempt-*",
            }
            if run_info["existing_attempts"]:
                attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
                info["parameters"]["Previous Attempts"] = attempts_str
        # Handle dynamic feedback_N steps
        elif step_id.startswith("feedback_"):
            run_num = int(step_id.split("_")[-1])
            ordinal = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}.get(run_num, f"{run_num}th")

            frc_idx = self._get_feedback_step_frc_index(step_id)
            feedback_runs = config.get("feedback_runs", [])
            if frc_idx is not None and frc_idx < len(feedback_runs):
                frc = feedback_runs[frc_idx]
                _feedback_task_type = frc.get("task", "ball")
                networks_str = ", ".join(self._resolve_frc_mask_names(frc))
            else:
                _feedback_task_type = (
                    config["feedback_runs"][0].get("task", "ball") if config.get("feedback_runs") else "ball"
                )
                networks_str = ", ".join(self._resolve_mask_names(config))

            _frc = frc if (frc_idx is not None and frc_idx < len(feedback_runs)) else None

            def _fb_get(k: str, d: Any = None) -> Any:
                return _frc.get(k, d) if _frc else d

            params = {
                "Task": f"feedback (run-{run_num:02d})",
                "Volumes": _fb_get("n_volumes", 150),
                "Baseline": _fb_get("n_baseline", 25),
                "Feedback": "ON",
                "Networks": networks_str,
            }

            # Show task state summary if session has adaptive state
            state_summary = get_task_state_summary(step_id)
            if state_summary != "initial":
                params["Task State"] = state_summary

            params["MoCo"] = {
                "siemens_moco": "Siemens (scanner)",
                "software": "ANTsPy (software)",
                "none": "Disabled",
            }.get(_fb_get("motion_correction", "siemens_moco"), "Unknown")
            params["Output"] = base_dir / "feedback" / "02_feedback_output"
            params["Simulator Speed"] = f"{config.get('simulator', {}).get('feedback_speed', 1.0)}x"

            info = {
                "label": f"Feedback Run {run_num}",
                "description": f"{ordinal} neurofeedback run with real-time activation feedback.",
                "parameters": params,
                "output_path": base_dir / "feedback" / "02_feedback_output",
                "output_pattern": f"{self._bids_prefix}_task-feedback_run-{run_num:02d}_attempt-*_feedbacklog.tsv",
            }
        else:
            info = step_info_map.get(
                step_id,
                {
                    "label": step_id,
                    "description": "Unknown step",
                    "parameters": {},
                    "output_path": None,
                },
            )

        # For feedback/transfer steps, show slot-based run info with attempt entity
        feedback_step_ids = [sid for sid, _ in _build_feedback_step_ids(config)]

        if step_id in feedback_step_ids:
            task_type = "transfer" if "transfer" in step_id else "feedback"
            output_dir = base_dir / "feedback" / "02_feedback_output"

            # Get slot-based run info with attempt number
            run_info = get_slot_run_info(
                output_dir=output_dir,
                subject=participant_id,
                session=self.session_id,
                task=task_type,
                step_id=step_id,
            )

            # Update parameters to show run number and attempt
            info["parameters"]["Task"] = f"{task_type} (run-{run_info['run']:02d})"
            info["parameters"]["Attempt"] = f"attempt-{run_info['attempt']:02d}"

            # If there are previous attempts, show them
            if run_info["existing_attempts"]:
                attempts_str = ", ".join(f"attempt-{a:02d}" for a in run_info["existing_attempts"])
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
        # Restore empty preview panel
        if self.right_panel_container is not None:
            self.right_panel_container.content = self._build_preview_panel()
            self.right_panel_container.update()

    def _on_preview_start(self, step_id: str):
        """Handle start button in preview panel.

        Args:
            step_id: Step identifier to execute
        """
        is_feedback_transfer = step_id.startswith("transfer_") or step_id.startswith("feedback_")
        is_collection = (
            step_id.startswith("rest_run_")
            or (step_id.startswith("localizer_") and "_run_" in step_id)
            or step_id in ("reference_initial", "reference_realtime")
        )
        scanner_touching = is_feedback_transfer or is_collection

        def _after_preflight():
            if not is_feedback_transfer and not is_collection:
                config = self.study_config.session_config
                base_dir = Path(config["paths"]["base_dir"])
                step_info = self._get_step_info(step_id, config, base_dir, self.participant_id)
                if step_info.get("outputs_exist", False):
                    self._show_overwrite_dialog(step_id)
                    return
            self._continue_preview_start(step_id, overwrite=False)

        if scanner_touching:
            self._preflight_scanner_connection(_after_preflight)
        else:
            _after_preflight()

    def _preflight_scanner_connection(self, on_proceed: Callable[[], None]):
        """Run a quick scanner connection test before starting a scanner-touching run.

        Skipped in simulator mode. On success, silently invokes ``on_proceed``.
        On failure, shows a warning dialog with Proceed Anyway / Cancel actions.
        Also refreshes the left-panel Scanner Connection status indicator.
        """
        config = self.study_config.session_config
        is_simulator = config.get("simulator", {}).get("enabled", True)
        if is_simulator:
            on_proceed()
            return

        scanner_ip = config.get("scanner", {}).get("ip", "")
        watch_dir = config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM")

        progress_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.SETTINGS_ETHERNET, color=ft.Colors.BLUE),
                    ft.Text("Checking Scanner Connection", weight=ft.FontWeight.BOLD),
                ],
                spacing=10,
            ),
            content=ft.Container(
                content=ft.Row(
                    [
                        ft.ProgressRing(width=20, height=20, stroke_width=2),
                        ft.Text("Testing connection to scanner...", size=13),
                    ],
                    spacing=12,
                ),
                width=400,
                padding=10,
            ),
        )
        self.page.overlay.append(progress_dialog)
        progress_dialog.open = True
        self.page.update()

        def _close_progress():
            progress_dialog.open = False
            with contextlib.suppress(Exception):
                self.page.update()

        def _update_status_indicator(ok: bool):
            icon = getattr(self, "_scanner_status_icon", None)
            text = getattr(self, "_scanner_status_text", None)
            if icon is None or text is None:
                return
            if ok:
                icon.name = ft.Icons.CHECK_CIRCLE
                icon.color = ft.Colors.GREEN
                text.value = "Connection OK"
                text.color = ft.Colors.GREEN
            else:
                icon.name = ft.Icons.ERROR
                icon.color = ft.Colors.RED
                text.value = "Issues detected"
                text.color = ft.Colors.RED
            text.italic = False
            with contextlib.suppress(Exception):
                self.page.update()

        def _show_failure_dialog(result):
            ping_detail = result.ping_detail or "(no ping info)"
            dir_detail = result.dir_detail or "(no directory info)"
            ping_ok = result.ping_ok is not False
            dir_ok = result.dir_exists and result.dir_writable

            def _ping_icon():
                if result.ping_ok is None:
                    return ft.Icon(ft.Icons.REMOVE_CIRCLE_OUTLINE, color=ft.Colors.GREY, size=20)
                return ft.Icon(
                    ft.Icons.CHECK_CIRCLE if ping_ok else ft.Icons.ERROR,
                    color=ft.Colors.GREEN if ping_ok else ft.Colors.RED,
                    size=20,
                )

            def _dir_icon():
                return ft.Icon(
                    ft.Icons.CHECK_CIRCLE if dir_ok else ft.Icons.ERROR,
                    color=ft.Colors.GREEN if dir_ok else ft.Colors.RED,
                    size=20,
                )

            def _row(icon: ft.Control, label: str, detail: str) -> ft.Row:
                return ft.Row(
                    [
                        icon,
                        ft.Text(f"{label}: ", weight=ft.FontWeight.W_500, size=13),
                        ft.Text(detail, size=12, color=ft.Colors.GREY_700, expand=True),
                    ],
                    spacing=8,
                )

            warning_dialog = ft.AlertDialog(modal=True)

            def _close_warning():
                warning_dialog.open = False
                with contextlib.suppress(Exception):
                    self.page.update()

            def _on_cancel(e):
                _close_warning()

            def _on_proceed_anyway(e):
                _close_warning()
                on_proceed()

            warning_dialog.title = ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING, color=ft.Colors.ORANGE),
                    ft.Text("Scanner Connection Issue", weight=ft.FontWeight.BOLD),
                ],
                spacing=10,
            )
            warning_dialog.content = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "The scanner connection check reported problems:",
                            size=13,
                        ),
                        ft.Container(height=6),
                        _row(_ping_icon(), "Network Ping", ping_detail),
                        _row(_dir_icon(), "Watch Directory", dir_detail),
                        ft.Container(height=10),
                        ft.Text(
                            "Proceed anyway?",
                            size=13,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    spacing=6,
                ),
                width=520,
            )
            warning_dialog.actions = [
                ft.TextButton("Cancel", on_click=_on_cancel),
                ft.FilledButton("Proceed Anyway", on_click=_on_proceed_anyway),
            ]
            warning_dialog.actions_alignment = ft.MainAxisAlignment.END

            self.page.overlay.append(warning_dialog)
            warning_dialog.open = True
            with contextlib.suppress(Exception):
                self.page.update()

        def _run_test():
            from pineuro.scanner_setup import test_scanner_connection

            try:
                result = test_scanner_connection(scanner_ip=scanner_ip, watch_dir=watch_dir)
            except Exception as e:
                logger.error(f"Scanner preflight test failed: {e}")
                result = None

            async def _apply():
                _close_progress()
                if result is None:
                    _update_status_indicator(False)
                    on_proceed()
                    return
                _update_status_indicator(result.overall_ok)
                if result.overall_ok:
                    on_proceed()
                else:
                    _show_failure_dialog(result)

            try:
                self.page.run_task(_apply)
            except Exception as e:
                logger.error(f"Failed to apply preflight result: {e}")

        self.page.run_thread(_run_test)

    def _show_overwrite_dialog(self, step_id: str):
        """Show dialog asking user whether to skip existing outputs or overwrite.

        Args:
            step_id: Step identifier to execute after user chooses
        """

        def on_skip(e):
            dialog.open = False
            self.page.update()
            self._continue_preview_start(step_id, overwrite=False)

        def on_overwrite(e):
            dialog.open = False
            self.page.update()
            self._continue_preview_start(step_id, overwrite=True)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Outputs Already Exist"),
            content=ft.Text(
                "Output files from a previous run were found.\n\n"
                "Skip existing will only re-run steps whose outputs are missing. "
                "Overwrite will re-run everything from scratch."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: _close(e)),
                ft.OutlinedButton("Overwrite All", on_click=on_overwrite),
                ft.FilledButton("Skip Existing", on_click=on_skip),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        def _close(e):
            dialog.open = False
            self.page.update()

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _continue_preview_start(self, step_id: str, overwrite: bool):
        """Continue starting a step after overwrite decision is made.

        Args:
            step_id: Step identifier to execute
            overwrite: Whether to overwrite existing outputs
        """
        # Store overwrite choice to pass to pipeline runner
        self._pending_overwrite = overwrite

        # Manual-mask import steps need this participant's files chosen first.
        manual_loc_idx = self._manual_localizer_for_step(step_id)
        if manual_loc_idx is not None:
            from pineuro.gui.components.manual_masks import manual_masks_complete

            base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
            loc = self.study_config.session_config["localizers"][manual_loc_idx]
            if not manual_masks_complete(base_dir, loc):
                self._selected_rest_runs = None
                self._selected_attempts = None
                self._show_manual_mask_config_dialog(
                    loc_idx=manual_loc_idx,
                    on_saved=lambda: self._start_step(step_id),
                )
                return

        # For preprocessing, show unified attempt selection dialog if needed
        if step_id == "preprocessing":
            if self._needs_unified_dialog():
                self._show_unified_attempt_dialog(lambda _: self._start_step(step_id))
                return
            self._selected_rest_runs = None
            self._selected_attempts = None
            self._start_step(step_id)
            return

        # Clear any previous selection for non-preprocessing steps
        self._selected_rest_runs = None
        self._selected_attempts = None
        self._start_step(step_id)

    def _start_step(self, step_id: str):
        """Actually start executing a pipeline step.

        This is called after any pre-step dialogs (like run selection) are completed.

        Args:
            step_id: Step identifier to execute
        """
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
                session_id=self.session_id,
                on_progress=self._on_pipeline_progress,
                on_complete=self._on_pipeline_complete,
                on_error=self._on_pipeline_error,
                on_collection_progress=self._on_collection_progress,
                on_qc_data=self._on_collection_qc_data,
            )

        # Pass overwrite choice to runner
        self.pipeline_runner.set_overwrite(getattr(self, "_pending_overwrite", False))

        # Pass selected runs and attempts to runner for preprocessing step
        if step_id == "preprocessing":
            self.pipeline_runner.set_selected_rest_runs(self._selected_rest_runs)
            self.pipeline_runner.set_selected_attempts(self._selected_attempts)
        else:
            self.pipeline_runner.set_selected_rest_runs(None)
            self.pipeline_runner.set_selected_attempts(None)

        # Show monitoring panel (creates task_mirror for feedback steps)
        self._show_monitoring_panel(step_id)

        # Connect task mirror callback for steps that show a TaskMirror:
        # - feedback/transfer steps
        # - all rest_run_ steps (fixation or task localizer — all launch a PsychoPy task)
        is_feedback_transfer = step_id.startswith("transfer_") or step_id.startswith("feedback_")
        is_localizer_run = step_id.startswith("rest_run_") or (step_id.startswith("localizer_") and "_run_" in step_id)

        if is_feedback_transfer or is_localizer_run:
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

    def _on_pipeline_complete(self, success: bool, result_message: str | None):
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

            stopped_by_user = not success and result_message and "stopped by user" in result_message

            if success:
                self._add_log(f"[SUCCESS] {result_message}")
            elif stopped_by_user:
                self._add_log(f"[INFO] {result_message}")
            else:
                self._add_log(f"[ERROR] {result_message}")

            # Always detect step status after completion (handles both success and interruption)
            self._detect_completed_steps()

            # Update UI lock state (re-enables navigation controls)
            self._update_ui_lock_state()

            if (
                not success
                and not stopped_by_user
                and completed_step_id is not None
                and self.right_panel_container is not None
            ):
                # Show interactive error recovery panel only for actual failures
                self._last_failed_step = completed_step_id
                error = RuntimeError(result_message or "Unknown error")
                self.right_panel_container.content = self._build_error_recovery_panel(error, completed_step_id)
                self.right_panel_container.update()
            elif self.right_panel_container is not None:
                # Restore empty preview panel on success or user-initiated stop
                self.right_panel_container.content = self._build_preview_panel()
                self.right_panel_container.update()

            # Show completion snackbar (success only; errors are shown in the recovery panel)
            if success:
                _snack = ft.SnackBar(
                    ft.Text(result_message if result_message else "Step completed"),
                    bgcolor=ft.Colors.GREEN,
                )
                _snack.open = True
                self.page.overlay.append(_snack)
            self.page.update()

            # Auto-show QC approval dialog after coregistration steps
            if success and completed_step_id == "coregistration":
                self._show_qc_approval_dialog("initial")
            elif success and completed_step_id == "coregistration_realtime":
                self._show_qc_approval_dialog("realtime")

            # Auto-show feedback summary dialog after feedback/transfer steps
            if completed_step_id is not None:
                is_feedback_transfer = completed_step_id.startswith("transfer_") or completed_step_id.startswith(
                    "feedback_"
                )
                if is_feedback_transfer:
                    self._show_feedback_summary(completed_step_id)

            # If this completion was a user-initiated stop with an in-flight
            # attempt on disk, offer Keep vs Discard now that the worker has
            # joined (so final flushes are done).
            if stopped_by_user and self._pending_discard_attempt is not None:
                attempt = self._pending_discard_attempt
                self._pending_discard_attempt = None
                self._show_discard_dialog(attempt)

        # Schedule the UI update on the main thread
        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Failed to schedule UI update: {e}")

    def _show_discard_dialog(self, attempt: dict[str, Any]) -> None:
        """Prompt the operator to keep or discard a just-stopped attempt's files.

        Default is Keep (no-op). Discard runs ``discard_attempt`` and logs the
        count; failures surface as an error dialog.
        """
        kind = attempt["kind"]
        subject = attempt["subject"]
        task = attempt["task"]
        attempt_num = attempt["attempt"]
        run = attempt["run"]

        def on_keep(_e):
            self.page.pop_dialog()
            self._add_log(f"[INFO] Kept partial files for {kind} run-{run:02d} attempt-{attempt_num:02d}")

        def on_discard(_e):
            self.page.pop_dialog()
            # Lazy import: pineuro.session is a heavy module per the GUI lazy-import rule.
            from pineuro.session import discard_attempt

            base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
            try:
                removed = discard_attempt(
                    base_dir,
                    subject=subject,
                    session=self.session_id,
                    task=task,
                    attempt=attempt_num,
                    run=run,
                    kind=kind,
                )
            except OSError as exc:
                self._add_log(f"[ERROR] Discard failed: {exc}")
                self.page.show_dialog(
                    ft.AlertDialog(
                        title=ft.Text("Discard Failed"),
                        content=ft.Text(str(exc)),
                        actions=[ft.TextButton("OK", on_click=lambda _e: self.page.pop_dialog())],
                    )
                )
                return
            self._add_log(
                f"[INFO] Discarded {len(removed)} artifact(s) for {kind} run-{run:02d} attempt-{attempt_num:02d}"
            )

        body = ft.Text(
            f"{kind.capitalize()} stopped at run-{run:02d} attempt-{attempt_num:02d}.\n\n"
            f"Keep the partial files (next attempt will be attempt-{attempt_num + 1:02d}), or "
            f"discard them (next attempt reuses attempt-{attempt_num:02d})?",
            size=14,
        )
        dialog = ft.AlertDialog(
            title=ft.Text("Discard this attempt?"),
            content=ft.Container(content=body, width=420),
            actions=[
                ft.TextButton(
                    "Discard",
                    icon=ft.Icons.DELETE_FOREVER,
                    style=ft.ButtonStyle(color=ft.Colors.RED_700),
                    on_click=on_discard,
                ),
                ft.FilledButton(
                    "Keep",
                    icon=ft.Icons.CHECK,
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREEN_700,
                    on_click=on_keep,
                ),
            ],
        )
        self.page.show_dialog(dialog)

    def _build_error_recovery_panel(self, error: Exception, step_id: str) -> ft.Container:
        """Build an interactive error recovery panel shown in the right panel area.

        Parameters
        ----------
        error : Exception
            The exception that caused the failure.
        step_id : str
            The pipeline step ID that failed (used by Retry).

        Returns
        -------
        ft.Container
            A red-tinted card with Retry / Continue Anyway / Scroll to Log buttons.
        """
        error_msg = truncate_text(str(error), 300)
        error_type = type(error).__name__

        def on_retry(e):
            if self.right_panel_container is not None:
                self.right_panel_container.content = self._build_preview_panel()
                self.right_panel_container.update()
            self._start_step(step_id)

        def on_continue(e):
            self._continue_after_error(step_id)

        def on_scroll_log(e):
            if self.log_listview is not None:
                with contextlib.suppress(Exception):
                    self.log_listview.scroll_to(offset=-1, duration=200)
                    self.page.update()

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.RED_700, size=24),
                            ft.Text(
                                f"Step failed: {error_type}",
                                size=15,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.RED_900,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Container(
                        content=ft.Text(
                            error_msg,
                            size=12,
                            color=ft.Colors.RED_800,
                            selectable=True,
                        ),
                        bgcolor=ft.Colors.RED_50,
                        border_radius=4,
                        padding=10,
                    ),
                    ft.Row(
                        [
                            ft.FilledButton(
                                "Retry Step",
                                icon=ft.Icons.REFRESH,
                                on_click=on_retry,
                            ),
                            ft.OutlinedButton(
                                "Continue Anyway",
                                icon=ft.Icons.SKIP_NEXT,
                                on_click=on_continue,
                            ),
                            ft.TextButton(
                                "Scroll to Log",
                                icon=ft.Icons.TERMINAL,
                                on_click=on_scroll_log,
                            ),
                        ],
                        spacing=8,
                    ),
                ],
                spacing=12,
            ),
            padding=20,
            bgcolor=ft.Colors.RED_50,
            border=ft.border.all(1, ft.Colors.RED_200),
            border_radius=8,
            margin=10,
        )

    def _continue_after_error(self, step_id: str):
        """Mark a failed step as interrupted and restore the preview panel.

        This allows the user to skip past a failed step and continue with the
        next one. The step is marked INTERRUPTED (not COMPLETE), so it will
        display a warning icon but not block later steps.

        Parameters
        ----------
        step_id : str
            The pipeline step ID to mark as interrupted.
        """
        self.step_statuses[step_id] = StepStatus.INTERRUPTED
        self._is_running = False
        self._update_ui_lock_state()

        if self.right_panel_container is not None:
            self.right_panel_container.content = self._build_preview_panel()
            self.right_panel_container.update()

        self._refresh_step_list()
        self._rebuild_header()

    def _on_pipeline_error(self, error: Exception):
        """Handle pipeline step error.

        Args:
            error: Exception that occurred
        """
        failed_step = self._current_step_id

        # Stop any feedback monitoring thread so it does not outlive a failed run.
        self._stop_feedback_monitoring()

        # This callback runs in the background thread
        async def update_ui():
            self._add_log(f"[ERROR] {str(error)}")
            self._is_running = False
            self._update_ui_lock_state()
            self._last_failed_step = failed_step

            if failed_step is not None and self.right_panel_container is not None:
                self.right_panel_container.content = self._build_error_recovery_panel(error, failed_step)
                self.right_panel_container.update()
            else:
                # Fallback: snackbar only (no step context)
                _snack = ft.SnackBar(
                    ft.Text(f"Error: {str(error)}"),
                    bgcolor=ft.Colors.RED,
                )
                _snack.open = True
                self.page.overlay.append(_snack)
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

    def _teardown_file_logging(self):
        """Remove and close the session log file handler if one is active."""
        if self._log_file_handler is not None:
            logging.getLogger("pineuro").removeHandler(self._log_file_handler)
            try:
                self._log_file_handler.close()
            except Exception:
                logger.debug("Failed to close session log file handler", exc_info=True)
            self._log_file_handler = None

    def _setup_file_logging(self):
        """Set up file-based logging to capture full debug output for export."""
        # Drop any handler from a previous session so handlers do not
        # accumulate on the shared "pineuro" logger across sessions.
        self._teardown_file_logging()
        base_dir = Path(self.study_config.session_config["paths"]["base_dir"])
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = logs_dir / f"session_{timestamp}.log"

        # Create file handler for the root logger to capture all pineuro logs
        self._log_file_handler = logging.FileHandler(self._log_file_path)
        self._log_file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self._log_file_handler.setFormatter(formatter)

        # Add handler to pineuro root logger to capture all module logs
        pineuro_logger = logging.getLogger("pineuro")
        pineuro_logger.addHandler(self._log_file_handler)

    def _clear_log(self, e=None):
        """Clear the log console display."""
        self.log_messages.clear()
        if self.log_listview is not None:
            self.log_listview.controls.clear()
            self.log_listview.update()

    def _show_monitoring_panel(self, step_id: str):
        """Show monitoring panel for a running step.

        Args:
            step_id: Step identifier
        """
        self._current_step_id = step_id
        config = self.study_config.session_config

        # Resolve the active FeedbackRunConfig for this step (if applicable)
        frc_idx = self._get_feedback_step_frc_index(step_id)
        feedback_runs = config.get("feedback_runs", [])
        self._active_frc: dict | None = (
            feedback_runs[frc_idx] if frc_idx is not None and frc_idx < len(feedback_runs) else None
        )

        # Check if this is a feedback/transfer step
        is_feedback_step = step_id.startswith("transfer_") or step_id.startswith("feedback_")

        if is_feedback_step:
            # Create activation plot for feedback/transfer runs
            sham_config = config.get("sham", {})
            blind_mode = sham_config.get("blind_mode", False)

            # Determine which masks to show based on blind mode and condition
            if blind_mode and sham_config.get("enabled", False):
                from pineuro.sham import get_condition_for_participant, load_randomization

                # In blind mode, only show the masks that control the task
                # Look up condition from randomization file
                condition = "real"  # Default
                randomization_file = sham_config.get("randomization_file", "")
                if randomization_file:
                    randomization_path = Path(randomization_file)
                    if randomization_path.exists():
                        try:
                            randomization = load_randomization(randomization_path)
                            condition_info = get_condition_for_participant(
                                randomization, self.participant_id, self.session_id
                            )
                            condition = condition_info.get("condition", "real")
                        except Exception as e:
                            logger.warning(f"Could not load randomization for blind mode: {e}")

                # Resolve masks from FRC if available, else fall back to global config
                frc_idx = self._get_feedback_step_frc_index(step_id)
                feedback_runs = config.get("feedback_runs", [])
                if frc_idx is not None and frc_idx < len(feedback_runs):
                    real_mask_names = self._resolve_frc_mask_names(feedback_runs[frc_idx])
                else:
                    real_mask_names = self._resolve_mask_names(config)

                if condition == "sham" and sham_config.get("delivery_method") == "alternative_masks":
                    # Sham condition: show sham masks
                    sham_mask_mode = sham_config.get("sham_mask_mode", "two_networks")
                    if sham_mask_mode == "hemisphere_split":
                        hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                        target_hemi = sham_config.get("target_hemisphere", "right")
                        non_target_hemi = "left" if target_hemi == "right" else "right"
                        # Show both hemisphere masks for sham
                        mask_names = [
                            f"{hemisphere_network}_{target_hemi}",
                            f"{hemisphere_network}_{non_target_hemi}",
                        ]
                    else:
                        # Two networks mode
                        alternative_targets = sham_config.get("alternative_mask_targets", [])
                        mask_names = alternative_targets[:2] if len(alternative_targets) >= 2 else real_mask_names
                else:
                    # Real condition: show real masks
                    mask_names = real_mask_names[:2]

                # In blind mode, always show as Target/Non-Target
                display_names = ["Target", "Non-Target"]
            else:
                # Non-blind mode: show all masks with real names
                frc_idx = self._get_feedback_step_frc_index(step_id)
                feedback_runs = config.get("feedback_runs", [])
                if frc_idx is not None and frc_idx < len(feedback_runs):
                    mask_names = self._resolve_frc_mask_names(feedback_runs[frc_idx])
                else:
                    mask_names = self._resolve_mask_names(config)
                if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
                    sham_mask_mode = sham_config.get("sham_mask_mode", "two_networks")
                    if sham_mask_mode == "hemisphere_split":
                        hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                        target_hemi = sham_config.get("target_hemisphere", "right")
                        non_target_hemi = "left" if target_hemi == "right" else "right"
                        for hemi in [target_hemi, non_target_hemi]:
                            hemi_name = f"{hemisphere_network}_{hemi}"
                            if hemi_name not in mask_names:
                                mask_names.append(hemi_name)
                    else:
                        alternative_targets = sham_config.get("alternative_mask_targets", [])
                        for alt_name in alternative_targets:
                            if alt_name and alt_name not in mask_names:
                                mask_names.append(alt_name)
                display_names = None

            n_volumes = self._fb("n_volumes", 150)

            # Store display names mapping for live metrics (blind mode)
            if display_names:
                self._feedback_display_names: dict[str, str] | None = {
                    mask_names[i]: display_names[i] for i in range(len(mask_names))
                }
            else:
                self._feedback_display_names = None

            # Resolve declarative live metrics for this task (declared on TaskMetadata).
            # The participant_workflow live monitoring panel renders one labeled
            # widget per metric and hides the row entirely if the task declared none.
            from pineuro.tasks.task_registry import get_task_metadata

            _live_metrics_task_type = (
                self._active_frc.get("task", "ball")
                if self._active_frc is not None
                else (config["feedback_runs"][0].get("task", "ball") if config.get("feedback_runs") else "ball")
            )
            try:
                self._live_metrics = list(get_task_metadata(_live_metrics_task_type).live_metrics)
            except (KeyError, ValueError):
                self._live_metrics = []

            # Determine what to plot: connectivity signal or individual activations
            feedback_mode = self._fb("feedback_mode", "activation")
            if feedback_mode == "connectivity" and len(mask_names) >= 2:
                if blind_mode and sham_config.get("enabled"):
                    # Blind mode: show only the condition's correlation
                    # mask_names already resolved to condition-specific masks above
                    conn_key = f"corr_{mask_names[0]}_{mask_names[1]}"
                    plot_mask_names = [conn_key]
                    plot_display_names = ["Connectivity"]
                else:
                    # Non-blind: show the real pair's correlation
                    real_conn_key = f"corr_{mask_names[0]}_{mask_names[1]}"
                    plot_mask_names = [real_conn_key]

                    # If alternative_masks sham: also show the sham pair's correlation
                    if sham_config.get("enabled") and sham_config.get("delivery_method") == "alternative_masks":
                        sham_mask_mode = sham_config.get("sham_mask_mode", "two_networks")
                        if sham_mask_mode == "hemisphere_split":
                            hemi_net = sham_config.get("hemisphere_network", "SomMotA")
                            t_hemi = sham_config.get("target_hemisphere", "right")
                            nt_hemi = "left" if t_hemi == "right" else "right"
                            sham_conn_key = f"corr_{hemi_net}_{t_hemi}_{hemi_net}_{nt_hemi}"
                        else:
                            alt_targets = sham_config.get("alternative_mask_targets", [])
                            if len(alt_targets) >= 2:
                                sham_conn_key = f"corr_{alt_targets[0]}_{alt_targets[1]}"
                            else:
                                sham_conn_key = None
                        if sham_conn_key and sham_conn_key not in plot_mask_names:
                            plot_mask_names.append(sham_conn_key)

                    plot_display_names = None  # show column names as-is

                self._plot_label = "Connectivity"
                plot_y_min, plot_y_max = -1.0, 1.0
            else:
                plot_mask_names = mask_names
                plot_display_names = display_names
                plot_y_min, plot_y_max = -2.0, 2.0
                self._plot_label = "Network Activation"

            # Store plot_mask_names for _start_feedback_monitoring (csv column extraction)
            self._feedback_mask_names = plot_mask_names

            self.activation_plot = ActivationPlot(
                mask_names=plot_mask_names,
                n_volumes=n_volumes,
                height=200,
                y_min=plot_y_min,
                y_max=plot_y_max,
                display_names=plot_display_names,
            )

            # Create FD plot for motion monitoring
            fd_threshold = self._fb("fd_threshold", 0.5)
            self.fd_plot = FDPlot(
                n_volumes=n_volumes,
                height=150,
                threshold=fd_threshold,
            )

            # Create DVARS plot for signal quality monitoring
            dvars_threshold = self._fb("dvars_threshold", 1.5)
            self.dvars_plot = DVARSPlot(
                n_volumes=n_volumes,
                height=150,
                threshold=dvars_threshold,
            )

            # Create tSNR plot for signal quality monitoring
            tsnr_threshold = self._fb("tsnr_threshold", 20)
            self.tsnr_plot = TSNRPlot(
                n_volumes=n_volumes,
                height=150,
                threshold=tsnr_threshold,
            )

            # Create task mirror for visual feedback display
            self.task_mirror = TaskMirror(width=None, height=None)

            # Create progress and status UI elements (compact for status strip)
            self._progress_bar = ft.ProgressBar(value=0, width=300)
            self._progress_text = ft.Text("0 / 0 volumes", size=12)
            self._status_text = ft.Text("Waiting...", size=12, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_600)
            self._metric_texts: dict[str, ft.Text] = {m.key: ft.Text("", size=12) for m in self._live_metrics}
            self._metrics_row = ft.Row(
                list(self._metric_texts.values()),
                spacing=16,
                visible=bool(self._live_metrics),
            )

            # Build monitoring panel:
            # Row 1: Activation/Connectivity plot + Task mirror (equal halves)
            # Row 2: FD, DVARS, tSNR plots side by side (compact, secondary)
            monitoring_panel = ft.Container(
                content=ft.Column(
                    [
                        # Header row: title + stop button
                        ft.Row(
                            [
                                ft.Text(
                                    self._get_step_label(step_id),
                                    size=18,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.FilledButton(
                                    "Stop",
                                    icon=ft.Icons.STOP,
                                    on_click=lambda e: self._on_stop_clicked(),
                                    bgcolor=ft.Colors.RED,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(),
                        # Status strip: status, progress, task-declared live metrics
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Row(
                                        [
                                            self._status_text,
                                            ft.Row(
                                                [self._progress_bar, self._progress_text],
                                                spacing=8,
                                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                            ),
                                        ],
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                    ),
                                    self._metrics_row,
                                ],
                                spacing=4,
                            ),
                            bgcolor=ft.Colors.GREY_50,
                            border_radius=8,
                            padding=ft.padding.symmetric(horizontal=12, vertical=6),
                        ),
                        # Row 1: Activation plot + Task mirror (equal halves)
                        ft.Row(
                            [
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text(self._plot_label, size=14, weight=ft.FontWeight.W_500),
                                            self.activation_plot.build(),
                                        ],
                                        spacing=5,
                                    ),
                                    expand=1,
                                    height=300,
                                ),
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text("Task Display", size=14, weight=ft.FontWeight.W_500),
                                            self.task_mirror.build(),
                                        ],
                                        spacing=5,
                                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                        expand=True,
                                    ),
                                    expand=1,
                                    height=300,
                                ),
                            ],
                            spacing=20,
                            vertical_alignment=ft.CrossAxisAlignment.START,
                        ),
                        # Row 2: QC plots side by side (secondary)
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Container(
                                        content=ft.Column(
                                            [
                                                ft.Text(
                                                    "Framewise Displacement",
                                                    size=11,
                                                    weight=ft.FontWeight.W_500,
                                                ),
                                                self.fd_plot.build(),
                                            ],
                                            spacing=3,
                                        ),
                                        expand=1,
                                        height=200,
                                    ),
                                    ft.Container(
                                        content=ft.Column(
                                            [
                                                ft.Text("DVARS", size=11, weight=ft.FontWeight.W_500),
                                                self.dvars_plot.build(),
                                            ],
                                            spacing=3,
                                        ),
                                        expand=1,
                                        height=200,
                                    ),
                                    ft.Container(
                                        content=ft.Column(
                                            [
                                                ft.Text("tSNR", size=11, weight=ft.FontWeight.W_500),
                                                self.tsnr_plot.build(),
                                            ],
                                            spacing=3,
                                        ),
                                        expand=1,
                                        height=200,
                                    ),
                                ],
                                spacing=12,
                            ),
                            padding=ft.padding.symmetric(horizontal=20),
                        ),
                    ],
                    spacing=8,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=20,
                expand=True,
            )

            # Start monitoring the feedback log file
            self._start_feedback_monitoring(step_id)

        else:
            # Check if this is a localizer run step (single or multi-localizer)
            is_rest_run_step = step_id.startswith("rest_run_")
            is_multi_loc_run = step_id.startswith("localizer_") and "_run_" in step_id
            is_localizer_run_step = is_rest_run_step or is_multi_loc_run
            # Clear stale collection_progress from previous steps (e.g. reference)
            # so the callback falls through to _progress_bar for localizer runs.
            self.collection_progress = None

            if is_localizer_run_step:
                n_volumes = self._get_collection_n_volumes(step_id)
                monitoring_panel = self._build_localizer_monitoring_panel(step_id, n_volumes)

            elif step_id in ["reference_initial", "reference_realtime"]:
                # Simple collection progress for short reference scans
                n_volumes = self._get_collection_n_volumes(step_id)
                tr = config["feedback_runs"][0].get("tr") if config.get("feedback_runs") else None

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
        # Handle multi-localizer step IDs: localizer_{idx}_run_{num},
        # localizer_{idx}_preprocessing, localizer_{idx}_extraction
        if step_id.startswith("localizer_") and "_run_" in step_id:
            parts = step_id.split("_")
            loc_idx = int(parts[1])
            run_num = parts[-1]
            config = self.study_config.session_config
            localizers = config.get("localizers", [])
            if loc_idx < len(localizers):
                loc = localizers[loc_idx]
                loc_type = loc.get("type", "rest")
                if loc_type == "rest":
                    return f"Collect Rest Run {run_num} (Loc {loc_idx + 1})"
                else:
                    try:
                        from pineuro.tasks import get_task_metadata

                        task_meta = get_task_metadata(loc_type)
                        task_name = task_meta.display_name.replace(" Localizer", "").replace(" Task", "")
                    except (ValueError, ImportError):
                        task_name = loc_type.title()
                    return f"Collect {task_name} Run {run_num} (Loc {loc_idx + 1})"
            return f"Localizer {loc_idx + 1} Run {run_num}"
        if step_id.startswith("localizer_") and step_id.endswith("_preprocessing"):
            loc_idx = int(step_id.split("_")[1])
            config = self.study_config.session_config
            localizers = config.get("localizers", [])
            loc_type = localizers[loc_idx].get("type", "rest") if loc_idx < len(localizers) else "rest"
            label = "Rest" if loc_type == "rest" else loc_type.upper()
            return f"Preprocess {label} (Loc {loc_idx + 1})"
        if step_id.startswith("localizer_") and step_id.endswith("_extraction"):
            loc_idx = int(step_id.split("_")[1])
            config = self.study_config.session_config
            localizers = config.get("localizers", [])
            loc_type = localizers[loc_idx].get("type", "rest") if loc_idx < len(localizers) else "rest"
            if loc_type == "manual":
                return f"Import Masks (Loc {loc_idx + 1})"
            label = "Rest" if loc_type == "rest" else loc_type.upper()
            return f"Extract {label} Masks (Loc {loc_idx + 1})"
        if step_id.startswith("localizer_") and step_id.endswith("_approval"):
            loc_idx = int(step_id.split("_")[1])
            config = self.study_config.session_config
            localizers = config.get("localizers", [])
            loc_type = localizers[loc_idx].get("type", "rest") if loc_idx < len(localizers) else "rest"
            label = "Manual" if loc_type == "manual" else ("Rest" if loc_type == "rest" else loc_type.upper())
            return f"Review & Approve {label} Masks (Loc {loc_idx + 1})"

        # Handle dynamic rest_run_* steps (single-localizer / legacy)
        if step_id.startswith("rest_run_"):
            run_num = step_id.split("_")[-1]
            # Dynamic label based on localizer type
            config = self.study_config.session_config
            loc_type = _loc(config).get("type", "rest")
            if loc_type == "rest":
                return f"Collect Rest Run {run_num}"
            try:
                from pineuro.tasks import get_task_metadata

                task_meta = get_task_metadata(loc_type)
                task_name = task_meta.display_name.replace(" Localizer", "").replace(" Task", "")
            except (ValueError, ImportError):
                task_name = loc_type.title()
            return f"Collect {task_name} Run {run_num}"

        labels = {
            "reference_initial": "Collect Initial Reference",
            "preprocessing": "Run Preprocessing",
            "mask_extraction": "Extract Masks",
            "mni_mask_extraction": "Extract MNI Atlas Masks",
            "coregistration": "Coregister to Reference",
            "reference_realtime": "Collect Real-time Reference",
            "coregistration_realtime": "Coregister to Real-time Reference",
            "transfer_1": "Transfer Run 1",
            "transfer_2": "Transfer Run 2",
        }
        # Handle dynamic feedback_N labels
        if step_id.startswith("feedback_"):
            fb_run_num = int(step_id.split("_")[-1])
            return f"Feedback Run {fb_run_num}"
        return labels.get(step_id, step_id)

    def _get_collection_n_volumes(self, step_id: str) -> int:
        """Get expected number of volumes for a collection step.

        Args:
            step_id: Step identifier

        Returns:
            Expected number of volumes
        """
        config = self.study_config.session_config

        if step_id == "reference_initial":
            return int(config.get("reference", {}).get("initial_n_volumes", 2))

        elif step_id == "reference_realtime":
            return int(config.get("reference", {}).get("realtime_n_volumes", 2))

        elif step_id.startswith("localizer_") and "_run_" in step_id:
            # Multi-localizer: get n_volumes from the localizer config
            loc_idx = int(step_id.split("_")[1])
            localizers = config.get("localizers", [])
            if loc_idx < len(localizers):
                return int(localizers[loc_idx].get("n_volumes", 250))
            return 250

        elif step_id.startswith("rest_run_"):
            # Single localizer
            n_volumes = config["localizers"][0].get("n_volumes") if config.get("localizers") else None
            if n_volumes is not None:
                return int(n_volumes)
            run_num = int(step_id.split("_")[-1])
            run_idx = run_num - 1
            rest_volumes = (
                [config["localizers"][0].get("n_volumes", 200)] * config["localizers"][0].get("n_runs", 1)
                if config.get("localizers")
                else [200]
            )
            if isinstance(rest_volumes, list) and run_idx < len(rest_volumes):
                return int(rest_volumes[run_idx])
            return int(rest_volumes[0]) if isinstance(rest_volumes, list) else int(rest_volumes)

        return 0

    def _build_localizer_monitoring_panel(self, step_id: str, n_volumes: int) -> ft.Container:
        """Build the right-hand monitoring panel for a localizer run.

        Used for both rest and task localizers. Creates the slice/task-mirror/QC/progress
        widget attributes that `_on_collection_progress` and `_on_collection_qc_data`
        update during the run, then returns the assembled container.
        """
        self._slice_image = ft.Image(
            src="",
            width=480,
            height=270,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
        )
        self._slice_placeholder = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.IMAGE_OUTLINED, size=48, color=ft.Colors.GREY_400),
                    ft.Text("Waiting for first volume...", size=12, color=ft.Colors.GREY_500),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            width=480,
            height=270,
            alignment=ft.Alignment(0, 0),
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=5,
            bgcolor=ft.Colors.BLACK,
        )

        self.task_mirror = TaskMirror(width=480, height=270)

        fd_threshold = self._fb("fd_threshold", 0.5)
        self.fd_plot = FDPlot(n_volumes=n_volumes, height=150, threshold=fd_threshold)
        dvars_threshold = self._fb("dvars_threshold", 1.5)
        self.dvars_plot = DVARSPlot(n_volumes=n_volumes, height=150, threshold=dvars_threshold)
        tsnr_threshold = self._fb("tsnr_threshold", 20)
        self.tsnr_plot = TSNRPlot(n_volumes=n_volumes, height=150, threshold=tsnr_threshold)

        self._progress_bar = ft.ProgressBar(value=0, width=300)
        self._progress_text = ft.Text("0 / 0 volumes", size=12)
        self._status_text = ft.Text("Waiting...", size=12, weight=ft.FontWeight.W_500, color=ft.Colors.GREY_600)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(
                                self._get_step_label(step_id),
                                size=18,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.FilledButton(
                                "Stop",
                                icon=ft.Icons.STOP,
                                on_click=lambda e: self._on_stop_clicked(),
                                bgcolor=ft.Colors.RED,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Divider(),
                    ft.Container(
                        content=ft.Row(
                            [
                                self._status_text,
                                ft.Row(
                                    [self._progress_bar, self._progress_text],
                                    spacing=8,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        bgcolor=ft.Colors.GREY_50,
                        border_radius=8,
                        padding=ft.padding.symmetric(horizontal=12, vertical=6),
                    ),
                    # Row 1: Brain slice + Task mirror
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Brain Slice", size=14, weight=ft.FontWeight.W_500),
                                        ft.Stack([self._slice_placeholder, self._slice_image]),
                                    ],
                                    spacing=5,
                                ),
                                expand=True,
                                height=300,
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Task Display", size=14, weight=ft.FontWeight.W_500),
                                        self.task_mirror.build(),
                                    ],
                                    spacing=5,
                                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                width=500,
                                height=300,
                            ),
                        ],
                        spacing=20,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                    # Row 2: QC plots side by side
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text("Framewise Displacement", size=11, weight=ft.FontWeight.W_500),
                                            self.fd_plot.build(),
                                        ],
                                        spacing=3,
                                    ),
                                    expand=1,
                                    height=200,
                                ),
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text("DVARS", size=11, weight=ft.FontWeight.W_500),
                                            self.dvars_plot.build(),
                                        ],
                                        spacing=3,
                                    ),
                                    expand=1,
                                    height=200,
                                ),
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Text("tSNR", size=11, weight=ft.FontWeight.W_500),
                                            self.tsnr_plot.build(),
                                        ],
                                        spacing=3,
                                    ),
                                    expand=1,
                                    height=200,
                                ),
                            ],
                            spacing=12,
                        ),
                        padding=ft.padding.symmetric(horizontal=20),
                    ),
                ],
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
            expand=True,
        )

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
        # Render DICOM slice in background thread (heavy I/O) before scheduling UI update
        # Skip when QC monitor is active — it provides properly unpacked slices
        # via _on_collection_qc_data, avoiding flicker between mosaic and single slice.
        slice_b64 = None
        has_qc_plots = hasattr(self, "fd_plot") and self.fd_plot is not None
        if dicom_path and hasattr(self, "_slice_image") and self._slice_image is not None and not has_qc_plots:
            try:
                from ..components.collection_progress import dicom_to_base64_png

                slice_b64 = dicom_to_base64_png(dicom_path)
            except Exception:
                pass

        # Schedule UI update on the main thread
        async def update_ui():
            if hasattr(self, "collection_progress") and self.collection_progress is not None:
                self.collection_progress.update(volumes_collected, elapsed_time, dicom_path)
                self.page.update()
            elif hasattr(self, "_progress_bar") and self._progress_bar is not None:
                # Task-based localizer progress (using separate progress bar)
                self._progress_bar.value = volumes_collected / n_volumes if n_volumes > 0 else 0
                self._progress_text.value = f"{volumes_collected} / {n_volumes} volumes"
                if hasattr(self, "_status_text") and self._status_text is not None:
                    self._status_text.value = f"Collecting DICOMs... ({elapsed_time:.1f}s)"
                # Update brain slice preview
                if slice_b64 and hasattr(self, "_slice_image") and self._slice_image is not None:
                    self._slice_image.src = slice_b64
                    self._slice_image.visible = True
                    if hasattr(self, "_slice_placeholder") and self._slice_placeholder is not None:
                        self._slice_placeholder.visible = False
                self.page.update()

        try:
            self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Failed to schedule collection progress update: {e}")

    def _on_collection_qc_data(
        self,
        volume: int,
        fd: float,
        dvars: float,
        tsnr: float,
        slice_b64: str | None,
    ):
        """Handle real-time QC data from CollectionQCMonitor during collection.

        Args:
            volume: Volume number (1-based)
            fd: Framewise displacement in mm
            dvars: DVARS value
            tsnr: Running tSNR estimate
            slice_b64: Base64 PNG data URI for brain slice (or None)
        """

        async def update_ui():
            try:
                # Update FD plot
                if hasattr(self, "fd_plot") and self.fd_plot and fd == fd:  # skip NaN
                    self.fd_plot.add_data(volume, fd)

                # Update DVARS plot
                if hasattr(self, "dvars_plot") and self.dvars_plot and dvars is not None:
                    self.dvars_plot.add_data(volume, dvars)

                # Update tSNR plot
                if hasattr(self, "tsnr_plot") and self.tsnr_plot and tsnr is not None and tsnr > 0:
                    self.tsnr_plot.add_data(volume, tsnr)

                # Update brain slice image
                if slice_b64 and hasattr(self, "_slice_image") and self._slice_image is not None:
                    self._slice_image.src = slice_b64
                    self._slice_image.visible = True
                    self._slice_image.update()
                    if hasattr(self, "_slice_placeholder") and self._slice_placeholder is not None:
                        self._slice_placeholder.visible = False
                        self._slice_placeholder.update()

            except Exception as e:
                logger.error(f"Error updating collection QC UI: {e}")

        with contextlib.suppress(Exception):
            self.page.run_task(update_ui)

    def _start_feedback_monitoring(self, step_id: str):
        """Start monitoring feedback log file for real-time updates.

        Args:
            step_id: Step identifier (transfer_1, feedback_1, etc.)
        """
        config = self.study_config.session_config
        base_dir = Path(config["paths"]["base_dir"])

        # Use the same mask_names that were set when building the monitoring panel
        if hasattr(self, "_feedback_mask_names"):
            mask_names = self._feedback_mask_names
        else:
            frc_idx = self._get_feedback_step_frc_index(step_id)
            feedback_runs = config.get("feedback_runs", [])
            if frc_idx is not None and frc_idx < len(feedback_runs):
                mask_names = self._resolve_frc_mask_names(feedback_runs[frc_idx])
            else:
                mask_names = self._resolve_mask_names(config)

        n_volumes = self._fb("n_volumes", 150)

        # Determine task type and get slot-based run info
        task_type = "transfer" if "transfer" in step_id else "feedback"
        output_dir = base_dir / "feedback" / "02_feedback_output"

        # Get the slot-based run info (run number and attempt number)
        from pineuro.feedback import get_slot_run_info

        run_info = get_slot_run_info(
            output_dir=output_dir,
            subject=self.participant_id,
            session=self.session_id,
            task=task_type,
            step_id=step_id,
        )

        # Construct log file path with attempt entity (format as 2-digit number)
        log_path = (
            output_dir
            / f"{self._bids_prefix}_task-{task_type}_run-{run_info['run']:02d}_attempt-{run_info['attempt']:02d}_feedbacklog.tsv"
        )
        # Store psychopylog path for hit count reading (same pattern but psychopylog.tsv)
        self._psychopylog_path = output_dir / log_path.name.replace("feedbacklog.tsv", "psychopylog.tsv")

        # Create and start feedback monitor
        self.feedback_monitor: FeedbackMonitor | None = FeedbackMonitor(
            log_path=log_path,
            mask_names=mask_names,
            n_volumes=n_volumes,
            on_update=self._on_feedback_update,
            poll_interval=0.5,
        )
        self.feedback_monitor.start()

    def _stop_feedback_monitoring(self):
        """Stop the feedback log monitor."""
        if hasattr(self, "feedback_monitor") and self.feedback_monitor:
            self.feedback_monitor.stop()
            self.feedback_monitor = None

    def _on_task_mirror_update(self, state: dict) -> None:
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
                if hasattr(self, "task_mirror") and self.task_mirror is not None:
                    self.task_mirror.update_state(state_copy)
            except Exception:
                pass

        with contextlib.suppress(Exception):
            self.page.run_task(update_mirror)

    def _on_feedback_update(
        self,
        volume: int,
        mask_activations: dict[str, float],
        motion_params: dict[str, float],
        fd: float | None,
        dvars: float | None = None,
        tsnr: float | None = None,
    ):
        """Handle real-time feedback data update.

        Args:
            volume: Volume number
            mask_activations: Dict mapping mask name to activation value
            motion_params: Dict of motion parameters
            fd: Framewise displacement in mm (None if not available)
            dvars: DVARS value (None if not available)
            tsnr: Running tSNR estimate (None if not available)
        """

        # Schedule UI update on main thread
        async def update_ui():
            try:
                # Update activation plot
                if self.activation_plot and mask_activations:
                    self.activation_plot.add_data(volume, mask_activations)

                # Update FD plot (skip NaN when motion correction is disabled)
                if hasattr(self, "fd_plot") and self.fd_plot and fd is not None and fd == fd:
                    self.fd_plot.add_data(volume, fd)

                # Update DVARS plot
                if hasattr(self, "dvars_plot") and self.dvars_plot and dvars is not None:
                    self.dvars_plot.add_data(volume, dvars)

                # Update tSNR plot
                if hasattr(self, "tsnr_plot") and self.tsnr_plot and tsnr is not None and tsnr > 0:
                    self.tsnr_plot.add_data(volume, tsnr)

                # Update progress
                n_volumes = self._fb("n_volumes", 150)
                progress = min(1.0, volume / n_volumes) if n_volumes > 0 else 0

                if self._progress_bar:
                    self._progress_bar.value = progress
                    self._progress_bar.update()

                if self._progress_text:
                    self._progress_text.value = f"{volume} / {n_volumes} volumes"
                    self._progress_text.update()

                if self._status_text:
                    n_baseline = self._fb("n_baseline", 25)
                    if volume < n_baseline:
                        self._status_text.value = f"Baseline ({volume}/{n_baseline})"
                        self._status_text.color = ft.Colors.ORANGE_600
                    else:
                        step_id = getattr(self, "_current_step_id", "")
                        if "transfer" in step_id:
                            self._status_text.value = "Transfer (no feedback)"
                        else:
                            self._status_text.value = "Task Active"
                        self._status_text.color = ft.Colors.GREEN_600
                    self._status_text.update()

                # Update declarative live metrics (from psychopylog.tsv)
                live_metrics = getattr(self, "_live_metrics", [])
                if live_metrics:
                    from ..components.csv_poller import read_live_metrics

                    snapshot = read_live_metrics(self._psychopylog_path, live_metrics)
                    display_map = getattr(self, "_feedback_display_names", None) or {}
                    for metric in live_metrics:
                        widget = self._metric_texts[metric.key]
                        cells = snapshot.get(metric.key, {})
                        if not cells:
                            widget.value = f"{metric.label}: --"
                        elif len(cells) == 1 and next(iter(cells)) == metric.column_pattern:
                            widget.value = f"{metric.label}: {next(iter(cells.values()))}"
                        else:
                            parts = []
                            for name, val in cells.items():
                                disp = display_map.get(name, name) if metric.remap_blind_names else name
                                parts.append(metric.item_format.format(name=disp, value=val))
                            widget.value = f"{metric.label}: {metric.join_with.join(parts)}"
                        widget.update()

            except Exception as e:
                logger.error(f"Error updating feedback UI: {e}")

        with contextlib.suppress(Exception):
            self.page.run_task(update_ui)

    def _on_stop_clicked(self):
        """Handle stop button click during execution - shows confirmation dialog."""
        # Check if this is a feedback/transfer step (which has post-run questions)
        step_id = self._current_step_id or ""
        is_feedback_step = step_id.startswith("transfer_") or step_id.startswith("feedback_")

        # Checkbox for showing questions (only relevant for feedback steps)
        show_questions_checkbox = ft.Checkbox(
            label="Show post-run questions to participant",
            value=False,  # Default to not showing questions on stop
        )

        def confirm_stop(e):
            self.page.pop_dialog()
            show_questions = bool(show_questions_checkbox.value) if is_feedback_step else False

            # Stop feedback monitoring
            self._stop_feedback_monitoring()

            if self.pipeline_runner is not None:
                # Capture the in-flight attempt so _on_pipeline_complete can
                # prompt Keep vs Discard once the worker thread has joined.
                self._pending_discard_attempt = self.pipeline_runner.stop(show_questions=show_questions)
                if show_questions:
                    self._add_log("[WARNING] Stop requested - showing questions first")
                else:
                    self._add_log("[WARNING] Stop requested by user")

        def cancel_stop(e):
            self.page.pop_dialog()

        # Build dialog content
        content_items: list[ft.Control] = [
            ft.Text(
                "Are you sure you want to stop the current run?",
                size=14,
            ),
        ]

        if is_feedback_step:
            content_items.extend(
                [
                    ft.Divider(),
                    show_questions_checkbox,
                    ft.Text(
                        "If checked, the participant will still see the post-run questionnaire.",
                        size=12,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ]
            )

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

        # Build parameter preview panel for this step
        preview_panel = self._build_step_preview(step_id)

        # Replace the right panel content with the preview
        if self.right_panel_container is not None:
            self.right_panel_container.content = preview_panel
            self.right_panel_container.update()
        else:
            logger.warning("right_panel_container not initialized")

    def _get_next_step(self) -> tuple[str, str] | None:
        """Determine the next step to run in the workflow.

        Considers both pending and interrupted steps. Interrupted steps
        (force-stopped runs) are treated as needing to be re-run.

        Returns:
            Tuple of (step_id, step_label) or None if all complete
        """
        # Build dynamic workflow order based on configuration
        config = self.study_config.session_config
        reference_initial_enabled = config.get("reference", {}).get("initial_enabled", True)
        reference_realtime_enabled = config.get("reference", {}).get("realtime_enabled", True)

        workflow: list[tuple[str, str]] = []

        # In mask reuse mode, skip localizer and mask extraction steps
        if self._is_mni_atlas_mode:
            # MNI atlas mode: reference + MNI extraction + approval + coregistration
            if reference_initial_enabled:
                workflow.append(("reference_initial", "Collect Initial Reference"))
            workflow.append(("mni_mask_extraction", "Extract MNI Atlas Masks"))
            workflow.append(("mask_approval", "Review & Approve Masks"))
            if reference_initial_enabled:
                workflow.append(("coregistration", "Coregister to Reference"))
        elif not self._is_mask_reuse_mode:
            # Add Initial Reference if enabled
            if reference_initial_enabled:
                workflow.append(("reference_initial", "Collect Initial Reference"))

            localizers = config.get("localizers", [])
            if localizers and len(localizers) > 1:
                # Multi-localizer: collect ALL runs first, then per-localizer preprocess→extract→approve.
                # Manual localizers import pre-built masks — no collection or preprocessing.
                for loc_idx, loc in enumerate(localizers):
                    if loc.get("type") == "manual":
                        continue
                    loc_n_runs = loc.get("n_runs", 2)
                    for run_num in range(1, loc_n_runs + 1):
                        step_id = f"localizer_{loc_idx}_run_{run_num}"
                        workflow.append((step_id, self._get_step_label(step_id)))
                for loc_idx, loc in enumerate(localizers):
                    loc_type = loc.get("type", "rest")
                    loc_label = (
                        "Manual" if loc_type == "manual" else (loc_type.upper() if loc_type != "rest" else "Rest")
                    )
                    if loc_type != "manual":
                        workflow.append(
                            (
                                f"localizer_{loc_idx}_preprocessing",
                                self._get_step_label(f"localizer_{loc_idx}_preprocessing"),
                            )
                        )
                    workflow.append(
                        (
                            f"localizer_{loc_idx}_extraction",
                            self._get_step_label(f"localizer_{loc_idx}_extraction"),
                        )
                    )
                    workflow.append(
                        (
                            f"localizer_{loc_idx}_approval",
                            f"Review & Approve Masks (Loc {loc_idx + 1}: {loc_label})",
                        )
                    )
            else:
                # Single localizer
                loc0 = config["localizers"][0] if config.get("localizers") else {}
                if loc0.get("type") == "manual":
                    # Manual import — no collection or preprocessing
                    workflow.append(("mask_extraction", "Import Masks"))
                    workflow.append(("mask_approval", "Review & Approve Masks"))
                else:
                    n_localizer_runs = loc0.get("n_runs", 2) if config.get("localizers") else 2
                    for run_num in range(1, n_localizer_runs + 1):
                        step_id = f"rest_run_{run_num}"
                        workflow.append((step_id, self._get_step_label(step_id)))
                    workflow.extend(
                        [
                            ("preprocessing", "Run Preprocessing"),
                            ("mask_extraction", "Extract Masks"),
                        ]
                    )
                    workflow.append(("mask_approval", "Review & Approve Masks"))

            # Add coregistration only if initial reference is enabled
            if reference_initial_enabled:
                workflow.append(("coregistration", "Coregister to Reference"))

        # Add realtime reference if enabled
        if reference_realtime_enabled:
            workflow.extend(
                [
                    ("reference_realtime", "Collect Real-time Reference"),
                    ("coregistration_realtime", "Coregister to Real-time Reference"),
                ]
            )

        # Add feedback/transfer steps
        workflow.extend(_build_feedback_step_ids(config))

        # Find first pending or interrupted step (interrupted takes priority as it needs re-run)
        for step_id, step_label in workflow:
            status = self.step_statuses.get(step_id)
            if status in (StepStatus.PENDING, StepStatus.INTERRUPTED):
                return (step_id, step_label)

        return None

    def _on_run_next_clicked(self):
        """Handle Run Next button click."""
        # Ignore clicks while running
        if self._is_running:
            return

        next_step = self._get_next_step()

        if next_step is None:
            _snack = ft.SnackBar(ft.Text("Session complete! All steps finished."))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        step_id, step_label = next_step

        # Handle mask_approval step specially - open the review dialog
        if step_id == "mask_approval":
            self._show_mask_review_dialog()
            return

        # Handle per-localizer approval steps
        if step_id.startswith("localizer_") and step_id.endswith("_approval"):
            # Extract localizer index from step_id like "localizer_0_approval"
            loc_idx = int(step_id.split("_")[1])
            self._show_mask_review_dialog(loc_idx=loc_idx)
            return

        # Trigger the step (same as clicking step button)
        self._on_step_clicked(step_id)

    def _on_back(self, e):
        """Handle back button click."""
        # Ignore clicks while running
        if self._is_running:
            return

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
        else:
            step_id, step_label = next_step
            self.run_next_button.text = f"Run Next: {step_label}"
            self.run_next_button.disabled = False
            self.run_next_button.icon = ft.icons.Icons.PLAY_ARROW

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
        if self.step_panel_list is None:
            return

        # Rebuild the expansion panel list with updated statuses
        panels = []
        is_simulator = self.study_config.session_config.get("simulator", {}).get("enabled", True)
        if not is_simulator:
            panels.append(self._build_scanner_connection_section())
        has_localizers = getattr(self, "_has_localizers", True)
        has_feedback = getattr(self, "_has_feedback", True)

        # Only show localizer and mask extraction sections if NOT in mask reuse mode
        if not self._is_mask_reuse_mode and has_localizers:
            panels.append(self._build_localizer_tasks_section())
        if not self._is_mask_reuse_mode and has_feedback:
            panels.append(self._build_mask_extraction_section())
        if has_feedback:
            panels.append(self._build_feedback_section())
        panels.append(self._build_utilities_section())
        self.step_panel_list.controls = panels

        # Force updates in the correct order
        with contextlib.suppress(Exception):
            self.step_panel_list.update()

        # Update the Run Next button by rebuilding the header
        self._rebuild_header()

        # Force a complete page refresh
        with contextlib.suppress(Exception):
            self.page.update()

    def build(self) -> ft.Control:
        """Build the participant workflow UI.

        Returns:
            Flet control for the workflow screen
        """
        # Header
        is_simulator = self.study_config.session_config.get("simulator", {}).get("enabled", True)
        simulator_text = " | Simulator Mode" if is_simulator else ""

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
        panels = []
        # Show scanner connection section when NOT in simulator mode
        is_simulator = self.study_config.session_config.get("simulator", {}).get("enabled", True)
        if not is_simulator:
            panels.append(self._build_scanner_connection_section())
        # Only show sections relevant to the session mode
        has_localizers = getattr(self, "_has_localizers", True)
        has_feedback = getattr(self, "_has_feedback", True)
        if not self._is_mask_reuse_mode and has_localizers:
            panels.append(self._build_localizer_tasks_section())
        if not self._is_mask_reuse_mode and has_feedback:
            panels.append(self._build_mask_extraction_section())
        if has_feedback:
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

        # Resumption banner (shown only when re-opening a partially-completed session)
        resumption_banner = self._build_resumption_banner()

        # Config change banner (shown when study config differs from saved snapshot)
        config_change_banner = self._build_config_change_banner()

        # Simulator banner (shown whenever simulator mode is enabled)
        simulator_banner = self._build_simulator_banner()

        main_controls: list[ft.Control] = [self.header_container]
        if simulator_banner is not None:
            main_controls.append(simulator_banner)
        if config_change_banner is not None:
            main_controls.append(config_change_banner)
        if resumption_banner is not None:
            main_controls.append(resumption_banner)
        main_controls.extend(
            [
                ft.Container(
                    content=content_area,
                    expand=True,
                ),
                log_console,
            ]
        )

        return ft.Column(
            main_controls,
            spacing=0,
            expand=True,
        )
