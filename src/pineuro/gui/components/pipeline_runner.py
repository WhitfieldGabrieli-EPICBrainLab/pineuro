"""Pipeline execution runner for background thread execution.

This module provides thread-safe execution of pineuro pipeline steps,
allowing the GUI to remain responsive during long-running operations.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from pineuro.preprocessing import collect_dicoms, preprocessing
from pineuro.mask_extraction import mask_extraction_rest, coregister_2vol, generate_2vol_qc_overlays
from pineuro.feedback import run_feedback_session, get_next_run_number, get_slot_run_info
from pineuro.tasks.ball import compute_boxcar_task_regressor
from pineuro.tasks.session import SessionState
from pineuro.sham import (
    load_randomization,
    get_condition_for_participant,
    find_feedbacklog_for_replay,
    ReplayActivationProvider,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute pipeline steps in background threads with progress callbacks."""

    def __init__(
        self,
        config: Dict[str, Any],
        participant_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[bool, Optional[str]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_collection_progress: Optional[Callable[[int, int, float, Path], None]] = None,
    ):
        """Initialize pipeline runner.

        Args:
            config: Session configuration dictionary
            participant_id: Participant ID
            on_progress: Callback for progress updates (message)
            on_complete: Callback when step completes (success, result_message)
            on_error: Callback when step fails (exception)
            on_collection_progress: Callback for DICOM collection progress
                (volumes_collected, n_volumes, elapsed_time, dicom_path)
        """
        self.config = config
        self.participant_id = participant_id
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_collection_progress = on_collection_progress

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_coordinator = None  # Track coordinator for feedback tasks
        self._task_mirror_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._selected_rest_runs: Optional[list] = None  # Selected runs for preprocessing

    def set_selected_rest_runs(self, runs: Optional[list]) -> None:
        """Set which rest runs to use for preprocessing.

        Args:
            runs: List of run IDs (1 and/or 2), or None to use all available
        """
        self._selected_rest_runs = runs

    def set_task_mirror_callback(
        self, callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        """Set callback for task visual state mirroring.

        The callback receives visual state dictionaries from tasks
        for rendering in a GUI mirror component.

        Parameters
        ----------
        callback : Callable or None
            Function that receives visual state dict from task.get_mirror_state()
        """
        self._task_mirror_callback = callback

    def run_step(self, step_id: str) -> None:
        """Execute a pipeline step in background thread.

        Args:
            step_id: Step identifier to execute
        """
        # Check if a step is already running
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"Step already running: {self._thread.name}")
            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run_step_impl,
            args=(step_id,),
            name=f"pipeline-{step_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Started background thread for step: {step_id}")

    def stop(self, show_questions: bool = False) -> None:
        """Signal the current step to stop (if running).

        Parameters
        ----------
        show_questions : bool, optional
            If True and a feedback task is running, show post-run questions
            to the participant before stopping. Default is False.
        """
        logger.info(f"Signaling pipeline step to stop (show_questions={show_questions})")
        self._stop_event.set()

        # Stop thread-based steps
        if self._thread is not None and self._thread.is_alive():
            # If a feedback task is running, stop the coordinator
            if self._current_coordinator is not None:
                logger.info("Stopping feedback coordinator")
                self._current_coordinator.stop(show_questions=show_questions)

    def _run_step_impl(self, step_id: str) -> None:
        """Implementation of step execution (runs in background thread).

        Args:
            step_id: Step identifier to execute
        """
        try:
            logger.info(f"Executing step: {step_id}")

            # Execute the appropriate step
            if step_id == "2vol_initial":
                self._run_2vol_initial()
            elif step_id.startswith("rest_run_"):
                # Handle dynamic rest run steps (rest_run_1, rest_run_2, etc.)
                run_num = int(step_id.split("_")[-1])
                self._run_rest_run(run_num)
            elif step_id == "preprocessing":
                self._run_preprocessing()
            elif step_id == "mask_extraction":
                self._run_mask_extraction()
            elif step_id == "coregistration":
                self._run_coregistration(run_id=1)
            elif step_id == "2vol_realtime":
                self._run_2vol_realtime()
            elif step_id == "coregistration_realtime":
                self._run_coregistration(run_id=2)
            elif step_id.startswith("transfer_") or step_id.startswith("feedback_"):
                self._run_feedback_task(step_id)
            else:
                raise ValueError(f"Unknown step ID: {step_id}")

            # Completion callback
            if self.on_complete:
                if self._stop_event.is_set():
                    self.on_complete(False, f"Step {step_id} stopped by user")
                else:
                    self.on_complete(True, f"Step {step_id} completed successfully")

        except Exception as e:
            logger.error(f"Step {step_id} failed: {e}", exc_info=True)
            if self.on_error:
                self.on_error(e)
            if self.on_complete:
                self.on_complete(False, str(e))

    def _emit_progress(self, message: str) -> None:
        """Emit progress update to callback.

        Args:
            message: Progress message
        """
        logger.info(f"Progress: {message}")
        if self.on_progress:
            self.on_progress(message)

    def _run_2vol_initial(self) -> None:
        """Collect initial 2vol for positioning verification."""
        config = self.config

        # Check if 2vol initial is enabled
        if not config["scan"].get("twovol_initial_enabled", True):
            self._emit_progress("2vol initial collection is disabled, skipping...")
            return

        self._emit_progress("Collecting initial 2vol DICOMs...")

        base_dir = Path(config["paths"]["base_dir"])
        output_dir = base_dir / "collection" / "task-2vol_run-01"

        # Get number of volumes (use new field, fall back to legacy)
        n_volumes = config["scan"].get("twovol_initial_n_volumes", config["scan"].get("twovol_n_volumes", 2))

        # Determine simulator vs real-time mode
        simulator_enabled = config["scan"].get("simulator_enabled", True)

        if simulator_enabled:
            # Simulator mode: create temp watch dir and use source DICOMs
            watch_dir = base_dir / "tmp" / "incoming_2vol_initial"
            watch_dir.mkdir(parents=True, exist_ok=True)
            simulator_source_dir = config["scan"]["twovol_initial_dicoms"]
            simulator_speed = config["scan"].get("collection_speed", 10.0)
        else:
            # Real-time mode: watch scanner DICOM directory
            watch_dir = Path(config["scan"]["scanner_dicom_dir"])
            simulator_source_dir = None
            simulator_speed = 1.0

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=n_volumes,
            simulator_source_dir=simulator_source_dir,
            simulator_speed=simulator_speed,
            clear_watch_dir=simulator_enabled,  # Only clear in simulator mode
            clear_output_dir=False,  # Don't clear if re-running
            progress_callback=self.on_collection_progress,
            stop_event=self._stop_event,
        )

        self._emit_progress(f"Collected {n_volumes} volumes to {output_dir}")

    def _run_rest_run(self, run_num: int) -> None:
        """Collect resting-state run.

        Args:
            run_num: Run number (1-based)
        """
        self._emit_progress(f"Collecting rest run {run_num} DICOMs...")

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        output_dir = base_dir / "collection" / f"task-rest_run-{run_num:02d}"

        # Get expected volumes (use new localizer fields, fall back to legacy)
        n_volumes = config["scan"].get("localizer_n_volumes")
        if n_volumes is None:
            rest_volumes = config["scan"].get("rest_n_volumes", [200])
            if isinstance(rest_volumes, list):
                if len(rest_volumes) >= run_num:
                    n_volumes = rest_volumes[run_num - 1]
                else:
                    n_volumes = rest_volumes[0] if rest_volumes else 200
            else:
                n_volumes = rest_volumes

        # Determine simulator vs real-time mode
        simulator_enabled = config["scan"].get("simulator_enabled", True)

        if simulator_enabled:
            # Simulator mode: get source directory and create temp watch dir
            localizer_sources = config["scan"].get("localizer_source_dicoms")
            if localizer_sources is None:
                localizer_sources = config["scan"].get("rest_source_dicoms", [])

            # Get source for this run (0-indexed)
            if isinstance(localizer_sources, list):
                if len(localizer_sources) >= run_num:
                    source_dir = localizer_sources[run_num - 1]
                elif len(localizer_sources) > 0:
                    source_dir = localizer_sources[0]
                else:
                    source_dir = None
            else:
                source_dir = localizer_sources

            watch_dir = base_dir / "tmp" / f"incoming_rest_run{run_num}"
            watch_dir.mkdir(parents=True, exist_ok=True)
            simulator_speed = config["scan"].get("collection_speed", 10.0)
        else:
            # Real-time mode: watch scanner DICOM directory
            source_dir = None
            watch_dir = Path(config["scan"]["scanner_dicom_dir"])
            simulator_speed = 1.0

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=n_volumes,
            simulator_source_dir=source_dir,
            simulator_speed=simulator_speed,
            clear_watch_dir=simulator_enabled,  # Only clear in simulator mode
            clear_output_dir=False,
            progress_callback=self.on_collection_progress,
            stop_event=self._stop_event,
        )

        self._emit_progress(f"Collected {n_volumes} volumes to {output_dir}")

    def _run_preprocessing(self) -> None:
        """Run preprocessing on collected rest runs."""
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"

        # Input DICOM directories
        rest_run_1_dir = base_dir / "collection" / "task-rest_run-01"
        rest_run_2_dir = base_dir / "collection" / "task-rest_run-02"

        # Use selected runs if specified, otherwise use all available
        input_dicoms = []
        if self._selected_rest_runs is not None:
            # Use explicitly selected runs
            if 1 in self._selected_rest_runs and rest_run_1_dir.exists():
                input_dicoms.append(rest_run_1_dir)
            if 2 in self._selected_rest_runs and rest_run_2_dir.exists():
                input_dicoms.append(rest_run_2_dir)

            self._emit_progress(f"Using selected runs: {self._selected_rest_runs}")
        else:
            # Default behavior: use all available runs
            if rest_run_1_dir.exists():
                input_dicoms.append(rest_run_1_dir)
            if rest_run_2_dir.exists():
                input_dicoms.append(rest_run_2_dir)

        if not input_dicoms:
            raise ValueError("No resting-state run directories found for preprocessing")

        # Set up progress callback for preprocessing to show in GUI log console
        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        preproc_file, mask_file, ref_file = preprocessing(
            participant_id=self.participant_id,
            task="rest",
            root_dir=preprocessing_dir,
            input_dicoms=input_dicoms,
            overwrite=True,  # Overwrite since user clicked "Start"
            qc_enabled=True,  # Generate QC report
            progress_callback=progress_callback,
        )

    def _run_mask_extraction(self) -> None:
        """Extract network masks using ICA."""
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"
        mask_extraction_dir = base_dir / "mask_extraction"

        # Input files from preprocessing
        preproc_file = preprocessing_dir / f"{self.participant_id}_task-rest_preproc.nii.gz"
        mask_file = preprocessing_dir / f"{self.participant_id}_task-rest_preproc_mask.nii.gz"
        ref_file = preprocessing_dir / f"{self.participant_id}_task-rest_boldref.nii.gz"

        # Mask extraction parameters
        n_yeo = config["mask"]["n_yeo"]
        yeo_targets = list(config["mask"]["yeo_targets"])  # Copy to avoid modifying original
        n_components = config["mask"].get("n_components", 15)
        smooth_fwhm = config["mask"]["smooth_fwhm"]
        highpass = config["mask"]["highpass"]
        lowpass = config["mask"].get("lowpass")

        # If sham is enabled with alternative masks, include those networks too
        sham_config = config.get("sham", {})
        sham_hemisphere_network = None
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
            if mask_mode == "hemisphere_split":
                # For hemisphere split, we need to include the network to split
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                if hemisphere_network and hemisphere_network not in yeo_targets:
                    yeo_targets.append(hemisphere_network)
                    self._emit_progress(f"Including sham hemisphere network: {hemisphere_network}")
                sham_hemisphere_network = hemisphere_network
            else:
                # For two_networks mode, include both alternative targets
                alternative_targets = sham_config.get("alternative_mask_targets", [])
                for target in alternative_targets:
                    if target and target not in yeo_targets:
                        yeo_targets.append(target)
                        self._emit_progress(f"Including sham alternative network: {target}")

        # Set up progress callback for mask extraction to show in GUI log console
        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        mask_files = mask_extraction_rest(
            participant_id=self.participant_id,
            root_dir=mask_extraction_dir,
            input_file=preproc_file,
            mask_file=mask_file,
            ref_file=ref_file,
            n_yeo=n_yeo,
            yeo_targets=yeo_targets,
            n_components=n_components,
            smooth_fwhm=smooth_fwhm,
            highpass=highpass,
            lowpass=lowpass,
            overwrite=True,
            qc_enabled=True,
            progress_callback=progress_callback,
            sham_hemisphere_network=sham_hemisphere_network,
        )

    def _run_coregistration(self, run_id: int = 1) -> None:
        """Coregister masks to 2vol feedback space.

        Args:
            run_id: Run number (1 for initial, 2 for realtime)
        """
        config = self.config

        # Check if 2vol is enabled for this run
        if run_id == 1:
            if not config["scan"].get("twovol_initial_enabled", True):
                self._emit_progress("2vol initial coregistration is disabled, skipping...")
                return
        else:
            if not config["scan"].get("twovol_realtime_enabled", True):
                self._emit_progress("2vol realtime coregistration is disabled, skipping...")
                return

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"
        mask_extraction_dir = base_dir / "mask_extraction"
        feedback_dir = base_dir / "feedback"

        # Check for mask reuse mode
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        mask_source_visit = visit_config.get("mask_source_visit")

        # Determine actual 2vol run number
        # In mask reuse mode, realtime (run_id=2 from caller) uses 2vol run-01 (since no initial 2vol)
        # In standard mode, run_id directly maps to 2vol run number
        if is_mask_reuse and run_id == 2:
            actual_2vol_run = 1
        else:
            actual_2vol_run = run_id

        run_str = f"run-{actual_2vol_run:02d}"
        self._emit_progress(f"Coregistering network masks to 2vol {run_str}...")

        # For QC overlay generation, we need the mask_extraction_dir where ICA outputs are
        source_mask_extraction_dir = None

        if is_mask_reuse and mask_source_visit:
            # Mask reuse mode: get masks and ref_file from source visit
            # The source visit is at: study_base_dir / participant_id / source_visit_id
            # Current base_dir is: study_base_dir / participant_id / current_visit_id
            # So we go up one level and into the source visit
            study_base_dir = base_dir.parent.parent  # Go up from participant/visit to study
            source_visit_dir = study_base_dir / self.participant_id / mask_source_visit
            source_preprocessing_dir = source_visit_dir / "preprocessing"
            source_mask_extraction_dir = source_visit_dir / "mask_extraction"

            ref_file = source_preprocessing_dir / f"{self.participant_id}_task-rest_boldref.nii.gz"
            masks_dir = source_mask_extraction_dir / "03_masks"

            self._emit_progress(f"Using masks from source visit: {mask_source_visit}")
        else:
            # Standard mode: get masks and ref_file from current visit
            ref_file = preprocessing_dir / f"{self.participant_id}_task-rest_boldref.nii.gz"
            masks_dir = mask_extraction_dir / "03_masks"

        twovol_dicom = base_dir / "collection" / f"task-2vol_{run_str}"

        # Get network mask files (thresholded only)
        network_files = list(masks_dir.glob(f"{self.participant_id}_*_thr.nii.gz"))

        # Filter out combined hemisphere network mask if using hemisphere_split sham mode
        # (we only want left/right split masks, not the combined one)
        sham_config = config.get("sham", {})
        if sham_config.get("enabled", False):
            delivery_method = sham_config.get("delivery_method", "alternative_masks")
            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
            if delivery_method == "alternative_masks" and mask_mode == "hemisphere_split":
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                combined_mask_name = f"{self.participant_id}_{hemisphere_network}_thr.nii.gz"
                network_files = [f for f in network_files if f.name != combined_mask_name]

        self._emit_progress(f"Coregistering {len(network_files)} masks to 2vol {run_str}...")

        # Initial 2vol (run-01) goes to mask_extraction, realtime (run-02+) goes to feedback
        if run_id == 1:
            root_dir = mask_extraction_dir
            output_subdir = "04_2vol"
        else:
            root_dir = feedback_dir
            output_subdir = "01_2vol"

        coreg_masks = coregister_2vol(
            participant_id=self.participant_id,
            root_dir=root_dir,
            input_dicom=twovol_dicom,
            network_files=[str(f) for f in network_files],
            ref_file=str(ref_file),
            run_id=actual_2vol_run,
            overwrite=True,
            output_subdir=output_subdir,
        )

        self._emit_progress(f"Coregistered {len(coreg_masks)} masks to 2vol {run_str}")

        # Generate 2vol QC overlays for visual approval
        self._emit_progress(f"Generating 2vol QC overlays for {run_str}...")
        try:
            # Determine session_dir and twovol_subdir based on run_id
            # For run-01 (initial): session_dir is mask_extraction, subdir is 04_2vol
            # For run-02 (realtime): session_dir is feedback, subdir is 01_2vol
            if run_id == 1:
                session_dir = mask_extraction_dir
                twovol_subdir = "04_2vol"
            else:
                session_dir = feedback_dir
                twovol_subdir = "01_2vol"

            # For QC overlays, use the mask_extraction_dir where ICA outputs are located
            # In mask reuse mode, this is the source visit's mask_extraction_dir
            qc_mask_extraction_dir = source_mask_extraction_dir if source_mask_extraction_dir else mask_extraction_dir

            generate_2vol_qc_overlays(
                participant_id=self.participant_id,
                session_dir=session_dir,
                ref_file=ref_file,
                mask_extraction_dir=qc_mask_extraction_dir,
                network_coreg_files=coreg_masks,
                run_str=str(actual_2vol_run).zfill(2),
                n_yeo=config["mask"]["n_yeo"],
                overwrite=True,
                twovol_subdir=twovol_subdir,
            )
            self._emit_progress(f"QC overlays generated for 2vol {run_str}")
        except Exception as e:
            logger.warning(f"2vol QC overlay generation failed: {e}")
            self._emit_progress(f"Warning: QC overlay generation failed: {e}")

    def _run_2vol_realtime(self) -> None:
        """Collect real-time 2vol before feedback session."""
        config = self.config

        # Check if 2vol realtime is enabled
        if not config["scan"].get("twovol_realtime_enabled", True):
            self._emit_progress("2vol realtime collection is disabled, skipping...")
            return

        self._emit_progress("Collecting real-time 2vol DICOMs...")

        base_dir = Path(config["paths"]["base_dir"])

        # In mask reuse mode, realtime 2vol is run-01 (no initial 2vol)
        # In standard mode, realtime 2vol is run-02 (after initial 2vol)
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        run_id = 1 if is_mask_reuse else 2
        output_dir = base_dir / "collection" / f"task-2vol_run-{run_id:02d}"

        # Get number of volumes (use new field, fall back to legacy)
        n_volumes = config["scan"].get("twovol_realtime_n_volumes", config["scan"].get("twovol_n_volumes", 2))

        # Determine simulator vs real-time mode
        simulator_enabled = config["scan"].get("simulator_enabled", True)

        if simulator_enabled:
            # Simulator mode: create temp watch dir and use source DICOMs
            watch_dir = base_dir / "tmp" / "incoming_2vol_realtime"
            watch_dir.mkdir(parents=True, exist_ok=True)
            simulator_source_dir = config["scan"]["twovol_realtime_dicoms"]
            simulator_speed = config["scan"].get("collection_speed", 10.0)
        else:
            # Real-time mode: watch scanner DICOM directory
            watch_dir = Path(config["scan"]["scanner_dicom_dir"])
            simulator_source_dir = None
            simulator_speed = 1.0

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=n_volumes,
            simulator_source_dir=simulator_source_dir,
            simulator_speed=simulator_speed,
            clear_watch_dir=simulator_enabled,  # Only clear in simulator mode
            clear_output_dir=False,
            progress_callback=self.on_collection_progress,
            stop_event=self._stop_event,
        )

        self._emit_progress(f"Collected {n_volumes} volumes to {output_dir}")

    def _run_feedback_task(self, step_id: str) -> None:
        """Run a feedback or transfer task.

        Args:
            step_id: One of transfer_1, transfer_2, feedback_1, feedback_2, feedback_3, etc.
        """
        # Parse step ID to determine task type
        if step_id.startswith("transfer_"):
            task_type = "transfer"
        elif step_id.startswith("feedback_"):
            task_type = "feedback"
        else:
            raise ValueError(f"Invalid feedback task step_id: {step_id}")

        feedback_mode = (task_type == "feedback")

        # Get output directory for slot-based run info
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"
        feedback_dir = base_dir / "feedback"
        output_dir = feedback_dir / "02_feedback_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use slot-based run numbering with acq entity for retry tracking
        run_info = get_slot_run_info(
            output_dir=output_dir,
            subject=self.participant_id,
            task=task_type,
            step_id=step_id,
        )
        task_run_number = run_info["run"]
        task_acq = run_info["acq"]

        self._emit_progress(f"Starting {task_type} run-{task_run_number:02d} acq-{task_acq:02d}...")

        # Get mask files - use 2vol-coregistered masks if realtime 2vol is enabled,
        # otherwise use boldref space masks from mask extraction
        #
        # For ball task, respect top_network/bottom_network assignment if specified
        task_config = config.get("task", {})
        ball_config = task_config.get("ball", {})
        top_network = ball_config.get("top_network")
        bottom_network = ball_config.get("bottom_network")

        if top_network and bottom_network:
            # Use explicit ball task network assignment for real condition
            real_mask_names = [top_network, bottom_network]
        else:
            # Fall back to config mask_names
            real_mask_names = config["mask"]["mask_names"]

        # Log the ball task network assignment for debugging
        self._emit_progress(f"Ball task networks: {real_mask_names[0]} (top/yellow) vs {real_mask_names[1]} (bottom/blue)")
        logger.info(f"Ball mask assignment: top={real_mask_names[0]}, bottom={real_mask_names[1]}")

        twovol_realtime_enabled = config["scan"].get("twovol_realtime_enabled", True)

        # Determine 2vol run number for mask file pattern
        # In mask reuse mode, 2vol is run-01 (no initial 2vol)
        # In standard mode, 2vol is run-02 (after initial 2vol)
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        realtime_2vol_run = 1 if is_mask_reuse else 2

        if twovol_realtime_enabled:
            # Use 2vol-coregistered masks
            masks_dir = feedback_dir / "01_2vol" / "masks"
            pattern_template = f"*_{{mask_name}}_thr_to_task-2vol_run-{realtime_2vol_run:02d}_*.nii.gz"
            mask_space = f"2vol run-{realtime_2vol_run:02d}"
        else:
            # Use boldref space masks from mask extraction
            masks_dir = mask_extraction_dir / "03_masks"
            pattern_template = "*_{mask_name}_thr.nii.gz"
            mask_space = "subject boldref"

        self._emit_progress(f"Using masks from {mask_space} space")

        # Helper function to find mask file for a given name
        def find_mask_file(mask_name: str) -> Path:
            pattern = pattern_template.format(mask_name=mask_name)
            matches = list(masks_dir.glob(pattern))
            if not matches:
                raise ValueError(
                    f"Could not find mask file for '{mask_name}' in {masks_dir}. "
                    f"Expected pattern: {pattern}"
                )
            if len(matches) > 1:
                logger.warning(f"Multiple mask files found for '{mask_name}', using first: {matches[0]}")
            return matches[0]

        # Find mask files for the real masks
        real_mask_files = [find_mask_file(name) for name in real_mask_names]

        # Collection directory for this run
        collection_dir = base_dir / "collection"

        # Determine simulator vs real-time mode
        simulator_enabled = config["scan"].get("simulator_enabled", True)

        if simulator_enabled:
            # Simulator mode: create temp watch dir and use source DICOMs
            watch_dir = base_dir / "tmp" / f"incoming_{task_type}_acq-{task_acq:02d}_run-{task_run_number:02d}"
            watch_dir.mkdir(parents=True, exist_ok=True)

            # Get per-run simulator source path based on step_id
            # step_id format: transfer_1, transfer_2, feedback_1, feedback_2, etc.
            run_num_in_step = int(step_id.split("_")[-1])  # Extract run number from step_id

            if task_type == "transfer":
                # Transfer runs use transfer_source_dicoms list (index 0 for transfer_1, 1 for transfer_2)
                transfer_sources = config["scan"].get("transfer_source_dicoms", [])
                # Handle legacy single-path format
                if isinstance(transfer_sources, str):
                    transfer_sources = [transfer_sources, transfer_sources]
                if run_num_in_step <= len(transfer_sources) and transfer_sources[run_num_in_step - 1]:
                    simulator_source_dir = transfer_sources[run_num_in_step - 1]
                else:
                    # Fallback to legacy feedback_dicoms or first available path
                    simulator_source_dir = config["scan"].get("feedback_dicoms", "")
                    if not simulator_source_dir and transfer_sources:
                        simulator_source_dir = transfer_sources[0]
            else:
                # Feedback runs use feedback_source_dicoms list (index 0 for feedback_1, etc.)
                feedback_sources = config["scan"].get("feedback_source_dicoms", [])
                # Handle legacy single-path format
                if isinstance(feedback_sources, str):
                    feedback_sources = [feedback_sources, feedback_sources]
                if run_num_in_step <= len(feedback_sources) and feedback_sources[run_num_in_step - 1]:
                    simulator_source_dir = feedback_sources[run_num_in_step - 1]
                else:
                    # Fallback to legacy feedback_dicoms or first available path
                    simulator_source_dir = config["scan"].get("feedback_dicoms", "")
                    if not simulator_source_dir and feedback_sources:
                        simulator_source_dir = feedback_sources[0]

            simulator_speed = config["scan"].get("feedback_simulator_speed", config["scan"].get("collection_speed", 10.0))
        else:
            # Real-time mode: watch scanner DICOM directory
            watch_dir = Path(config["scan"]["scanner_dicom_dir"])
            simulator_source_dir = None
            simulator_speed = 1.0

        self._emit_progress(f"Preparing {task_type} run (feedback={'ON' if feedback_mode else 'OFF'})...")

        # Compute task regressor (HRF-convolved boxcar for ball task)
        n_baseline = config["feedback"]["n_baseline"]
        n_volumes = config["feedback"]["n_volumes"]
        # Use new feedback_tr field, fall back to legacy tr
        tr = config["scan"].get("feedback_tr") or config["scan"].get("tr", 1.2)
        task_regressor = compute_boxcar_task_regressor(n_baseline, n_volumes, tr)

        # Determine sham condition and build complete mask list
        # New approach: compute ALL masks (real + sham alternatives), let BallTask decide
        # which two masks control the ball based on condition
        condition = "real"
        sham_mask_names = []
        sham_mask_files = []
        sham_provider = None  # ReplayActivationProvider for replay mode
        sham_config = config.get("sham", {})

        if sham_config.get("enabled", False):
            randomization_file = sham_config.get("randomization_file")
            if randomization_file:
                randomization_path = Path(randomization_file)
                if randomization_path.exists():
                    try:
                        randomization = load_randomization(randomization_path)
                        condition_info = get_condition_for_participant(
                            randomization, self.participant_id
                        )
                        condition = condition_info["condition"]
                        self._emit_progress(f"Participant condition: {condition}")

                        delivery_method = sham_config.get("delivery_method", "alternative_masks")

                        if delivery_method == "alternative_masks":
                            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
                            if mask_mode == "hemisphere_split":
                                # Hemisphere split: use left/right halves of a single network
                                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                                target_hemisphere = sham_config.get("target_hemisphere", "right")
                                non_target_hemisphere = "left" if target_hemisphere == "right" else "right"

                                # Target hemisphere is sham target, non-target is sham non-target
                                for hemi in [target_hemisphere, non_target_hemisphere]:
                                    alt_name = f"{hemisphere_network}_{hemi}"
                                    if alt_name not in real_mask_names:
                                        try:
                                            sham_mask_files.append(find_mask_file(alt_name))
                                            sham_mask_names.append(alt_name)
                                        except ValueError as e:
                                            logger.warning(f"Sham hemisphere mask not found: {e}")
                            else:
                                # Two networks mode: use two distinct networks
                                alternative_targets = sham_config.get("alternative_mask_targets", [])
                                for alt_name in alternative_targets:
                                    # Only add if not already in real_mask_names
                                    if alt_name not in real_mask_names:
                                        try:
                                            sham_mask_files.append(find_mask_file(alt_name))
                                            sham_mask_names.append(alt_name)
                                        except ValueError as e:
                                            logger.warning(f"Sham mask not found: {e}")

                        elif delivery_method == "replay" and condition == "sham":
                            # Replay mode: find feedbacklog from matched participant
                            matched_participant = condition_info.get("matched_participant")
                            matched_visit = condition_info.get("matched_visit")

                            if not matched_participant:
                                raise ValueError(
                                    f"Sham participant {self.participant_id} has replay mode "
                                    f"but no matched_participant in randomization file"
                                )

                            self._emit_progress(
                                f"Replay mode: looking up feedbacklog from {matched_participant}"
                                + (f" visit {matched_visit}" if matched_visit else "")
                            )

                            try:
                                # base_dir is participant-specific (e.g., /data/sub-02)
                                # For replay, we need the study base dir (e.g., /data)
                                # to find other participants' feedbacklogs
                                study_base_dir = base_dir.parent
                                feedbacklog_path = find_feedbacklog_for_replay(
                                    source_dir=study_base_dir,
                                    matched_participant=matched_participant,
                                    matched_visit=matched_visit,
                                    task=task_type,
                                    run=task_run_number,
                                )
                                self._emit_progress(f"Found feedbacklog: {feedbacklog_path.name}")

                                # Create replay provider with the real mask names
                                sham_provider = ReplayActivationProvider(
                                    feedbacklog_path=feedbacklog_path,
                                    mask_names=real_mask_names,
                                )
                                self._emit_progress(
                                    f"Replay provider initialized with {sham_provider._n_source_volumes} volumes"
                                )
                            except FileNotFoundError as e:
                                logger.error(f"Could not find feedbacklog for replay: {e}")
                                self._emit_progress(f"ERROR: {e}")
                                raise

                    except Exception as e:
                        logger.error(f"Error loading randomization: {e}")
                        self._emit_progress(f"Warning: Could not load randomization file: {e}")
                        raise
                else:
                    logger.warning(f"Randomization file not found: {randomization_path}")

        # Combine all masks: real masks + any additional sham masks
        all_mask_files = real_mask_files + sham_mask_files
        all_mask_names = real_mask_names + sham_mask_names
        self._emit_progress(f"Computing activations for masks: {all_mask_names}")

        # Determine which masks control the ball based on condition
        # For real: use real_mask_names (e.g., ContA, DefaultA)
        # For sham: substitute first real mask with first sham mask (e.g., SomMotA replaces ContA)
        if condition == "sham" and sham_mask_names:
            # Sham: use sham mask for top (first), keep real mask for bottom (second)
            ball_mask_names = [sham_mask_names[0], real_mask_names[1]]
            self._emit_progress(f"Sham: ball controlled by {ball_mask_names} (instead of {real_mask_names})")
        else:
            ball_mask_names = real_mask_names

        # Run feedback session (this will handle DICOM collection + processing)
        # Pass ALL masks so all activations are computed and logged
        try:
            # Get MURFI design path (only used if non-empty and in simulator mode)
            murfi_design_path = config["scan"].get("murfi_design_path", "")
            if not murfi_design_path or not simulator_enabled:
                murfi_design_path = None

            session_info = run_feedback_session(
                mask_paths=[str(f) for f in all_mask_files],
                subject=self.participant_id,
                task=task_type,
                run=task_run_number,
                output_dir=output_dir,
                collection_dir=collection_dir,
                incoming_dir=watch_dir,
                n_baseline=n_baseline,
                n_volumes=n_volumes,
                tr=tr,
                task_regressors=task_regressor,  # Pass the computed task regressor (overridden by MURFI if provided)
                mask_names=all_mask_names,  # All masks for logging
                use_efficiency_weights=config["feedback"].get("use_efficiency_weights", True),
                simulator_source_dir=simulator_source_dir,
                simulator_speed=simulator_speed,
                step_id=step_id,  # Use slot-based naming with acq entity
                qc_enabled=True,
                fd_threshold=config["feedback"].get("fd_threshold", 0.5),
                condition=condition,  # Log the actual condition
                sham_provider=sham_provider,  # Replay provider for sham participants
                moco_enabled=config["feedback"].get("moco_enabled", True),  # Motion correction mode
                murfi_design_path=murfi_design_path,  # External design matrix (MURFI format)
            )

            # Wait for the coordinator thread to complete
            coordinator = session_info['coordinator']
            coordinator_thread = session_info['thread']

            # Track coordinator for stop signal
            self._current_coordinator = coordinator

            # Launch PsychoPy task in separate thread
            self._emit_progress(f"Launching PsychoPy task...")

            # Import task factory
            from pineuro.tasks import create_task

            # Get task configuration
            task_config = config.get("task", {})
            feedback_task_type = task_config.get("task_type", "ball")
            task_specific_config = task_config.get(feedback_task_type, {})

            # Get display configuration
            display_config = config.get("display", {})
            presenter_monitor = display_config.get("presenter_monitor", "right")

            # Load session state for adaptive difficulty persistence across runs
            session_state = SessionState.load(output_dir, task_id=feedback_task_type)

            # Update max_feedback_runs from config if different
            max_feedback_runs = task_config.get("max_feedback_runs", 2)
            if session_state.max_feedback_runs != max_feedback_runs:
                session_state.max_feedback_runs = max_feedback_runs
                session_state.save()

            self._emit_progress(f"Session state: task={feedback_task_type}, runs={len(session_state.completed_runs)}")

            # Build factory config with task parameters plus runtime settings
            factory_config = {
                **task_specific_config,
                "mask_names": ball_mask_names,
                "feedback_mode": feedback_mode,
                "presenter_monitor": presenter_monitor,
            }

            # Create task via factory (validates config against task metadata)
            task = create_task(
                task_id=feedback_task_type,
                coordinator=coordinator,
                config=factory_config,
                session_state=session_state,
                output_dir=str(output_dir),
            )

            # Register callback so coordinator updates task with new activations
            coordinator._activation_callback = task._update_data

            # Connect GUI mirror callback if set
            if self._task_mirror_callback is not None:
                task.set_mirror_callback(self._task_mirror_callback)

            # Run task in separate thread (task will call coordinator.send_trigger())
            def run_task():
                try:
                    self._emit_progress(f"Task thread started...")
                    task.run()
                    self._emit_progress(f"Task completed successfully")
                except Exception as e:
                    logger.error(f"Task error: {e}", exc_info=True)
                    self._emit_progress(f"Task error: {e}")

            task_thread = threading.Thread(target=run_task, daemon=False, name="PsychoPyTask")
            task_thread.start()

            self._emit_progress(f"Session running, waiting for completion...")

            # Wait for both threads to complete
            coordinator_thread.join()  # Wait for coordinator to finish
            task_thread.join()  # Wait for task to finish

            # Extract results from coordinator and task
            volumes_processed = coordinator.n_processed

            # Build results message (handle tasks with/without hit_counters)
            result_msg = f"{task_type.capitalize()} run {task_run_number} complete! Processed {volumes_processed} volumes."
            if hasattr(task, 'hit_counters') and task.hit_counters:
                hits = {
                    ball_mask_names[i]: task.hit_counters[i]
                    for i in range(min(len(ball_mask_names), len(task.hit_counters)))
                }
                result_msg += f" Hits: {hits}"

            self._emit_progress(result_msg)
        finally:
            # Clear coordinator reference
            self._current_coordinator = None
