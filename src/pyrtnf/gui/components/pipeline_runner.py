"""Pipeline execution runner for background thread execution.

This module provides thread-safe execution of pyrtnf pipeline steps,
allowing the GUI to remain responsive during long-running operations.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from pyrtnf.preprocessing import collect_dicoms, preprocessing
from pyrtnf.mask_extraction import mask_extraction_rest, coregister_2vol, generate_2vol_qc_overlays
from pyrtnf.feedback import run_feedback_session, get_next_run_number, get_slot_run_info
from pyrtnf.tasks.ball import compute_boxcar_task_regressor

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
    ):
        """Initialize pipeline runner.

        Args:
            config: Session configuration dictionary
            participant_id: Participant ID
            on_progress: Callback for progress updates (message)
            on_complete: Callback when step completes (success, result_message)
            on_error: Callback when step fails (exception)
        """
        self.config = config
        self.participant_id = participant_id
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_coordinator = None  # Track coordinator for feedback tasks
        self._task_mirror_callback: Optional[Callable[[Dict[str, Any]], None]] = None

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
        if self._thread is not None and self._thread.is_alive():
            logger.info(f"Signaling pipeline step to stop (show_questions={show_questions})")
            self._stop_event.set()

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
            elif step_id == "rest_run_1":
                self._run_rest_run_1()
            elif step_id == "rest_run_2":
                self._run_rest_run_2()
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
            elif step_id in ["transfer_1", "feedback_1", "feedback_2", "transfer_2"]:
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
        self._emit_progress("Collecting initial 2vol DICOMs...")

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        output_dir = base_dir / "collection" / "task-2vol_run-01"

        # Create temporary watch directory
        watch_dir = base_dir / "tmp" / "incoming_2vol_initial"
        watch_dir.mkdir(parents=True, exist_ok=True)

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=config["scan"]["twovol_n_volumes"],
            simulator_source_dir=config["scan"]["twovol_initial_dicoms"],
            simulator_speed=config["scan"]["collection_speed"],
            clear_watch_dir=True,
            clear_output_dir=False,  # Don't clear if re-running
        )

        self._emit_progress(f"Collected {config['scan']['twovol_n_volumes']} volumes to {output_dir}")

    def _run_rest_run_1(self) -> None:
        """Collect resting-state run 1."""
        self._emit_progress("Collecting rest run 1 DICOMs...")

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        output_dir = base_dir / "collection" / "task-rest_run-01"

        # Get expected volumes
        rest_volumes = config["scan"]["rest_n_volumes"]
        if isinstance(rest_volumes, list):
            n_volumes = rest_volumes[0]
        else:
            n_volumes = rest_volumes

        # Get source directory
        rest_sources = config["scan"]["rest_source_dicoms"]
        if isinstance(rest_sources, list):
            source_dir = rest_sources[0]
        else:
            source_dir = rest_sources

        # Create temporary watch directory
        watch_dir = base_dir / "tmp" / "incoming_rest_run1"
        watch_dir.mkdir(parents=True, exist_ok=True)

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=n_volumes,
            simulator_source_dir=source_dir,
            simulator_speed=config["scan"]["collection_speed"],
            clear_watch_dir=True,
            clear_output_dir=False,
        )

        self._emit_progress(f"Collected {n_volumes} volumes to {output_dir}")

    def _run_rest_run_2(self) -> None:
        """Collect resting-state run 2."""
        self._emit_progress("Collecting rest run 2 DICOMs...")

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        output_dir = base_dir / "collection" / "task-rest_run-02"

        # Get expected volumes
        rest_volumes = config["scan"]["rest_n_volumes"]
        if isinstance(rest_volumes, list) and len(rest_volumes) > 1:
            n_volumes = rest_volumes[1]
        else:
            n_volumes = rest_volumes if not isinstance(rest_volumes, list) else rest_volumes[0]

        # Get source directory
        rest_sources = config["scan"]["rest_source_dicoms"]
        if isinstance(rest_sources, list) and len(rest_sources) > 1:
            source_dir = rest_sources[1]
        else:
            source_dir = rest_sources if not isinstance(rest_sources, list) else rest_sources[0]

        # Create temporary watch directory
        watch_dir = base_dir / "tmp" / "incoming_rest_run2"
        watch_dir.mkdir(parents=True, exist_ok=True)

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=n_volumes,
            simulator_source_dir=source_dir,
            simulator_speed=config["scan"]["collection_speed"],
            clear_watch_dir=True,
            clear_output_dir=False,
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

        # Combine run directories
        input_dicoms = [rest_run_1_dir]
        if rest_run_2_dir.exists():
            input_dicoms.append(rest_run_2_dir)

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
        yeo_targets = config["mask"]["yeo_targets"]
        smooth_fwhm = config["mask"]["smooth_fwhm"]
        highpass = config["mask"]["highpass"]
        lowpass = config["mask"].get("lowpass")

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
            smooth_fwhm=smooth_fwhm,
            highpass=highpass,
            lowpass=lowpass,
            overwrite=True,
            qc_enabled=True,
            progress_callback=progress_callback,
        )

    def _run_coregistration(self, run_id: int = 1) -> None:
        """Coregister masks to 2vol feedback space.

        Args:
            run_id: Run number (1 for initial, 2 for realtime)
        """
        run_str = f"run-{run_id:02d}"
        self._emit_progress(f"Coregistering network masks to 2vol {run_str}...")

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"
        mask_extraction_dir = base_dir / "mask_extraction"
        feedback_dir = base_dir / "feedback"

        # Input files
        ref_file = preprocessing_dir / f"{self.participant_id}_task-rest_boldref.nii.gz"
        masks_dir = mask_extraction_dir / "03_masks"
        twovol_dicom = base_dir / "collection" / f"task-2vol_{run_str}"

        # Get network mask files (thresholded only)
        network_files = list(masks_dir.glob(f"{self.participant_id}_*_thr.nii.gz"))

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
            run_id=run_id,
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

            generate_2vol_qc_overlays(
                participant_id=self.participant_id,
                session_dir=session_dir,
                ref_file=ref_file,
                mask_extraction_dir=mask_extraction_dir,
                network_coreg_files=coreg_masks,
                run_str=str(run_id).zfill(2),
                overwrite=True,
                twovol_subdir=twovol_subdir,
            )
            self._emit_progress(f"QC overlays generated for 2vol {run_str}")
        except Exception as e:
            logger.warning(f"2vol QC overlay generation failed: {e}")
            self._emit_progress(f"Warning: QC overlay generation failed: {e}")

    def _run_2vol_realtime(self) -> None:
        """Collect real-time 2vol before feedback session."""
        self._emit_progress("Collecting real-time 2vol DICOMs...")

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        output_dir = base_dir / "collection" / "task-2vol_run-02"

        # Create temporary watch directory
        watch_dir = base_dir / "tmp" / "incoming_2vol_realtime"
        watch_dir.mkdir(parents=True, exist_ok=True)

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=config["scan"]["twovol_n_volumes"],
            simulator_source_dir=config["scan"]["twovol_realtime_dicoms"],
            simulator_speed=config["scan"]["collection_speed"],
            clear_watch_dir=True,
            clear_output_dir=False,
        )

        self._emit_progress(f"Collected {config['scan']['twovol_n_volumes']} volumes to {output_dir}")

    def _run_feedback_task(self, step_id: str) -> None:
        """Run a feedback or transfer task.

        Args:
            step_id: One of transfer_1, feedback_1, feedback_2, transfer_2
        """
        # Parse step ID to determine task type
        task_type_map = {
            "transfer_1": "transfer",
            "feedback_1": "feedback",
            "feedback_2": "feedback",
            "transfer_2": "transfer",
        }

        task_type = task_type_map[step_id]
        feedback_mode = (task_type == "feedback")

        # Get output directory for slot-based run info
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
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

        # Get coregistered mask files (use thresholded masks from realtime 2vol run-02)
        # IMPORTANT: Match files to mask_names order, not alphabetically sorted
        twovol_masks_dir = feedback_dir / "01_2vol" / "masks"
        mask_names = config["mask"]["mask_names"]

        # Find mask files matching each mask name in order
        mask_files = []
        for mask_name in mask_names:
            pattern = f"*_{mask_name}_thr_to_task-2vol_run-02_*.nii.gz"
            matches = list(twovol_masks_dir.glob(pattern))
            if not matches:
                raise ValueError(
                    f"Could not find mask file for '{mask_name}' in {twovol_masks_dir}. "
                    f"Expected pattern: {pattern}"
                )
            if len(matches) > 1:
                logger.warning(f"Multiple mask files found for '{mask_name}', using first: {matches[0]}")
            mask_files.append(matches[0])

        # Verify mask count matches
        if len(mask_files) != len(mask_names):
            raise ValueError(
                f"Found {len(mask_files)} mask files but config specifies {len(mask_names)} mask names. "
                f"Mask files: {[f.name for f in mask_files]}, "
                f"Mask names: {mask_names}"
            )

        # Create temporary watch directory
        watch_dir = base_dir / "tmp" / f"incoming_{task_type}_acq-{task_acq:02d}_run-{task_run_number:02d}"
        watch_dir.mkdir(parents=True, exist_ok=True)

        # Collection directory for this run
        collection_dir = base_dir / "collection"

        self._emit_progress(f"Preparing {task_type} run (feedback={'ON' if feedback_mode else 'OFF'})...")

        # Compute task regressor (HRF-convolved boxcar for ball task)
        n_baseline = config["feedback"]["n_baseline"]
        n_volumes = config["feedback"]["n_volumes"]
        tr = config["scan"]["tr"]
        task_regressor = compute_boxcar_task_regressor(n_baseline, n_volumes, tr)

        # Run feedback session (this will handle DICOM collection + processing)
        try:
            session_info = run_feedback_session(
                mask_paths=[str(f) for f in mask_files],
                subject=self.participant_id,
                task=task_type,
                run=task_run_number,
                output_dir=output_dir,
                collection_dir=collection_dir,
                incoming_dir=watch_dir,
                n_baseline=n_baseline,
                n_volumes=n_volumes,
                tr=tr,
                task_regressors=task_regressor,  # Pass the computed task regressor
                mask_names=mask_names,
                use_efficiency_weights=config["feedback"].get("use_efficiency_weights", True),
                simulator_source_dir=config["scan"]["feedback_dicoms"],
                simulator_speed=config["scan"].get("simulator_speed", 1.0),
                step_id=step_id,  # Use slot-based naming with acq entity
                qc_enabled=True,
            )

            # Wait for the coordinator thread to complete
            coordinator = session_info['coordinator']
            coordinator_thread = session_info['thread']

            # Track coordinator for stop signal
            self._current_coordinator = coordinator

            # Launch PsychoPy task in separate thread
            self._emit_progress(f"Launching PsychoPy task...")

            # Import task class
            from pyrtnf.tasks import BallTask

            # Get task configuration
            task_config = config.get("task", {})

            # Create and run task
            task = BallTask(
                coordinator=coordinator,
                mask_names=mask_names,
                scale_factor=task_config.get("initial_scale_factor", 10.0),
                fullscreen=task_config.get("fullscreen", True),
                trigger_keys=task_config.get("trigger_keys", ["t", "+", "5", "equal"]),
                left_button=task_config.get("left_button", "1"),
                right_button=task_config.get("right_button", "2"),
                enter_button=task_config.get("enter_button", "0"),
                min_hits_threshold=task_config.get("min_hits_threshold", 3),
                max_hits_threshold=task_config.get("max_hits_threshold", 5),
                scale_adjust_factor=task_config.get("scale_adjust_factor", 0.25),
                feedback_mode=feedback_mode,
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
            hits = {
                mask_names[0]: task.hit_counters[0],
                mask_names[1]: task.hit_counters[1],
            }

            self._emit_progress(
                f"{task_type.capitalize()} run {task_run_number} complete! "
                f"Processed {volumes_processed} volumes. Hits: {hits}"
            )
        finally:
            # Clear coordinator reference
            self._current_coordinator = None
