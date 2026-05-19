"""Pipeline execution runner for background thread execution.

This module provides thread-safe execution of pineuro pipeline steps,
allowing the GUI to remain responsive during long-running operations.
"""

import contextlib
import logging
import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Heavy pineuro imports are deferred to method bodies to avoid loading
# numpy/nibabel/scipy when the GUI module is first imported.

logger = logging.getLogger(__name__)


def _select_sham_task_mask_names(
    sham_mask_names: list[str],
    real_mask_names: list[str],
) -> list[str]:
    """Pick the pair of masks that drive the feedback task in sham condition.

    Assumes ``sham_mask_names`` is non-empty. When two or more sham masks are
    available, both the target and non-target roles are driven by sham masks
    (e.g. ``[target_hemi, non_target_hemi]`` for ``hemisphere_split``). With
    only a single sham mask, the target role is swapped and the real
    non-target is kept.
    """
    if len(sham_mask_names) >= 2:
        return list(sham_mask_names[:2])
    return [sham_mask_names[0], real_mask_names[1]]


class _DispatchedTask:
    """Thin wrapper around a :class:`~concurrent.futures.Future`.

    Exposes ``.is_alive()`` and ``.join()`` so it can be used as a drop-in
    replacement for the :class:`threading.Thread` returned by
    ``_launch_localizer_task()`` on Windows/Linux.
    """

    def __init__(self, future):
        self._future = future

    def is_alive(self) -> bool:
        return not self._future.done()

    def join(self, timeout: float | None = None) -> None:
        with contextlib.suppress(Exception):
            self._future.result(timeout=timeout)


class PipelineRunner:
    """Execute pipeline steps in background threads with progress callbacks."""

    def __init__(
        self,
        config: dict[str, Any],
        participant_id: str,
        on_progress: Callable[[str], None] | None = None,
        on_complete: Callable[[bool, str | None], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_collection_progress: Callable[[int, int, float, Path], None] | None = None,
        on_qc_data: Callable[[int, float, float, float, str | None], None] | None = None,
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
            on_qc_data: Callback for real-time QC data during collection
                (volume, fd, dvars, tsnr, slice_b64)
        """
        self.config = config
        self.participant_id = participant_id
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_collection_progress = on_collection_progress
        self.on_qc_data = on_qc_data

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_coordinator = None  # Track coordinator for feedback tasks
        self._current_feedback_task: Any = None  # Track feedback task for stop support
        self._current_localizer_task = None  # Track localizer task for stop support
        self._task_mirror_callback: Callable[[dict[str, Any]], None] | None = None
        self.overwrite: bool = False
        self._selected_rest_runs: list | None = None  # Selected runs for preprocessing
        self._selected_acqs: dict[int, list[int]] | None = None  # {run_id: [acq_nums]}

    def _resolve_feedback_run_config(self, step_id: str) -> dict[str, Any] | None:
        """Return the ``feedback_runs`` entry for *step_id*, or None.

        The mapping is derived from ``config["feedback_runs"]``.  Each entry
        contributes transfer + feedback steps in order, so the cumulative
        step count determines which entry owns a given step_id.
        """
        feedback_runs = self.config.get("feedback_runs")
        if not feedback_runs:
            return None

        step_to_frc: dict[str, dict[str, Any]] = {}
        fb_counter = 0
        tf_counter = 0
        for frc in feedback_runs:
            use_transfers = frc.get("use_transfer_runs", False)
            if use_transfers:
                tf_counter += 1
                step_to_frc[f"transfer_{tf_counter}"] = frc
            for _ in range(frc.get("max_runs", 1)):
                fb_counter += 1
                step_to_frc[f"feedback_{fb_counter}"] = frc
            if use_transfers:
                tf_counter += 1
                step_to_frc[f"transfer_{tf_counter}"] = frc

        return step_to_frc.get(step_id)

    @staticmethod
    def _derive_differential_pair(frc: dict[str, Any] | None, task_type: str) -> list[str] | None:
        """Compute the ``[positive, negative]`` mask pair from a run's network_assignment.

        Returns None if the task is not a 2-network task or the assignment is incomplete.
        Replaces the legacy ``feedback_runs[*].differential_pair`` field.
        """
        if not frc:
            return None
        from pineuro.tasks.task_registry import get_task_metadata

        try:
            metadata = get_task_metadata(task_type)
        except (KeyError, ValueError):
            return None
        if len(metadata.network_roles) != 2:
            return None
        assignment = frc.get("network_assignment", {}) or {}
        pos = assignment.get(metadata.network_roles[0].key)
        neg = assignment.get(metadata.network_roles[1].key)
        if pos and neg:
            return [pos, neg]
        return None

    def set_selected_acqs(self, acqs: dict[int, list[int]] | None) -> None:
        """Set which acquisitions to use for each run.

        Args:
            acqs: Dict {run_id: [acq_nums]}, or None to auto-select highest complete
        """
        self._selected_acqs = acqs

    def set_selected_rest_runs(self, runs: list | None) -> None:
        """Set which rest runs to use for preprocessing.

        Args:
            runs: List of run IDs (1 and/or 2), or None to use all available
        """
        self._selected_rest_runs = runs

    def set_overwrite(self, overwrite: bool) -> None:
        """Set whether pipeline steps should overwrite existing outputs.

        Args:
            overwrite: If True, re-run all sub-steps. If False, skip sub-steps
                whose output files already exist.
        """
        self.overwrite = overwrite

    def set_task_mirror_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """Set callback for task visual state mirroring.

        The callback receives visual state dictionaries from tasks
        for rendering in a GUI mirror component.

        Parameters
        ----------
        callback : Callable or None
            Function that receives visual state dict from task.get_mirror_state()
        """
        self._task_mirror_callback = callback

    @staticmethod
    def find_available_collections(
        collection_dir: Path, task: str, run: int, expected_volumes: int
    ) -> list[dict[str, Any]]:
        """Find all complete collection folders for a run.

        Parameters
        ----------
        collection_dir : Path
            Collection directory to scan
        task : str
            Task name (e.g., 'task-rest', 'task-reference')
        run : int
            Run number
        expected_volumes : int
            Expected number of volumes for a complete collection

        Returns
        -------
        list[dict]
            List of dicts with keys:
            - acq: int - Acquisition number
            - path: Path - Folder path
            - n_volumes: int - Number of volumes present
            - is_complete: bool - True if >= expected_volumes

            Sorted by acq number (highest first)
        """
        import re

        if not collection_dir.exists():
            return []

        pattern = f"{task}_acq-*_run-{run:02d}"
        collections: list[dict[str, Any]] = []

        for folder in collection_dir.glob(pattern):
            if folder.is_dir():
                match = re.search(r"acq-(\d+)_run-", folder.name)
                if match:
                    acq_num = int(match.group(1))
                    n_files = len([f for f in folder.iterdir() if f.is_file()])
                    is_complete = n_files >= expected_volumes

                    collections.append(
                        {
                            "acq": acq_num,
                            "path": folder,
                            "n_volumes": n_files,
                            "is_complete": is_complete,
                        }
                    )

        # Sort by acq number (highest first)
        collections.sort(key=lambda x: x["acq"], reverse=True)
        return collections

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
            # If a feedback task is running, stop the coordinator and the task
            if self._current_coordinator is not None:
                logger.info("Stopping feedback coordinator")
                self._current_coordinator.stop(show_questions=show_questions)
            if self._current_feedback_task is not None:
                logger.info("Stopping feedback task (PsychoPy)")
                self._current_feedback_task.stop()
            # If a localizer task is running, stop it
            if self._current_localizer_task is not None:
                logger.info("Stopping localizer task")
                self._current_localizer_task.stop()

    def _run_step_impl(self, step_id: str) -> None:
        """Implementation of step execution (runs in background thread).

        Args:
            step_id: Step identifier to execute
        """
        try:
            logger.info(f"Executing step: {step_id}")

            # Execute the appropriate step
            if step_id == "reference_initial":
                self._run_reference_initial()
            elif step_id.startswith("rest_run_"):
                # Handle dynamic rest run steps (rest_run_1, rest_run_2, etc.)
                run_num = int(step_id.split("_")[-1])
                self._run_rest_run(run_num)
            elif step_id == "preprocessing":
                self._run_preprocessing()
            elif step_id == "mask_extraction":
                self._run_mask_extraction()
            elif step_id == "mni_mask_extraction":
                self._run_mni_mask_extraction()
            elif step_id == "coregistration":
                self._run_coregistration(run_id=1)
            elif step_id == "reference_realtime":
                self._run_reference_realtime()
            elif step_id == "coregistration_realtime":
                self._run_coregistration(run_id=2)
            elif step_id.startswith("transfer_") or step_id.startswith("feedback_"):
                self._run_feedback_task(step_id)
            elif step_id.startswith("localizer_") and "_run_" in step_id:
                # Multi-localizer collection: localizer_{idx}_run_{run_num}
                parts = step_id.split("_")
                loc_idx = int(parts[1])
                run_num = int(parts[-1])
                self._run_localizer_collection(loc_idx, run_num)
            elif step_id.startswith("localizer_") and step_id.endswith("_preprocessing"):
                # Multi-localizer preprocessing: localizer_{idx}_preprocessing
                loc_idx = int(step_id.split("_")[1])
                self._run_localizer_preprocessing(loc_idx)
            elif step_id.startswith("localizer_") and step_id.endswith("_extraction"):
                # Multi-localizer extraction: localizer_{idx}_extraction
                loc_idx = int(step_id.split("_")[1])
                self._run_localizer_extraction(loc_idx)
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

    def _emit_progress_file_only(self, message: str) -> None:
        """Log a progress message to the file log only (not forwarded to GUI)."""
        logger.info(f"Progress: {message}")

    def _is_blind_mode(self) -> bool:
        """Check if sham blind mode is active."""
        sham_config = self.config.get("sham", {})
        return bool(sham_config.get("enabled", False) and sham_config.get("blind_mode", False))

    def _run_reference_initial(self) -> None:
        """Collect initial reference scan for positioning verification."""
        from pineuro.preprocessing import collect_dicoms
        from pineuro.session import get_localizer_run_info

        config = self.config

        # Check if reference initial is enabled
        if not config.get("reference", {}).get("initial_enabled", True):
            self._emit_progress("Reference initial collection is disabled, skipping...")
            return

        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        # Get number of volumes (use new field, fall back to legacy)
        n_volumes = config.get("reference", {}).get("initial_n_volumes", 2)

        # Get run info with retry detection
        run_info = get_localizer_run_info(
            collection_dir=collection_dir, task="task-reference", run=1, expected_volumes=n_volumes
        )
        acq = run_info["acq"]

        self._emit_progress(f"Collecting initial reference DICOMs (acq-{acq:02d})...")
        if run_info["existing_acqs"]:
            self._emit_progress(f"Previous attempts: {', '.join(f'acq-{a:02d}' for a in run_info['existing_acqs'])}")

        output_dir = collection_dir / f"task-reference_acq-{acq:02d}_run-01"

        # Determine simulator vs real-time mode
        simulator_enabled = config.get("simulator", {}).get("enabled", True)

        if simulator_enabled:
            # Simulator mode: create temp watch dir and use source DICOMs
            watch_dir = base_dir / "tmp" / "incoming_reference_initial"
            watch_dir.mkdir(parents=True, exist_ok=True)
            simulator_source_dir = config.get("reference", {}).get("initial_dicoms", "")

            # Try mapping file if individual field is empty
            if not simulator_source_dir:
                mapping_path = config.get("simulator", {}).get("source_mapping", "")
                if mapping_path and self.participant_id:
                    try:
                        from pineuro.session import SimulatorSourceMapping

                        mapping = SimulatorSourceMapping.from_file(Path(mapping_path))
                        resolved = mapping.get_source(self.participant_id, "reference_initial")
                        if resolved is not None:
                            simulator_source_dir = str(resolved)
                            logger.info(f"reference_initial source from mapping: {simulator_source_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to resolve reference_initial from mapping: {e}")

            simulator_speed = config.get("simulator", {}).get("collection_speed", 1.0)
        else:
            # Real-time mode: watch scanner DICOM directory
            watch_dir = Path(config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM"))
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
        """Collect resting-state or localizer task run.

        For rest localizers, this just collects DICOMs.
        For task localizers, this also launches the localizer PsychoPy task
        concurrently with DICOM collection.

        Args:
            run_num: Run number (1-based)
        """
        from pineuro.preprocessing import collect_dicoms
        from pineuro.session import get_localizer_run_info

        config = self.config
        localizer_type = config.get("mask", {}).get("localizer_type", "rest")
        localizer_task_type = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
        task_name = localizer_task_type if localizer_task_type != "rest" else "rest"

        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        # Get expected volumes (use new localizer fields, fall back to legacy)
        n_volumes = config["localizers"][0].get("n_volumes") if config.get("localizers") else None
        if n_volumes is None:
            rest_volumes = (
                [config["localizers"][0].get("n_volumes", 200)] * config["localizers"][0].get("n_runs", 1)
                if config.get("localizers")
                else [200]
            )
            if isinstance(rest_volumes, list):
                if len(rest_volumes) >= run_num:
                    n_volumes = rest_volumes[run_num - 1]
                else:
                    n_volumes = rest_volumes[0] if rest_volumes else 200
            else:
                n_volumes = rest_volumes

        # If the localizer task declares its own worst-case duration, override
        # the user-configured n_volumes with the computed ceiling. The existing
        # task_done_event mechanism still lets collect_dicoms stop early when
        # the task finishes, so this only caps collection — it never forces it
        # to run longer than the task actually does.
        if localizer_type == "task" and localizer_task_type != "rest":
            from pineuro.tasks.task_registry import get_task_metadata

            meta = get_task_metadata(localizer_task_type)
            if meta.compute_max_run_seconds is not None:
                task_params = config["localizers"][0].get("task_params", {}) if config.get("localizers") else {}
                tr = config["localizers"][0].get("tr", 1.2) if config.get("localizers") else 1.2
                computed = math.ceil(meta.compute_max_run_seconds(task_params) / tr)
                self._emit_progress(
                    f"{localizer_task_type}: auto-computed n_volumes={computed} "
                    f"(worst-case ceiling; collection ends early on task_done_event)"
                )
                n_volumes = computed

        # Get run info with retry detection
        run_info = get_localizer_run_info(
            collection_dir=collection_dir, task=f"task-{task_name}", run=run_num, expected_volumes=n_volumes
        )
        acq = run_info["acq"]

        self._emit_progress(f"Collecting {task_name} run {run_num} (acq-{acq:02d}) DICOMs...")
        if run_info["existing_acqs"]:
            self._emit_progress(f"Previous attempts: {', '.join(f'acq-{a:02d}' for a in run_info['existing_acqs'])}")

        output_dir = collection_dir / f"task-{task_name}_acq-{acq:02d}_run-{run_num:02d}"

        # Determine simulator vs real-time mode
        simulator_enabled = config.get("simulator", {}).get("enabled", True)

        if simulator_enabled:
            # Simulator mode: get source directory and create temp watch dir
            source_dir = None

            # 1. Try simulator_source_mapping first (per-subject paths)
            mapping_path = config.get("simulator", {}).get("source_mapping", "")
            if mapping_path and self.participant_id:
                try:
                    from pineuro.session import SimulatorSourceMapping

                    mapping = SimulatorSourceMapping.from_file(Path(mapping_path))
                    run_type = f"rest{run_num}"
                    resolved = mapping.get_source(self.participant_id, run_type)
                    if resolved is not None:
                        source_dir = str(resolved)
                        logger.info(f"Localizer source from mapping: {source_dir}")
                except Exception as e:
                    logger.warning(f"Failed to resolve localizer source from mapping: {e}")
            else:
                logger.debug("Skipping source mapping (no mapping_path or participant_id)")

            # 2. Fall back to individual localizer_source_dicoms field
            if not source_dir:
                localizer_sources = config["localizers"][0].get("source_dicoms") if config.get("localizers") else None
                if localizer_sources is None:
                    localizer_sources = (
                        config["localizers"][0].get("source_dicoms", []) if config.get("localizers") else []
                    )

                if isinstance(localizer_sources, list):
                    if len(localizer_sources) >= run_num:
                        source_dir = localizer_sources[run_num - 1]
                    elif len(localizer_sources) > 0:
                        source_dir = localizer_sources[0]
                else:
                    source_dir = localizer_sources

            watch_dir = base_dir / "tmp" / f"incoming_{task_name}_run{run_num}"
            watch_dir.mkdir(parents=True, exist_ok=True)
            # Clear stale files from previous runs so collect_dicoms doesn't
            # catalog them as "already processed" (the simulator will re-copy
            # files with the same names, which would then be skipped).
            for f in watch_dir.iterdir():
                if f.is_file():
                    f.unlink()
            simulator_speed = config.get("simulator", {}).get("collection_speed", 1.0)
        else:
            # Real-time mode: watch scanner DICOM directory
            source_dir = None
            watch_dir = Path(config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM"))
            simulator_speed = 1.0

        # Create simulator if in simulator mode
        simulator = None
        if simulator_enabled and source_dir:
            from pineuro.simulator import VolumeSimulator

            simulator = VolumeSimulator(
                source_dir=source_dir,
                target_dir=watch_dir,
                tr=(
                    config["localizers"][0].get("tr", 1.2) if config.get("localizers") else 1.2
                ),  # Use configured TR from study
                speed=simulator_speed,
                clear_target=True,
                trigger_mode=config.get("simulator", {}).get("trigger_mode", "manual"),
            )

        # Determine which localizer task to use
        if localizer_type == "task" and localizer_task_type != "rest":
            # Task-based localizer (e.g., self/other)
            actual_task_id = localizer_task_type
        else:
            # Rest localizer - use fixation task
            actual_task_id = "fixation"

        # Shared event so collect_dicoms knows when acquisition actually starts
        trigger_event = threading.Event()

        # Event set when the localizer task's paradigm finishes.
        # For variable-duration tasks (e.g., RNT), this signals collect_dicoms
        # to drain remaining volumes and stop, rather than waiting for n_volumes.
        task_done_event = threading.Event()

        # Always launch a localizer task - it controls simulator start timing
        localizer_thread = self._launch_localizer_task(
            run_num,
            actual_task_id,
            watch_dir,
            acq=acq,
            simulator=simulator,
            trigger_event=trigger_event,
            task_done_event=task_done_event,
        )

        # collect_dicoms just watches the directory - no simulator
        # (simulator is controlled by the task for synchronization)

        # Create QC monitor for real-time motion/quality metrics during collection
        from pineuro.collection_qc import CollectionQCMonitor

        motion_mode = config.get("scan", {}).get(
            "localizer_motion_correction",
            config.get("feedback", {}).get("motion_correction", "siemens_moco"),
        )
        qc_monitor = CollectionQCMonitor(
            log_path=output_dir / "collectionlog.tsv",
            motion_correction=motion_mode,
            on_qc_data=self.on_qc_data,
        )

        collect_dicoms(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=n_volumes,
            simulator_source_dir=None,  # No simulator - task controls it
            simulator_speed=simulator_speed,
            clear_watch_dir=False,  # Task's simulator handles clearing
            clear_output_dir=False,
            progress_callback=self.on_collection_progress,
            stop_event=self._stop_event,
            trigger_event=trigger_event,
            task_done_event=task_done_event,
            qc_monitor=qc_monitor,
        )

        # Report actual collected count (may be less than n_volumes for
        # variable-duration tasks that finish before the configured max)
        actual_count = len(list(output_dir.iterdir())) if output_dir.exists() else 0
        self._emit_progress(f"Collected {actual_count} volumes to {output_dir}")

        # Stop and wait for localizer task (collection is done, task can end)
        if self._current_localizer_task is not None:
            self._current_localizer_task.stop()
        if localizer_thread is not None and localizer_thread.is_alive():
            self._emit_progress("Waiting for localizer task to complete...")
            localizer_thread.join(timeout=10)
            self._current_localizer_task = None

    def _launch_localizer_task(
        self,
        run_num: int,
        task_id: str,
        watch_dir: Path,
        acq: int = 1,
        simulator=None,
        trigger_event=None,
        task_done_event: threading.Event | None = None,
    ):
        """Launch a localizer PsychoPy task on the main thread.

        The task runs concurrently with DICOM collection, presenting stimuli
        while the scanner acquires volumes.  The task is dispatched to the
        main thread via :func:`~pineuro.gui.dispatch.get_dispatcher` (required
        on macOS for Cocoa NSWindow creation).

        Args:
            run_num: Run number (1-based)
            task_id: Task identifier (e.g., "fixation", "self_other")
            watch_dir: Directory to watch for first DICOM arrival (auto-trigger)
            acq: Acquisition number (for retry handling)
            simulator: VolumeSimulator instance (optional, for offline testing)
            trigger_event: Event set when scanner trigger is received
            task_done_event: Event to set when task paradigm finishes, signaling
                collect_volumes to drain remaining volumes and stop

        Returns:
            _DispatchedTask wrapping the main-thread Future
        """
        from pineuro.tasks.factory import create_localizer_task

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])

        # Get task-specific parameters from config
        if task_id == "fixation":
            # For fixation task, use scan parameters
            localizer_params = {
                "duration": (config["localizers"][0].get("n_volumes", 250) if config.get("localizers") else 250)
                * (config["localizers"][0].get("tr", 2.0) if config.get("localizers") else 2.0)
            }
        else:
            # For other tasks, get from localizer_task config section
            localizer_params = config.get("localizer_task", {}).get(task_id, {})

        # Debug: log the parameters being passed
        logger.debug(f"Localizer task ID: {task_id}")
        logger.debug(f"Localizer params: {localizer_params}")

        # Output events to the preprocessing directory (survives cleanup)
        events_dir = base_dir / "preprocessing"
        events_dir.mkdir(parents=True, exist_ok=True)

        # Get presenter monitor from display config
        display_config = config.get("display", {})
        presenter_monitor = display_config.get("presenter_monitor", "right")

        task = create_localizer_task(
            task_id=task_id,
            output_dir=events_dir,
            participant_id=self.participant_id,
            config=localizer_params,
            run=run_num,
            acq=acq,  # Pass acq for events file naming
            watch_dir=str(watch_dir),  # Pass watch_dir for auto-trigger on first DICOM
            simulator=simulator,  # Pass simulator for task-controlled start timing
            monitor_position=presenter_monitor,
        )

        # Set trigger event for timing synchronization with collect_dicoms
        if trigger_event is not None:
            task.trigger_event = trigger_event

        # Set mirror callback if available
        if self._task_mirror_callback is not None:
            task.set_mirror_callback(self._task_mirror_callback)

        self._current_localizer_task = task

        def run_localizer():
            try:
                self._emit_progress(f"Localizer task '{task_id}' started (run {run_num})")
                events_path = task.run()
                if events_path:
                    self._emit_progress(f"Localizer events saved to {events_path}")
            except Exception as e:
                logger.error(f"Localizer task error: {e}", exc_info=True)
                self._emit_progress(f"Localizer task error: {e}")
            finally:
                # Signal that the task paradigm is done, allowing collect_volumes
                # to drain remaining volumes and stop for variable-duration tasks
                if task_done_event is not None:
                    task_done_event.set()

        # Dispatch task.run() to the main thread (required on macOS for
        # Cocoa NSWindow; used on all platforms for a single code path).
        from pineuro.gui.dispatch import get_dispatcher

        future = get_dispatcher().submit(run_localizer)
        logger.info(f"Dispatched localizer task '{task_id}' to main thread")
        return _DispatchedTask(future)

    def _resolve_acq_paths(self, task_name: str, runs_to_process: list[int], rest_volumes: list[int]) -> list[Path]:
        """Resolve collection folder paths using selected acqs.

        Uses self._selected_acqs if set (from GUI dialog), otherwise
        auto-selects highest complete acq for each run.

        Parameters
        ----------
        task_name : str
            Task name (e.g., 'rest', 'self_other')
        runs_to_process : list[int]
            List of run numbers to process
        rest_volumes : list[int]
            Expected volumes per run

        Returns
        -------
        list[Path]
            Selected DICOM folder paths
        """
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        selected_paths = []
        for run_num in runs_to_process:
            expected_vols = rest_volumes[run_num - 1] if len(rest_volumes) >= run_num else rest_volumes[0]

            # If user selected specific acqs, use those directly
            if self._selected_acqs and run_num in self._selected_acqs:
                acq_nums = self._selected_acqs[run_num]
                found_any = False
                for acq_num in acq_nums:
                    acq_path = collection_dir / f"task-{task_name}_acq-{acq_num:02d}_run-{run_num:02d}"
                    if acq_path.exists():
                        self._emit_progress(f"Run {run_num}: Using selected acq-{acq_num:02d}")
                        selected_paths.append(acq_path)
                        found_any = True
                    else:
                        logger.warning(f"Selected acq-{acq_num:02d} for run {run_num} not found")
                if found_any:
                    continue

            # Auto-select: find highest complete acq
            collections = self.find_available_collections(
                collection_dir=collection_dir, task=f"task-{task_name}", run=run_num, expected_volumes=expected_vols
            )

            if not collections:
                logger.warning(f"No collections found for run {run_num}")
                continue

            complete = [c for c in collections if c["is_complete"]]
            selected = complete[0] if complete else collections[0]
            self._emit_progress(f"Run {run_num}: Using acq-{selected['acq']:02d} ({selected['n_volumes']} volumes)")
            selected_paths.append(selected["path"])

        return selected_paths

    def _run_preprocessing(self) -> None:
        """Run preprocessing on collected rest runs."""
        from pineuro.preprocessing import preprocessing

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"
        preprocessing_dir = base_dir / "preprocessing"

        # Determine task name for localizer
        localizer_task_type = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
        task_name = localizer_task_type if localizer_task_type != "rest" else "rest"

        # Get expected volumes for finding complete collections
        rest_volumes = (
            [config["localizers"][0].get("n_volumes", 200)] * config["localizers"][0].get("n_runs", 2)
            if config.get("localizers")
            else [200, 200]
        )
        if not isinstance(rest_volumes, list):
            rest_volumes = [rest_volumes]

        # Determine which runs to process
        if self._selected_rest_runs is not None:
            runs_to_process = self._selected_rest_runs
            self._emit_progress(f"Using selected runs: {runs_to_process}")
        else:
            # Auto-detect available runs (up to 2)
            runs_to_process = []
            for run_num in [1, 2]:
                expected_vols = rest_volumes[run_num - 1] if len(rest_volumes) >= run_num else rest_volumes[0]
                collections = self.find_available_collections(
                    collection_dir=collection_dir, task=f"task-{task_name}", run=run_num, expected_volumes=expected_vols
                )
                if collections:
                    runs_to_process.append(run_num)

        # Select acquisitions (uses GUI selections if set, else auto-selects)
        input_dicoms = self._resolve_acq_paths(
            task_name=task_name, runs_to_process=runs_to_process, rest_volumes=rest_volumes
        )

        if not input_dicoms:
            raise ValueError(f"No {task_name} collection directories selected for preprocessing")

        # Log selected acquisitions
        for path in input_dicoms:
            self._emit_progress(f"Using: {path.name}")

        # Set up progress callback for preprocessing to show in GUI log console
        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        preproc_file, mask_file, ref_file = preprocessing(
            participant_id=self.participant_id,
            task=task_name,
            root_dir=preprocessing_dir,
            input_dicoms=[str(d) for d in input_dicoms],
            overwrite=self.overwrite,
            qc_enabled=True,  # Generate QC report
            progress_callback=progress_callback,
        )

    def _manual_reference_file(self, base_dir: Path, root_dir: Path, loc: dict) -> Path:
        """Resolve a manual localizer's reference EPI for coregistration.

        Prefers the reference recorded in the imported ``03_masks/mask_selections.json``
        (the volume ``import_manual_masks`` actually registered against), falling back
        to the per-participant ``manual_masks.json`` record.
        """
        import json

        from pineuro.gui.components.manual_masks import read_manual_masks

        sel = root_dir / "03_masks" / "mask_selections.json"
        if sel.exists():
            try:
                ref = json.loads(sel.read_text()).get("ref_file")
            except (json.JSONDecodeError, OSError):
                ref = None
            if ref and Path(ref).exists():
                return Path(ref)
        record = read_manual_masks(base_dir, loc.get("id", "manual"))
        ref = (record or {}).get("reference")
        if not ref:
            raise ValueError(f"Manual localizer '{loc.get('id', 'manual')}' reference EPI is not configured")
        return Path(ref)

    def _run_manual_mask_import(
        self,
        loc: dict,
        base_dir: Path,
        root_dir: Path,
        qc_dir: Path,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> list[Path]:
        """Import user-supplied masks for a manual localizer.

        The mask / reference / T1 files are participant-specific and read from
        ``base_dir/manual_masks.json``. Masks land in ``root_dir/03_masks/`` —
        the same location ICA/GLM extraction writes to — so downstream
        coregistration and feedback consume them unchanged.
        """
        from pineuro.gui.components.manual_masks import read_manual_masks
        from pineuro.mask_extraction import import_manual_masks

        loc_id = loc.get("id", "manual")
        record = read_manual_masks(base_dir, loc_id)
        if not record:
            raise ValueError(
                f"Manual mask files for localizer '{loc_id}' are not configured for this "
                "participant — use 'Configure / edit mask files…' first."
            )

        masks = record.get("masks", {}) or {}
        # Import in the network-name order from the study config (falls back to
        # whatever the per-participant record contains).
        names = list(loc.get("manual_mask_names") or []) or list(masks.keys())
        mask_space = record.get("mask_space") or loc.get("manual_mask_space", "native")
        ref_file = record.get("reference")
        anat_file = record.get("anat")

        missing = [n for n in names if n not in masks]
        if missing:
            raise ValueError(f"Manual mask file(s) not configured for: {', '.join(missing)}")
        if not ref_file:
            raise ValueError("Manual localizer requires a reference EPI file")
        if mask_space == "anat" and not anat_file:
            raise ValueError("Manual localizer with anatomical-space masks requires a T1 file")

        mask_paths = [masks[n] for n in names]
        self._emit_progress(f"Importing {len(mask_paths)} user-supplied mask(s) ({mask_space} space)")
        return import_manual_masks(
            participant_id=self.participant_id,
            root_dir=root_dir,
            mask_paths=mask_paths,
            mask_names=names,
            ref_file=ref_file,
            mask_space=mask_space,
            anat_file=anat_file,
            overwrite=self.overwrite,
            qc_enabled=True,
            qc_dir=qc_dir,
            progress_callback=progress_callback,
        )

    def _run_mask_extraction(self) -> None:
        """Extract network masks using ICA (rest), GLM contrasts (task), or manual import."""
        from pineuro.mask_extraction import mask_extraction_rest

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"
        mask_extraction_dir = base_dir / "mask_extraction"

        localizer_type = config.get("mask", {}).get("localizer_type", "rest")
        localizer_task_type = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
        task_name = localizer_task_type if localizer_task_type != "rest" else "rest"

        # Input files from preprocessing
        preproc_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_preproc.nii.gz"
        mask_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_preproc_mask.nii.gz"
        ref_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_boldref.nii.gz"

        # Set up progress callback for mask extraction to show in GUI log console
        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        # Manual import — user-supplied masks; no preprocessing inputs needed.
        loc0 = config["localizers"][0] if config.get("localizers") else {}
        if loc0.get("type") == "manual":
            from pineuro.gui.components.manual_masks import read_manual_masks

            self._emit_progress("Using manual mask import")
            self._run_manual_mask_import(
                loc0,
                base_dir=base_dir,
                root_dir=mask_extraction_dir,
                qc_dir=base_dir / "qc",
                progress_callback=progress_callback,
            )
            manual_record = read_manual_masks(base_dir, loc0.get("id", "manual")) or {}
            self._create_tissue_masks_if_enabled(
                mask_extraction_dir,
                manual_record.get("reference"),
                progress_callback,
            )
            return

        if localizer_type == "task":
            from pineuro.mask_extraction import mask_extraction_task

            self._emit_progress("Using task-based (GLM) mask extraction")

            contrasts = config.get("mask", {}).get("contrasts", {})

            # Fall back to task metadata's default_contrasts if not in config
            localizer_task_type = config["localizers"][0].get("type", "") if config.get("localizers") else ""
            if not contrasts and localizer_task_type:
                from pineuro.tasks.task_registry import get_task_metadata

                try:
                    metadata = get_task_metadata(localizer_task_type)
                    contrasts = metadata.default_contrasts or {}
                    if contrasts:
                        self._emit_progress(f"Using default contrasts from {metadata.display_name}: {contrasts}")
                except KeyError:
                    pass

            # Auto-resolve events file from localizer task output
            events_dir = base_dir / "preprocessing"
            events_glob = sorted(events_dir.glob(f"*_task-{localizer_task_type}_*_events.tsv"))
            if events_glob:
                # If acqs were selected, prefer events file matching selected acq
                events_file = None
                if self._selected_acqs:
                    for run_num, acq_nums in self._selected_acqs.items():
                        for acq_num in acq_nums:
                            acq_pattern = f"acq-{acq_num:02d}_run-{run_num:02d}"
                            for ef in events_glob:
                                if acq_pattern in ef.name:
                                    events_file = str(ef)
                                    break
                            if events_file:
                                break
                        if events_file:
                            break
                if events_file is None:
                    events_file = str(events_glob[-1])  # Default: latest
                self._emit_progress(f"Found events file: {Path(events_file).name}")
            else:
                raise FileNotFoundError(
                    f"No events file found in {events_dir} for task '{localizer_task_type}'. "
                    f"Run the localizer task first to generate the events TSV."
                )

            tr = config["localizers"][0].get("tr", 1.2) if config.get("localizers") else 1.2
            threshold = config.get("mask", {}).get("threshold", 2000)
            smooth_fwhm = (
                config["localizers"][0].get("extraction", {}).get("smooth_fwhm", 5.0)
                if config.get("localizers")
                else 5.0
            )
            highpass = (
                config["localizers"][0].get("extraction", {}).get("highpass", 0.01)
                if config.get("localizers")
                else 0.01
            )
            threshold_method = config.get("mask", {}).get("threshold_method", "top_n")
            atlas_mask_cfg = config.get("mask", {}).get("atlas_mask")
            atlas_name_cfg = config.get("mask", {}).get("atlas_name")
            atlas_source_cfg = config.get("mask", {}).get("atlas_source")
            atlas_regions_cfg = config.get("mask", {}).get("atlas_regions")
            cluster_start_threshold = config.get("mask", {}).get("cluster_start_threshold", 2.0)
            cluster_step = config.get("mask", {}).get("cluster_step", 0.1)
            cluster_return_all = config.get("mask", {}).get("cluster_return_all", False)
            min_cluster_size = config.get("mask", {}).get("min_cluster_size", 50)
            max_total_voxels = config.get("mask", {}).get("max_total_voxels")

            include_reverse = config.get("mask", {}).get("include_reverse_contrasts", False)

            _mask_files = mask_extraction_task(
                participant_id=self.participant_id,
                root_dir=mask_extraction_dir,
                input_file=preproc_file,
                mask_file=mask_file,
                events=events_file,
                contrasts=contrasts,
                tr=tr,
                threshold=threshold,
                threshold_method=threshold_method,
                atlas_mask=atlas_mask_cfg,
                atlas_name=atlas_name_cfg,
                atlas_source=atlas_source_cfg,
                atlas_regions=atlas_regions_cfg,
                ref_file=str(ref_file),
                cluster_start_threshold=cluster_start_threshold,
                cluster_step=cluster_step,
                cluster_return_all=cluster_return_all,
                min_cluster_size=min_cluster_size,
                max_total_voxels=max_total_voxels,
                include_reverse=include_reverse,
                overwrite=self.overwrite,
                smoothing_fwhm=smooth_fwhm,
                high_pass=highpass,
                qc_enabled=True,
                progress_callback=progress_callback,
            )
        else:
            # ICA-based extraction (existing behavior)
            n_yeo = config["localizers"][0]["extraction"]["n_yeo"]
            network_atlas = config["localizers"][0].get("extraction", {}).get("network_atlas", f"yeo_{n_yeo}")
            yeo_targets = list(config["localizers"][0].get("networks", []))  # Copy to avoid modifying original
            n_components = (
                config["localizers"][0].get("extraction", {}).get("n_components", 30)
                if config.get("localizers")
                else 30
            )
            threshold = config.get("mask", {}).get("threshold", 2000)
            smooth_fwhm = config["localizers"][0]["extraction"]["smooth_fwhm"]
            highpass = config["localizers"][0]["extraction"]["highpass"]
            lowpass = config["localizers"][0].get("extraction", {}).get("lowpass")

            # If sham is enabled with alternative masks, include those networks too
            sham_config = config.get("sham", {})
            sham_hemisphere_network = None
            if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
                mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
                if mask_mode == "hemisphere_split":
                    hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                    if hemisphere_network and hemisphere_network not in yeo_targets:
                        yeo_targets.append(hemisphere_network)
                        if self._is_blind_mode():
                            self._emit_progress_file_only(f"Including sham hemisphere network: {hemisphere_network}")
                            self._emit_progress("Including additional network for mask extraction")
                        else:
                            self._emit_progress(f"Including sham hemisphere network: {hemisphere_network}")
                    sham_hemisphere_network = hemisphere_network
                else:
                    alternative_targets = sham_config.get("alternative_mask_targets", [])
                    for target in alternative_targets:
                        if target and target not in yeo_targets:
                            yeo_targets.append(target)
                            if self._is_blind_mode():
                                self._emit_progress_file_only(f"Including sham alternative network: {target}")
                                self._emit_progress("Including additional network for mask extraction")
                            else:
                                self._emit_progress(f"Including sham alternative network: {target}")

            _mask_files = mask_extraction_rest(
                participant_id=self.participant_id,
                root_dir=mask_extraction_dir,
                input_file=preproc_file,
                mask_file=mask_file,
                ref_file=ref_file,
                n_yeo=n_yeo,
                yeo_targets=yeo_targets,
                network_atlas=network_atlas,
                n_components=n_components,
                threshold=threshold,
                smooth_fwhm=smooth_fwhm,
                highpass=highpass,
                lowpass=lowpass,
                overwrite=self.overwrite,
                qc_enabled=True,
                progress_callback=progress_callback,
                sham_hemisphere_network=sham_hemisphere_network,
            )

        # Create WM/CSF tissue masks for nuisance regression (if enabled)
        self._create_tissue_masks_if_enabled(mask_extraction_dir, ref_file, progress_callback)

    def _create_tissue_masks_if_enabled(
        self,
        mask_extraction_dir: Path,
        ref_file: str | Path | None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """Create WM/CSF tissue masks for nuisance regression when enabled."""
        fb_cfg = self.config.get("feedback", {})
        if not (fb_cfg.get("include_wm_regressor", False) or fb_cfg.get("include_csf_regressor", False)):
            return
        if not ref_file:
            self._emit_progress("WM/CSF regressors enabled but no reference volume available; skipping tissue masks")
            return

        from pineuro.mask_extraction import create_tissue_masks

        self._emit_progress("Creating WM/CSF tissue masks for nuisance regression...")
        tissue_dir = mask_extraction_dir / "05_tissue"

        # Try to reuse MNI inverse transforms from the ICA/MNI pathway's 02_mnireg/
        inv_xfm = None
        mnireg_dir = mask_extraction_dir / "02_mnireg"
        if mnireg_dir.exists():
            inv_files = sorted(mnireg_dir.glob("*_inv_*"))
            if inv_files:
                inv_xfm = [str(f) for f in inv_files]
                self._emit_progress(f"Reusing MNI inverse transforms from {mnireg_dir.name}/")

        create_tissue_masks(
            ref_file=str(ref_file),
            output_dir=tissue_dir,
            participant_id=self.participant_id,
            inv_transform_files=inv_xfm,
            overwrite=self.overwrite,
            progress_callback=progress_callback,
        )
        self._emit_progress("WM/CSF tissue masks created")

    # ------------------------------------------------------------------
    # Multi-localizer pipeline methods
    # ------------------------------------------------------------------

    def _get_localizer_config(self, loc_idx: int) -> dict:
        """Return the localizer config dict at the given index."""
        localizers = self.config.get("localizers", [])
        if loc_idx < len(localizers):
            result: dict = localizers[loc_idx]
            return result
        raise IndexError(f"No localizer config at index {loc_idx}")

    def _run_localizer_collection(self, loc_idx: int, run_num: int) -> None:
        """Collect DICOMs for a specific localizer run.

        Delegates to :meth:`_run_rest_run` after temporarily overriding the
        relevant config fields so the existing collection logic uses the
        correct parameters for this localizer.
        """
        # Temporarily swap the active localizer to index 0 so the downstream
        # logic (which reads `config["localizers"][0]`) sees the right localizer.
        config = self.config
        localizers = config.get("localizers", [])
        if loc_idx == 0 or loc_idx >= len(localizers):
            # Nothing to swap — the active localizer is already index 0 (or invalid).
            self._run_rest_run(run_num)
            return

        localizers[0], localizers[loc_idx] = localizers[loc_idx], localizers[0]
        try:
            self._run_rest_run(run_num)
        finally:
            localizers[0], localizers[loc_idx] = localizers[loc_idx], localizers[0]

    def _run_localizer_preprocessing(self, loc_idx: int) -> None:
        """Run preprocessing for a specific localizer."""
        from pineuro.preprocessing import preprocessing

        loc = self._get_localizer_config(loc_idx)
        loc_type = loc.get("type", "rest")
        task_name = "rest" if loc_type == "rest" else loc_type

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"
        preprocessing_dir = base_dir / "preprocessing"

        n_runs = loc.get("n_runs", 2)
        n_volumes = loc.get("n_volumes", 250)

        self._emit_progress(f"Preprocessing localizer {loc_idx}: {task_name}")

        # Find available collection directories for this localizer
        runs_to_process = []
        for run_num in range(1, n_runs + 1):
            collections = self.find_available_collections(
                collection_dir=collection_dir,
                task=f"task-{task_name}",
                run=run_num,
                expected_volumes=n_volumes,
            )
            if collections:
                runs_to_process.append(run_num)

        # Select acquisitions
        rest_volumes = [n_volumes] * n_runs
        input_dicoms = self._resolve_acq_paths(
            task_name=task_name,
            runs_to_process=runs_to_process,
            rest_volumes=rest_volumes,
        )

        if not input_dicoms:
            raise ValueError(f"No {task_name} collection directories found for preprocessing")

        for path in input_dicoms:
            self._emit_progress(f"Using: {path.name}")

        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        preprocessing(
            participant_id=self.participant_id,
            task=task_name,
            root_dir=preprocessing_dir,
            input_dicoms=[str(d) for d in input_dicoms],
            overwrite=self.overwrite,
            qc_enabled=True,
            progress_callback=progress_callback,
        )

    def _run_localizer_extraction(self, loc_idx: int) -> None:
        """Run mask extraction for a specific localizer.

        Output goes to ``mask_extraction/loc_{idx}_{task_name}/`` and
        results are copied to the shared ``mask_extraction/03_masks/``.
        """
        import shutil

        loc = self._get_localizer_config(loc_idx)
        loc_type = loc.get("type", "rest")
        task_name = "rest" if loc_type == "rest" else loc_type

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"
        mask_extraction_dir = base_dir / "mask_extraction"
        loc_dir = mask_extraction_dir / f"loc_{loc_idx}_{task_name}"

        preproc_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_preproc.nii.gz"
        mask_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_preproc_mask.nii.gz"
        ref_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_boldref.nii.gz"

        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        self._emit_progress(f"Extracting masks for localizer {loc_idx}: {task_name}")

        if loc_type == "manual":
            mask_files = self._run_manual_mask_import(
                loc,
                base_dir=base_dir,
                root_dir=loc_dir,
                qc_dir=base_dir / "qc" / f"loc_{loc_idx}_manual",
                progress_callback=progress_callback,
            )
        elif loc_type == "rest":
            from pineuro.mask_extraction import mask_extraction_rest

            n_yeo = loc.get("n_yeo", 17)
            network_atlas = loc.get("network_atlas", f"yeo_{n_yeo}")
            yeo_targets = list(loc.get("yeo_targets", ["DefaultA", "ContA"]))
            threshold = loc.get("threshold", 2000)
            smooth_fwhm = loc.get("smooth_fwhm", 5.0)
            highpass = loc.get("highpass", 0.01)
            lowpass = loc.get("lowpass")

            # Include sham networks for rest localizers
            sham_config = config.get("sham", {})
            sham_hemisphere_network = None
            if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
                mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
                if mask_mode == "hemisphere_split":
                    hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                    if hemisphere_network and hemisphere_network not in yeo_targets:
                        yeo_targets.append(hemisphere_network)
                    sham_hemisphere_network = hemisphere_network
                else:
                    for target in sham_config.get("alternative_mask_targets", []):
                        if target and target not in yeo_targets:
                            yeo_targets.append(target)

            mask_files = mask_extraction_rest(
                participant_id=self.participant_id,
                root_dir=loc_dir,
                input_file=preproc_file,
                mask_file=mask_file,
                ref_file=ref_file,
                n_yeo=n_yeo,
                yeo_targets=yeo_targets,
                network_atlas=network_atlas,
                threshold=threshold,
                smooth_fwhm=smooth_fwhm,
                highpass=highpass,
                lowpass=lowpass,
                overwrite=self.overwrite,
                qc_enabled=True,
                progress_callback=progress_callback,
                sham_hemisphere_network=sham_hemisphere_network,
            )
        else:
            from pineuro.mask_extraction import mask_extraction_task

            contrasts = loc.get("contrasts", {})
            if not contrasts:
                from pineuro.tasks.task_registry import get_task_metadata

                try:
                    metadata = get_task_metadata(loc_type)
                    contrasts = metadata.default_contrasts or {}
                    if contrasts:
                        self._emit_progress(f"Using default contrasts from {metadata.display_name}: {contrasts}")
                except KeyError:
                    pass

            # Auto-resolve events file
            events_dir = base_dir / "preprocessing"
            events_glob = sorted(events_dir.glob(f"*_task-{loc_type}_*_events.tsv"))
            if events_glob:
                events_file = str(events_glob[-1])
                self._emit_progress(f"Found events file: {Path(events_file).name}")
            else:
                raise FileNotFoundError(
                    f"No events file found in {events_dir} for task '{loc_type}'. "
                    f"Run the localizer task first to generate the events TSV."
                )

            mask_files = mask_extraction_task(
                participant_id=self.participant_id,
                root_dir=loc_dir,
                input_file=preproc_file,
                mask_file=mask_file,
                events=events_file,
                contrasts=contrasts,
                tr=loc.get("tr", 1.2),
                threshold=loc.get("threshold", 2000),
                threshold_method=loc.get("threshold_method", "top_n"),
                atlas_mask=loc.get("atlas_mask"),
                atlas_name=loc.get("atlas_name"),
                atlas_source=loc.get("atlas_source"),
                atlas_regions=loc.get("atlas_regions"),
                ref_file=str(ref_file),
                cluster_start_threshold=loc.get("cluster_start_threshold", 2.0),
                cluster_step=loc.get("cluster_step", 0.1),
                cluster_return_all=loc.get("cluster_return_all", False),
                min_cluster_size=loc.get("min_cluster_size", 50),
                max_total_voxels=loc.get("max_total_voxels"),
                include_reverse=loc.get("include_reverse_contrasts", False),
                overwrite=self.overwrite,
                smoothing_fwhm=loc.get("smooth_fwhm", 5.0),
                high_pass=loc.get("highpass", 0.01),
                drift_model=loc.get("drift_model", "cosine"),
                hrf_model=loc.get("hrf_model", "spm"),
                qc_enabled=True,
                progress_callback=progress_callback,
            )

        # Copy masks to shared 03_masks/
        shared_mask_dir = mask_extraction_dir / "03_masks"
        shared_mask_dir.mkdir(parents=True, exist_ok=True)
        for f in mask_files:
            dest = shared_mask_dir / f.name
            shutil.copy2(f, dest)
            self._emit_progress(f"Copied {f.name} -> 03_masks/")

    def _find_best_reference(self) -> Path:
        """Find the best available reference image for MNI registration.

        Priority:
        1. boldref from preprocessing (best quality, brain-extracted)
        2. Mean of initial reference DICOMs (converted to NIfTI)
        3. Mean of realtime reference DICOMs (for 0-localizer MNI atlas mode)

        Returns:
            Path to reference NIfTI file

        Raises:
            FileNotFoundError: If no suitable reference is available
        """
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        localizer_task_type = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
        task_name = localizer_task_type if localizer_task_type != "rest" else "rest"

        # Option 1: boldref from preprocessing
        preprocessing_dir = base_dir / "preprocessing"
        boldref = preprocessing_dir / f"{self.participant_id}_task-{task_name}_boldref.nii.gz"
        if boldref.exists():
            self._emit_progress(f"Using boldref from preprocessing: {boldref.name}")
            return boldref

        collection_dir = base_dir / "collection"
        mni_ref_dir = base_dir / "mask_extraction" / "00_mni_ref"

        # Option 2: Initial reference DICOMs → convert to NIfTI → compute mean
        ref_dirs = sorted(collection_dir.glob("task-reference_acq-*_run-01"))
        for ref_dir in reversed(ref_dirs):  # newest acq first
            dicoms = list(ref_dir.iterdir())
            if dicoms:
                from pineuro.utils import compute_mean, dicom_to_nifti

                mni_ref_dir.mkdir(parents=True, exist_ok=True)

                mean_file = mni_ref_dir / f"{self.participant_id}_task-reference_mean.nii.gz"
                if mean_file.exists():
                    self._emit_progress(f"Using existing reference mean: {mean_file.name}")
                    return mean_file

                self._emit_progress(f"Converting reference DICOMs from {ref_dir.name}...")
                nifti_file = dicom_to_nifti(
                    dicom_dir=ref_dir,
                    output_dir=mni_ref_dir,
                    output_name=f"{self.participant_id}_task-reference",
                )
                nifti_path = Path(nifti_file)
                if nifti_path.exists():
                    self._emit_progress("Computing mean reference volume...")
                    compute_mean(input_file=nifti_path, output_file=mean_file)
                    if mean_file.exists():
                        return mean_file

        # Option 3: Realtime reference DICOMs (for 0-localizer MNI atlas mode)
        rt_ref_dirs = sorted(collection_dir.glob("task-reference_acq-*_run-02"))
        for ref_dir in reversed(rt_ref_dirs):  # newest acq first
            dicoms = list(ref_dir.iterdir())
            if dicoms:
                from pineuro.utils import compute_mean, dicom_to_nifti

                mni_ref_dir.mkdir(parents=True, exist_ok=True)

                mean_file = mni_ref_dir / f"{self.participant_id}_task-reference-realtime_mean.nii.gz"
                if mean_file.exists():
                    self._emit_progress(f"Using existing realtime reference mean: {mean_file.name}")
                    return mean_file

                self._emit_progress(f"Converting realtime reference DICOMs from {ref_dir.name}...")
                nifti_file = dicom_to_nifti(
                    dicom_dir=ref_dir,
                    output_dir=mni_ref_dir,
                    output_name=f"{self.participant_id}_task-reference-realtime",
                )
                nifti_path = Path(nifti_file)
                if nifti_path.exists():
                    self._emit_progress("Computing mean realtime reference volume...")
                    compute_mean(input_file=nifti_path, output_file=mean_file)
                    if mean_file.exists():
                        return mean_file

        raise FileNotFoundError(
            "No reference image available for MNI registration. Collect a reference scan or run preprocessing first."
        )

    def _run_mni_mask_extraction(self) -> None:
        """Extract network masks from MNI atlas (no localizer/ICA required).

        This is the fallback pathway that registers a network atlas from MNI
        space to the participant's reference scan and extracts binary masks.
        """
        from pineuro.mask_extraction import mask_extraction_mni

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"

        # Find the best available reference image
        ref_file = self._find_best_reference()

        # Read atlas/network config (same settings used for ICA pipeline)
        mask_cfg = config.get("mask", {})
        n_yeo = mask_cfg.get("n_yeo", 17)
        network_atlas = mask_cfg.get("network_atlas", f"yeo_{n_yeo}")
        yeo_targets = list(mask_cfg.get("yeo_targets", ["DefaultA", "ContA"]))
        threshold = mask_cfg.get("threshold", 2000)

        # Handle sham networks (same logic as _run_mask_extraction)
        sham_config = config.get("sham", {})
        sham_hemisphere_network = None
        if sham_config.get("enabled", False) and sham_config.get("delivery_method") == "alternative_masks":
            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
            if mask_mode == "hemisphere_split":
                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                if hemisphere_network and hemisphere_network not in yeo_targets:
                    yeo_targets.append(hemisphere_network)
                    if self._is_blind_mode():
                        self._emit_progress_file_only(f"Including sham hemisphere network: {hemisphere_network}")
                        self._emit_progress("Including additional network for mask extraction")
                    else:
                        self._emit_progress(f"Including sham hemisphere network: {hemisphere_network}")
                sham_hemisphere_network = hemisphere_network
            else:
                alternative_targets = sham_config.get("alternative_mask_targets", [])
                for target in alternative_targets:
                    if target and target not in yeo_targets:
                        yeo_targets.append(target)
                        if self._is_blind_mode():
                            self._emit_progress_file_only(f"Including sham alternative network: {target}")
                            self._emit_progress("Including additional network for mask extraction")
                        else:
                            self._emit_progress(f"Including sham alternative network: {target}")

        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        _mask_files = mask_extraction_mni(
            participant_id=self.participant_id,
            root_dir=mask_extraction_dir,
            ref_file=ref_file,
            n_yeo=n_yeo,
            yeo_targets=yeo_targets,
            network_atlas=network_atlas,
            threshold=threshold,
            overwrite=self.overwrite,
            qc_enabled=True,
            progress_callback=progress_callback,
            sham_hemisphere_network=sham_hemisphere_network,
        )

        # Create WM/CSF tissue masks for nuisance regression (if enabled)
        fb_cfg = config.get("feedback", {})
        if fb_cfg.get("include_wm_regressor", False) or fb_cfg.get("include_csf_regressor", False):
            from pineuro.mask_extraction import create_tissue_masks

            self._emit_progress("Creating WM/CSF tissue masks for nuisance regression...")
            tissue_dir = mask_extraction_dir / "05_tissue"

            # Reuse MNI inverse transforms from 02_mnireg/
            inv_xfm = None
            mnireg_dir = mask_extraction_dir / "02_mnireg"
            if mnireg_dir.exists():
                inv_files = sorted(mnireg_dir.glob("*_inv_*"))
                if inv_files:
                    inv_xfm = [str(f) for f in inv_files]
                    self._emit_progress(f"Reusing MNI inverse transforms from {mnireg_dir.name}/")

            create_tissue_masks(
                ref_file=str(ref_file),
                output_dir=tissue_dir,
                participant_id=self.participant_id,
                inv_transform_files=inv_xfm,
                overwrite=self.overwrite,
                progress_callback=progress_callback,
            )
            self._emit_progress("WM/CSF tissue masks created")

    def _run_coregistration(self, run_id: int = 1) -> None:
        """Coregister masks to reference feedback space.

        Args:
            run_id: Run number (1 for initial, 2 for realtime)
        """
        from pineuro.mask_extraction import coregister_reference, generate_reference_qc_overlays

        config = self.config

        # Check if reference is enabled for this run
        if run_id == 1:
            if not config.get("reference", {}).get("initial_enabled", True):
                self._emit_progress("Reference initial coregistration is disabled, skipping...")
                return
        else:
            if not config.get("reference", {}).get("realtime_enabled", True):
                self._emit_progress("Reference realtime coregistration is disabled, skipping...")
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

        # Determine actual reference run number
        # In mask reuse mode, realtime (run_id=2 from caller) uses reference run-01 (since no initial reference)
        # In standard mode, run_id directly maps to reference run number
        if is_mask_reuse and run_id == 2:
            actual_reference_run = 1
        else:
            actual_reference_run = run_id

        run_str = f"run-{actual_reference_run:02d}"
        self._emit_progress(f"Coregistering network masks to reference {run_str}...")

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

            localizer_task = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
            task_name = localizer_task if localizer_task != "rest" else "rest"
            ref_file = source_preprocessing_dir / f"{self.participant_id}_task-{task_name}_boldref.nii.gz"
            masks_dir = source_mask_extraction_dir / "03_masks"

            self._emit_progress(f"Using masks from source visit: {mask_source_visit}")
        else:
            # Standard mode: get masks and ref_file from current visit
            loc0 = config["localizers"][0] if config.get("localizers") else {}
            localizer_task = loc0.get("type", "rest")
            if localizer_task == "manual":
                # Manual localizer: masks were imported against the participant's
                # user-supplied reference EPI — there is no preprocessing boldref.
                ref_file = self._manual_reference_file(base_dir, mask_extraction_dir, loc0)
            else:
                task_name = localizer_task if localizer_task != "rest" else "rest"
                ref_file = preprocessing_dir / f"{self.participant_id}_task-{task_name}_boldref.nii.gz"
            masks_dir = mask_extraction_dir / "03_masks"

        # Find the most recent complete reference collection for this run
        n_volumes = config.get("reference", {}).get("initial_n_volumes", 2)
        collections = self.find_available_collections(
            collection_dir=base_dir / "collection",
            task="task-reference",
            run=actual_reference_run,
            expected_volumes=n_volumes,
        )
        if not collections:
            raise ValueError(f"No reference collection found for run {actual_reference_run}")

        # Use highest complete acq (or highest incomplete if none complete)
        complete = [c for c in collections if c["is_complete"]]
        selected = complete[0] if complete else collections[0]
        reference_dicom = selected["path"]

        if not selected["is_complete"]:
            logger.warning(f"Using incomplete reference collection: {selected['n_volumes']}/{n_volumes} volumes")

        # Initial reference (run-01) goes to mask_extraction, realtime (run-02+) goes to feedback
        if run_id == 1:
            root_dir = mask_extraction_dir
            output_subdir = "04_reference"
        else:
            root_dir = feedback_dir
            output_subdir = "01_reference"

        # Build mask-to-localizer map for multi-localizer coregistration
        localizers = config.get("localizers", [])
        is_multi_localizer = len(localizers) > 1 and not is_mask_reuse

        if is_multi_localizer:
            # Multi-localizer: each localizer's masks need its own boldref
            # Build mask name -> localizer index mapping
            mask_name_to_loc: dict[str, int] = {}
            for loc_idx, loc in enumerate(localizers):
                loc_type = loc.get("type", "rest")
                if loc_type == "rest":
                    for target in loc.get("yeo_targets", []):
                        mask_name_to_loc[target] = loc_idx
                elif loc_type == "manual":
                    for mask_name in loc.get("manual_mask_names", []):
                        mask_name_to_loc[mask_name] = loc_idx
                else:
                    for mask_name in loc.get("contrasts", {}):
                        mask_name_to_loc[mask_name] = loc_idx

            # Group masks by localizer, each with its own boldref
            # Masks live in shared 03_masks/ but originated from different localizer spaces
            all_mask_files = list(masks_dir.glob(f"{self.participant_id}_*_thr.nii.gz"))

            # Filter sham hemisphere masks
            sham_config = config.get("sham", {})
            if sham_config.get("enabled", False):
                delivery_method = sham_config.get("delivery_method", "alternative_masks")
                mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
                if delivery_method == "alternative_masks" and mask_mode == "hemisphere_split":
                    hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                    combined_mask_name = f"{self.participant_id}_{hemisphere_network}_thr.nii.gz"
                    all_mask_files = [f for f in all_mask_files if f.name != combined_mask_name]

            # Map each mask file to its localizer index by matching mask names
            loc_groups: dict[int, list[Path]] = {}
            unmatched: list[Path] = []
            prefix = f"{self.participant_id}_"
            suffix = "_thr.nii.gz"
            for mask_file in all_mask_files:
                name = mask_file.name
                if name.startswith(prefix) and name.endswith(suffix):
                    mask_name = name[len(prefix) : -len(suffix)]
                else:
                    mask_name = name
                matched_loc_idx = mask_name_to_loc.get(mask_name)
                if matched_loc_idx is not None:
                    loc_groups.setdefault(matched_loc_idx, []).append(mask_file)
                else:
                    unmatched.append(mask_file)

            # Tissue masks go with the first localizer (they're in the same space)
            tissue_mask_files = []
            tissue_masks_dir = mask_extraction_dir / "05_tissue"
            if tissue_masks_dir.exists():
                tissue_mask_files = list(tissue_masks_dir.glob(f"{self.participant_id}_tissue_*_mask.nii.gz"))
                if tissue_mask_files:
                    self._emit_progress(f"Including {len(tissue_mask_files)} tissue masks for coregistration")

            # Coregister each localizer group with its own boldref
            all_coreg_masks: list[Path] = []
            for loc_idx, mask_files in sorted(loc_groups.items()):
                loc = localizers[loc_idx]
                loc_type = loc.get("type", "rest")
                loc_task_name = "rest" if loc_type == "rest" else loc_type
                if loc_type == "manual":
                    # Manual localizer has no preprocessing boldref — its masks
                    # were imported against the user-supplied reference EPI.
                    loc_dir = mask_extraction_dir / f"loc_{loc_idx}_manual"
                    loc_ref_file = self._manual_reference_file(base_dir, loc_dir, loc)
                else:
                    loc_ref_file = preprocessing_dir / f"{self.participant_id}_task-{loc_task_name}_boldref.nii.gz"

                # Add tissue masks to the first group only
                group_files = [str(f) for f in mask_files]
                if loc_idx == min(loc_groups.keys()) and tissue_mask_files:
                    group_files += [str(f) for f in tissue_mask_files]
                if unmatched and loc_idx == min(loc_groups.keys()):
                    group_files += [str(f) for f in unmatched]
                    unmatched = []

                self._emit_progress(
                    f"Coregistering {len(group_files)} masks from localizer {loc_idx} "
                    f"({loc_task_name}) with its boldref..."
                )

                coreg_result = coregister_reference(
                    participant_id=self.participant_id,
                    root_dir=root_dir,
                    input_dicom=reference_dicom,
                    network_files=group_files,
                    ref_file=str(loc_ref_file),
                    run_id=actual_reference_run,
                    overwrite=self.overwrite,
                    output_subdir=output_subdir,
                )
                all_coreg_masks.extend(coreg_result)

            coreg_masks = all_coreg_masks
            self._emit_progress(f"Coregistered {len(coreg_masks)} masks across {len(loc_groups)} localizers")
        else:
            # Single localizer (or mask reuse): all masks use the same boldref
            network_files = list(masks_dir.glob(f"{self.participant_id}_*_thr.nii.gz"))

            # Filter out combined hemisphere network mask if using hemisphere_split sham mode
            sham_config = config.get("sham", {})
            if sham_config.get("enabled", False):
                delivery_method = sham_config.get("delivery_method", "alternative_masks")
                mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
                if delivery_method == "alternative_masks" and mask_mode == "hemisphere_split":
                    hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                    combined_mask_name = f"{self.participant_id}_{hemisphere_network}_thr.nii.gz"
                    network_files = [f for f in network_files if f.name != combined_mask_name]

            # Include tissue masks for nuisance regression (WM/CSF) if they exist
            tissue_mask_files = []
            tissue_masks_dir = mask_extraction_dir / "05_tissue"
            if tissue_masks_dir.exists():
                tissue_mask_files = list(tissue_masks_dir.glob(f"{self.participant_id}_tissue_*_mask.nii.gz"))
                if tissue_mask_files:
                    self._emit_progress(f"Including {len(tissue_mask_files)} tissue masks for coregistration")

            all_mask_files_for_coreg = [str(f) for f in network_files] + [str(f) for f in tissue_mask_files]
            self._emit_progress(f"Coregistering {len(all_mask_files_for_coreg)} masks to reference {run_str}...")

            coreg_masks = coregister_reference(
                participant_id=self.participant_id,
                root_dir=root_dir,
                input_dicom=reference_dicom,
                network_files=all_mask_files_for_coreg,
                ref_file=str(ref_file),
                run_id=actual_reference_run,
                overwrite=self.overwrite,
                output_subdir=output_subdir,
            )
            self._emit_progress(f"Coregistered {len(coreg_masks)} masks to reference {run_str}")

        # Generate reference QC overlays for visual approval
        self._emit_progress(f"Generating reference QC overlays for {run_str}...")
        try:
            # Determine session_dir and reference_subdir based on run_id
            # For run-01 (initial): session_dir is mask_extraction, subdir is 04_reference
            # For run-02 (realtime): session_dir is feedback, subdir is 01_reference
            if run_id == 1:
                session_dir = mask_extraction_dir
                reference_subdir = "04_reference"
            else:
                session_dir = feedback_dir
                reference_subdir = "01_reference"

            # For QC overlays, use the mask_extraction_dir where ICA outputs are located
            # In mask reuse mode, this is the source visit's mask_extraction_dir
            qc_mask_extraction_dir = source_mask_extraction_dir if source_mask_extraction_dir else mask_extraction_dir

            generate_reference_qc_overlays(
                participant_id=self.participant_id,
                session_dir=session_dir,
                ref_file=ref_file,
                mask_extraction_dir=qc_mask_extraction_dir,
                network_coreg_files=coreg_masks,
                run_str=str(actual_reference_run).zfill(2),
                n_yeo=(
                    config["localizers"][0].get("extraction", {}).get("n_yeo", 17) if config.get("localizers") else 17
                ),
                overwrite=self.overwrite,
                reference_subdir=reference_subdir,
            )
            self._emit_progress(f"QC overlays generated for reference {run_str}")
        except Exception as e:
            logger.warning(f"Reference QC overlay generation failed: {e}")
            self._emit_progress(f"Warning: QC overlay generation failed: {e}")

    def _run_reference_realtime(self) -> None:
        """Collect real-time reference scan before feedback session."""
        from pineuro.preprocessing import collect_dicoms
        from pineuro.session import get_localizer_run_info

        config = self.config

        # Check if reference realtime is enabled
        if not config.get("reference", {}).get("realtime_enabled", True):
            self._emit_progress("Reference realtime collection is disabled, skipping...")
            return

        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        # In mask reuse mode, realtime reference is run-01 (no initial reference)
        # In standard mode, realtime reference is run-02 (after initial reference)
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        run_id = 1 if is_mask_reuse else 2

        # Get number of volumes (use new field, fall back to legacy)
        n_volumes = config.get("reference", {}).get("realtime_n_volumes", 2)

        # Get run info with retry detection
        run_info = get_localizer_run_info(
            collection_dir=collection_dir, task="task-reference", run=run_id, expected_volumes=n_volumes
        )
        acq = run_info["acq"]

        self._emit_progress(f"Collecting real-time reference DICOMs (acq-{acq:02d}, run-{run_id:02d})...")
        if run_info["existing_acqs"]:
            self._emit_progress(f"Previous attempts: {', '.join(f'acq-{a:02d}' for a in run_info['existing_acqs'])}")

        output_dir = collection_dir / f"task-reference_acq-{acq:02d}_run-{run_id:02d}"

        # Determine simulator vs real-time mode
        simulator_enabled = config.get("simulator", {}).get("enabled", True)

        if simulator_enabled:
            # Simulator mode: create temp watch dir and use source DICOMs
            watch_dir = base_dir / "tmp" / "incoming_reference_realtime"
            watch_dir.mkdir(parents=True, exist_ok=True)
            simulator_source_dir = config.get("reference", {}).get("realtime_dicoms", "")

            # Try mapping file if individual field is empty
            if not simulator_source_dir:
                mapping_path = config.get("simulator", {}).get("source_mapping", "")
                if mapping_path and self.participant_id:
                    try:
                        from pineuro.session import SimulatorSourceMapping

                        mapping = SimulatorSourceMapping.from_file(Path(mapping_path))
                        resolved = mapping.get_source(self.participant_id, "reference_realtime")
                        if resolved is not None:
                            simulator_source_dir = str(resolved)
                            logger.info(f"reference_realtime source from mapping: {simulator_source_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to resolve reference_realtime from mapping: {e}")

            simulator_speed = config.get("simulator", {}).get("collection_speed", 1.0)
        else:
            # Real-time mode: watch scanner DICOM directory
            watch_dir = Path(config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM"))
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
        from pineuro.feedback import get_slot_run_info, run_feedback_session
        from pineuro.sham import (
            ReplayActivationProvider,
            find_feedbacklog_for_replay,
            get_condition_for_participant,
            load_randomization,
        )
        from pineuro.tasks.designs import BlockDesign, NuisanceOnlyDesign
        from pineuro.tasks.session import SessionState
        from pineuro.tasks.task_registry import get_task_metadata

        # Parse step ID to determine task type
        if step_id.startswith("transfer_"):
            task_type = "transfer"
        elif step_id.startswith("feedback_"):
            task_type = "feedback"
        else:
            raise ValueError(f"Invalid feedback task step_id: {step_id}")

        feedback_mode = task_type == "feedback"

        # Resolve per-run config from feedback_runs if available
        frc = self._resolve_feedback_run_config(step_id)
        # Map old-shape flat keys to the new nested sub-block they live in
        # (qc / glm / connectivity / intermittent). For backwards compatibility
        # within this helper, the legacy flat key is also checked at the top level
        # of the per-run dict.
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

        def _fb(key: str, default: Any = None) -> Any:
            """Read a feedback-run setting from the per-run config (new nested schema)."""
            if frc is None:
                return default
            # Direct top-level read (task, max_runs, tr, n_volumes, n_baseline, feedback_mode, etc.)
            if key in frc:
                return frc[key]
            # Look up in nested sub-block
            sub_key = _NESTED_KEY_MAP.get(key)
            if sub_key is not None:
                block, inner = sub_key
                return frc.get(block, {}).get(inner, default)
            return default

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

        # Get mask files - use reference-coregistered masks if realtime reference is enabled,
        # otherwise use boldref space masks from mask extraction
        feedback_task_type = (
            frc.get("task", "ball")
            if frc
            else (config.get("feedback_runs", [{}])[0].get("task", "ball") if config.get("feedback_runs") else "ball")
        )

        # Determine mask names based on localizer type.
        # For task-based (GLM) extraction, mask names come from contrast keys.
        # For ICA-based (rest) extraction, mask names come from mask_names/yeo_targets.
        localizer_type = config["localizers"][0].get("type", "rest") if config.get("localizers") else "rest"
        signal_mode = _fb("feedback_mode", "activation")
        include_reverse = config.get("mask", {}).get("include_reverse_contrasts", False)
        if localizer_type == "task":
            contrasts = config["localizers"][0].get("contrasts", {}) if config.get("localizers") else {}
            real_mask_names = list(contrasts.keys())
            # Fallback to task metadata default_contrasts if config has none
            if not real_mask_names:
                localizer_task_type = config["localizers"][0].get("type") if config.get("localizers") else None
                if localizer_task_type:
                    try:
                        loc_meta = get_task_metadata(localizer_task_type)
                        real_mask_names = list((loc_meta.default_contrasts or {}).keys())
                    except KeyError:
                        pass
            # Append reverse contrast mask names
            if include_reverse:
                real_mask_names = real_mask_names + [f"{n}_reverse" for n in real_mask_names]
        else:
            real_mask_names = [
                name
                for loc in config.get("localizers", [])
                for name in (
                    list(loc.get("networks", []))
                    if loc.get("type") == "rest"
                    else list(loc.get("contrasts", {}).keys())
                )
            ]

        # Resolve task mask names from network assignment in task config.
        # Read per-run task_params from frc.
        metadata = get_task_metadata(feedback_task_type)
        task_specific_config = dict(frc.get("task_params", {})) if frc else {}

        if metadata.network_roles:
            # network_assignment is the single source of truth for role-to-mask mapping.
            # The legacy differential_pair / task_params[role.key] fallbacks were removed
            # along with the old singular config blocks.
            assigned_names = []
            if frc:
                net_assign = frc.get("network_assignment", {})
                for role in metadata.network_roles:
                    name = net_assign.get(role.key)
                    if name:
                        assigned_names.append(name)
            if len(assigned_names) >= 2:
                real_mask_names = assigned_names

        self._emit_progress(f"Task networks: {real_mask_names}")

        reference_realtime_enabled = config.get("reference", {}).get("realtime_enabled", True)

        # Determine reference run number for mask file pattern
        # In mask reuse mode, reference is run-01 (no initial reference)
        # In standard mode, reference is run-02 (after initial reference)
        visit_config = config.get("visit", {})
        is_mask_reuse = visit_config.get("mask_source") == "reuse"
        realtime_reference_run = 1 if is_mask_reuse else 2

        if reference_realtime_enabled:
            # Use reference-coregistered masks
            masks_dir = feedback_dir / "01_reference" / "masks"
            pattern_template = f"*_{{mask_name}}_thr_to_task-reference_run-{realtime_reference_run:02d}_*.nii.gz"
            mask_space = f"reference run-{realtime_reference_run:02d}"
        else:
            # Use boldref space masks from mask extraction
            masks_dir = mask_extraction_dir / "03_masks"
            pattern_template = "*_{mask_name}_thr.nii.gz"
            mask_space = "subject boldref"

        self._emit_progress(f"Using masks from {mask_space} space")

        # Find tissue masks for nuisance regression (WM/CSF)
        wm_mask_path = None
        csf_mask_path = None
        if _fb("include_wm_regressor", False) or _fb("include_csf_regressor", False):
            if reference_realtime_enabled:
                wm_matches = list(
                    masks_dir.glob(f"*_tissue_wm_mask_to_task-reference_run-{realtime_reference_run:02d}_*.nii.gz")
                )
                csf_matches = list(
                    masks_dir.glob(f"*_tissue_csf_mask_to_task-reference_run-{realtime_reference_run:02d}_*.nii.gz")
                )
            else:
                tissue_dir = mask_extraction_dir / "05_tissue"
                wm_matches = list(tissue_dir.glob("*_tissue_wm_mask.nii.gz")) if tissue_dir.exists() else []
                csf_matches = list(tissue_dir.glob("*_tissue_csf_mask.nii.gz")) if tissue_dir.exists() else []
            if wm_matches:
                wm_mask_path = str(wm_matches[0])
            if csf_matches:
                csf_mask_path = str(csf_matches[0])
            if wm_mask_path or csf_mask_path:
                self._emit_progress(f"Using tissue masks for nuisance regression from {mask_space} space")

        # Helper function to find mask file for a given name
        def find_mask_file(mask_name: str) -> Path:
            pattern = pattern_template.format(mask_name=mask_name)
            matches = list(masks_dir.glob(pattern))
            if not matches:
                raise ValueError(
                    f"Could not find mask file for '{mask_name}' in {masks_dir}. Expected pattern: {pattern}"
                )
            if len(matches) > 1:
                logger.warning(f"Multiple mask files found for '{mask_name}', using first: {matches[0]}")
            return matches[0]

        # Find mask files for the real masks
        real_mask_files = [find_mask_file(name) for name in real_mask_names]

        # Collection directory for this run
        collection_dir = base_dir / "collection"

        # Determine simulator vs real-time mode
        simulator_enabled = config.get("simulator", {}).get("enabled", True)

        if simulator_enabled:
            # Simulator mode: create temp watch dir and use source DICOMs
            watch_dir = base_dir / "tmp" / f"incoming_{task_type}_acq-{task_acq:02d}_run-{task_run_number:02d}"
            watch_dir.mkdir(parents=True, exist_ok=True)

            # Get per-run simulator source path based on step_id
            # step_id format: transfer_1, transfer_2, feedback_1, feedback_2, etc.
            run_num_in_step = int(step_id.split("_")[-1])  # Extract run number from step_id
            simulator_source_dir = None

            # 1. Try simulator_source_mapping first (per-subject paths)
            mapping_path = config.get("simulator", {}).get("source_mapping", "")
            if mapping_path and self.participant_id:
                try:
                    from pineuro.session import SimulatorSourceMapping

                    mapping = SimulatorSourceMapping.from_file(Path(mapping_path))
                    run_type = f"{task_type}_run-{run_num_in_step:02d}"
                    resolved = mapping.get_source(self.participant_id, run_type)
                    if resolved is not None:
                        simulator_source_dir = str(resolved)
                        logger.info(f"Feedback source from mapping: {simulator_source_dir}")
                except Exception as e:
                    logger.warning(f"Failed to resolve feedback source from mapping: {e}")

            # 2. Fall back to individual source fields
            if not simulator_source_dir:
                if task_type == "transfer":
                    transfer_sources = [fr.get("transfer_source_dicoms", "") for fr in config.get("feedback_runs", [])]
                    if isinstance(transfer_sources, str):
                        transfer_sources = [transfer_sources, transfer_sources]
                    if run_num_in_step <= len(transfer_sources) and transfer_sources[run_num_in_step - 1]:
                        simulator_source_dir = transfer_sources[run_num_in_step - 1]
                    else:
                        simulator_source_dir = ""
                        if not simulator_source_dir and transfer_sources:
                            simulator_source_dir = transfer_sources[0]
                else:
                    feedback_sources = [fr.get("source_dicoms", "") for fr in config.get("feedback_runs", [])]
                    if isinstance(feedback_sources, str):
                        feedback_sources = [feedback_sources, feedback_sources]
                    if run_num_in_step <= len(feedback_sources) and feedback_sources[run_num_in_step - 1]:
                        simulator_source_dir = feedback_sources[run_num_in_step - 1]
                    else:
                        simulator_source_dir = ""
                        if not simulator_source_dir and feedback_sources:
                            simulator_source_dir = feedback_sources[0]

            simulator_speed = config.get("simulator", {}).get(
                "feedback_speed", config.get("simulator", {}).get("collection_speed", 1.0)
            )
        else:
            # Real-time mode: watch scanner DICOM directory
            watch_dir = Path(config.get("scanner", {}).get("dicom_dir", "/BOLD_DICOM"))
            simulator_source_dir = None
            simulator_speed = 1.0

        self._emit_progress(f"Preparing {task_type} run (feedback={'ON' if feedback_mode else 'OFF'})...")

        # Compute task regressor from task metadata design type
        n_baseline = _fb("n_baseline", 25)
        n_volumes = _fb("n_volumes", 150)
        # Use per-run TR, fall back to scan config
        tr = _fb("tr", 1.2)
        metadata = get_task_metadata(feedback_task_type)
        design: NuisanceOnlyDesign | BlockDesign
        if metadata.design_type == "nuisance_only":
            design = NuisanceOnlyDesign()
        else:  # "block" (default)
            design = BlockDesign(n_baseline)
        task_regressor = design.compute_regressors(n_volumes, tr)

        # Determine sham condition and build complete mask list
        # Compute ALL masks (real + sham alternatives); the task receives
        # the appropriate subset based on condition
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
                        condition_info = get_condition_for_participant(randomization, self.participant_id)
                        condition = condition_info["condition"]
                        if self._is_blind_mode():
                            self._emit_progress_file_only(f"Participant condition: {condition}")
                        else:
                            self._emit_progress(f"Participant condition: {condition}")

                        delivery_method = sham_config.get("delivery_method", "alternative_masks")

                        if delivery_method == "alternative_masks":
                            mask_mode = sham_config.get("sham_mask_mode", "hemisphere_split")
                            if mask_mode == "hemisphere_split":
                                # Hemisphere split: use left/right halves of a single network.
                                # Sham target = {network}_{target_hemisphere},
                                # sham non-target = contralateral hemisphere.
                                hemisphere_network = sham_config.get("hemisphere_network", "SomMotA")
                                target_hemisphere = sham_config.get("target_hemisphere", "right")
                                non_target_hemisphere = "left" if target_hemisphere == "right" else "right"

                                for hemi in [target_hemisphere, non_target_hemisphere]:
                                    alt_name = f"{hemisphere_network}_{hemi}"
                                    try:
                                        sham_mask_files.append(find_mask_file(alt_name))
                                        sham_mask_names.append(alt_name)
                                    except ValueError as e:
                                        logger.warning(f"Sham hemisphere mask not found: {e}")
                            else:
                                # Two networks mode: use two distinct networks
                                alternative_targets = sham_config.get("alternative_mask_targets", [])
                                for alt_name in alternative_targets:
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

                            if self._is_blind_mode():
                                self._emit_progress_file_only(
                                    f"Replay mode: looking up feedbacklog from {matched_participant}"
                                    + (f" visit {matched_visit}" if matched_visit else "")
                                )
                                self._emit_progress("Replay mode: loading activation source")
                            else:
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
                                if self._is_blind_mode():
                                    self._emit_progress_file_only(f"Found feedbacklog: {feedbacklog_path.name}")
                                    self._emit_progress("Activation source loaded")
                                else:
                                    self._emit_progress(f"Found feedbacklog: {feedbacklog_path.name}")

                                # Create replay provider with the real mask names
                                sham_provider = ReplayActivationProvider(
                                    feedbacklog_path=feedbacklog_path,
                                    mask_names=real_mask_names,
                                )
                                if self._is_blind_mode():
                                    self._emit_progress_file_only(
                                        f"Replay provider initialized with {sham_provider._n_source_volumes} volumes"
                                    )
                                    self._emit_progress("Activation replay ready")
                                else:
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

        # Combine all masks: real masks + any additional sham masks.
        # Dedupe by mask name so a mask present in both lists (e.g. the real
        # target happens to equal one of the hemisphere-split sham masks) is
        # only computed once.
        all_mask_files = list(real_mask_files)
        all_mask_names = list(real_mask_names)
        for name, path in zip(sham_mask_names, sham_mask_files, strict=True):
            if name not in all_mask_names:
                all_mask_names.append(name)
                all_mask_files.append(path)
        if self._is_blind_mode():
            self._emit_progress_file_only(f"Computing activations for masks: {all_mask_names}")
            generic_labels = ["Target", "Non-Target"] + [f"Network {i + 1}" for i in range(2, len(all_mask_names))]
            self._emit_progress(f"Computing activations for masks: {generic_labels[: len(all_mask_names)]}")
        else:
            self._emit_progress(f"Computing activations for masks: {all_mask_names}")

        # Determine which masks control the task based on condition.
        # For real: use real_mask_names. For sham: delegate to helper.
        if condition == "sham" and sham_mask_names:
            task_mask_names = _select_sham_task_mask_names(sham_mask_names, real_mask_names)
            if self._is_blind_mode():
                self._emit_progress_file_only(
                    f"Sham: task controlled by {task_mask_names} (instead of {real_mask_names})"
                )
            else:
                self._emit_progress(f"Sham: task controlled by {task_mask_names} (instead of {real_mask_names})")
        else:
            task_mask_names = real_mask_names

        # Build connectivity_pairs, adding sham pair if needed
        connectivity_pairs = _fb("connectivity_pairs")
        if (
            signal_mode == "connectivity"
            and connectivity_pairs
            and sham_mask_names
            and len(sham_mask_names) >= 2
            and sham_config.get("delivery_method") == "alternative_masks"
        ):
            sham_pair = [sham_mask_names[0], sham_mask_names[1]]
            if sham_pair not in connectivity_pairs:
                connectivity_pairs = list(connectivity_pairs) + [sham_pair]

        # Run feedback session (this will handle DICOM collection + processing)
        # Pass ALL masks so all activations are computed and logged
        try:
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
                task_regressors=task_regressor,
                mask_names=all_mask_names,  # All masks for logging
                roi_combine_method=_fb("roi_combine_method", "inverse_sd"),
                simulator_source_dir=simulator_source_dir,
                simulator_speed=simulator_speed,
                simulator_trigger_mode=config.get("simulator", {}).get("trigger_mode", "manual"),
                step_id=step_id,  # Use slot-based naming with acq entity
                qc_enabled=True,
                fd_threshold=_fb("fd_threshold", 0.5),
                condition=condition,  # Log the actual condition
                sham_provider=sham_provider,  # Replay provider for sham participants
                motion_correction=_fb("motion_correction", "siemens_moco"),
                include_drift=_fb("include_drift", True),
                include_motion_regressors=_fb("include_motion_regressors", True),
                include_wm_regressor=_fb("include_wm_regressor", False),
                include_csf_regressor=_fb("include_csf_regressor", False),
                wm_mask_path=wm_mask_path,
                csf_mask_path=csf_mask_path,
                smooth_fwhm=_fb("smooth_fwhm"),  # Real-time spatial smoothing
                differential_pair=self._derive_differential_pair(frc, feedback_task_type),
                feedback_interval=_fb("feedback_interval"),  # Intermittent feedback mode
                connectivity_pairs=connectivity_pairs,  # Sliding-window correlation pairs (may include sham pair)
                connectivity_window=_fb("connectivity_window", 15),  # Correlation window size
                connectivity_target=_fb("connectivity_target", "increase"),  # Target direction
                task_id=feedback_task_type,  # Registered task_id for live-metric QC summary
            )

            # Wait for the coordinator thread to complete
            coordinator = session_info["coordinator"]
            coordinator_thread = session_info["thread"]

            # Track coordinator for stop signal
            self._current_coordinator = coordinator

            # Launch PsychoPy task in separate thread
            self._emit_progress("Launching PsychoPy task...")

            # Import task factory
            from pineuro.tasks import create_task

            # Get task configuration — per-run from frc.
            task_specific_config = frc.get("task_params", {}) if frc else {}

            # Get display configuration
            display_config = config.get("display", {})
            presenter_monitor = display_config.get("presenter_monitor", "right")

            # Load session state for adaptive difficulty persistence across runs
            session_state = SessionState.load(output_dir, task_id=feedback_task_type)

            # Update max_feedback_runs from config if different (sum across all feedback_runs)
            max_feedback_runs = sum(fr.get("max_runs", 1) for fr in config.get("feedback_runs", [])) or 2
            if session_state.max_feedback_runs != max_feedback_runs:
                session_state.max_feedback_runs = max_feedback_runs
                session_state.save()

            self._emit_progress(f"Session state: task={feedback_task_type}, runs={len(session_state.completed_runs)}")

            # Build factory config with task parameters plus runtime settings
            signal_mode = _fb("feedback_mode", "activation")

            connectivity_target = _fb("connectivity_target", "increase")

            factory_config = {
                **task_specific_config,
                "mask_names": task_mask_names,
                "feedback_mode": feedback_mode,
                "signal_mode": signal_mode,
                "connectivity_target": connectivity_target,
                "presenter_monitor": presenter_monitor,
            }

            # Create task via factory (validates config against task metadata)
            simulator = session_info.get("simulator")
            task = create_task(
                task_id=feedback_task_type,
                coordinator=coordinator,
                config=factory_config,
                session_state=session_state,
                output_dir=str(output_dir),
                simulator=simulator,
            )

            # Register callback so coordinator updates task with new activations
            coordinator._activation_callback = task._update_data

            # Track task for stop support
            self._current_feedback_task = task

            # Connect GUI mirror callback if set
            if self._task_mirror_callback is not None:
                task.set_mirror_callback(self._task_mirror_callback)

            # Dispatch task.run() to the main thread (required on macOS for
            # Cocoa NSWindow; used on all platforms for a single code path).
            def run_task():
                try:
                    self._emit_progress("Task started...")
                    task.run()
                    self._emit_progress("Task completed successfully")
                except Exception as e:
                    logger.error(f"Task error: {e}", exc_info=True)
                    self._emit_progress(f"Task error: {e}")

            from pineuro.gui.dispatch import get_dispatcher

            task_future = get_dispatcher().submit(run_task)
            logger.info("Dispatched feedback task to main thread")

            self._emit_progress("Session running, waiting for completion...")

            # Wait for both coordinator and task to complete
            coordinator_thread.join()
            with contextlib.suppress(Exception):
                task_future.result(timeout=30)

            # Extract results generically via task contract
            results = task.get_run_results()
            volumes_processed = results.get("volumes_processed", coordinator.n_processed)

            result_msg = (
                f"{task_type.capitalize()} run {task_run_number} complete! Processed {volumes_processed} volumes."
            )
            extra = {k: v for k, v in results.items() if k != "volumes_processed"}
            if extra:
                result_msg += f" {extra}"

            self._emit_progress(result_msg)
        finally:
            # Clear coordinator and task references
            self._current_coordinator = None
            self._current_feedback_task = None
