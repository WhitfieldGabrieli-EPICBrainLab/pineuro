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


def _loc(config: dict, idx: int = 0) -> dict:
    """Return ``config["localizers"][idx]`` as a dict, or ``{}`` if missing."""
    locs = config.get("localizers") or []
    return locs[idx] if idx < len(locs) else {}


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


def _require_sham_mask(condition: str, mask_name: str, error: Exception) -> None:
    """Handle a sham alternative mask that could not be found.

    For a sham participant a missing mask means the task would silently fall back to
    the real target masks (the original bug), so this raises. The operator-facing
    message is generic — the network name goes only to the log — so the condition is
    not leaked in blind mode. For a real participant the sham masks exist only for
    blinding/logging and never drive the task, so a missing one is merely logged.
    """
    if condition == "sham":
        logger.error(f"Sham mask '{mask_name}' missing: {error}")
        raise ValueError(
            "Sham masks were not coregistered to reference space; re-run reference "
            "coregistration before this feedback run (see log for details)."
        ) from error
    logger.warning(f"Sham mask not found (real participant, ignored): {error}")


def _coreg_localizer_networks(config: dict[str, Any], loc: dict[str, Any]) -> set[str]:
    """Network names whose ``03_masks`` thresholded masks a localizer must coregister.

    The localizer's own networks/contrasts UNION the sham construction networks
    (hemisphere-split network or alternative targets). Mask extraction adds these same
    sham networks via ``sham_networks_for_localizer``; coregistration MUST mirror that or
    the sham masks (e.g. ``SomMotA_left``/``SomMotA_right``) never reach reference space
    and a sham participant silently receives the real feedback. The combined
    hemisphere-split network itself is dropped separately by the caller (replaced by its
    left/right splits).
    """
    from ..utils import localizer_mask_names, sham_networks_for_localizer

    return set(localizer_mask_names(loc)) | set(sham_networks_for_localizer(config, loc))


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
        session_id: str,
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
            session_id: BIDS session identifier (e.g., 'ses-01')
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
        self.session_id = session_id
        # BIDS prefix matching preprocessing/mask-extraction output naming
        # ({participant}_{session}_task-...); used when resolving pipeline I/O paths.
        self._bids_prefix = f"{participant_id}_{session_id}"
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
        # Tracks the in-flight run/attempt so stop() can return enough info
        # for the workflow to offer Keep/Discard. Cleared when the step finishes.
        self._current_attempt: dict[str, Any] | None = None
        # Guards the four _current_* refs above: written by the worker
        # thread, read by stop() on the UI thread.
        self._task_lock = threading.Lock()
        self._task_mirror_callback: Callable[[dict[str, Any]], None] | None = None
        self.overwrite: bool = False
        self._selected_rest_runs: list | None = None  # Selected runs for preprocessing
        self._selected_attempts: dict[int, list[int]] | None = None  # {run_id: [attempt_nums]}

    def _resolve_feedback_run_config(self, step_id: str) -> dict[str, Any] | None:
        """Return the ``feedback_runs`` entry for *step_id*, or None.

        The mapping is derived from ``config["feedback_runs"]``.  Each entry
        contributes transfer + feedback steps in order, so the cumulative
        step count determines which entry owns a given step_id.
        """
        feedback_runs = self.config.get("feedback_runs")
        if not feedback_runs:
            return None

        from ..utils import resolve_use_transfer_runs

        step_to_frc: dict[str, dict[str, Any]] = {}
        fb_counter = 0
        tf_counter = 0
        for frc in feedback_runs:
            use_transfers = resolve_use_transfer_runs(frc)
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

    def set_selected_attempts(self, attempts: dict[int, list[int]] | None) -> None:
        """Set which attempts to use for each run.

        Args:
            attempts: Dict {run_id: [attempt_nums]}, or None to auto-select highest complete
        """
        self._selected_attempts = attempts

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
            - attempt: int - Attempt number
            - path: Path - Folder path
            - n_volumes: int - Number of volumes present
            - is_complete: bool - True if >= expected_volumes

            Sorted by attempt number (highest first)
        """
        import re

        if not collection_dir.exists():
            return []

        # Lazy import: pineuro.utils pulls numpy/nibabel — keep it out of GUI
        # module-load time (see architecture.md GUI lazy-import rule).
        from pineuro.utils import collection_is_complete, count_volume_files

        pattern = f"{task}_run-{run:02d}_attempt-*"
        collections: list[dict[str, Any]] = []

        for folder in collection_dir.glob(pattern):
            if folder.is_dir():
                match = re.search(r"attempt-(\d+)", folder.name)
                if match:
                    attempt_num = int(match.group(1))
                    collections.append(
                        {
                            "attempt": attempt_num,
                            "path": folder,
                            "n_volumes": count_volume_files(folder),
                            "is_complete": collection_is_complete(folder, expected_volumes),
                        }
                    )

        # Sort by attempt number (highest first)
        collections.sort(key=lambda x: x["attempt"], reverse=True)
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

    def stop(self, show_questions: bool = False) -> dict[str, Any] | None:
        """Signal the current step to stop (if running).

        Parameters
        ----------
        show_questions : bool, optional
            If True and a feedback task is running, show post-run questions
            to the participant before stopping. Default is False.

        Returns
        -------
        dict or None
            Snapshot of the in-flight attempt with keys ``kind``
            (``"localizer"`` | ``"feedback"``), ``subject``, ``task``, ``attempt``,
            ``run`` — for the workflow to offer a Keep/Discard prompt. Returns
            ``None`` if no attempt was active or if the localizer's collection
            directory hasn't been created yet (nothing to discard).
        """
        logger.info(f"Signaling pipeline step to stop (show_questions={show_questions})")
        self._stop_event.set()

        # Snapshot the shared refs under the lock, then act outside it.
        with self._task_lock:
            coordinator = self._current_coordinator
            feedback_task = self._current_feedback_task
            localizer_task = self._current_localizer_task
            attempt = dict(self._current_attempt) if self._current_attempt else None

        # Stop thread-based steps
        if self._thread is not None and self._thread.is_alive():
            # If a feedback task is running, stop the coordinator and the task
            if coordinator is not None:
                logger.info("Stopping feedback coordinator")
                coordinator.stop(show_questions=show_questions)
            if feedback_task is not None:
                logger.info("Stopping feedback task (PsychoPy)")
                feedback_task.stop()
            # If a localizer task is running, stop it
            if localizer_task is not None:
                logger.info("Stopping localizer task")
                localizer_task.stop()

        # Suppress the discard prompt for localizer stops that happened before
        # the collection dir was created (nothing on disk to discard yet).
        if attempt is not None and attempt["kind"] == "localizer":
            base_dir = Path(self.config["paths"]["base_dir"])
            collection_dir = (
                base_dir
                / "collection"
                / f"task-{attempt['task']}_run-{attempt['run']:02d}_attempt-{attempt['attempt']:02d}"
            )
            if not collection_dir.exists():
                return None

        return attempt

    def current_task_has_post_run_questions(self) -> bool:
        """Whether the in-flight feedback task would show post-run questions if stopped now.

        Queries the live task so the answer matches exactly what the stop path
        would display (its ``feedback_mode`` / ``_task_state`` are unchanged by a
        stop, since stopped runs aren't recorded as completed). Returns False when
        no feedback task is running (e.g. localizer steps), so the Stop dialog
        omits the questions checkbox.
        """
        with self._task_lock:
            task = self._current_feedback_task
        if task is None:
            return False
        try:
            return bool(task.has_pending_post_run_questions())
        except Exception:  # never block the stop dialog on a metadata/state hiccup
            logger.exception("Failed to check for pending post-run questions")
            return False

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
            # A stop requested mid-flight (e.g. preprocessing raising
            # PreprocessingStopped at a checkpoint) is a clean cancellation,
            # not a failure — report it as stopped and skip the error panel.
            if self._stop_event.is_set():
                logger.info(f"Step {step_id} stopped by user")
                if self.on_complete:
                    self.on_complete(False, f"Step {step_id} stopped by user")
                return
            logger.error(f"Step {step_id} failed: {e}", exc_info=True)
            if self.on_error:
                self.on_error(e)
            if self.on_complete:
                self.on_complete(False, str(e))
        finally:
            # Clear the in-flight attempt so a later stop() (e.g. on the next
            # step) doesn't return stale info for a step that's already done.
            with self._task_lock:
                self._current_attempt = None

    def _set_current_attempt(self, *, kind: str, subject: str, task: str, attempt: int, run: int) -> None:
        """Record the in-flight run/attempt so stop() can offer Keep/Discard."""
        with self._task_lock:
            self._current_attempt = {
                "kind": kind,
                "subject": subject,
                "task": task,
                "attempt": attempt,
                "run": run,
            }

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
        from pineuro.preprocessing import collect_volumes
        from pineuro.session import get_localizer_run_info

        config = self.config

        # Check if reference initial is enabled
        if not config.get("reference", {}).get("initial_enabled", True):
            self._emit_progress("Reference initial collection is disabled, skipping...")
            return

        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        n_volumes = config.get("reference", {}).get("initial_n_volumes", 2)

        # Get run info with retry detection
        run_info = get_localizer_run_info(collection_dir=collection_dir, task="task-reference", run=1)
        attempt = run_info["attempt"]
        self._set_current_attempt(
            kind="localizer", subject=self.participant_id, task="reference", attempt=attempt, run=1
        )

        self._emit_progress(f"Collecting initial reference DICOMs (attempt-{attempt:02d})...")
        if run_info["existing_attempts"]:
            self._emit_progress(
                f"Previous attempts: {', '.join(f'attempt-{a:02d}' for a in run_info['existing_attempts'])}"
            )

        output_dir = collection_dir / f"task-reference_run-01_attempt-{attempt:02d}"

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

        collect_volumes(
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
        from pineuro.preprocessing import collect_volumes
        from pineuro.session import get_localizer_run_info

        config = self.config
        loc = _loc(config)
        task_name = loc.get("type", "rest")
        is_task_localizer = task_name not in ("rest", "manual")

        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        # Get expected volumes from the localizer entry
        n_volumes = loc.get("n_volumes", 200)

        # If the localizer task declares its own worst-case duration, override
        # the user-configured n_volumes with the computed ceiling. The existing
        # task_done_event mechanism still lets collect_volumes stop early when
        # the task finishes, so this only caps collection — it never forces it
        # to run longer than the task actually does.
        if is_task_localizer:
            from pineuro.tasks.task_registry import get_task_metadata

            meta = get_task_metadata(task_name)
            if meta.compute_max_run_seconds is not None:
                task_params = loc.get("task_params", {})
                tr = loc.get("tr", 1.2)
                computed = math.ceil(meta.compute_max_run_seconds(task_params) / tr)
                self._emit_progress(
                    f"{task_name}: auto-computed n_volumes={computed} "
                    f"(worst-case ceiling; collection ends early on task_done_event)"
                )
                n_volumes = computed

        # Get run info with retry detection
        run_info = get_localizer_run_info(collection_dir=collection_dir, task=f"task-{task_name}", run=run_num)
        attempt = run_info["attempt"]
        self._set_current_attempt(
            kind="localizer", subject=self.participant_id, task=task_name, attempt=attempt, run=run_num
        )

        self._emit_progress(f"Collecting {task_name} run {run_num} (attempt-{attempt:02d}) DICOMs...")
        if run_info["existing_attempts"]:
            self._emit_progress(
                f"Previous attempts: {', '.join(f'attempt-{a:02d}' for a in run_info['existing_attempts'])}"
            )

        output_dir = collection_dir / f"task-{task_name}_run-{run_num:02d}_attempt-{attempt:02d}"

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
            # Clear stale files from previous runs so collect_volumes doesn't
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
        actual_task_id = task_name if is_task_localizer else "rest"

        # Shared event so collect_volumes knows when acquisition actually starts
        trigger_event = threading.Event()

        # Event set when the localizer task's paradigm finishes.
        # For variable-duration tasks (e.g., RIT), this signals collect_volumes
        # to drain remaining volumes and stop, rather than waiting for n_volumes.
        task_done_event = threading.Event()

        # Always launch a localizer task - it controls simulator start timing
        localizer_thread = self._launch_localizer_task(
            run_num,
            actual_task_id,
            watch_dir,
            attempt=attempt,
            simulator=simulator,
            trigger_event=trigger_event,
            task_done_event=task_done_event,
        )

        # collect_volumes just watches the directory - no simulator
        # (simulator is controlled by the task for synchronization)

        # Create QC monitor for real-time motion/quality metrics during collection
        from pineuro.collection_qc import CollectionQCMonitor

        motion_mode = loc.get("motion_correction", "siemens_moco")
        qc_monitor = CollectionQCMonitor(
            log_path=output_dir / "collectionlog.tsv",
            motion_correction=motion_mode,
            on_qc_data=self.on_qc_data,
        )

        collect_volumes(
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
        from pineuro.utils import count_volume_files

        actual_count = count_volume_files(output_dir)
        self._emit_progress(f"Collected {actual_count} volumes to {output_dir}")

        # Stop and wait for localizer task (collection is done, task can end)
        if self._current_localizer_task is not None:
            self._current_localizer_task.stop()
        if localizer_thread is not None and localizer_thread.is_alive():
            self._emit_progress("Waiting for localizer task to complete...")
            localizer_thread.join(timeout=10)
            with self._task_lock:
                self._current_localizer_task = None

    def _launch_localizer_task(
        self,
        run_num: int,
        task_id: str,
        watch_dir: Path,
        attempt: int = 1,
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
            task_id: Task identifier (e.g., "rest", "selfother")
            watch_dir: Directory to watch for first DICOM arrival (auto-trigger)
            attempt: Attempt number (for retry handling)
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

        # Get task-specific parameters from config. Fixed-duration localizers
        # (flagged via TaskMetadata.localizer_params_from_scan) derive their
        # params from the scan parameters; others read from the per-task
        # localizer_task config section.
        from pineuro.tasks.task_registry import get_task_metadata

        try:
            params_from_scan = get_task_metadata(task_id).localizer_params_from_scan
        except Exception:
            params_from_scan = False

        loc = _loc(config)
        if params_from_scan:
            localizer_params = {"duration": loc.get("n_volumes", 250) * loc.get("tr", 2.0)}
        else:
            # Task-localizer parameters live on the localizer entry itself
            localizer_params = loc.get("task_params", {})

        # Output events to the preprocessing directory (survives cleanup)
        events_dir = base_dir / "preprocessing"
        events_dir.mkdir(parents=True, exist_ok=True)

        # Every localizer gets a TaskState: the completed_runs ledger (resume/audit)
        # plus any adaptive per-run snapshot the task records. Lives in a per-task-id
        # subdir of the (shared) preprocessing dir so localizer types don't collide.
        from pineuro.tasks.state import TaskState

        task_state = TaskState.load(events_dir / f"task_state_{task_id}", task_id=task_id)

        # Get presenter monitor from display config
        display_config = config.get("display", {})
        presenter_monitor = display_config.get("presenter_monitor", "right")

        scanner_cfg = config.get("scanner", {}) or {}
        trigger_keys = scanner_cfg.get("trigger_keys") or ["t", "+", "5", "="]

        task = create_localizer_task(
            task_id=task_id,
            output_dir=events_dir,
            participant_id=self.participant_id,
            session_id=self.session_id,
            config=localizer_params,
            task_state=task_state,
            run=run_num,
            attempt=attempt,  # Pass attempt for events file naming
            watch_dir=str(watch_dir),  # Pass watch_dir for auto-trigger on first DICOM
            simulator=simulator,  # Pass simulator for task-controlled start timing
            monitor_position=presenter_monitor,
            trigger_keys=trigger_keys,
        )

        # Set trigger event for timing synchronization with collect_volumes
        if trigger_event is not None:
            task.trigger_event = trigger_event

        # Set mirror callback if available
        if self._task_mirror_callback is not None:
            task.set_mirror_callback(self._task_mirror_callback)

        with self._task_lock:
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

    def _resolve_attempt_paths(
        self, task_name: str, runs_to_process: list[int], rest_volumes: list[int]
    ) -> list[tuple[Path, int, int]]:
        """Resolve collection folder paths using selected attempts.

        Uses self._selected_attempts if set (from GUI dialog), otherwise
        auto-selects highest complete attempt for each run.

        Parameters
        ----------
        task_name : str
            Task name (e.g., 'rest', 'selfother')
        runs_to_process : list[int]
            List of run numbers to process
        rest_volumes : list[int]
            Expected volumes per run

        Returns
        -------
        list[tuple[Path, int, int]]
            ``(path, attempt, run)`` triples in selection order. ``attempt`` and
            ``run`` are the integers the GUI assigned; callers forward them
            to ``preprocessing()`` so intermediate filenames carry the
            ``run-NN_attempt-NN`` identity (and ``{1: [2, 3]}``-style
            multi-attempt selections don't get mis-numbered into run-01/run-02).
        """
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        selected: list[tuple[Path, int, int]] = []
        for run_num in runs_to_process:
            expected_vols = rest_volumes[run_num - 1] if len(rest_volumes) >= run_num else rest_volumes[0]

            # If user selected specific attempts, use those directly
            if self._selected_attempts and run_num in self._selected_attempts:
                attempt_nums = self._selected_attempts[run_num]
                found_any = False
                for attempt_num in attempt_nums:
                    attempt_path = collection_dir / f"task-{task_name}_run-{run_num:02d}_attempt-{attempt_num:02d}"
                    if attempt_path.exists():
                        self._emit_progress(f"Run {run_num}: Using selected attempt-{attempt_num:02d}")
                        selected.append((attempt_path, attempt_num, run_num))
                        found_any = True
                    else:
                        logger.warning(f"Selected attempt-{attempt_num:02d} for run {run_num} not found")
                if found_any:
                    continue

            # Auto-select: find highest complete attempt
            collections = self.find_available_collections(
                collection_dir=collection_dir, task=f"task-{task_name}", run=run_num, expected_volumes=expected_vols
            )

            if not collections:
                logger.warning(f"No collections found for run {run_num}")
                continue

            complete = [c for c in collections if c["is_complete"]]
            picked = complete[0] if complete else collections[0]
            self._emit_progress(f"Run {run_num}: Using attempt-{picked['attempt']:02d} ({picked['n_volumes']} volumes)")
            selected.append((picked["path"], int(picked["attempt"]), run_num))

        return selected

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

        # Select attempts (uses GUI selections if set, else auto-selects)
        resolved = self._resolve_attempt_paths(
            task_name=task_name, runs_to_process=runs_to_process, rest_volumes=rest_volumes
        )

        if not resolved:
            raise ValueError(f"No {task_name} collection directories selected for preprocessing")

        input_paths = [path for path, _attempt, _run in resolved]
        attempt_per_input = [attempt for _path, attempt, _run in resolved]
        run_per_input = [run for _path, _attempt, run in resolved]

        # Log selected attempts
        for path in input_paths:
            self._emit_progress(f"Using: {path.name}")

        # Set up progress callback for preprocessing to show in GUI log console
        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        preproc_file, mask_file, ref_file = preprocessing(
            participant_id=self.participant_id,
            session_id=self.session_id,
            task=task_name,
            root_dir=preprocessing_dir,
            input_dicoms=[str(d) for d in input_paths],
            overwrite=self.overwrite,
            qc_enabled=True,  # Generate QC report
            progress_callback=progress_callback,
            attempt_per_input=attempt_per_input,
            run_per_input=run_per_input,
            stop_event=self._stop_event,
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
            session_id=self.session_id,
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
        from pineuro.mask_extraction import run_mask_extraction_from_config

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        preprocessing_dir = base_dir / "preprocessing"
        mask_extraction_dir = base_dir / "mask_extraction"

        loc0 = _loc(config)
        loc_type = loc0.get("type", "rest")
        task_name = loc_type
        extraction = loc0.get("extraction", {})

        # Input files from preprocessing
        preproc_file = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_preproc.nii.gz"
        mask_file = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_preproc_mask.nii.gz"
        ref_file = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_boldref.nii.gz"

        # Set up progress callback for mask extraction to show in GUI log console
        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        # Manual import — user-supplied masks; no preprocessing inputs needed.
        if loc_type == "manual":
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

        if loc_type != "rest":
            self._emit_progress("Using task-based (GLM) mask extraction")

            contrasts = loc0.get("contrasts", {})

            # Auto-resolve events file from localizer task output
            events_dir = base_dir / "preprocessing"
            events_glob = sorted(events_dir.glob(f"*_task-{loc_type}_*_events.tsv"))
            if events_glob:
                # If attempts were selected, prefer events file matching selected attempt
                events_file = None
                if self._selected_attempts:
                    for run_num, attempt_nums in self._selected_attempts.items():
                        for attempt_num in attempt_nums:
                            attempt_pattern = f"run-{run_num:02d}_attempt-{attempt_num:02d}"
                            for ef in events_glob:
                                if attempt_pattern in ef.name:
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
                    f"No events file found in {events_dir} for task '{loc_type}'. "
                    f"Run the localizer task first to generate the events TSV."
                )

            tr = loc0.get("tr", 1.2)
            threshold = extraction.get("threshold", 2000)
            smooth_fwhm = extraction.get("smooth_fwhm", 5.0)
            highpass = extraction.get("highpass", 0.01)
            threshold_method = extraction.get("threshold_method", "top_n")
            atlas_mask_cfg = extraction.get("atlas_mask")
            atlas_source_cfg = extraction.get("atlas_source")
            atlas_regions_cfg = extraction.get("atlas_regions")
            cluster_start_threshold = extraction.get("cluster_start_threshold", 2.0)
            cluster_step = extraction.get("cluster_step", 0.1)
            cluster_return_all = extraction.get("cluster_return_all", False)
            min_cluster_size = extraction.get("min_cluster_size", 50)
            max_total_voxels = extraction.get("max_total_voxels")

            include_reverse = loc0.get("include_reverse_contrasts", False)

            params = {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "root_dir": mask_extraction_dir,
                "input_file": preproc_file,
                "mask_file": mask_file,
                "events": events_file,
                "contrasts": contrasts,
                "tr": tr,
                "threshold": threshold,
                "threshold_method": threshold_method,
                "atlas_mask": atlas_mask_cfg,
                "atlas_source": atlas_source_cfg,
                "atlas_regions": atlas_regions_cfg,
                "ref_file": str(ref_file),
                "cluster_start_threshold": cluster_start_threshold,
                "cluster_step": cluster_step,
                "cluster_return_all": cluster_return_all,
                "min_cluster_size": min_cluster_size,
                "max_total_voxels": max_total_voxels,
                "include_reverse": include_reverse,
                "overwrite": self.overwrite,
                "smoothing_fwhm": smooth_fwhm,
                "high_pass": highpass,
                "qc_enabled": True,
            }
            _mask_files = run_mask_extraction_from_config("task", params, progress_callback=progress_callback)
            self._write_reproduce_script("task", params)
        else:
            # ICA-based extraction (existing behavior)
            network_atlas = extraction.get("network_atlas", "yeo_17")
            networks = list(loc0.get("networks", []))  # Copy to avoid modifying original
            n_components = extraction.get("n_components", 30)
            threshold = extraction.get("threshold", 2000)
            smooth_fwhm = extraction.get("smooth_fwhm", 5.0)
            highpass = extraction.get("highpass", 0.01)
            lowpass = extraction.get("lowpass")

            # Sham gating is study-wide; the construction networks come from the per-run
            # sham config of whichever feedback runs source masks from this localizer. All
            # sham networks are extracted for ALL participants (blinding requirement).
            from ..utils import sham_hemisphere_network_for_localizer, sham_networks_for_localizer

            sham_hemisphere_network = sham_hemisphere_network_for_localizer(config, loc0)
            for net in sham_networks_for_localizer(config, loc0):
                if net not in networks:
                    networks.append(net)
                    if self._is_blind_mode():
                        self._emit_progress_file_only(f"Including sham network: {net}")
                        self._emit_progress("Including additional network for mask extraction")
                    else:
                        self._emit_progress(f"Including sham network: {net}")

            params = {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "root_dir": mask_extraction_dir,
                "input_file": preproc_file,
                "mask_file": mask_file,
                "ref_file": ref_file,
                "networks": networks,
                "network_atlas": network_atlas,
                "n_components": n_components,
                "threshold": threshold,
                "smooth_fwhm": smooth_fwhm,
                "highpass": highpass,
                "lowpass": lowpass,
                "overwrite": self.overwrite,
                "qc_enabled": True,
                "sham_hemisphere_network": sham_hemisphere_network,
            }
            _mask_files = run_mask_extraction_from_config(
                "rest", params, progress_callback=progress_callback, stop_event=self._stop_event
            )
            self._write_reproduce_script("rest", params)

        # Create WM/CSF tissue masks for nuisance regression (if enabled)
        self._create_tissue_masks_if_enabled(mask_extraction_dir, ref_file, progress_callback)

    def _write_reproduce_script(
        self,
        mode: str,
        params: dict,
        selections: dict | None = None,
    ) -> None:
        """Write a standalone reproduce_masks.py capturing this extraction call.

        The script replays the exact ``mode`` + ``params`` through
        ``run_mask_extraction_from_config`` — the same dispatcher the pipeline
        just used — so it can never drift from the core API. Best-effort: a
        failure here must not break the pipeline run.
        """
        try:
            from pineuro.script_generator import generate_mask_extraction_script, save_script

            script = generate_mask_extraction_script(mode, params, selections=selections)
            out = save_script(script, Path(params["root_dir"]) / "reproduce_masks.py")
            self._emit_progress(f"Wrote reproduction script: {out.name}")
        except Exception as ex:
            logger.warning(f"Could not write reproduction script: {ex}")

    def _create_tissue_masks_if_enabled(
        self,
        mask_extraction_dir: Path,
        ref_file: str | Path | None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """Create WM/CSF tissue masks for nuisance regression when enabled.

        Triggered when *any* feedback run requests WM or CSF nuisance regressors.
        """
        feedback_runs = self.config.get("feedback_runs", []) or []
        own_need = any(
            frc.get("glm", {}).get("include_wm_regressor", False)
            or frc.get("glm", {}).get("include_csf_regressor", False)
            for frc in feedback_runs
        )
        # ``_tissue_required`` (stamped when the study is flattened) also covers later
        # sessions that reuse this session's masks and need nuisance regressors, so a
        # feedback-only session has tissue masks to coregister from here.
        if not self.config.get("_tissue_required", own_need):
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
            session_id=self.session_id,
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

        # Select attempts
        rest_volumes = [n_volumes] * n_runs
        resolved = self._resolve_attempt_paths(
            task_name=task_name,
            runs_to_process=runs_to_process,
            rest_volumes=rest_volumes,
        )

        if not resolved:
            raise ValueError(f"No {task_name} collection directories found for preprocessing")

        input_paths = [path for path, _attempt, _run in resolved]
        attempt_per_input = [attempt for _path, attempt, _run in resolved]
        run_per_input = [run for _path, _attempt, run in resolved]

        for path in input_paths:
            self._emit_progress(f"Using: {path.name}")

        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        preprocessing(
            participant_id=self.participant_id,
            session_id=self.session_id,
            task=task_name,
            root_dir=preprocessing_dir,
            input_dicoms=[str(d) for d in input_paths],
            overwrite=self.overwrite,
            qc_enabled=True,
            progress_callback=progress_callback,
            attempt_per_input=attempt_per_input,
            run_per_input=run_per_input,
            stop_event=self._stop_event,
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

        preproc_file = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_preproc.nii.gz"
        mask_file = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_preproc_mask.nii.gz"
        ref_file = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_boldref.nii.gz"

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
            from pineuro.mask_extraction import run_mask_extraction_from_config

            extraction = loc.get("extraction", {})
            network_atlas = extraction.get("network_atlas", "yeo_17")
            networks = list(loc.get("networks", []))
            threshold = extraction.get("threshold", 2000)
            smooth_fwhm = extraction.get("smooth_fwhm", 5.0)
            highpass = extraction.get("highpass", 0.01)
            lowpass = extraction.get("lowpass")

            # Include sham construction networks (per-run sham config of the feedback runs
            # sourcing masks from this localizer; gated study-wide by config["sham"].enabled).
            from ..utils import sham_hemisphere_network_for_localizer, sham_networks_for_localizer

            sham_hemisphere_network = sham_hemisphere_network_for_localizer(config, loc)
            for net in sham_networks_for_localizer(config, loc):
                if net not in networks:
                    networks.append(net)

            params = {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "root_dir": loc_dir,
                "input_file": preproc_file,
                "mask_file": mask_file,
                "ref_file": ref_file,
                "networks": networks,
                "network_atlas": network_atlas,
                "n_components": extraction.get("n_components", 30),
                "threshold": threshold,
                "smooth_fwhm": smooth_fwhm,
                "highpass": highpass,
                "lowpass": lowpass,
                "overwrite": self.overwrite,
                "qc_enabled": True,
                "sham_hemisphere_network": sham_hemisphere_network,
            }
            mask_files = run_mask_extraction_from_config(
                "rest", params, progress_callback=progress_callback, stop_event=self._stop_event
            )
            self._write_reproduce_script("rest", params)
        else:
            from pineuro.mask_extraction import run_mask_extraction_from_config

            extraction = loc.get("extraction", {})
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

            params = {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "root_dir": loc_dir,
                "input_file": preproc_file,
                "mask_file": mask_file,
                "events": events_file,
                "contrasts": contrasts,
                "tr": loc.get("tr", 1.2),
                "threshold": extraction.get("threshold", 2000),
                "threshold_method": extraction.get("threshold_method", "top_n"),
                "atlas_mask": extraction.get("atlas_mask"),
                "atlas_source": extraction.get("atlas_source"),
                "atlas_regions": extraction.get("atlas_regions"),
                "ref_file": str(ref_file),
                "cluster_start_threshold": extraction.get("cluster_start_threshold", 2.0),
                "cluster_step": extraction.get("cluster_step", 0.1),
                "cluster_return_all": extraction.get("cluster_return_all", False),
                "min_cluster_size": extraction.get("min_cluster_size", 50),
                "max_total_voxels": extraction.get("max_total_voxels"),
                "include_reverse": loc.get("include_reverse_contrasts", False),
                "overwrite": self.overwrite,
                "smoothing_fwhm": extraction.get("smooth_fwhm", 5.0),
                "high_pass": extraction.get("highpass", 0.01),
                "drift_model": loc.get("drift_model", "cosine"),
                "hrf_model": loc.get("hrf_model", "spm"),
                "qc_enabled": True,
            }
            mask_files = run_mask_extraction_from_config("task", params, progress_callback=progress_callback)
            self._write_reproduce_script("task", params)

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
        boldref = preprocessing_dir / f"{self._bids_prefix}_task-{task_name}_boldref.nii.gz"
        if boldref.exists():
            self._emit_progress(f"Using boldref from preprocessing: {boldref.name}")
            return boldref

        collection_dir = base_dir / "collection"
        mni_ref_dir = base_dir / "mask_extraction" / "00_mni_ref"

        # Option 2: Initial reference DICOMs → convert to NIfTI → compute mean
        ref_dirs = sorted(collection_dir.glob("task-reference_run-01_attempt-*"))
        for ref_dir in reversed(ref_dirs):  # newest attempt first
            dicoms = list(ref_dir.iterdir())
            if dicoms:
                from pineuro.utils import compute_mean, dicom_to_nifti

                mni_ref_dir.mkdir(parents=True, exist_ok=True)

                mean_file = mni_ref_dir / f"{self._bids_prefix}_task-reference_mean.nii.gz"
                if mean_file.exists():
                    self._emit_progress(f"Using existing reference mean: {mean_file.name}")
                    return mean_file

                self._emit_progress(f"Converting reference DICOMs from {ref_dir.name}...")
                nifti_file = dicom_to_nifti(
                    dicom_dir=ref_dir,
                    output_dir=mni_ref_dir,
                    output_name=f"{self._bids_prefix}_task-reference",
                )
                nifti_path = Path(nifti_file)
                if nifti_path.exists():
                    self._emit_progress("Computing mean reference volume...")
                    compute_mean(input_file=nifti_path, output_file=mean_file)
                    if mean_file.exists():
                        return mean_file

        # Option 3: Realtime reference DICOMs (for 0-localizer MNI atlas mode)
        rt_ref_dirs = sorted(collection_dir.glob("task-reference_run-02_attempt-*"))
        for ref_dir in reversed(rt_ref_dirs):  # newest attempt first
            dicoms = list(ref_dir.iterdir())
            if dicoms:
                from pineuro.utils import compute_mean, dicom_to_nifti

                mni_ref_dir.mkdir(parents=True, exist_ok=True)

                mean_file = mni_ref_dir / f"{self._bids_prefix}_task-reference-realtime_mean.nii.gz"
                if mean_file.exists():
                    self._emit_progress(f"Using existing realtime reference mean: {mean_file.name}")
                    return mean_file

                self._emit_progress(f"Converting realtime reference DICOMs from {ref_dir.name}...")
                nifti_file = dicom_to_nifti(
                    dicom_dir=ref_dir,
                    output_dir=mni_ref_dir,
                    output_name=f"{self._bids_prefix}_task-reference-realtime",
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
        from pineuro.mask_extraction import run_mask_extraction_from_config

        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"

        # Find the best available reference image
        ref_file = self._find_best_reference()

        # Read atlas/network config (same settings used for ICA pipeline)
        loc0 = _loc(config)
        extraction = loc0.get("extraction", {})
        network_atlas = extraction.get("network_atlas", "yeo_17")
        networks = list(loc0.get("networks", ["DefaultA", "ContA"]))
        threshold = extraction.get("threshold", 2000)

        # Handle sham construction networks (same per-run resolution as _run_mask_extraction)
        from ..utils import sham_hemisphere_network_for_localizer, sham_networks_for_localizer

        sham_hemisphere_network = sham_hemisphere_network_for_localizer(config, loc0)
        for net in sham_networks_for_localizer(config, loc0):
            if net not in networks:
                networks.append(net)
                if self._is_blind_mode():
                    self._emit_progress_file_only(f"Including sham network: {net}")
                    self._emit_progress("Including additional network for mask extraction")
                else:
                    self._emit_progress(f"Including sham network: {net}")

        def progress_callback(step: str, detail: str = ""):
            if detail:
                self._emit_progress(f"{step}: {detail}")
            else:
                self._emit_progress(step)

        params = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "root_dir": mask_extraction_dir,
            "ref_file": ref_file,
            "networks": networks,
            "network_atlas": network_atlas,
            "threshold": threshold,
            "overwrite": self.overwrite,
            "qc_enabled": True,
            "sham_hemisphere_network": sham_hemisphere_network,
        }
        _mask_files = run_mask_extraction_from_config("mni", params, progress_callback=progress_callback)
        self._write_reproduce_script("mni", params)

        # Create WM/CSF tissue masks for nuisance regression (if any feedback run needs them)
        feedback_runs = config.get("feedback_runs", []) or []
        if any(
            frc.get("glm", {}).get("include_wm_regressor", False)
            or frc.get("glm", {}).get("include_csf_regressor", False)
            for frc in feedback_runs
        ):
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
                session_id=self.session_id,
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
        from pineuro.mask_extraction import ATLAS_SOURCES, coregister_reference, generate_reference_qc_overlays

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

        from ..utils import has_fresh_localizer

        localizers = config.get("localizers", [])
        study_base_dir = base_dir.parent.parent  # study root (holds reuse source sessions)

        # Run-number rule: an all-reused session collects only the realtime reference
        # (run-01); any other session has an initial reference (run-01) so realtime is
        # run-02. (run_id=1 coregistration is only emitted when a fresh localizer exists.)
        all_reused = bool(localizers) and not has_fresh_localizer(config)
        actual_reference_run = 1 if run_id == 1 else (1 if all_reused else 2)

        run_str = f"run-{actual_reference_run:02d}"
        self._emit_progress(f"Coregistering network masks to reference {run_str}...")

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

        # Use highest complete attempt (or highest incomplete if none complete)
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

        # Sham hemisphere-split: the combined network mask is dropped per localizer below
        # (its left/right split masks are used instead); the network is resolved from the
        # per-run sham config of the feedback runs sourcing masks from each localizer.
        # The sham construction networks are folded into the coregistration name set via
        # _coreg_localizer_networks (so the split/alternative masks reach reference space).
        from ..utils import sham_hemisphere_network_for_localizer

        # Tissue masks (WM/CSF). A fresh-localizer session created them locally and
        # coregisters them with its first fresh localizer group (same space). A
        # feedback-only / all-reused session has none locally: when nuisance
        # regression is needed it pulls them from a source session's 05_tissue and
        # coregisters with that source's reused group (whose boldref->reference
        # transform matches the tissue masks' space).
        local_tissue_files: list[Path] = []
        local_tissue_dir = mask_extraction_dir / "05_tissue"
        if local_tissue_dir.exists():
            local_tissue_files = sorted(local_tissue_dir.glob(f"{self._bids_prefix}_tissue_*_mask.nii.gz"))
        first_fresh_idx = next((i for i, loc in enumerate(localizers) if not loc.get("reuse_from_session")), None)

        own_tissue_need = any(
            frc.get("glm", {}).get("include_wm_regressor", False)
            or frc.get("glm", {}).get("include_csf_regressor", False)
            for frc in (config.get("feedback_runs", []) or [])
        )
        tissue_required = config.get("_tissue_required", own_tissue_need)
        reused_tissue_idx: int | None = None
        reused_tissue_files: list[Path] = []
        if tissue_required and not local_tissue_files:
            for i, loc in enumerate(localizers):
                src = loc.get("reuse_from_session")
                if not src:
                    continue
                src_prefix = f"{self.participant_id}_{src}"
                src_tissue_dir = study_base_dir / self.participant_id / src / "mask_extraction" / "05_tissue"
                if src_tissue_dir.exists():
                    files = sorted(src_tissue_dir.glob(f"{src_prefix}_tissue_*_mask.nii.gz"))
                    if files:
                        reused_tissue_idx = i
                        reused_tissue_files = files
                        break

        # Coregister each localizer's masks with the reference (boldref) from its
        # own source: the current session for fresh localizers, the source session
        # for reused ones. A session may mix the two.
        qc_ref_file: Path | None = None
        source_mask_extraction_dir: Path | None = None  # ICA outputs dir for QC overlays
        coreg_masks: list[Path] = []
        for loc_idx, loc in enumerate(localizers):
            loc_type = loc.get("type", "rest")
            loc_task_name = "rest" if loc_type == "rest" else loc_type
            reuse_from = loc.get("reuse_from_session")
            # Real networks + sham construction networks (so the sham hemisphere/alternative
            # masks are coregistered too — mirrors mask extraction; see _coreg_localizer_networks).
            names = _coreg_localizer_networks(config, loc)

            if reuse_from:
                src_session_dir = study_base_dir / self.participant_id / reuse_from
                loc_me_dir = src_session_dir / "mask_extraction"
                src_masks_dir = loc_me_dir / "03_masks"
                src_prefix = f"{self.participant_id}_{reuse_from}"
                loc_ref_file = src_session_dir / "preprocessing" / f"{src_prefix}_task-{loc_task_name}_boldref.nii.gz"
            else:
                loc_me_dir = mask_extraction_dir
                src_masks_dir = mask_extraction_dir / "03_masks"
                src_prefix = self._bids_prefix
                if loc_type == "manual":
                    loc_dir = mask_extraction_dir / f"loc_{loc_idx}_manual"
                    loc_ref_file = self._manual_reference_file(base_dir, loc_dir, loc)
                else:
                    loc_ref_file = preprocessing_dir / f"{src_prefix}_task-{loc_task_name}_boldref.nii.gz"

            # This localizer's thresholded masks from its source dir: exact name
            # matches plus derived variants (e.g. hemisphere "{name}_left"), minus the
            # combined network mask when hemisphere-split sham is active (its left/right
            # splits are coregistered instead).
            sham_combined_name = sham_hemisphere_network_for_localizer(config, loc)
            group_files: list[Path] = []
            for f in sorted(src_masks_dir.glob(f"{src_prefix}_*_thr.nii.gz")):
                mask_name = f.name[len(src_prefix) + 1 : -len("_thr.nii.gz")]
                if sham_combined_name and mask_name == sham_combined_name:
                    continue
                if mask_name in names or any(mask_name.startswith(f"{n}_") for n in names):
                    group_files.append(f)

            if not group_files:
                self._emit_progress(
                    f"Warning: no masks found for localizer {loc_idx} ({loc_task_name}) in {src_masks_dir}"
                )
                continue

            group_strs = [str(f) for f in group_files]
            group_tissue: list[Path] = []
            if loc_idx == first_fresh_idx:
                group_tissue = local_tissue_files
            elif loc_idx == reused_tissue_idx:
                group_tissue = reused_tissue_files
            if group_tissue:
                group_strs += [str(f) for f in group_tissue]
                self._emit_progress(f"Including {len(group_tissue)} tissue masks for coregistration")

            origin = f"reuse {reuse_from}" if reuse_from else "this session"
            self._emit_progress(
                f"Coregistering {len(group_strs)} mask(s) from localizer {loc_idx} ({loc_task_name}, {origin})..."
            )
            coreg_result = coregister_reference(
                participant_id=self.participant_id,
                session_id=self.session_id,
                root_dir=root_dir,
                input_dicom=reference_dicom,
                network_files=group_strs,
                ref_file=str(loc_ref_file),
                run_id=actual_reference_run,
                overwrite=self.overwrite,
                output_subdir=output_subdir,
            )
            coreg_masks.extend(coreg_result)
            if qc_ref_file is None:
                qc_ref_file = loc_ref_file
                source_mask_extraction_dir = loc_me_dir

        ref_file = qc_ref_file
        self._emit_progress(f"Coregistered {len(coreg_masks)} masks to reference {run_str}")

        if ref_file is None:
            # No localizer produced masks (nothing coregistered) — skip QC overlays.
            return

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
            # In mask reuse mode, this is the source session's mask_extraction_dir
            qc_mask_extraction_dir = source_mask_extraction_dir if source_mask_extraction_dir else mask_extraction_dir

            generate_reference_qc_overlays(
                participant_id=self.participant_id,
                session_id=self.session_id,
                session_dir=session_dir,
                ref_file=ref_file,
                mask_extraction_dir=qc_mask_extraction_dir,
                network_coreg_files=coreg_masks,
                run_str=str(actual_reference_run).zfill(2),
                n_networks=ATLAS_SOURCES.get(
                    config["localizers"][0].get("extraction", {}).get("network_atlas", "yeo_17")
                    if config.get("localizers")
                    else "yeo_17",
                    {},
                ).get("n_networks", 17),
                overwrite=self.overwrite,
                reference_subdir=reference_subdir,
            )
            self._emit_progress(f"QC overlays generated for reference {run_str}")
        except Exception as e:
            logger.warning(f"Reference QC overlay generation failed: {e}")
            self._emit_progress(f"Warning: QC overlay generation failed: {e}")

    def _run_reference_realtime(self) -> None:
        """Collect real-time reference scan before feedback tasks."""
        from pineuro.preprocessing import collect_volumes
        from pineuro.session import get_localizer_run_info

        config = self.config

        # Check if reference realtime is enabled
        if not config.get("reference", {}).get("realtime_enabled", True):
            self._emit_progress("Reference realtime collection is disabled, skipping...")
            return

        base_dir = Path(config["paths"]["base_dir"])
        collection_dir = base_dir / "collection"

        # Realtime reference is run-01 for an all-reused session (no initial reference
        # collected), else run-02 (after the initial reference).
        from ..utils import has_fresh_localizer

        localizers = config.get("localizers", [])
        all_reused = bool(localizers) and not has_fresh_localizer(config)
        run_id = 1 if all_reused else 2

        n_volumes = config.get("reference", {}).get("realtime_n_volumes", 2)

        # Get run info with retry detection
        run_info = get_localizer_run_info(collection_dir=collection_dir, task="task-reference", run=run_id)
        attempt = run_info["attempt"]
        self._set_current_attempt(
            kind="localizer", subject=self.participant_id, task="reference", attempt=attempt, run=run_id
        )

        self._emit_progress(f"Collecting real-time reference DICOMs (run-{run_id:02d}, attempt-{attempt:02d})...")
        if run_info["existing_attempts"]:
            self._emit_progress(
                f"Previous attempts: {', '.join(f'attempt-{a:02d}' for a in run_info['existing_attempts'])}"
            )

        output_dir = collection_dir / f"task-reference_run-{run_id:02d}_attempt-{attempt:02d}"

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

        collect_volumes(
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
        from pineuro.feedback import get_slot_run_info, run_feedback_task
        from pineuro.sham import (
            ReplayActivationProvider,
            find_feedbacklog_for_replay,
            get_condition_for_participant,
            load_randomization,
        )
        from pineuro.tasks.designs import BlockDesign, NuisanceOnlyDesign
        from pineuro.tasks.state import TaskState
        from pineuro.tasks.task_registry import get_task_metadata

        from ..utils import feedback_step_for

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
        # Convenience map from a short setting name to the nested sub-block it lives
        # in (qc / glm / connectivity / intermittent), so call sites can use flat names.
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
            """Read a feedback-run setting from the per-run config (nested schema)."""
            if frc is None:
                return default
            # Settings that live in a nested sub-block.
            sub_key = _NESTED_KEY_MAP.get(key)
            if sub_key is not None:
                block, inner = sub_key
                return frc.get(block, {}).get(inner, default)
            # Top-level settings (task, max_runs, tr, n_volumes, n_baseline, feedback_mode, etc.)
            return frc.get(key, default)

        # Get output directory for slot-based run info
        config = self.config
        base_dir = Path(config["paths"]["base_dir"])
        mask_extraction_dir = base_dir / "mask_extraction"
        feedback_dir = base_dir / "feedback"
        output_dir = feedback_dir / "02_feedback_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the BIDS task entity (real task name; transfer runs get a "transfer"
        # suffix) and the per-task run number from the shared step plan.
        step = feedback_step_for(config, step_id)
        if step is None:
            raise ValueError(f"Could not resolve feedback step for step_id: {step_id}")
        task_entity = step.task_entity
        task_run_number = step.run_number
        # Real registered task id (e.g. "ball") — drives TaskState/factory/QC, not filenames.
        feedback_task_type = step.task_id

        # Next attempt number for this (task entity, run) slot, for retry tracking.
        run_info = get_slot_run_info(
            output_dir=output_dir,
            subject=self.participant_id,
            session=self.session_id,
            task=task_entity,
            run=task_run_number,
        )
        task_attempt = run_info["attempt"]
        self._set_current_attempt(
            kind="feedback",
            subject=self.participant_id,
            task=task_entity,
            attempt=task_attempt,
            run=task_run_number,
        )

        self._emit_progress(f"Starting {task_type} run-{task_run_number:02d} attempt-{task_attempt:02d}...")

        # Get mask files - use reference-coregistered masks if realtime reference is enabled,
        # otherwise use boldref space masks from mask extraction

        # Determine mask names based on the localizer entry's type.
        loc0 = _loc(config)
        loc_type = loc0.get("type", "rest")
        signal_mode = _fb("feedback_mode", "activation")
        include_reverse = loc0.get("include_reverse_contrasts", False)
        if loc_type not in ("rest", "manual"):
            contrasts = loc0.get("contrasts", {})
            real_mask_names = list(contrasts.keys())
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
            assigned_names = []
            if frc:
                net_assign = frc.get("network_assignment", {})
                for role in metadata.network_roles:
                    name = net_assign.get(role.key)
                    if name:
                        assigned_names.append(name)
            # Use whatever roles were assigned (single-network tasks resolve to
            # one mask; optional roles add more). Empty -> keep the default.
            if assigned_names:
                real_mask_names = assigned_names

        self._emit_progress(f"Task networks: {real_mask_names}")

        reference_realtime_enabled = config.get("reference", {}).get("realtime_enabled", True)

        # Reference run number for the coregistered-mask file pattern: run-01 for an
        # all-reused session (no initial reference), else run-02.
        from ..utils import has_fresh_localizer

        all_reused = bool(config.get("localizers", [])) and not has_fresh_localizer(config)
        realtime_reference_run = 1 if all_reused else 2

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
            watch_dir = (
                base_dir / "tmp" / f"incoming_{task_entity}_run-{task_run_number:02d}_attempt-{task_attempt:02d}"
            )
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
        sham_config = config.get("sham", {})  # study-wide gating: enabled + randomization_file
        frc_sham = (frc.get("sham", {}) if frc else {}) or {}  # per-run mask construction params

        if sham_config.get("enabled", False):
            randomization_file = sham_config.get("randomization_file")
            if randomization_file:
                randomization_path = Path(randomization_file)
                if randomization_path.exists():
                    try:
                        randomization = load_randomization(randomization_path)
                        # Sham is enabled, so an unlisted participant cannot be assigned a
                        # condition. Fail loud instead of silently defaulting to "real"
                        # (get_condition_for_participant's permissive default is wrong here).
                        if self.participant_id not in randomization:
                            raise ValueError(
                                f"Participant {self.participant_id} is not in the randomization file "
                                f"{randomization_path}. Sham is enabled, so a condition cannot be "
                                f"assigned; check for a typo in the participant ID or add them to the file."
                            )
                        condition_info = get_condition_for_participant(randomization, self.participant_id)
                        condition = condition_info["condition"]
                        if self._is_blind_mode():
                            self._emit_progress_file_only(f"Participant condition: {condition}")
                        else:
                            self._emit_progress(f"Participant condition: {condition}")

                        delivery_method = frc_sham.get("delivery_method", "alternative_masks")

                        if delivery_method == "alternative_masks":
                            mask_mode = frc_sham.get("sham_mask_mode", "hemisphere_split")
                            if mask_mode == "hemisphere_split":
                                # Hemisphere split: use left/right halves of a single network.
                                # Sham target = {network}_{target_hemisphere},
                                # sham non-target = contralateral hemisphere.
                                hemisphere_network = frc_sham.get("hemisphere_network", "SomMotA")
                                target_hemisphere = frc_sham.get("target_hemisphere", "right")
                                non_target_hemisphere = "left" if target_hemisphere == "right" else "right"

                                for hemi in [target_hemisphere, non_target_hemisphere]:
                                    alt_name = f"{hemisphere_network}_{hemi}"
                                    try:
                                        sham_mask_files.append(find_mask_file(alt_name))
                                        sham_mask_names.append(alt_name)
                                    except ValueError as e:
                                        _require_sham_mask(condition, alt_name, e)
                            else:
                                # Two networks mode: use two distinct networks
                                alternative_targets = frc_sham.get("alternative_mask_targets", [])
                                for alt_name in alternative_targets:
                                    try:
                                        sham_mask_files.append(find_mask_file(alt_name))
                                        sham_mask_names.append(alt_name)
                                    except ValueError as e:
                                        _require_sham_mask(condition, alt_name, e)

                        elif delivery_method == "replay" and condition == "sham":
                            # Replay mode: find feedbacklog from matched participant
                            matched_participant = condition_info.get("matched_participant")
                            matched_session = condition_info.get("matched_session")

                            if not matched_participant:
                                raise ValueError(
                                    f"Sham participant {self.participant_id} has replay mode "
                                    f"but no matched_participant in randomization file"
                                )

                            if self._is_blind_mode():
                                self._emit_progress_file_only(
                                    f"Replay mode: looking up feedbacklog from {matched_participant}"
                                    + (f" session {matched_session}" if matched_session else "")
                                )
                                self._emit_progress("Replay mode: loading activation source")
                            else:
                                self._emit_progress(
                                    f"Replay mode: looking up feedbacklog from {matched_participant}"
                                    + (f" session {matched_session}" if matched_session else "")
                                )

                            try:
                                # base_dir is participant-specific (e.g., /data/sub-02)
                                # For replay, we need the study base dir (e.g., /data)
                                # to find other participants' feedbacklogs
                                study_base_dir = base_dir.parent
                                feedbacklog_path = find_feedbacklog_for_replay(
                                    source_dir=study_base_dir,
                                    matched_participant=matched_participant,
                                    matched_session=matched_session or self.session_id,
                                    task=task_entity,
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

                        # A sham participant must end up with sham masks (alternative_masks
                        # modes) — otherwise task_mask_names falls back to the real masks
                        # below and sham silently delivers real. This catches an empty
                        # alternative_mask_targets config (the per-mask check above covers
                        # masks that exist in config but not on disk).
                        if condition == "sham" and delivery_method == "alternative_masks" and not sham_mask_names:
                            raise ValueError(
                                "Sham condition resolved to no sham masks; check the feedback "
                                "run's sham configuration (see log for details)."
                            )

                    except Exception as e:
                        logger.error(f"Sham setup failed: {e}")
                        self._emit_progress(f"ERROR: Sham setup failed: {e}")
                        raise
                else:
                    raise FileNotFoundError(
                        f"Sham is enabled but the randomization file was not found: "
                        f"{randomization_path}. A sham study cannot run without it "
                        f"(every participant would default to real)."
                    )
            else:
                raise ValueError("Sham is enabled for this study but no randomization_file is configured.")

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
            and frc_sham.get("delivery_method") == "alternative_masks"
        ):
            sham_pair = [sham_mask_names[0], sham_mask_names[1]]
            if sham_pair not in connectivity_pairs:
                connectivity_pairs = list(connectivity_pairs) + [sham_pair]

        # Run feedback task (this will handle DICOM collection + processing)
        # Pass ALL masks so all activations are computed and logged
        try:
            session_info = run_feedback_task(
                mask_paths=[str(f) for f in all_mask_files],
                subject=self.participant_id,
                session=self.session_id,
                task=task_entity,
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
                step_id=step_id,  # Use slot-based naming with attempt entity
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
            with self._task_lock:
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

            # Load task state for adaptive difficulty persistence across runs
            task_state = TaskState.load(output_dir, task_id=feedback_task_type)

            # Update max_feedback_runs from config if different (sum across all feedback_runs)
            max_feedback_runs = sum(fr.get("max_runs", 1) for fr in config.get("feedback_runs", [])) or 2
            if task_state.max_feedback_runs != max_feedback_runs:
                task_state.max_feedback_runs = max_feedback_runs
                task_state.save()

            self._emit_progress(f"Task state: task={feedback_task_type}, runs={len(task_state.completed_runs)}")

            # Build factory config with task parameters plus runtime settings
            signal_mode = _fb("feedback_mode", "activation")

            connectivity_target = _fb("connectivity_target", "increase")

            # First run of its type (transfer/feedback) gets the detailed
            # instruction slides; later runs get the brief reminder.
            run_type = "feedback" if feedback_mode else "transfer"
            is_first_of_type = not any(r.run_type == run_type for r in task_state.completed_runs)

            factory_config = {
                **task_specific_config,
                "mask_names": task_mask_names,
                "feedback_mode": feedback_mode,
                "signal_mode": signal_mode,
                "connectivity_target": connectivity_target,
                "presenter_monitor": presenter_monitor,
                "is_first_of_type": is_first_of_type,
            }

            # Create task via factory (validates config against task metadata)
            simulator = session_info.get("simulator")
            task = create_task(
                task_id=feedback_task_type,
                coordinator=coordinator,
                config=factory_config,
                task_state=task_state,
                output_dir=str(output_dir),
                simulator=simulator,
            )

            # Register callback so coordinator updates task with new activations
            coordinator._activation_callback = task._update_data

            # Gate "last_feedback_only" post-run questions on the last feedback run for this task.
            task._is_last_feedback_run = feedback_mode and step.is_last_feedback

            # Track task for stop support
            with self._task_lock:
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

            # Surface any error recorded by the coordinator background thread (e.g. a
            # MoCo data/config mismatch) so the step fails instead of reporting fake
            # success. check_error() is a no-op on a clean or user-stopped run (a stop
            # sets _stop_event but not _error), so this only raises on genuine errors.
            coordinator.check_error()  # re-raises coordinator._error if set

            # Extract results generically via task contract
            results = task.get_run_results()
            volumes_processed = results.get("volumes_processed", coordinator.n_processed)

            # Defense in depth: a run that processed nothing produced no feedback.
            # Fail it (unless the user stopped it) so other silent-0-volume causes
            # (wrong watch dir, scanner not sending, all volumes failing validation)
            # don't masquerade as success.
            if volumes_processed == 0 and not coordinator.was_stopped() and coordinator._error is None:
                raise RuntimeError(
                    f"{task_type.capitalize()} run {task_run_number} processed 0 volumes. "
                    f"No feedback was produced. Check that the scanner is sending volumes to the "
                    f"watch directory and that the motion correction mode matches the incoming data."
                )

            result_msg = (
                f"{task_type.capitalize()} run {task_run_number} complete! Processed {volumes_processed} volumes."
            )
            extra = {k: v for k, v in results.items() if k != "volumes_processed"}
            if extra:
                result_msg += f" {extra}"

            self._emit_progress(result_msg)
        finally:
            # Clear coordinator and task references
            with self._task_lock:
                self._current_coordinator = None
                self._current_feedback_task = None
