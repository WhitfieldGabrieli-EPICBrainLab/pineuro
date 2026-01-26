"""
Real-time fMRI preprocessing pipeline for neurofeedback research.

Summary
-------
DICOM conversion, motion correction, brain extraction, and multi-run
coregistration with consistent, numbered output folders.

Functions
---------
- dicom_to_nifti: Convert DICOM files to NIfTI
- motion_correction: Apply motion correction using FSL's mcflirt
- brain_extraction: Perform brain extraction using FSL's bet
- preprocessing: Complete preprocessing pipeline

Workflow
--------
- Convert raw DICOMs to NIfTI
- Run motion correction and brain extraction
- Optionally smooth/filter and coregister across runs
- Produce standardized outputs for downstream mask extraction/feedback

Dependencies
------------
FSL, nibabel, numpy

Notes
-----
Outputs use a standard layout:
- 01_nifti : Converted NIfTI files
- 02_motion_correction : Motion-corrected data
- 03_brain_extraction : Means and brain-extracted data
- 04_coregistered : Coregistered and concatenated runs

FSL binaries must be available in PATH. Each step honors overwrite=False to
skip work when outputs already exist.
Further reading: see docs/references.md for reference list.

References
----------
Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. NeuroImage, 62(2), 782-790.
"""

# Standard library imports
from typing import Union, List, Optional, Callable
from pathlib import Path
import heapq
import shutil
import copy
import threading
import time

# Third-party imports
import numpy as np
import nibabel as nib
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from nilearn.image import smooth_img
from scipy.signal import butter, filtfilt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent

# Local imports
from pineuro.utils import (
    run_command, dicom_to_nifti, compute_xfm, apply_xfm, compute_mean, apply_mask,
    DICOM_EXTENSIONS, is_valid_dicom, get_tr_from_dicom, Timer, parallel_run_command,
    erode_mask
)

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

def motion_correction(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Perform motion correction on fMRI data using FSL's mcflirt.

    Parameters
    ----------
    input_file : str or Path
        Path to input 4D NIfTI file containing fMRI timeseries
    output_file : str or Path
        Path for output motion-corrected NIfTI file. Parent directory will be created if it doesn't exist
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """

    cmd = [
        'mcflirt',
        '-in', str(input_file),
        '-out', str(output_file),
        '-plots'  # Generate .par file with motion parameters for QC
    ]

    run_command(
        cmd_name='mcflirt',
        cmd=cmd,
        description='Motion correction',
        inputs=input_file,
        outputs=[output_file, str(output_file) + '.par'],  # Include .par file in outputs
        overwrite=overwrite,
        timing_category='preprocessing'
    )

def concat(
        input_files: List[Union[str, Path]],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Concatenate a set of fMRI data along the time axis.

    Parameters
    ----------
    input_files : List of str or Path
        List of path to input NIfTI files
    output_file: str or Path
        Path to save concatenated data
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping concatenation.")
        return
    
    logger.info("Concatenating...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all images and concatenate
    images = [nib.load(str(f)) for f in input_files]
    
    # Get data arrays and concatenate along time axis
    data_arrays = [img.get_fdata() for img in images]
    concatenated = np.concatenate(data_arrays, axis=-1)
    
    # Use affine and header from first image
    first_img = images[0]
    out_img = Nifti1Image(concatenated, first_img.affine, first_img.header)
    
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")

def spatial_smoothing(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        fwhm: float = 5.0,
        overwrite: bool = False
) -> None:
    """
    Apply Gaussian spatial smoothing to fMRI data using nilearn.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_file : str or Path
        Path for output smoothed NIfTI file. Parent directory will be created if it doesn't exist
    fwhm : float, default=5.0
        Full width at half maximum (FWHM) of the Gaussian smoothing kernel in mm
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping spatial smoothing.")
        return
    
    logger.info("Spatial smoothing...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use nilearn for smoothing
    smoothed = smooth_img(str(input_file), fwhm=fwhm)
    smoothed.to_filename(str(output_path))
    logger.info(f"Output: {output_path}")

def temporal_filtering(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        overwrite: bool = False
) -> None:
    """
    Apply temporal bandpass filtering to fMRI data using scipy Butterworth filter.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_file : str or Path
        Path for output filtered NIfTI file. Parent directory will be created if it doesn't exist
    highpass : float, optional
        High-pass filter cutoff in Hz. If None, no high-pass filtering is applied
    lowpass : float, optional
        Low-pass filter cutoff in Hz. If None, no low-pass filtering is applied
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping temporal filtering.")
        return
    
    # Load image
    img = nib.load(str(input_file))
    data = img.get_fdata()
    
    # Get TR from header (4th element of zooms is temporal spacing)
    # type: ignore needed because nibabel typing doesn't recognize header.get_zooms()[3]
    tr = img.header.get_zooms()[3]  # type: ignore
    fs = 1.0 / tr  # Sampling frequency in Hz
    nyquist = fs / 2.0
    
    logger.info(f"Temporal filtering (TR={tr}s, fs={fs:.3f}Hz)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine filter type and cutoffs
    if highpass is not None and lowpass is not None:
        # Bandpass filter
        btype = 'bandpass'
        Wn = [highpass / nyquist, lowpass / nyquist]
    elif highpass is not None:
        # High-pass filter
        btype = 'highpass'
        Wn = highpass / nyquist
    elif lowpass is not None:
        # Low-pass filter  
        btype = 'lowpass'
        Wn = lowpass / nyquist
    else:
        # No filtering requested
        logger.info("No filter cutoffs specified. Saving unchanged.")
        save(img, output_path)
        return
    
    # Design Butterworth filter (order 2 to match FSL behavior)
    b, a = butter(2, Wn, btype=btype)
    
    # Apply filter along time axis (axis=-1)
    # Reshape to 2D for filtering, then reshape back
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])
    
    # Apply zero-phase filtering
    filtered_2d = filtfilt(b, a, data_2d, axis=-1)
    filtered = filtered_2d.reshape(original_shape)
    
    # Save output
    out_img = Nifti1Image(filtered, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")

def brain_extraction(
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        overwrite: bool = False
) -> tuple[Path, Path]:
    """
    Perform brain extraction using FSL's bet.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_dir : str or Path
        Directory where the NIfTI file will be saved. Will be created if it doesn't exist
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    
    # Construct the expected outputs
    output_name = Path(input_file).with_suffix("").stem
    brain_file = Path(output_dir) / f"{output_name}_brain.nii.gz"
    mask_file = Path(output_dir) / f"{output_name}_brain_mask.nii.gz"
    
    # Build bet command
    cmd = [
        'bet',
        str(input_file),
        str(brain_file),
        "-m"
    ]

    # Run bet
    run_command(
        cmd_name='bet',
        cmd=cmd,
        description='Brain extraction',
        inputs=input_file,
        outputs=[brain_file, mask_file],
        overwrite=overwrite,
        timing_category='preprocessing'
    )

    return brain_file, mask_file

def preprocessing(
        participant_id: str,
        task: str,
        root_dir: Union[str, Path],
        input_dicoms: Union[str, Path, List[Union[str, Path]]],
        overwrite: bool = False,
        n_jobs: int = -1,
        qc_enabled: bool = True,
        progress_callback: Optional[callable] = None
) -> tuple[Path, Path, Path]:
    """
    Run the preprocessing pipeline for fMRI data (task or rest).

    Parameters
    ----------
    participant_id : str
        Participant ID
    task : str
        Task name (e.g., rest)
    root_dir : str or Path
        Path to the root directory for all outputs
    input_dicoms : str or Path or list of str or Path
        Path to a single DICOM folder, or list of paths to multiple files
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    n_jobs : int, default=-1
        Number of parallel jobs for DICOM conversion and motion correction.
        -1 uses all available CPUs. Set to 1 for sequential processing.
    qc_enabled : bool, default=True
        If True, generate QC report after preprocessing completes.
        Report includes motion parameters, framewise displacement, brain mask coverage,
        and multi-run coregistration overlays (if applicable).
    progress_callback : callable, optional
        Callback function for progress updates. Called with (step, detail) arguments.
        Used by GUI to display real-time progress in log console.

    Returns
    -------
    preproc_file : Path
        Preprocessed file
    mask_file : Path
        Mask file
    ref_file : Path
        BOLD reference file

    Raises
    ------
    FileNotFoundError
        If any folder in input_dicoms does not exist or is empty

    """

    # Helper to emit progress to both logger and callback
    def emit_progress(step: str, detail: str = ""):
        if detail:
            logger.info(f"{step}: {detail}")
        else:
            logger.info(step)
        if progress_callback:
            progress_callback(step, detail)

    # Log the command
    emit_progress(f"Running preprocessing for {participant_id}")

    # Convert to Path object
    root_dir = Path(root_dir)

    # Convert inputs to list if single file provided
    if isinstance(input_dicoms, (str, Path)):
        input_dicoms = [input_dicoms]

    # Convert to strings and validate
    input_paths: list[str] = []
    logger.info(f"Input DICOM folders:\n")
    for f in input_dicoms:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"Folder not found: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")
        if not any(path.iterdir()):
            raise FileNotFoundError(f"Directory is empty: {path}")
        input_paths.append(str(path))
        logger.info(f"  {str(path)}")

    # Step 1: Convert DICOMs to NIfTI
    emit_progress("Step 1/4: DICOM to NIfTI conversion")
    nifti_dir = root_dir / "01_nifti"
    nifti_files = []

    # Use parallel processing for DICOM conversion if multiple runs
    if n_jobs != 1 and len(input_paths) > 1:
        emit_progress("Step 1/4", f"Converting {len(input_paths)} runs in parallel...")
        # Build job list for parallel execution
        dcm_jobs = []
        nifti_file_paths = []
        for run, path in enumerate(input_paths):
            output_name = f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}"
            nifti_file_path = nifti_dir / f"{output_name}.nii.gz"
            nifti_file_paths.append(nifti_file_path)

            dcm_jobs.append({
                'cmd_name': 'dcm2niix',
                'cmd': ['dcm2niix', '-z', 'y', '-o', str(nifti_dir), '-f', output_name, str(path)],
                'description': f'DICOM to NIfTI conversion run {run+1}',
                'inputs': path,
                'outputs': nifti_file_path,
                'overwrite': overwrite,
                'verbose': False
            })

        parallel_run_command(dcm_jobs, n_jobs=n_jobs, timing_category='preprocessing')
        nifti_files = nifti_file_paths
    else:
        # Sequential processing
        for run, path in enumerate(input_paths):
            emit_progress("Step 1/4", f"Converting run {run+1}/{len(input_paths)}...")
            nifti_file = dicom_to_nifti(
                dicom_dir=path,
                output_dir=nifti_dir,
                output_name=f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}",
                overwrite=overwrite
            )
            nifti_files.append(nifti_file)

    emit_progress("Step 1/4", f"Complete ({len(nifti_files)} runs converted)")

    # Step 2: Motion correction
    emit_progress("Step 2/4: Motion correction (mcflirt)")
    mcf_dir = root_dir / "02_motion_correction"
    mcf_files = []

    # Use parallel processing for motion correction if multiple runs
    if n_jobs != 1 and len(nifti_files) > 1:
        emit_progress("Step 2/4", f"Running mcflirt on {len(nifti_files)} runs in parallel...")
        # Build job list for parallel execution
        mc_jobs = []
        mcf_file_paths = []
        for run, nifti in enumerate(nifti_files):
            mcf_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mcf.nii.gz")
            mcf_file_paths.append(mcf_file)

            mc_jobs.append({
                'cmd_name': 'mcflirt',
                'cmd': ['mcflirt', '-in', str(nifti), '-out', str(mcf_file)],
                'description': f'Motion correction run {run+1}',
                'inputs': nifti,
                'outputs': mcf_file,
                'overwrite': overwrite,
                'verbose': False
            })

        parallel_run_command(mc_jobs, n_jobs=n_jobs, timing_category='preprocessing')
        mcf_files = mcf_file_paths
    else:
        # Sequential processing
        for run, nifti in enumerate(nifti_files):
            emit_progress("Step 2/4", f"Running mcflirt on run {run+1}/{len(nifti_files)}...")
            mcf_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mcf.nii.gz")
            motion_correction(input_file=nifti, output_file=mcf_file, overwrite=overwrite)
            mcf_files.append(mcf_file)

    emit_progress("Step 2/4", f"Complete ({len(mcf_files)} runs corrected)")

    # Step 3: Brain extraction - compute temporal means and extract brain mask
    emit_progress("Step 3/4: Brain extraction (bet)")
    bet_dir = root_dir / "03_brain_extraction"

    # Compute mean of each motion-corrected run for use in coregistration
    emit_progress("Step 3/4", "Computing temporal mean images...")
    mean_files = []
    for run, nifti in enumerate(mcf_files):
        mean_file = bet_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean.nii.gz")
        compute_mean(input_file=nifti, output_file=mean_file, overwrite=overwrite)
        mean_files.append(mean_file)

    # Extract brain mask from first run's mean (used for all runs)
    emit_progress("Step 3/4", "Running brain extraction (bet)...")
    sub_ref = mean_files[0]
    _, mean_mask = brain_extraction(input_file=sub_ref, output_dir=bet_dir, overwrite=overwrite)
    emit_progress("Step 3/4", "Complete (brain mask extracted)")

    # Step 4: Co-registration across runs (if more than one)
    emit_progress("Step 4/4: Coregistration and finalization")
    coreg_dir = root_dir / "04_coregistered"
    coreg_file = coreg_dir / Path(f"{participant_id}_task-{task}.nii.gz")
    xfm_dir = coreg_dir / "xfm"

    if len(mcf_files) == 1:
        # Single run: skip coregistration and use motion-corrected data directly
        emit_progress("Step 4/4", "Single run - skipping coregistration")
        coreg_file = mcf_files[0]

    else:
        # Multiple runs: coregister each to the first run's mean, then concatenate
        emit_progress("Step 4/4", f"Coregistering {len(mcf_files)} runs to run 1...")
        sub_ref = mean_files[0]
        concat_files = []

        for run, mean_file in enumerate(mean_files):

            if run == 0:
                # First run serves as reference - no coregistration needed
                concat_files.append(mcf_files[run])

            else:
                # Compute rigid transformation from this run's mean to reference mean
                emit_progress("Step 4/4", f"Computing transform for run {run+1}/{len(mean_files)}...")
                output_prefix = xfm_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean_to_run-01_mean")
                _, xfm_files = compute_xfm(moving=mean_file, fixed=sub_ref, output_prefix=output_prefix, type_of_transform='Rigid', overwrite=overwrite)

                # Apply transformation to full 4D timeseries (not just mean)
                emit_progress("Step 4/4", f"Applying transform to run {run+1}/{len(mean_files)}...")
                output_file = coreg_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_to_run-01.nii.gz")
                apply_xfm(moving=mcf_files[run], fixed=sub_ref, transform_files=xfm_files, output_file=output_file, interpolator='linear', overwrite=overwrite)
                concat_files.append(output_file)

        # Concatenate all coregistered runs along time axis
        emit_progress("Step 4/4", f"Concatenating {len(concat_files)} runs...")
        concat(input_files=concat_files, output_file=coreg_file, overwrite=overwrite)

    # Erode brain mask (2 voxels) and save to final output location
    emit_progress("Step 4/4", "Eroding and saving brain mask...")
    mask_file = root_dir / Path(f"{participant_id}_task-{task}_preproc_mask.nii.gz")
    erode_mask(mean_mask, mask_file, overwrite=overwrite)

    # Apply mask to preprocessed timeseries (final masked 4D output)
    preproc_file = root_dir / Path(f"{participant_id}_task-{task}_preproc.nii.gz")
    if not preproc_file.exists() or overwrite:
        apply_mask(input_file=coreg_file, mask_file=mask_file, output_file=preproc_file, overwrite=overwrite)

    # Apply mask to BOLD reference (masked mean of first run)
    ref_file = root_dir / Path(f"{participant_id}_task-{task}_boldref.nii.gz")
    if not ref_file.exists() or overwrite:
        apply_mask(input_file=sub_ref, mask_file=mask_file, output_file=ref_file, overwrite=overwrite)

    emit_progress("Step 4/4", "Complete")

    # Generate QC report
    if qc_enabled:
        emit_progress("Generating QC report...")
        try:
            from . import qc
            qc_report = qc.generate_preprocessing_report(
                preprocessing_dir=root_dir,
                participant_id=participant_id,
                task=task,
                overwrite=overwrite
            )
            emit_progress("QC report", f"Saved to {qc_report.name}")
        except Exception as e:
            logger.warning(f"QC report generation failed: {e}")

    # Return preprocessed file
    return preproc_file, mask_file, ref_file


# ==============================================================================
# DICOM Collection - Watch for incoming DICOMs during scanner acquisition
# ==============================================================================


class _CollectorEventHandler(FileSystemEventHandler):
    """Watchdog event handler for DICOM file detection during collection."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def on_created(self, event):
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            self._callback(Path(src_path.decode()))
        elif isinstance(src_path, str):
            self._callback(Path(src_path))


def collect_dicoms(
    watch_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_volumes: int,
    timeout: float = 300.0,
    clear_watch_dir: bool = True,
    clear_output_dir: bool = True,
    simulator_source_dir: Optional[Union[str, Path]] = None,
    simulator_speed: float = 1.0,
    poll_interval: float = 0.5,
    progress_callback: Optional[Callable[[int, int, float, Path], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Path:
    """
    Watch for incoming DICOMs during scanner acquisition and collect them.

    This function mirrors MURFI's resting_state step - it watches an incoming
    directory for DICOMs from the scanner, copies them to an output folder,
    and returns when the expected number of volumes have been collected.

    Works for both resting-state and task data collection.

    Parameters
    ----------
    watch_dir : str or Path
        Directory where scanner exports DICOM files (incoming folder).
    output_dir : str or Path
        Directory to copy collected DICOMs to.
    n_volumes : int
        Number of volumes to collect.
    timeout : float, default=300.0
        Maximum time to wait for all volumes (seconds).
    clear_watch_dir : bool, default=True
        Clear watch_dir before starting (like MURFI's rm -rf tmp/murfi_input).
    clear_output_dir : bool, default=True
        Clear output_dir before starting.
    simulator_source_dir : str or Path, optional
        If provided, runs DicomSimulator to feed files into watch_dir.
        Useful for offline testing.
    simulator_speed : float, default=1.0
        Speed multiplier for simulation (e.g., 10.0 for 10x speed).
        Only used if simulator_source_dir is provided.
    poll_interval : float, default=0.5
        How often to check for new files when queue is empty (seconds).
    progress_callback : callable, optional
        Function called after each volume with signature:
        callback(volumes_collected, n_volumes, elapsed_time, dicom_path).
        Use this to update progress displays during collection.
    stop_event : threading.Event, optional
        Event to signal early termination. If set, collection stops gracefully.

    Returns
    -------
    Path
        Path to output_dir containing collected DICOMs (ready for preprocessing).

    Raises
    ------
    TimeoutError
        If all volumes are not collected within timeout.
    FileNotFoundError
        If watch_dir does not exist (and not using simulator).

    Examples
    --------
    Collect resting-state DICOMs from scanner:

    >>> dicom_folder = collect_dicoms(
    ...     watch_dir='/scanner/export/incoming',
    ...     output_dir='/data/sub-01/raw/rest',
    ...     n_volumes=200,
    ...     clear_watch_dir=True,
    ... )

    Offline testing with simulator:

    >>> dicom_folder = collect_dicoms(
    ...     watch_dir='/tmp/incoming',
    ...     output_dir='/data/sub-01/raw/rest',
    ...     n_volumes=200,
    ...     simulator_source_dir='/path/to/source_dicoms',
    ...     simulator_speed=10.0,
    ... )

    """
    watch_dir = Path(watch_dir)
    output_dir = Path(output_dir)

    # Validate or create directories
    if simulator_source_dir is None:
        if not watch_dir.exists():
            raise FileNotFoundError(f"Watch directory not found: {watch_dir}")
    else:
        watch_dir.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear directories if requested
    if clear_watch_dir and watch_dir.exists():
        for f in watch_dir.iterdir():
            if f.is_file():
                f.unlink()
        logger.info(f"Cleared watch directory: {watch_dir}")

    if clear_output_dir and output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file():
                f.unlink()
        logger.info(f"Cleared output directory: {output_dir}")

    # Collection state
    pending_heap: list[tuple[str, Path]] = []
    processed_files: set[str] = set()
    collected_count = 0
    lock = threading.Lock()

    def on_new_file(file_path: Path) -> None:
        """Add file to pending queue."""
        with lock:
            if file_path.name not in processed_files:
                existing_names = {item[0] for item in pending_heap}
                if file_path.name not in existing_names:
                    heapq.heappush(pending_heap, (file_path.name, file_path))

    def scan_for_existing_files() -> None:
        """Scan watch directory for any files not yet queued."""
        found_files: list[Path] = []
        for ext in DICOM_EXTENSIONS:
            found_files.extend(watch_dir.glob(ext))

        with lock:
            existing_names = {item[0] for item in pending_heap}
            for file_path in found_files:
                if file_path.name not in processed_files:
                    if file_path.name not in existing_names:
                        heapq.heappush(pending_heap, (file_path.name, file_path))
                        existing_names.add(file_path.name)

    def get_next_file() -> Optional[Path]:
        """Get the next valid DICOM file in order."""
        with lock:
            while pending_heap:
                filename, file_path = pending_heap[0]

                if filename in processed_files:
                    heapq.heappop(pending_heap)
                    continue

                if is_valid_dicom(file_path):
                    heapq.heappop(pending_heap)
                    processed_files.add(filename)
                    return file_path
                else:
                    heapq.heappop(pending_heap)
                    logger.debug(f"Skipping invalid DICOM: {filename}")

            return None

    # Set up watchdog observer
    event_handler = _CollectorEventHandler(on_new_file)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()

    # Set up simulator if requested
    simulator = None
    simulator_thread = None
    if simulator_source_dir is not None:
        from .simulator import DicomSimulator

        simulator = DicomSimulator(
            source_dir=simulator_source_dir,
            target_dir=watch_dir,
            speed=simulator_speed,
            clear_target=False,  # We already cleared it
            cleanup_on_complete=False,
        )

        def run_simulator():
            for _ in simulator.copy_volumes_iter(stop_event=stop_event):
                pass

        simulator_thread = threading.Thread(target=run_simulator, daemon=True)
        simulator_thread.start()
        logger.info(f"Started DICOM simulator (speed={simulator_speed}x)")

    logger.info(f"Collecting {n_volumes} volumes from {watch_dir}")
    logger.info(f"Output directory: {output_dir}")

    start_time = time.time()

    try:
        while collected_count < n_volumes:
            # Check for stop signal
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Collection stopped by user after {collected_count}/{n_volumes} volumes")
                break

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout after {elapsed:.1f}s: collected {collected_count}/{n_volumes} volumes"
                )

            # Scan for any missed files
            scan_for_existing_files()

            # Get next file to process
            file_path = get_next_file()

            if file_path is not None:
                # Copy file to output directory
                dest_path = output_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                collected_count += 1

                # Call progress callback
                if progress_callback is not None:
                    elapsed = time.time() - start_time
                    try:
                        progress_callback(collected_count, n_volumes, elapsed, dest_path)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

                if collected_count % 10 == 0 or collected_count == n_volumes:
                    logger.info(f"Collected {collected_count}/{n_volumes} volumes")
            else:
                # Wait for more files
                time.sleep(poll_interval)

    finally:
        # Stop watchdog observer
        observer.stop()
        observer.join()

        # Wait for simulator to finish (skip if stopped early to avoid blocking)
        if simulator_thread is not None:
            if stop_event is not None and stop_event.is_set():
                # Don't wait for simulator when stopped - it's a daemon thread
                logger.debug("Skipping simulator join (stopped by user)")
            else:
                simulator_thread.join()  # Wait for completion
                logger.debug("Simulator thread completed")

        # Clean up watch directory (remove incoming DICOMs)
        if clear_watch_dir and watch_dir.exists():
            for f in watch_dir.iterdir():
                if f.is_file():
                    f.unlink()
            logger.debug(f"Cleaned up watch directory: {watch_dir}")

    elapsed = time.time() - start_time
    logger.info(f"✓ Collection complete: {collected_count} volumes in {elapsed:.1f}s")

    return output_dir
