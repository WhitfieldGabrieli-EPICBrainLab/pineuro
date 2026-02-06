"""
Real-time fMRI preprocessing pipeline for neurofeedback research.

This module handles the complete preprocessing workflow for fMRI data before
neurofeedback sessions. It converts raw DICOM images from the scanner into
analysis-ready NIfTI volumes with motion correction and brain extraction.

The preprocessing pipeline is designed for real-time neurofeedback studies where:
- Multiple functional runs need to be combined (e.g., localizer + feedback runs)
- Speed matters for between-run processing during scanner sessions
- Consistent output structure enables automated downstream processing

Key Components
--------------
preprocessing : Main pipeline function
    Runs the complete 4-step preprocessing workflow.
collect_volumes : DICOM/NIfTI collection function
    Watches for incoming volumes during scanner acquisition.
motion_correction : Motion correction wrapper
    Rigid-body registration using ANTsPy for head motion correction.
brain_extraction : Brain extraction wrapper
    Brain mask extraction using ANTsPy's get_mask.

Workflow
--------
The pipeline follows these steps (with numbered output directories for traceability):

1. **DICOM to NIfTI** (01_nifti/)
   Convert raw DICOMs to NIfTI format using dcm2niix.
   Preserves scanner metadata (TR, slice timing, etc.) in JSON sidecars.

2. **Motion Correction** (02_motion_correction/)
   Estimate and correct head motion using ANTsPy rigid registration.
   Outputs .par files with 6 motion parameters for QC.

3. **Brain Extraction** (03_brain_extraction/)
   Compute temporal mean of each run and extract brain mask using ANTsPy.
   The mask excludes non-brain tissue to improve GLM estimation.

4. **Coregistration** (04_coregistered/)
   Align all runs to the first run's coordinate space using ANTs.
   Concatenate aligned runs into a single 4D timeseries.

Output Files
------------
After preprocessing, the root directory contains:
- {participant}_task-{task}_preproc.nii.gz : Masked, preprocessed 4D timeseries
- {participant}_task-{task}_preproc_mask.nii.gz : Eroded brain mask
- {participant}_task-{task}_boldref.nii.gz : Masked BOLD reference (mean of run-01)

Dependencies
------------
External:
    - dcm2niix - For DICOM conversion

Python:
    - ANTsPy - Motion correction, brain extraction, coregistration
    - nibabel : NIfTI I/O
    - nilearn : Spatial smoothing
    - scipy : Temporal filtering
    - watchdog : Filesystem monitoring for DICOM collection

Notes
-----
**Performance Optimization:**
Motion correction uses `-stages 1` (single refinement stage) which is 1.66x faster
than the default 4 stages while maintaining r=0.9999 correlation with full results.
This tradeoff is appropriate for real-time use where speed matters.

**Overwrite Behavior:**
All functions support `overwrite=False` (the default). When False, existing outputs
are skipped, enabling efficient re-runs of the pipeline. Set `overwrite=True` to
force regeneration of all outputs.

**Parallel Processing:**
Multi-run preprocessing supports parallel execution via `n_jobs`. DICOM conversion
and motion correction run in parallel across runs when n_jobs != 1.

See Also
--------
mask_extraction : Extracts network masks from preprocessed resting-state data
feedback : Real-time activation estimation using preprocessed reference data

References
----------
.. [1] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M.
       (2012). FSL. NeuroImage, 62(2), 782-790.
.. [2] Li, X., Morgan, P. S., Ashburner, J., Smith, J., & Rorden, C. (2016).
       The first step for neuroimaging data analysis: DICOM to NIfTI conversion.
       Journal of Neuroscience Methods, 264, 47-56.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# --- Standard Library ---
# Type hints for function signatures (improves IDE support and documentation)
from typing import Union, List, Optional, Callable
from pathlib import Path
import heapq      # Priority queue for ordered volume collection
import shutil     # File operations (copy during DICOM collection)
import copy       # Deep copy for data structures
import threading  # Multi-threading for volume collection with watchdog
import time       # Timing for collection timeout and progress tracking
import logging    # Structured logging throughout the pipeline

# --- Third-Party: Neuroimaging ---
import numpy as np
import nibabel as nib
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from nilearn.image import smooth_img        # Gaussian spatial smoothing
from scipy.signal import butter, filtfilt   # Butterworth temporal filtering
import ants                                  # Brain extraction via get_mask()

# --- Third-Party: Filesystem Monitoring ---
# Watchdog provides cross-platform filesystem event monitoring for detecting
# new DICOM files as they arrive from the scanner during acquisition
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent

# --- Local Imports ---
from pineuro.utils import (
    # Command execution
    parallel_run_command,  # Parallel execution for multi-run dcm2niix

    # DICOM/NIfTI operations
    dicom_to_nifti,        # dcm2niix wrapper
    is_valid_volume,       # Validate DICOM/NIfTI files
    is_nifti_file,         # Check if file is NIfTI format
    get_json_sidecar_path, # Get path to JSON sidecar for NIfTI

    # Registration (ANTs-backed)
    compute_xfm,           # Compute transformation matrix
    apply_xfm,             # Apply transformation to volume
    compute_mean,          # Compute temporal mean of 4D volume

    # Masking
    apply_mask,            # Apply brain mask to volume
    erode_mask,            # Erode mask to reduce edge effects

    # Timing
    Timer,                 # Performance timing context manager
)

# --- Module Logger ---
# Use __name__ to create a logger hierarchy (pineuro.preprocessing)
# This allows users to configure logging at the pineuro level
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 1: CORE PREPROCESSING FUNCTIONS
# =============================================================================
# These functions handle individual preprocessing steps using ANTsPy.
# Each function follows a consistent pattern:
#   - Validate inputs
#   - Check if outputs exist (skip if overwrite=False)
#   - Run the processing step
#   - Return output paths


def motion_correction(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Perform motion correction on fMRI data using ANTsPy rigid registration.

    Motion correction aligns each volume in the 4D timeseries to a reference
    volume (middle volume by default) to compensate for head movement during
    the scan. This is critical for neurofeedback because uncorrected motion
    can cause spurious activation signals.

    Parameters
    ----------
    input_file : str or Path
        Path to input 4D NIfTI file containing fMRI timeseries.
    output_file : str or Path
        Path for output motion-corrected NIfTI file.
        Parent directory will be created if it doesn't exist.
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if output file already exists.

    Returns
    -------
    None
        Writes output file to disk. Also creates {output_file}.par with
        motion parameters (6 columns: rot_x, rot_y, rot_z, trans_x, trans_y, trans_z).

    Notes
    -----
    **Implementation:**
    Uses ANTsPy rigid registration with a tuned multi-resolution strategy
    (5 iterations at 2x shrink, 2 at full resolution). Benchmarked at ~0.9s
    per volume with r=0.9999 voxel-wise correlation vs mcflirt.

    **Reference Volume:**
    The middle volume is used as the reference (matching mcflirt default).

    See Also
    --------
    preprocessing : Main pipeline function that calls this.
    """
    from scipy.spatial.transform import Rotation

    input_file = Path(input_file)
    output_file = Path(output_file)
    par_file = Path(str(output_file) + '.par')

    # Check if outputs exist
    if not overwrite and output_file.exists() and par_file.exists():
        logger.info(f"Skipping motion correction (outputs exist): {output_file.name}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with Timer('motion_correction', category='preprocessing'):
        logger.info(f"Running motion correction: {input_file.name}")

        # Load 4D image
        img = nib.load(input_file)
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine
        n_vols = data.shape[3]

        # Extract voxel spacing from affine
        spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).tolist()

        # Use middle volume as reference (matches mcflirt default)
        ref_idx = n_vols // 2
        ref_ants = ants.from_numpy(data[..., ref_idx], spacing=spacing)

        corrected = np.zeros_like(data)
        corrected[..., ref_idx] = data[..., ref_idx]
        motion_params = np.zeros((n_vols, 6))

        for i in range(n_vols):
            if i == ref_idx:
                continue

            mov_ants = ants.from_numpy(data[..., i], spacing=spacing)

            result = ants.registration(
                fixed=ref_ants,
                moving=mov_ants,
                type_of_transform='Rigid',
                aff_iterations=(5, 2),
                aff_shrink_factors=(2, 1),
                aff_smoothing_sigmas=(1, 0),
                verbose=False,
            )

            corrected[..., i] = result['warpedmovout'].numpy()

            # Extract motion parameters from transform
            tx = ants.read_transform(result['fwdtransforms'][0])
            params = np.array(tx.parameters)

            if len(params) == 6:
                # Euler3DTransform: [angleX, angleY, angleZ, transX, transY, transZ]
                motion_params[i] = params
            elif len(params) == 12:
                # MatrixOffsetTransform: [M00..M22, tx, ty, tz]
                matrix = params[:9].reshape(3, 3)
                offset = params[9:12]
                fp = tx.fixed_parameters
                fixed_p = np.array(fp) if fp is not None and len(fp) > 0 else np.zeros(3)
                translation = offset + fixed_p - matrix @ fixed_p
                euler = Rotation.from_matrix(matrix).as_euler('xyz')
                motion_params[i] = [euler[0], euler[1], euler[2],
                                    translation[0], translation[1], translation[2]]

        # Save corrected 4D image
        corrected_img = nib.Nifti1Image(corrected, affine, img.header)
        nib.save(corrected_img, output_file)

        # Save motion parameters in .par format (mcflirt-compatible)
        # 6 columns: rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm)
        np.savetxt(par_file, motion_params, fmt='%10.6f')

        logger.info(f"Motion correction complete: {output_file.name} ({n_vols} volumes)")

def concat(
        input_files: List[Union[str, Path]],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Concatenate multiple fMRI volumes along the time axis.

    This function combines multiple 4D NIfTI files (e.g., from separate runs)
    into a single 4D timeseries. Used after coregistration to create a unified
    dataset for ICA-based mask extraction.

    Parameters
    ----------
    input_files : list of str or Path
        List of paths to input 4D NIfTI files. Files are concatenated in the
        order provided, so ensure chronological ordering (run-01, run-02, etc.).
    output_file : str or Path
        Path for output concatenated NIfTI file.
    overwrite : bool, default=False
        If True, always concatenate and overwrite existing output.
        If False, skip if output file already exists.

    Returns
    -------
    None
        Writes concatenated 4D volume to disk.

    Notes
    -----
    **Header Handling:**
    The affine transformation matrix and header are taken from the first input
    file. This assumes all runs are in the same coordinate space (i.e., have
    been coregistered).

    **Memory Usage:**
    All input files are loaded into memory simultaneously for concatenation.
    For very long acquisitions, this may require significant RAM.

    Examples
    --------
    >>> concat(
    ...     input_files=['run-01_mcf.nii.gz', 'run-02_coreg.nii.gz'],
    ...     output_file='all_runs.nii.gz'
    ... )
    """
    output_path = Path(output_file)

    # --- Check for existing output (skip if not overwriting) ---
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping concatenation.")
        return

    logger.info("Concatenating...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load all input images ---
    # nibabel loads headers lazily, but get_fdata() loads the full array
    images = [nib.load(str(f)) for f in input_files]

    # --- Concatenate along time axis (axis=-1 = 4th dimension) ---
    # For 4D volumes with shape (X, Y, Z, T), this appends timepoints
    data_arrays = [img.get_fdata() for img in images]
    concatenated = np.concatenate(data_arrays, axis=-1)

    # --- Create output image ---
    # Use affine and header from first image (assumes all are coregistered)
    first_img = images[0]
    out_img = Nifti1Image(concatenated, first_img.affine, first_img.header)

    save(out_img, output_path)
    logger.info(f"Output: {output_path}")

# =============================================================================
# SECTION 2: OPTIONAL PREPROCESSING STEPS
# =============================================================================
# These functions are not part of the default pipeline but can be applied
# for specific use cases (e.g., smoothing for group-level analysis).


def spatial_smoothing(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        fwhm: float = 5.0,
        overwrite: bool = False
) -> None:
    """
    Apply Gaussian spatial smoothing to fMRI data.

    Spatial smoothing increases signal-to-noise ratio (SNR) by averaging
    neighboring voxels, at the cost of spatial resolution. This is typically
    used for group-level analysis rather than individual neurofeedback.

    Parameters
    ----------
    input_file : str or Path
        Path to input 3D or 4D NIfTI file.
    output_file : str or Path
        Path for output smoothed NIfTI file.
        Parent directory will be created if it doesn't exist.
    fwhm : float, default=5.0
        Full width at half maximum (FWHM) of the Gaussian smoothing kernel
        in millimeters. Common values:
        - 5-6 mm : Standard for single-subject analysis
        - 8 mm : Standard for group analysis
        - 0 or None : No smoothing
    overwrite : bool, default=False
        If True, always smooth and overwrite existing output.
        If False, skip if output file already exists.

    Returns
    -------
    None
        Writes smoothed volume to disk.

    Notes
    -----
    **When to Use:**
    For real-time neurofeedback, smoothing is typically applied during mask
    extraction (before ICA) rather than during preprocessing. The default
    pipeline does NOT apply smoothing.

    **FWHM vs Sigma:**
    FWHM = sigma * 2.355 (for Gaussian kernels). A 6mm FWHM corresponds to
    sigma ≈ 2.55mm.
    """
    output_path = Path(output_file)

    # --- Check for existing output ---
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping spatial smoothing.")
        return

    logger.info("Spatial smoothing...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Apply Gaussian smoothing via nilearn ---
    # nilearn handles 3D and 4D inputs automatically
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
    Apply temporal bandpass filtering to fMRI data.

    Temporal filtering removes unwanted frequency components from the BOLD signal.
    High-pass filtering removes slow drifts (scanner drift, physiological), while
    low-pass filtering removes high-frequency noise.

    Parameters
    ----------
    input_file : str or Path
        Path to input 4D NIfTI file.
    output_file : str or Path
        Path for output filtered NIfTI file.
        Parent directory will be created if it doesn't exist.
    highpass : float, optional
        High-pass filter cutoff in Hz. Removes frequencies below this value.
        Common value: 0.01 Hz (removes drifts slower than 100s).
        If None, no high-pass filtering is applied.
    lowpass : float, optional
        Low-pass filter cutoff in Hz. Removes frequencies above this value.
        Common value: 0.1 Hz (removes noise faster than 10s).
        If None, no low-pass filtering is applied.
    overwrite : bool, default=False
        If True, always filter and overwrite existing output.
        If False, skip if output file already exists.

    Returns
    -------
    None
        Writes filtered volume to disk.

    Notes
    -----
    **Filter Design:**
    Uses a 2nd-order Butterworth filter (same as FSL's default) for smooth
    frequency response without ringing artifacts.

    **Zero-Phase Filtering:**
    The filter is applied forward and backward (filtfilt) to eliminate phase
    distortion. This is important for preserving the timing of neural events.

    **TR Extraction:**
    The repetition time (TR) is read from the NIfTI header to compute the
    sampling frequency. Ensure your NIfTI files have correct TR metadata.

    **When to Use:**
    For resting-state analysis, bandpass filtering (0.01-0.1 Hz) is standard.
    For task-based neurofeedback, filtering may remove task-related signal
    and is generally NOT recommended.
    """
    output_path = Path(output_file)

    # --- Check for existing output ---
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping temporal filtering.")
        return

    # --- Load image and extract timing parameters ---
    img = nib.load(str(input_file))
    data = img.get_fdata()

    # Extract TR from NIfTI header (4th element of zooms is temporal spacing)
    # nibabel's type hints don't recognize this, hence type: ignore
    tr = img.header.get_zooms()[3]  # type: ignore
    fs = 1.0 / tr          # Sampling frequency in Hz
    nyquist = fs / 2.0     # Nyquist frequency (max representable frequency)

    logger.info(f"Temporal filtering (TR={tr}s, fs={fs:.3f}Hz)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Determine filter type based on provided cutoffs ---
    if highpass is not None and lowpass is not None:
        # Bandpass: keep frequencies between highpass and lowpass
        btype = 'bandpass'
        Wn = [highpass / nyquist, lowpass / nyquist]  # Normalize to Nyquist
    elif highpass is not None:
        # High-pass: remove frequencies below highpass (removes slow drifts)
        btype = 'highpass'
        Wn = highpass / nyquist
    elif lowpass is not None:
        # Low-pass: remove frequencies above lowpass (removes high-freq noise)
        btype = 'lowpass'
        Wn = lowpass / nyquist
    else:
        # No filtering requested - save unchanged
        logger.info("No filter cutoffs specified. Saving unchanged.")
        save(img, output_path)
        return

    # --- Design Butterworth filter ---
    # Order 2 matches FSL's default and provides good balance between
    # sharpness (frequency selectivity) and smoothness (minimal ringing)
    b, a = butter(2, Wn, btype=btype)

    # --- Apply filter to each voxel's timeseries ---
    # Reshape 4D (X, Y, Z, T) to 2D (voxels, timepoints) for efficient filtering
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])

    # filtfilt applies filter forward then backward to eliminate phase shift
    # This is critical for task timing - phase distortion would shift events
    filtered_2d = filtfilt(b, a, data_2d, axis=-1)
    filtered = filtered_2d.reshape(original_shape)

    # --- Save filtered output ---
    out_img = Nifti1Image(filtered, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")

# =============================================================================
# SECTION 3: BRAIN EXTRACTION
# =============================================================================


def brain_extraction(
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        overwrite: bool = False
) -> tuple[Path, Path]:
    """
    Perform brain extraction using ANTsPy's get_mask.

    Brain extraction (skull stripping) creates a binary mask that identifies
    brain voxels vs. non-brain tissue (skull, scalp, air). This mask is used
    to restrict analysis to brain tissue, which improves GLM estimation and
    reduces computation.

    Parameters
    ----------
    input_file : str or Path
        Path to input 3D NIfTI file (typically temporal mean of fMRI data).
    output_dir : str or Path
        Directory for output files. Will be created if it doesn't exist.
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if output files already exist.

    Returns
    -------
    brain_file : Path
        Path to brain-extracted volume ({input_name}_brain.nii.gz).
    mask_file : Path
        Path to binary brain mask ({input_name}_brain_mask.nii.gz).

    Notes
    -----
    **Implementation:**
    Uses ANTsPy's get_mask() which applies Otsu thresholding followed by
    morphological cleanup operations. This is faster than FSL's bet (~0.5s
    vs ~1.2s) while achieving ~94% Dice overlap with bet masks.

    **Why Use Temporal Mean:**
    The temporal mean of an fMRI run has better SNR than individual volumes,
    making brain boundaries easier to detect.

    **Mask Erosion:**
    The mask often includes edge voxels that contain partial-volume effects.
    The main preprocessing function erodes the mask by 2 voxels to exclude
    these unreliable edge voxels.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Construct output paths ---
    output_name = input_file.with_suffix("").stem
    brain_file = output_dir / f"{output_name}_brain.nii.gz"
    mask_file = output_dir / f"{output_name}_brain_mask.nii.gz"

    # --- Check if outputs exist ---
    if not overwrite and brain_file.exists() and mask_file.exists():
        logger.info(f"Skipping brain extraction (outputs exist): {mask_file.name}")
        return brain_file, mask_file

    # --- Run brain extraction with ANTsPy ---
    with Timer('brain_extraction', category='preprocessing'):
        logger.info(f"Running brain extraction: {input_file.name}")

        # Load image with ANTs
        img = ants.image_read(str(input_file))

        # Get brain mask using Otsu thresholding + morphological cleanup
        # low_thresh: use mean * 0.5 as lower threshold (similar to bet's approach)
        # cleanup: level of morphological cleanup (2 = moderate)
        mask = ants.get_mask(img, low_thresh=img.numpy().mean() * 0.5, cleanup=2)

        # Apply mask to get brain-extracted volume
        brain = img * mask

        # Save outputs
        ants.image_write(brain, str(brain_file))
        ants.image_write(mask, str(mask_file))

        logger.info(f"Brain extraction complete: {mask_file.name}")

    return brain_file, mask_file

# =============================================================================
# SECTION 4: MAIN PREPROCESSING PIPELINE
# =============================================================================


def preprocessing(
        participant_id: str,
        task: str,
        root_dir: Union[str, Path],
        input_dicoms: Union[str, Path, List[Union[str, Path]]],
        overwrite: bool = False,
        n_jobs: int = -1,
        qc_enabled: bool = True,
        progress_callback: Optional[Callable[[str, str], None]] = None
) -> tuple[Path, Path, Path]:
    """
    Run the complete preprocessing pipeline for fMRI data.

    This is the main entry point for preprocessing. It executes the full 4-step
    pipeline: DICOM conversion, motion correction, brain extraction, and
    multi-run coregistration. Outputs are written to numbered subdirectories
    for easy traceability.

    Parameters
    ----------
    participant_id : str
        Participant identifier (e.g., 'sub-01'). Used in output filenames.
    task : str
        Task name (e.g., 'rest', 'nback'). Used in output filenames following
        BIDS naming conventions.
    root_dir : str or Path
        Root directory for all preprocessing outputs. Subdirectories
        (01_nifti/, 02_motion_correction/, etc.) are created here.
    input_dicoms : str or Path or list of str or Path
        Path(s) to DICOM folder(s). Can be:
        - Single path: One run of data
        - List of paths: Multiple runs to be coregistered and concatenated
    overwrite : bool, default=False
        If True, regenerate all outputs even if they exist.
        If False, skip steps where outputs already exist (faster re-runs).
    n_jobs : int, default=-1
        Number of parallel jobs for DICOM conversion and motion correction.
        - -1: Use all available CPU cores
        - 1: Sequential processing (useful for debugging)
        - N: Use exactly N cores
    qc_enabled : bool, default=True
        If True, generate HTML QC report after preprocessing completes.
        Report includes motion timeseries, framewise displacement, and
        brain mask coverage visualizations.
    progress_callback : callable, optional
        Callback function for progress updates with signature:
        ``callback(step: str, detail: str)``.
        Used by GUI to display real-time progress.

    Returns
    -------
    preproc_file : Path
        Path to preprocessed 4D timeseries (masked, motion-corrected,
        coregistered if multi-run).
    mask_file : Path
        Path to eroded brain mask (binary NIfTI).
    ref_file : Path
        Path to BOLD reference image (masked mean of first run).

    Raises
    ------
    FileNotFoundError
        If any DICOM folder does not exist or is empty.

    Notes
    -----
    **Output Directory Structure:**
    ::

        root_dir/
        ├── 01_nifti/                    # DICOM → NIfTI conversion
        │   └── {participant}_task-{task}_run-{N}.nii.gz
        ├── 02_motion_correction/        # Motion-corrected volumes
        │   ├── {participant}_task-{task}_run-{N}_mcf.nii.gz
        │   └── {participant}_task-{task}_run-{N}_mcf.nii.gz.par
        ├── 03_brain_extraction/         # Temporal means and brain mask
        │   ├── {participant}_task-{task}_run-{N}_mean.nii.gz
        │   └── {participant}_task-{task}_run-01_mean_brain_mask.nii.gz
        ├── 04_coregistered/             # Multi-run alignment (if >1 run)
        │   ├── {participant}_task-{task}.nii.gz
        │   └── xfm/                     # ANTs transform files
        ├── {participant}_task-{task}_preproc.nii.gz      # Final masked output
        ├── {participant}_task-{task}_preproc_mask.nii.gz # Eroded brain mask
        ├── {participant}_task-{task}_boldref.nii.gz      # BOLD reference
        └── qc/                          # QC report (if enabled)
            └── preprocessing_qc.html

    **Multi-Run Coregistration:**
    When multiple DICOM folders are provided, all runs are registered to the
    first run's coordinate space using rigid-body (6 DOF) ANTs registration.
    This ensures voxel correspondence across runs for concatenation.

    **Mask Erosion:**
    The brain mask is eroded by 2 voxels to exclude edge voxels that may
    contain partial-volume effects or be affected by motion artifacts.

    Examples
    --------
    Single run preprocessing:

    >>> preproc_file, mask_file, ref_file = preprocessing(
    ...     participant_id='sub-01',
    ...     task='rest',
    ...     root_dir='/data/sub-01/preproc',
    ...     input_dicoms='/scanner/sub-01/rest_dicoms',
    ... )

    Multi-run preprocessing (coregistered):

    >>> preproc_file, mask_file, ref_file = preprocessing(
    ...     participant_id='sub-01',
    ...     task='rest',
    ...     root_dir='/data/sub-01/preproc',
    ...     input_dicoms=[
    ...         '/scanner/sub-01/rest_run1_dicoms',
    ...         '/scanner/sub-01/rest_run2_dicoms',
    ...     ],
    ...     n_jobs=4,  # Use 4 cores for parallel processing
    ... )

    See Also
    --------
    collect_volumes : Collect DICOMs from scanner before preprocessing.
    mask_extraction : Extract network masks from preprocessed resting-state data.
    """

    # -------------------------------------------------------------------------
    # Setup: Progress reporting helper
    # -------------------------------------------------------------------------
    # This helper function sends progress updates to both the Python logger
    # and an optional GUI callback. This dual output ensures progress is
    # visible in terminal mode (via logger) and GUI mode (via callback).

    def emit_progress(step: str, detail: str = ""):
        """Emit progress to logger and optional GUI callback."""
        if detail:
            logger.info(f"{step}: {detail}")
        else:
            logger.info(step)
        if progress_callback:
            progress_callback(step, detail)

    emit_progress(f"Running preprocessing for {participant_id}")

    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    # Validate all input DICOM directories before starting any processing.
    # This catches common issues (wrong path, empty folder) early rather than
    # failing mid-pipeline.

    root_dir = Path(root_dir)

    # Normalize input_dicoms to a list (single path becomes a 1-element list)
    if isinstance(input_dicoms, (str, Path)):
        input_dicoms = [input_dicoms]

    # Validate each input directory
    input_paths: list[str] = []
    logger.info(f"Input DICOM folders:\n")
    for f in input_dicoms:
        path = Path(f)

        # Check existence
        if not path.exists():
            raise FileNotFoundError(f"Folder not found: {path}")

        # Check it's a directory (not a file)
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")

        # Check it contains at least one file (catches common "wrong path" errors)
        if not any(path.iterdir()):
            raise FileNotFoundError(f"Directory is empty: {path}")

        input_paths.append(str(path))
        logger.info(f"  {str(path)}")

    # =========================================================================
    # STEP 1: DICOM TO NIFTI CONVERSION
    # =========================================================================
    # Convert raw DICOM files from the scanner into NIfTI format (.nii.gz).
    # NIfTI is the standard format for neuroimaging analysis, supported by
    # FSL, ANTs, nilearn, and other tools.
    #
    # Why dcm2niix?
    # - Preserves all scanner metadata (TR, slice timing, phase encoding)
    # - Handles all major scanner vendors (Siemens, GE, Philips)
    # - Outputs JSON sidecars with BIDS-compatible metadata

    emit_progress("Step 1/4: DICOM to NIfTI conversion")
    nifti_dir = root_dir / "01_nifti"
    nifti_files = []

    # --- Parallel Conversion (for multi-run datasets) ---
    # When we have multiple runs AND parallel processing is enabled,
    # convert all runs simultaneously. This is significantly faster
    # for multi-run datasets on multi-core machines.
    if n_jobs != 1 and len(input_paths) > 1:
        emit_progress("Step 1/4", f"Converting {len(input_paths)} runs in parallel...")

        # Build job specifications for parallel execution
        dcm_jobs = []
        nifti_file_paths = []
        for run, path in enumerate(input_paths):
            # BIDS-style naming: sub-XX_task-YY_run-01.nii.gz
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
                'verbose': False  # Suppress per-job output in parallel mode
            })

        # Execute all jobs in parallel using joblib
        parallel_run_command(dcm_jobs, n_jobs=n_jobs, timing_category='preprocessing')
        nifti_files = nifti_file_paths

    else:
        # --- Sequential Conversion (single run or n_jobs=1) ---
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

    # =========================================================================
    # STEP 2: MOTION CORRECTION
    # =========================================================================
    # Estimate and correct for head motion during scanning. Even small head
    # movements (1-2mm) can cause significant artifacts in fMRI data because
    # the BOLD signal changes we're measuring are only 1-5% of the baseline.
    #
    # How motion correction works:
    # 1. Select reference volume (middle of the timeseries by default)
    # 2. For each volume, estimate 6 rigid-body parameters (3 translation, 3 rotation)
    #    using ANTsPy rigid registration with tuned multi-resolution strategy
    # 3. Apply transformation to align each volume to reference
    #
    # The .par file contains the motion parameters for QC visualization.

    emit_progress("Step 2/4: Motion correction (ANTsPy)")
    mcf_dir = root_dir / "02_motion_correction"
    mcf_files = []

    for run, nifti in enumerate(nifti_files):
        emit_progress("Step 2/4", f"Running motion correction on run {run+1}/{len(nifti_files)}...")
        mcf_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mcf.nii.gz")
        motion_correction(input_file=nifti, output_file=mcf_file, overwrite=overwrite)
        mcf_files.append(mcf_file)

    emit_progress("Step 2/4", f"Complete ({len(mcf_files)} runs corrected)")

    # =========================================================================
    # STEP 3: BRAIN EXTRACTION
    # =========================================================================
    # Create a brain mask to exclude non-brain tissue from analysis.
    # This step has two sub-parts:
    #
    # 1. Compute temporal mean of each run
    #    - Averaging over time improves SNR and makes brain boundaries clearer
    #    - These means are also used as reference images for coregistration
    #
    # 2. Extract brain mask from first run
    #    - ANTsPy get_mask identifies brain vs. non-brain voxels
    #    - We only run extraction on the first run because all runs will be aligned
    #      to the same space (via coregistration in Step 4)

    emit_progress("Step 3/4: Brain extraction (ANTsPy)")
    bet_dir = root_dir / "03_brain_extraction"

    # --- Compute temporal mean of each motion-corrected run ---
    # The temporal mean has better SNR than individual volumes, making it
    # easier to find accurate brain boundaries. These means also
    # serve as reference images for multi-run coregistration.
    emit_progress("Step 3/4", "Computing temporal mean images...")
    mean_files = []
    for run, nifti in enumerate(mcf_files):
        mean_file = bet_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean.nii.gz")
        compute_mean(input_file=nifti, output_file=mean_file, overwrite=overwrite)
        mean_files.append(mean_file)

    # --- Extract brain mask from first run ---
    # We use run-01 as the reference space for the entire dataset.
    # All subsequent runs will be aligned to this space, so we only need
    # one brain mask (created from run-01's mean).
    emit_progress("Step 3/4", "Running brain extraction...")
    sub_ref = mean_files[0]  # Run-01 mean serves as the subject reference
    _, mean_mask = brain_extraction(input_file=sub_ref, output_dir=bet_dir, overwrite=overwrite)
    emit_progress("Step 3/4", "Complete (brain mask extracted)")

    # =========================================================================
    # STEP 4: COREGISTRATION AND FINALIZATION
    # =========================================================================
    # For multi-run datasets, align all runs to a common coordinate space
    # (the first run), then concatenate into a single 4D timeseries.
    #
    # Why coregister?
    # - Between runs, the participant may shift position in the scanner
    # - Without coregistration, the same brain region would be at different
    #   voxel coordinates in different runs
    # - ICA and other analyses require consistent voxel correspondence
    #
    # Coregistration method:
    # - Rigid-body (6 DOF): 3 translations + 3 rotations
    # - This preserves brain shape while correcting for position changes
    # - We use ANTs for robust, accurate registration

    emit_progress("Step 4/4: Coregistration and finalization")
    coreg_dir = root_dir / "04_coregistered"
    coreg_file = coreg_dir / Path(f"{participant_id}_task-{task}.nii.gz")
    xfm_dir = coreg_dir / "xfm"  # Store ANTs transform files

    if len(mcf_files) == 1:
        # --- Single Run: No Coregistration Needed ---
        # When there's only one run, skip coregistration entirely.
        # The motion-corrected data is used directly for downstream analysis.
        emit_progress("Step 4/4", "Single run - skipping coregistration")
        coreg_file = mcf_files[0]

    else:
        # --- Multiple Runs: Coregister to First Run ---
        # Run-01 serves as the reference. All subsequent runs are aligned to it.
        emit_progress("Step 4/4", f"Coregistering {len(mcf_files)} runs to run 1...")
        sub_ref = mean_files[0]  # Run-01 mean is the target for registration
        concat_files = []

        for run, mean_file in enumerate(mean_files):

            if run == 0:
                # First run IS the reference - no transformation needed
                # Just add its motion-corrected data to the concatenation list
                concat_files.append(mcf_files[run])

            else:
                # --- Compute Rigid Transform ---
                # Estimate the 6 rigid-body parameters (3 translation, 3 rotation)
                # that align this run's mean to run-01's mean
                emit_progress("Step 4/4", f"Computing transform for run {run+1}/{len(mean_files)}...")
                output_prefix = xfm_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean_to_run-01_mean")
                _, xfm_files = compute_xfm(
                    moving=mean_file,       # This run's mean (to be transformed)
                    fixed=sub_ref,          # Run-01 mean (target space)
                    output_prefix=output_prefix,
                    type_of_transform='Rigid',  # 6 DOF: preserves brain shape
                    overwrite=overwrite
                )

                # --- Apply Transform to Full 4D Timeseries ---
                # The transform was computed on the mean, but we apply it to the
                # full 4D timeseries so all volumes are in the same space
                emit_progress("Step 4/4", f"Applying transform to run {run+1}/{len(mean_files)}...")
                output_file = coreg_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_to_run-01.nii.gz")
                apply_xfm(
                    moving=mcf_files[run],  # Full 4D timeseries
                    fixed=sub_ref,          # Reference space
                    transform_files=xfm_files,
                    output_file=output_file,
                    interpolator='linear',  # Linear interpolation for smooth results
                    overwrite=overwrite
                )
                concat_files.append(output_file)

        # --- Concatenate All Runs ---
        # Combine all coregistered runs into a single 4D timeseries.
        # The order is chronological (run-01, run-02, ...).
        emit_progress("Step 4/4", f"Concatenating {len(concat_files)} runs...")
        concat(input_files=concat_files, output_file=coreg_file, overwrite=overwrite)

    # -------------------------------------------------------------------------
    # Finalization: Create Final Output Files
    # -------------------------------------------------------------------------
    # Generate the three main outputs that downstream processing expects:
    # 1. Masked preprocessed timeseries (for ICA, feedback estimation)
    # 2. Eroded brain mask (for masking operations)
    # 3. BOLD reference (for registration, overlay visualization)

    # --- Erode Brain Mask ---
    # bet's mask often includes edge voxels with partial-volume effects.
    # Erosion by 2 voxels removes these unreliable edge voxels, which
    # improves GLM estimation by excluding noisy boundary regions.
    emit_progress("Step 4/4", "Eroding and saving brain mask...")
    mask_file = root_dir / Path(f"{participant_id}_task-{task}_preproc_mask.nii.gz")
    erode_mask(mean_mask, mask_file, overwrite=overwrite)

    # --- Apply Mask to Preprocessed Timeseries ---
    # Zero out non-brain voxels to reduce file size and exclude them from
    # downstream analysis. This creates the final analysis-ready 4D file.
    preproc_file = root_dir / Path(f"{participant_id}_task-{task}_preproc.nii.gz")
    if not preproc_file.exists() or overwrite:
        apply_mask(input_file=coreg_file, mask_file=mask_file, output_file=preproc_file, overwrite=overwrite)

    # --- Create BOLD Reference ---
    # The masked mean of run-01 serves as a reference image for:
    # - Registration to MNI space (mask extraction)
    # - Overlay visualization (QC reports)
    # - 2-volume coregistration (feedback sessions)
    ref_file = root_dir / Path(f"{participant_id}_task-{task}_boldref.nii.gz")
    if not ref_file.exists() or overwrite:
        apply_mask(input_file=sub_ref, mask_file=mask_file, output_file=ref_file, overwrite=overwrite)

    emit_progress("Step 4/4", "Complete")

    # -------------------------------------------------------------------------
    # QC Report Generation (Optional)
    # -------------------------------------------------------------------------
    # Generate an HTML report with visualizations for quality control:
    # - Motion parameter timeseries (detect sudden movements)
    # - Framewise displacement (quantify motion severity)
    # - Brain mask coverage (verify brain extraction quality)
    # - Multi-run coregistration overlays (verify alignment)

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
            # QC failure shouldn't stop preprocessing from succeeding
            logger.warning(f"QC report generation failed: {e}")

    # -------------------------------------------------------------------------
    # Return Output Paths
    # -------------------------------------------------------------------------
    return preproc_file, mask_file, ref_file


# =============================================================================
# SECTION 5: VOLUME COLLECTION (REAL-TIME SCANNER INTERFACE)
# =============================================================================
# These functions handle real-time volume collection from the scanner.
# During an fMRI acquisition, the scanner exports DICOM or NIfTI files
# one at a time to an "incoming" directory. These functions watch that
# directory and copy files as they arrive, enabling:
#
# - Real-time neurofeedback: Process volumes immediately after acquisition
# - Offline simulation: Test pipelines without scanner access
# - Resting-state collection: Gather all volumes before preprocessing
#
# The collector uses filesystem monitoring (watchdog) for responsive
# detection of new files, combined with periodic scanning for reliability.


class _CollectorEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that detects new files in the watch directory.

    This handler is triggered by the operating system's filesystem event
    notification system (inotify on Linux, FSEvents on macOS, etc.), providing
    near-instant detection of new files.

    Parameters
    ----------
    callback : callable
        Function to call when a new file is detected.
        Called with Path object for the new file.

    Notes
    -----
    We use filesystem events for speed (sub-second detection), but also do
    periodic scanning as a backup because filesystem events can be missed
    under heavy I/O load or network filesystem issues.
    """

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def on_created(self, event):
        """Handle file creation events from the filesystem."""
        # Ignore directory creation events (we only want files)
        if event.is_directory:
            return

        # Handle both str and bytes paths (Windows vs Unix compatibility)
        src_path = event.src_path
        if isinstance(src_path, bytes):
            self._callback(Path(src_path.decode()))
        elif isinstance(src_path, str):
            self._callback(Path(src_path))


def collect_volumes(
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
    Watch for incoming volumes (DICOM or NIfTI) during scanner acquisition and collect them.

    This function mirrors MURFI's resting_state step - it watches an incoming
    directory for volumes from the scanner, copies them to an output folder,
    and returns when the expected number of volumes have been collected.

    Supports both DICOM and NIfTI input files. For NIfTI files, also copies
    the JSON sidecar file if present.

    Works for both resting-state and task data collection.

    Parameters
    ----------
    watch_dir : str or Path
        Directory where scanner exports volume files (incoming folder).
    output_dir : str or Path
        Directory to copy collected volumes to.
    n_volumes : int
        Number of volumes to collect.
    timeout : float, default=300.0
        Maximum time to wait for all volumes (seconds).
    clear_watch_dir : bool, default=True
        Clear watch_dir before starting (like MURFI's rm -rf tmp/murfi_input).
    clear_output_dir : bool, default=True
        Clear output_dir before starting.
    simulator_source_dir : str or Path, optional
        If provided, runs VolumeSimulator to feed files into watch_dir.
        Useful for offline testing. Supports both DICOM and NIfTI source files.
    simulator_speed : float, default=1.0
        Speed multiplier for simulation (e.g., 10.0 for 10x speed).
        Only used if simulator_source_dir is provided.
    poll_interval : float, default=0.5
        How often to check for new files when queue is empty (seconds).
    progress_callback : callable, optional
        Function called after each volume with signature:
        callback(volumes_collected, n_volumes, elapsed_time, volume_path).
        Use this to update progress displays during collection.
    stop_event : threading.Event, optional
        Event to signal early termination. If set, collection stops gracefully.

    Returns
    -------
    Path
        Path to output_dir containing collected volumes (ready for preprocessing).

    Raises
    ------
    TimeoutError
        If all volumes are not collected within timeout.
    FileNotFoundError
        If watch_dir does not exist (and not using simulator).

    Examples
    --------
    Collect resting-state DICOMs from scanner:

    >>> dicom_folder = collect_volumes(
    ...     watch_dir='/scanner/export/incoming',
    ...     output_dir='/data/sub-01/raw/rest',
    ...     n_volumes=200,
    ...     clear_watch_dir=True,
    ... )

    Offline testing with simulator (supports DICOM or NIfTI source):

    >>> volume_folder = collect_volumes(
    ...     watch_dir='/tmp/incoming',
    ...     output_dir='/data/sub-01/raw/rest',
    ...     n_volumes=200,
    ...     simulator_source_dir='/path/to/source_volumes',
    ...     simulator_speed=10.0,
    ... )

    """
    # -------------------------------------------------------------------------
    # Directory Setup and Validation
    # -------------------------------------------------------------------------
    watch_dir = Path(watch_dir)
    output_dir = Path(output_dir)

    # --- Validate/Create Directories ---
    # In real-time mode (no simulator), watch_dir must exist (scanner export folder).
    # In simulation mode, we create watch_dir if needed.
    if simulator_source_dir is None:
        # Real-time mode: watch_dir must be pre-configured by scanner
        if not watch_dir.exists():
            raise FileNotFoundError(f"Watch directory not found: {watch_dir}")
    else:
        # Simulation mode: create watch_dir to receive simulated volumes
        watch_dir.mkdir(parents=True, exist_ok=True)

    # Always create output_dir (where we'll store collected volumes)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Clear Directories (Optional) ---
    # Clearing ensures we start fresh without leftover files from previous runs.
    # For real-time use, this mimics MURFI's "rm -rf tmp/murfi_input" pattern.
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

    # -------------------------------------------------------------------------
    # Collection State
    # -------------------------------------------------------------------------
    # We use a min-heap (priority queue) to process files in alphabetical order.
    # This is important because scanner naming conventions (img001, img002, ...)
    # encode chronological order, and we need to maintain this order.
    #
    # Thread safety: Multiple threads access this state (watchdog handler,
    # main loop, scanner callback), so all access is protected by a lock.

    pending_heap: list[tuple[str, Path]] = []  # (filename, path) tuples sorted by filename
    processed_files: set[str] = set()           # Track processed files to avoid duplicates
    collected_count = 0
    lock = threading.Lock()

    def on_new_file(file_path: Path) -> None:
        """
        Callback invoked by watchdog when a new file appears.

        Adds the file to the pending queue for ordered processing.
        Thread-safe: acquires lock before modifying shared state.
        """
        with lock:
            # Skip if already processed (shouldn't happen but defensive)
            if file_path.name not in processed_files:
                # Check if already in queue (watchdog may fire multiple times)
                existing_names = {item[0] for item in pending_heap}
                if file_path.name not in existing_names:
                    # Add to heap - files are sorted by filename (alphabetical order)
                    heapq.heappush(pending_heap, (file_path.name, file_path))

    def scan_for_existing_files() -> None:
        """
        Periodically scan watch directory for any missed files.

        Filesystem event notifications can be missed under heavy load or on
        network filesystems. This backup scan ensures we eventually find all
        files even if watchdog misses some events.
        """
        # Get all files in watch directory (supports extensionless DICOMs)
        found_files = [f for f in watch_dir.iterdir() if f.is_file()]

        with lock:
            existing_names = {item[0] for item in pending_heap}
            for file_path in found_files:
                if file_path.name not in processed_files:
                    if file_path.name not in existing_names:
                        heapq.heappush(pending_heap, (file_path.name, file_path))
                        existing_names.add(file_path.name)

    def get_next_file() -> Optional[Path]:
        """
        Get the next valid volume file to process, in alphabetical order.

        Returns None if no valid files are pending. Files are validated
        (DICOM header check or NIfTI format check) before being returned.
        Invalid files are logged and skipped.
        """
        with lock:
            while pending_heap:
                # Peek at the smallest (alphabetically first) file
                filename, file_path = pending_heap[0]

                # Skip if already processed (defensive check)
                if filename in processed_files:
                    heapq.heappop(pending_heap)
                    continue

                # Validate the file (check DICOM header or NIfTI format)
                if is_valid_volume(file_path):
                    heapq.heappop(pending_heap)
                    processed_files.add(filename)
                    return file_path
                else:
                    # Invalid file (corrupt, incomplete, or not a volume)
                    heapq.heappop(pending_heap)
                    logger.debug(f"Skipping invalid volume: {filename}")

            return None

    # -------------------------------------------------------------------------
    # Filesystem Watcher Setup
    # -------------------------------------------------------------------------
    # Use watchdog to monitor the watch directory for new files.
    # Watchdog uses the OS's native filesystem event system (inotify on Linux,
    # FSEvents on macOS) for responsive, low-overhead file detection.

    event_handler = _CollectorEventHandler(on_new_file)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)  # No subdirectories
    observer.start()

    # -------------------------------------------------------------------------
    # Simulator Setup (Optional - for Offline Testing)
    # -------------------------------------------------------------------------
    # When simulator_source_dir is provided, we spin up a VolumeSimulator
    # that copies files from the source at TR intervals, mimicking scanner
    # acquisition. This enables testing the full pipeline without scanner access.

    simulator = None
    simulator_thread = None
    if simulator_source_dir is not None:
        from .simulator import VolumeSimulator

        # Create simulator with user-specified speed multiplier
        # speed=10.0 means 10x real-time (TR/10 delay between volumes)
        simulator = VolumeSimulator(
            source_dir=simulator_source_dir,
            target_dir=watch_dir,
            speed=simulator_speed,
            clear_target=False,      # Already cleared above
            cleanup_on_complete=False,  # Leave files for collection
        )

        def run_simulator():
            """Background thread that runs the simulator."""
            for _ in simulator.copy_volumes_iter(stop_event=stop_event):
                pass  # Iterator handles timing and copying

        # Run simulator in a daemon thread (automatically stops when main exits)
        simulator_thread = threading.Thread(target=run_simulator, daemon=True)
        simulator_thread.start()
        logger.info(f"Started {simulator.source_type} simulator (speed={simulator_speed}x)")

    logger.info(f"Collecting {n_volumes} volumes from {watch_dir}")
    logger.info(f"Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # Main Collection Loop
    # -------------------------------------------------------------------------
    # Poll for new files until we've collected the expected number of volumes.
    # The loop combines filesystem events (fast) with periodic scanning (reliable)
    # to ensure we never miss files.

    start_time = time.time()

    try:
        while collected_count < n_volumes:
            # --- Check for Early Termination ---
            # The stop_event allows external code (e.g., GUI "Stop" button) to
            # gracefully terminate collection before all volumes arrive
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Collection stopped by user after {collected_count}/{n_volumes} volumes")
                break

            # --- Check Timeout ---
            # Prevent infinite waiting if scanner stops sending files
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout after {elapsed:.1f}s: collected {collected_count}/{n_volumes} volumes"
                )

            # --- Periodic Backup Scan ---
            # Filesystem events can be missed under heavy load or on network
            # filesystems. This backup scan ensures we eventually find all files.
            scan_for_existing_files()

            # --- Get Next File (in alphabetical/chronological order) ---
            file_path = get_next_file()

            if file_path is not None:
                # --- Copy File to Output Directory ---
                # We copy rather than move to leave the scanner's export intact
                # (some scanner software expects files to remain)
                dest_path = output_dir / file_path.name
                shutil.copy2(file_path, dest_path)  # copy2 preserves metadata

                # --- Handle JSON Sidecars for NIfTI Files ---
                # dcm2niix outputs create JSON sidecars with acquisition parameters.
                # If collecting NIfTI files (from dcm2niix), copy the sidecar too.
                if is_nifti_file(file_path):
                    json_src = get_json_sidecar_path(file_path)
                    if json_src.exists():
                        json_dst = get_json_sidecar_path(dest_path)
                        shutil.copy2(json_src, json_dst)

                collected_count += 1

                # --- Progress Callback ---
                # Notify external code (GUI) of progress
                if progress_callback is not None:
                    elapsed = time.time() - start_time
                    try:
                        progress_callback(collected_count, n_volumes, elapsed, dest_path)
                    except Exception as e:
                        # Don't let callback errors stop collection
                        logger.warning(f"Progress callback error: {e}")

                # Log progress every 10 volumes (avoid log spam)
                if collected_count % 10 == 0 or collected_count == n_volumes:
                    logger.info(f"Collected {collected_count}/{n_volumes} volumes")
            else:
                # --- Wait for More Files ---
                # No files ready yet; sleep briefly before checking again
                time.sleep(poll_interval)

    finally:
        # -------------------------------------------------------------------------
        # Cleanup
        # -------------------------------------------------------------------------
        # Always clean up resources, even if collection was interrupted

        # Stop filesystem watcher
        observer.stop()
        observer.join()

        # Wait for simulator thread (if running) to complete
        if simulator_thread is not None:
            if stop_event is not None and stop_event.is_set():
                # User stopped early - don't wait for simulator (it's a daemon thread)
                logger.debug("Skipping simulator join (stopped by user)")
            else:
                # Normal completion - wait for simulator to finish
                simulator_thread.join()
                logger.debug("Simulator thread completed")

        # Clean up watch directory (remove leftover incoming files)
        # This prevents stale files from affecting future runs
        if clear_watch_dir and watch_dir.exists():
            for f in watch_dir.iterdir():
                if f.is_file():
                    f.unlink()
            logger.debug(f"Cleaned up watch directory: {watch_dir}")

    elapsed = time.time() - start_time
    logger.info(f"Collection complete: {collected_count} volumes in {elapsed:.1f}s")

    return output_dir


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================
# Old name for collect_volumes - kept for existing code that uses it
collect_dicoms = collect_volumes
