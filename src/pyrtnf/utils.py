"""
Utilities and command execution framework for pyrtnf.

This module provides centralized command execution, FSL wrappers, and utility
functions for the pyrtnf package. All external commands (FSL, dcm2niix) are
routed through a common execution interface with validation and logging.

Key Functions
-------------
run_command : Universal command execution with validation and overwrite control
dicom_to_nifti : Convert DICOM files to NIfTI format using dcm2niix
get_yeo_labels : Get Yeo network labels for 7 or 17 network parcellations
compute_mean : Compute temporal mean of fMRI data (FSL fslmaths)
compute_xfm : Compute transformation matrix using FSL FLIRT
apply_xfm : Apply transformation matrix using FSL FLIRT
apply_xfm_sform_to_mni : Apply sform/qform transformation to MNI space
invert_xfm : Invert transformation matrix using FSL convert_xfm
compute_xfm_to_mni : Register to MNI152 space using templateflow BOLD reference
apply_mask : Apply binary mask to volume (FSL fslmaths)

Dependencies
------------
- FSL (FMRIB Software Library) - for registration and image manipulation
- dcm2niix - for DICOM to NIfTI conversion
- templateflow - for MNI152 template access
- numpy, pathlib

Notes
-----
The run_command() function provides centralized handling for all external
commands with features including:
- Automatic command availability checking
- Input/output file validation
- Overwrite control (skip if outputs exist)
- Comprehensive error logging
- Subprocess execution with error handling

All FSL commands use standardized flags (-noresample, -noresampblur) for
consistent behavior across the pipeline.

References
----------
[1] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith,
    S. M. (2012). FSL. NeuroImage, 62(2), 782-790.
[2] Li, X., Morgan, P. S., Ashburner, J., Smith, J., & Rorden, C. (2016).
    The first step for neuroimaging data analysis: DICOM to NIfTI conversion.
    Journal of Neuroscience Methods, 264, 47-56.
"""

# Standard library imports
import shutil
from typing import Union, List, Optional, cast
from pathlib import Path
import subprocess
import tempfile

# Third-party imports
import numpy as np
import nibabel as nib
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
from templateflow import api as tflow

# Local imports

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

# ==============================================================
# Installation Check
# ==============================================================

class CommandNotFoundError(Exception):
    """Raised when an FSL command is not installed or found in PATH."""

def _check_command_installation(cmd: Union[str, Path] = 'fsl') -> None:
    """
    Check if command is installed and available in PATH.

    Parameters
    ----------
    cmd
        Command to look for
    
    Raises
    ------
    CommandNotFoundError
        If command is not found
        
    """

    if shutil.which(cmd) is None:
        raise CommandNotFoundError(
            f"{cmd} is not found in PATH."
        )
    
    logger.debug(f"{cmd} installation verified")

# ==============================================================
#  Helper Functions
# ==============================================================
def run_command(
        cmd_name: str,
        cmd: List[str],
        description: str,
        inputs: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        outputs: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        overwrite: bool = False,
        verbose: bool = True
) -> None:
    """
    Run a shell command with error handling and logging

    Parameters
    ----------
    cmd_name : str
        Name of command to execute
    cmd : list of str
        Command and arguments to execute
    description : str
        Human readable description of what the command does (for logging)
    inputs : list of str or Path, optional
        Paths to files/folders that should exist before command starts
    outputs : list of str or Path, optional
        Paths to files/folders that should exist after command completes
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    verbose : bool, default=True
        If False, suppresses informational logging messages.

    Raises
    ------
    FileNotFoundError
        If input files are not found
    RuntimeError
        If command fails or if expected output files are not created

    """

    # Log the command
    if verbose:
        logger.info(f"{description}...")
    
    # Convert outputs to list if single file provided
    if isinstance(outputs, (str, Path)):
        outputs = [outputs]
    
    # Check if outputs exist and skip if overwrite=False
    if not overwrite and outputs is not None:
        output_paths_check = [Path(f) for f in outputs]
        if all(p.exists() for p in output_paths_check):
            if verbose:
                logger.info(f"All output files exist and overwrite=False. Skipping {description}.")
            return
    
    # Log the command
    if verbose:
        logger.info(f"Running command: {' '.join(cmd)}")

    # Check if command is installed
    _check_command_installation(cmd_name)
    
    # Convert inputs to list if single file provided
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    # Convert to Path objects and validate
    input_paths: list[Path] = []
    if inputs is not None:
        if verbose:
            logger.info(f"Input files:")
        for f in inputs:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            input_paths.append(path)
            if verbose:
                logger.info(f"  {f}")
    output_paths: list[Path] = []
    if outputs is not None:
        if verbose:
            logger.info(f"Output files:")
        for f in outputs:
            path = Path(f)
            (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
            output_paths.append(path)
            if verbose:
                logger.info(f"  {f}")

    # Run command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Log output
        if result.stdout and verbose:
            logger.debug(f"stdout:\n{result.stdout}")
        if result.stderr and verbose:
            logger.debug(f"stderr:\n{result.stderr}")
        
        if verbose:
            logger.info(f"{description} completed successfully")
    
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"{description} failed:\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {e.returncode}\n"
            f"stdout: {e.stdout}"
            f"stderr: {e.stderr}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # Verify output files if specified
    if outputs is not None:
        missing_files: List[Path] = []
        for output_path in output_paths:
            if not output_path.exists():
                missing_files.append(output_path)
        if len(missing_files) > 0:
            error_msg = (
                f"{description} completed but expected output file(s) not found:\n"
                f"{missing_files}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

def dicom_to_nifti(
        dicom_dir: Union[str, Path],
        output_dir: Union[str, Path],
        output_name: str,
        overwrite: bool = False,
        verbose: bool = True,
        compress: bool = True
) -> Path:
    """
    Convert DICOM file(s) to NIfTI format using dcm2niix.

    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing DICOM files or path to a single DICOM file to convert.
    output_dir : str or Path
        Directory where the NIfTI file will be saved. Will be created if it doesn't exist
    output_name : str
        Base name for output file (without extension)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    verbose : bool, default=True
        If False, suppresses dcm2niix output messages.
    compress : bool, default=True
        If True, output gzip-compressed NIfTI (.nii.gz).
        If False, output uncompressed NIfTI (.nii) for faster writes.

    Returns
    -------
    nifti_path : Path
        Path to the created NIfTI file

    Notes
    -----
    Set compress=False for real-time processing where speed is critical.
    Uncompressed output saves ~50-100ms per volume but produces larger files.

    """

    dicom_path = Path(dicom_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add quiet flag if not verbose
    quiet_flag = [] if verbose else ['-q']

    # Compression flag: -z y for gzip, -z n for uncompressed
    compress_flag = 'y' if compress else 'n'
    ext = '.nii.gz' if compress else '.nii'
    nifti_path = output_dir / f"{output_name}{ext}"

    # Handle single file vs directory
    # If a single file is given, copy it to a temp directory to avoid converting
    # all DICOMs in the same directory
    temp_dir = None
    if dicom_path.is_file():
        temp_dir = tempfile.mkdtemp(prefix='dcm2niix_')
        temp_dicom = Path(temp_dir) / dicom_path.name
        shutil.copy2(dicom_path, temp_dicom)
        conversion_path = Path(temp_dir)
    else:
        conversion_path = dicom_path

    try:
        # Build command
        cmd = ['dcm2niix'] + quiet_flag + ['-z', compress_flag, '-o', str(output_dir), '-f', str(output_name), str(conversion_path)]

        # Run dcm2niix
        run_command(
            cmd_name='dcm2niix',
            cmd=cmd,
            description='Converting DICOM to NIfTI',
            inputs=dicom_path,
            outputs=nifti_path,
            overwrite=overwrite,
            verbose=verbose
        )
    finally:
        # Clean up temp directory if created
        if temp_dir is not None and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    return nifti_path

def get_yeo_labels(
        n_networks: int
) -> List[str]:
    """
    Function to get the Yeo network labels.

    Parameters
    ---------
    n_networks : int
        7 or 17 network parcellation
    
    Returns
    -------
    labels : List of str
        List of network labels

    Raises
    ------
    ValueError
        If n_networks parameter is invalid

    """

    # Validate n_networks
    if n_networks != 7 and n_networks != 17:
        raise ValueError(
            f"Invalid number of networks: {n_networks}. "
            f"Valid options are 7 or 17"
        )
    
    # Assign labels
    if n_networks == 7:
        labels = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    else:
        labels = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB",
        "SalVentAttnA", "SalVentAttnB", "LimbicA", "LimbicB", "ContC", "ContA", "ContB",
        "TempPar", "DefaultC", "DefaultA", "DefaultB"]

    return labels

# ==============================================================
#  FSL FUNCTIONS
# ==============================================================

def compute_mean(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Compute temporal mean of fMRI data using numpy.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should be 4D)
    output_file: str or Path
        Path to save output mean image
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping compute mean.")
        return
    
    logger.info("Computing mean...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load 4D image and compute mean along time axis
    img = nib.load(str(input_file))
    data = img.get_fdata()
    mean_data = data.mean(axis=-1)
    
    # Save output
    out_img = Nifti1Image(mean_data, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}") 

def compute_xfm(
        input_file: Union[str, Path],
        reference_file: Union[str, Path],
        output_name: Union[str, Path],
        cost: str = 'corratio',
        dof: int = 6,
        overwrite: bool = False
) -> tuple[Path, Path]:
    """
    Compute a transformation matrix using FSL's FLIRT

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    reference_file : str or Path
        Path to reference image (target space for alignment)
    output_name : str or Path
        Base name for output file (without extension)
    cost : str, default='corratio'
        Cost function for registration (per FSL, e.g., corratio, leastsq)
    dof: int, default=6
        Degrees of freedom for transformation (per FSL, e.g., 6 for rigid body)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Returns
    -------
    output_file : Path
        Path for output NIfTI file
    output_xfm : Path
        Path for output transformation

    Raises
    ------
    ValueError
        If cost or dof parameters are invalid

    """

    # Store output NIfTI file and transformation
    output_file = Path(output_name).with_suffix(".nii.gz")
    output_xfm = Path(output_name).with_suffix(".mat")

    # Validate cost function
    valid_costs = ['mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq', 'labeldiff', 'bbr']
    if cost not in valid_costs:
        raise ValueError(
            f"Invalid cost function: {cost}. "
            f"Valid options are: {', '.join(valid_costs)}"
        )
    
    # Validate DOF
    valid_dofs = [6, 7, 9, 12]
    if dof not in valid_dofs:
        raise ValueError(
            f"Invalid dof: {dof}. "
            f"Valid options are: {', '.join(map(str, valid_dofs))}"
        )

    # Build flirt command
    cmd = [
        'flirt',
        '-in', str(input_file),
        '-ref', str(reference_file),
        '-out', str(output_file),
        '-omat', str(output_xfm),
        '-cost', cost,
        '-dof', str(dof),
        '-noresample',
        '-noresampblur'
    ]

    # Run flirt
    run_command(
        cmd_name='flirt',
        cmd=cmd,
        description='Computing transformation',
        inputs=[input_file, reference_file],
        outputs=[output_file, output_xfm],
        overwrite=overwrite
    ) 

    return output_file, output_xfm

def apply_xfm(
        input_file: Union[str, Path],
        reference_file: Union[str, Path],
        xfm_file: Union[str, Path],
        output_file: Union[str, Path],
        interp: str = 'nearestneighbour',
        overwrite: bool = False
) -> None:
    """
    Apply a transformation matrix using FSL's FLIRT

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    reference_file : str or Path
        Path to reference image (target space for alignment)
    xfm_file : str or Path
        Path to transformation file
    output_file : str or Path
        Path for output file 
    interp : str, default='nearestneighbour'
        Interpolation for registration (per FSL, e.g., trilinear, nearestneighbour)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Raises
    ------
    ValueError
        If interp parameter is invalid

    """

    # Validate interp
    valid_interps = ['nearestneighbour', 'trilinear', 'sinc', 'spline', 'nn']
    if interp not in valid_interps:
        raise ValueError(
            f"Invalid interpolation: {interp}. "
            f"Valid options are: {', '.join(valid_interps)}"
        )
    
    # Build flirt command
    cmd = [
        'flirt',
        '-in', str(input_file),
        '-ref', str(reference_file),
        '-out', str(output_file),
        '-init', str(xfm_file),
        '-applyxfm',
        '-interp', interp,
        '-noresample',
        '-noresampblur'
    ]

    run_command(
        cmd_name='flirt',
        cmd=cmd,
        description='Applying transformation',
        inputs=[input_file, reference_file, xfm_file],
        outputs=output_file,
        overwrite=overwrite
    )

def apply_xfm_sform_to_mni(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        interp: str = 'nearestneighbour',
        overwrite: bool = False
) -> None:
    """
    Apply transformation to MNI space using sform/qform matrices from NIfTI headers

    This function applies the transformation encoded in the sform/qform matrices
    to transform the input file to MNI152 space using the templateflow BOLD reference.
    No explicit transformation matrix file is required.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should have sform/qform set to MNI space)
    output_file : str or Path
        Path for output file in MNI space
    interp : str, default='nearestneighbour'
        Interpolation method (per FSL, e.g., trilinear, nearestneighbour)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Raises
    ------
    ValueError
        If interp parameter is invalid

    Notes
    -----
    This function uses FLIRT's -applyxfm -usesqform flags to apply the
    transformation encoded in the NIfTI headers rather than using an
    external transformation matrix file. The reference template is the
    MNI152NLin2009cAsym BOLD reference from templateflow.

    """

    # Validate interp
    valid_interps = ['nearestneighbour', 'trilinear', 'sinc', 'spline', 'nn']
    if interp not in valid_interps:
        raise ValueError(
            f"Invalid interpolation: {interp}. "
            f"Valid options are: {', '.join(valid_interps)}"
        )

    # Load in the MNI152 BOLD reference
    epi_tpl_path = cast(Path, tflow.get(
        "MNI152NLin2009cAsym",
        resolution=2,          # 2 mm
        desc="fMRIPrep",       # fMRIPrep-style boldref
        suffix="boldref"
    )) # type: ignore

    # Build flirt command
    cmd = [
        'flirt',
        '-in', str(input_file),
        '-ref', str(epi_tpl_path),
        '-out', str(output_file),
        '-applyxfm',
        '-usesqform',
        '-interp', interp
    ]

    run_command(
        cmd_name='flirt',
        cmd=cmd,
        description='Applying sform/qform transformation to MNI',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    )

def invert_xfm(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Invert a transformation matrix using numpy.

    Parameters
    ----------
    input_file : str or Path
        Path to input transformation file (FSL .mat format)
    output_file : str or Path
        Path to output transformation file
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping invert_xfm.")
        return
    
    logger.info("Inverting transformation matrix...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load FSL .mat file (4x4 matrix, space-separated)
    matrix = np.loadtxt(str(input_file))
    
    # Invert the matrix
    inverted = np.linalg.inv(matrix)
    
    # Save in FSL format
    np.savetxt(str(output_path), inverted, fmt='%.10f')
    logger.info(f"Output: {output_path}")

def compute_xfm_to_mni(
        input_file: Union[str, Path],
        output_name: Union[str, Path],
        cost: str = 'corratio',
        dof: int = 6,
        overwrite: bool = False
) -> tuple[Path, Path]:
    """
    Wrapper for compute_xfm to specifically compute transformation to MNI space using the BOLD reference

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_name : str or Path
        Base name for output file (without extension)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Returns
    -------
    output_file : Path
        Path for output NIfTI file
    output_xfm : Path
        Path for output transformation

    """

    # Check if outputs exist and skip if overwrite=False
    if not overwrite:
        sub2mni_file = Path(output_name).with_suffix(".nii.gz")
        sub2mni_xfm = Path(output_name).with_suffix(".mat")
        output_paths_check = [Path(sub2mni_file), Path(sub2mni_xfm)]
        if all(p.exists() for p in output_paths_check):
            logger.info(f"All output files exist and overwrite=False. Skipping MNI registration.")
            return sub2mni_file, sub2mni_xfm

    
    # Load in the MNI152 BOLD reference
    epi_tpl_path = cast(Path, tflow.get( 
        "MNI152NLin2009cAsym",
        resolution=2,          # 2 mm
        desc="fMRIPrep",       # fMRIPrep-style boldref
        suffix="boldref"
    )) # type: ignore

    # Compute xfm
    sub2mni_file, sub2mni_xfm = compute_xfm(
        input_file=input_file,
        reference_file=epi_tpl_path,
        output_name=output_name,
        cost='corratio',
        dof=6,
        overwrite=overwrite
    )

    return sub2mni_file, sub2mni_xfm

def apply_mask(
        input_file: Union[str, Path],
        mask_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Apply a binary mask to a NIfTI image using numpy.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    mask_file : str or Path
        Path to mask NIfTI file
    output_file : str or Path
        Path to output NIfTI file
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping apply_mask.")
        return
    
    logger.info("Applying mask...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load images
    img = nib.load(str(input_file))
    mask = nib.load(str(mask_file))
    
    # Get data
    data = img.get_fdata()
    mask_data = mask.get_fdata()
    
    # Apply mask (multiply)
    masked = data * (mask_data > 0).astype(data.dtype)
    
    # Save output
    out_img = Nifti1Image(masked, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")


# ==============================================================
#  ANTs FUNCTIONS
# ==============================================================

def compute_mean_ants(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Compute temporal mean of fMRI data using ANTsPy.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should be 4D)
    output_file : str or Path
        Path to save output mean image
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    """
    import ants
    
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping compute mean.")
        return
    
    logger.info("Computing mean (ANTs)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load 4D image and compute mean
    img_4d = ants.image_read(str(input_file))
    mean_data = img_4d.numpy().mean(axis=-1)
    
    # Create mean image with same spatial properties
    mean_img = ants.from_numpy(
        mean_data,
        origin=img_4d.origin[:3],
        spacing=img_4d.spacing[:3],
        direction=img_4d.direction[:3, :3]
    )
    
    ants.image_write(mean_img, str(output_file))
    logger.info(f"Output: {output_file}")


def compute_xfm_ants(
        input_file: Union[str, Path],
        reference_file: Union[str, Path],
        output_name: Union[str, Path],
        type_of_transform: str = 'Rigid',
        overwrite: bool = False
) -> tuple[Path, list]:
    """
    Compute a transformation using ANTsPy registration.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (moving image)
    reference_file : str or Path
        Path to reference image (fixed/target space for alignment)
    output_name : str or Path
        Base name for output file (without extension)
    type_of_transform : str, default='Rigid'
        Type of transform: 'Rigid', 'Affine', 'SyN', etc.
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Returns
    -------
    output_file : Path
        Path for output registered NIfTI file
    transform_files : list
        List of paths to transform files
    """
    import ants
    
    output_file = Path(output_name).with_suffix(".nii.gz")
    xfm_dir = output_file.parent
    xfm_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if outputs exist
    if not overwrite and output_file.exists():
        # Try to find existing transform files
        existing_transforms = list(xfm_dir.glob(f"{Path(output_name).stem}_transform_*.mat"))
        if existing_transforms:
            logger.info(f"Output files exist and overwrite=False. Skipping registration.")
            return output_file, [str(t) for t in sorted(existing_transforms)]
    
    logger.info(f"Computing transformation (ANTs {type_of_transform})...")
    logger.info(f"Input: {input_file}")
    logger.info(f"Reference: {reference_file}")
    
    # Load images
    fixed = ants.image_read(str(reference_file))
    moving = ants.image_read(str(input_file))
    
    # Run registration
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=type_of_transform,
        verbose=False
    )
    
    # Save registered image
    ants.image_write(registration['warpedmovout'], str(output_file))
    logger.info(f"Output: {output_file}")
    
    # Copy and rename transform files to output directory
    transform_files = []
    for i, fwd in enumerate(registration['fwdtransforms']):
        suffix = Path(fwd).suffix
        new_path = xfm_dir / f"{Path(output_name).stem}_transform_{i}{suffix}"
        shutil.copy(fwd, new_path)
        transform_files.append(str(new_path))
        logger.info(f"Transform: {new_path}")
    
    return output_file, transform_files


def apply_xfm_ants(
        input_file: Union[str, Path],
        reference_file: Union[str, Path],
        transform_files: list,
        output_file: Union[str, Path],
        interpolator: str = 'nearestNeighbor',
        overwrite: bool = False
) -> None:
    """
    Apply a transformation using ANTsPy.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    reference_file : str or Path
        Path to reference image (target space)
    transform_files : list
        List of paths to transform files from compute_xfm_ants
    output_file : str or Path
        Path for output file
    interpolator : str, default='nearestNeighbor'
        Interpolation method: 'nearestNeighbor', 'linear', 'bSpline', etc.
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    """
    import ants
    
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping apply transform.")
        return
    
    logger.info(f"Applying transformation (ANTs)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load images
    fixed = ants.image_read(str(reference_file))
    moving = ants.image_read(str(input_file))
    
    # Apply transforms
    transformed = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transform_files,
        interpolator=interpolator
    )
    
    ants.image_write(transformed, str(output_file))
    logger.info(f"Output: {output_file}")
