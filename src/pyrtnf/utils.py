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
        verbose: bool = True
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

    Returns
    -------
    nifti_path : Path
        Path to the created NIfTI file

    Notes
    -----
    If dicom_dir is a file path, the file is copied to a temporary directory
    before conversion to ensure only that file is converted.
    If dicom_dir is a directory path, all DICOM files in that directory will be converted.

    """
    
    dicom_path = Path(dicom_dir)
    
    # Add quiet flag if not verbose
    quiet_flag = [] if verbose else ['-q']
    
    # If a single file is provided, create a temp directory with just that file
    if dicom_path.is_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Copy the DICOM file to temp directory
            tmp_dicom = tmpdir_path / dicom_path.name
            shutil.copy2(dicom_path, tmp_dicom)
            
            # Build command with temp directory
            cmd = ['dcm2niix'] + quiet_flag + ['-z', 'y', '-o', str(output_dir), '-f', str(output_name), str(tmpdir_path)]
            nifti_path = Path(output_dir) / f"{output_name}.nii.gz"
            
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
    else:
        # Directory path - process normally
        cmd = ['dcm2niix'] + quiet_flag + ['-z', 'y', '-o', str(output_dir), '-f', str(output_name), str(dicom_path)]
        nifti_path = Path(output_dir) / f"{output_name}.nii.gz"
        
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
    Compute temporal mean of fMRI data.

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

    # Build fslmaths command
    cmd = [
        'fslmaths',
        str(input_file),
        '-Tmean',
        str(output_file)
    ]

    # Run fslmaths
    run_command(
        cmd_name='fslmaths',
        cmd=cmd,
        description='Compute mean',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    ) 

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

def invert_xfm(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Invert a transformation matrix using FSL's convert_xfm

    Parameters
    ----------
    input_file : str or Path
        Path to input transformation file
    output_file : str or Path
        Path to output transformation file
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """

    # Build convert_xfm command
    cmd = [
        'convert_xfm',
        '-omat', str(output_file),
        '-inverse', str(input_file),
    ]

    # Run convert_xfm
    run_command(
        cmd_name='convert_xfm',
        cmd=cmd,
        description='Applying transformation',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    )

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
    Invert a transformation matrix using FSL's convert_xfm

    Parameters
    ----------
    input_file : str or Path
        Path to input transformation file
    mask_file : str or Path
    output_file : str or Path
        Path to output transformation file
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """

    # Build fslmaths command
    cmd = [
        'fslmaths',
        str(input_file),
        "-mas", str(mask_file),
        str(output_file)
    ]

    # Run fslmaths
    run_command(
        cmd_name='fslmaths',
        cmd=cmd,
        description='Applying mask',
        inputs=[input_file, mask_file],
        outputs=output_file,
        overwrite=overwrite
    )
