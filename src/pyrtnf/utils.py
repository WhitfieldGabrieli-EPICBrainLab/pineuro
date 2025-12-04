"""
Utilities for pyrtnf

This module provides utilities for the pyrtnf package

Notes
-----

Examples
--------

References
----------
"""

# Standard library imports
import shutil
from typing import Union, List, Optional, cast
from pathlib import Path
import subprocess

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
        overwrite: bool = False
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
    overwrite : bool, default=True
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Raises
    ------
    FileNotFoundError
        If input files are not found
    RuntimeError
        If command fails or if expected output files are not created

    """

    # Log the command
    logger.info(f"{description}...")
    
    # Convert outputs to list if single file provided
    if isinstance(outputs, (str, Path)):
        outputs = [outputs]
    
    # Check if outputs exist and skip if overwrite=False
    if not overwrite and outputs is not None:
        output_paths_check = [Path(f) for f in outputs]
        if all(p.exists() for p in output_paths_check):
            logger.info(f"All output files exist and overwrite=False. Skipping {description}.")
            return
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")

    # Check if command is installed
    _check_command_installation(cmd_name)
    
    # Convert inputs to list if single file provided
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    # Convert to Path objects and validate
    input_paths: list[Path] = []
    if inputs is not None:
        logger.info(f"Input files:")
        for f in inputs:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            input_paths.append(path)
            logger.info(f"  {f}")
    output_paths: list[Path] = []
    if outputs is not None:
        logger.info(f"Output files:")
        for f in outputs:
            path = Path(f)
            (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
            output_paths.append(path)
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
        if result.stdout:
            logger.debug(f"stdout:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"stderr:\n{result.stderr}")
        
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
        labels = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB"
        "SalVentAttnA", "SalVentAttnB", "LimbicA", "LimbicB", "ContC", "ContA", "ContB",
        "TempPar", "DefaultC", "DefaultA", "DefaultB"]

    return labels

# ==============================================================
#  FSL FUNCTIONS
# ==============================================================

def compute_mean(
        input_file: Union[str, Path],
        output_file: Union[str, Path]
) -> None:
    """
    Compute temporal mean of fMRI data.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should be 4D)
    output_file: str or Path
        Path to save output mean image

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
        outputs=output_file
    ) 

def compute_xfm(
        input_file: Union[str, Path],
        reference_file: Union[str, Path],
        output_name: Union[str, Path],
        cost: str = 'corratio',
        dof: int = 6
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
        outputs=[output_file, output_xfm]
    ) 

    return output_file, output_xfm

def apply_xfm(
        input_file: Union[str, Path],
        reference_file: Union[str, Path],
        xfm_file: Union[str, Path],
        output_file: Union[str, Path],
        interp: str = 'nearestneighbour',
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
        '-interp', interp,
        '-noresample',
        '-noresampblur'
    ]

    run_command(
        cmd_name='flirt',
        cmd=cmd,
        description='Applying transformation',
        inputs=[input_file, reference_file, xfm_file],
        outputs=output_file
    )

def invert_xfm(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
) -> None:
    """
    Invert a transformation matrix using FSL's convert_xfm

    Parameters
    ----------
    input_file : str or Path
        Path to input transformation file
    output_file : str or Path
        Path to output transformation file

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
        outputs=output_file
    )

def compute_xfm_to_mni(
        input_file: Union[str, Path],
        output_name: Union[str, Path],
        cost: str = 'corratio',
        dof: int = 6
) -> tuple[Path, Path]:
    """
    Wrapper for compute_xfm to specifically compute transformation to MNI space using the BOLD reference

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_name : str or Path
        Base name for output file (without extension)

    Returns
    -------
    output_file : Path
        Path for output NIfTI file
    output_xfm : Path
        Path for output transformation

    """

    # Load in the MNI152 BOLD reference
    epi_tpl_path = cast(Path, tflow.get( 
        "MNI152NLin2009cAsym",
        resolution=2,          # 2 mm
        desc="fMRIPrep",       # fMRIPrep-style boldref
        suffix="boldref"
    ))

    # Compute xfm
    sub2mni_file, sub2mni_xfm = compute_xfm(
        input_file=input_file,
        reference_file=epi_tpl_path,
        output_name=output_name,
        cost='corratio',
        dof=6
    )

    return sub2mni_file, sub2mni_xfm

    
    



