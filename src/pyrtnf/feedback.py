"""

"""

# Standard library imports
from typing import Union, List
from pathlib import Path

# Third-party imports

# Local imports
from pyrtnf.utils import dicom_to_nifti, compute_mean, compute_xfm, apply_xfm

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

def coregister_2vol(
        participant_id: str,
        root_dir: Union[str, Path],
        input_dicom: Union[str, Path],
        network_files: Union[str, Path, List[Union[str, Path]]],
        ref_file: Union[str, Path],
        overwrite: bool = False
) -> List[Path]:
    """
    Coregister extracted networks to participant's current position in the scanner
    using a two volume (2vol) acquisition.

    Parameters
    ----------
    participant_id : str
        Participant ID
    root_dir : str or Path
        Path to the root directory for all outputs
    input_dicom : str or Path
        Path to a single DICOM folder
    network_files : str or Path or list of str of Path
        Path to a single network file, or list of multiple network files
    ref_file : str or Path
        Path to NIfTI file to use as a BOLD reference for network space
    overwrite : bool, default=True
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Returns
    -------
    network_coreg_files
        List of co-registered networks

    Raises
    ------
    FileNotFoundError
        If input_dicoms, network_files, or ref_file does not exist or is empty

    """

    # Log the command
    logger.info(f"Running preprocessing for {participant_id}...")
    
    # Convert to Path object
    root_dir = Path(root_dir)
    
    # Convert to Path objects and validate
    input_path = Path(input_dicom)
    logger.info(f"Input DICOM folder: {str(input_path)}")
    if not input_path.exists():
        raise FileNotFoundError(f"Folder not found: {input_path}")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {input_path}")
    if not any(input_path.iterdir()):
        raise FileNotFoundError(f"Directory is empty: {input_path}")

    # Convert outputs to list if single file provided
    if isinstance(network_files, (str, Path)):
        network_files = [network_files]

    # Convert to Path objects and validate
    network_paths: list[Path] = [] 
    logger.info(f"Input network files:\n")
    for f in network_files:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        network_paths.append(path)
        logger.info(f"  {str(path)}")

    # Convert to Path object and validate
    ref_file = Path(ref_file)
    if not ref_file.exists():
        raise FileNotFoundError(f"File not found: {ref_file}")

    # Step 1: Check for existing 2vol and get largest run number
    # TODO: Smarter way to handle whether 2vol already eixsts
    twovol_dir = root_dir / "01_2vol"
    last_run = 0
    if twovol_dir.exists() and overwrite:
        pattern = f"{participant_id}_task-2vol_run-*.nii.gz"
        matching_files = list(twovol_dir.glob(pattern))
        for file in matching_files:
            for part in file.stem.split('_'):
                if part.startswith('run-'):
                    try:
                        run_numbers.append(int(part.split('-')[1]))
                    except ValueError:
                        pass
        if run_numbers:
            last_run = max(run_numbers)
    else:
        twovol_dir.mkdir(parents=True, exist_ok=True)
    next_run = last_run + 1

    # Step 2: Convert DICOMs to NIfTI
    nifti_file = dicom_to_nifti(
        dicom_dir=input_path,
        output_dir=twovol_dir,
        output_name=f"{participant_id}_task-2vol_run-{str(next_run).zfill(2)}",
        overwrite=overwrite
    )

    # Step 3: Compute mean
    mean_file = twovol_dir / Path(f"{participant_id}_task-2vol_run-{str(next_run).zfill(2)}_mean.nii.gz")
    compute_mean(input_file=nifti_file, output_file=mean_file, overwrite=overwrite)
    
    # Step 4: Compute transformation
    xfm_dir = twovol_dir / "xfm"
    output_name = xfm_dir / Path(f"{participant_id}_boldref_to_task-2vol_run-{str(next_run).zfill(2)}_mean")
    _, xfm_file = compute_xfm(input_file=ref_file, reference_file=mean_file, output_name=output_name, cost='corratio', dof=6, overwrite=overwrite)

    # Step 5: Apply transformation
    mask_dir = twovol_dir / "masks"
    network_coreg_files = []
    for network in network_paths:
        output_file = mask_dir / (network.with_suffix("").stem + f"_to_task-2vol_run-{str(next_run).zfill(2)}_mean.nii.gz")
        apply_xfm(input_file=network, reference_file=mean_file, xfm_file=xfm_file, output_file=output_file, interp='nearestneighbour', overwrite=overwrite)
        network_coreg_files.append(output_file)

    return network_coreg_files



