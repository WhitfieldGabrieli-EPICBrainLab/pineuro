"""Script generation utilities for reproducible mask extraction.

This module provides functions to generate standalone Python scripts that
reproduce the mask extraction pipeline with all parameters captured. Users
can modify these scripts and re-run them outside the GUI for custom workflows.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from pineuro.mask_extraction import MaskSelection


def generate_mask_extraction_script(
    participant_id: str,
    mask_extraction_dir: Union[str, Path],
    input_file: Union[str, Path],
    mask_file: Union[str, Path],
    ref_file: Union[str, Path],
    n_yeo: int = 17,
    yeo_targets: Optional[List[str]] = None,
    n_components: int = 20,
    threshold: int = 2000,
    smooth_fwhm: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    selections: Optional[Dict[str, MaskSelection]] = None,
) -> str:
    """Generate a Python script that reproduces the mask extraction pipeline.

    Parameters
    ----------
    participant_id : str
        Participant ID (e.g., "sub-01")
    mask_extraction_dir : str or Path
        Root directory for mask extraction outputs
    input_file : str or Path
        Path to preprocessed fMRI data
    mask_file : str or Path
        Path to brain mask
    ref_file : str or Path
        Path to BOLD reference image
    n_yeo : int
        Yeo parcellation (7 or 17 networks)
    yeo_targets : list of str, optional
        Target network names to extract
    n_components : int
        Number of ICA components
    threshold : int
        Voxel count for mask thresholding
    smooth_fwhm : float, optional
        Gaussian smoothing FWHM in mm
    highpass : float, optional
        High-pass filter frequency in Hz
    lowpass : float, optional
        Low-pass filter frequency in Hz
    selections : dict, optional
        Dictionary mapping network names to MaskSelection objects
        for any manual modifications

    Returns
    -------
    str
        Python script content as a string
    """
    mask_extraction_dir = Path(mask_extraction_dir)
    input_file = Path(input_file)
    mask_file = Path(mask_file)
    ref_file = Path(ref_file)

    if yeo_targets is None:
        yeo_targets = ["DefaultA", "ContA"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build mask selection code if there are manual modifications
    mask_selection_code = _generate_selection_code(
        selections, mask_extraction_dir, n_yeo, threshold
    )

    script = f'''#!/usr/bin/env python3
"""
Mask Extraction Script for {participant_id}
Generated: {timestamp}

This script reproduces the mask extraction pipeline with all parameters.
You can modify parameters below and re-run to regenerate masks.

Usage:
    python reproduce_masks.py

To regenerate only the QC report (after manual edits):
    python reproduce_masks.py --qc-only
"""

import shutil
import sys
from pathlib import Path

from pineuro import mask_extraction_rest
from pineuro.mask_extraction import (
    MaskSelection,
    save_mask_selections,
    regenerate_mask_from_selection,
    load_mask_selections,
)
from pineuro import qc
from pineuro.qc import generate_alternative_overlays, generate_yeo_network_overlay


def clean_stale_outputs(root_dir: Path):
    """Remove directories and files that would become stale after regenerating masks.

    - 04_2vol/: Coregistered masks from localizer session (must be re-run)
    - qc/: QC report and overlays (regenerated at the end)
    - .masks_approved: Approval marker (masks need re-approval after regeneration)
    """
    stale_dirs = ["04_2vol", "qc"]
    for dirname in stale_dirs:
        dir_path = root_dir / dirname
        if dir_path.exists():
            print(f"  Removing stale directory: {{dirname}}/")
            shutil.rmtree(dir_path)

    # Remove approval marker - masks need re-approval after regeneration
    approval_marker = root_dir / ".masks_approved"
    if approval_marker.exists():
        print("  Removing approval marker: .masks_approved")
        approval_marker.unlink()


# =============================================================================
# INPUT FILES
# =============================================================================
participant_id = "{participant_id}"
root_dir = Path("{mask_extraction_dir}")
input_file = Path("{input_file}")
mask_file = Path("{mask_file}")
ref_file = Path("{ref_file}")


# =============================================================================
# ICA PARAMETERS
# Modify these to change the ICA decomposition
# =============================================================================
n_components = {n_components}  # Number of ICA components to extract
smooth_fwhm = {repr(smooth_fwhm)}  # Spatial smoothing FWHM in mm (None = no smoothing)
highpass = {repr(highpass)}  # High-pass filter frequency in Hz (None = no filtering)
lowpass = {repr(lowpass)}  # Low-pass filter frequency in Hz (None = no filtering)


# =============================================================================
# MASK PARAMETERS
# Modify these to change network selection and thresholding
# =============================================================================
n_yeo = {n_yeo}  # Yeo parcellation: 7 or 17 networks
yeo_targets = {repr(yeo_targets)}  # Networks to extract
threshold = {threshold}  # Number of voxels to keep per mask


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    qc_only = "--qc-only" in sys.argv

    if not qc_only:
        # -----------------------------------------------------------------
        # Clean up stale outputs from previous runs
        # -----------------------------------------------------------------
        print("=" * 60)
        print("Cleaning up stale outputs")
        print("=" * 60)
        clean_stale_outputs(root_dir)
        print()

        # -----------------------------------------------------------------
        # STEP 1: Run mask extraction pipeline
        # -----------------------------------------------------------------
        print("=" * 60)
        print("STEP 1: Running mask extraction pipeline")
        print("=" * 60)
        print(f"  Participant: {{participant_id}}")
        print(f"  Input file: {{input_file}}")
        print(f"  ICA components: {{n_components}}")
        print(f"  Target networks: {{yeo_targets}}")
        print(f"  Threshold: {{threshold}} voxels")
        print()

        mask_files = mask_extraction_rest(
            participant_id=participant_id,
            root_dir=root_dir,
            input_file=input_file,
            mask_file=mask_file,
            ref_file=ref_file,
            n_yeo=n_yeo,
            yeo_targets=yeo_targets,
            n_components=n_components,
            threshold=threshold,
            smooth_fwhm=smooth_fwhm,
            highpass=highpass,
            lowpass=lowpass,
            overwrite=True,  # Set to False to skip existing outputs
            qc_enabled=False,  # QC generated separately below
        )

        print(f"  Created {{len(mask_files)}} mask files")
        print()

{mask_selection_code}

    # -----------------------------------------------------------------
    # STEP 3: Regenerate all overlays for GUI preview
    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Regenerating overlay images")
    print("=" * 60)

    qc_dir = root_dir / "qc"

    # Generate alternative ICA component overlays
    print(f"  Generating ICA component overlays for {{len(yeo_targets)}} networks...")
    generate_alternative_overlays(
        mask_extraction_dir=root_dir,
        participant_id=participant_id,
        network_names=yeo_targets,
        n_alternatives=n_components,  # All ICA components
        output_dir=qc_dir,
        n_yeo=n_yeo,
        overwrite=True,
    )

    # Generate Yeo atlas fallback overlays for each network
    print(f"  Generating Yeo atlas overlays...")
    for network_name in yeo_targets:
        generate_yeo_network_overlay(
            mask_extraction_dir=root_dir,
            participant_id=participant_id,
            network_name=network_name,
            output_dir=qc_dir,
            n_yeo=n_yeo,
            overwrite=True,
        )

    print()

    # -----------------------------------------------------------------
    # STEP 4: Regenerate QC report
    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Generating QC report")
    print("=" * 60)

    qc.generate_mask_extraction_report(
        mask_extraction_dir=root_dir,
        participant_id=participant_id,
        network_names=yeo_targets,
        n_yeo=n_yeo,
        overwrite=True,
        open_browser=False,  # Set to True to auto-open in browser
    )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Masks saved to: {{root_dir / '03_masks'}}")
    print(f"  QC report: {{root_dir / 'qc' / 'mask_extraction_qc.html'}}")

    if not qc_only:
        print()
        print("NOTE: Masks require re-approval in the GUI before proceeding.")
        print("The 04_2vol/ directory was also removed because it contained")
        print("coregistered masks from the previous pipeline run.")


if __name__ == "__main__":
    main()
'''

    return script


def _generate_selection_code(
    selections: Optional[Dict[str, MaskSelection]],
    mask_extraction_dir: Path,
    n_yeo: int,
    threshold: int,
) -> str:
    """Generate Python code for applying manual mask selections.

    Parameters
    ----------
    selections : dict or None
        Dictionary mapping network names to MaskSelection objects
    mask_extraction_dir : Path
        Root directory for mask extraction
    n_yeo : int
        Yeo parcellation (7 or 17)
    threshold : int
        Voxel threshold

    Returns
    -------
    str
        Python code block as a string
    """
    if not selections:
        return '''        # -----------------------------------------------------------------
        # STEP 2: No manual mask selections to apply
        # -----------------------------------------------------------------
        # (All masks use automatic ICA component selection)
        print("  No manual mask selections to apply")
        print()
'''

    # Check if any selections are non-auto
    manual_selections = {
        name: sel for name, sel in selections.items()
        if sel.selection_reason != "auto"
    }

    if not manual_selections:
        return '''        # -----------------------------------------------------------------
        # STEP 2: No manual mask selections to apply
        # -----------------------------------------------------------------
        # (All masks use automatic ICA component selection)
        print("  No manual mask selections to apply")
        print()
'''

    # Generate code for manual selections - COMMENTED OUT by default
    # so running the script fresh gives auto-selected ICA components
    selection_defs = []
    for name, sel in manual_selections.items():
        if sel.source_type == "ica_combined":
            # Combined ICA selection - include secondary component
            selection_defs.append(f'''        #     "{name}": MaskSelection(
        #         network_name="{sel.network_name}",
        #         source_type="{sel.source_type}",
        #         component_index={sel.component_index},
        #         correlation={sel.correlation},
        #         selection_reason="{sel.selection_reason}",
        #         secondary_component_index={sel.secondary_component_index},
        #         secondary_correlation={sel.secondary_correlation},
        #     ),''')
        else:
            # Single ICA or Yeo atlas selection
            selection_defs.append(f'''        #     "{name}": MaskSelection(
        #         network_name="{sel.network_name}",
        #         source_type="{sel.source_type}",
        #         component_index={sel.component_index},
        #         correlation={sel.correlation},
        #         selection_reason="{sel.selection_reason}",
        #     ),''')

    selections_dict = "\n".join(selection_defs)

    return f'''        # -----------------------------------------------------------------
        # STEP 2: Apply manual mask selections (COMMENTED OUT)
        # -----------------------------------------------------------------
        # Uncomment the code below to re-apply your manual mask selections.
        # By default, running this script uses the auto-selected ICA components.
        #
        # masks_dir = root_dir / "03_masks"
        # ica_dir = root_dir / "01_ica"
        # mnireg_dir = root_dir / "02_mnireg"
        #
        # # Find ICA components file
        # components_files = sorted(ica_dir.glob("*_components.nii.gz"))
        # if not components_files:
        #     raise FileNotFoundError("ICA components file not found")
        # components_file = components_files[0]
        #
        # # Find Yeo atlas in subject space
        # yeo_files = sorted(mnireg_dir.glob("Yeo_to_*.nii.gz"))
        # if not yeo_files:
        #     raise FileNotFoundError("Yeo atlas in subject space not found")
        # yeo_subject_space = yeo_files[0]
        #
        # # Manual selections to apply
        # manual_selections = {{
{selections_dict}
        # }}
        #
        # for name, selection in manual_selections.items():
        #     print(f"  {{name}}: {{selection.source_type}}", end="")
        #     if selection.source_type == "ica":
        #         print(f" (IC {{selection.component_index}}, r={{selection.correlation:.2f}})")
        #     elif selection.source_type == "ica_combined":
        #         print(f" (IC {{selection.component_index}} + IC {{selection.secondary_component_index}})")
        #     else:
        #         print(" (Yeo atlas fallback)")
        #
        #     regenerate_mask_from_selection(
        #         participant_id=participant_id,
        #         masks_dir=masks_dir,
        #         selection=selection,
        #         components_file=components_file,
        #         yeo_subject_space=yeo_subject_space,
        #         n_yeo={n_yeo},
        #         threshold={threshold},
        #     )
        #
        # # Save selections to JSON
        # existing_selections = load_mask_selections(masks_dir)
        # existing_selections.update(manual_selections)
        # save_mask_selections(masks_dir, existing_selections)
        #
        # print()
'''


def save_script(
    script_content: str,
    output_path: Union[str, Path],
) -> Path:
    """Save a generated script to a file.

    Parameters
    ----------
    script_content : str
        The script content to save
    output_path : str or Path
        Path to save the script

    Returns
    -------
    Path
        Path to the saved script file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_content)
    # Make executable on Unix
    output_path.chmod(0o755)
    return output_path
