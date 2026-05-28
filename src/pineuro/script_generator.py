"""Script generation for reproducible mask extraction.

Generates a standalone ``reproduce_masks.py`` that re-runs the mask extraction
pipeline with the exact parameters captured from a GUI run. Researchers can
edit the parameters and re-run outside the GUI for custom workflows.

Design
------
The generated script does **not** re-derive how to call mask extraction. It
serializes the *mode* (``"rest" | "task" | "mni"``) and the resolved *params*
dict the pipeline actually passed, then replays the call through the single
shared dispatcher :func:`pineuro.mask_extraction.run_mask_extraction_from_config`.
Because both the live pipeline and the generated script go through that one
function, the script cannot drift from the ``mask_extraction_*`` signatures.

For ``rest`` mode the script additionally regenerates the ICA alternative-component
overlays + Yeo overlays + QC report (the GUI mask-selection dialog needs these),
and supports a ``--qc-only`` flag to refresh QC without re-running ICA. ``task``
and ``mni`` modes rely on the QC produced inside the extraction call itself.
"""

import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat
from string import Template
from typing import Any, Union

from pineuro.atlas_registry import ATLAS_SOURCES
from pineuro.mask_extraction import MaskSelection

VALID_MODES = ("rest", "task", "mni")


def generate_mask_extraction_script(
    mode: str,
    params: dict[str, Any],
    *,
    selections: dict[str, MaskSelection] | None = None,
) -> str:
    """Generate a standalone script that reproduces a mask extraction run.

    Parameters
    ----------
    mode : str
        Extraction mode: ``"rest"`` (ICA), ``"task"`` (GLM), or ``"mni"`` (atlas).
    params : dict
        The keyword arguments the pipeline passed to the corresponding
        ``mask_extraction_*`` function (``participant_id``, ``session_id``,
        ``root_dir``, inputs, atlas/threshold settings, ``overwrite``,
        ``qc_enabled``, ...). Path-like values are stringified for portability.
        Runtime-only args (``progress_callback``, ``stop_event``) must be omitted.
    selections : dict, optional
        Rest-only. Maps network names to :class:`MaskSelection` objects; used to
        emit a commented-out block documenting any manual component picks.

    Returns
    -------
    str
        Python script content.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mask extraction mode: {mode!r}. Expected one of {VALID_MODES}.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    participant_id = str(params.get("participant_id", "unknown"))
    norm = _normalize_for_repr(params)

    if mode == "rest":
        # Rest regenerates QC + alternative overlays itself, so the recorded call
        # disables the built-in QC to avoid doing it twice.
        norm = {**norm, "qc_enabled": False}
        selection_block = _generate_selection_code(
            selections,
            n_networks=ATLAS_SOURCES.get(params.get("network_atlas", "yeo_17"), {}).get("n_networks", 17),
            threshold=params.get("threshold", 2000),
        )
        return _REST_TEMPLATE.substitute(
            participant_id=participant_id,
            timestamp=timestamp,
            params_block=pformat(norm, indent=4, width=100, sort_dicts=False),
            selection_block=selection_block,
        )

    return _SIMPLE_TEMPLATE.substitute(
        mode=mode,
        participant_id=participant_id,
        timestamp=timestamp,
        params_block=pformat(norm, indent=4, width=100, sort_dicts=False),
    )


def _normalize_for_repr(value: Any) -> Any:
    """Recursively convert Path values to str so the dict reprs as portable code."""
    if isinstance(value, dict):
        return {k: _normalize_for_repr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_repr(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _generate_selection_code(
    selections: dict[str, MaskSelection] | None,
    n_networks: int,
    threshold: int,
) -> str:
    """Build the (commented-out) STEP 2 block documenting manual mask selections."""
    none_block = """        # -----------------------------------------------------------------
        # STEP 2: No manual mask selections to apply
        # -----------------------------------------------------------------
        # (All masks use automatic ICA component selection)
        print("  No manual mask selections to apply")
        print()
"""
    if not selections:
        return none_block

    manual_selections = {name: sel for name, sel in selections.items() if sel.selection_reason != "auto"}
    if not manual_selections:
        return none_block

    selection_defs = []
    for name, sel in manual_selections.items():
        if sel.source_type == "ica_combined":
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
            selection_defs.append(f'''        #     "{name}": MaskSelection(
        #         network_name="{sel.network_name}",
        #         source_type="{sel.source_type}",
        #         component_index={sel.component_index},
        #         correlation={sel.correlation},
        #         selection_reason="{sel.selection_reason}",
        #     ),''')

    selections_dict = "\n".join(selection_defs)

    return f"""        # -----------------------------------------------------------------
        # STEP 2: Apply manual mask selections (COMMENTED OUT)
        # -----------------------------------------------------------------
        # Uncomment to re-apply the manual selections made in the GUI. By
        # default this script uses the auto-selected ICA components.
        #
        # from pineuro.mask_extraction import (
        #     MaskSelection,
        #     load_mask_selections,
        #     regenerate_mask_from_selection,
        #     save_mask_selections,
        # )
        #
        # masks_dir = root_dir / "03_masks"
        # ica_dir = root_dir / "01_ica"
        # mnireg_dir = root_dir / "02_mnireg"
        #
        # components_files = sorted(ica_dir.glob("*_components.nii.gz"))
        # if not components_files:
        #     raise FileNotFoundError("ICA components file not found")
        # components_file = components_files[0]
        #
        # yeo_files = sorted(mnireg_dir.glob("Yeo_to_*.nii.gz"))
        # if not yeo_files:
        #     raise FileNotFoundError("Yeo atlas in subject space not found")
        # yeo_subject_space = yeo_files[0]
        #
        # manual_selections = {{
{selections_dict}
        # }}
        #
        # for name, selection in manual_selections.items():
        #     regenerate_mask_from_selection(
        #         participant_id=PARAMS["participant_id"],
        #         session_id=PARAMS["session_id"],
        #         masks_dir=masks_dir,
        #         selection=selection,
        #         components_file=components_file,
        #         yeo_subject_space=yeo_subject_space,
        #         n_networks={n_networks},
        #         threshold={threshold},
        #     )
        #
        # existing_selections = load_mask_selections(masks_dir)
        # existing_selections.update(manual_selections)
        # save_mask_selections(masks_dir, existing_selections)
        # print()
"""


_REST_TEMPLATE = Template('''#!/usr/bin/env python3
"""
Mask Extraction Script (rest / ICA) for ${participant_id}
Generated: ${timestamp}

Reproduces the mask extraction pipeline. All parameters captured from the GUI
run are in PARAMS below; edit and re-run to customize.

Usage:
    python reproduce_masks.py

To regenerate only the QC report (after manual edits):
    python reproduce_masks.py --qc-only
"""

import shutil
import sys
from pathlib import Path

from pineuro import qc
from pineuro.atlas_registry import ATLAS_SOURCES
from pineuro.mask_extraction import run_mask_extraction_from_config
from pineuro.qc import generate_alternative_overlays, generate_yeo_network_overlay

MODE = "rest"

# Resolved parameters captured from the GUI run. Edit and re-run to customize.
# qc_enabled is False here because this script regenerates QC + overlays below.
PARAMS = ${params_block}


def clean_stale_outputs(root_dir):
    """Remove outputs that go stale when masks are regenerated."""
    for dirname in ["04_reference", "qc"]:
        dir_path = root_dir / dirname
        if dir_path.exists():
            print(f"  Removing stale directory: {dirname}/")
            shutil.rmtree(dir_path)
    approval_marker = root_dir / ".masks_approved"
    if approval_marker.exists():
        print("  Removing approval marker: .masks_approved")
        approval_marker.unlink()


def main():
    root_dir = Path(PARAMS["root_dir"])
    network_names = PARAMS["networks"]
    n_networks = ATLAS_SOURCES.get(PARAMS["network_atlas"], {}).get("n_networks", 17)
    qc_only = "--qc-only" in sys.argv

    if not qc_only:
        print("=" * 60)
        print("Cleaning up stale outputs")
        print("=" * 60)
        clean_stale_outputs(root_dir)
        print()

        print("=" * 60)
        print("STEP 1: Running mask extraction (ICA)")
        print("=" * 60)
        mask_files = run_mask_extraction_from_config(MODE, PARAMS)
        print(f"  Created {len(mask_files)} mask files")
        print()

${selection_block}
    print("=" * 60)
    print("STEP 3: Regenerating overlay images")
    print("=" * 60)
    qc_dir = root_dir / "qc"
    generate_alternative_overlays(
        mask_extraction_dir=root_dir,
        participant_id=PARAMS["participant_id"],
        session_id=PARAMS["session_id"],
        network_names=network_names,
        n_alternatives=PARAMS["n_components"],
        output_dir=qc_dir,
        n_networks=n_networks,
        overwrite=True,
    )
    for network_name in network_names:
        generate_yeo_network_overlay(
            mask_extraction_dir=root_dir,
            participant_id=PARAMS["participant_id"],
            session_id=PARAMS["session_id"],
            network_name=network_name,
            output_dir=qc_dir,
            n_networks=n_networks,
            overwrite=True,
        )
    print()

    print("=" * 60)
    print("STEP 4: Generating QC report")
    print("=" * 60)
    qc.generate_mask_extraction_report(
        mask_extraction_dir=root_dir,
        participant_id=PARAMS["participant_id"],
        session_id=PARAMS["session_id"],
        network_names=network_names,
        n_networks=n_networks,
        overwrite=True,
        open_browser=False,
    )
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Masks saved to: {root_dir / '03_masks'}")
    print(f"  QC report: {root_dir / 'qc' / 'mask_extraction_qc.html'}")
    if not qc_only:
        print()
        print("NOTE: Masks require re-approval in the GUI before proceeding.")


if __name__ == "__main__":
    main()
''')


_SIMPLE_TEMPLATE = Template('''#!/usr/bin/env python3
"""
Mask Extraction Script (${mode}) for ${participant_id}
Generated: ${timestamp}

Reproduces the mask extraction pipeline. All parameters captured from the GUI
run are in PARAMS below; edit and re-run to customize.

Usage:
    python reproduce_masks.py
"""

import shutil
from pathlib import Path

from pineuro.mask_extraction import run_mask_extraction_from_config

MODE = "${mode}"

# Resolved parameters captured from the GUI run. Edit and re-run to customize.
PARAMS = ${params_block}


def clean_stale_outputs(root_dir):
    """Remove outputs that go stale when masks are regenerated."""
    for dirname in ["04_reference", "qc"]:
        dir_path = root_dir / dirname
        if dir_path.exists():
            print(f"  Removing stale directory: {dirname}/")
            shutil.rmtree(dir_path)
    approval_marker = root_dir / ".masks_approved"
    if approval_marker.exists():
        approval_marker.unlink()


def main():
    root_dir = Path(PARAMS["root_dir"])

    print("=" * 60)
    print("Cleaning up stale outputs")
    print("=" * 60)
    clean_stale_outputs(root_dir)
    print()

    print("=" * 60)
    print(f"Running mask extraction (mode: {MODE})")
    print("=" * 60)
    mask_files = run_mask_extraction_from_config(MODE, PARAMS)
    print(f"  Created {len(mask_files)} mask files")
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Masks saved to: {root_dir / '03_masks'}")


if __name__ == "__main__":
    main()
''')


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
    # Make executable on Unix (Windows doesn't support chmod)
    if sys.platform != "win32":
        output_path.chmod(0o755)
    return output_path
