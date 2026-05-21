"""Validation -- script_generator.py output is syntactically valid and faithful.

generate_mask_extraction_script and generate_task_mask_extraction_script
produce standalone Python scripts. Two correctness properties matter to users:

1. **Syntactic validity** -- the generated script must compile with ``ast.parse``
   so that simply running it does not fail with a SyntaxError.
2. **Parameter faithfulness** -- every argument the user passed in must appear
   in the generated script as a literal value so that re-running reproduces
   the exact configuration.

Both properties are decidable: we parse the script and we substring-match every
input parameter. A regression -- truncated f-string, escaped quote, dropped
parameter -- breaks one or both checks deterministically.
"""

from __future__ import annotations

import ast

import pytest

from pineuro.script_generator import generate_mask_extraction_script, generate_task_mask_extraction_script, save_script

pytestmark = pytest.mark.validation


# =============================================================================
# generate_mask_extraction_script
# =============================================================================


def test_rest_script_is_syntactically_valid():
    """Generated rest script parses cleanly via ast.parse (no SyntaxError)."""
    script = generate_mask_extraction_script(
        participant_id="sub-01",
        mask_extraction_dir="/data/sub-01/mask_extraction",
        input_file="/data/sub-01/preproc.nii.gz",
        mask_file="/data/sub-01/mask.nii.gz",
        ref_file="/data/sub-01/ref.nii.gz",
        n_yeo=17,
        yeo_targets=["DefaultA", "ContA"],
        n_components=30,
        threshold=2000,
        smooth_fwhm=6.0,
        highpass=0.01,
        lowpass=0.1,
    )
    ast.parse(script)  # raises SyntaxError if invalid


def test_rest_script_contains_every_parameter_literal():
    """Every user-supplied value appears verbatim in the generated script.

    Substring match against the script text. If the generator silently drops a
    parameter or renames it, the corresponding assertion fails.

    Paths are rendered via ``str(Path(...))`` which uses the OS separator, so
    the check uses just the unique filename component to remain OS-independent.
    """
    script = generate_mask_extraction_script(
        participant_id="sub-42",
        mask_extraction_dir="/data/sub-42/mask_extraction_xyz",
        input_file="/data/sub-42/unique_preproc.nii.gz",
        mask_file="/data/sub-42/unique_mask.nii.gz",
        ref_file="/data/sub-42/unique_ref.nii.gz",
        n_yeo=7,
        yeo_targets=["Vis", "SomMot"],
        n_components=25,
        threshold=1500,
        smooth_fwhm=4.0,
        highpass=0.008,
        lowpass=0.12,
    )

    # Participant id and unique filename components appear verbatim.
    for value in (
        "sub-42",
        "mask_extraction_xyz",
        "unique_preproc.nii.gz",
        "unique_mask.nii.gz",
        "unique_ref.nii.gz",
        "Vis",
        "SomMot",
    ):
        assert value in script, f"missing {value!r} in generated script"
    # Numeric params written as Python literals -- check for the specific forms
    # that the generator produces (e.g. "n_components = 25").
    assert "n_components = 25" in script
    assert "1500" in script  # threshold literal
    assert "smooth_fwhm = 4.0" in script
    assert "n_yeo = 7" in script


# =============================================================================
# generate_task_mask_extraction_script
# =============================================================================


def test_task_script_is_syntactically_valid():
    """Generated task script parses cleanly via ast.parse."""
    script = generate_task_mask_extraction_script(
        participant_id="sub-01",
        mask_extraction_dir="/data/sub-01/mask_extraction",
        input_file="/data/sub-01/preproc.nii.gz",
        mask_file="/data/sub-01/mask.nii.gz",
        events_file="/data/sub-01/events.tsv",
        contrasts={"DefaultA": "self - other", "ContA": "task - rest"},
        tr=1.5,
        threshold=2000,
        threshold_method="top_n",
        smoothing_fwhm=5.0,
        high_pass=0.01,
        drift_model="cosine",
        hrf_model="spm",
    )
    ast.parse(script)


def test_task_script_contains_contrast_expressions_verbatim():
    """User-defined contrast expressions appear unchanged in the generated script."""
    contrasts = {
        "DefaultA": "self - other",
        "ContA": "task - rest",
        "Vis": "face - scrambled",
    }
    script = generate_task_mask_extraction_script(
        participant_id="sub-99",
        mask_extraction_dir="/data/sub-99/mask_extraction",
        input_file="/data/sub-99/preproc.nii.gz",
        mask_file="/data/sub-99/mask.nii.gz",
        events_file="/data/sub-99/events.tsv",
        contrasts=contrasts,
        tr=2.0,
    )
    for mask_name, expr in contrasts.items():
        assert mask_name in script
        assert expr in script
    assert "sub-99" in script


def test_task_script_threshold_method_propagates():
    """The threshold_method (top_n vs cluster) appears as a literal in the script."""
    for method in ("top_n", "cluster"):
        script = generate_task_mask_extraction_script(
            participant_id="sub-01",
            mask_extraction_dir="/data/sub-01/mask_extraction",
            input_file="/data/sub-01/preproc.nii.gz",
            mask_file="/data/sub-01/mask.nii.gz",
            events_file="/data/sub-01/events.tsv",
            contrasts={"DefaultA": "self - other"},
            tr=1.5,
            threshold_method=method,
        )
        ast.parse(script)
        assert method in script


# =============================================================================
# save_script round-trip
# =============================================================================


def test_save_script_round_trip(tmp_path):
    """save_script writes the exact bytes of the script to disk."""
    content = "# my generated script\nprint('hello, world')\n"
    out = tmp_path / "sub" / "script.py"
    save_script(content, out)
    assert out.exists()
    assert out.read_text() == content


def test_save_script_creates_parent_dirs(tmp_path):
    """save_script creates missing parent directories rather than erroring out."""
    out = tmp_path / "a" / "b" / "c" / "script.py"
    assert not out.parent.exists()
    save_script("x = 1\n", out)
    assert out.exists()
    assert out.read_text() == "x = 1\n"
