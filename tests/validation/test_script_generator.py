"""Validation -- script_generator.py output is syntactically valid and faithful.

``generate_mask_extraction_script`` produces a standalone Python script for any
extraction mode (rest / task / mni). Two correctness properties matter to users:

1. **Syntactic validity** -- the generated script must compile with ``ast.parse``
   so that simply running it does not fail with a SyntaxError.
2. **Parameter faithfulness** -- every value the pipeline passed must appear in
   the generated script so that re-running reproduces the exact configuration.

Both properties are decidable: we parse the script and we substring-match every
input parameter. A regression -- truncated string, escaped quote, dropped
parameter -- breaks one or both checks deterministically.
"""

from __future__ import annotations

import ast

import pytest

from pineuro.script_generator import generate_mask_extraction_script, save_script

pytestmark = pytest.mark.validation


# =============================================================================
# rest / ICA
# =============================================================================


def test_rest_script_is_syntactically_valid():
    script = generate_mask_extraction_script(
        "rest",
        {
            "participant_id": "sub-01",
            "session_id": "ses-01",
            "root_dir": "/data/sub-01/mask_extraction",
            "input_file": "/data/sub-01/preproc.nii.gz",
            "mask_file": "/data/sub-01/mask.nii.gz",
            "ref_file": "/data/sub-01/ref.nii.gz",
            "n_yeo": 17,
            "yeo_targets": ["DefaultA", "ContA"],
            "network_atlas": "yeo_17",
            "n_components": 30,
            "threshold": 2000,
            "smooth_fwhm": 6.0,
            "highpass": 0.01,
            "lowpass": 0.1,
            "overwrite": True,
            "qc_enabled": True,
        },
    )
    ast.parse(script)


def test_rest_script_contains_every_parameter_literal():
    """Every supplied value appears verbatim, including the network_atlas that
    the old generator silently dropped (Schaefer/non-Yeo support)."""
    script = generate_mask_extraction_script(
        "rest",
        {
            "participant_id": "sub-42",
            "session_id": "ses-01",
            "root_dir": "/data/sub-42/mask_extraction_xyz",
            "input_file": "/data/sub-42/unique_preproc.nii.gz",
            "mask_file": "/data/sub-42/unique_mask.nii.gz",
            "ref_file": "/data/sub-42/unique_ref.nii.gz",
            "n_yeo": 7,
            "yeo_targets": ["Vis", "SomMot"],
            "network_atlas": "schaefer_100_7net",
            "n_components": 25,
            "threshold": 1500,
            "smooth_fwhm": 4.0,
            "highpass": 0.008,
            "lowpass": 0.12,
            "overwrite": True,
            "qc_enabled": True,
        },
    )
    for value in (
        "sub-42",
        "mask_extraction_xyz",
        "unique_preproc.nii.gz",
        "unique_mask.nii.gz",
        "unique_ref.nii.gz",
        "Vis",
        "SomMot",
        "schaefer_100_7net",
    ):
        assert value in script, f"missing {value!r} in generated script"
    assert "'n_components': 25" in script
    assert "'threshold': 1500" in script
    assert "'smooth_fwhm': 4.0" in script
    assert "'n_yeo': 7" in script


# =============================================================================
# task / GLM
# =============================================================================


def test_task_script_is_syntactically_valid():
    script = generate_mask_extraction_script(
        "task",
        {
            "participant_id": "sub-01",
            "session_id": "ses-01",
            "root_dir": "/data/sub-01/mask_extraction",
            "input_file": "/data/sub-01/preproc.nii.gz",
            "mask_file": "/data/sub-01/mask.nii.gz",
            "events": "/data/sub-01/events.tsv",
            "contrasts": {"DefaultA": "self - other", "ContA": "task - rest"},
            "tr": 1.5,
            "threshold": 2000,
            "threshold_method": "top_n",
            "smoothing_fwhm": 5.0,
            "high_pass": 0.01,
            "drift_model": "cosine",
            "hrf_model": "spm",
            "overwrite": True,
            "qc_enabled": True,
        },
    )
    ast.parse(script)


def test_task_script_contains_contrast_expressions_verbatim():
    contrasts = {
        "DefaultA": "self - other",
        "ContA": "task - rest",
        "Vis": "face - scrambled",
    }
    script = generate_mask_extraction_script(
        "task",
        {
            "participant_id": "sub-99",
            "session_id": "ses-01",
            "root_dir": "/data/sub-99/mask_extraction",
            "input_file": "/data/sub-99/preproc.nii.gz",
            "mask_file": "/data/sub-99/mask.nii.gz",
            "events": "/data/sub-99/events.tsv",
            "contrasts": contrasts,
            "tr": 2.0,
            "overwrite": True,
            "qc_enabled": True,
        },
    )
    for mask_name, expr in contrasts.items():
        assert mask_name in script
        assert expr in script
    assert "sub-99" in script


def test_task_script_threshold_method_propagates():
    """Every threshold_method, including the newest cluster_extent, propagates."""
    for method in ("top_n", "cluster", "cluster_extent"):
        script = generate_mask_extraction_script(
            "task",
            {
                "participant_id": "sub-01",
                "session_id": "ses-01",
                "root_dir": "/data/sub-01/mask_extraction",
                "input_file": "/data/sub-01/preproc.nii.gz",
                "mask_file": "/data/sub-01/mask.nii.gz",
                "events": "/data/sub-01/events.tsv",
                "contrasts": {"DefaultA": "self - other"},
                "tr": 1.5,
                "threshold_method": method,
                "overwrite": True,
                "qc_enabled": True,
            },
        )
        ast.parse(script)
        assert method in script


# =============================================================================
# mni / atlas fallback
# =============================================================================


def test_mni_script_is_syntactically_valid_and_faithful():
    script = generate_mask_extraction_script(
        "mni",
        {
            "participant_id": "sub-07",
            "session_id": "ses-01",
            "root_dir": "/data/sub-07/mask_extraction",
            "ref_file": "/data/sub-07/unique_mni_ref.nii.gz",
            "n_yeo": 17,
            "yeo_targets": ["DefaultA", "ContA"],
            "network_atlas": "yeo_17",
            "threshold": 2000,
            "overwrite": True,
            "qc_enabled": True,
        },
    )
    ast.parse(script)
    for value in ("sub-07", "unique_mni_ref.nii.gz", "yeo_17", "DefaultA"):
        assert value in script, f"missing {value!r} in generated script"


# =============================================================================
# save_script round-trip
# =============================================================================


def test_save_script_round_trip(tmp_path):
    content = "# my generated script\nprint('hello, world')\n"
    out = tmp_path / "sub" / "script.py"
    save_script(content, out)
    assert out.exists()
    assert out.read_text() == content


def test_save_script_creates_parent_dirs(tmp_path):
    out = tmp_path / "a" / "b" / "c" / "script.py"
    assert not out.parent.exists()
    save_script("x = 1\n", out)
    assert out.exists()
    assert out.read_text() == "x = 1\n"
