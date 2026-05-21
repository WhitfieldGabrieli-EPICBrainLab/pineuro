"""Validation -- collection_qc.py FD/DVARS and Siemens MoCo motion extraction.

collection_qc.py reimplements the FD/DVARS math (without the GLM/design-matrix
machinery of feedback.FeedbackEstimator) and parses Siemens MoCo motion params
from a DICOM ImageComments string. Both paths must produce the same numbers as
feedback.py for the same input -- a divergence between the two implementations
would mean the real-time QC monitor and the feedback estimator disagree on the
same data.

The FD/DVARS hand-calculated cases are reused from test_feedback_estimator.py;
the DICOM parser is tested with synthesised ImageComments strings.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pineuro.collection_qc import (
    _compute_dvars,
    _compute_fd,
    _extract_motion_from_dicom_header,
)

pytestmark = pytest.mark.validation

# Reuse the same definition-derived constant as test_feedback_estimator.py:
# arc length at HEAD_RADIUS_MM=50 mm per degree of rotation.
_DEG_TO_MM = 50.0 * np.pi / 180.0

# Hand-calculated FD cases (mirror of test_feedback_estimator._FD_CASES; the math
# is intentionally identical because the two implementations must agree).
_FD_CASES = [
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, -2.0, 1.0, 0.0, 0.0, 0.0], 6.0),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], _DEG_TO_MM),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.0], 2.0 * _DEG_TO_MM),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1.0 + _DEG_TO_MM),
    ([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [2.0, 2.0, 3.0, 0.0, 0.0, 0.0], 1.0),
    ([0.5, -0.5, 0.5, 0.1, -0.1, 0.0], [0.5, -0.5, 0.5, 0.1, -0.1, 0.0], 0.0),
]

_DVARS_CASES = [
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
    ([100.0, 100.0], [102.0, 102.0], 2.0),
    ([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], 0.0),
    ([0.0, 0.0], [3.0, 4.0], float(np.sqrt(12.5))),
    ([7.0], [10.0], 3.0),
]


def test_fd_first_volume_returns_zero():
    """_compute_fd returns 0.0 when prev_motion is None (contract)."""
    curr = np.array([0.25, -0.05, 0.40, -0.01, 0.04, -0.02])
    assert _compute_fd(curr, prev_motion=None) == 0.0


@pytest.mark.parametrize("prev, curr, expected_fd", _FD_CASES)
def test_fd_hand_calculated(record, prev, curr, expected_fd):
    """Quality-monitor head-motion (FD) matches a hand calculation.

    Expected values derived from the FD definition (sum of |Δ| with rotations at
    HEAD_RADIUS_MM=50 mm arc length), independent of pineuro's implementation.
    A bug -- wrong head radius, dropped abs, wrong axis split -- fails the case.
    """
    produced = _compute_fd(
        np.asarray(curr, dtype=np.float64),
        prev_motion=np.asarray(prev, dtype=np.float64),
    )
    err = abs(produced - expected_fd)
    case = f"prev{prev}_curr{curr}".replace(" ", "")
    assert (
        record(
            "collection_qc",
            f"fd_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "The live scan-quality monitor computes framewise displacement (FD) "
                "separately from the feedback estimator. Its FD is checked against "
                "values worked out by hand, so the two paths agree."
            ),
        )
        < 1e-12
    )


def test_dvars_first_volume_returns_zero():
    """_compute_dvars returns 0.0 when prev_voxel_values is None (contract)."""
    curr = np.array([1.0, 2.0, 3.0])
    assert _compute_dvars(curr, prev_voxel_values=None) == 0.0


@pytest.mark.parametrize("prev, curr, expected_dvars", _DVARS_CASES)
def test_dvars_hand_calculated(record, prev, curr, expected_dvars):
    """Quality-monitor image-stability (DVARS) matches a hand calculation."""
    produced = _compute_dvars(
        np.asarray(curr, dtype=np.float64),
        prev_voxel_values=np.asarray(prev, dtype=np.float64),
    )
    err = abs(produced - expected_dvars)
    case = f"prev{prev}_curr{curr}".replace(" ", "")
    assert (
        record(
            "collection_qc",
            f"dvars_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "The live scan-quality monitor's DVARS is checked against values "
                "worked out by hand, confirming it matches the feedback estimator "
                "on the same input."
            ),
        )
        < 1e-12
    )


# =============================================================================
# DICOM motion-param extraction
# =============================================================================


def _fake_dcm(image_comments: str | None) -> SimpleNamespace:
    """Build a minimal pydicom-dataset-like object with one attribute."""
    if image_comments is None:
        return SimpleNamespace()
    return SimpleNamespace(ImageComments=image_comments)


def test_extract_motion_missing_image_comments_returns_none():
    """No ImageComments -> None (contract)."""
    assert _extract_motion_from_dicom_header(_fake_dcm(None)) is None


def test_extract_motion_reference_volume_returns_zeros():
    """Siemens reference volume comment -> zeros vector (per code path)."""
    dcm = _fake_dcm("Reference volume for motion correction")
    motion = _extract_motion_from_dicom_header(dcm)
    assert motion is not None
    np.testing.assert_array_equal(motion, np.zeros(6, dtype=np.float64))


def test_extract_motion_parses_siemens_pattern_exactly(record):
    """Scanner head-motion values are read back exactly.

    The comment format Siemens MoCo writes is the literal "Motion: a,b,c,d,e,f".
    Each parsed value must equal the corresponding literal source value to
    floating-point precision: at 6 decimal places the round-trip is exact, since
    the printed digits are exactly representable in float64 for these inputs.
    """
    values = [0.123, -0.456, 0.789, -0.012, 0.034, -0.056]
    comment = "Motion: " + ",".join(f"{v}" for v in values)
    dcm = _fake_dcm(comment)
    parsed = _extract_motion_from_dicom_header(dcm)
    assert parsed is not None
    max_err = float(np.max(np.abs(parsed - np.array(values, dtype=np.float64))))
    assert (
        record(
            "collection_qc",
            "dicom_motion_parse_max_err",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "Scanners record head motion in a text field of each image. This "
                "checks the six values pineuro reads back match the known values "
                "exactly."
            ),
        )
        == 0.0
    )


def test_extract_motion_no_match_returns_none():
    """ImageComments without the Motion: pattern -> None."""
    dcm = _fake_dcm("some unrelated DICOM comment")
    assert _extract_motion_from_dicom_header(dcm) is None
