"""Validation -- shared motion-QC primitives in pineuro.utils.

utils.compute_fd, utils.compute_dvars and utils.parse_siemens_motion_comment are
the single implementation of framewise displacement, DVARS, and Siemens MoCo
motion parsing. Both the real-time feedback estimator (feedback.FeedbackEstimator)
and the live collection-QC monitor (collection_qc.CollectionQCMonitor) delegate to
them, so validating these functions once validates both paths.

The hand-calculated cases below derive every expected value *from the definition*
-- without re-running pineuro's algorithm -- so a bug (wrong head radius, dropped
abs/sqrt, sum vs mean, wrong axis split) flips at least one expected value.
"""

from __future__ import annotations

import numpy as np
import pytest

from pineuro.utils import compute_dvars, compute_fd, parse_siemens_motion_comment

pytestmark = pytest.mark.validation

# Arc length at HEAD_RADIUS_MM=50 mm per degree of rotation, written as an inert
# expression so the test anchors to the definition, not pineuro's compiled formula.
# 50.0 * pi / 180.0 ~= 0.8727 mm/deg is the standard arc-length conversion.
_DEG_TO_MM = 50.0 * np.pi / 180.0

# Hand-calculated FD cases: (prev, curr, expected_fd_mm). FD = sum of absolute
# frame-to-frame differences across the 6 motion params, rotations (indices 3:6,
# in degrees) converted to mm of arc at a 50 mm head radius.
_FD_CASES = [
    # Pure translation along one axis. FD = |delta| for that axis, zero elsewhere.
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1.0),
    # Multi-axis translation. FD = sum of per-axis |delta|.
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, -2.0, 1.0, 0.0, 0.0, 0.0], 6.0),
    # Pure rotation about each axis (degrees -> mm arc).
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], _DEG_TO_MM),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], _DEG_TO_MM),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.0], 2.0 * _DEG_TO_MM),
    # Mixed translation + rotation.
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1.0 + _DEG_TO_MM),
    # Nonzero prev: only the *difference* contributes.
    ([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [2.0, 2.0, 3.0, 0.0, 0.0, 0.0], 1.0),
    # No motion -> FD == 0.
    ([0.5, -0.5, 0.5, 0.1, -0.1, 0.0], [0.5, -0.5, 0.5, 0.1, -0.1, 0.0], 0.0),
]

# Hand-calculated DVARS cases: (prev, curr, expected_dvars). DVARS =
# sqrt(mean((I_t - I_{t-1})^2)) for a small voxel set whose mean-of-squared-diffs
# is trivially known.
_DVARS_CASES = [
    # All voxels move by the same amount k -> mean(k^2) = k^2 -> sqrt = |k|.
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
    ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 2.0),
    ([100.0, 100.0], [102.0, 102.0], 2.0),
    # No change -> dvars exactly 0.
    ([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], 0.0),
    # Asymmetric diff [3, 4] -> mean(9, 16) = 12.5 -> sqrt(12.5).
    ([0.0, 0.0], [3.0, 4.0], float(np.sqrt(12.5))),
    # Mixed sign diff [-1, 1] both contribute -> mean(1, 1) = 1 -> sqrt = 1.0.
    ([5.0, 5.0], [4.0, 6.0], 1.0),
    # Single voxel -> dvars == |diff|.
    ([7.0], [10.0], 3.0),
]


# =============================================================================
# compute_fd
# =============================================================================


def test_fd_first_volume_returns_zero():
    """compute_fd returns 0.0 when prev_motion is None (contract)."""
    curr = np.array([0.25, -0.05, 0.40, -0.01, 0.04, -0.02])
    assert compute_fd(curr, prev_motion=None) == 0.0


@pytest.mark.parametrize("prev, curr, expected_fd", _FD_CASES)
def test_fd_hand_calculated(record, prev, curr, expected_fd):
    """Head-motion (FD) matches a hand calculation.

    Each expected value is derived from the FD definition (sum of |delta| with
    rotations at HEAD_RADIUS_MM=50 mm arc length) without running pineuro's
    algorithm. A bug -- wrong head radius, wrong rotation-to-arc factor, wrong
    sign/abs/sum -- moves the produced value off the expected literal. The 1e-12
    tolerance bounds floating-point ulp noise, not algorithmic correctness.
    """
    produced = compute_fd(
        np.asarray(curr, dtype=np.float64),
        prev_motion=np.asarray(prev, dtype=np.float64),
    )
    err = abs(produced - expected_fd)
    case = f"prev{prev}_curr{curr}".replace(" ", "")
    assert (
        record(
            "utils",
            f"fd_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "Framewise displacement (FD) is how far the head moved between "
                "two volumes. The shared FD function is compared against values "
                "worked out by hand for each test case."
            ),
        )
        < 1e-12
    )


# =============================================================================
# compute_dvars
# =============================================================================


def test_dvars_first_volume_returns_zero():
    """compute_dvars returns 0.0 when prev_voxel_values is None (contract)."""
    curr = np.array([1.0, 2.0, 3.0])
    assert compute_dvars(curr, prev_voxel_values=None) == 0.0


@pytest.mark.parametrize("prev, curr, expected_dvars", _DVARS_CASES)
def test_dvars_hand_calculated(record, prev, curr, expected_dvars):
    """Image-stability (DVARS) matches a hand calculation.

    Each expected value is derived from the DVARS definition (sqrt of the mean of
    squared frame-to-frame differences) without running pineuro's algorithm. A
    bug -- dropped sqrt, sum vs mean, wrong reduction -- moves the produced value
    off the expected literal. The 1e-12 tolerance bounds one-ulp sqrt rounding.
    """
    produced = compute_dvars(
        np.asarray(curr, dtype=np.float64),
        prev_voxel_values=np.asarray(prev, dtype=np.float64),
    )
    err = abs(produced - expected_dvars)
    case = f"prev{prev}_curr{curr}".replace(" ", "")
    assert (
        record(
            "utils",
            f"dvars_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "DVARS measures how much the image changed between two volumes. "
                "The shared DVARS function is compared against values worked out "
                "by hand for each test case."
            ),
        )
        < 1e-12
    )


# =============================================================================
# parse_siemens_motion_comment
# =============================================================================


def test_parse_reference_volume_returns_zeros():
    """The Siemens reference-volume sentinel parses to a zeros vector."""
    motion = parse_siemens_motion_comment("Reference volume for motion correction")
    assert motion is not None
    np.testing.assert_array_equal(motion, np.zeros(6, dtype=np.float64))


def test_parse_no_match_returns_none():
    """A comment without the Motion: pattern parses to None."""
    assert parse_siemens_motion_comment("some unrelated DICOM comment") is None
    assert parse_siemens_motion_comment("") is None


def test_parse_siemens_pattern_exactly(record):
    """Scanner head-motion values are read back exactly.

    The comment format Siemens MoCo writes is the literal "Motion: a,b,c,d,e,f".
    Each parsed value must equal the corresponding literal source value to
    floating-point precision -- at 6 decimal places the round-trip is exact,
    since the printed digits are exactly representable in float64 for these
    inputs.
    """
    values = [0.123, -0.456, 0.789, -0.012, 0.034, -0.056]
    comment = "Motion: " + ",".join(f"{v}" for v in values)
    parsed = parse_siemens_motion_comment(comment)
    assert parsed is not None
    max_err = float(np.max(np.abs(parsed - np.array(values, dtype=np.float64))))
    assert (
        record(
            "utils",
            "motion_parse_max_err",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "Scanners record head motion in a text field of each image. "
                "This checks the six values pineuro reads back match the known "
                "values exactly."
            ),
        )
        == 0.0
    )
