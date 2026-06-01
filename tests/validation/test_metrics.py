"""Self-tests for the validation recovery metrics (tests/validation/metrics.py).

The whole validation suite's verdicts rest on these helpers: if ``pearson_r`` had a
sign bug or ``dice`` an off-by-one, every recovery check would be quietly wrong. These
assert each metric against hand-calculated values from first principles, so the suite's
measuring stick is itself verified.

Pure numpy -- no ANTs/nilearn -- so this is marked ``validation`` (it exercises
validation-only infrastructure) but not ``slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from .metrics import (
    best_match_correlation,
    dice,
    euclidean_offset,
    pearson_r,
    relative_error,
    rmse_z,
)

pytestmark = pytest.mark.validation


class TestPearsonR:
    def test_perfect_positive(self):
        assert pearson_r([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert pearson_r([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_constant_input_is_nan(self):
        # std == 0 on either side is undefined -> nan (never 0 or 1, which would be
        # silently wrong recovery verdicts).
        assert math.isnan(pearson_r([5, 5, 5], [1, 2, 3]))
        assert math.isnan(pearson_r([1, 2, 3], [7, 7, 7]))


class TestRmseZ:
    def test_identical_after_zscore_is_zero(self):
        # Same series (any affine rescale) z-scores identically -> 0.
        assert rmse_z([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(0.0, abs=1e-12)

    def test_anticorrelated_is_two(self):
        # z(x) and z(-x) differ by 2*z; with unit-variance z, RMS == 2.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert rmse_z(x, [-v for v in x]) == pytest.approx(2.0)


class TestDice:
    def test_partial_overlap(self):
        # |A∩B| = 1, |A| = 2, |B| = 2 -> 2*1/(2+2) = 0.5
        assert dice([0, 1, 1], [1, 1, 0]) == pytest.approx(0.5)

    def test_both_empty_is_one(self):
        assert dice([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        assert dice([1, 1, 0], [0, 0, 1]) == pytest.approx(0.0)

    def test_identical_is_one(self):
        assert dice([1, 0, 1, 1], [1, 0, 1, 1]) == pytest.approx(1.0)


class TestRelativeError:
    def test_basic(self):
        assert relative_error(11.0, 10.0) == pytest.approx(0.1)

    def test_both_zero_is_zero(self):
        assert relative_error(0.0, 0.0) == pytest.approx(0.0)

    def test_true_zero_estimate_nonzero_is_nan(self):
        assert math.isnan(relative_error(5.0, 0.0))


class TestEuclideanOffset:
    def test_345_triangle(self):
        assert euclidean_offset([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)

    def test_zero_when_identical(self):
        assert euclidean_offset([1.5, -2.0, 3.0], [1.5, -2.0, 3.0]) == pytest.approx(0.0)


class TestBestMatchCorrelation:
    def test_diagonal_match_is_one(self):
        # 3 estimated maps that exactly contain the 2 true maps -> each true map's
        # best |r| over the estimated set is 1.0. Order/sign are arbitrary, so put a
        # sign-flipped and a shuffled copy in to exercise the abs + best-match search.
        true = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        estimated = np.array(
            [
                [-1.0, -2.0, -3.0, -4.0],  # true[0] sign-flipped -> |r| == 1
                [0.0, 1.0, 0.0, 1.0],  # noise
                [4.0, 3.0, 2.0, 1.0],  # exactly true[1] -> r == 1
            ]
        )
        out = best_match_correlation(estimated, true)
        np.testing.assert_allclose(out, [1.0, 1.0], atol=1e-9)
