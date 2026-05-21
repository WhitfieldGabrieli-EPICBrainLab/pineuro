"""Validation -- ConnectivityEstimator sliding-window Pearson correlation.

The estimator's contract: for each configured (m0, m1) pair, compute Pearson r
between the last ``window`` samples of the two timeseries. Return 0.0 if either
timeseries is shorter than ``window``, or if either has near-zero variance.

These are pure-numpy operations on caller-supplied float lists, so every test
case below is constructed so the expected r is computable by hand:

- ``r = +1`` if y is a positive linear transform of x.
- ``r = -1`` if y is a negative linear transform of x.
- ``r = 0`` if Σ(x - x̄)(y - ȳ) = 0 (orthogonal centred signals).

A bug -- dropped centring, wrong window slice, sign error, wrong reduction --
flips at least one of these expected values.
"""

from __future__ import annotations

import pytest

from pineuro.feedback import ConnectivityEstimator

pytestmark = pytest.mark.validation


def test_construct_rejects_small_window():
    """window < 3 is rejected (insufficient samples for a meaningful correlation)."""
    with pytest.raises(ValueError, match="window"):
        ConnectivityEstimator([("a", "b")], window=2)


def test_construct_rejects_empty_pairs():
    """At least one pair is required."""
    with pytest.raises(ValueError, match="pair"):
        ConnectivityEstimator([], window=10)


def test_name_format_is_corr_m0_m1():
    """Result keys follow the documented ``corr_{m0}_{m1}`` convention."""
    est = ConnectivityEstimator([("net_a", "net_b"), ("net_b", "net_c")], window=3)
    assert est.names == ["corr_net_a_net_b", "corr_net_b_net_c"]


def test_warmup_returns_zero(record):
    """Before the sliding window is full, every pair returns exactly 0.0."""
    est = ConnectivityEstimator([("a", "b")], window=10)
    result = est.compute({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})  # only 3 of 10
    assert (
        record(
            "connectivity",
            "warmup_value",
            result["corr_a_b"],
            threshold=0.0,
            op="==",
            description=(
                "Connectivity needs a full window of samples before it means "
                "anything. Until then it returns exactly zero, not an undefined value."
            ),
        )
        == 0.0
    )


def test_perfect_positive_correlation(record):
    """Two signals moving together correlate at +1.

    For any linear b * x + c with b > 0, centred series are positively
    proportional and the Pearson formula reduces to +1; a missing centring or
    sign flip would not give exactly 1.0.
    """
    est = ConnectivityEstimator([("a", "b")], window=5)
    ts_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    ts_b = [2.0, 4.0, 6.0, 8.0, 10.0]  # = 2 * ts_a (linear, positive)
    result = est.compute({"a": ts_a, "b": ts_b})
    err = abs(result["corr_a_b"] - 1.0)
    assert (
        record(
            "connectivity",
            "r_positive_one_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "Two signals where one is just a scaled copy of the other. Their "
                "correlation must come out at exactly +1."
            ),
        )
        < 1e-12
    )


def test_perfect_negative_correlation(record):
    """Two signals moving oppositely correlate at -1."""
    est = ConnectivityEstimator([("a", "b")], window=5)
    ts_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    ts_b = [5.0, 4.0, 3.0, 2.0, 1.0]  # reverse
    # centred a = [-2, -1, 0, 1, 2]; centred b = [2, 1, 0, -1, -2];
    # dot = -4 -1 + 0 -1 -4 = -10; sum_sq each = 10; r = -10 / sqrt(10*10) = -1.0.
    result = est.compute({"a": ts_a, "b": ts_b})
    err = abs(result["corr_a_b"] - (-1.0))
    assert (
        record(
            "connectivity",
            "r_negative_one_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "Two signals where one rises exactly as the other falls. Their correlation must come out at exactly -1."
            ),
        )
        < 1e-12
    )


def test_orthogonal_signals_zero_correlation(record):
    """Two unrelated signals correlate at zero.

    With a = [+1,-1,+1,-1] (mean 0) and b = [+1,+1,-1,-1] (mean 0),
    the dot product Σ a_i b_i = 1 - 1 - 1 + 1 = 0, so the Pearson r is 0.
    A missing-centring bug would NOT give 0 here -- it would produce a nonzero
    cosine-similarity value.
    """
    est = ConnectivityEstimator([("a", "b")], window=4)
    result = est.compute({"a": [1.0, -1.0, 1.0, -1.0], "b": [1.0, 1.0, -1.0, -1.0]})
    err = abs(result["corr_a_b"] - 0.0)
    assert (
        record(
            "connectivity",
            "r_orthogonal_error",
            err,
            threshold=1e-12,
            op="<",
            description=("Two signals with no shared pattern. Their correlation must come out at exactly zero."),
        )
        < 1e-12
    )


def test_constant_signal_returns_zero(record):
    """A flat signal yields a connectivity of zero."""
    est = ConnectivityEstimator([("a", "b")], window=5)
    result = est.compute({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [42.0, 42.0, 42.0, 42.0, 42.0]})
    # b has zero std -> result is the explicit 0.0 fallback, not a NaN, not the
    # mathematical "undefined". Recorded as exact equality.
    assert (
        record(
            "connectivity",
            "constant_signal_value",
            result["corr_a_b"],
            threshold=0.0,
            op="==",
            description=(
                "When one region's signal never changes, its correlation is "
                "undefined. pineuro returns zero rather than an undefined value."
            ),
        )
        == 0.0
    )


def test_sliding_window_uses_only_last_n_samples(record):
    """Only the most recent samples count toward connectivity.

    Pad the start of both timeseries with constants. The full timeseries would
    have very different r (the constants dominate), but with the sliding window
    only the trailing samples remain -- which were chosen so that r = +1.
    """
    est = ConnectivityEstimator([("a", "b")], window=3)
    # Trailing 3 samples: a = [1, 2, 3], b = [2, 4, 6] -> r = +1.0
    # Padded prefix is constant but does not affect the trailing window.
    ts_a = [100.0, 100.0, 100.0, 1.0, 2.0, 3.0]
    ts_b = [50.0, 50.0, 50.0, 2.0, 4.0, 6.0]
    result = est.compute({"a": ts_a, "b": ts_b})
    err = abs(result["corr_a_b"] - 1.0)
    assert (
        record(
            "connectivity",
            "sliding_window_recent_only_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "Both signals are padded with stale constant values up front. "
                "Connectivity must use only the recent samples and ignore the padding."
            ),
        )
        < 1e-12
    )


def test_multiple_pairs_independent(record):
    """Each pair is computed independently; no cross-talk between pairs."""
    est = ConnectivityEstimator([("a", "b"), ("a", "c")], window=5)
    result = est.compute(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # = a -> r = +1
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],  # = reverse a -> r = -1
        }
    )
    err_ab = abs(result["corr_a_b"] - 1.0)
    err_ac = abs(result["corr_a_c"] - (-1.0))
    max_err = max(err_ab, err_ac)
    assert (
        record(
            "connectivity",
            "multipair_max_error",
            max_err,
            threshold=1e-12,
            op="<",
            description=(
                "Two region pairs are tracked in the same call, one correlating at "
                "+1 and one at -1. Each must be computed without disturbing the other."
            ),
        )
        < 1e-12
    )


def test_unequal_length_uses_shorter(record):
    """Connectivity uses the shorter series when two regions differ in length."""
    est = ConnectivityEstimator([("a", "b")], window=3)
    # b has only 3 samples, a has 5; min length is 3 -- exactly window size.
    # Trailing 3 of both: a = [3, 4, 5], b = [3, 4, 5] -> r = +1.0.
    result = est.compute({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [3.0, 4.0, 5.0]})
    err = abs(result["corr_a_b"] - 1.0)
    assert (
        record(
            "connectivity",
            "unequal_length_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "When two regions have different numbers of samples, connectivity "
                "should use the recent samples that both share."
            ),
        )
        < 1e-12
    )
