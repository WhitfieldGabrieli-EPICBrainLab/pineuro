"""Validation — numeric core (fast, no external tools).

Known-answer checks for the deterministic numerical machinery: the HRF and
task-design regressors, the activation smoother / normalizer, and the
incremental-GLM solver at the heart of real-time feedback. Every value here has
an analytically or independently computable ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gamma

from pineuro.feedback import IncrementalGLM, _combine_roi
from pineuro.tasks.designs import BlockDesign, EventRelatedDesign, NuisanceOnlyDesign
from pineuro.tasks.utils import ActivationNormalizer, ActivationSmoother
from pineuro.utils import compute_boxcar_task_regressor, compute_hrf

pytestmark = pytest.mark.validation


# =============================================================================
# HRF
# =============================================================================


def _reference_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """Independent reimplementation of pineuro's compute_hrf algorithm.

    Built from ``scipy.stats.gamma`` (pineuro rolls its own gamma PDF), so a
    match confirms the implementation, not a shared bug. pineuro samples a
    double-gamma at 16 Hz, normalizes the high-res kernel to sum 1, then
    *down-samples* by picking every ``int(tr*16)``-th sample.
    """
    hires_dt = 1.0 / 16.0
    n_hires = int(duration / hires_dt)
    t = np.arange(n_hires, dtype=np.float64)
    hires = gamma.pdf(t, 6.0, scale=16.0) - gamma.pdf(t, 16.0, scale=16.0) / 6.0
    hires = hires / hires.sum()
    samples_per_tr = int(tr / hires_dt)
    n_tr = int(duration / tr)
    hrf = np.zeros(n_tr, dtype=np.float64)
    for i in range(n_tr):
        idx = i * samples_per_tr
        if idx < len(hires):
            hrf[i] = hires[idx]
    return hrf


@pytest.mark.parametrize("tr", [1.0, 1.2, 2.0])
def test_hrf_matches_double_gamma(tr, record):
    """The HRF matches a reference double-gamma curve."""
    produced = compute_hrf(tr)
    reference = _reference_hrf(tr)
    assert produced.shape == reference.shape
    max_err = float(np.max(np.abs(produced - reference)))
    assert (
        record(
            "hrf",
            f"hrf_max_abs_error_tr{tr}",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "The HRF curve pineuro generates, compared against an "
                "independently coded double-gamma reference. The two should be "
                "identical apart from floating-point rounding."
            ),
        )
        < 1e-9
    )


def test_hrf_shape_peak_and_undershoot(record):
    """The HRF peaks at exactly 5 s post-onset and has a negative undershoot."""
    tr = 1.0
    hrf = compute_hrf(tr)
    # peak_time = argmax(hrf) * tr — an integer sample index times the TR. At tr=1.0
    # the canonical double-gamma peak lands deterministically on the 5 s sample.
    peak_time = float(np.argmax(hrf)) * tr
    assert (
        record(
            "hrf",
            "hrf_peak_time_s",
            peak_time,
            threshold=5.0,
            op="==",
            units="s",
            description=(
                "Where the HRF reaches its maximum. The standard double-gamma shape peaks 5 s after the event."
            ),
        )
        == 5.0
    )
    assert hrf.min() < 0.0  # the undershoot lobe
    assert abs(hrf[-1]) < 0.01  # decays back toward zero


def test_boxcar_regressor_recovers_block(record):
    """A block-task regressor is flat at zero during baseline and rises to one."""
    n_baseline, n_volumes, tr = 20, 120, 1.0
    reg = compute_boxcar_task_regressor(n_baseline, n_volumes, tr)
    assert reg.shape == (n_volumes,)
    # A causal HRF convolved with a boxcar that is zero through baseline leaves the
    # baseline samples a sum of literal zeros — exactly 0, not merely small.
    baseline_max = float(np.max(np.abs(reg[:n_baseline])))
    assert (
        record(
            "design",
            "boxcar_baseline_max",
            baseline_max,
            threshold=0.0,
            op="==",
            description=(
                "The largest regressor value during the baseline, before the task "
                "starts. Nothing has happened yet, so every baseline value must be "
                "exactly zero."
            ),
        )
        == 0.0
    )
    # reg[-1] is np.convolve's running sum of the normalised HRF kernel: mathematically
    # 1.0, but the summation rounding is not portably exact -- a tolerance, not ==.
    steady_error = abs(float(reg[-1]) - 1.0)
    assert (
        record(
            "design",
            "boxcar_steady_state_error",
            steady_error,
            threshold=1e-9,
            op="<",
            description=(
                "How far the regressor's final value sits from one. A sustained "
                "block drives the modelled response to a steady level of one."
            ),
        )
        < 1e-9
    )


# =============================================================================
# Task designs
# =============================================================================


def test_block_design_matches_boxcar_util():
    """BlockDesign delegates to compute_boxcar_task_regressor exactly."""
    n_baseline, n_volumes, tr = 25, 150, 1.2
    design = BlockDesign(n_baseline).compute_regressors(n_volumes, tr)
    expected = compute_boxcar_task_regressor(n_baseline, n_volumes, tr).reshape(-1, 1)
    assert design.shape == (n_volumes, 1)
    np.testing.assert_array_equal(design, expected)


def test_nuisance_only_design_is_empty():
    """NuisanceOnlyDesign produces a (n_volumes, 0) regressor matrix."""
    design = NuisanceOnlyDesign().compute_regressors(n_volumes=100, tr=1.0)
    assert design.shape == (100, 0)


def test_event_design_recovers_onset(record):
    """An event regressor peaks exactly one HRF lag (5 s) after the event onset."""
    tr, n_volumes, onset = 1.0, 60, 20.0
    design = EventRelatedDesign({"cond": [onset]})
    reg = design.compute_regressors(n_volumes, tr)
    assert reg.shape == (n_volumes, 1)
    # lag = peak_vol * tr - onset — an integer sample index times the TR, minus the
    # onset; all exactly representable, so the HRF lag is deterministically 5.0 s.
    peak_vol = int(np.argmax(reg[:, 0]))
    lag_s = peak_vol * tr - onset
    assert (
        record(
            "design",
            "event_hrf_lag_s",
            lag_s,
            threshold=5.0,
            op="==",
            units="s",
            description=(
                "The gap between an event and the peak of its modelled response. The HRF delays the response by 5 s."
            ),
        )
        == 5.0
    )
    assert reg[:, 0].max() == 1.0  # EventRelatedDesign divides by the peak: max/max == 1


def test_event_design_multi_condition():
    """EventRelatedDesign yields one column per condition, each localised."""
    design = EventRelatedDesign({"a": [10.0], "b": [35.0]})
    reg = design.compute_regressors(n_volumes=60, tr=1.0)
    assert reg.shape == (60, 2)
    assert int(np.argmax(reg[:, 0])) < int(np.argmax(reg[:, 1]))  # a precedes b


# =============================================================================
# Activation smoother / normalizer
# =============================================================================


@pytest.mark.parametrize("alpha", [0.2, 0.5])
def test_smoother_follows_ema_formula(alpha, record):
    """Activation smoothing follows the expected exponential curve."""
    smoother = ActivationSmoother(alpha=alpha)
    max_err = 0.0
    for t in range(1, 41):
        value = smoother.update(1.0)
        expected = 1.0 - (1.0 - alpha) ** t
        max_err = max(max_err, abs(value - expected))
    # alpha=0.5 is dyadic, so the EMA recursion and 1-(1-alpha)^t are bit-exact; alpha=0.2
    # is not representable, so its rounding gets a machine-precision tolerance instead.
    if alpha == 0.5:
        assert (
            record(
                "smoothing",
                f"smoother_max_error_a{alpha}",
                max_err,
                threshold=0.0,
                op="==",
                description=(
                    "pineuro's smoothed activation compared against the exact "
                    "exponential-smoothing curve. Checked at two smoothing strengths."
                ),
            )
            == 0.0
        )
    else:
        assert (
            record(
                "smoothing",
                f"smoother_max_error_a{alpha}",
                max_err,
                threshold=1e-12,
                op="<",
                description=(
                    "pineuro's smoothed activation compared against the exact "
                    "exponential-smoothing curve. Checked at two smoothing strengths."
                ),
            )
            < 1e-12
        )


def test_normalizer_warmup_returns_neutral():
    """ActivationNormalizer returns 0.5 until it has at least three samples."""
    norm = ActivationNormalizer(window_size=50)
    assert norm.update(7.0) == 0.5
    assert norm.update(9.0) == 0.5


def test_normalizer_maps_to_known_points(record):
    """Activation rescaling maps known input positions to the expected output.

    With a window of {0.0, 2.0} values the 10th/90th percentiles fall on the flat
    0.0 and 2.0 regions, so the bounds are exactly 0.0 and 2.0. (1-0)/(2-0) is
    exactly 0.5 in IEEE-754, so the window midpoint normalizes to exactly 0.5; the
    bounds map to exactly 0.0 and 1.0, and an out-of-range value clips to exactly 1.0.
    """
    norm = ActivationNormalizer(window_size=50, lower_percentile=10, upper_percentile=90)
    norm.update(0.0)  # warmup -- < 3 samples returns 0.5
    norm.update(2.0)  # warmup
    norm.update(0.0)
    norm.update(2.0)
    midpoint_error = abs(norm.update(1.0) - 0.5)
    assert (
        record(
            "smoothing",
            "normalizer_midpoint_error",
            midpoint_error,
            threshold=0.0,
            op="==",
            description=(
                "Activation is rescaled to a 0-1 range from its recent spread. A "
                "value at the middle of that spread must land at 0.5."
            ),
        )
        == 0.0
    )
    assert norm.update(0.0) == 0.0  # lower bound -> exactly 0
    assert norm.update(2.0) == 1.0  # upper bound -> exactly 1
    assert norm.update(100.0) == 1.0  # above the window -> clipped to exactly 1


# =============================================================================
# IncrementalGLM — the real-time least-squares solver
# =============================================================================


def _solve(glm: IncrementalGLM) -> np.ndarray:
    """Back-solve the GLM's current QR state for the parameter estimates."""
    p = glm.n_regressors
    return glm._solve_via_backsubstitution(glm._R[:, :p], glm._R[:, p:])


def test_glm_matches_lstsq(record):
    """The real-time GLM matches a standard batch regression."""
    rng = np.random.default_rng(0)
    n_obs, p = 80, 4
    design = rng.standard_normal((n_obs, p))
    beta_true = np.array([2.0, -0.5, 0.3, 1.1])
    y = design @ beta_true + rng.normal(0.0, 0.1, n_obs)

    glm = IncrementalGLM(n_regressors=p, n_nuisance=1, n_outputs=1)
    for t in range(n_obs):
        glm.update(design[t], y[t])
    beta_glm = _solve(glm)[:, 0]
    beta_lstsq, *_ = np.linalg.lstsq(design, y, rcond=None)

    max_err = float(np.max(np.abs(beta_glm - beta_lstsq)))
    assert (
        record(
            "feedback_glm",
            "glm_vs_lstsq_max_error",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "pineuro's real-time GLM updates one volume at a time; a standard "
                "batch regression sees all volumes at once. On identical data the "
                "two must reach the same fit."
            ),
        )
        < 1e-9
    )


def test_glm_qr_factor_satisfies_normal_equations(record):
    """The real-time GLM solves the correct regression."""
    rng = np.random.default_rng(1)
    n_obs, p = 60, 5
    design = rng.standard_normal((n_obs, p))
    y = rng.standard_normal(n_obs)

    glm = IncrementalGLM(n_regressors=p, n_nuisance=1, n_outputs=1)
    for t in range(n_obs):
        glm.update(design[t], y[t])

    r_factor = glm._R[:, :p]
    reconstructed = r_factor.T @ r_factor
    gram = design.T @ design
    max_err = float(np.max(np.abs(reconstructed - gram)))
    rel_err = max_err / float(np.max(np.abs(gram)))
    assert (
        record(
            "feedback_glm",
            "glm_qr_relative_error",
            rel_err,
            threshold=1e-9,
            op="<",
            description=(
                "A self-consistency check on the running fit pineuro accumulates "
                "volume by volume. Near zero means it is solving the intended "
                "regression."
            ),
        )
        < 1e-9
    )


def test_glm_recovers_known_betas_noise_free(record):
    """The GLM recovers known coefficients from noise-free data."""
    rng = np.random.default_rng(2)
    n_obs, p = 50, 3
    design = rng.standard_normal((n_obs, p))
    beta_true = np.array([1.5, -2.0, 0.75])
    y = design @ beta_true  # noise-free

    glm = IncrementalGLM(n_regressors=p, n_nuisance=1, n_outputs=1)
    for t in range(n_obs):
        glm.update(design[t], y[t])
    beta_glm = _solve(glm)[:, 0]

    max_err = float(np.max(np.abs(beta_glm - beta_true)))
    assert (
        record(
            "feedback_glm",
            "glm_known_beta_error",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "Test data is built from known regression coefficients with no "
                "noise added. The GLM should recover those coefficients exactly."
            ),
        )
        < 1e-9
    )


def test_glm_handles_rank_deficiency():
    """A degenerate (all-zero) regressor gets beta 0 and triggers one warning."""
    rng = np.random.default_rng(3)
    n_obs, p = 40, 3
    design = rng.standard_normal((n_obs, p))
    design[:, 0] = 0.0  # e.g. a task regressor that is zero throughout baseline
    y = design @ np.array([0.0, 1.0, -1.0]) + rng.normal(0.0, 0.05, n_obs)

    glm = IncrementalGLM(n_regressors=p, n_nuisance=1, n_outputs=1)
    for t in range(n_obs):
        activation = glm.update(design[t], y[t])
    assert activation is not None  # the solver does not crash on the dead column

    beta_glm = _solve(glm)[:, 0]
    assert beta_glm[0] == 0.0  # dead column forced to zero
    assert glm._rank_deficient_warned is True  # warned (exactly once per session)


def test_glm_activation_tracks_unmodelled_signal():
    """Activation is higher post-baseline once an unmodelled signal appears.

    Directional check (not a recorded metric): the GLM models only a nuisance
    intercept, so the unmodelled +50 post-baseline step stays in the residual and
    lifts the activation. The lift is not an exact quantity -- the IncrementalGLM's
    activation carries warmup transients (it is nonzero even for a constant input,
    before the steady state settles), so only the *direction* is asserted.
    """
    n_baseline, n_obs = 20, 90
    task_bump = np.where(np.arange(n_obs) >= n_baseline, 50.0, 0.0)
    y = 1000.0 + task_bump  # noise-free; the +50 post-baseline step is unmodelled

    glm = IncrementalGLM(n_regressors=1, n_nuisance=1, n_outputs=1)
    activations: list[float | None] = []
    for t in range(n_obs):
        result = glm.update(np.array([1.0]), y[t])
        activations.append(None if result is None else float(result[0]))

    baseline = [a for t, a in enumerate(activations) if a is not None and t < n_baseline]
    post = [a for t, a in enumerate(activations) if a is not None and t >= n_baseline]
    assert np.mean(post) > np.mean(baseline)


def test_glm_freeze_steady_state_is_stable():
    """After freeze_steady_state the frozen RSS no longer changes."""
    rng = np.random.default_rng(5)
    glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=3)
    for _ in range(25):
        glm.update(rng.standard_normal(2), rng.standard_normal(3))

    assert glm.is_frozen() is False
    glm.freeze_steady_state()
    assert glm.is_frozen() is True
    frozen_rss = glm._steady_state_rss.copy()
    frozen_mse = glm._steady_state_residual_mse.copy()

    # Feed high-variance data; the frozen steady-state values must not move.
    for _ in range(25):
        glm.update(rng.standard_normal(2), rng.normal(0.0, 100.0, 3))
    np.testing.assert_array_equal(glm._steady_state_rss, frozen_rss)
    np.testing.assert_array_equal(glm._steady_state_residual_mse, frozen_mse)


# =============================================================================
# ROI combination
# =============================================================================


def test_combine_roi_mean_is_unweighted_average():
    """_combine_roi('mean') returns the plain mean of per-voxel activations."""
    activations = np.array([1.0, 2.0, 3.0, 4.0])
    rss = np.array([1.0, 10.0, 100.0, 1000.0])
    assert _combine_roi(activations, rss, "mean") == 2.5  # mean of 1,2,3,4 is exactly 2.5


# Hand-calculated inverse-SD weighting cases. By definition (Hinds et al. 2011)
# weight_i = 1/sigma_i with sigma_i = sqrt(rss_i), and the combined value is
# sum(w_i * a_i) / sum(w_i). Each (activations, rss, expected) below is derived
# from that definition for inputs where the result is trivially known, and is
# chosen to *discriminate* between the correct formula and likely-buggy variants
# (e.g. w = 1/rss instead of 1/sqrt(rss), missing normalisation, wrong reduction).
_INVERSE_SD_CASES = [
    # Equal weights (rss all equal) -> plain mean of activations. Reduces to the
    # mean under any reasonable weighting; validates the basic combination.
    ([10.0, 10.0], [1.0, 1.0], 10.0),
    ([3.0, 5.0, 7.0], [4.0, 4.0, 4.0], 5.0),
    # Discriminative — rss=[1, 100] -> sigma=[1, 10] -> w=[1, 0.1].
    # With activation=10 on the clean voxel and 0 on the noisy voxel:
    # combined = (10*1 + 0*0.1) / (1 + 0.1) = 10 / 1.1.
    # A bug using w = 1/rss instead would give 10 / 1.01 -- a different number.
    ([10.0, 0.0], [1.0, 100.0], 10.0 / 1.1),
    # Discriminative — rss=[1, 4] -> sigma=[1, 2] -> w=[1, 0.5].
    # With activations=[6, 0]: combined = (6*1 + 0*0.5) / 1.5 = 4.0 exactly.
    # A bug using w = 1/rss would give 6 / 1.25 = 4.8 -- different.
    ([6.0, 0.0], [1.0, 4.0], 4.0),
    # Single voxel -> activation passes through unchanged regardless of weight.
    ([5.0], [4.0], 5.0),
    # All activations equal -> result equals that value regardless of weighting.
    ([7.5, 7.5, 7.5, 7.5], [1.0, 9.0, 16.0, 25.0], 7.5),
]


@pytest.mark.parametrize("activations, rss, expected", _INVERSE_SD_CASES)
def test_combine_roi_inverse_sd_hand_calculated(record, activations, rss, expected):
    """Region averaging weights steadier voxels more, matching a hand calculation.

    Each expected value is derived from the Hinds et al. 2011 definition
    (w_i = 1/sigma_i; combined = sum(w_i * a_i) / sum(w_i)) without re-running
    pineuro's formula. The discriminative cases (rss with different values, mixed
    activations) would yield different numbers under common buggy variants such
    as ``w = 1/rss`` instead of ``w = 1/sqrt(rss)`` -- so a real algorithmic bug
    fails the test rather than silently passing.
    """
    a = np.asarray(activations, dtype=np.float64)
    r = np.asarray(rss, dtype=np.float64)
    combined = _combine_roi(a, r, "inverse_sd")
    err = abs(combined - expected)
    # Per-case label keeps the report rows distinguishable.
    case = f"a{activations}_rss{rss}".replace(" ", "")
    assert (
        record(
            "feedback_glm",
            f"combine_roi_inverse_sd_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "pineuro averages a region's voxels with steadier voxels weighted "
                "more heavily than noisy ones. This compares that weighted average "
                "against one computed by hand."
            ),
        )
        < 1e-12
    )


def test_combine_roi_inverse_sd_downweights_noisy_voxels(record):
    """Directional property: clean voxels dominate the weighted average.

    With clean voxels at activation 10 and noisy voxels (100x noisier) at activation
    2, the inverse-SD weighting must pull the result much closer to 10 than to a
    plain mean of 6 would. This is a property check distinct from the exact
    hand-calculated values above -- it catches sign / weighting-direction bugs.
    """
    activations = np.array([10.0, 10.0, 2.0, 2.0])
    rss = np.array([1.0, 1.0, 100.0, 100.0])  # last two voxels are 100x noisier
    combined = _combine_roi(activations, rss, "inverse_sd")
    assert combined > 6.0  # plain-mean baseline: clean voxels must pull above it
    assert combined < 10.0  # but not all the way to the clean value
