"""Validation — real-time feedback estimation.

Drives FeedbackEstimator.process_volume directly with in-memory phantom volumes
(no disk I/O, no external tools) and checks that the activation it reports
recovers the signal injected into a known ROI.
"""

from __future__ import annotations

import numpy as np
import pytest

from pineuro.feedback import FeedbackEstimator

from .metrics import pearson_r
from .phantom import PhantomSpec, ROISpec, build_phantom, hrf_convolved_events, write_nifti
from .references import _reference_inverse_sd_timeseries, _reference_with_drift_timeseries

pytestmark = pytest.mark.validation

_HEAD_RADIUS_MM = 50.0  # pineuro's framewise-displacement head-radius constant


def _expected_recovery(signal: np.ndarray) -> np.ndarray:
    """The FeedbackEstimator's *defined* output for a noise-free ROI signal.

    The estimator reports, per volume, the deviation of that volume from the
    GLM's running intercept fit -- not the raw injected signal. For an
    intercept-only nuisance model that fit is the inclusive running mean of the
    data, so the recovered timeseries is an exact positive scaling of
    ``signal - cummean(signal)``. Recovery tests correlate against this.
    """
    arr = np.asarray(signal, dtype=np.float64)
    cummean = np.cumsum(arr) / np.arange(1, len(arr) + 1)
    return arr - cummean


def _run_estimator(data, gt, mask_paths, mask_names, **kwargs) -> FeedbackEstimator:
    """Feed every volume of a phantom through a FeedbackEstimator.

    Defaults to an intercept-only nuisance model so signal-recovery checks are
    not confounded by a drift regressor absorbing slow signal components; tests
    that exercise drift/motion regressors override these explicitly.
    """
    kwargs.setdefault("motion_correction", "none")
    kwargs.setdefault("include_drift", False)
    kwargs.setdefault("include_motion_regressors", False)
    estimator = FeedbackEstimator(
        mask_paths,
        n_baseline=gt.n_baseline,
        n_volumes=gt.n_volumes,
        tr=gt.tr,
        mask_names=mask_names,
        **kwargs,
    )
    for t in range(gt.n_volumes):
        estimator.process_volume("volume.nii", volume_data=data[..., t], volume_affine=gt.affine)
    return estimator


def _single_roi_estimator(tmp_path, **kwargs):
    """A FeedbackEstimator over a one-ROI phantom; returns (estimator, gt)."""
    data, gt = build_phantom(
        PhantomSpec(
            vol_shape=(24, 24, 18),
            n_volumes=100,
            n_baseline=20,
            seed=3,
            background_noise_sd=0.0,
            rois=[ROISpec("roi", noise_sd=0.0)],
        )
    )
    mask_path = tmp_path / "roi_mask.nii.gz"
    write_nifti(gt.roi("roi").mask(gt.vol_shape), gt.affine, mask_path)
    return _run_estimator(data, gt, [mask_path], ["roi"], **kwargs), gt


def test_process_volume_recovers_injected_signal(tmp_path, record):
    """Real-time activation recovers a known injected signal.

    On a noise-free phantom the recovered timeseries is an exact positive scaling
    of ``signal - cummean(signal)`` (the estimator's defined output -- see
    _expected_recovery), so it correlates with that target at 1.0 to machine
    precision. The reference uses np.cumsum; the estimator uses incremental QR.
    """
    estimator, gt = _single_roi_estimator(tmp_path)
    recovered = np.array(estimator.timeseries["roi"])
    nb = gt.n_baseline
    expected = _expected_recovery(gt.roi("roi").signal)
    r = pearson_r(recovered[nb:], expected[nb:])
    err = abs(1.0 - r)
    assert (
        record(
            "feedback",
            "process_volume_recovery_error",
            err,
            threshold=1e-9,
            op="<",
            description=(
                "A known signal is injected into a region of a noise-free phantom. "
                "The real-time estimator should recover it; this is how far the "
                "recovered timecourse departs from a perfect match."
            ),
        )
        < 1e-9
    )


def test_inverse_sd_matches_batch_reference(tmp_path, record):
    """Reliability-weighted region averaging matches a batch reference.

    The phantom uses a fixed-seed RNG and FeedbackEstimator with
    motion_correction="none" is pure numpy -- both deterministic. The reference
    (see ``_reference_inverse_sd_timeseries``) solves the per-voxel, per-volume
    least-squares problem via ``numpy.linalg.lstsq`` (LAPACK ``gelsd``, SVD-based)
    -- a genuinely different numerical algorithm from pineuro's Gentleman's
    incremental QR. Agreement to floating-point precision means a shared bug
    would have to manifest in both LAPACK SVD and Gentleman's QR independently,
    which is essentially impossible. The dedicated inverse-SD weighting math
    check is ``combine_roi_inverse_sd_error == 0`` in test_numeric_core.
    """
    spec = PhantomSpec(
        vol_shape=(24, 24, 18),
        n_volumes=100,
        n_baseline=20,
        seed=11,
        rois=[ROISpec("roi", n_voxels=60, noise_sd=6.0, noisy_fraction=0.5, noisy_sd=100.0)],
    )
    data, gt = build_phantom(spec)
    mask_path = tmp_path / "roi_mask.nii.gz"
    write_nifti(gt.roi("roi").mask(gt.vol_shape), gt.affine, mask_path)
    nb = gt.n_baseline

    est_inv = _run_estimator(data, gt, [mask_path], ["roi"], roi_combine_method="inverse_sd")
    pineuro_inv = np.array(est_inv.timeseries["roi"], dtype=np.float64)
    reference_inv = _reference_inverse_sd_timeseries(data, gt.roi("roi").mask(gt.vol_shape), nb)

    max_err = float(np.max(np.abs(pineuro_inv[nb:] - reference_inv[nb:])))
    assert (
        record(
            "feedback",
            "inverse_sd_vs_batch_reference_max_err",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "pineuro combines a region's voxels one volume at a time, weighting "
                "steadier voxels more. This is the largest gap from an independently "
                "coded batch version run on the same data."
            ),
        )
        < 1e-9
    )


def test_multimask_independent_recovery(tmp_path, record):
    """Two regions with different signals are each recovered independently.

    On a noise-free phantom each mask's recovered timeseries matches that ROI's
    defined estimator output (signal minus its running mean) to machine precision,
    and tracks its own signal far better than the other ROI's.
    """
    n_volumes, n_baseline, tr = 100, 20, 1.0
    signal_a = hrf_convolved_events(n_volumes, tr, target_sd=30.0, seed=1, n_baseline=n_baseline)
    signal_b = hrf_convolved_events(n_volumes, tr, target_sd=30.0, seed=2, n_baseline=n_baseline)
    spec = PhantomSpec(
        vol_shape=(32, 32, 20),
        n_volumes=n_volumes,
        n_baseline=n_baseline,
        tr=tr,
        seed=4,
        background_noise_sd=0.0,
        rois=[
            ROISpec("net_a", center=(10, 16, 10), signal=signal_a, noise_sd=0.0),
            ROISpec("net_b", center=(22, 16, 10), signal=signal_b, noise_sd=0.0),
        ],
    )
    data, gt = build_phantom(spec)
    mask_a = tmp_path / "a.nii.gz"
    mask_b = tmp_path / "b.nii.gz"
    write_nifti(gt.roi("net_a").mask(gt.vol_shape), gt.affine, mask_a)
    write_nifti(gt.roi("net_b").mask(gt.vol_shape), gt.affine, mask_b)

    estimator = _run_estimator(data, gt, [mask_a, mask_b], ["net_a", "net_b"])
    rec_a = np.array(estimator.timeseries["net_a"])[n_baseline:]
    rec_b = np.array(estimator.timeseries["net_b"])[n_baseline:]
    expected_a = _expected_recovery(signal_a)[n_baseline:]
    expected_b = _expected_recovery(signal_b)[n_baseline:]
    own_a = pearson_r(rec_a, expected_a)
    own_b = pearson_r(rec_b, expected_b)
    cross = pearson_r(rec_a, expected_b)
    err = max(abs(1.0 - own_a), abs(1.0 - own_b))
    assert (
        record(
            "feedback",
            "multimask_recovery_error",
            err,
            threshold=1e-9,
            op="<",
            description=(
                "Two regions carry different signals at once. This is the worst "
                "recovery error across the two; near zero means neither region's "
                "signal bleeds into the other."
            ),
        )
        < 1e-9
    )
    assert cross < min(own_a, own_b)  # each mask tracks its own signal, not the other's


def test_drift_regressor_matches_batch_reference(tmp_path, record):
    """The real-time GLM with a drift term matches a batch reference.

    Replaces the previous correlation-based "drift improves recovery" smoke check
    with a deterministic, machine-precision contract: pineuro's incremental
    Gentleman's QR with design ``[intercept, drift=(t+1)]`` must produce
    per-volume combined activations that match an independent
    ``numpy.linalg.lstsq``-based reference to 1e-9. A shared bug would have to
    manifest in both LAPACK SVD and Gentleman's QR independently, which is
    essentially impossible.

    Why this test is exact rather than a recovery-quality floor: the
    *correctness* of pineuro's drift modelling is decided by whether the online
    GLM with drift in the design matches a batch GLM with the same design; the
    *quality* of recovery (i.e. how much drift modelling improves the output) is
    a downstream consequence of that math being right. Testing the GLM-vs-batch
    agreement is the more discriminating check -- a bug in pineuro's design-row
    construction (wrong drift formula, wrong offset, dropped column) breaks the
    agreement, where a correlation-based test would only flag it if the
    recovery quality dropped past a fuzzy threshold.
    """
    n_volumes, n_baseline, tr = 100, 20, 1.0
    clean = hrf_convolved_events(n_volumes, tr, target_sd=30.0, seed=5, n_baseline=n_baseline)
    drift = np.linspace(0.0, 300.0, n_volumes)  # linear, 10x the signal amplitude
    spec = PhantomSpec(
        vol_shape=(24, 24, 18),
        n_volumes=n_volumes,
        n_baseline=n_baseline,
        tr=tr,
        seed=5,
        # noisy_fraction>0 gives the inverse-SD weighting non-trivial work
        # (varying voxel variance) without compromising determinism -- the
        # phantom rng is seeded and pineuro is pure numpy.
        rois=[ROISpec("roi", n_voxels=60, signal=clean + drift, noise_sd=6.0, noisy_fraction=0.5, noisy_sd=100.0)],
    )
    data, gt = build_phantom(spec)
    mask_path = tmp_path / "roi_mask.nii.gz"
    write_nifti(gt.roi("roi").mask(gt.vol_shape), gt.affine, mask_path)

    estimator = _run_estimator(
        data,
        gt,
        [mask_path],
        ["roi"],
        include_drift=True,
        roi_combine_method="inverse_sd",
    )
    pineuro_out = np.array(estimator.timeseries["roi"], dtype=np.float64)
    reference = _reference_with_drift_timeseries(data, gt.roi("roi").mask(gt.vol_shape), n_baseline)

    nb = gt.n_baseline
    max_err = float(np.max(np.abs(pineuro_out[nb:] - reference[nb:])))
    assert (
        record(
            "feedback",
            "drift_glm_vs_batch_reference_max_err",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "With a slow drift term added to the model, the real-time GLM is "
                "compared against a batch fit using the same model. This is the "
                "largest per-volume gap between them."
            ),
        )
        < 1e-9
    )


def test_baseline_freeze_timing(tmp_path):
    """The GLM steady state freezes exactly when n_volumes == n_baseline - 1."""
    data, gt = build_phantom(PhantomSpec(vol_shape=(20, 20, 14), n_volumes=60, n_baseline=20, seed=6))
    mask_path = tmp_path / "roi_mask.nii.gz"
    write_nifti(gt.roi("roi").mask(gt.vol_shape), gt.affine, mask_path)
    estimator = FeedbackEstimator(
        [mask_path],
        n_baseline=gt.n_baseline,
        n_volumes=gt.n_volumes,
        tr=gt.tr,
        mask_names=["roi"],
        motion_correction="none",
    )
    for t in range(gt.n_volumes):
        estimator.process_volume("volume.nii", volume_data=data[..., t], volume_affine=gt.affine)
        expected_frozen = estimator.n_volumes >= gt.n_baseline - 1
        assert estimator._glm.is_frozen() is expected_frozen


def test_motion_json_parsed_correctly(tmp_path):
    """Motion parameters are read back exactly from a dcm2niix JSON sidecar."""
    from .phantom import write_json_sidecar

    data, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=8))
    mask_path = tmp_path / "roi_mask.nii.gz"
    vol_path = tmp_path / "vol_0001.nii.gz"
    write_nifti(gt.roi("roi").mask(gt.vol_shape), gt.affine, mask_path)
    write_nifti(data[..., 0], gt.affine, vol_path)
    estimator = FeedbackEstimator(
        [mask_path],
        n_baseline=gt.n_baseline,
        n_volumes=gt.n_volumes,
        tr=gt.tr,
        mask_names=["roi"],
    )

    motion = np.array([0.12, -0.34, 0.56, -0.01, 0.02, -0.03])
    sidecar = write_json_sidecar(vol_path, gt.tr, motion=motion)
    parsed = estimator._extract_motion_params_from_json(sidecar)
    # motion_image_comment formats with ":.6f"; these 2-decimal inputs round-trip exactly.
    np.testing.assert_array_equal(parsed, motion)


# Hand-calculated FD test vectors. Each (prev, curr, expected_fd_mm) is derived
# *from the definition* — FD = sum of absolute frame-to-frame differences across the
# 6 motion parameters, with rotations converted to mm of arc at HEAD_RADIUS_MM=50 —
# without re-running pineuro's algorithm. A bug in pineuro (wrong head radius, wrong
# sign convention, wrong axis ordering, dropped abs/sum) flips at least one expected
# value below to a different number, and the corresponding case fails.
#
# The factor 50.0 * np.pi / 180.0 ≈ 0.8727 mm/deg is the standard arc-length conversion
# at a 50 mm head radius; written here as an inert expression so the test anchors to
# the definition, not to pineuro's compiled formula.
_DEG_TO_MM = 50.0 * np.pi / 180.0
_FD_CASES = [
    # Pure translation along one axis. FD = |Δ| for that axis, zero elsewhere.
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1.0),
    # Multi-axis translation. FD = sum of per-axis |Δ|.
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, -2.0, 1.0, 0.0, 0.0, 0.0], 6.0),
    # Pure rotation about each axis (degrees → mm arc).
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


def _make_bare_estimator(tmp_path, gt):
    """Return a FeedbackEstimator whose _prev_motion / _prev_voxel_values are unset."""
    mask_path = tmp_path / "roi_mask.nii.gz"
    write_nifti(gt.roi("roi").mask(gt.vol_shape), gt.affine, mask_path)
    return FeedbackEstimator(
        [mask_path],
        n_baseline=gt.n_baseline,
        n_volumes=gt.n_volumes,
        tr=gt.tr,
        mask_names=["roi"],
    )


def test_fd_first_volume_returns_zero(tmp_path):
    """_compute_fd returns 0.0 when there is no previous frame (contract)."""
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=9))
    estimator = _make_bare_estimator(tmp_path, gt)
    assert estimator._prev_motion is None
    assert estimator._compute_fd(np.array([0.25, -0.05, 0.40, -0.01, 0.04, -0.02])) == 0.0


@pytest.mark.parametrize("prev, curr, expected_fd", _FD_CASES)
def test_fd_hand_calculated(tmp_path, record, prev, curr, expected_fd):
    """Head-motion (FD) matches a hand calculation.

    Each expected value is derived from the FD definition (sum of |Δ| with rotations
    at HEAD_RADIUS_MM=50 mm arc length) without running pineuro's algorithm. A bug in
    pineuro -- wrong head radius, wrong rotation→arc factor, wrong sign/abs/sum --
    moves the produced value off the expected literal and the case fails. The 1e-12
    tolerance bounds floating-point ulp noise from the dot-product reduction, not the
    algorithmic correctness (a real bug would deviate by orders of magnitude).
    """
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=9))
    estimator = _make_bare_estimator(tmp_path, gt)
    estimator._prev_motion = np.asarray(prev, dtype=np.float64)
    produced = estimator._compute_fd(np.asarray(curr, dtype=np.float64))
    err = abs(produced - expected_fd)
    case = f"prev{prev}_curr{curr}".replace(" ", "")
    assert (
        record(
            "feedback",
            f"fd_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "Framewise displacement (FD) is how far the head moved between two "
                "volumes. pineuro's FD is compared against values worked out by hand "
                "for each test case."
            ),
        )
        < 1e-12
    )


# Hand-calculated DVARS test vectors. DVARS = sqrt(mean((I_t - I_{t-1})^2)) by
# definition (Power et al. 2012). Each (prev, curr, expected_dvars) below is
# derived from that definition for a small voxel set whose mean-of-squared-diffs
# is trivially known. A bug in pineuro (mean vs sum, dropped square, dropped sqrt,
# wrong reduction axis) flips at least one expected value.
_DVARS_CASES = [
    # All voxels move by the same amount k -> mean(k^2) = k^2 -> sqrt = |k|.
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
    ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 2.0),
    ([100.0, 100.0], [102.0, 102.0], 2.0),
    # No motion -> dvars exactly 0.
    ([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], 0.0),
    # Asymmetric diff [3, 4] -> mean(9, 16) = 12.5 -> sqrt(12.5).
    ([0.0, 0.0], [3.0, 4.0], float(np.sqrt(12.5))),
    # Mixed sign diff: |[-1, 1]| both contribute -> mean(1, 1) = 1 -> sqrt = 1.0.
    ([5.0, 5.0], [4.0, 6.0], 1.0),
    # Single voxel -> dvars == |diff|.
    ([7.0], [10.0], 3.0),
]


def test_dvars_first_volume_returns_zero(tmp_path):
    """_compute_dvars returns 0.0 when there is no previous volume (contract)."""
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=10))
    estimator = _make_bare_estimator(tmp_path, gt)
    assert estimator._prev_voxel_values is None
    assert estimator._compute_dvars(np.array([1.0, 2.0, 3.0])) == 0.0


@pytest.mark.parametrize("prev, curr, expected_dvars", _DVARS_CASES)
def test_dvars_hand_calculated(tmp_path, record, prev, curr, expected_dvars):
    """Image-stability (DVARS) matches a hand calculation.

    Each expected value is derived from the DVARS definition (sqrt of the mean of
    squared frame-to-frame differences) without running pineuro's algorithm. A bug
    in pineuro -- dropped sqrt, sum vs mean, wrong reduction -- moves the produced
    value off the expected literal and the case fails. The 1e-12 tolerance bounds
    one-ulp sqrt rounding; a real algorithmic bug deviates by orders of magnitude.
    """
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=10))
    estimator = _make_bare_estimator(tmp_path, gt)
    estimator._prev_voxel_values = np.asarray(prev, dtype=np.float64)
    produced = estimator._compute_dvars(np.asarray(curr, dtype=np.float64))
    err = abs(produced - expected_dvars)
    case = f"prev{prev}_curr{curr}".replace(" ", "")
    assert (
        record(
            "feedback",
            f"dvars_error_{case}",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "DVARS measures how much the image changed between two volumes. "
                "pineuro's DVARS is compared against values worked out by hand for "
                "each test case."
            ),
        )
        < 1e-12
    )


# =============================================================================
# tSNR (Welford's online algorithm)
# =============================================================================


def test_tsnr_first_volume_returns_zero(tmp_path):
    """_update_tsnr returns 0.0 on the first call (Welford state not initialised)."""
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=11))
    estimator = _make_bare_estimator(tmp_path, gt)
    voxels = np.array([1000.0, 1000.0, 1000.0])
    assert estimator._update_tsnr(voxels) == 0.0


def test_tsnr_constant_timeseries_returns_zero(tmp_path):
    """Voxels with zero variance -> tSNR fallback is 0.0 (no usable std)."""
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=12))
    estimator = _make_bare_estimator(tmp_path, gt)
    voxels = np.array([1000.0, 1000.0, 1000.0])
    for _ in range(5):
        # All updates carry the same values -> sample variance is exactly 0.
        result = estimator._update_tsnr(voxels)
    assert result == 0.0


def test_tsnr_known_mean_and_sd(tmp_path, record):
    """Temporal signal-to-noise ratio matches a hand calculation.

    Two volumes per voxel: 999 and 1001.
    - per-voxel mean = (999 + 1001) / 2 = 1000
    - per-voxel sample variance = ((999-1000)^2 + (1001-1000)^2) / (n-1) = 2 / 1 = 2
    - per-voxel sample SD = sqrt(2)
    - per-voxel tSNR = 1000 / sqrt(2)
    - median across voxels (all identical) = 1000 / sqrt(2).
    """
    _, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=30, n_baseline=10, seed=13))
    estimator = _make_bare_estimator(tmp_path, gt)
    estimator._update_tsnr(np.array([999.0, 999.0, 999.0]))
    produced = estimator._update_tsnr(np.array([1001.0, 1001.0, 1001.0]))
    expected = 1000.0 / float(np.sqrt(2.0))
    err = abs(produced - expected)
    assert (
        record(
            "feedback",
            "tsnr_known_mean_sd_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "Temporal signal-to-noise ratio is a voxel's mean signal divided by "
                "its variability over time. This compares pineuro's tSNR against a "
                "value worked out by hand."
            ),
        )
        < 1e-12
    )
