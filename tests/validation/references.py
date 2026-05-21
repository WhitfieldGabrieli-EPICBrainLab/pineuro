"""Independent SVD-based references for the FeedbackEstimator GLM.

These reimplement the FeedbackEstimator's per-volume least-squares output via
``numpy.linalg.lstsq`` (LAPACK ``gelsd``, SVD-based) -- a genuinely different
numerical algorithm from pineuro's Gentleman's incremental QR. A validation test
that asserts pineuro matches one of these to floating-point precision rules out
both an incremental-update bug in the QR path and any error confined to a single
solver: a shared bug would have to manifest in both LAPACK SVD and Gentleman's QR
the same way, which is essentially impossible.

Shared by ``test_feedback_estimator.py`` (direct GLM tests) and
``test_full_pipeline.py`` (end-to-end signal-recovery, computed on the
motion-corrected data the estimator actually consumes).
"""

from __future__ import annotations

import numpy as np


def _reference_inverse_sd_timeseries(data4d: np.ndarray, roi_mask: np.ndarray, n_baseline: int) -> np.ndarray:
    """SVD-based reference for FeedbackEstimator(intercept-only, inverse_sd).

    Per voxel ``v`` at volume ``t`` (0-indexed), with the intercept-only design
    ``X = ones((t+1, 1))``:

    - ``beta[v, t]`` = ``lstsq(X, y[v, :t+1]).beta``  (= sample mean, via SVD).
    - ``error[v, t] = y[v, t] - beta[v, t]``.
    - ``rss[v, t] = sum_{i<=t} (y[v, i] - beta[v, t])**2`` (running, from fit).
    - Freeze: pineuro freezes when ``n_volumes == n_baseline - 1``, i.e. just
      after processing 0-indexed volume ``t_freeze = n_baseline - 2``; snapshot
      is ``rss`` / ``n_obs`` at that point (``n_obs_at_freeze = n_baseline - 1``).
    - Deviation post-freeze: ``sqrt(frozen_rss / n_obs_at_freeze)``.
    - Activation: ``error / deviation``.
    - ROI inverse-SD combine uses *current* (running) rss, matching pineuro's
      ``current_rss = self._glm.get_rss()`` call site: ``w = 1 / sqrt(rss)``.
    - Combined: ``sum_v w[v,t] * activation[v,t] / sum_v w[v,t]``.
    """
    voxels = data4d[roi_mask > 0].astype(np.float64)  # (n_vox, n_vol)
    n_vox, n_vol = voxels.shape

    error = np.zeros((n_vox, n_vol), dtype=np.float64)
    running_rss = np.zeros((n_vox, n_vol), dtype=np.float64)
    # Per-volume SVD-based LS solve over a growing window.
    for t in range(n_vol):
        x = np.ones((t + 1, 1), dtype=np.float64)
        y = voxels[:, : t + 1].T  # (t+1, n_vox)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        intercept = beta[0]  # (n_vox,)
        residuals = y - intercept[None, :]
        running_rss[:, t] = np.sum(residuals**2, axis=0)
        error[:, t] = voxels[:, t] - intercept

    t_freeze = n_baseline - 2
    n_obs_at_freeze = t_freeze + 1
    deviation = np.sqrt(running_rss[:, t_freeze] / n_obs_at_freeze)
    activation = error / deviation[:, None]

    # weight = 0 for the first observation (rss==0) -- it has no information; pineuro
    # likewise returns 0 there because n_obs < n_regressors.
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(running_rss > 0, 1.0 / np.sqrt(running_rss), 0.0)
    weight_sum = weights.sum(axis=0)
    combined = np.zeros_like(weight_sum)
    nz = weight_sum > 0
    combined[nz] = (weights * activation).sum(axis=0)[nz] / weight_sum[nz]
    return combined


def _reference_with_drift_timeseries(data4d: np.ndarray, roi_mask: np.ndarray, n_baseline: int) -> np.ndarray:
    """SVD-based reference for FeedbackEstimator(intercept+drift, inverse_sd).

    Extends ``_reference_inverse_sd_timeseries`` to the two-regressor model
    pineuro builds when ``include_drift=True``: design row at vol_idx t is
    ``[1, t+1]`` (intercept + 1-indexed linear drift), per ``feedback.py:1675``.

    The minimum-norm SVD solution at t < n_regressors - 1 is not meaningful
    (system is underdetermined); callers restrict comparison to the post-baseline
    window where the system is overdetermined.

    Per voxel ``v`` at volume ``t`` (0-indexed):

    - ``X = column_stack([ones(t+1), arange(1, t+2)])``  (intercept + drift).
    - ``beta[v, t] = lstsq(X, y[v, :t+1]).beta``  (SVD-based LS).
    - Fit at t: ``intercept_hat + drift_hat * (t+1)``.
    - ``error[v, t] = y[v, t] - fit[v, t]``.
    - ``rss[v, t] = sum_{i<=t} (y[v, i] - X[i] @ beta[v, t])**2`` (running, from fit).
    - Freeze and activation: identical to the intercept-only reference.
    """
    voxels = data4d[roi_mask > 0].astype(np.float64)  # (n_vox, n_vol)
    n_vox, n_vol = voxels.shape

    error = np.zeros((n_vox, n_vol), dtype=np.float64)
    running_rss = np.zeros((n_vox, n_vol), dtype=np.float64)
    for t in range(n_vol):
        # Design at time t: [intercept, drift=(i+1)] for i in 0..t.
        ones_col = np.ones(t + 1, dtype=np.float64)
        drift_col = np.arange(1, t + 2, dtype=np.float64)
        x = np.column_stack([ones_col, drift_col])
        y = voxels[:, : t + 1].T  # (t+1, n_vox)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)  # (2, n_vox)
        pred_full = x @ beta  # (t+1, n_vox)
        residuals = y - pred_full
        running_rss[:, t] = np.sum(residuals**2, axis=0)
        # Current-volume fit and error
        fit_at_t = beta[0] + beta[1] * (t + 1)  # (n_vox,)
        error[:, t] = voxels[:, t] - fit_at_t

    t_freeze = n_baseline - 2
    n_obs_at_freeze = t_freeze + 1
    deviation = np.sqrt(running_rss[:, t_freeze] / n_obs_at_freeze)
    activation = error / deviation[:, None]

    # ROI inverse-SD combine: w = 1/sqrt(running_rss).
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(running_rss > 0, 1.0 / np.sqrt(running_rss), 0.0)
    weight_sum = weights.sum(axis=0)
    combined = np.zeros_like(weight_sum)
    nz = weight_sum > 0
    combined[nz] = (weights * activation).sum(axis=0)[nz] / weight_sum[nz]
    return combined
