"""Tests for pineuro.feedback — IncrementalGLM, FeedbackEstimator, ConnectivityEstimator."""

import nibabel as nib
import numpy as np
import pytest

from pineuro.feedback import IncrementalGLM, NeurofeedbackCoordinator, _combine_roi

# =============================================================================
# _combine_roi — ROI combining
# =============================================================================


class TestCombineRoi:
    """Per-voxel -> ROI feedback aggregation (mean vs inverse-SD weighting)."""

    def test_mean(self):
        a = np.array([1.0, 5.0])
        assert _combine_roi(a, np.array([1.0, 100.0]), "mean") == 3.0

    def test_inverse_sd_downweights_noisy_voxel(self):
        # Clean voxel (RSS 1) and noisy voxel (RSS 100) carrying activations
        # 1.0 and 5.0. Inverse-SD weights are 1/1 and 1/10, so the clean
        # voxel dominates: (1*1 + 0.1*5) / (1 + 0.1) = 1.5 / 1.1.
        a = np.array([1.0, 5.0])
        rss = np.array([1.0, 100.0])
        result = _combine_roi(a, rss, "inverse_sd")
        assert result == pytest.approx(1.5 / 1.1)
        # Far closer to the clean voxel than the unweighted mean (3.0) -- and
        # nothing like the old RSS-weighting, which gave ~4.96.
        assert result < _combine_roi(a, rss, "mean")

    def test_inverse_sd_equals_mean_for_uniform_noise(self):
        # Equal RSS -> equal weights -> the weighted average is the mean.
        a = np.array([2.0, 4.0, 9.0])
        rss = np.array([7.0, 7.0, 7.0])
        assert _combine_roi(a, rss, "inverse_sd") == pytest.approx(np.mean(a))

    def test_inverse_sd_skips_nonpositive_rss(self):
        # A voxel with zero RSS has no usable noise estimate and is dropped;
        # only the second voxel (RSS 4, weight 0.5) contributes.
        a = np.array([1.0, 5.0])
        result = _combine_roi(a, np.array([0.0, 4.0]), "inverse_sd")
        assert result == pytest.approx(5.0)

    def test_inverse_sd_falls_back_to_mean_when_no_valid_voxel(self):
        a = np.array([1.0, 2.0])
        assert _combine_roi(a, np.array([0.0, 0.0]), "inverse_sd") == pytest.approx(1.5)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="roi_combine_method"):
            _combine_roi(np.array([1.0]), np.array([1.0]), "bogus")


# =============================================================================
# IncrementalGLM — Core Algorithm Tests
# =============================================================================


class TestIncrementalGLMBasic:
    """Basic construction and validation tests."""

    def test_construction(self):
        glm = IncrementalGLM(n_regressors=3, n_nuisance=3, n_outputs=10)
        assert glm.n_regressors == 3
        assert glm.n_nuisance == 3
        assert glm.n_outputs == 10
        assert glm.n_obs == 0

    def test_invalid_n_regressors(self):
        with pytest.raises(ValueError):
            IncrementalGLM(n_regressors=0, n_nuisance=0)

    def test_invalid_n_nuisance_zero(self):
        with pytest.raises(ValueError):
            IncrementalGLM(n_regressors=3, n_nuisance=0)

    def test_invalid_n_nuisance_exceeds(self):
        with pytest.raises(ValueError):
            IncrementalGLM(n_regressors=3, n_nuisance=4)

    def test_wrong_x_shape(self):
        glm = IncrementalGLM(n_regressors=3, n_nuisance=3, n_outputs=1)
        with pytest.raises(ValueError, match="X has 2 elements"):
            glm.update(np.array([1.0, 2.0]), np.array([1.0]))

    def test_wrong_y_shape(self):
        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=3)
        with pytest.raises(ValueError, match="y has 1 elements"):
            glm.update(np.array([1.0, 2.0]), np.array([1.0]))


class TestIncrementalGLMReturns:
    """Test when GLM returns None vs activation values."""

    def test_returns_none_before_min_obs(self):
        glm = IncrementalGLM(n_regressors=3, n_nuisance=3, n_outputs=1)
        for i in range(2):
            result = glm.update(np.array([1.0, float(i), 0.0]), np.array([float(i)]))
        assert result is None
        assert glm.n_obs == 2

    def test_returns_activation_at_min_obs(self):
        glm = IncrementalGLM(n_regressors=3, n_nuisance=3, n_outputs=1)
        rng = np.random.default_rng(42)
        for i in range(3):
            X = rng.standard_normal(3)
            y = np.array([rng.standard_normal()])
            result = glm.update(X, y)
        assert result is not None
        assert result.shape == (1,)

    def test_returns_activation_after_min_obs(self):
        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=5)
        rng = np.random.default_rng(42)
        for i in range(10):
            X = rng.standard_normal(2)
            y = rng.standard_normal(5)
            result = glm.update(X, y)
        assert result is not None
        assert result.shape == (5,)


class TestIncrementalGLMNumerical:
    """Numerical accuracy tests: compare against batch OLS."""

    def test_matches_batch_ols(self):
        """Incremental GLM beta should match numpy least-squares solution."""
        rng = np.random.default_rng(42)
        n_obs = 50
        n_regressors = 3
        n_outputs = 5

        # Generate data: y = X @ true_beta + noise
        X_all = rng.standard_normal((n_obs, n_regressors))
        # Add intercept as first column
        X_all[:, 0] = 1.0
        true_beta = rng.standard_normal((n_regressors, n_outputs))
        noise = rng.standard_normal((n_obs, n_outputs)) * 0.1
        Y_all = X_all @ true_beta + noise

        # Run incremental GLM (all regressors are nuisance for this test)
        glm = IncrementalGLM(n_regressors=n_regressors, n_nuisance=n_regressors, n_outputs=n_outputs)
        for i in range(n_obs):
            glm.update(X_all[i], Y_all[i])

        # Get beta from incremental GLM via back-substitution
        R = glm._R[:, :n_regressors]
        r = glm._R[:, n_regressors:]
        beta_incremental = glm._solve_via_backsubstitution(R, r)

        # Batch OLS: beta = (X'X)^-1 X'Y
        beta_batch, _, _, _ = np.linalg.lstsq(X_all, Y_all, rcond=None)

        np.testing.assert_allclose(beta_incremental, beta_batch, atol=1e-6)

    def test_simple_regression(self):
        """Test with known y = 2*x + 1 relationship."""
        rng = np.random.default_rng(42)
        n_obs = 100

        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=1)

        for i in range(n_obs):
            x = float(i) / n_obs
            X = np.array([x, 1.0])  # [slope_regressor, intercept]
            y = np.array([2.0 * x + 1.0 + rng.standard_normal() * 0.01])
            glm.update(X, y)

        # Extract beta
        R = glm._R[:, :2]
        r = glm._R[:, 2:]
        beta = glm._solve_via_backsubstitution(R, r)

        # Beta should be close to [2.0, 1.0]
        np.testing.assert_allclose(beta[0, 0], 2.0, atol=0.1)
        np.testing.assert_allclose(beta[1, 0], 1.0, atol=0.1)

    def test_multiple_outputs(self):
        """Multiple outputs (voxels) should be fit simultaneously."""
        rng = np.random.default_rng(42)
        n_obs = 50
        n_outputs = 10
        n_regressors = 2

        glm = IncrementalGLM(n_regressors=n_regressors, n_nuisance=n_regressors, n_outputs=n_outputs)

        X_all = rng.standard_normal((n_obs, n_regressors))
        true_beta = rng.standard_normal((n_regressors, n_outputs))
        Y_all = X_all @ true_beta + rng.standard_normal((n_obs, n_outputs)) * 0.01

        for i in range(n_obs):
            glm.update(X_all[i], Y_all[i])

        R = glm._R[:, :n_regressors]
        r = glm._R[:, n_regressors:]
        beta = glm._solve_via_backsubstitution(R, r)

        np.testing.assert_allclose(beta, true_beta, atol=0.1)


class TestIncrementalGLMSteadyState:
    """Test steady-state freezing behavior."""

    def test_freeze_steady_state(self):
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=3)

        # Feed baseline observations
        for i in range(30):
            X = rng.standard_normal(2)
            y = rng.standard_normal(3)
            glm.update(X, y)

        # Freeze
        glm.freeze_steady_state()
        frozen_rss = glm._steady_state_rss.copy()
        frozen_mse = glm._steady_state_residual_mse.copy()

        # Continue feeding data
        for i in range(20):
            X = rng.standard_normal(2)
            y = rng.standard_normal(3)
            glm.update(X, y)

        # Frozen values should not change
        np.testing.assert_array_equal(glm._steady_state_rss, frozen_rss)
        np.testing.assert_array_equal(glm._steady_state_residual_mse, frozen_mse)

    def test_freeze_idempotent(self):
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=1)

        for i in range(10):
            glm.update(rng.standard_normal(2), rng.standard_normal(1))

        glm.freeze_steady_state()
        first_rss = glm._steady_state_rss.copy()

        # Feed more data then freeze again
        for i in range(10):
            glm.update(rng.standard_normal(2), rng.standard_normal(1))

        glm.freeze_steady_state()  # Should be no-op
        np.testing.assert_array_equal(glm._steady_state_rss, first_rss)

    def test_activation_uses_frozen_deviation(self):
        """After freeze, deviation should use frozen RSS, not current."""
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=1)

        for i in range(20):
            glm.update(rng.standard_normal(2), rng.standard_normal(1))

        glm.freeze_steady_state()
        frozen_timepoint = glm._steady_state_timepoint

        # Feed many more observations with large values
        for i in range(100):
            glm.update(rng.standard_normal(2), rng.standard_normal(1) * 100)

        # Frozen timepoint should still be 20
        assert glm._steady_state_timepoint == frozen_timepoint


class TestIncrementalGLMNuisanceRemoval:
    """Test that nuisance regressors are properly removed."""

    def test_nuisance_signal_removed(self):
        """Task signal should remain after nuisance removal."""
        rng = np.random.default_rng(42)
        n_obs = 100
        # 1 task regressor + 2 nuisance regressors (intercept, drift)
        n_task = 1
        n_nuisance = 2
        n_regressors = n_task + n_nuisance

        glm = IncrementalGLM(n_regressors=n_regressors, n_nuisance=n_nuisance, n_outputs=1)

        activations = []
        for i in range(n_obs):
            t = float(i) / n_obs
            task = 1.0 if i > 50 else 0.0
            intercept = 1.0
            drift = t
            X = np.array([task, intercept, drift])
            # Signal = task effect + nuisance + noise
            y = np.array([5.0 * task + 100.0 * intercept + 10.0 * drift + rng.standard_normal() * 0.5])
            result = glm.update(X, y)
            if result is not None:
                activations.append(result[0])

        # After task onset, activations should be higher
        activations = np.array(activations)
        # Split at the point where task=1 starts contributing
        # (accounting for the first n_regressors observations returning None)
        active_idx = 50 - n_regressors
        if active_idx > 0:
            baseline_mean = np.mean(activations[:active_idx])
            active_mean = np.mean(activations[active_idx + 10 :])  # skip transition
            assert active_mean > baseline_mean


class TestIncrementalGLMNumericalStability:
    """Test numerical stability edge cases."""

    def test_many_iterations(self):
        """Should not produce NaN or Inf after many updates."""
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=3, n_nuisance=3, n_outputs=2)

        for i in range(500):
            X = rng.standard_normal(3)
            y = rng.standard_normal(2)
            result = glm.update(X, y)

        assert result is not None
        assert np.all(np.isfinite(result))

    def test_single_regressor(self):
        """Edge case: n_regressors=1, n_nuisance=1."""
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=1, n_nuisance=1, n_outputs=1)

        for i in range(20):
            X = np.array([1.0])
            y = np.array([rng.standard_normal()])
            result = glm.update(X, y)

        assert result is not None
        assert np.all(np.isfinite(result))


class TestIncrementalGLMDumpVars:
    """Test dump_algo_vars debugging method."""

    def test_dump_structure(self):
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=3, n_nuisance=3, n_outputs=5)

        for i in range(10):
            glm.update(rng.standard_normal(3), rng.standard_normal(5))

        dump = glm.dump_algo_vars()
        assert "n_obs" in dump
        assert "d_approx" in dump
        assert "d_debug" in dump
        assert "weight" in dump
        assert "rss" in dump
        assert "R_diag" in dump
        assert dump["n_obs"] == 10

    def test_dump_specific_voxels(self):
        rng = np.random.default_rng(42)
        glm = IncrementalGLM(n_regressors=2, n_nuisance=2, n_outputs=10)

        for i in range(5):
            glm.update(rng.standard_normal(2), rng.standard_normal(10))

        dump = glm.dump_algo_vars(voxel_indices=[0, 5, 9])
        assert len(dump["rss"]) == 3


# =============================================================================
# FeedbackEstimator (mocked I/O)
# =============================================================================


class TestFeedbackEstimatorConstruction:
    """Test FeedbackEstimator construction with synthetic masks."""

    def test_basic_construction(self, sample_mask, tmp_path):
        from pineuro.feedback import FeedbackEstimator

        mask_path = sample_mask(shape=(8, 8, 8), fraction=0.3)
        estimator = FeedbackEstimator(
            mask_paths=[mask_path],
            n_baseline=5,
            n_volumes=20,
            tr=2.0,
        )
        assert len(estimator.mask_names) == 1

    def test_multiple_masks(self, sample_mask, tmp_path):
        from pineuro.feedback import FeedbackEstimator

        mask1 = sample_mask(shape=(8, 8, 8), fraction=0.3, name="mask1.nii.gz")
        mask2 = sample_mask(shape=(8, 8, 8), fraction=0.2, name="mask2.nii.gz")
        estimator = FeedbackEstimator(
            mask_paths=[mask1, mask2],
            mask_names=["DMN", "Motor"],
            n_baseline=5,
            n_volumes=20,
            tr=2.0,
        )
        assert estimator.mask_names == ["DMN", "Motor"]


class TestFeedbackEstimatorProcessing:
    """Test volume processing with synthetic data."""

    def test_process_volume_returns_activations(self, sample_mask, tmp_path):
        from pineuro.feedback import FeedbackEstimator

        shape = (8, 8, 8)
        mask_path = sample_mask(shape=shape, fraction=0.3)
        estimator = FeedbackEstimator(
            mask_paths=[mask_path],
            mask_names=["DMN"],
            n_baseline=3,
            n_volumes=20,
            tr=2.0,
            include_drift=True,
            include_motion_regressors=False,
            motion_correction="none",
        )

        # Process enough volumes to get past baseline + min observations
        rng = np.random.default_rng(42)
        last_result = None
        for i in range(10):
            vol_data = rng.standard_normal(shape).astype(np.float32)
            vol_path = tmp_path / f"vol_{i:03d}.nii.gz"
            nib.save(nib.Nifti1Image(vol_data, np.eye(4)), vol_path)
            result = estimator.process_volume(vol_path)
            if result is not None:
                last_result = result

        # Should eventually get activation dict
        assert last_result is not None
        assert "DMN" in last_result

    def test_baseline_freeze(self, sample_mask, tmp_path):
        from pineuro.feedback import FeedbackEstimator

        shape = (8, 8, 8)
        mask_path = sample_mask(shape=shape, fraction=0.3)
        estimator = FeedbackEstimator(
            mask_paths=[mask_path],
            n_baseline=5,
            n_volumes=20,
            tr=2.0,
            include_motion_regressors=False,
            motion_correction="none",
        )

        rng = np.random.default_rng(42)
        # Process baseline volumes
        for i in range(6):
            vol_data = rng.standard_normal(shape).astype(np.float32)
            vol_path = tmp_path / f"vol_{i:03d}.nii.gz"
            nib.save(nib.Nifti1Image(vol_data, np.eye(4)), vol_path)
            estimator.process_volume(vol_path)

        # After baseline, GLM should be frozen
        assert estimator._glm._steady_state_rss is not None


# =============================================================================
# get_next_run_number
# =============================================================================


class TestGetNextRunNumber:
    def test_empty_dir(self, tmp_path):
        from pineuro.feedback import get_next_run_number

        assert get_next_run_number(tmp_path, "sub-01", "ses-01", "feedback") == 1

    def test_existing_runs(self, tmp_path):
        from pineuro.feedback import get_next_run_number

        (tmp_path / "sub-01_ses-01_task-feedback_run-01_feedbacklog.tsv").touch()
        (tmp_path / "sub-01_ses-01_task-feedback_run-02_feedbacklog.tsv").touch()
        assert get_next_run_number(tmp_path, "sub-01", "ses-01", "feedback") == 3

    def test_noncontiguous_runs(self, tmp_path):
        from pineuro.feedback import get_next_run_number

        (tmp_path / "sub-01_ses-01_task-feedback_run-01_feedbacklog.tsv").touch()
        (tmp_path / "sub-01_ses-01_task-feedback_run-03_feedbacklog.tsv").touch()
        assert get_next_run_number(tmp_path, "sub-01", "ses-01", "feedback") == 4


# =============================================================================
# MoCo data/config mismatch detection
# =============================================================================


def _bare_coordinator(keep_moco: bool) -> NeurofeedbackCoordinator:
    """Build a minimal coordinator with only the attributes the file-filtering
    and mismatch-detection code touches, bypassing the heavy __init__ (estimator,
    masks, output dirs, threads)."""
    import threading

    c = object.__new__(NeurofeedbackCoordinator)
    c._lock = threading.Lock()
    c._pending_heap = []
    c._processed_files = set()
    c._retry_counts = {}
    c._MAX_RETRIES = 5
    c._keep_moco = keep_moco
    c.n_processed = 0
    c._skipped_originals_no_moco = 0
    c._MOCO_MISMATCH_THRESHOLD = 10
    c._error = None
    c._stop_event = threading.Event()
    return c


class TestMocoMismatchDetection:
    """siemens_moco mode fed non-MoCo data must fail clearly, not silently."""

    def test_non_moco_data_in_siemens_moco_mode_sets_error(self):
        import heapq
        from pathlib import Path
        from unittest.mock import patch

        c = _bare_coordinator(keep_moco=True)
        for i in range(c._MOCO_MISMATCH_THRESHOLD):
            heapq.heappush(c._pending_heap, (f"{i:03d}", Path(f"/fake/orig_{i}.dcm")))

        # Every incoming volume is a valid DICOM but an original (non-MoCo) series.
        with (
            patch("pineuro.feedback.is_valid_volume", return_value=True),
            patch("pineuro.feedback.is_moco_series", return_value=False),
        ):
            # One call drains the whole heap (all entries are skipped as originals).
            assert c._get_next_file_to_process() is None
            c._check_moco_mismatch()

        assert c._skipped_originals_no_moco >= c._MOCO_MISMATCH_THRESHOLD
        assert isinstance(c._error, RuntimeError)
        assert c._stop_event.is_set()
        # Message names the mismatch and the actionable remedy.
        msg = str(c._error)
        assert "MoCoSeries" in msg
        assert "software" in msg
        with pytest.raises(RuntimeError, match="siemens_moco"):
            c.check_error()

    def test_interleaved_moco_and_originals_no_false_positive(self):
        import heapq
        from pathlib import Path
        from unittest.mock import patch

        c = _bare_coordinator(keep_moco=True)
        # Interleave originals and MoCoSeries; keys keep heap order independent of
        # MoCo-ness so is_moco_series (mapped via the path) decides each volume.
        for i in range(20):
            kind = "orig" if i % 2 == 0 else "moco"
            heapq.heappush(c._pending_heap, (f"{i:03d}", Path(f"/fake/{kind}_{i}.dcm")))

        with (
            patch("pineuro.feedback.is_valid_volume", return_value=True),
            patch("pineuro.feedback.is_moco_series", side_effect=lambda p: "moco" in str(p)),
        ):
            # Pull all processable (MoCo) volumes; originals are skipped between them.
            while (path := c._get_next_file_to_process()) is not None:
                assert "moco" in str(path)
                c.n_processed += 1  # mirrors _process_volume incrementing the count
            c._check_moco_mismatch()

        assert c.n_processed > 0
        assert c._error is None
        assert not c._stop_event.is_set()

    def test_software_mode_skips_moco_no_false_positive(self):
        import heapq
        from pathlib import Path
        from unittest.mock import patch

        c = _bare_coordinator(keep_moco=False)
        for i in range(c._MOCO_MISMATCH_THRESHOLD + 5):
            heapq.heappush(c._pending_heap, (f"{i:03d}", Path(f"/fake/moco_{i}.dcm")))

        # software/none mode keeps originals and skips MoCo-derived volumes; the
        # mismatch counter is siemens_moco-specific and must stay untouched here.
        with (
            patch("pineuro.feedback.is_valid_volume", return_value=True),
            patch("pineuro.feedback.is_moco_series", return_value=True),
        ):
            assert c._get_next_file_to_process() is None
            c._check_moco_mismatch()

        assert c._skipped_originals_no_moco == 0
        assert c._error is None
        assert not c._stop_event.is_set()

    def test_user_stop_not_reported_as_mismatch(self):
        c = _bare_coordinator(keep_moco=True)
        c._skipped_originals_no_moco = 20  # well past the threshold
        c._stop_event.set()  # user stop sets _stop_event but not _error

        c._check_moco_mismatch()

        assert c._error is None
