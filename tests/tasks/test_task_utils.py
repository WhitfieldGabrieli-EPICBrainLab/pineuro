"""Tests for pineuro.tasks.utils — ActivationSmoother, ActivationNormalizer, BIDS helpers."""

import numpy as np
import pytest

from pineuro.tasks.utils import (
    ActivationNormalizer,
    ActivationSmoother,
    is_activation_outlier,
    make_bids_run_prefix,
    make_bids_subject_prefix,
)


class TestMakeBidsSubjectPrefix:
    def test_combines_subject_and_session(self):
        assert make_bids_subject_prefix("sub-01", "ses-02") == "sub-01_ses-02"


class TestMakeBidsRunPrefix:
    def test_pads_run_and_acq_to_two_digits(self):
        assert make_bids_run_prefix("sub-01", "ses-01", "ball", run=2, acq=3) == "sub-01_ses-01_task-ball_acq-03_run-02"

    def test_run_padding(self):
        assert make_bids_run_prefix("sub-01", "ses-01", "ball", run=1, acq=1).endswith("acq-01_run-01")


class TestIsActivationOutlier:
    def test_scalar_below_threshold(self):
        assert is_activation_outlier(0.5, threshold=1.0) is False

    def test_scalar_above_threshold(self):
        assert is_activation_outlier(2.0, threshold=1.0) is True

    def test_negative_magnitude_counts(self):
        # threshold compares against abs(act)
        assert is_activation_outlier(-2.0, threshold=1.0) is True

    def test_list_any_outlier(self):
        assert is_activation_outlier([0.1, 0.2, 5.0], threshold=1.0) is True

    def test_list_all_within(self):
        assert is_activation_outlier([0.1, -0.5, 0.9], threshold=1.0) is False


class TestActivationSmoother:
    def test_alpha_range_validation(self):
        with pytest.raises(ValueError):
            ActivationSmoother(alpha=-0.1)
        with pytest.raises(ValueError):
            ActivationSmoother(alpha=1.5)

    def test_alpha_one_passes_through(self):
        # alpha=1 → instant update, value equals input
        sm = ActivationSmoother(alpha=1.0)
        assert sm.update(0.7) == pytest.approx(0.7)
        assert sm.update(0.3) == pytest.approx(0.3)

    def test_alpha_zero_never_updates(self):
        sm = ActivationSmoother(alpha=0.0)
        assert sm.update(1.0) == 0.0
        assert sm.update(5.0) == 0.0

    def test_converges_toward_constant_input(self):
        sm = ActivationSmoother(alpha=0.3)
        for _ in range(50):
            sm.update(1.0)
        assert sm.value == pytest.approx(1.0, abs=1e-3)

    def test_reset_restores_zero(self):
        sm = ActivationSmoother(alpha=0.3)
        sm.update(0.8)
        sm.reset()
        assert sm.value == 0.0


class TestActivationNormalizer:
    def test_window_size_validation(self):
        with pytest.raises(ValueError):
            ActivationNormalizer(window_size=1)

    def test_percentile_order_validation(self):
        with pytest.raises(ValueError):
            ActivationNormalizer(lower_percentile=90, upper_percentile=10)

    def test_returns_neutral_during_warmup(self):
        norm = ActivationNormalizer()
        # Fewer than 3 samples → neutral 0.5
        assert norm.update(0.0) == 0.5
        assert norm.update(1.0) == 0.5

    def test_identical_window_returns_neutral(self):
        norm = ActivationNormalizer()
        for _ in range(10):
            assert norm.update(0.5) == 0.5

    def test_normalizes_to_unit_interval(self):
        norm = ActivationNormalizer(window_size=50, lower_percentile=10, upper_percentile=90)
        rng = np.random.default_rng(0)
        outputs = [norm.update(float(v)) for v in rng.standard_normal(200)]
        assert all(0.0 <= o <= 1.0 for o in outputs[3:])

    def test_clips_extremes(self):
        norm = ActivationNormalizer(window_size=20, lower_percentile=25, upper_percentile=75)
        for v in np.linspace(-1.0, 1.0, 20):
            norm.update(float(v))
        # 100.0 is far above the upper percentile of [-1, 1] → clipped to 1.0
        assert norm.update(100.0) == 1.0
        # -100.0 → clipped to 0.0
        assert norm.update(-100.0) == 0.0

    def test_reset_clears_window(self):
        norm = ActivationNormalizer()
        for v in (0.1, 0.2, 0.3):
            norm.update(v)
        norm.reset()
        assert norm.update(0.5) == 0.5  # back to warmup neutral
