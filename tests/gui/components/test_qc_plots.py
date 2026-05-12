"""Tests for pineuro.gui.components.qc_plots — pure plotting helpers.

Skips chart-building entirely; only covers the numerical and string-formatting
hooks (standardization, axis range, summary, flag direction). Plot rendering is
exercised by integration runs, not unit tests.
"""


def _populate(plot, values):
    for i, v in enumerate(values, start=1):
        plot.data.add_point(i, v)
    return plot


# =============================================================================
# Flag direction
# =============================================================================


class TestIsFlagged:
    def test_fd_flags_above_threshold(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=200, threshold=0.5)
        assert plot._is_flagged(0.6) is True
        assert plot._is_flagged(0.4) is False
        # Boundary excluded (strict >)
        assert plot._is_flagged(0.5) is False

    def test_tsnr_flags_below_threshold(self):
        from pineuro.gui.components.qc_plots import TSNRPlot

        plot = TSNRPlot(n_volumes=200, threshold=40.0)
        assert plot._is_flagged(35.0) is True
        assert plot._is_flagged(45.0) is False


# =============================================================================
# DVARS standardization
# =============================================================================


class TestDVARSStandardization:
    def test_empty_returns_empty(self):
        from pineuro.gui.components.qc_plots import DVARSPlot

        plot = DVARSPlot(n_volumes=200)
        assert plot._get_standardized_values() == []

    def test_zero_pool_returns_zeros(self):
        from pineuro.gui.components.qc_plots import DVARSPlot

        plot = DVARSPlot(n_volumes=200)
        # Only zeros (pool excludes them) and fewer than 6 → no usable median
        _populate(plot, [0.0, 0.0, 0.0])
        assert plot._get_standardized_values() == [0.0, 0.0, 0.0]

    def test_division_by_sqrt_median_sq(self):
        from pineuro.gui.components.qc_plots import DVARSPlot

        plot = DVARSPlot(n_volumes=200)
        # 5 unused prefix + a stable pool: [2, 2, 4, 4, 4]
        _populate(plot, [99.0, 99.0, 99.0, 99.0, 99.0, 2.0, 2.0, 4.0, 4.0, 4.0])
        # squared pool sorted: [4, 4, 16, 16, 16] → median = 16 → denom = 4
        std = plot._get_standardized_values()
        # Standardize all values (including the prefix) by denom=4
        assert std[5] == 0.5  # 2.0 / 4
        assert std[8] == 1.0  # 4.0 / 4
        assert std[0] == 24.75  # 99.0 / 4

    def test_skip_point_zero_in_dvars(self):
        from pineuro.gui.components.qc_plots import DVARSPlot

        plot = DVARSPlot(n_volumes=200)
        _populate(plot, [1.0, 0.0, 2.0])
        # _skip_point returns True for raw-zero entries
        assert plot._skip_point(0) is False
        assert plot._skip_point(1) is True
        assert plot._skip_point(2) is False


# =============================================================================
# Axis range
# =============================================================================


class TestAxisRange:
    def test_fd_axis_uses_threshold_floor(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=200, threshold=0.5)
        _populate(plot, [0.1, 0.2, 0.1])  # all well below threshold
        _, _, min_y, max_y = plot._calculate_axis_range()
        assert min_y == 0.0
        # max_y has a threshold-based floor (>= threshold * 1.2 = 0.6, and >= threshold + 0.1 = 0.6)
        assert max_y >= 0.6

    def test_fd_axis_scales_to_data(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=200, threshold=0.5)
        _populate(plot, [2.0])  # far above threshold
        _, _, _, max_y = plot._calculate_axis_range()
        assert max_y >= 2.0 * 1.2

    def test_tsnr_axis_minimum_60(self):
        from pineuro.gui.components.qc_plots import TSNRPlot

        plot = TSNRPlot(n_volumes=200, threshold=40.0)
        _populate(plot, [10.0, 15.0, 12.0])  # all small
        _, _, min_y, max_y = plot._calculate_axis_range()
        assert min_y == 0.0
        # Even for small data, max_y has a 60 floor
        assert max_y >= 60.0


# =============================================================================
# Grid interval
# =============================================================================


class TestGridInterval:
    def test_fd_fixed_half_unit(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=200)
        assert plot._get_grid_interval(2.0) == 0.5
        assert plot._get_grid_interval(0.6) == 0.5  # never below 0.5

    def test_dvars_scales_with_range(self):
        from pineuro.gui.components.qc_plots import DVARSPlot

        plot = DVARSPlot(n_volumes=200)
        assert plot._get_grid_interval(2.0) == 0.5  # floor of 0.5
        assert plot._get_grid_interval(8.0) == 2.0  # max_y / 4

    def test_tsnr_scales_with_range(self):
        from pineuro.gui.components.qc_plots import TSNRPlot

        plot = TSNRPlot(n_volumes=200)
        assert plot._get_grid_interval(40.0) == 10.0  # floor
        assert plot._get_grid_interval(100.0) == 25.0  # max_y / 4


# =============================================================================
# Summary formatting
# =============================================================================


class TestSummaryFormatting:
    def test_fd_summary_counts_above_threshold(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=200, threshold=0.5)
        _populate(plot, [0.1, 0.6, 0.4, 0.9, 0.2])  # 2 above
        text, _ = plot._format_summary()
        assert "2 > 0.5mm" in text
        assert "(40.0%)" in text

    def test_fd_initial_summary(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=200, threshold=0.5)
        text, _ = plot._format_initial_summary()
        assert text == "0 > 0.5mm (0%)"

    def test_tsnr_summary_uses_latest(self):
        from pineuro.gui.components.qc_plots import TSNRPlot

        plot = TSNRPlot(n_volumes=200, threshold=40.0)
        _populate(plot, [30.0, 45.0])
        text, _ = plot._format_summary()
        # Uses last value, formatted to 1 decimal
        assert text == "tSNR: 45.0"

    def test_threshold_label_format(self):
        from pineuro.gui.components.qc_plots import DVARSPlot, FDPlot

        assert FDPlot(n_volumes=200, threshold=0.5)._format_threshold_label() == "0.5mm"
        assert DVARSPlot(n_volumes=200, threshold=1.5)._format_threshold_label() == "1.5"


# =============================================================================
# X-axis right-edge rounding (constructor logic)
# =============================================================================


class TestXMaxRoundedToTwentyFives:
    def test_n_volumes_412_rounds_to_425(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=412)
        assert plot._x_max == 425

    def test_n_volumes_exact_multiple_unchanged(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=400)
        assert plot._x_max == 400

    def test_small_n_volumes_rounds_up(self):
        from pineuro.gui.components.qc_plots import FDPlot

        plot = FDPlot(n_volumes=1)
        assert plot._x_max == 25
