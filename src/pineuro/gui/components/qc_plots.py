"""Real-time QC plot components for neurofeedback monitoring.

This module provides Flet-based line charts for displaying real-time
framewise displacement (FD), DVARS, and tSNR timeseries during feedback runs.

Each plot is a thin subclass of BaseQCPlot that provides metric-specific
configuration (colors, axis labels, threshold direction, summary format)
while the base class handles all chart lifecycle logic.

Requires flet-charts package: pip install flet-charts
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import flet as ft

# flet_charts is a hard dependency (see pyproject.toml [project.dependencies]).
import flet_charts as fch  # type: ignore[import-untyped]

from pineuro.gui import styles as st

LineChart = fch.LineChart
LineChartData = fch.LineChartData
LineChartDataPoint = fch.LineChartDataPoint
ChartGridLines = fch.ChartGridLines
ChartAxis = fch.ChartAxis
ChartCirclePoint = fch.ChartCirclePoint

logger = logging.getLogger(__name__)

# Colors
FD_LINE_COLOR = ft.Colors.ORANGE_500
THRESHOLD_LINE_COLOR = ft.Colors.RED_600
THRESHOLD_FILL_COLOR = ft.Colors.with_opacity(0.15, ft.Colors.RED_600)
DVARS_LINE_COLOR = ft.Colors.CYAN_500
TSNR_LINE_COLOR = ft.Colors.CYAN_500

# Default thresholds
DEFAULT_FD_THRESHOLD = 0.5  # mm
DEFAULT_DVARS_THRESHOLD = 1.5  # Standardized (Nichols 2013, fMRIPrep default)
DEFAULT_TSNR_THRESHOLD = 20.0  # Below this = flagged


# =============================================================================
# Data container
# =============================================================================


@dataclass
class QCData:
    """Container for QC metric timeseries data (FD, DVARS, tSNR)."""

    volumes: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add_point(self, volume: int, value: float):
        self.volumes.append(volume)
        self.values.append(value)

    def clear(self):
        self.volumes.clear()
        self.values.clear()

    @property
    def n_points(self) -> int:
        return len(self.volumes)


# =============================================================================
# Base plot class
# =============================================================================


class BaseQCPlot:
    """Base class for real-time QC timeseries plots.

    Subclasses set class attributes for configuration and override hook
    methods for metric-specific behavior.

    For most metrics, add_data() performs an incremental O(1) append to the
    existing chart series. Subclasses that require all y-values to be
    recalculated on each new point (e.g. DVARSPlot standardization) should
    set ``_incremental_append = False`` to force a full O(n) rebuild instead.
    """

    # --- Subclass configuration ---
    line_color: str = ft.Colors.CYAN_500
    axis_title: str = ""
    axis_title_size: int = 12
    legend_label: str = ""
    flag_above: bool = True  # True = flag above threshold, False = flag below
    _incremental_append: bool = True  # Set False when all y-values change on each update

    def __init__(self, n_volumes: int, height: float, threshold: float):
        self.n_volumes = n_volumes
        # Round up to next multiple of 25 so auto-generated x-axis labels
        # don't overlap with the endpoint (e.g. 412 → 425)
        self._x_max = math.ceil(n_volumes / 25) * 25
        self.height = height
        self.threshold = threshold
        self.data = QCData()
        self.chart: Any = None
        self._summary_text: ft.Text | None = None

    # --- Public API ---

    def build(self) -> ft.Container:
        """Build the chart + legend + summary container."""
        self._data_series = self._create_data_series()
        min_x, max_x, min_y, max_y = self._calculate_axis_range()

        self.chart = LineChart(
            data_series=self._data_series,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            horizontal_grid_lines=ChartGridLines(
                interval=self._get_grid_interval(max_y),
                color=ft.Colors.with_opacity(0.2, ft.Colors.GREY_500),
                width=1,
            ),
            vertical_grid_lines=ChartGridLines(
                interval=25,
                color=ft.Colors.with_opacity(0.2, ft.Colors.GREY_500),
                width=1,
            ),
            left_axis=ChartAxis(
                label_size=40,
                title=ft.Text(self.axis_title, size=self.axis_title_size),
            ),
            bottom_axis=ChartAxis(
                label_size=32,
                title=ft.Text("Volume", size=12),
            ),
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_500),
            expand=True,
        )

        summary_text, summary_color = self._format_initial_summary()
        self._summary_text = ft.Text(summary_text, size=10, color=summary_color)

        legend = ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(width=14, height=3, bgcolor=self.line_color, border_radius=2),
                        ft.Text(self.legend_label, size=10),
                    ],
                    spacing=3,
                ),
                ft.Row(
                    [
                        ft.Container(
                            width=14,
                            height=0,
                            border=ft.Border.only(top=ft.BorderSide(1.5, THRESHOLD_LINE_COLOR)),
                        ),
                        ft.Text(self._format_threshold_label(), size=10),
                    ],
                    spacing=3,
                ),
                self._summary_text,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            spacing=8,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(content=self.chart, height=self.height, expand=True),
                    legend,
                ],
                spacing=5,
            ),
            padding=10,
        )

    def add_data(self, volume: int, value: float):
        """Add a data point and update the chart.

        Uses an incremental O(1) append when possible (``_incremental_append``
        is True and the chart has already been built). Falls back to a full
        O(n) rebuild for subclasses whose y-values change retroactively (e.g.
        DVARSPlot DVARS standardization).
        """
        self.data.add_point(volume, value)
        if self._incremental_append and self.chart is not None and self._data_series:
            self._append_latest_point()
            self._update_axis_and_summary()
        else:
            self._update_chart()

    def clear(self):
        """Clear all data and reset chart."""
        self.data.clear()
        self._update_chart()

    # --- Shared internals ---

    def _append_latest_point(self) -> None:
        """Append only the newest data point to the existing series (O(1))."""
        if not self._data_series:
            return

        i = len(self.data.volumes) - 1
        if i < 0 or self._skip_point(i):
            return

        plot_values = self._get_plot_values()
        if i >= len(plot_values):
            return

        val = plot_values[i]
        point_marker = None
        if ChartCirclePoint is not None and self._is_flagged(val):
            point_marker = ChartCirclePoint(
                color=THRESHOLD_LINE_COLOR,
                radius=4,
                stroke_color=ft.Colors.RED_900,
                stroke_width=1,
            )
        self._data_series[0].points.append(
            LineChartDataPoint(
                x=float(self.data.volumes[i]),
                y=val,
                point=point_marker,
            )
        )

    def _update_axis_and_summary(self) -> None:
        """Recalculate axis range and summary label, then call chart.update()."""
        if self.chart is None:
            return

        min_x, max_x, min_y, max_y = self._calculate_axis_range()
        self.chart.min_x = min_x
        self.chart.max_x = max_x
        self.chart.min_y = min_y
        self.chart.max_y = max_y

        if self.chart.horizontal_grid_lines:
            self.chart.horizontal_grid_lines.interval = self._get_grid_interval(max_y)

        if self._summary_text is not None:
            text, color = self._format_summary()
            self._summary_text.value = text
            self._summary_text.color = color

        try:
            self.chart.update()
            if self._summary_text is not None:
                self._summary_text.update()
        except Exception:
            # Chart update can fail during shutdown; safe to ignore.
            pass

    def _create_data_series(self) -> list:
        min_x, max_x, _, _ = self._calculate_axis_range()
        plot_values = self._get_plot_values()
        series = []

        # Main timeseries line with flagged-point markers
        points = []
        for i in range(len(self.data.volumes)):
            if self._skip_point(i):
                continue
            val = plot_values[i] if i < len(plot_values) else 0.0
            point_marker = None
            if ChartCirclePoint is not None and self._is_flagged(val):
                point_marker = ChartCirclePoint(
                    color=THRESHOLD_LINE_COLOR,
                    radius=4,
                    stroke_color=ft.Colors.RED_900,
                    stroke_width=1,
                )
            points.append(
                LineChartDataPoint(
                    x=float(self.data.volumes[i]),
                    y=val,
                    point=point_marker,
                )
            )

        data_line_kwargs = {
            "points": points,
            "color": self.line_color,
            "stroke_width": 2,
            "curved": False,
            "rounded_stroke_cap": True,
        }
        data_line_kwargs.update(self._get_data_line_kwargs())
        series.append(LineChartData(**data_line_kwargs))

        # Threshold reference line (dashed)
        series.append(
            LineChartData(
                points=[
                    LineChartDataPoint(x=float(min_x), y=self.threshold),
                    LineChartDataPoint(x=float(max_x), y=self.threshold),
                ],
                color=THRESHOLD_LINE_COLOR,
                stroke_width=1.5,
                dash_pattern=[5, 5],
                curved=False,
            )
        )

        return series

    def _update_chart(self):
        """Full O(n) rebuild of data series, then update axis and summary."""
        if self.chart is None:
            return

        self.chart.data_series = self._create_data_series()
        self._update_axis_and_summary()

    def _is_flagged(self, value: float) -> bool:
        if self.flag_above:
            return value > self.threshold
        return value < self.threshold

    # --- Hooks for subclass customization ---

    def _get_plot_values(self) -> list[float]:
        """Values to plot. Override for transformations (e.g. DVARS standardization)."""
        return self.data.values

    def _skip_point(self, index: int) -> bool:
        """Whether to skip plotting a point at this index."""
        return False

    def _get_data_line_kwargs(self) -> dict:
        """Extra kwargs for the main data LineChartData (e.g. FD fill region)."""
        return {}

    def _calculate_axis_range(self) -> tuple[float, float, float, float]:
        raise NotImplementedError

    def _get_grid_interval(self, max_y: float) -> float:
        raise NotImplementedError

    def _format_summary(self) -> tuple[str, str]:
        """Return (text, color) for the summary label."""
        raise NotImplementedError

    def _format_initial_summary(self) -> tuple[str, str]:
        """Return (text, color) for the summary label before any data."""
        raise NotImplementedError

    def _format_threshold_label(self) -> str:
        """Label text for the threshold legend entry."""
        raise NotImplementedError


# =============================================================================
# FD plot
# =============================================================================


class FDPlot(BaseQCPlot):
    """Real-time framewise displacement timeseries plot."""

    line_color = FD_LINE_COLOR
    axis_title = "FD (mm)"
    legend_label = "FD"
    flag_above = True

    def __init__(self, n_volumes: int, height: float = 120, threshold: float = DEFAULT_FD_THRESHOLD):
        super().__init__(n_volumes, height, threshold)

    def _get_data_line_kwargs(self) -> dict:
        return {
            "below_line_bgcolor": THRESHOLD_FILL_COLOR,
            "below_line_cutoff_y": self.threshold,
        }

    def _calculate_axis_range(self):
        data_max = max(self.data.values) if self.data.values else 0.0
        max_y = max(self.threshold * 1.2, data_max * 1.2, self.threshold + 0.1)
        return (1, self._x_max, 0.0, max_y)

    def _get_grid_interval(self, max_y: float) -> float:
        return 0.5

    def _format_threshold_label(self) -> str:
        return f"{self.threshold}mm"

    def _format_summary(self) -> tuple[str, str]:
        n = sum(1 for v in self.data.values if v > self.threshold)
        pct = (n / len(self.data.values) * 100) if self.data.values else 0.0
        text = f"{n} > {self.threshold}mm ({pct:.1f}%)"
        color = THRESHOLD_LINE_COLOR if n > 0 else st.MUTED
        return text, color

    def _format_initial_summary(self) -> tuple[str, str]:
        return f"0 > {self.threshold}mm (0%)", st.MUTED


# =============================================================================
# DVARS plot
# =============================================================================


class DVARSPlot(BaseQCPlot):
    """Real-time standardized DVARS timeseries plot.

    Raw DVARS (Power 2012) is standardized by dividing by sqrt(median(DVARS^2))
    from stable volumes (Nichols 2013). Threshold default is 1.5 (fMRIPrep).

    Requires full rebuild on every update because standardization divides all
    y-values by a median that changes as new data arrives.
    """

    line_color = DVARS_LINE_COLOR
    axis_title = "Std. DVARS"
    axis_title_size = 10
    legend_label = "DVARS"
    flag_above = True
    _incremental_append = False  # Standardization changes all y-values retroactively

    def __init__(self, n_volumes: int, height: float = 100, threshold: float = DEFAULT_DVARS_THRESHOLD):
        super().__init__(n_volumes, height, threshold)

    def _get_plot_values(self) -> list[float]:
        return self._get_standardized_values()

    def _skip_point(self, index: int) -> bool:
        return self.data.values[index] == 0.0

    def _get_standardized_values(self) -> list[float]:
        """Standardize raw DVARS: divide by sqrt(median(DVARS^2)) from stable volumes."""
        vals = self.data.values
        if not vals:
            return []

        # Prefer volumes 6+ (past scanner stabilization) for robust median
        pool = [v for v in vals[5:] if v > 0] if len(vals) > 5 else [v for v in vals if v > 0]
        if not pool:
            return [0.0] * len(vals)

        squared = sorted(v * v for v in pool)
        median_sq = squared[len(squared) // 2]
        if median_sq <= 0:
            return [0.0] * len(vals)

        denom = math.sqrt(median_sq)
        return [v / denom for v in vals]

    def _calculate_axis_range(self):
        std_values = self._get_standardized_values()
        stable = [v for v in std_values[5:] if v > 0] if len(std_values) > 5 else [v for v in std_values if v > 0]
        data_max = max(stable) if stable else 0.0
        max_y = max(self.threshold * 1.5, data_max * 1.2)
        return (1, self._x_max, 0.0, max_y)

    def _get_grid_interval(self, max_y: float) -> float:
        return max(0.5, max_y / 4)

    def _format_threshold_label(self) -> str:
        return f"{self.threshold}"

    def _format_summary(self) -> tuple[str, str]:
        std_values = self._get_standardized_values()
        n = sum(1 for v in std_values if v > self.threshold)
        pct = (n / len(std_values) * 100) if std_values else 0.0
        text = f"{n} > {self.threshold} ({pct:.1f}%)"
        color = THRESHOLD_LINE_COLOR if n > 0 else st.MUTED
        return text, color

    def _format_initial_summary(self) -> tuple[str, str]:
        return f"0 > {self.threshold} (0%)", st.MUTED


# =============================================================================
# tSNR plot
# =============================================================================


class TSNRPlot(BaseQCPlot):
    """Real-time tSNR timeseries plot.

    Displays running tSNR over time. Values below the threshold are flagged.
    """

    line_color = TSNR_LINE_COLOR
    axis_title = "tSNR"
    legend_label = "tSNR"
    flag_above = False  # flags BELOW threshold

    def __init__(self, n_volumes: int, height: float = 80, threshold: float = DEFAULT_TSNR_THRESHOLD):
        super().__init__(n_volumes, height, threshold)

    def _calculate_axis_range(self):
        data_max = max(self.data.values) if self.data.values else 0.0
        max_y = max(60.0, data_max * 1.2)
        return (1, self._x_max, 0.0, max_y)

    def _get_grid_interval(self, max_y: float) -> float:
        return max(10.0, max_y / 4)

    def _format_threshold_label(self) -> str:
        return f"{self.threshold:g}"

    def _format_summary(self) -> tuple[str, str]:
        latest = self.data.values[-1] if self.data.values else 0.0
        text = f"tSNR: {latest:.1f}"
        color = TSNR_LINE_COLOR if latest >= self.threshold else THRESHOLD_LINE_COLOR
        return text, color

    def _format_initial_summary(self) -> tuple[str, str]:
        return "tSNR: --", st.MUTED
