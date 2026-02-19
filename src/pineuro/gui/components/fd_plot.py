"""Real-time QC plot components for neurofeedback monitoring.

This module provides Flet-based line charts for displaying real-time
framewise displacement (FD), DVARS, and tSNR timeseries during feedback runs.

Requires flet-charts package: pip install flet-charts
"""

import logging
from dataclasses import dataclass, field

import flet as ft

# Import chart types from flet_charts package
try:
    import flet_charts as fch

    LineChart = fch.LineChart
    LineChartData = fch.LineChartData
    LineChartDataPoint = fch.LineChartDataPoint
    ChartGridLines = fch.ChartGridLines
    ChartAxis = fch.ChartAxis
    ChartCirclePoint = fch.ChartCirclePoint
    CHARTS_AVAILABLE = True
except ImportError:
    # Fallback: charts not available
    LineChart = None
    LineChartData = None
    LineChartDataPoint = None
    ChartGridLines = None
    ChartAxis = None
    ChartCirclePoint = None
    CHARTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Colors
FD_LINE_COLOR = ft.Colors.ORANGE_500
THRESHOLD_LINE_COLOR = ft.Colors.RED_600
THRESHOLD_FILL_COLOR = ft.Colors.with_opacity(0.15, ft.Colors.RED_600)
DVARS_LINE_COLOR = ft.Colors.CYAN_500
DVARS_THRESHOLD = 1.5  # Standardized DVARS threshold (Nichols 2013, fMRIPrep default)
TSNR_LINE_COLOR = ft.Colors.GREEN_600
TSNR_GOOD_THRESHOLD = 40.0
TSNR_MODERATE_THRESHOLD = 20.0

# Default threshold value
DEFAULT_THRESHOLD = 0.5  # mm


@dataclass
class FDData:
    """Container for framewise displacement timeseries data.

    Attributes:
        volumes: List of volume numbers (x-axis)
        fd_values: List of FD values in mm (y-axis)
    """

    volumes: list[int] = field(default_factory=list)
    fd_values: list[float] = field(default_factory=list)

    def add_point(self, volume: int, fd: float):
        """Add a new data point.

        Args:
            volume: Volume number
            fd: Framewise displacement value in mm
        """
        self.volumes.append(volume)
        self.fd_values.append(fd)

    def clear(self):
        """Clear all data."""
        self.volumes.clear()
        self.fd_values.clear()

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.volumes)

    def get_mean_fd(self) -> float:
        """Get mean FD across all volumes."""
        if not self.fd_values:
            return 0.0
        return sum(self.fd_values) / len(self.fd_values)

    def get_max_fd(self) -> float:
        """Get maximum FD value."""
        if not self.fd_values:
            return 0.0
        return max(self.fd_values)

    def count_above_threshold(self, threshold: float = 0.5) -> int:
        """Count frames with FD above threshold.

        Args:
            threshold: FD threshold in mm

        Returns:
            Number of frames exceeding threshold
        """
        return sum(1 for fd in self.fd_values if fd > threshold)

    def get_pct_above_threshold(self, threshold: float = 0.5) -> float:
        """Get percentage of frames above threshold.

        Args:
            threshold: FD threshold in mm

        Returns:
            Percentage (0-100) of frames exceeding threshold
        """
        if not self.fd_values:
            return 0.0
        return (self.count_above_threshold(threshold) / len(self.fd_values)) * 100


class FDPlot:
    """Real-time framewise displacement timeseries plot using Flet LineChart.

    Displays FD values over time with:
    - Single line for FD timeseries
    - Auto-scaling Y axis based on data
    """

    def __init__(
        self,
        n_volumes: int,
        height: float = 120,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Initialize the FD plot.

        Args:
            n_volumes: Total expected number of volumes (fixes x-axis range)
            height: Chart height in pixels
            threshold: FD threshold in mm for flagging (default: 0.5)
        """
        self.n_volumes = n_volumes
        self.height = height
        self.threshold = threshold

        # Data storage
        self.data = FDData()

        # Y-axis range (dynamic, starts at threshold * 1.2)
        self.y_min = 0.0
        self.y_max = max(1.0, threshold * 1.2)

        # Chart control (created in build())
        self.chart: LineChart | None = None
        self._data_series: list[LineChartData] = []

        # Summary text control (created in build())
        self._summary_text: ft.Text | None = None

    def _create_data_series(self) -> list[LineChartData]:
        """Create LineChartData objects for FD line and threshold reference.

        Returns:
            List of LineChartData controls
        """
        if LineChartData is None:
            logger.warning("LineChart not available in this Flet version")
            return []

        series = []
        min_x, max_x, _, _ = self._calculate_axis_range()

        # 1. FD timeseries line with threshold shading and point markers
        fd_points = []
        for i in range(len(self.data.volumes)):
            fd_value = self.data.fd_values[i]

            # Add point marker for frames above threshold
            point_marker = None
            if ChartCirclePoint is not None and fd_value > self.threshold:
                point_marker = ChartCirclePoint(
                    color=THRESHOLD_LINE_COLOR,
                    radius=4,
                    stroke_color=ft.Colors.RED_900,
                    stroke_width=1,
                )

            fd_points.append(
                LineChartDataPoint(
                    x=float(self.data.volumes[i]),
                    y=fd_value,
                    point=point_marker,
                )
            )

        series.append(
            LineChartData(
                points=fd_points,
                color=FD_LINE_COLOR,
                stroke_width=2,
                curved=False,  # Keep FD line sharp for quick motion detection
                rounded_stroke_cap=True,
                below_line_bgcolor=THRESHOLD_FILL_COLOR,
                below_line_cutoff_y=self.threshold,
            )
        )

        # 2. Threshold reference line (dashed horizontal line)
        threshold_points = [
            LineChartDataPoint(x=float(min_x), y=self.threshold),
            LineChartDataPoint(x=float(max_x), y=self.threshold),
        ]
        series.append(
            LineChartData(
                points=threshold_points,
                color=THRESHOLD_LINE_COLOR,
                stroke_width=1.5,
                dash_pattern=[5, 5],
                curved=False,
            )
        )

        return series

    def _calculate_axis_range(self) -> tuple[float, float, float, float]:
        """Calculate axis ranges based on expected volumes and data.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        # X axis: fixed range from 1 to n_volumes
        min_x = 1
        max_x = self.n_volumes

        # Y axis: dynamic based on data, but always show threshold with padding
        min_y = 0.0
        data_max = self.data.get_max_fd() if self.data.fd_values else 0.0
        max_y = max(
            self.threshold * 1.2,  # At least threshold + 20%
            data_max * 1.2,
        )

        # Ensure reasonable minimum y_max (slightly above threshold)
        max_y = max(max_y, self.threshold + 0.1)

        return (min_x, max_x, min_y, max_y)

    def build(self) -> ft.Container:
        """Build the chart control.

        Returns:
            Container with chart, legend, and summary
        """
        # Check if LineChart is available
        if LineChart is None:
            logger.warning("LineChart not available - showing placeholder")
            return ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Chart not available", size=16, color=ft.Colors.GREY_500),
                        ft.Text("Flet charts module not found", size=12, color=ft.Colors.GREY_400),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=self.height,
                alignment=ft.Alignment(0, 0),  # Center alignment
            )

        # Create initial data series
        self._data_series = self._create_data_series()
        min_x, max_x, min_y, max_y = self._calculate_axis_range()

        # Create chart
        self.chart = LineChart(
            data_series=self._data_series,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            horizontal_grid_lines=ChartGridLines(
                interval=0.5,
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
                title=ft.Text("FD (mm)", size=12),
            ),
            bottom_axis=ChartAxis(
                label_size=32,
                title=ft.Text("Volume", size=12),
            ),
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_500),
            expand=True,
        )

        # Create summary text for threshold exceedances
        self._summary_text = ft.Text(
            f"0 > {self.threshold}mm (0%)",
            size=10,
            color=ft.Colors.GREY_600,
        )

        # Create legend row (compact for narrow column layout)
        legend = ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(width=14, height=3, bgcolor=FD_LINE_COLOR, border_radius=2),
                        ft.Text("FD", size=10),
                    ],
                    spacing=3,
                ),
                ft.Row(
                    [
                        ft.Container(
                            width=14,
                            height=0,
                            border=ft.border.only(top=ft.BorderSide(1.5, THRESHOLD_LINE_COLOR)),
                        ),
                        ft.Text(f"{self.threshold}mm", size=10),
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
                    ft.Container(
                        content=self.chart,
                        height=self.height,
                        expand=True,
                    ),
                    legend,
                ],
                spacing=5,
            ),
            padding=10,
        )

    def add_data(self, volume: int, fd: float):
        """Add new FD data and update chart.

        Args:
            volume: Volume number
            fd: Framewise displacement value in mm
        """
        self.data.add_point(volume, fd)
        self._update_chart()

    def _update_chart(self):
        """Update chart with current data."""
        if self.chart is None:
            return

        # Update data series
        self.chart.data_series = self._create_data_series()

        # Update axis ranges (may change with new max FD)
        min_x, max_x, min_y, max_y = self._calculate_axis_range()
        self.chart.min_x = min_x
        self.chart.max_x = max_x
        self.chart.min_y = min_y
        self.chart.max_y = max_y

        # Update summary text
        if self._summary_text is not None:
            n_above = self.data.count_above_threshold(self.threshold)
            pct_above = self.data.get_pct_above_threshold(self.threshold)
            self._summary_text.value = f"{n_above} > {self.threshold}mm ({pct_above:.1f}%)"

            # Change color to red if there are frames above threshold
            if n_above > 0:
                self._summary_text.color = THRESHOLD_LINE_COLOR
            else:
                self._summary_text.color = ft.Colors.GREY_600

        try:
            self.chart.update()
            if self._summary_text is not None:
                self._summary_text.update()
        except Exception as e:
            logger.debug(f"Chart update failed (may be expected during shutdown): {e}")

    def clear(self):
        """Clear all data and reset chart."""
        self.data.clear()
        self._update_chart()


# =============================================================================
# DVARS Plot
# =============================================================================


@dataclass
class DVARSData:
    """Container for DVARS timeseries data.

    Attributes:
        volumes: List of volume numbers (x-axis)
        dvars_values: List of DVARS values (y-axis)
    """

    volumes: list[int] = field(default_factory=list)
    dvars_values: list[float] = field(default_factory=list)

    def add_point(self, volume: int, dvars: float):
        """Add a new data point."""
        self.volumes.append(volume)
        self.dvars_values.append(dvars)

    def clear(self):
        """Clear all data."""
        self.volumes.clear()
        self.dvars_values.clear()

    @property
    def n_points(self) -> int:
        return len(self.volumes)

    def get_mean_dvars(self) -> float:
        if not self.dvars_values:
            return 0.0
        return sum(self.dvars_values) / len(self.dvars_values)

    def get_max_dvars(self) -> float:
        if not self.dvars_values:
            return 0.0
        return max(self.dvars_values)


class DVARSPlot:
    """Real-time standardized DVARS timeseries plot using Flet LineChart.

    Raw DVARS (Power 2012) is standardized by dividing by sqrt(median(DVARS²))
    from stable volumes (Nichols 2013). This makes the values dimensionless
    with a standard threshold of 1.5 (fMRIPrep default).
    """

    def __init__(self, n_volumes: int, height: float = 100, threshold: float = DVARS_THRESHOLD):
        self.n_volumes = n_volumes
        self.height = height
        self.threshold = threshold
        self.data = DVARSData()
        self.chart: LineChart | None = None
        self._summary_text: ft.Text | None = None

    def _get_standardized_values(self) -> list[float]:
        """Standardize raw DVARS: divide by sqrt(median(DVARS²)) from stable volumes.

        Always standardizes using whatever non-zero data is available.
        Prefers volumes 6+ (past scanner stabilization) for the median estimate.
        Volume 1 (DVARS=0, no previous frame) is excluded from plotting.
        """
        import math

        vals = self.data.dvars_values
        if not vals:
            return []

        # Collect non-zero values for robust median estimate
        # Prefer volumes 6+ (past scanner stabilization) for the median
        if len(vals) > 5:
            pool = [v for v in vals[5:] if v > 0]
        else:
            pool = [v for v in vals if v > 0]

        if not pool:
            return [0.0] * len(vals)

        squared = sorted(v * v for v in pool)
        median_sq = squared[len(squared) // 2]
        if median_sq <= 0:
            return [0.0] * len(vals)

        denom = math.sqrt(median_sq)
        return [v / denom for v in vals]

    def _create_data_series(self) -> list:
        if LineChartData is None:
            return []

        std_values = self._get_standardized_values()
        min_x, max_x, _, _ = self._calculate_axis_range()
        series = []

        # DVARS timeseries — skip volume 1 (DVARS=0 is undefined, no previous frame)
        points = []
        for i in range(len(self.data.volumes)):
            if i < len(std_values) and self.data.dvars_values[i] == 0.0:
                continue  # skip undefined DVARS (first volume)
            point_marker = None
            val = std_values[i] if i < len(std_values) else 0.0
            if ChartCirclePoint is not None and val > self.threshold:
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

        series.append(
            LineChartData(
                points=points,
                color=DVARS_LINE_COLOR,
                stroke_width=2,
                curved=False,
                rounded_stroke_cap=True,
            )
        )

        # Threshold reference line at 1.5
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

    def _calculate_axis_range(self) -> tuple[float, float, float, float]:
        min_x = 1
        max_x = self.n_volumes
        min_y = 0.0
        std_values = self._get_standardized_values()
        # Use stable standardized values (skip first 5 + zeros) for axis scaling
        stable = [v for v in std_values[5:] if v > 0] if len(std_values) > 5 else [v for v in std_values if v > 0]
        data_max = max(stable) if stable else 0.0
        max_y = max(self.threshold * 1.5, data_max * 1.2)
        return (min_x, max_x, min_y, max_y)

    def build(self) -> ft.Container:
        if LineChart is None:
            return ft.Container(
                content=ft.Text("Chart not available", size=16, color=ft.Colors.GREY_500),
                height=self.height,
            )

        self._data_series = self._create_data_series()
        min_x, max_x, min_y, max_y = self._calculate_axis_range()

        self.chart = LineChart(
            data_series=self._data_series,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            horizontal_grid_lines=ChartGridLines(
                interval=max(0.5, max_y / 4),
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
                title=ft.Text("Std. DVARS", size=10),
            ),
            bottom_axis=ChartAxis(
                label_size=32,
                title=ft.Text("Volume", size=12),
            ),
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_500),
            expand=True,
        )

        self._summary_text = ft.Text(
            f"0 > {self.threshold} (0%)",
            size=10,
            color=ft.Colors.GREY_600,
        )

        legend = ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(width=14, height=3, bgcolor=DVARS_LINE_COLOR, border_radius=2),
                        ft.Text("DVARS", size=10),
                    ],
                    spacing=3,
                ),
                ft.Row(
                    [
                        ft.Container(
                            width=14,
                            height=0,
                            border=ft.border.only(top=ft.BorderSide(1.5, THRESHOLD_LINE_COLOR)),
                        ),
                        ft.Text(f"{self.threshold}", size=10),
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

    def add_data(self, volume: int, dvars: float):
        """Add new DVARS data and update chart."""
        self.data.add_point(volume, dvars)
        self._update_chart()

    def _update_chart(self):
        if self.chart is None:
            return

        self.chart.data_series = self._create_data_series()
        min_x, max_x, min_y, max_y = self._calculate_axis_range()
        self.chart.min_x = min_x
        self.chart.max_x = max_x
        self.chart.min_y = min_y
        self.chart.max_y = max_y

        if self.chart.horizontal_grid_lines:
            self.chart.horizontal_grid_lines.interval = max(0.5, max_y / 4)

        if self._summary_text is not None:
            std_values = self._get_standardized_values()
            n_above = sum(1 for v in std_values if v > self.threshold)
            pct = (n_above / len(std_values) * 100) if std_values else 0
            self._summary_text.value = f"{n_above} > {self.threshold} ({pct:.1f}%)"
            self._summary_text.color = THRESHOLD_LINE_COLOR if n_above > 0 else ft.Colors.GREY_600

        try:
            self.chart.update()
            if self._summary_text is not None:
                self._summary_text.update()
        except Exception as e:
            logger.debug(f"DVARS chart update failed: {e}")

    def clear(self):
        """Clear all data and reset chart."""
        self.data.clear()
        self._update_chart()


# =============================================================================
# tSNR Plot
# =============================================================================


@dataclass
class TSNRData:
    """Container for tSNR timeseries data.

    Attributes:
        volumes: List of volume numbers (x-axis)
        tsnr_values: List of running tSNR values (y-axis)
    """

    volumes: list[int] = field(default_factory=list)
    tsnr_values: list[float] = field(default_factory=list)

    def add_point(self, volume: int, tsnr: float):
        """Add a new data point."""
        self.volumes.append(volume)
        self.tsnr_values.append(tsnr)

    def clear(self):
        """Clear all data."""
        self.volumes.clear()
        self.tsnr_values.clear()

    @property
    def n_points(self) -> int:
        return len(self.volumes)

    def get_latest_tsnr(self) -> float:
        if not self.tsnr_values:
            return 0.0
        return self.tsnr_values[-1]

    def get_max_tsnr(self) -> float:
        if not self.tsnr_values:
            return 0.0
        return max(self.tsnr_values)


class TSNRPlot:
    """Real-time tSNR timeseries plot using Flet LineChart.

    Displays running tSNR over time with reference lines at quality
    thresholds (20 = moderate, 40 = good). Shows convergence as the
    Welford estimate stabilizes.
    """

    def __init__(self, n_volumes: int, height: float = 80):
        """Initialize the tSNR plot.

        Args:
            n_volumes: Total expected number of volumes (fixes x-axis range)
            height: Chart height in pixels
        """
        self.n_volumes = n_volumes
        self.height = height
        self.data = TSNRData()
        self.chart: LineChart | None = None
        self._summary_text: ft.Text | None = None

    def _create_data_series(self) -> list:
        if LineChartData is None:
            return []

        min_x, max_x, _, _ = self._calculate_axis_range()
        series = []

        # tSNR timeseries line
        points = [
            LineChartDataPoint(
                x=float(self.data.volumes[i]),
                y=self.data.tsnr_values[i],
            )
            for i in range(len(self.data.volumes))
        ]
        series.append(
            LineChartData(
                points=points,
                color=TSNR_LINE_COLOR,
                stroke_width=2,
                curved=False,
                rounded_stroke_cap=True,
            )
        )

        # Reference line at tSNR = 40 (good quality threshold)
        series.append(
            LineChartData(
                points=[
                    LineChartDataPoint(x=float(min_x), y=TSNR_GOOD_THRESHOLD),
                    LineChartDataPoint(x=float(max_x), y=TSNR_GOOD_THRESHOLD),
                ],
                color=ft.Colors.GREEN_400,
                stroke_width=1,
                dash_pattern=[5, 5],
                curved=False,
            )
        )

        # Reference line at tSNR = 20 (moderate quality threshold)
        series.append(
            LineChartData(
                points=[
                    LineChartDataPoint(x=float(min_x), y=TSNR_MODERATE_THRESHOLD),
                    LineChartDataPoint(x=float(max_x), y=TSNR_MODERATE_THRESHOLD),
                ],
                color=ft.Colors.ORANGE_400,
                stroke_width=1,
                dash_pattern=[5, 5],
                curved=False,
            )
        )

        return series

    def _calculate_axis_range(self) -> tuple[float, float, float, float]:
        min_x = 1
        max_x = self.n_volumes
        min_y = 0.0
        data_max = self.data.get_max_tsnr() if self.data.tsnr_values else 0.0
        max_y = max(60.0, data_max * 1.2)
        return (min_x, max_x, min_y, max_y)

    def build(self) -> ft.Container:
        if LineChart is None:
            return ft.Container(
                content=ft.Text("Chart not available", size=16, color=ft.Colors.GREY_500),
                height=self.height,
            )

        self._data_series = self._create_data_series()
        min_x, max_x, min_y, max_y = self._calculate_axis_range()

        self.chart = LineChart(
            data_series=self._data_series,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            horizontal_grid_lines=ChartGridLines(
                interval=max(10.0, max_y / 4),
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
                title=ft.Text("tSNR", size=12),
            ),
            bottom_axis=ChartAxis(
                label_size=32,
                title=ft.Text("Volume", size=12),
            ),
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_500),
            expand=True,
        )

        self._summary_text = ft.Text(
            "tSNR: --",
            size=10,
            color=ft.Colors.GREY_600,
        )

        legend = ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(width=14, height=3, bgcolor=TSNR_LINE_COLOR, border_radius=2),
                        ft.Text("tSNR", size=10),
                    ],
                    spacing=3,
                ),
                ft.Row(
                    [
                        ft.Container(
                            width=14,
                            height=0,
                            border=ft.border.only(top=ft.BorderSide(1, ft.Colors.GREEN_400)),
                        ),
                        ft.Text("40", size=10, color=ft.Colors.GREEN_600),
                    ],
                    spacing=3,
                ),
                ft.Row(
                    [
                        ft.Container(
                            width=14,
                            height=0,
                            border=ft.border.only(top=ft.BorderSide(1, ft.Colors.ORANGE_400)),
                        ),
                        ft.Text("20", size=10, color=ft.Colors.ORANGE_600),
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

    def add_data(self, volume: int, tsnr: float):
        """Add new tSNR data and update chart."""
        self.data.add_point(volume, tsnr)
        self._update_chart()

    def _update_chart(self):
        if self.chart is None:
            return

        self.chart.data_series = self._create_data_series()
        min_x, max_x, min_y, max_y = self._calculate_axis_range()
        self.chart.min_x = min_x
        self.chart.max_x = max_x
        self.chart.min_y = min_y
        self.chart.max_y = max_y

        if self.chart.horizontal_grid_lines:
            self.chart.horizontal_grid_lines.interval = max(10.0, max_y / 4)

        if self._summary_text is not None:
            latest = self.data.get_latest_tsnr()
            self._summary_text.value = f"tSNR: {latest:.1f}"
            if latest >= TSNR_GOOD_THRESHOLD:
                self._summary_text.color = ft.Colors.GREEN_600
            elif latest >= TSNR_MODERATE_THRESHOLD:
                self._summary_text.color = ft.Colors.ORANGE_600
            else:
                self._summary_text.color = ft.Colors.RED_600

        try:
            self.chart.update()
            if self._summary_text is not None:
                self._summary_text.update()
        except Exception as e:
            logger.debug(f"tSNR chart update failed: {e}")

    def clear(self):
        """Clear all data and reset chart."""
        self.data.clear()
        self._update_chart()
