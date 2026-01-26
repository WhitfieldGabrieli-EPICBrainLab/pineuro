"""Real-time framewise displacement plot component for neurofeedback monitoring.

This module provides a Flet-based line chart for displaying real-time
framewise displacement (FD) timeseries during feedback runs.

Requires flet-charts package: pip install flet-charts
"""

import logging
from typing import List, Optional
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
    CHARTS_AVAILABLE = True
except ImportError:
    # Fallback: charts not available
    LineChart = None
    LineChartData = None
    LineChartDataPoint = None
    ChartGridLines = None
    ChartAxis = None
    CHARTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Colors
FD_LINE_COLOR = ft.Colors.ORANGE_500


@dataclass
class FDData:
    """Container for framewise displacement timeseries data.

    Attributes:
        volumes: List of volume numbers (x-axis)
        fd_values: List of FD values in mm (y-axis)
    """
    volumes: List[int] = field(default_factory=list)
    fd_values: List[float] = field(default_factory=list)

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
    ):
        """Initialize the FD plot.

        Args:
            n_volumes: Total expected number of volumes (fixes x-axis range)
            height: Chart height in pixels
        """
        self.n_volumes = n_volumes
        self.height = height

        # Data storage
        self.data = FDData()

        # Y-axis range (dynamic, starts at 0-1)
        self.y_min = 0.0
        self.y_max = 1.0

        # Chart control (created in build())
        self.chart: Optional[LineChart] = None
        self._data_series: List[LineChartData] = []

    def _create_data_series(self) -> List[LineChartData]:
        """Create LineChartData objects for FD line.

        Returns:
            List of LineChartData controls
        """
        if LineChartData is None:
            logger.warning("LineChart not available in this Flet version")
            return []

        series = []

        # FD timeseries line
        fd_points = []
        for i in range(len(self.data.volumes)):
            fd_points.append(
                LineChartDataPoint(
                    x=float(self.data.volumes[i]),
                    y=self.data.fd_values[i],
                )
            )

        series.append(
            LineChartData(
                points=fd_points,
                color=FD_LINE_COLOR,
                stroke_width=2,
                curved=False,  # Keep FD line sharp for quick motion detection
                rounded_stroke_cap=True,
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

        # Y axis: dynamic based on data
        min_y = 0.0
        max_y = max(self.y_max, self.data.get_max_fd() * 1.2)

        # Ensure reasonable minimum y_max
        max_y = max(max_y, 0.5)

        return (min_x, max_x, min_y, max_y)

    def build(self) -> ft.Container:
        """Build the chart control.

        Returns:
            Container with chart and legend
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

        return ft.Container(
            content=ft.Container(
                content=self.chart,
                height=self.height,
                expand=True,
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

        try:
            self.chart.update()
        except Exception as e:
            logger.debug(f"Chart update failed (may be expected during shutdown): {e}")

    def clear(self):
        """Clear all data and reset chart."""
        self.data.clear()
        self._update_chart()
