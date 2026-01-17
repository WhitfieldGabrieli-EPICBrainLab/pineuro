"""Real-time activation plot component for neurofeedback monitoring.

This module provides a Flet-based line chart for displaying real-time
network activation timeseries during feedback runs.

Requires flet-charts package: pip install flet-charts
"""

import logging
from typing import List, Optional, Dict
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

# Color palette for network activations
NETWORK_COLORS = [
    ft.Colors.BLUE_500,
    ft.Colors.RED_500,
    ft.Colors.GREEN_500,
    ft.Colors.ORANGE_500,
    ft.Colors.PURPLE_500,
    ft.Colors.CYAN_500,
]


@dataclass
class ActivationData:
    """Container for activation timeseries data.

    Attributes:
        volumes: List of volume numbers (x-axis)
        activations: Dict mapping mask name to list of activation values
    """
    volumes: List[int] = field(default_factory=list)
    activations: Dict[str, List[float]] = field(default_factory=dict)

    def add_point(self, volume: int, mask_activations: Dict[str, float]):
        """Add a new data point.

        Args:
            volume: Volume number
            mask_activations: Dict mapping mask name to activation value
        """
        self.volumes.append(volume)
        for mask_name, value in mask_activations.items():
            if mask_name not in self.activations:
                self.activations[mask_name] = []
            self.activations[mask_name].append(value)

    def clear(self):
        """Clear all data."""
        self.volumes.clear()
        self.activations.clear()

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.volumes)


class ActivationPlot:
    """Real-time activation timeseries plot using Flet LineChart.

    Displays network activations over time with:
    - Separate lines for each network (mask)
    - Auto-scaling Y axis
    - Rolling window display (shows last N volumes)
    - Color-coded legend
    """

    def __init__(
        self,
        mask_names: List[str],
        max_display_points: int = 150,
        height: float = 250,
        y_min: float = -2.0,
        y_max: float = 2.0,
    ):
        """Initialize the activation plot.

        Args:
            mask_names: Names of network masks (one line per mask)
            max_display_points: Maximum points to show (rolling window)
            height: Chart height in pixels
            y_min: Minimum Y axis value
            y_max: Maximum Y axis value
        """
        self.mask_names = mask_names
        self.max_display_points = max_display_points
        self.height = height
        self.y_min = y_min
        self.y_max = y_max

        # Data storage
        self.data = ActivationData()

        # Assign colors to masks
        self.mask_colors = {
            name: NETWORK_COLORS[i % len(NETWORK_COLORS)]
            for i, name in enumerate(mask_names)
        }

        # Chart control (created in build())
        self.chart: Optional[LineChart] = None
        self._data_series: List[LineChartData] = []

    def _create_data_series(self) -> List[LineChartData]:
        """Create LineChartData objects for each mask.

        Returns:
            List of LineChartData controls
        """
        if LineChartData is None:
            logger.warning("LineChart not available in this Flet version")
            return []

        series = []
        for mask_name in self.mask_names:
            data_points = []

            # Get activation values for this mask
            if mask_name in self.data.activations:
                activations = self.data.activations[mask_name]
                volumes = self.data.volumes

                # Apply rolling window
                start_idx = max(0, len(volumes) - self.max_display_points)
                for i in range(start_idx, len(volumes)):
                    data_points.append(
                        LineChartDataPoint(
                            x=float(volumes[i]),
                            y=activations[i],
                        )
                    )

            series.append(
                LineChartData(
                    data_points=data_points,
                    color=self.mask_colors[mask_name],
                    stroke_width=2,
                    curved=True,
                    stroke_cap_round=True,
                )
            )

        return series

    def _calculate_axis_range(self) -> tuple[float, float, float, float]:
        """Calculate axis ranges based on current data.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        if not self.data.volumes:
            return (0, self.max_display_points, self.y_min, self.y_max)

        # X axis: show rolling window
        max_vol = max(self.data.volumes)
        min_vol = max(0, max_vol - self.max_display_points)

        # Y axis: use fixed range or auto-scale
        # For now, use fixed range for stability
        return (min_vol, max(max_vol, min_vol + 10), self.y_min, self.y_max)

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
                interval=1,
                color=ft.Colors.with_opacity(0.2, ft.Colors.GREY_500),
                width=1,
            ),
            vertical_grid_lines=ChartGridLines(
                interval=25,
                color=ft.Colors.with_opacity(0.2, ft.Colors.GREY_500),
                width=1,
            ),
            left_axis=ChartAxis(
                labels_size=40,
                title=ft.Text("Activation (normalized)", size=12),
            ),
            bottom_axis=ChartAxis(
                labels_size=32,
                title=ft.Text("Volume", size=12),
            ),
            tooltip_bgcolor=ft.Colors.with_opacity(0.8, ft.Colors.GREY_900),
            expand=True,
        )

        # Create legend
        legend_items = []
        for mask_name in self.mask_names:
            legend_items.append(
                ft.Row(
                    [
                        ft.Container(
                            width=20,
                            height=4,
                            bgcolor=self.mask_colors[mask_name],
                            border_radius=2,
                        ),
                        ft.Text(mask_name, size=12),
                    ],
                    spacing=5,
                )
            )

        legend = ft.Row(
            legend_items,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
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

    def add_data(self, volume: int, mask_activations: Dict[str, float]):
        """Add new activation data and update chart.

        Args:
            volume: Volume number
            mask_activations: Dict mapping mask name to activation value
        """
        self.data.add_point(volume, mask_activations)
        self._update_chart()

    def _update_chart(self):
        """Update chart with current data."""
        if self.chart is None:
            return

        # Update data series
        self.chart.data_series = self._create_data_series()

        # Update axis ranges
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

    def set_y_range(self, y_min: float, y_max: float):
        """Set Y axis range.

        Args:
            y_min: Minimum Y value
            y_max: Maximum Y value
        """
        self.y_min = y_min
        self.y_max = y_max
        self._update_chart()
