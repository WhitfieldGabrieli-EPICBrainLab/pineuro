"""Real-time activation plot component for neurofeedback monitoring.

This module provides a Flet-based line chart for displaying real-time
network activation timeseries during feedback runs.

Requires flet-charts package: pip install flet-charts
"""

import logging
import math
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
    ChartAxisLabel = fch.ChartAxisLabel
    CHARTS_AVAILABLE = True
except ImportError:
    # Fallback: charts not available
    LineChart = None
    LineChartData = None
    LineChartDataPoint = None
    ChartGridLines = None
    ChartAxis = None
    ChartAxisLabel = None
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

    volumes: list[int] = field(default_factory=list)
    activations: dict[str, list[float]] = field(default_factory=dict)

    def add_point(self, volume: int, mask_activations: dict[str, float]):
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
        mask_names: list[str],
        n_volumes: int,
        height: float = 250,
        y_min: float = -2.0,
        y_max: float = 2.0,
        display_names: list[str] | None = None,
    ):
        """Initialize the activation plot.

        Args:
            mask_names: Names of network masks (one line per mask)
            n_volumes: Total expected number of volumes (fixes x-axis range)
            height: Chart height in pixels
            y_min: Minimum Y axis value
            y_max: Maximum Y axis value
            display_names: Optional display names for legend (for blind mode).
                If None, uses mask_names.
        """
        self.mask_names = mask_names
        self.display_names = display_names if display_names else mask_names
        self.n_volumes = n_volumes
        # Round up to next multiple of 25 so x-axis labels don't overlap
        self._x_max = math.ceil(n_volumes / 25) * 25
        self.height = height
        self.y_min = y_min
        self.y_max = y_max

        # Data storage
        self.data = ActivationData()

        # Assign colors to masks
        self.mask_colors = {name: NETWORK_COLORS[i % len(NETWORK_COLORS)] for i, name in enumerate(mask_names)}

        # Chart control (created in build())
        self.chart: LineChart | None = None
        self._data_series: list[LineChartData] = []

    def _create_data_series(self) -> list[LineChartData]:
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

                # Plot all data points
                for i in range(len(volumes)):
                    data_points.append(
                        LineChartDataPoint(
                            x=float(volumes[i]),
                            y=activations[i],
                        )
                    )

            series.append(
                LineChartData(
                    points=data_points,
                    color=self.mask_colors[mask_name],
                    stroke_width=2,
                    curved=True,
                    rounded_stroke_cap=True,
                )
            )

        return series

    def _calculate_axis_range(self) -> tuple[float, float, float, float]:
        """Calculate axis ranges based on expected volumes and current data.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        # X axis: fixed range from 1 to rounded max
        min_x, max_x = 1, self._x_max

        # Y axis: dynamic based on data
        if self.data.n_points == 0:
            # No data yet, use defaults
            return (min_x, max_x, self.y_min, self.y_max)

        # Find min/max across all activations
        all_values = []
        for values in self.data.activations.values():
            all_values.extend(values)

        if not all_values:
            return (min_x, max_x, self.y_min, self.y_max)

        data_min = min(all_values)
        data_max = max(all_values)

        # Add padding (10% of range on each side)
        data_range = data_max - data_min
        padding = max(data_range * 0.1, 0.1)  # At least 0.1 padding

        # Ensure minimum range for readability
        min_range = 0.5
        if data_range < min_range:
            center = (data_max + data_min) / 2
            data_min = center - min_range / 2
            data_max = center + min_range / 2

        return (min_x, max_x, data_min - padding, data_max + padding)

    def _create_y_axis_labels(self, min_y: float, max_y: float) -> list:
        """Create custom Y axis labels with smaller font size.

        Args:
            min_y: Minimum Y value
            max_y: Maximum Y value

        Returns:
            List of ChartAxisLabel objects
        """
        if ChartAxisLabel is None:
            return []

        labels = []
        # Create labels at regular intervals
        y_range = max_y - min_y
        # Use 5 labels for a nice distribution
        interval = y_range / 4
        for i in range(5):
            value = min_y + i * interval
            labels.append(
                ChartAxisLabel(
                    value=value,
                    label=ft.Text(f"{value:.1f}", size=10),
                )
            )
        return labels

    def _create_x_axis_labels(self, min_x: float, max_x: float) -> list:
        """Create custom X axis labels at round intervals.

        Args:
            min_x: Minimum X value
            max_x: Maximum X value

        Returns:
            List of ChartAxisLabel objects
        """
        if ChartAxisLabel is None:
            return []

        # Pick a nice interval: 25, 50, or 100 depending on range
        x_range = max_x - min_x
        if x_range <= 150:
            interval = 25
        elif x_range <= 300:
            interval = 50
        else:
            interval = 100

        labels = []
        value = 0
        while value <= max_x:
            if value >= min_x:
                labels.append(
                    ChartAxisLabel(
                        value=float(value),
                        label=ft.Text(f"{value}", size=10),
                    )
                )
            value += interval
        return labels

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

        # Create custom axis labels with smaller font
        y_labels = self._create_y_axis_labels(min_y, max_y)
        x_labels = self._create_x_axis_labels(min_x, max_x)

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
                label_size=30,
                labels=y_labels,
                title=ft.Text("Activation (normalized)", size=12),
            ),
            bottom_axis=ChartAxis(
                label_size=24,
                labels=x_labels,
                title=ft.Text("Volume", size=12),
            ),
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_500),
            expand=True,
        )

        # Create legend (use display_names for blind mode)
        legend_items = []
        for i, mask_name in enumerate(self.mask_names):
            display_name = self.display_names[i] if i < len(self.display_names) else mask_name
            legend_items.append(
                ft.Row(
                    [
                        ft.Container(
                            width=20,
                            height=4,
                            bgcolor=self.mask_colors[mask_name],
                            border_radius=2,
                        ),
                        ft.Text(display_name, size=12),
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

    def add_data(self, volume: int, mask_activations: dict[str, float]):
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

        # Update axis ranges and labels
        min_x, max_x, min_y, max_y = self._calculate_axis_range()
        self.chart.min_x = min_x
        self.chart.max_x = max_x
        self.chart.min_y = min_y
        self.chart.max_y = max_y

        # Update axis labels for new range
        if self.chart.left_axis:
            self.chart.left_axis.labels = self._create_y_axis_labels(min_y, max_y)

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
