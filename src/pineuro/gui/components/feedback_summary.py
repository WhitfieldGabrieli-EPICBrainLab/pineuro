"""Feedback run summary dialog component.

Displays a summary after feedback runs complete, showing:
- Status (completed/interrupted)
- Network hit counts (from PsychoPy log)
- Motion quality statistics (from feedbacklog)
- Activation and FD timeseries plots
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import flet as ft
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSummaryData:
    """Summary data for a completed feedback run."""

    total_volumes: int
    baseline_volumes: int
    feedback_volumes: int
    expected_volumes: int
    hits_per_network: Dict[str, int]  # {network_name: final_hit_count}
    fd_mean: float
    fd_max: float
    fd_above_threshold: int
    fd_pct_above_threshold: float
    fd_threshold: float
    completed: bool


def compute_feedback_summary(
    feedbacklog_path: Path,
    psychopy_log_path: Optional[Path],
    n_baseline: int,
    n_expected_volumes: int,
    fd_threshold: float = 0.5,
) -> FeedbackSummaryData:
    """Compute summary statistics from feedback and PsychoPy logs.

    Args:
        feedbacklog_path: Path to feedbacklog.tsv file
        psychopy_log_path: Path to psychopylog.tsv file (optional, for hit counts)
        n_baseline: Number of baseline volumes
        n_expected_volumes: Expected total number of volumes
        fd_threshold: FD threshold in mm for counting excessive motion

    Returns:
        FeedbackSummaryData with computed statistics
    """
    # Load feedback log for FD statistics
    df = pd.read_csv(feedbacklog_path, sep="\t")
    df["fd"] = pd.to_numeric(df["fd"], errors="coerce")

    total_volumes = len(df)
    baseline_volumes = min(n_baseline, total_volumes)
    feedback_volumes = max(0, total_volumes - n_baseline)

    # FD statistics
    fd_values = df["fd"].values
    fd_mean = float(np.nanmean(fd_values)) if len(fd_values) > 0 else 0.0
    fd_max = float(np.nanmax(fd_values)) if len(fd_values) > 0 else 0.0
    fd_above = int(np.sum(fd_values > fd_threshold))
    fd_pct = (fd_above / total_volumes * 100) if total_volumes > 0 else 0.0

    # Parse hits from PsychoPy log
    hits_per_network: Dict[str, int] = {}
    if psychopy_log_path and psychopy_log_path.exists():
        try:
            psych_df = pd.read_csv(psychopy_log_path, sep="\t")
            # Find columns ending with _hits
            hit_cols = [col for col in psych_df.columns if col.endswith("_hits")]
            for col in hit_cols:
                # Extract network name (e.g., "DefaultA_hits" -> "DefaultA")
                network_name = col.replace("_hits", "")
                # Get max value (cumulative, so last row has final count)
                hits = int(psych_df[col].max())
                hits_per_network[network_name] = hits
        except Exception as e:
            logger.warning(f"Failed to parse PsychoPy log for hits: {e}")

    completed = total_volumes >= n_expected_volumes

    return FeedbackSummaryData(
        total_volumes=total_volumes,
        baseline_volumes=baseline_volumes,
        feedback_volumes=feedback_volumes,
        expected_volumes=n_expected_volumes,
        hits_per_network=hits_per_network,
        fd_mean=fd_mean,
        fd_max=fd_max,
        fd_above_threshold=fd_above,
        fd_pct_above_threshold=fd_pct,
        fd_threshold=fd_threshold,
        completed=completed,
    )


class FeedbackSummaryDialog:
    """Modal dialog showing feedback run summary after completion.

    Displays:
    - Run status (completed/interrupted)
    - Network hit counts
    - Motion quality (FD statistics)
    - Activation and FD timeseries plots
    """

    def __init__(
        self,
        page: ft.Page,
        summary_data: FeedbackSummaryData,
        subject: str,
        task: str,
        run: int,
        activation_plot_path: Optional[Path] = None,
        fd_plot_path: Optional[Path] = None,
        on_close: Optional[Callable[[], None]] = None,
        on_view_full_qc: Optional[Callable[[], None]] = None,
        display_names: Optional[Dict[str, str]] = None,
    ):
        """Initialize the feedback summary dialog.

        Args:
            page: Flet Page object
            summary_data: Computed summary statistics
            subject: Subject ID
            task: Task type (feedback/transfer)
            run: Run number
            activation_plot_path: Path to activation timeseries plot PNG
            fd_plot_path: Path to FD timeseries plot PNG
            on_close: Callback when dialog is closed
            on_view_full_qc: Callback to open full QC report
            display_names: Optional mapping of network names to display names (for blind mode)
        """
        self.page = page
        self.summary_data = summary_data
        self.subject = subject
        self.task = task
        self.run = run
        self.activation_plot_path = activation_plot_path
        self.fd_plot_path = fd_plot_path
        self.on_close = on_close
        self.on_view_full_qc = on_view_full_qc
        self.display_names = display_names or {}
        self._dialog: Optional[ft.AlertDialog] = None

    def show(self):
        """Display the modal dialog."""
        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.ASSESSMENT, color=ft.Colors.BLUE),
                    ft.Text(
                        f"Feedback Summary - {self.subject} ({self.task} run {self.run})",
                        weight=ft.FontWeight.BOLD,
                    ),
                ],
                spacing=10,
            ),
            content=self._build_content(),
            actions=self._build_actions(),
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(self._dialog)
        self._dialog.open = True
        self.page.update()

    def _build_content(self) -> ft.Control:
        """Build the dialog content with summary stats and plots."""
        data = self.summary_data

        # Status indicator
        if data.completed:
            status_icon = ft.Icons.CHECK_CIRCLE
            status_color = ft.Colors.GREEN
            status_text = f"Completed ({data.total_volumes}/{data.expected_volumes} volumes)"
        else:
            status_icon = ft.Icons.WARNING_AMBER
            status_color = ft.Colors.ORANGE
            status_text = f"Interrupted ({data.total_volumes}/{data.expected_volumes} volumes)"

        status_row = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(status_icon, color=status_color, size=24),
                    ft.Text(
                        status_text,
                        size=16,
                        weight=ft.FontWeight.W_500,
                        color=status_color,
                    ),
                ],
                spacing=10,
            ),
            padding=10,
            bgcolor=ft.Colors.GREEN_50 if data.completed else ft.Colors.ORANGE_50,
            border_radius=8,
        )

        # Stats cards row
        stats_row = ft.Row(
            [
                self._build_hits_card(),
                self._build_motion_card(),
            ],
            spacing=15,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        # Build content column
        content_children = [
            status_row,
            ft.Container(height=15),
            stats_row,
        ]

        # Add plots if available
        if self.activation_plot_path and self.activation_plot_path.exists():
            content_children.append(ft.Container(height=15))
            content_children.append(
                ft.Container(
                    content=ft.Image(
                        src=str(self.activation_plot_path),
                        width=700,
                    ),
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=8,
                    padding=5,
                )
            )

        if self.fd_plot_path and self.fd_plot_path.exists():
            content_children.append(ft.Container(height=10))
            content_children.append(
                ft.Container(
                    content=ft.Image(
                        src=str(self.fd_plot_path),
                        width=700,
                    ),
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=8,
                    padding=5,
                )
            )

        return ft.Container(
            content=ft.Column(
                content_children,
                spacing=0,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=750,
            height=600,
            padding=10,
        )

    def _build_hits_card(self) -> ft.Control:
        """Build the network hits summary card."""
        data = self.summary_data

        hit_rows = []
        total_hits = 0
        for network, hits in data.hits_per_network.items():
            total_hits += hits
            # Use display name if available (for blind mode)
            display_name = self.display_names.get(network, network)
            hit_rows.append(
                ft.Row(
                    [
                        ft.Text(f"{display_name}:", size=13, color=ft.Colors.GREY_700),
                        ft.Text(
                            str(hits),
                            size=13,
                            weight=ft.FontWeight.W_500,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                )
            )

        # Add total row
        if data.hits_per_network:
            hit_rows.append(ft.Divider(height=1))
            hit_rows.append(
                ft.Row(
                    [
                        ft.Text("Total:", size=13, weight=ft.FontWeight.W_500),
                        ft.Text(
                            str(total_hits),
                            size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                )
            )
        else:
            hit_rows.append(
                ft.Text(
                    "No hit data available",
                    size=12,
                    color=ft.Colors.GREY_500,
                    italic=True,
                )
            )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.TRACK_CHANGES, color=ft.Colors.BLUE, size=20),
                            ft.Text(
                                "Network Hits",
                                size=14,
                                weight=ft.FontWeight.BOLD,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Container(height=8),
                    *hit_rows,
                ],
                spacing=4,
            ),
            padding=15,
            bgcolor=ft.Colors.GREY_100,
            border_radius=8,
            border=ft.border.all(1, ft.Colors.GREY_300),
            width=200,
        )

    def _build_motion_card(self) -> ft.Control:
        """Build the motion quality summary card."""
        data = self.summary_data

        # Determine motion quality color
        if data.fd_pct_above_threshold < 5:
            quality_color = ft.Colors.GREEN
            quality_text = "Good"
        elif data.fd_pct_above_threshold < 15:
            quality_color = ft.Colors.ORANGE
            quality_text = "Moderate"
        else:
            quality_color = ft.Colors.RED
            quality_text = "High motion"

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.SPEED, color=quality_color, size=20),
                            ft.Text(
                                "Motion Quality",
                                size=14,
                                weight=ft.FontWeight.BOLD,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Container(height=8),
                    ft.Row(
                        [
                            ft.Text("Mean FD:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(
                                f"{data.fd_mean:.2f} mm",
                                size=13,
                                weight=ft.FontWeight.W_500,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Text("Max FD:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(
                                f"{data.fd_max:.2f} mm",
                                size=13,
                                weight=ft.FontWeight.W_500,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Text(
                                f">{data.fd_threshold}mm:",
                                size=13,
                                color=ft.Colors.GREY_700,
                            ),
                            ft.Text(
                                f"{data.fd_above_threshold} vols ({data.fd_pct_above_threshold:.1f}%)",
                                size=13,
                                weight=ft.FontWeight.W_500,
                                color=quality_color,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Divider(height=1),
                    ft.Row(
                        [
                            ft.Text("Quality:", size=13, weight=ft.FontWeight.W_500),
                            ft.Text(
                                quality_text,
                                size=13,
                                weight=ft.FontWeight.BOLD,
                                color=quality_color,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=4,
            ),
            padding=15,
            bgcolor=ft.Colors.GREY_100,
            border_radius=8,
            border=ft.border.all(1, ft.Colors.GREY_300),
            width=200,
        )

    def _build_actions(self) -> list:
        """Build dialog action buttons."""
        actions = []

        if self.on_view_full_qc:
            actions.append(
                ft.TextButton(
                    "View Full QC Report",
                    icon=ft.Icons.OPEN_IN_NEW,
                    on_click=self._on_view_qc_clicked,
                )
            )

        actions.append(
            ft.FilledButton(
                "Close",
                on_click=self._on_close_clicked,
            )
        )

        return actions

    def _on_view_qc_clicked(self, e):
        """Handle View Full QC Report button click."""
        if self.on_view_full_qc:
            self.on_view_full_qc()

    def _on_close_clicked(self, e):
        """Handle close button click."""
        self._close_dialog()
        if self.on_close:
            self.on_close()

    def _close_dialog(self):
        """Close the dialog."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
