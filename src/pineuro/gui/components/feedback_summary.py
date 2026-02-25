"""Feedback run summary dialog component.

Displays a summary after feedback runs complete, showing:
- Status (completed/interrupted)
- Network hit counts (from PsychoPy log)
- FD, DVARS, and tSNR quality cards (from feedbacklog)
- Activation timeseries plot
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import flet as ft
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .base_dialog import BaseDialog

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSummaryData:
    """Summary data for a completed feedback run."""

    total_volumes: int
    baseline_volumes: int
    feedback_volumes: int
    expected_volumes: int
    hits_per_network: dict[str, int]  # {network_name: final_hit_count}
    fd_mean: float
    fd_max: float
    fd_above_threshold: int
    fd_pct_above_threshold: float
    fd_threshold: float
    completed: bool
    dvars_mean: float
    dvars_max: float
    dvars_above_threshold: int
    dvars_pct_above_threshold: float
    dvars_threshold: float
    tsnr_final: float


def compute_feedback_summary(
    feedbacklog_path: Path,
    psychopy_log_path: Path | None,
    n_baseline: int,
    n_expected_volumes: int,
    fd_threshold: float = 0.5,
    dvars_threshold: float = 1.5,
) -> FeedbackSummaryData:
    """Compute summary statistics from feedback and PsychoPy logs.

    Args:
        feedbacklog_path: Path to feedbacklog.tsv file
        psychopy_log_path: Path to psychopylog.tsv file (optional, for hit counts)
        n_baseline: Number of baseline volumes
        n_expected_volumes: Expected total number of volumes
        fd_threshold: FD threshold in mm for counting excessive motion
        dvars_threshold: DVARS threshold (standardized) for counting high-variability volumes

    Returns:
        FeedbackSummaryData with computed statistics
    """
    # Load feedback log
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
    hits_per_network: dict[str, int] = {}
    if psychopy_log_path and psychopy_log_path.exists():
        try:
            psych_df = pd.read_csv(psychopy_log_path, sep="\t")
            # Find columns ending with _hits
            hit_cols = [col for col in psych_df.columns if col.endswith("_hits")]
            for col in hit_cols:
                # Extract network name (e.g., "DefaultA_hits" -> "DefaultA")
                network_name = col.replace("_hits", "")
                # Get max value (cumulative, so last row has final count)
                max_val = psych_df[col].max()
                if pd.isna(max_val):
                    continue
                hits = int(max_val)
                hits_per_network[network_name] = hits
        except Exception as e:
            logger.warning(f"Failed to parse PsychoPy log for hits: {e}")

    completed = total_volumes >= n_expected_volumes

    # DVARS statistics (standardized to match real-time display)
    raw_dvars = pd.to_numeric(df["dvars"], errors="coerce").values
    # Standardize: divide by sqrt(median(DVARS^2)) from stable volumes (6+)
    pool = raw_dvars[5:] if len(raw_dvars) > 5 else raw_dvars
    pool = pool[(pool > 0) & np.isfinite(pool)]
    if len(pool) > 0:
        median_sq = float(np.median(pool**2))
        denom = math.sqrt(median_sq) if median_sq > 0 else 1.0
    else:
        denom = 1.0
    dvars_values = raw_dvars / denom
    dvars_mean = float(np.nanmean(dvars_values)) if len(dvars_values) > 0 else 0.0
    dvars_max = float(np.nanmax(dvars_values)) if len(dvars_values) > 0 else 0.0
    dvars_above = int(np.nansum(dvars_values > dvars_threshold))
    dvars_pct = (dvars_above / total_volumes * 100) if total_volumes > 0 else 0.0

    # tSNR: use last value (most stable estimate after convergence)
    tsnr_values = pd.to_numeric(df["tsnr"], errors="coerce").values
    valid_tsnr = tsnr_values[np.isfinite(tsnr_values) & (tsnr_values > 0)]
    tsnr_final = float(valid_tsnr[-1]) if len(valid_tsnr) > 0 else 0.0

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
        dvars_mean=dvars_mean,
        dvars_max=dvars_max,
        dvars_above_threshold=dvars_above,
        dvars_pct_above_threshold=dvars_pct,
        dvars_threshold=dvars_threshold,
        tsnr_final=tsnr_final,
    )


class FeedbackSummaryDialog(BaseDialog):
    """Modal dialog showing feedback run summary after completion.

    Displays:
    - Run status (completed/interrupted)
    - Network hit counts
    - FD, DVARS, and tSNR quality cards
    - Activation timeseries plot
    """

    def __init__(
        self,
        page: ft.Page,
        summary_data: FeedbackSummaryData,
        subject: str,
        task: str,
        run: int,
        activation_plot_path: Path | None = None,
        connectivity_plot_path: Path | None = None,
        on_close: Callable[[], None] | None = None,
        on_view_full_qc: Callable[[], None] | None = None,
        display_names: dict[str, str] | None = None,
    ):
        """Initialize the feedback summary dialog.

        Args:
            page: Flet Page object
            summary_data: Computed summary statistics
            subject: Subject ID
            task: Task type (feedback/transfer)
            run: Run number
            activation_plot_path: Path to activation timeseries plot PNG
            connectivity_plot_path: Path to connectivity timeseries plot PNG
            on_close: Callback when dialog is closed
            on_view_full_qc: Callback to open full QC report
            display_names: Optional mapping of network names to display names (for blind mode)
        """
        super().__init__(page)
        self.summary_data = summary_data
        self.subject = subject
        self.task = task
        self.run = run
        self.activation_plot_path = activation_plot_path
        self.connectivity_plot_path = connectivity_plot_path
        self.on_close = on_close
        self.on_view_full_qc = on_view_full_qc
        self.display_names = display_names or {}

    def show(self):
        """Display the modal dialog."""
        self._open_dialog(
            ft.AlertDialog(
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
        )

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

        # Stats cards row: Network Hits | FD | DVARS | tSNR
        stats_row = ft.Row(
            [
                self._build_hits_card(),
                self._build_fd_card(),
                self._build_dvars_card(),
                self._build_tsnr_card(),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        # Build content column
        content_children: list[ft.Control] = [
            status_row,
            ft.Container(height=15),
            stats_row,
        ]

        # Timeseries plots
        for plot_path in [self.activation_plot_path, self.connectivity_plot_path]:
            if plot_path and plot_path.exists():
                content_children.append(ft.Container(height=10))
                content_children.append(
                    ft.Container(
                        content=ft.Image(
                            src=str(plot_path),
                            width=750,
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
            width=800,
            height=600,
            padding=10,
        )

    def _build_hits_card(self) -> ft.Control:
        """Build the network hits summary card."""
        data = self.summary_data

        hit_rows: list[ft.Control] = []
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
            width=170,
        )

    def _build_fd_card(self) -> ft.Control:
        """Build the framewise displacement summary card."""
        data = self.summary_data

        # Determine quality color
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
                            ft.Text("FD", size=14, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=8,
                    ),
                    ft.Container(height=8),
                    ft.Row(
                        [
                            ft.Text("Mean:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(f"{data.fd_mean:.2f} mm", size=13, weight=ft.FontWeight.W_500),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Text("Max:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(f"{data.fd_max:.2f} mm", size=13, weight=ft.FontWeight.W_500),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Text(f">{data.fd_threshold}mm:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(
                                f"{data.fd_above_threshold} ({data.fd_pct_above_threshold:.1f}%)",
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
                            ft.Text(quality_text, size=13, weight=ft.FontWeight.BOLD, color=quality_color),
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
            width=170,
        )

    def _build_dvars_card(self) -> ft.Control:
        """Build the DVARS summary card."""
        data = self.summary_data

        # Determine quality color (same thresholds as FD: <5% good, <15% moderate)
        if data.dvars_pct_above_threshold < 5:
            quality_color = ft.Colors.GREEN
            quality_text = "Good"
        elif data.dvars_pct_above_threshold < 15:
            quality_color = ft.Colors.ORANGE
            quality_text = "Moderate"
        else:
            quality_color = ft.Colors.RED
            quality_text = "High variability"

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.SHOW_CHART, color=quality_color, size=20),
                            ft.Text("DVARS", size=14, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=8,
                    ),
                    ft.Container(height=8),
                    ft.Row(
                        [
                            ft.Text("Mean:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(f"{data.dvars_mean:.1f}", size=13, weight=ft.FontWeight.W_500),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Text("Max:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(f"{data.dvars_max:.1f}", size=13, weight=ft.FontWeight.W_500),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Row(
                        [
                            ft.Text(f">{data.dvars_threshold}:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(
                                f"{data.dvars_above_threshold} ({data.dvars_pct_above_threshold:.1f}%)",
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
                            ft.Text(quality_text, size=13, weight=ft.FontWeight.BOLD, color=quality_color),
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
            width=170,
        )

    def _build_tsnr_card(self) -> ft.Control:
        """Build the temporal SNR summary card."""
        data = self.summary_data

        # Determine tSNR quality color
        if data.tsnr_final >= 40:
            tsnr_color = ft.Colors.GREEN
            tsnr_text = "Good"
        elif data.tsnr_final >= 20:
            tsnr_color = ft.Colors.ORANGE
            tsnr_text = "Moderate"
        elif data.tsnr_final > 0:
            tsnr_color = ft.Colors.RED
            tsnr_text = "Low"
        else:
            tsnr_color = ft.Colors.GREY_500
            tsnr_text = "N/A"

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.INSIGHTS, color=tsnr_color, size=20),
                            ft.Text("tSNR", size=14, weight=ft.FontWeight.BOLD),
                        ],
                        spacing=8,
                    ),
                    ft.Container(height=8),
                    ft.Row(
                        [
                            ft.Text("Final:", size=13, color=ft.Colors.GREY_700),
                            ft.Text(
                                f"{data.tsnr_final:.1f}" if data.tsnr_final > 0 else "N/A",
                                size=13,
                                weight=ft.FontWeight.BOLD,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Divider(height=1),
                    ft.Row(
                        [
                            ft.Text("Quality:", size=13, weight=ft.FontWeight.W_500),
                            ft.Text(tsnr_text, size=13, weight=ft.FontWeight.BOLD, color=tsnr_color),
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
            width=170,
        )

    def _build_actions(self) -> list[ft.Control]:
        """Build dialog action buttons."""
        actions: list[ft.Control] = []

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

