"""Feedback run summary dialog component.

Displays a summary after feedback runs complete, showing:
- Status (completed/interrupted)
- Task-declared live metrics (from PsychoPy log; declared on TaskMetadata)
- FD, DVARS, and tSNR quality cards (from feedbacklog)
- Activation timeseries plot
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import flet as ft

from pineuro.gui import styles as st
from pineuro.tasks.task_registry import LiveMetric, read_live_metrics

from .base_dialog import BaseDialog

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSummaryData:
    """Summary data for a completed feedback run."""

    total_volumes: int
    baseline_volumes: int
    feedback_volumes: int
    expected_volumes: int
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
    # {metric.key: {item_name: formatted_value, ...}} from psychopylog last row
    live_metric_snapshots: dict[str, dict[str, str]] = field(default_factory=dict)


def compute_feedback_summary(
    feedbacklog_path: Path,
    psychopy_log_path: Path | None,
    n_baseline: int,
    n_expected_volumes: int,
    fd_threshold: float = 0.5,
    dvars_threshold: float = 1.5,
    live_metrics: list[LiveMetric] | None = None,
) -> FeedbackSummaryData:
    """Compute summary statistics from feedback and PsychoPy logs.

    Args:
        feedbacklog_path: Path to feedbacklog.tsv file
        psychopy_log_path: Path to psychopylog.tsv file (optional, for live metrics)
        n_baseline: Number of baseline volumes
        n_expected_volumes: Expected total number of volumes
        fd_threshold: FD threshold in mm for counting excessive motion
        dvars_threshold: DVARS threshold (standardized) for counting high-variability volumes
        live_metrics: Task-declared live metrics (from TaskMetadata.live_metrics) to read
            from the psychopylog last row. Empty list -> no metric cards shown.

    Returns:
        FeedbackSummaryData with computed statistics
    """
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]

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

    # Read declared live metrics from psychopylog (last row).
    live_metric_snapshots: dict[str, dict[str, str]] = {}
    if psychopy_log_path is not None and live_metrics:
        live_metric_snapshots = read_live_metrics(psychopy_log_path, live_metrics)

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
        live_metric_snapshots=live_metric_snapshots,
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
        live_metrics: list[LiveMetric] | None = None,
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
            live_metrics: Task-declared live metrics (from TaskMetadata.live_metrics).
                One card is rendered per metric; an empty list shows none.
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
        self.live_metrics = list(live_metrics) if live_metrics else []

    def show(self):
        """Display the modal dialog."""
        self._open_dialog(
            ft.AlertDialog(
                modal=True,
                title=ft.Row(
                    [
                        ft.Icon(ft.Icons.ASSESSMENT, color=st.ACCENT),
                        ft.Text(
                            f"Feedback Summary - {self.subject} ({self.task} run {self.run})",
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    spacing=st.GAP_SM,
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
            status_color = st.SUCCESS
            status_text = f"Completed ({data.total_volumes}/{data.expected_volumes} volumes)"
        else:
            status_icon = ft.Icons.WARNING_AMBER
            status_color = st.WARNING
            status_text = f"Interrupted ({data.total_volumes}/{data.expected_volumes} volumes)"

        status_row = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(status_icon, color=status_color, size=st.ICON_LG),
                    ft.Text(
                        status_text,
                        size=st.SECTION,
                        weight=ft.FontWeight.W_500,
                        color=status_color,
                    ),
                ],
                spacing=st.GAP_SM,
            ),
            padding=st.GAP_MD,
            bgcolor=st.SUCCESS_BG if data.completed else st.WARNING_BG,
            border_radius=8,
        )

        # Stats cards row: one combined task-metrics card (when any), then FD | DVARS | tSNR
        metric_card = self._build_metric_card()
        stats_row = ft.Row(
            [
                *([metric_card] if metric_card is not None else []),
                self._build_fd_card(),
                self._build_dvars_card(),
                self._build_tsnr_card(),
            ],
            spacing=st.GAP_MD,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        # Build content column
        content_children: list[ft.Control] = [
            status_row,
            ft.Container(height=st.GAP),
            stats_row,
        ]

        # Timeseries plots
        for plot_path in [self.activation_plot_path, self.connectivity_plot_path]:
            if plot_path and plot_path.exists():
                content_children.append(ft.Container(height=st.GAP_MD))
                content_children.append(
                    ft.Container(
                        content=ft.Image(
                            src=str(plot_path),
                            width=st.PLOT_IMAGE_W,
                        ),
                        border=ft.border.all(1, st.PANEL_BORDER),
                        border_radius=8,
                        padding=st.GAP_XS,
                    )
                )

        return ft.Container(
            content=ft.Column(
                content_children,
                spacing=0,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=st.DIALOG_XL,
            height=600,
            padding=st.GAP_MD,
        )

    def _build_metric_card(self) -> ft.Control | None:
        """Build one combined card with all declared live metrics.

        Returns None when no declared metric has data (so the caller can skip
        rendering an empty card). All metrics share a single "Task Metrics"
        card with one section per metric:

        - Single-cell metric (e.g. gauge "Score"): a single `Label: value` row.
        - Multi-cell metric (e.g. ball "Hits"): a label header, indented
          per-item rows, and an indented "Total" row when every cell parses as
          an int.

        Multiple metrics on the same card are separated by a thin divider.
        """
        rendered_blocks: list[list[ft.Control]] = []
        for metric in self.live_metrics:
            cells = self.summary_data.live_metric_snapshots.get(metric.key, {})
            if not cells:
                continue
            rendered_blocks.append(self._render_metric_block(metric, cells))

        if not rendered_blocks:
            return None

        rows: list[ft.Control] = []
        for i, block in enumerate(rendered_blocks):
            if i > 0:
                rows.append(ft.Divider(height=1))
            rows.extend(block)

        return st.card_shell(ft.Icons.TRACK_CHANGES, "Task Metrics", *rows, width=200)

    def _render_metric_block(self, metric: LiveMetric, cells: dict[str, str]) -> list[ft.Control]:
        """Render a single metric's rows for inclusion in the combined card."""
        block: list[ft.Control] = []
        is_literal_single = len(cells) == 1 and next(iter(cells)) == metric.column_pattern

        if is_literal_single:
            value = next(iter(cells.values()))
            block.append(st.stat_row(f"{metric.label}:", value))
            return block

        # Multi-cell: label header, indented per-item rows, optional total.
        block.append(ft.Text(metric.label, size=st.BODY, weight=ft.FontWeight.W_500))
        int_total = 0
        all_int = True
        for name, value in cells.items():
            disp = self.display_names.get(name, name) if metric.remap_blind_names else name
            block.append(st.stat_row(f"  {disp}:", value))
            try:
                int_total += int(value)
            except ValueError:
                all_int = False

        if all_int and len(cells) > 1:
            block.append(
                st.stat_row(
                    "  Total:",
                    str(int_total),
                    value_color=st.ACCENT,
                    value_weight=ft.FontWeight.BOLD,
                    label_weight=ft.FontWeight.W_500,
                    label_color=None,
                )
            )

        return block

    def _build_fd_card(self) -> ft.Control:
        """Build the framewise displacement summary card."""
        data = self.summary_data

        # Determine quality color
        if data.fd_pct_above_threshold < 5:
            color, verdict = st.SUCCESS, "Good"
        elif data.fd_pct_above_threshold < 15:
            color, verdict = st.WARNING, "Moderate"
        else:
            color, verdict = st.ERROR, "High motion"

        return st.card_shell(
            ft.Icons.SPEED,
            "FD",
            st.stat_row("Mean:", f"{data.fd_mean:.2f} mm"),
            st.stat_row("Max:", f"{data.fd_max:.2f} mm"),
            st.stat_row(
                f">{data.fd_threshold}mm:",
                f"{data.fd_above_threshold} ({data.fd_pct_above_threshold:.1f}%)",
                value_color=color,
            ),
            ft.Divider(height=1),
            st.stat_row(
                "Quality:",
                verdict,
                value_color=color,
                value_weight=ft.FontWeight.BOLD,
                label_weight=ft.FontWeight.W_500,
                label_color=None,
            ),
            icon_color=color,
        )

    def _build_dvars_card(self) -> ft.Control:
        """Build the DVARS summary card."""
        data = self.summary_data

        # Determine quality color (same thresholds as FD: <5% good, <15% moderate)
        if data.dvars_pct_above_threshold < 5:
            color, verdict = st.SUCCESS, "Good"
        elif data.dvars_pct_above_threshold < 15:
            color, verdict = st.WARNING, "Moderate"
        else:
            color, verdict = st.ERROR, "High variability"

        return st.card_shell(
            ft.Icons.SHOW_CHART,
            "DVARS",
            st.stat_row("Mean:", f"{data.dvars_mean:.1f}"),
            st.stat_row("Max:", f"{data.dvars_max:.1f}"),
            st.stat_row(
                f">{data.dvars_threshold}:",
                f"{data.dvars_above_threshold} ({data.dvars_pct_above_threshold:.1f}%)",
                value_color=color,
            ),
            ft.Divider(height=1),
            st.stat_row(
                "Quality:",
                verdict,
                value_color=color,
                value_weight=ft.FontWeight.BOLD,
                label_weight=ft.FontWeight.W_500,
                label_color=None,
            ),
            icon_color=color,
        )

    def _build_tsnr_card(self) -> ft.Control:
        """Build the temporal SNR summary card."""
        data = self.summary_data

        # Determine tSNR quality color
        if data.tsnr_final >= 40:
            color, verdict = st.SUCCESS, "Good"
        elif data.tsnr_final >= 20:
            color, verdict = st.WARNING, "Moderate"
        elif data.tsnr_final > 0:
            color, verdict = st.ERROR, "Low"
        else:
            color, verdict = st.NEUTRAL, "N/A"

        return st.card_shell(
            ft.Icons.INSIGHTS,
            "tSNR",
            st.stat_row(
                "Final:",
                f"{data.tsnr_final:.1f}" if data.tsnr_final > 0 else "N/A",
                value_weight=ft.FontWeight.BOLD,
            ),
            ft.Divider(height=1),
            st.stat_row(
                "Quality:",
                verdict,
                value_color=color,
                value_weight=ft.FontWeight.BOLD,
                label_weight=ft.FontWeight.W_500,
                label_color=None,
            ),
            icon_color=color,
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
