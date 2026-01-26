"""QC plot viewer modal component for displaying QC images inline.

This module provides a modal dialog that displays key QC plots
directly in the GUI, with an option to open the full HTML report
in an external browser.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import flet as ft

from .qc_viewer import open_qc_report

logger = logging.getLogger(__name__)


class QCPlotViewer:
    """Modal dialog for viewing QC plots inline.

    Displays a scrollable gallery of PNG plots with labels,
    and provides a button to open the full HTML report.

    In approval mode (require_approval=True), shows Approve/Reject buttons
    instead of Close, for mandatory QC review gates.
    """

    def __init__(
        self,
        page: ft.Page,
        title: str,
        plots: List[Tuple[str, Path]],
        html_report_path: Optional[Path] = None,
        require_approval: bool = False,
        on_approve: Optional[Callable[[], None]] = None,
        on_reject: Optional[Callable[[], None]] = None,
    ):
        """Initialize the QC plot viewer.

        Args:
            page: Flet Page object
            title: Dialog title (e.g., "Preprocessing QC")
            plots: List of (label, path) tuples for PNG images
            html_report_path: Optional path to full HTML report
            require_approval: If True, show Approve/Reject buttons instead of Close
            on_approve: Callback when Approve is clicked (approval mode only)
            on_reject: Callback when Reject is clicked (approval mode only)
        """
        self.page = page
        self.title = title
        self.plots = plots
        self.html_report_path = html_report_path
        self.require_approval = require_approval
        self.on_approve = on_approve
        self.on_reject = on_reject
        self._dialog: Optional[ft.AlertDialog] = None

    def show(self):
        """Display the modal dialog."""
        if not self.plots:
            # Show a message if no plots available
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"No QC plots available for {self.title}"),
                open=True,
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.update()
            return

        # Use different styling for approval mode
        if self.require_approval:
            title_icon = ft.Icon(ft.Icons.APPROVAL, color=ft.Colors.ORANGE)
        else:
            title_icon = ft.Icon(ft.Icons.ASSESSMENT, color=ft.Colors.BLUE)

        self._dialog = ft.AlertDialog(
            title=ft.Row(
                [
                    title_icon,
                    ft.Text(self.title, weight=ft.FontWeight.BOLD),
                ],
                spacing=10,
            ),
            content=self._build_content(),
            actions=self._build_actions(),
            actions_alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.page.show_dialog(self._dialog)

    def _build_content(self) -> ft.Control:
        """Build scrollable plot gallery.

        Returns:
            Container with scrollable plot images
        """
        plot_controls = []

        for label, path in self.plots:
            if not path.exists():
                logger.warning(f"Plot file not found: {path}")
                continue

            plot_controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                label,
                                size=14,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.BLUE_800,
                            ),
                            ft.Container(
                                content=ft.Image(
                                    src=str(path),
                                    fit="contain",
                                ),
                                border=ft.border.all(1, ft.Colors.GREY_300),
                                border_radius=5,
                                padding=5,
                            ),
                        ],
                        spacing=5,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=10,
                )
            )

        # Wrap in a scrollable column
        return ft.Container(
            content=ft.Column(
                plot_controls,
                spacing=15,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=800,
            height=600,
        )

    def _build_actions(self) -> List[ft.Control]:
        """Build dialog action buttons.

        Returns:
            List of action button controls
        """
        actions = []

        # View full report button (if HTML path available)
        if self.html_report_path and self.html_report_path.exists():
            actions.append(
                ft.TextButton(
                    "View Full Report",
                    icon=ft.Icons.OPEN_IN_NEW,
                    on_click=self._on_view_full_report,
                )
            )
        else:
            # Add spacer to push buttons to the right
            actions.append(ft.Container())

        if self.require_approval:
            # Approval mode: Reject and Approve buttons
            actions.append(
                ft.TextButton(
                    "Reject",
                    icon=ft.Icons.CANCEL,
                    on_click=self._on_reject_clicked,
                    style=ft.ButtonStyle(color=ft.Colors.RED),
                )
            )
            actions.append(
                ft.FilledButton(
                    "Approve",
                    icon=ft.Icons.CHECK_CIRCLE,
                    on_click=self._on_approve_clicked,
                    bgcolor=ft.Colors.GREEN,
                )
            )
        else:
            # Normal mode: just Close button
            actions.append(
                ft.FilledButton(
                    "Close",
                    on_click=self._on_close,
                )
            )

        return actions

    def _on_view_full_report(self, e):
        """Handle View Full Report button click."""
        if self.html_report_path:
            logger.info(f"Opening full report: {self.html_report_path}")
            open_qc_report(self.html_report_path)

    def _on_close(self, e):
        """Handle Close button click."""
        self.page.pop_dialog()

    def _on_approve_clicked(self, e):
        """Handle Approve button click."""
        logger.info("QC approved by user")
        self.page.pop_dialog()
        if self.on_approve:
            self.on_approve()

    def _on_reject_clicked(self, e):
        """Handle Reject button click."""
        logger.info("QC rejected by user")
        self.page.pop_dialog()
        if self.on_reject:
            self.on_reject()
