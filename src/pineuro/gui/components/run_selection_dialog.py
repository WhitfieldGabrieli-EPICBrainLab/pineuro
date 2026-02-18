"""Run selection dialog for preprocessing step.

Allows users to select which resting-state runs to include
when data is incomplete or partial.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import flet as ft

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Information about a single resting-state run."""

    run_id: int  # 1 or 2
    run_name: str  # "Rest Run 1" or "Rest Run 2"
    directory: Path
    expected_volumes: int
    actual_volumes: int
    is_complete: bool
    selected: bool = True  # Default to selected if has data


class RunSelectionDialog:
    """Modal dialog for selecting resting-state runs for preprocessing.

    Displays available runs with their volume counts and completion status,
    allowing the user to choose which runs to include in preprocessing.
    """

    def __init__(
        self,
        page: ft.Page,
        runs: List[RunInfo],
        on_confirm: Callable[[List[int]], None],
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        """Initialize the run selection dialog.

        Args:
            page: Flet Page object
            runs: List of RunInfo objects for available runs
            on_confirm: Callback with list of selected run IDs (1 and/or 2)
            on_cancel: Optional callback when dialog is cancelled
        """
        self.page = page
        self.runs = runs
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self._dialog: Optional[ft.AlertDialog] = None
        self._checkboxes: dict[int, ft.Checkbox] = {}

    def show(self):
        """Display the modal dialog."""
        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.PLAYLIST_ADD_CHECK, color=ft.Colors.BLUE),
                    ft.Text(
                        "Select Runs for Preprocessing", weight=ft.FontWeight.BOLD
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
        """Build the dialog content with run selection options."""
        run_controls = []

        for run in self.runs:
            # Determine status color and icon
            if run.actual_volumes == 0:
                status_icon = ft.Icons.BLOCK
                status_color = ft.Colors.GREY_400
                status_text = "No data"
                can_select = False
            elif run.is_complete:
                status_icon = ft.Icons.CHECK_CIRCLE
                status_color = ft.Colors.GREEN
                status_text = "Complete"
                can_select = True
            else:
                status_icon = ft.Icons.WARNING_AMBER
                status_color = ft.Colors.ORANGE
                status_text = "Incomplete"
                can_select = True

            # Volume progress indicator
            progress_value = (
                run.actual_volumes / run.expected_volumes
                if run.expected_volumes > 0
                else 0
            )

            checkbox = ft.Checkbox(
                value=run.selected and can_select,
                disabled=not can_select,
                on_change=lambda e, r=run: self._on_checkbox_change(e, r.run_id),
            )
            self._checkboxes[run.run_id] = checkbox

            run_card = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                checkbox,
                                ft.Text(
                                    run.run_name,
                                    size=16,
                                    weight=ft.FontWeight.W_500,
                                ),
                                ft.Container(expand=True),
                                ft.Icon(status_icon, color=status_color, size=20),
                                ft.Text(status_text, color=status_color, size=13),
                            ],
                            alignment=ft.MainAxisAlignment.START,
                        ),
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Row(
                                        [
                                            ft.Text(
                                                "Volumes:",
                                                size=13,
                                                color=ft.Colors.GREY_700,
                                            ),
                                            ft.Text(
                                                f"{run.actual_volumes} / {run.expected_volumes}",
                                                size=13,
                                                weight=ft.FontWeight.W_500,
                                                color=status_color,
                                            ),
                                        ],
                                        spacing=10,
                                    ),
                                    ft.ProgressBar(
                                        value=progress_value,
                                        color=status_color,
                                        bgcolor=ft.Colors.GREY_200,
                                        height=8,
                                    ),
                                ],
                                spacing=5,
                            ),
                            padding=ft.padding.only(left=40),
                        ),
                    ],
                    spacing=5,
                ),
                padding=15,
                bgcolor=ft.Colors.GREY_100 if can_select else ft.Colors.GREY_50,
                border_radius=8,
                border=ft.border.all(1, ft.Colors.GREY_300),
            )

            run_controls.append(run_card)

        # Add info text about incomplete runs
        has_incomplete = any(
            not r.is_complete and r.actual_volumes > 0 for r in self.runs
        )
        info_text = None
        if has_incomplete:
            info_text = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(
                            ft.Icons.INFO_OUTLINE, color=ft.Colors.BLUE_700, size=18
                        ),
                        ft.Text(
                            "Incomplete runs may still be usable but could affect mask quality.",
                            size=12,
                            color=ft.Colors.BLUE_700,
                        ),
                    ],
                    spacing=10,
                ),
                padding=10,
                bgcolor=ft.Colors.BLUE_50,
                border_radius=5,
            )

        content_children = run_controls
        if info_text:
            content_children.append(ft.Container(height=10))
            content_children.append(info_text)

        return ft.Container(
            content=ft.Column(
                content_children,
                spacing=10,
            ),
            width=450,
            padding=10,
        )

    def _build_actions(self) -> List[ft.Control]:
        """Build dialog action buttons."""
        return [
            ft.TextButton(
                "Cancel",
                on_click=self._on_cancel_clicked,
            ),
            ft.FilledButton(
                "Continue",
                icon=ft.Icons.ARROW_FORWARD,
                on_click=self._on_confirm_clicked,
            ),
        ]

    def _on_checkbox_change(self, e, run_id: int):
        """Handle checkbox state change."""
        # Update the run's selected state
        for run in self.runs:
            if run.run_id == run_id:
                run.selected = e.control.value
                break

    def _on_confirm_clicked(self, e):
        """Handle confirm button click."""
        # Get selected run IDs
        selected_runs = [
            run.run_id for run in self.runs if run.selected and run.actual_volumes > 0
        ]

        if not selected_runs:
            # Show error - at least one run must be selected
            _snack = ft.SnackBar(
                ft.Text("Please select at least one run with data."),
                bgcolor=ft.Colors.ORANGE
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        self._close_dialog()
        self.on_confirm(selected_runs)

    def _on_cancel_clicked(self, e):
        """Handle cancel button click."""
        self._close_dialog()
        if self.on_cancel:
            self.on_cancel()

    def _close_dialog(self):
        """Close the dialog."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
