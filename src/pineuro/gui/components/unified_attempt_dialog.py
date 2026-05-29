"""Unified attempt selection dialog for preprocessing step.

Replaces the separate run-selection and attempt-selection dialogs with a single
dialog that lists all attempts grouped by run, each with a checkbox.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import flet as ft

from .base_dialog import BaseDialog

logger = logging.getLogger(__name__)


@dataclass
class AttemptInfo:
    """Information about a single attempt."""

    attempt_num: int
    directory: Path
    n_volumes: int
    expected_volumes: int
    is_complete: bool


@dataclass
class RunGroup:
    """All attempts for one run, for display grouping."""

    run_id: int
    run_name: str
    attempts: list[AttemptInfo] = field(default_factory=list)


def scan_collection_attempts(
    collection_dir: Path,
    task: str,
    run: int,
    expected_volumes: int,
) -> list[AttemptInfo]:
    """Scan collection directory for attempts for a given run.

    Parameters
    ----------
    collection_dir : Path
        Collection directory to scan
    task : str
        Task prefix (e.g., 'task-rest', 'task-selfother')
    run : int
        Run number
    expected_volumes : int
        Expected number of volumes for a complete collection

    Returns
    -------
    list[AttemptInfo]
        Available attempts sorted by attempt number (highest first)
    """
    if not collection_dir.exists():
        return []

    pattern = f"{task}_run-{run:02d}_attempt-*"
    attempts = []

    for folder in collection_dir.glob(pattern):
        if folder.is_dir():
            match = re.search(r"attempt-(\d+)", folder.name)
            if match:
                attempt_num = int(match.group(1))
                n_files = len([f for f in folder.iterdir() if f.is_file()])
                attempts.append(
                    AttemptInfo(
                        attempt_num=attempt_num,
                        directory=folder,
                        n_volumes=n_files,
                        expected_volumes=expected_volumes,
                        is_complete=n_files >= expected_volumes,
                    )
                )

    attempts.sort(key=lambda a: a.attempt_num, reverse=True)
    return attempts


class UnifiedAttemptDialog(BaseDialog):
    """Modal dialog for selecting attempts for preprocessing.

    Displays all attempts grouped by run, each with a checkbox.
    The user can freely check/uncheck any attempt across any run.
    """

    def __init__(
        self,
        page: ft.Page,
        run_groups: list[RunGroup],
        on_confirm: Callable[[dict[int, list[int]]], None],
        on_cancel: Callable[[], None] | None = None,
    ):
        super().__init__(page)
        self.run_groups = run_groups
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        # Keyed by (run_id, attempt_num)
        self._checkboxes: dict[tuple[int, int], ft.Checkbox] = {}

    def show(self):
        """Display the modal dialog."""
        self._open_dialog(
            ft.AlertDialog(
                modal=True,
                title=ft.Row(
                    [
                        ft.Icon(ft.Icons.PLAYLIST_ADD_CHECK, color=ft.Colors.BLUE),
                        ft.Text(
                            "Select Attempts for Preprocessing",
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
        """Build the dialog content with checkboxes per attempt."""
        run_controls: list[ft.Control] = []

        for run_group in self.run_groups:
            if not run_group.attempts:
                continue

            # Determine default selection: highest complete, or highest overall
            default_attempt = None
            complete = [a for a in run_group.attempts if a.is_complete]
            if complete:
                default_attempt = max(a.attempt_num for a in complete)
            elif run_group.attempts:
                best = [a for a in run_group.attempts if a.n_volumes > 0]
                if best:
                    default_attempt = max(a.attempt_num for a in best)

            # Build checkbox rows for each attempt
            attempt_rows: list[ft.Control] = []
            for attempt in run_group.attempts:
                if attempt.n_volumes == 0:
                    status_icon = ft.Icons.BLOCK
                    status_color = ft.Colors.GREY_400
                    status_text = "No data"
                    can_select = False
                elif attempt.is_complete:
                    status_icon = ft.Icons.CHECK_CIRCLE
                    status_color = ft.Colors.GREEN
                    status_text = "Complete"
                    can_select = True
                else:
                    status_icon = ft.Icons.WARNING_AMBER
                    status_color = ft.Colors.ORANGE
                    status_text = "Incomplete"
                    can_select = True

                progress_value = attempt.n_volumes / attempt.expected_volumes if attempt.expected_volumes > 0 else 0

                checked = can_select and attempt.attempt_num == default_attempt
                checkbox = ft.Checkbox(
                    value=checked,
                    disabled=not can_select,
                )
                self._checkboxes[(run_group.run_id, attempt.attempt_num)] = checkbox

                attempt_row = ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    checkbox,
                                    ft.Text(
                                        f"attempt-{attempt.attempt_num:02d}",
                                        size=14,
                                        weight=ft.FontWeight.W_500,
                                    ),
                                    ft.Container(expand=True),
                                    ft.Icon(status_icon, color=status_color, size=18),
                                    ft.Text(status_text, color=status_color, size=12),
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
                                                    size=12,
                                                    color=ft.Colors.GREY_700,
                                                ),
                                                ft.Text(
                                                    f"{attempt.n_volumes} / {attempt.expected_volumes}",
                                                    size=12,
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
                                            height=6,
                                        ),
                                    ],
                                    spacing=3,
                                ),
                                padding=ft.padding.only(left=40),
                            ),
                        ],
                        spacing=3,
                    ),
                    padding=ft.padding.only(top=2, bottom=2),
                )
                attempt_rows.append(attempt_row)

            run_card = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            run_group.run_name,
                            size=16,
                            weight=ft.FontWeight.W_500,
                        ),
                        *attempt_rows,
                    ],
                    spacing=5,
                ),
                padding=15,
                bgcolor=ft.Colors.GREY_100,
                border_radius=8,
                border=ft.border.all(1, ft.Colors.GREY_300),
            )
            run_controls.append(run_card)

        # Info text
        has_incomplete = any(not a.is_complete and a.n_volumes > 0 for rg in self.run_groups for a in rg.attempts)
        if has_incomplete:
            run_controls.append(ft.Container(height=5))
            run_controls.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.BLUE_700, size=18),
                            ft.Text(
                                "Incomplete attempts may still be usable but could affect quality.",
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
            )

        return ft.Container(
            content=ft.Column(
                run_controls,
                spacing=10,
            ),
            width=480,
            padding=10,
        )

    def _build_actions(self) -> list[ft.Control]:
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

    def _on_confirm_clicked(self, e):
        """Handle confirm button click."""
        # Collect all checked attempts grouped by run
        selected: dict[int, list[int]] = defaultdict(list)
        for (run_id, attempt_num), cb in self._checkboxes.items():
            if cb.value:
                selected[run_id].append(attempt_num)

        if not selected:
            _snack = ft.SnackBar(
                ft.Text("Please select at least one attempt."),
                bgcolor=ft.Colors.ORANGE,
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Sort attempt lists so ordering is deterministic
        result = {run_id: sorted(attempts) for run_id, attempts in sorted(selected.items())}

        self._close_dialog()
        self.on_confirm(result)

    def _on_cancel_clicked(self, e):
        """Handle cancel button click."""
        self._close_dialog()
        if self.on_cancel:
            self.on_cancel()
