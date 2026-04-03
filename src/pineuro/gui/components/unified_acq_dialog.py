"""Unified acquisition selection dialog for preprocessing step.

Replaces the separate run-selection and acq-selection dialogs with a single
dialog that lists all acquisitions grouped by run, each with a checkbox.
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
class AcqInfo:
    """Information about a single acquisition attempt."""

    acq_num: int
    directory: Path
    n_volumes: int
    expected_volumes: int
    is_complete: bool


@dataclass
class RunGroup:
    """All acquisitions for one run, for display grouping."""

    run_id: int
    run_name: str
    acquisitions: list[AcqInfo] = field(default_factory=list)


def scan_collection_acqs(
    collection_dir: Path,
    task: str,
    run: int,
    expected_volumes: int,
) -> list[AcqInfo]:
    """Scan collection directory for acquisition attempts for a given run.

    Parameters
    ----------
    collection_dir : Path
        Collection directory to scan
    task : str
        Task prefix (e.g., 'task-rest', 'task-self_other')
    run : int
        Run number
    expected_volumes : int
        Expected number of volumes for a complete collection

    Returns
    -------
    list[AcqInfo]
        Available acquisitions sorted by acq number (highest first)
    """
    if not collection_dir.exists():
        return []

    pattern = f"{task}_acq-*_run-{run:02d}"
    acqs = []

    for folder in collection_dir.glob(pattern):
        if folder.is_dir():
            match = re.search(r"acq-(\d+)_run-", folder.name)
            if match:
                acq_num = int(match.group(1))
                n_files = len([f for f in folder.iterdir() if f.is_file()])
                acqs.append(
                    AcqInfo(
                        acq_num=acq_num,
                        directory=folder,
                        n_volumes=n_files,
                        expected_volumes=expected_volumes,
                        is_complete=n_files >= expected_volumes,
                    )
                )

    acqs.sort(key=lambda a: a.acq_num, reverse=True)
    return acqs


class UnifiedAcqDialog(BaseDialog):
    """Modal dialog for selecting acquisitions for preprocessing.

    Displays all acquisition attempts grouped by run, each with a checkbox.
    The user can freely check/uncheck any acquisition across any run.
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
        # Keyed by (run_id, acq_num)
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
                            "Select Acquisitions for Preprocessing",
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
        """Build the dialog content with checkboxes per acquisition."""
        run_controls: list[ft.Control] = []

        for run_group in self.run_groups:
            if not run_group.acquisitions:
                continue

            # Determine default selection: highest complete, or highest overall
            default_acq = None
            complete = [a for a in run_group.acquisitions if a.is_complete]
            if complete:
                default_acq = max(a.acq_num for a in complete)
            elif run_group.acquisitions:
                best = [a for a in run_group.acquisitions if a.n_volumes > 0]
                if best:
                    default_acq = max(a.acq_num for a in best)

            # Build checkbox rows for each acquisition
            acq_rows: list[ft.Control] = []
            for acq in run_group.acquisitions:
                if acq.n_volumes == 0:
                    status_icon = ft.Icons.BLOCK
                    status_color = ft.Colors.GREY_400
                    status_text = "No data"
                    can_select = False
                elif acq.is_complete:
                    status_icon = ft.Icons.CHECK_CIRCLE
                    status_color = ft.Colors.GREEN
                    status_text = "Complete"
                    can_select = True
                else:
                    status_icon = ft.Icons.WARNING_AMBER
                    status_color = ft.Colors.ORANGE
                    status_text = "Incomplete"
                    can_select = True

                progress_value = acq.n_volumes / acq.expected_volumes if acq.expected_volumes > 0 else 0

                checked = can_select and acq.acq_num == default_acq
                checkbox = ft.Checkbox(
                    value=checked,
                    disabled=not can_select,
                )
                self._checkboxes[(run_group.run_id, acq.acq_num)] = checkbox

                acq_row = ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    checkbox,
                                    ft.Text(
                                        f"acq-{acq.acq_num:02d}",
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
                                                    f"{acq.n_volumes} / {acq.expected_volumes}",
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
                acq_rows.append(acq_row)

            run_card = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            run_group.run_name,
                            size=16,
                            weight=ft.FontWeight.W_500,
                        ),
                        *acq_rows,
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
        has_incomplete = any(not a.is_complete and a.n_volumes > 0 for rg in self.run_groups for a in rg.acquisitions)
        if has_incomplete:
            run_controls.append(ft.Container(height=5))
            run_controls.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.BLUE_700, size=18),
                            ft.Text(
                                "Incomplete acquisitions may still be usable but could affect quality.",
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
        # Collect all checked acquisitions grouped by run
        selected: dict[int, list[int]] = defaultdict(list)
        for (run_id, acq_num), cb in self._checkboxes.items():
            if cb.value:
                selected[run_id].append(acq_num)

        if not selected:
            _snack = ft.SnackBar(
                ft.Text("Please select at least one acquisition."),
                bgcolor=ft.Colors.ORANGE,
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Sort acq lists so ordering is deterministic
        result = {run_id: sorted(acqs) for run_id, acqs in sorted(selected.items())}

        self._close_dialog()
        self.on_confirm(result)

    def _on_cancel_clicked(self, e):
        """Handle cancel button click."""
        self._close_dialog()
        if self.on_cancel:
            self.on_cancel()
