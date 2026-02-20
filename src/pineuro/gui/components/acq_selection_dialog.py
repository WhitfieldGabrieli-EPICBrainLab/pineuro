"""Acquisition selection dialog for preprocessing step.

When multiple acquisition attempts exist for a run (e.g., after retries),
allows users to select which acquisition to use for each run.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import flet as ft

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
class RunAcqInfo:
    """Available acquisitions for a single run."""

    run_id: int
    run_name: str
    acquisitions: list[AcqInfo]
    selected_acq: int | None = None  # Default to highest complete

    def __post_init__(self):
        if self.selected_acq is None and self.acquisitions:
            # Default: highest complete acq, or highest overall
            complete = [a for a in self.acquisitions if a.is_complete]
            if complete:
                self.selected_acq = max(a.acq_num for a in complete)
            else:
                self.selected_acq = max(a.acq_num for a in self.acquisitions)


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


class AcqSelectionDialog:
    """Modal dialog for selecting acquisitions for preprocessing.

    Displays available acquisition attempts per run with volume counts
    and completion status. Uses radio buttons since only one acq per run
    can be selected.
    """

    def __init__(
        self,
        page: ft.Page,
        runs: list[RunAcqInfo],
        on_confirm: Callable[[dict[int, int]], None],
        on_cancel: Callable[[], None] | None = None,
    ):
        """Initialize the acquisition selection dialog.

        Args:
            page: Flet Page object
            runs: List of RunAcqInfo objects for available runs
            on_confirm: Callback with dict {run_id: selected_acq_num}
            on_cancel: Optional callback when dialog is cancelled
        """
        self.page = page
        self.runs = runs
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self._dialog: ft.AlertDialog | None = None
        self._radio_groups: dict[int, ft.RadioGroup] = {}

    def show(self):
        """Display the modal dialog."""
        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.TUNE, color=ft.Colors.BLUE),
                    ft.Text(
                        "Select Acquisition for Each Run",
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
        """Build the dialog content with radio buttons per run."""
        run_controls: list[ft.Control] = []

        for run_info in self.runs:
            if not run_info.acquisitions:
                continue

            # Run header
            run_controls.append(
                ft.Text(
                    run_info.run_name,
                    size=16,
                    weight=ft.FontWeight.W_500,
                )
            )

            # Radio buttons for each acq
            radios: list[ft.Control] = []
            for acq in run_info.acquisitions:
                if acq.is_complete:
                    status_icon = ft.Icons.CHECK_CIRCLE
                    status_color = ft.Colors.GREEN
                    status_text = "Complete"
                else:
                    status_icon = ft.Icons.WARNING_AMBER
                    status_color = ft.Colors.ORANGE
                    status_text = "Incomplete"

                progress_value = acq.n_volumes / acq.expected_volumes if acq.expected_volumes > 0 else 0

                _radio_label = ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        f"acq-{acq.acq_num:02d}",
                                        size=14,
                                        weight=ft.FontWeight.W_500,
                                    ),
                                    ft.Container(expand=True),
                                    ft.Icon(status_icon, color=status_color, size=18),
                                    ft.Text(status_text, color=status_color, size=12),
                                ],
                            ),
                            ft.Row(
                                [
                                    ft.Text(
                                        f"{acq.n_volumes} / {acq.expected_volumes} volumes",
                                        size=12,
                                        color=ft.Colors.GREY_700,
                                    ),
                                ],
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
                    width=350,
                    padding=ft.padding.only(left=5, top=2, bottom=2),
                )

                radios.append(
                    ft.Radio(
                        value=str(acq.acq_num),
                        label=f"acq-{acq.acq_num:02d} ({acq.n_volumes}/{acq.expected_volumes} vols)",
                    )
                )

            rg = ft.RadioGroup(
                value=str(run_info.selected_acq),
                content=ft.Column(radios, spacing=5),
            )
            self._radio_groups[run_info.run_id] = rg

            run_card = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            run_info.run_name,
                            size=16,
                            weight=ft.FontWeight.W_500,
                        ),
                        rg,
                    ],
                    spacing=8,
                ),
                padding=15,
                bgcolor=ft.Colors.GREY_100,
                border_radius=8,
                border=ft.border.all(1, ft.Colors.GREY_300),
            )
            run_controls.append(run_card)

        return ft.Container(
            content=ft.Column(
                run_controls,
                spacing=10,
            ),
            width=450,
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
        selected_acqs = {}
        for run_id, rg in self._radio_groups.items():
            if rg.value:
                selected_acqs[run_id] = int(rg.value)

        if not selected_acqs:
            _snack = ft.SnackBar(ft.Text("Please select an acquisition for each run."), bgcolor=ft.Colors.ORANGE)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        self._close_dialog()
        self.on_confirm(selected_acqs)

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
