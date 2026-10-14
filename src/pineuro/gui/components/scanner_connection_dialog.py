"""Scanner connection test dialog for participant workflow.

Runs connectivity checks (ping, directory validation) and displays
results before starting scanner data collection.
"""

import logging
from collections.abc import Callable

import flet as ft

from pineuro.gui import styles as st
from pineuro.scanner_setup import ConnectionTestResult, test_scanner_connection

logger = logging.getLogger(__name__)


class ScannerConnectionDialog:
    """Modal dialog that tests scanner connectivity and watch directory status.

    Runs the test in a background thread and displays animated results.
    """

    def __init__(
        self,
        page: ft.Page,
        scanner_ip: str | None,
        watch_dir: str,
        on_close: Callable[[ConnectionTestResult | None], None] | None = None,
    ):
        self.page = page
        self.scanner_ip = scanner_ip
        self.watch_dir = watch_dir
        self.on_close = on_close
        self._dialog: ft.AlertDialog | None = None
        self._result: ConnectionTestResult | None = None

    def show(self):
        """Display the dialog and start the connection test."""
        self._ping_status = ft.Container(
            content=self._build_checking_row("Network Ping"),
        )
        self._dir_status = ft.Container(
            content=self._build_checking_row("Watch Directory"),
        )
        self._account_status = ft.Container()  # filled only when the account check finds a problem
        self._overall_status = ft.Container()

        ip_display = self.scanner_ip if self.scanner_ip else "not configured"

        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.SETTINGS_ETHERNET, color=st.ACCENT),
                    ft.Text("Scanner Connection Test", weight=ft.FontWeight.BOLD),
                ],
                spacing=st.GAP_SM,
            ),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            f"Scanner IP: {ip_display}",
                            size=st.CAPTION,
                            color=st.MUTED,
                        ),
                        ft.Text(
                            f"Watch dir: {self.watch_dir}",
                            size=st.CAPTION,
                            color=st.MUTED,
                        ),
                        ft.Container(height=st.GAP_MD),
                        self._ping_status,
                        self._dir_status,
                        self._account_status,
                        ft.Container(height=st.GAP_MD),
                        self._overall_status,
                    ],
                    spacing=st.GAP_SM,
                ),
                width=st.DIALOG_MD,
            ),
            actions=[
                ft.TextButton("Close", on_click=lambda e: self._close()),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(self._dialog)
        self._dialog.open = True
        self.page.update()

        self.page.run_thread(self._run_test)

    def _run_test(self):
        """Execute connection test in background thread."""
        result = test_scanner_connection(
            scanner_ip=self.scanner_ip,
            watch_dir=self.watch_dir,
        )
        self._result = result

        try:
            self._update_results(result)

            async def _apply():
                self.page.update()

            self.page.run_task(_apply)
        except Exception as e:
            logger.error(f"Failed to update connection test UI: {e}")

    def _update_results(self, result: ConnectionTestResult):
        """Update dialog with test results."""
        # Ping result
        if result.ping_ok is None:
            self._ping_status.content = self._build_result_row(
                "Network Ping",
                "Skipped (no IP configured)",
                ft.Icons.REMOVE_CIRCLE_OUTLINE,
                st.NEUTRAL,
            )
        elif result.ping_ok:
            self._ping_status.content = self._build_result_row(
                "Network Ping",
                result.ping_detail,
                ft.Icons.CHECK_CIRCLE,
                st.SUCCESS,
            )
        else:
            self._ping_status.content = self._build_result_row(
                "Network Ping",
                result.ping_detail,
                ft.Icons.ERROR,
                st.ERROR,
            )

        # Directory result
        if result.dir_exists and result.dir_writable:
            if result.dir_empty:
                icon = ft.Icons.CHECK_CIRCLE
                color = st.SUCCESS
            else:
                icon = ft.Icons.WARNING
                color = st.WARNING
        else:
            icon = ft.Icons.ERROR
            color = st.ERROR

        self._dir_status.content = self._build_result_row("Watch Directory", result.dir_detail, icon, color)

        # Account: pineuro must run as the account the scanner writes as (checked on macOS/Linux)
        if result.account_detail:
            self._account_status.content = self._build_result_row(
                "Account", result.account_detail, ft.Icons.ERROR, st.ERROR
            )

        # Overall banner
        if result.overall_ok:
            self._overall_status.content = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=st.SUCCESS, size=st.ICON_LG),
                        ft.Text(
                            "Scanner connection ready",
                            size=st.SUBSECTION,
                            weight=ft.FontWeight.BOLD,
                            color=st.SUCCESS,
                        ),
                    ],
                    spacing=st.GAP_SM,
                ),
                padding=st.GAP_MD,
                bgcolor=st.SUCCESS_BG,
                border_radius=5,
            )
        else:
            self._overall_status.content = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.ERROR, color=st.ERROR, size=st.ICON_LG),
                        ft.Text(
                            "Connection issues detected",
                            size=st.SUBSECTION,
                            weight=ft.FontWeight.BOLD,
                            color=st.ERROR,
                        ),
                    ],
                    spacing=st.GAP_SM,
                ),
                padding=st.GAP_MD,
                bgcolor=st.ERROR_BG,
                border_radius=5,
            )

    def _build_checking_row(self, label: str) -> ft.Row:
        """Build a row with a spinning progress indicator."""
        return ft.Row(
            [
                ft.ProgressRing(width=16, height=16, stroke_width=2),
                st.subsection(f"{label}: "),
                ft.Text("Testing...", size=st.BODY, color=st.MUTED),
            ],
            spacing=st.GAP_SM,
        )

    def _build_result_row(self, label: str, detail: str, icon: ft.IconData, color: str) -> ft.Row:
        """Build a result row with status icon."""
        return ft.Row(
            [
                ft.Icon(icon, color=color, size=st.ICON_MD),
                st.subsection(f"{label}: "),
                ft.Text(detail, size=st.CAPTION, color=st.MUTED, expand=True),
            ],
            spacing=st.GAP_SM,
        )

    def _close(self):
        """Close the dialog and invoke callback."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
        if self.on_close:
            self.on_close(self._result)
