"""Session selection dialog for multi-session studies.

This dialog allows users to:
- View existing sessions for a participant
- Select an existing session to continue
- Create a new session (with mask compute or reuse options)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

if TYPE_CHECKING:
    # pineuro.session is a heavy module; import it lazily inside methods.
    from pineuro.session import SessionMetadata

logger = logging.getLogger(__name__)


class SessionSelectionDialog:
    """Dialog for selecting or creating sessions for a participant."""

    def __init__(
        self,
        page: ft.Page,
        participant_id: str,
        base_dir: Path,
        on_select: Callable[[str, SessionMetadata | None], None],
    ):
        """Initialize the session selection dialog.

        Args:
            page: Flet Page object
            participant_id: Participant ID
            base_dir: Study base directory
            on_select: Callback when session is selected/created.
                       Called with (session_id, session_metadata) where
                       session_metadata is None when continuing an existing session.
        """
        self.page = page
        self.participant_id = participant_id
        self.base_dir = Path(base_dir)
        self.on_select = on_select
        from pineuro.session import detect_existing_sessions

        self.existing_sessions = detect_existing_sessions(self.base_dir, participant_id)
        self.dialog: ft.AlertDialog | None = None

    def show(self):
        """Show the session selection dialog."""
        if not self.existing_sessions:
            self._show_new_participant_dialog()
        else:
            self._show_multi_session_dialog()

    def _show_new_participant_dialog(self):
        """Show dialog for new participant without existing data."""
        default_session_id = "ses-01"

        content = ft.Column(
            [
                ft.Text(
                    f"Participant: {self.participant_id}",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Divider(),
                ft.Text(
                    "No existing data found for this participant.",
                    size=12,
                    color=ft.Colors.GREY_600,
                ),
                ft.Text(
                    f"A new session ({default_session_id}) will be created.",
                    size=12,
                ),
            ],
            tight=True,
            spacing=10,
        )

        def start_new(e):
            from pineuro.session import SessionMetadata

            self.page.pop_dialog()
            session_metadata = SessionMetadata(
                session_id=default_session_id,
                session_number=1,
                mask_source="compute",
            )
            self.on_select(default_session_id, session_metadata)

        def cancel(e):
            self.page.pop_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("Start New Session"),
            content=ft.Container(content=content, width=400),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.FilledButton("Start", icon=ft.Icons.PLAY_ARROW, on_click=start_new),
            ],
        )
        self.page.show_dialog(self.dialog)

    def _show_multi_session_dialog(self):
        """Show dialog when one or more sessions already exist for this participant."""
        session_tiles: list[ft.Control] = []
        sessions_with_approved_masks = []

        for session in self.existing_sessions:
            session_id = session["session_id"]

            status_parts = []
            if session["has_masks"]:
                if session["masks_approved"]:
                    status_parts.append("Masks approved")
                    sessions_with_approved_masks.append(session_id)
                else:
                    status_parts.append("Masks extracted")
            if session["has_feedback"]:
                status_parts.append("Feedback started")

            status_text = ", ".join(status_parts) if status_parts else "Not started"

            session_tiles.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FOLDER_OPEN),
                    title=ft.Text(session_id),
                    subtitle=ft.Text(status_text, size=11),
                    on_click=lambda e, vid=session_id: self._select_existing_session(vid),
                )
            )

        # Add "New Session" option
        from pineuro.session import next_session_id

        new_session_id = next_session_id(self.base_dir, self.participant_id)

        content = ft.Column(
            [
                ft.Text(
                    f"Participant: {self.participant_id}",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Divider(),
                ft.Text(
                    "Existing sessions:",
                    size=12,
                    weight=ft.FontWeight.W_500,
                ),
                ft.Container(
                    content=ft.Column(session_tiles, spacing=0),
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=8,
                    padding=0,
                ),
                ft.Container(height=10),
                ft.Text(
                    "Or create a new session:",
                    size=12,
                    weight=ft.FontWeight.W_500,
                ),
            ],
            tight=True,
            spacing=8,
        )

        def create_new_session(e):
            self.page.pop_dialog()
            self._show_new_session_dialog(new_session_id, sessions_with_approved_masks)

        def cancel(e):
            self.page.pop_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("Select Session"),
            content=ft.Container(content=content, width=450, height=350),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.FilledButton(
                    f"New Session ({new_session_id})",
                    icon=ft.Icons.ADD,
                    on_click=create_new_session,
                ),
            ],
        )
        self.page.show_dialog(self.dialog)

    def _select_existing_session(self, session_id: str):
        """Handle selection of existing session."""
        self.page.pop_dialog()
        # Continue existing session - no new SessionMetadata needed
        self.on_select(session_id, None)

    def _show_new_session_dialog(self, new_session_id: str, sessions_with_masks: list[str]):
        """Show dialog for creating a new session."""
        # Mask source dropdown options
        mask_options = [
            ft.dropdown.Option("compute", "Compute new masks (run localizer)"),
        ]

        for source_session in sessions_with_masks:
            mask_options.append(
                ft.dropdown.Option(
                    f"reuse:{source_session}",
                    f"Reuse masks from {source_session}",
                )
            )

        mask_source_dropdown = ft.Dropdown(
            label="Mask Source",
            options=mask_options,
            value="compute",
            width=350,
        )

        notes_field = ft.TextField(
            label="Notes (optional)",
            multiline=True,
            max_lines=3,
            width=350,
        )

        content = ft.Column(
            [
                ft.Text(
                    f"Creating session: {new_session_id}",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Divider(),
                mask_source_dropdown,
                ft.Container(height=5),
                ft.Text(
                    "• 'Compute new masks': Runs full localizer and mask extraction",
                    size=11,
                    color=ft.Colors.GREY_600,
                ),
                ft.Text(
                    "• 'Reuse masks': Skips localizer, only collects reference scan for coregistration",
                    size=11,
                    color=ft.Colors.GREY_600,
                ),
                ft.Container(height=10),
                notes_field,
            ],
            tight=True,
            spacing=8,
        )

        def create_session(e):
            from pineuro.session import SessionMetadata

            self.page.pop_dialog()

            mask_value = mask_source_dropdown.value or ""
            if mask_value.startswith("reuse:"):
                mask_source = "reuse"
                mask_source_session = mask_value.split(":", 1)[1]
            else:
                mask_source = "compute"
                mask_source_session = None

            # Determine session number
            existing_multi = [v for v in self.existing_sessions if v["session_id"] is not None]
            session_number = len(existing_multi) + 1

            session_metadata = SessionMetadata(
                session_id=new_session_id,
                session_number=session_number,
                mask_source=mask_source,
                mask_source_session=mask_source_session,
                notes=notes_field.value.strip(),
            )

            logger.info(
                f"Creating new session: {new_session_id}, mask_source={mask_source}, source_session={mask_source_session}"
            )

            self.on_select(new_session_id, session_metadata)

        def back(e):
            self.page.pop_dialog()
            self._show_multi_session_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("New Session Options"),
            content=ft.Container(content=content, width=400),
            actions=[
                ft.TextButton("Back", on_click=back),
                ft.FilledButton("Create", icon=ft.Icons.CHECK, on_click=create_session),
            ],
        )
        self.page.show_dialog(self.dialog)
