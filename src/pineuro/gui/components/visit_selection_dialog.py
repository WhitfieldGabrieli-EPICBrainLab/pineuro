"""Visit selection dialog for multi-visit studies.

This dialog allows users to:
- View existing visits for a participant
- Select an existing visit to continue
- Create a new visit (with mask compute or reuse options)
"""

import logging
from collections.abc import Callable
from pathlib import Path

import flet as ft

from pineuro.session import VisitConfig, detect_existing_visits, next_visit_id

logger = logging.getLogger(__name__)


class VisitSelectionDialog:
    """Dialog for selecting or creating visits for a participant."""

    def __init__(
        self,
        page: ft.Page,
        participant_id: str,
        base_dir: Path,
        on_select: Callable[[str | None, VisitConfig | None], None],
    ):
        """Initialize the visit selection dialog.

        Args:
            page: Flet Page object
            participant_id: Participant ID
            base_dir: Study base directory
            on_select: Callback when visit is selected/created.
                       Called with (visit_id, visit_config) where:
                       - visit_id is None for single-visit mode
                       - visit_config is None when continuing existing visit
        """
        self.page = page
        self.participant_id = participant_id
        self.base_dir = Path(base_dir)
        self.on_select = on_select
        self.existing_visits = detect_existing_visits(self.base_dir, participant_id)
        self.dialog: ft.AlertDialog | None = None

    def show(self):
        """Show the visit selection dialog."""
        # Check if participant has any existing visits
        if not self.existing_visits:
            # No existing data - offer single-visit vs multi-visit choice
            self._show_new_participant_dialog()
        elif len(self.existing_visits) == 1 and self.existing_visits[0]["visit_id"] is None:
            # Single-visit mode data exists - continue or migrate
            self._show_single_visit_dialog()
        else:
            # Multi-visit data exists - select or create new
            self._show_multi_visit_dialog()

    def _show_new_participant_dialog(self):
        """Show dialog for new participant without existing data."""
        # For new participants, just start with ses-01 in multi-visit mode
        # This simplifies the UX while maintaining the new structure
        default_visit_id = "ses-01"

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
                    f"A new visit ({default_visit_id}) will be created.",
                    size=12,
                ),
            ],
            tight=True,
            spacing=10,
        )

        def start_new(e):
            self.page.pop_dialog()
            visit_config = VisitConfig(
                visit_id=default_visit_id,
                visit_number=1,
                mask_source="compute",
            )
            self.on_select(default_visit_id, visit_config)

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

    def _show_single_visit_dialog(self):
        """Show dialog when single-visit data exists."""
        visit_info = self.existing_visits[0]

        status_items = []
        if visit_info["has_masks"]:
            status_items.append("Masks: extracted")
            if visit_info["masks_approved"]:
                status_items.append("(approved)")
        if visit_info["has_feedback"]:
            status_items.append("Feedback: started")

        status_text = ", ".join(status_items) if status_items else "No completed steps"

        content = ft.Column(
            [
                ft.Text(
                    f"Participant: {self.participant_id}",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Divider(),
                ft.Text(
                    "Existing single-visit session found.",
                    size=12,
                ),
                ft.Text(
                    f"Status: {status_text}",
                    size=12,
                    color=ft.Colors.GREY_600,
                ),
                ft.Container(height=10),
                ft.Text(
                    "Choose an option:",
                    size=12,
                    weight=ft.FontWeight.W_500,
                ),
            ],
            tight=True,
            spacing=8,
        )

        def continue_session(e):
            self.page.pop_dialog()
            # Continue in single-visit mode (backward compatible)
            self.on_select(None, None)

        def cancel(e):
            self.page.pop_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("Continue Session"),
            content=ft.Container(content=content, width=400),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.FilledButton(
                    "Continue",
                    icon=ft.Icons.PLAY_ARROW,
                    on_click=continue_session,
                ),
            ],
        )
        self.page.show_dialog(self.dialog)

    def _show_multi_visit_dialog(self):
        """Show dialog when multi-visit data exists."""
        # Build visit list
        visit_tiles = []
        visits_with_approved_masks = []

        for visit in self.existing_visits:
            visit_id = visit["visit_id"]
            if visit_id is None:
                continue  # Skip single-visit entries in multi-visit mode

            status_parts = []
            if visit["has_masks"]:
                if visit["masks_approved"]:
                    status_parts.append("Masks approved")
                    visits_with_approved_masks.append(visit_id)
                else:
                    status_parts.append("Masks extracted")
            if visit["has_feedback"]:
                status_parts.append("Feedback started")

            status_text = ", ".join(status_parts) if status_parts else "Not started"

            visit_tiles.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FOLDER_OPEN),
                    title=ft.Text(visit_id),
                    subtitle=ft.Text(status_text, size=11),
                    on_click=lambda e, vid=visit_id: self._select_existing_visit(vid),
                )
            )

        # Add "New Visit" option
        new_visit_id = next_visit_id(self.base_dir, self.participant_id)

        content = ft.Column(
            [
                ft.Text(
                    f"Participant: {self.participant_id}",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Divider(),
                ft.Text(
                    "Existing visits:",
                    size=12,
                    weight=ft.FontWeight.W_500,
                ),
                ft.Container(
                    content=ft.Column(visit_tiles, spacing=0),
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=8,
                    padding=0,
                ),
                ft.Container(height=10),
                ft.Text(
                    "Or create a new visit:",
                    size=12,
                    weight=ft.FontWeight.W_500,
                ),
            ],
            tight=True,
            spacing=8,
        )

        def create_new_visit(e):
            self.page.pop_dialog()
            self._show_new_visit_dialog(new_visit_id, visits_with_approved_masks)

        def cancel(e):
            self.page.pop_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("Select Visit"),
            content=ft.Container(content=content, width=450, height=350),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.FilledButton(
                    f"New Visit ({new_visit_id})",
                    icon=ft.Icons.ADD,
                    on_click=create_new_visit,
                ),
            ],
        )
        self.page.show_dialog(self.dialog)

    def _select_existing_visit(self, visit_id: str):
        """Handle selection of existing visit."""
        self.page.pop_dialog()
        # Continue existing visit - no new VisitConfig needed
        self.on_select(visit_id, None)

    def _show_new_visit_dialog(self, new_visit_id: str, visits_with_masks: list[str]):
        """Show dialog for creating a new visit."""
        # Mask source dropdown options
        mask_options = [
            ft.dropdown.Option("compute", "Compute new masks (run localizer)"),
        ]

        for source_visit in visits_with_masks:
            mask_options.append(
                ft.dropdown.Option(
                    f"reuse:{source_visit}",
                    f"Reuse masks from {source_visit}",
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
                    f"Creating visit: {new_visit_id}",
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
                    "• 'Reuse masks': Skips localizer, only collects 2vol for coregistration",
                    size=11,
                    color=ft.Colors.GREY_600,
                ),
                ft.Container(height=10),
                notes_field,
            ],
            tight=True,
            spacing=8,
        )

        def create_visit(e):
            self.page.pop_dialog()

            mask_value = mask_source_dropdown.value
            if mask_value.startswith("reuse:"):
                mask_source = "reuse"
                mask_source_visit = mask_value.split(":", 1)[1]
            else:
                mask_source = "compute"
                mask_source_visit = None

            # Determine visit number
            existing_multi = [v for v in self.existing_visits if v["visit_id"] is not None]
            visit_number = len(existing_multi) + 1

            visit_config = VisitConfig(
                visit_id=new_visit_id,
                visit_number=visit_number,
                mask_source=mask_source,
                mask_source_visit=mask_source_visit,
                notes=notes_field.value.strip(),
            )

            logger.info(
                f"Creating new visit: {new_visit_id}, mask_source={mask_source}, source_visit={mask_source_visit}"
            )

            self.on_select(new_visit_id, visit_config)

        def back(e):
            self.page.pop_dialog()
            self._show_multi_visit_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("New Visit Options"),
            content=ft.Container(content=content, width=400),
            actions=[
                ft.TextButton("Back", on_click=back),
                ft.FilledButton("Create", icon=ft.Icons.CHECK, on_click=create_visit),
            ],
        )
        self.page.show_dialog(self.dialog)
