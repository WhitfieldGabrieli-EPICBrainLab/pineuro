"""Main Flet application entry point for the pineuro GUI.

This module provides the root application class and serves as the
central hub for all GUI navigation. It implements a single-page
application pattern where different "screens" are rendered by
clearing and rebuilding the page content.

Screens
-------
**Study Manager** (default screen)
    Browse saved studies as cards with metadata preview.
    Actions: New Study, Edit, Duplicate, Delete, Start Session.

**Study Editor**
    Configure study parameters in a tabbed interface.
    Tabs: General, Paths, Preprocessing, Mask Extraction, Feedback, Sham.

**Participant Workflow**
    Execute and monitor pipeline steps for a specific participant.
    Shows real-time progress, QC reports, and run controls.

Navigation Flow
---------------
1. App starts → Study Manager
2. Click study card → Prompt for participant ID → Participant Workflow
3. Click "Edit" → Study Editor → Save → Study Manager
4. Click "Back" in any screen → Study Manager

Key Components
--------------
PyrtnfApp : Main application class
    Manages screen transitions, current study state, and page configuration.

main() : Entry point function
    Creates the Flet app and launches the GUI window.

Configuration
-------------
The app uses ConfigManager to persist study configurations in
~/.pineuro/studies/ as YAML files.

Window Configuration
--------------------
- Title: "pineuro - Real-time fMRI Neurofeedback"
- Default size: 1400×900, maximized
- Theme: Light mode
- Padding: 0 (screens manage their own padding)

Usage
-----
From command line:
    python -m pineuro.gui.app

Programmatically:
    >>> from pineuro.gui.app import main
    >>> main()

Dependencies
------------
flet : Cross-platform GUI framework
ConfigManager : Study configuration persistence
StudyConfig : Study data model

See Also
--------
screens.study_editor : Study configuration screen
screens.participant_workflow : Pipeline execution screen
"""

import logging
from typing import TYPE_CHECKING

import flet as ft

from .config_manager import ConfigManager
from .models import StudyConfig
from .utils import validate_participant_id

if TYPE_CHECKING:
    from pineuro.session import VisitConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pre-built study templates — applied as overrides on top of StudyEditorScreen._create_default_config()
STUDY_TEMPLATES: dict[str, dict] = {
    "Ball Task — Single Visit": {
        "description": "Standard single-visit neurofeedback with the ball task (default settings).",
        "icon": "sports_soccer",
        "overrides": {},
    },
    "Ball Task — Multi-Visit": {
        "description": "Longitudinal neurofeedback study with the ball task across multiple visits.",
        "icon": "calendar_month",
        "overrides": {
            "study": {"multi_visit_enabled": True},
        },
    },
    "Thermometer Task": {
        "description": "Single-network neurofeedback using the thermometer display (ContA target).",
        "icon": "thermostat",
        "overrides": {
            "task": {"task_type": "thermometer"},
            "mask": {"yeo_targets": ["ContA"], "mask_names": ["ContA"]},
        },
    },
    "Sham-Controlled (Ball)": {
        "description": "Ball task with blinded sham condition — hemisphere-split alternative masks.",
        "icon": "science",
        "overrides": {
            "sham": {
                "enabled": True,
                "blind_mode": True,
                "delivery_method": "alternative_masks",
            },
        },
    },
}


class PyrtnfApp:
    """Main application class managing screen navigation and state."""

    def __init__(self, page: ft.Page):
        """Initialize the application.

        Args:
            page: Flet Page object
        """
        self.page = page
        self.config_manager = ConfigManager()
        self.current_study: StudyConfig | None = None

        # Configure page
        self._setup_page()

        # Initialize with study manager screen
        self.show_study_manager()

    def _setup_page(self):
        """Configure the Flet page properties."""
        self.page.title = "pineuro - Real-time fMRI Neurofeedback"
        self.page.window.width = 1400
        self.page.window.height = 900
        self.page.window.maximizable = True
        self.page.window.maximized = True
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.theme = ft.Theme(
            visual_density=ft.VisualDensity.COMPACT,
        )

        logger.info("pineuro GUI initialized")

    def show_study_manager(self):
        """Show the study manager screen."""
        logger.info("Navigating to Study Manager")

        # Clear current content
        self.page.controls.clear()

        # Create study manager screen
        self.page.add(
            ft.Container(
                content=ft.Column(
                    [
                        # Header
                        ft.Container(
                            content=ft.Row(
                                [
                                    ft.Text(
                                        "pineuro Study Manager",
                                        size=24,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Row(
                                        [
                                            ft.OutlinedButton(
                                                "New from Template",
                                                icon=ft.Icons.AUTO_AWESOME,
                                                on_click=lambda _: self._show_template_dialog(),
                                            ),
                                            ft.OutlinedButton(
                                                "Import Study",
                                                icon=ft.Icons.DOWNLOAD,
                                                on_click=lambda _: self._import_study(),
                                            ),
                                            ft.FilledButton(
                                                "+ New Study",
                                                on_click=lambda _: self.show_study_editor(None),
                                            ),
                                        ],
                                        spacing=8,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            padding=20,
                            bgcolor=ft.Colors.BLUE_50,
                        ),
                        # Study list
                        ft.Container(
                            content=self._build_study_list(),
                            padding=20,
                            expand=True,
                        ),
                    ],
                    spacing=0,
                ),
                expand=True,
            )
        )

        self.page.update()

    def _build_study_list(self) -> ft.Control:
        """Build the grid of study cards.

        Returns:
            Flet control containing study cards
        """
        studies = self.config_manager.list_studies()

        if not studies:
            return ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(ft.Icons.FOLDER_OPEN, size=100, color=ft.Colors.GREY_400),
                        ft.Text(
                            "No studies found",
                            size=20,
                            color=ft.Colors.GREY_600,
                        ),
                        ft.Text(
                            "Create your first study to get started!",
                            size=14,
                            color=ft.Colors.GREY_500,
                        ),
                        ft.ElevatedButton(
                            "Create New Study",
                            icon=ft.Icons.CREATE,
                            on_click=lambda _: self.show_study_editor(None),
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                alignment=ft.Alignment(0, 0),
            )

        # Create study cards in a grid
        cards: list[ft.Control] = []
        for study_meta in studies:
            cards.append(self._create_study_card(study_meta))

        return ft.GridView(
            controls=cards,
            runs_count=3,
            max_extent=500,
            child_aspect_ratio=1.2,
            spacing=10,
            run_spacing=10,
        )

    def _create_study_card(self, study_meta) -> ft.Card:
        """Create a card for a single study.

        Args:
            study_meta: StudyMetadata object

        Returns:
            Flet Card control
        """
        from .utils import format_timestamp, truncate_text

        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            study_meta.name,
                            size=24,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Text(
                                truncate_text(study_meta.description, 80),
                                size=16,
                                color=ft.Colors.GREY_700,
                            ),
                            height=60,
                        ),
                        ft.Text(
                            f"Modified: {format_timestamp(study_meta.modified)}",
                            size=14,
                            color=ft.Colors.GREY_500,
                        ),
                        ft.Divider(),
                        ft.Row(
                            [
                                ft.IconButton(
                                    icon=ft.Icons.EDIT,
                                    icon_size=24,
                                    tooltip="Edit Study Configuration",
                                    on_click=lambda _, name=study_meta.name: self.show_study_editor(name),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.CONTENT_COPY,
                                    icon_size=24,
                                    tooltip="Duplicate",
                                    on_click=lambda _, name=study_meta.name: self._duplicate_study(name),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.UPLOAD_FILE,
                                    icon_size=24,
                                    tooltip="Export study config to file",
                                    on_click=lambda _, name=study_meta.name: self._export_study(name),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE,
                                    icon_size=24,
                                    tooltip="Delete",
                                    on_click=lambda _, name=study_meta.name: self._confirm_delete_study(name),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=5,
                        ),
                    ],
                    spacing=12,
                ),
                padding=20,
                ink=True,
                on_click=lambda _, name=study_meta.name: self._prompt_start_session(name),
            ),
        )

    def _prompt_start_session(self, study_name: str):
        """Prompt for participant ID to start a session.

        Args:
            study_name: Name of the study to start
        """
        from pathlib import Path

        from .components.visit_selection_dialog import VisitSelectionDialog

        logger.info(f"Starting session for study: {study_name}")

        # Load study to get example participant ID
        study_config = self.config_manager.load_study(study_name)
        if not study_config:
            _snack = ft.SnackBar(ft.Text(f"Failed to load study: {study_name}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        example_id = study_config.session_config.get("paths", {}).get("example_participant_id", "sub-001")

        # Create text field for participant ID
        def on_id_change(e):
            pid = (participant_id_field.value or "").strip()
            if pid:
                _, error = validate_participant_id(pid)
                participant_id_field.error_text = error
            else:
                participant_id_field.error_text = None
            participant_id_field.update()

        participant_id_field = ft.TextField(
            label="Participant ID",
            hint_text=f"e.g., {example_id}",
            autofocus=True,
            on_change=on_id_change,
        )

        def start_session(e):
            participant_id = (participant_id_field.value or "").strip()

            is_valid, error = validate_participant_id(participant_id)
            if not is_valid:
                participant_id_field.error_text = error
                participant_id_field.update()
                return

            self.page.pop_dialog()

            # Check if multi-visit mode is enabled for this study
            multi_visit_enabled = study_config.session_config.get("study", {}).get("multi_visit_enabled", False)

            if multi_visit_enabled:
                # Get base directory from config
                base_dir = Path(study_config.session_config["paths"]["base_dir"])

                # Show visit selection dialog
                def on_visit_selected(visit_id, visit_config):
                    self.show_participant_workflow(study_config, participant_id, visit_id, visit_config)

                visit_dialog = VisitSelectionDialog(
                    page=self.page,
                    participant_id=participant_id,
                    base_dir=base_dir,
                    on_select=on_visit_selected,
                )
                visit_dialog.show()
            else:
                # Single-visit mode - go directly to workflow
                self.show_participant_workflow(study_config, participant_id)

        def cancel_start(e):
            self.page.pop_dialog()

        dialog = ft.AlertDialog(
            title=ft.Text("Start Participant Session"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            f"Study: {study_name}",
                            size=14,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Divider(),
                        participant_id_field,
                        ft.Text(
                            "Enter the participant ID to begin the neurofeedback session.",
                            size=12,
                            color=ft.Colors.GREY_600,
                            italic=True,
                        ),
                    ],
                    tight=True,
                    spacing=10,
                ),
                width=400,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=cancel_start),
                ft.FilledButton("Start", icon=ft.Icons.PLAY_ARROW, on_click=start_session),
            ],
        )

        self.page.show_dialog(dialog)

    def _duplicate_study(self, study_name: str):
        """Duplicate a study.

        Args:
            study_name: Name of the study to duplicate
        """
        logger.info(f"Duplicating study: {study_name}")
        new_study = self.config_manager.duplicate_study(study_name)

        if new_study:
            _snack = ft.SnackBar(ft.Text(f"Study duplicated: {new_study.metadata.name}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            self.show_study_manager()  # Refresh

    def _export_study(self, study_name: str):
        """Export a study configuration YAML to a user-chosen file.

        Uses tkinter filedialog (same pattern as study_editor.py directory picker).

        Args:
            study_name: Name of the study to export
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            _snack = ft.SnackBar(ft.Text("File browser requires tkinter"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        dest = filedialog.asksaveasfilename(
            title=f"Export '{study_name}'",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialfile=f"{study_name.replace(' ', '_')}.yaml",
        )
        root.destroy()

        if not dest:
            return

        from pathlib import Path

        success = self.config_manager.export_study(study_name, Path(dest))
        msg = f"Exported to {dest}" if success else "Export failed: study not found"
        _snack = ft.SnackBar(ft.Text(msg))
        _snack.open = True
        self.page.overlay.append(_snack)
        self.page.update()

    def _import_study(self):
        """Import a study configuration YAML from a user-chosen file.

        Uses tkinter filedialog (same pattern as study_editor.py file picker).
        """
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            _snack = ft.SnackBar(ft.Text("File browser requires tkinter"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        src = filedialog.askopenfilename(
            title="Import Study",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        root.destroy()

        if not src:
            return

        from pathlib import Path

        study = self.config_manager.import_study(Path(src))
        if study:
            _snack = ft.SnackBar(ft.Text(f"Imported study: {study.metadata.name}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            self.show_study_manager()
        else:
            _snack = ft.SnackBar(ft.Text("Import failed: could not read study file"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

    def _confirm_delete_study(self, study_name: str):
        """Show confirmation dialog before deleting a study.

        Args:
            study_name: Name of the study to delete
        """

        def delete_confirmed(e):
            self.page.pop_dialog()

            success = self.config_manager.delete_study(study_name)
            if success:
                _snack = ft.SnackBar(ft.Text(f"Study deleted: {study_name}"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                self.show_study_manager()  # Refresh

        def cancel_delete(e):
            self.page.pop_dialog()

        dialog = ft.AlertDialog(
            title=ft.Text("Delete Study?"),
            content=ft.Text(f"Are you sure you want to delete '{study_name}'? This cannot be undone."),
            actions=[
                ft.TextButton("Cancel", on_click=cancel_delete),
                ft.TextButton("Delete", on_click=delete_confirmed),
            ],
        )

        self.page.show_dialog(dialog)

    def _show_template_dialog(self):
        """Show a dialog for selecting a study template."""

        def _use_template(template_name: str):
            self.page.pop_dialog()
            self._create_study_from_template(template_name)

        template_cards: list[ft.Control] = []
        for name, info in STUDY_TEMPLATES.items():
            tname = name  # capture for lambda
            template_cards.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.AUTO_AWESOME, color=ft.Colors.BLUE_400),
                    title=ft.Text(tname, weight=ft.FontWeight.BOLD),
                    subtitle=ft.Text(info["description"], size=12, color=ft.Colors.GREY_600),
                    trailing=ft.OutlinedButton(
                        "Use",
                        on_click=lambda _, n=tname: _use_template(n),
                    ),
                )
            )

        dialog = ft.AlertDialog(
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.AUTO_AWESOME, color=ft.Colors.BLUE_400),
                    ft.Text("New Study from Template", weight=ft.FontWeight.BOLD),
                ],
                spacing=8,
            ),
            content=ft.Container(
                content=ft.Column(template_cards, spacing=4, tight=True),
                width=520,
            ),
            actions=[ft.TextButton("Cancel", on_click=lambda _: self.page.pop_dialog())],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

    def _create_study_from_template(self, template_name: str):
        """Open the study editor pre-filled with a template configuration.

        Args:
            template_name: Key in STUDY_TEMPLATES dict
        """
        import copy

        from .models import StudyConfig, StudyMetadata
        from .screens.study_editor import StudyEditorScreen

        template = STUDY_TEMPLATES.get(template_name)
        if template is None:
            return

        # Build default config using the same helper as StudyEditorScreen
        default_config = StudyEditorScreen(
            page=self.page,
            config_manager=self.config_manager,
        ).study_config

        # Deep-copy the session config and apply template overrides
        session_config = copy.deepcopy(default_config.session_config)
        for section, values in template["overrides"].items():
            if section not in session_config:
                session_config[section] = {}
            session_config[section].update(values)

        # Build the template StudyConfig
        template_config = StudyConfig(
            metadata=StudyMetadata(name=template_name, description=template["description"]),
            session_config=session_config,
            file_path=None,
        )

        logger.info(f"Opening study editor from template: {template_name}")
        self.current_study = template_config
        self.page.controls.clear()

        from .screens.study_editor import StudyEditorScreen

        editor = StudyEditorScreen(
            page=self.page,
            config_manager=self.config_manager,
            study_config=template_config,
            on_cancel=lambda: self.show_study_manager(),
            on_save=lambda: self.show_study_manager(),
        )
        self.page.add(editor.build())
        self.page.update()

    def show_study_editor(self, study_name: str | None):
        """Show the study editor screen.

        Args:
            study_name: Name of study to edit, or None for new study
        """
        from .screens.study_editor import StudyEditorScreen

        logger.info(f"Navigating to Study Editor (study: {study_name})")

        # Load study if editing
        if study_name:
            self.current_study = self.config_manager.load_study(study_name)
        else:
            self.current_study = None

        # Clear current content
        self.page.controls.clear()

        # Create study editor
        editor = StudyEditorScreen(
            page=self.page,
            config_manager=self.config_manager,
            study_config=self.current_study,
            on_cancel=lambda: self.show_study_manager(),
            on_save=lambda: self.show_study_manager(),
            on_start_session=lambda participant_id, visit_id=None, visit_config=None: (
                self.show_participant_workflow(self.current_study, participant_id, visit_id, visit_config)
                if self.current_study is not None
                else None
            ),
        )

        # Add editor UI
        self.page.add(editor.build())
        self.page.update()

    def show_participant_workflow(
        self,
        study_config: StudyConfig,
        participant_id: str,
        visit_id: str | None = None,
        visit_config: "VisitConfig | None" = None,
    ):
        """Show the participant workflow screen.

        Args:
            study_config: Study configuration to use
            participant_id: Participant ID
            visit_id: Visit identifier (e.g., 'ses-01'). None for single-visit mode.
            visit_config: VisitConfig for new visits. None when continuing existing visit.
        """
        from .screens.participant_workflow import ParticipantWorkflowScreen

        visit_str = f", visit: {visit_id}" if visit_id else ""
        logger.info(f"Navigating to Participant Workflow (participant: {participant_id}{visit_str})")

        self.current_study = study_config

        # Clear current content
        self.page.controls.clear()

        # Create participant workflow screen
        workflow = ParticipantWorkflowScreen(
            page=self.page,
            config_manager=self.config_manager,
            study_config=study_config,
            participant_id=participant_id,
            visit_id=visit_id,
            visit_config=visit_config,
            on_back=lambda: self.show_study_manager(),
        )

        # Add workflow UI
        self.page.add(workflow.build())
        self.page.update()


def main():
    """Main entry point for the Flet app."""

    def create_app(page: ft.Page):
        PyrtnfApp(page)

    ft.app(target=create_app)


if __name__ == "__main__":
    main()
