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

from . import styles as st
from .config_manager import ConfigManager
from .models import StudyConfig
from .utils import validate_participant_id

if TYPE_CHECKING:
    from pineuro.session import SessionMetadata

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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

        # Discover third-party task plugins via entry points
        self._discover_task_plugins()

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
        self.page.theme = st.theme()

        logger.info("pineuro GUI initialized")

    def _discover_task_plugins(self):
        """Discover and register third-party task plugins.

        Uses importlib.metadata entry points (group='pineuro.tasks') to
        find and load externally installed task packages.
        """
        from pineuro.tasks.task_registry import discover_tasks

        try:
            count = discover_tasks()
            if count > 0:
                logger.info(f"Discovered {count} task plugin(s) via entry points")
        except Exception as exc:
            logger.warning(f"Task plugin discovery failed: {exc}")

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
                                    st.title("pineuro Study Manager"),
                                    ft.Row(
                                        [
                                            ft.OutlinedButton(
                                                "Import Study",
                                                icon=ft.Icons.DOWNLOAD,
                                                on_click=lambda _: self._import_study(),
                                            ),
                                            ft.FilledButton(
                                                "New Study",
                                                icon=ft.Icons.ADD,
                                                on_click=lambda _: self.show_study_editor(None),
                                            ),
                                        ],
                                        spacing=st.GAP_SM,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            padding=st.PAD,
                            bgcolor=st.HEADER_BG,
                        ),
                        # Study list
                        ft.Container(
                            content=self._build_study_list(),
                            padding=st.PAD,
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
                        ft.Icon(ft.Icons.FOLDER_OPEN, size=st.ICON_HERO, color=ft.Colors.OUTLINE),
                        ft.Text("No studies found", size=st.TITLE, color=st.MUTED),
                        st.caption("Create your first study to get started!", italic=False),
                        ft.FilledButton(
                            "New Study",
                            icon=ft.Icons.ADD,
                            on_click=lambda _: self.show_study_editor(None),
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=st.GAP_MD,
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
            spacing=st.GAP_SM,
            run_spacing=st.GAP_SM,
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
                        st.section(study_meta.name, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Text(
                                truncate_text(study_meta.description, 80),
                                size=st.BODY,
                                color=st.MUTED,
                            ),
                            height=60,
                        ),
                        st.caption(f"Modified: {format_timestamp(study_meta.modified)}", italic=False),
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
                                    icon=ft.Icons.HISTORY,
                                    icon_size=24,
                                    tooltip="Version History",
                                    on_click=lambda _, name=study_meta.name: self._show_version_history(name),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE,
                                    icon_size=24,
                                    tooltip="Delete",
                                    on_click=lambda _, name=study_meta.name: self._confirm_delete_study(name),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=st.GAP_XS,
                        ),
                    ],
                    spacing=st.GAP_MD,
                ),
                padding=st.PAD,
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

        from .components.session_selection_dialog import SessionSelectionDialog

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

            base_dir = Path(study_config.session_config["paths"]["base_dir"])

            def on_session_selected(session_id, session_metadata):
                self.show_participant_workflow(study_config, participant_id, session_id, session_metadata)

            session_dialog = SessionSelectionDialog(
                page=self.page,
                participant_id=participant_id,
                base_dir=base_dir,
                on_select=on_session_selected,
                study_config=study_config,
            )
            session_dialog.show()

        def cancel_start(e):
            self.page.pop_dialog()

        dialog = ft.AlertDialog(
            title=ft.Text("Start Participant Session"),
            content=ft.Container(
                content=ft.Column(
                    [
                        st.subsection(f"Study: {study_name}"),
                        ft.Divider(),
                        participant_id_field,
                        st.caption("Enter the participant ID to begin the neurofeedback session."),
                    ],
                    tight=True,
                    spacing=st.GAP_MD,
                ),
                width=st.DIALOG_SM,
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

        Args:
            study_name: Name of the study to export
        """
        from pineuro.gui.utils import pick_save_file

        dest = pick_save_file(
            title=f"Export '{study_name}'",
            default_name=f"{study_name.replace(' ', '_')}.yaml",
            default_extension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )

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
        """Import a study configuration YAML from a user-chosen file."""
        from pineuro.gui.utils import pick_file

        src = pick_file(
            title="Import Study",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )

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

    def _show_version_history(self, study_name: str):
        """Show version history dialog for a study.

        Args:
            study_name: Name of the study
        """
        versions = self.config_manager.list_versions(study_name)

        if not versions:
            _snack = ft.SnackBar(ft.Text(f"No version history for '{study_name}'"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        def restore_version(version_info):
            self.page.pop_dialog()
            try:
                self.config_manager.restore_version(study_name, version_info["path"])
                _snack = ft.SnackBar(ft.Text(f"Restored version {version_info['version']} of '{study_name}'"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                self.show_study_manager()
            except Exception as ex:
                logger.error(f"Failed to restore version: {ex}")
                _snack = ft.SnackBar(ft.Text(f"Restore failed: {ex}"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()

        # Build version list (newest first for display)
        version_rows: list[ft.Control] = []
        for v in reversed(versions):
            version_rows.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.SAVE, color=st.ACCENT),
                    title=ft.Text(f"Version {v['version']}"),
                    subtitle=st.caption(v["timestamp"], italic=False),
                    trailing=ft.OutlinedButton(
                        "Restore",
                        on_click=lambda _, info=v: restore_version(info),
                    ),
                )
            )

        dialog = ft.AlertDialog(
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.HISTORY, color=st.ACCENT),
                    ft.Text(f"Version History: {study_name}", weight=ft.FontWeight.BOLD),
                ],
                spacing=st.GAP_SM,
            ),
            content=ft.Container(
                content=ft.Column(
                    version_rows,
                    spacing=st.GAP_XS,
                    tight=True,
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=st.DIALOG_MD,
                height=400,
            ),
            actions=[ft.TextButton("Close", on_click=lambda _: self.page.pop_dialog())],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

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
            on_start_session=lambda study_config, participant_id, session_id, session_metadata: (
                self.show_participant_workflow(study_config, participant_id, session_id, session_metadata)
            ),
        )

        # Add editor UI
        self.page.add(editor.build())
        self.page.update()

    def show_participant_workflow(
        self,
        study_config: StudyConfig,
        participant_id: str,
        session_id: str,
        session_metadata: "SessionMetadata | None" = None,
    ):
        """Show the participant workflow screen.

        Args:
            study_config: Study configuration to use
            participant_id: Participant ID
            session_id: Session identifier (e.g., 'ses-01').
            session_metadata: SessionMetadata for new sessions. None when continuing existing session.
        """
        from .screens.participant_workflow import ParticipantWorkflowScreen

        logger.info(f"Navigating to Participant Workflow (participant: {participant_id}, session: {session_id})")

        self.current_study = study_config

        # Clear current content
        self.page.controls.clear()

        # Create participant workflow screen
        workflow = ParticipantWorkflowScreen(
            page=self.page,
            config_manager=self.config_manager,
            study_config=study_config,
            participant_id=participant_id,
            session_id=session_id,
            session_metadata=session_metadata,
            on_back=lambda: self.show_study_manager(),
        )

        # Add workflow UI
        self.page.add(workflow.build())
        self.page.update()


def main():
    """Main entry point for the Flet app."""
    import signal
    import threading

    from pineuro.gui.dispatch import MainThreadDispatcher, set_dispatcher

    def create_app(page: ft.Page):
        PyrtnfApp(page)

    # Run Flet in a background thread and keep the main thread free for
    # PsychoPy task dispatch.  Flet's desktop viewer runs as a separate
    # process, so ft.app() doesn't need the main thread.  On macOS this
    # is required (Cocoa NSWindow must be created on the main thread);
    # on all platforms it gives a consistent single-code-path architecture.
    dispatcher = MainThreadDispatcher()
    set_dispatcher(dispatcher)

    # Python only allows signal.signal() from the main thread.  Flet
    # (or its dependencies) registers signal handlers internally — this
    # fails when ft.app() runs in a background thread.  Patch it to
    # silently skip registration from non-main threads.
    _original_signal = signal.signal

    def _safe_signal(signum, handler):
        if threading.current_thread() is threading.main_thread():
            return _original_signal(signum, handler)
        return signal.getsignal(signum)

    signal.signal = _safe_signal

    flet_thread = threading.Thread(target=ft.app, args=(create_app,), daemon=True, name="FletApp")
    flet_thread.start()

    # When the user closes the Flet window, ft.app() returns and the
    # daemon thread exits.  Shut down the dispatcher so the main thread
    # (run_forever) can exit too.
    def _watch_flet_exit():
        flet_thread.join()
        dispatcher.shutdown()

    threading.Thread(target=_watch_flet_exit, daemon=True, name="FletExitWatcher").start()

    dispatcher.run_forever()


if __name__ == "__main__":
    main()
