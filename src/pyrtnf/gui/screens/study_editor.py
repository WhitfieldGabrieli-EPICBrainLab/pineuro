"""Study Editor screen for creating and editing study configurations.

This module provides a tabbed interface for configuring all aspects of a pyrtnf study:
- Metadata and scanner parameters
- Network selection and preprocessing
- Task-specific parameters (Ball task, future tasks)
- File paths for simulator mode

The editor supports both creating new studies and editing existing ones, with
real-time validation and user-friendly file browsing via tkinter dialogs.
"""

import logging
from typing import Optional, Dict

import flet as ft

from pyrtnf.tasks import get_task_metadata, list_available_tasks, TaskParameter
from ..config_manager import ConfigManager
from ..models import StudyConfig, StudyMetadata
from ..utils import (
    ValidationError,
    validate_positive_number,
    validate_positive_int,
    validate_yeo_networks,
)

logger = logging.getLogger(__name__)


class StudyEditorScreen:
    """Study editor screen with tabbed parameter sections."""

    def __init__(
        self,
        page: ft.Page,
        config_manager: ConfigManager,
        study_config: Optional[StudyConfig] = None,
        on_cancel=None,
        on_save=None,
        on_start_session=None
    ):
        """Initialize the study editor.

        Args:
            page: Flet Page object
            config_manager: ConfigManager instance
            study_config: Study to edit, or None for new study
            on_cancel: Callback when cancel is clicked
            on_save: Callback when save is successful
            on_start_session: Callback when start session is clicked
        """
        self.page = page
        self.config_manager = config_manager
        self.study_config = study_config
        self.is_new = study_config is None
        self.on_cancel_callback = on_cancel
        self.on_save_callback = on_save
        self.on_start_session_callback = on_start_session

        # Create default config if new study
        if self.is_new:
            self.study_config = self._create_default_config()

        # Validation state
        self.validation_errors = {}

        # Dynamic task parameter fields (populated based on task type)
        self.task_param_fields: Dict[str, ft.Control] = {}

        # Create form fields
        self._create_form_fields()

    def _create_default_config(self) -> StudyConfig:
        """Create a default study configuration.

        Returns:
            Default StudyConfig
        """
        metadata = StudyMetadata(
            name="New Study",
            description=""
        )

        session_config = {
            "paths": {
                "participant_id": "",
                "base_dir": "/data/pyrtnf",
                "example_participant_id": "sub-001",
            },
            "scan": {
                "tr": 1.2,
                "n_volumes": 150,
                "n_baseline": 25,
                "rest_source_dicoms": ["", ""],
                "rest_n_volumes": [200, 200],
                "rest_max_runs": 2,
                "twovol_initial_dicoms": "",
                "twovol_realtime_dicoms": "",
                "twovol_n_volumes": 2,
                "feedback_dicoms": "",
                "collection_speed": 10.0,
                "simulator_speed": 1.0,
            },
            "mask": {
                "n_yeo": 17,
                "yeo_targets": ["DefaultA", "ContA"],
                "smooth_fwhm": 5.0,
                "highpass": 0.01,
                "lowpass": None,
                "mask_names": ["DefaultA", "ContA"],
            },
            "feedback": {
                "n_baseline": 25,
                "n_volumes": 150,
                "use_efficiency_weights": True,
            },
            "task": {
                "task_type": "ball",  # Task type identifier
                "max_feedback_runs": 2,
                # Ball task parameters
                "ball": {
                    "use_transfer_runs": True,
                    "initial_scale_factor": 10.0,
                    "fullscreen": True,
                    "trigger_keys": ["t", "+", "5", "equal"],
                    "left_button": "1",
                    "right_button": "2",
                    "enter_button": "0",
                    "min_hits_threshold": 3,
                    "max_hits_threshold": 5,
                    "scale_adjust_factor": 0.25,
                },
            }
        }

        return StudyConfig(metadata=metadata, session_config=session_config)

    def _create_form_fields(self):
        """Create all form field controls."""
        config = self.study_config.session_config

        # Metadata fields
        self.name_field = ft.TextField(
            label="Study Name",
            value=self.study_config.metadata.name,
            on_change=self._on_name_change,
        )

        self.description_field = ft.TextField(
            label="Description",
            value=self.study_config.metadata.description,
            multiline=True,
            min_lines=2,
            max_lines=4,
        )

        self.example_participant_id_field = ft.TextField(
            label="Example Participant ID",
            value=config["paths"].get("example_participant_id", "sub-001"),
            hint_text="Shown as hint when starting a session (e.g., sub-001, sub-mindbpd001)",
        )

        # Scanner fields
        self.tr_field = ft.TextField(
            label="TR (seconds)",
            value=str(config["scan"]["tr"]),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("tr", self.tr_field, validate_positive_number),
        )

        self.n_volumes_field = ft.TextField(
            label="Volumes per feedback run",
            value=str(config["scan"]["n_volumes"]),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_volumes", self.n_volumes_field, validate_positive_int),
        )

        self.n_baseline_field = ft.TextField(
            label="Baseline volumes",
            value=str(config["scan"]["n_baseline"]),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_baseline", self.n_baseline_field, validate_positive_int),
        )

        self.rest_max_runs_field = ft.TextField(
            label="Number of rest runs",
            value=str(config["scan"]["rest_max_runs"]),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("rest_max_runs", self.rest_max_runs_field, validate_positive_int),
        )

        self.collection_speed_field = ft.TextField(
            label="Simulator speed (collection, × realtime)",
            value=str(config["scan"]["collection_speed"]),
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        # Network fields
        self.n_yeo_radio = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="7", label="7 Networks"),
                ft.Radio(value="17", label="17 Networks"),
            ]),
            value=str(config["mask"]["n_yeo"]),
        )

        # Yeo network checkboxes
        self.yeo_checkboxes = self._create_yeo_checkboxes(config["mask"]["n_yeo"], config["mask"]["yeo_targets"])

        self.smooth_fwhm_field = ft.TextField(
            label="Spatial smoothing (mm FWHM)",
            value=str(config["mask"]["smooth_fwhm"]),
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        self.highpass_field = ft.TextField(
            label="Highpass filter (Hz)",
            value=str(config["mask"]["highpass"]),
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        # Task type selection (populated from registry)
        available_tasks = list_available_tasks()
        self.task_type_dropdown = ft.Dropdown(
            label="Task Type",
            options=[
                ft.dropdown.Option(
                    key=task_id,
                    text=metadata.display_name,
                )
                for task_id, metadata in available_tasks.items()
            ],
            value=config["task"].get("task_type", "ball"),
        )
        # Set event handler as property (not in constructor)
        self.task_type_dropdown.on_change = self._on_task_type_changed

        self.max_feedback_runs_field = ft.TextField(
            label="Number of feedback runs",
            value=str(config["task"]["max_feedback_runs"]),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Total feedback runs (transfer runs are optional)",
        )

        # Create dynamic task parameter fields for current task type
        self._create_task_parameter_fields()

        # Paths fields
        self.base_dir_field = ft.TextField(
            label="Base data directory",
            value=config["paths"]["base_dir"],
        )

        # Simulator source paths
        self.rest_source_1_field = ft.TextField(
            label="Rest run 1 source DICOMs",
            value=config["scan"]["rest_source_dicoms"][0] if len(config["scan"]["rest_source_dicoms"]) > 0 else "",
            hint_text="Path to DICOM directory for simulator (optional)",
        )

        self.rest_source_2_field = ft.TextField(
            label="Rest run 2 source DICOMs",
            value=config["scan"]["rest_source_dicoms"][1] if len(config["scan"]["rest_source_dicoms"]) > 1 else "",
            hint_text="Path to DICOM directory for simulator (optional)",
        )

        self.twovol_initial_field = ft.TextField(
            label="Initial 2vol source DICOMs",
            value=config["scan"]["twovol_initial_dicoms"],
            hint_text="Path to DICOM directory for simulator (optional)",
        )

        self.twovol_realtime_field = ft.TextField(
            label="Real-time 2vol source DICOMs",
            value=config["scan"]["twovol_realtime_dicoms"],
            hint_text="Path to DICOM directory for simulator (optional)",
        )

        self.feedback_source_field = ft.TextField(
            label="Feedback runs source DICOMs",
            value=config["scan"]["feedback_dicoms"],
            hint_text="Path to DICOM directory for simulator (optional)",
        )

    def _create_yeo_checkboxes(self, n_yeo: int, selected_targets: list) -> ft.Column:
        """Create Yeo network checkboxes based on atlas size.

        Args:
            n_yeo: 7 or 17
            selected_targets: Currently selected network names

        Returns:
            Column of checkboxes
        """
        from ..utils import YEO_7_LABELS, YEO_17_LABELS

        labels = YEO_7_LABELS if n_yeo == 7 else YEO_17_LABELS

        checkboxes = []
        for label in labels:
            checkbox = ft.Checkbox(
                label=label,
                value=label in selected_targets,
            )
            checkboxes.append(checkbox)

        return ft.Column(checkboxes, scroll=ft.ScrollMode.AUTO, height=300)

    def _validate_field(self, field_name: str, field: ft.TextField, validator):
        """Validate a form field and update UI.

        Args:
            field_name: Field identifier
            field: TextField control
            validator: Validation function
        """
        try:
            validator(field.value, field_name)
            field.error_text = None
            if field_name in self.validation_errors:
                del self.validation_errors[field_name]
        except ValidationError as e:
            field.error_text = str(e)
            self.validation_errors[field_name] = str(e)

        field.update()

    def _on_name_change(self, e):
        """Handle study name change."""
        if not self.name_field.value.strip():
            self.name_field.error_text = "Study name is required"
            self.validation_errors["name"] = "Study name is required"
        else:
            self.name_field.error_text = None
            if "name" in self.validation_errors:
                del self.validation_errors["name"]

        self.name_field.update()

    def _create_task_parameter_fields(self):
        """Create form fields for current task type's parameters."""
        task_type = self.task_type_dropdown.value
        metadata = get_task_metadata(task_type)

        config = self.study_config.session_config
        task_config = config["task"].get(task_type, {})

        self.task_param_fields.clear()

        for param in metadata.parameters:
            field = self._create_field_for_parameter(param, task_config)
            self.task_param_fields[param.key] = field

    def _create_field_for_parameter(self, param: TaskParameter, task_config: dict) -> ft.Control:
        """Create appropriate Flet control for a parameter.

        Args:
            param: TaskParameter definition
            task_config: Task-specific config dictionary

        Returns:
            Flet control for the parameter
        """
        current_value = task_config.get(param.key, param.default)

        if param.type == "bool":
            return ft.Checkbox(
                label=param.label,
                value=current_value,
            )

        elif param.type in ["float", "int"]:
            return ft.TextField(
                label=param.label,
                value=str(current_value),
                keyboard_type=ft.KeyboardType.NUMBER,
                hint_text=param.hint,
            )

        elif param.type == "str":
            return ft.TextField(
                label=param.label,
                value=current_value,
                hint_text=param.hint,
            )

        elif param.type == "list[str]":
            # Convert list to comma-separated string
            value_str = ", ".join(current_value) if isinstance(current_value, list) else current_value
            return ft.TextField(
                label=param.label,
                value=value_str,
                hint_text=f"{param.hint} (comma-separated)" if param.hint else "Comma-separated values",
            )

    def _on_task_type_changed(self, e):
        """Handle task type dropdown change (dynamic tab rebuild)."""
        logger.info(f"Task type changed to: {self.task_type_dropdown.value}")

        # Recreate task parameter fields for new task type
        self._create_task_parameter_fields()

        # Rebuild the task tab dynamically
        # Find the TabsWidget and update its content
        # For now, show a message that changes will appear on save
        self.page.snack_bar = ft.SnackBar(
            ft.Text(f"Task type changed to {self.task_type_dropdown.value}. Task parameters will update when you reopen this study."),
            open=True,
        )
        self.page.update()

    def _create_browse_button(self, field: ft.TextField) -> ft.IconButton:
        """Create a browse button for directory selection.

        Args:
            field: TextField to update with selected directory

        Returns:
            IconButton that opens directory picker
        """
        def pick_directory(e):
            """Handle directory picker click event."""
            logger.info(f"Directory picker clicked for field: {field.label}")

            # Use tkinter file dialog as a workaround for Flet FilePicker issues on Linux
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                logger.error("tkinter is not available")
                self.page.snack_bar = ft.SnackBar(
                    ft.Text("File browser requires tkinter. Install: sudo apt-get install python3-tk"),
                    open=True,
                )
                self.page.update()
                return

            try:
                # Create a temporary tkinter root window
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.attributes('-topmost', True)  # Bring to front

                # Open directory picker
                path = filedialog.askdirectory(
                    title=f"Select {field.label}",
                    initialdir=field.value if field.value else None
                )

                # Clean up
                root.destroy()

                logger.info(f"Selected path: {path}")
                if path:
                    field.value = path
                    field.update()

            except Exception as ex:
                logger.error(f"Error opening directory picker: {ex}")
                # Show error to user
                self.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Error opening file browser: {ex}"),
                    open=True,
                )
                self.page.update()

        return ft.IconButton(
            icon=ft.icons.Icons.FOLDER_OPEN,
            tooltip="Browse for folder",
            on_click=pick_directory,
        )

    def build(self) -> ft.Control:
        """Build the study editor UI.

        Returns:
            Flet control for the study editor
        """
        # Create tab bar (headers)
        tab_bar = ft.TabBar(
            tabs=[
                ft.Tab(label="Metadata & Scanner"),
                ft.Tab(label="Networks"),
                ft.Tab(label="Task"),
                ft.Tab(label="File Paths"),
            ],
        )

        # Create tab content panels
        tab_view = ft.TabBarView(
            controls=[
                self._build_scanner_tab(),
                self._build_networks_tab(),
                self._build_task_tab(),
                self._build_paths_tab(),
            ],
            expand=True,
        )

        # Wrap TabBar and TabBarView in a Column for Tabs
        tabs_content = ft.Column(
            [tab_bar, tab_view],
            spacing=0,
            expand=True,
        )

        # Create Tabs control
        tabs = ft.Tabs(
            content=tabs_content,
            length=4,  # Number of tabs
        )

        # Create action buttons
        action_buttons = ft.Row(
            [
                ft.TextButton("Cancel", on_click=self._on_cancel),
                ft.Row(
                    [
                        ft.ElevatedButton("Save", on_click=self._on_save),
                        ft.ElevatedButton("Save As", on_click=self._on_save_as),
                        ft.FilledButton(
                            "Start Session",
                            icon=ft.icons.Icons.PLAY_ARROW,
                            on_click=self._on_start_session,
                            disabled=self.is_new,  # Disable for new unsaved studies
                        ),
                    ],
                    spacing=5,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(
                                f"Study Editor - {self.study_config.metadata.name}",
                                size=24,
                                weight=ft.FontWeight.BOLD,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    padding=20,
                    bgcolor=ft.Colors.BLUE_50,
                ),
                # Tabs (contains TabBar and TabBarView)
                ft.Container(
                    content=tabs,
                    padding=20,
                    expand=True,
                ),
                # Actions
                ft.Container(
                    content=action_buttons,
                    padding=20,
                    bgcolor=ft.Colors.GREY_100,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _build_scanner_tab(self) -> ft.Container:
        """Build the scanner parameters tab."""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Study Information", size=16, weight=ft.FontWeight.BOLD),
                    self.name_field,
                    self.description_field,
                    self.example_participant_id_field,
                    ft.Divider(),
                    ft.Text("Scanner Parameters", size=16, weight=ft.FontWeight.BOLD),
                    self.tr_field,
                    self.n_volumes_field,
                    self.n_baseline_field,
                    self.rest_max_runs_field,
                    ft.Divider(),
                    ft.Text("Simulator Settings", size=16, weight=ft.FontWeight.BOLD),
                    self.collection_speed_field,
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    def _build_networks_tab(self) -> ft.Container:
        """Build the networks parameters tab."""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Yeo Atlas", size=16, weight=ft.FontWeight.BOLD),
                    self.n_yeo_radio,
                    ft.Divider(),
                    ft.Text("Target Networks (select 2+)", size=16, weight=ft.FontWeight.BOLD),
                    self.yeo_checkboxes,
                    ft.Divider(),
                    ft.Text("Preprocessing", size=16, weight=ft.FontWeight.BOLD),
                    self.smooth_fwhm_field,
                    self.highpass_field,
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    def _build_task_tab(self) -> ft.Container:
        """Build the task parameters tab."""
        # Common task configuration (all tasks)
        common_section = [
            ft.Text("Task Selection", size=16, weight=ft.FontWeight.BOLD),
            self.task_type_dropdown,
            ft.Divider(),
            ft.Text("Session Structure", size=16, weight=ft.FontWeight.BOLD),
            self.max_feedback_runs_field,
            ft.Text(
                "Note: Transfer runs are optional (configure in task-specific parameters below)",
                size=12,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            ft.Divider(),
        ]

        # Task-specific parameters
        task_specific_section = self._build_task_specific_params()

        return ft.Container(
            content=ft.Column(
                common_section + task_specific_section,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    def _build_task_specific_params(self) -> list:
        """Build task-specific parameter controls dynamically from task metadata.

        Returns:
            List of Flet controls for the current task type
        """
        task_type = self.task_type_dropdown.value

        try:
            metadata = get_task_metadata(task_type)
        except ValueError:
            return [
                ft.Text(
                    f"Unknown task type: {task_type}",
                    size=14,
                    color=ft.Colors.ORANGE_700,
                ),
            ]

        # Build controls from task metadata
        controls = [
            ft.Text(
                f"{metadata.display_name} Parameters",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            ft.Text(
                metadata.description,
                size=12,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            ft.Divider(),
        ]

        # Add each parameter field
        for param in metadata.parameters:
            field = self.task_param_fields.get(param.key)
            if field is not None:
                controls.append(field)

                # Add hint text if provided
                if param.hint and param.type != "bool":  # Bools show label, not separate hint
                    controls.append(
                        ft.Text(
                            param.hint,
                            size=11,
                            color=ft.Colors.GREY_600,
                            italic=True,
                        )
                    )

        return controls

    def _build_paths_tab(self) -> ft.Container:
        """Build the file paths tab."""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Output Directory", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        self.base_dir_field,
                        self._create_browse_button(self.base_dir_field),
                    ]),
                    ft.Divider(),
                    ft.Text("Simulator Source Paths (Optional)", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Configure DICOM source directories for offline testing with the simulator",
                        size=12,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Row([
                        self.rest_source_1_field,
                        self._create_browse_button(self.rest_source_1_field),
                    ]),
                    ft.Row([
                        self.rest_source_2_field,
                        self._create_browse_button(self.rest_source_2_field),
                    ]),
                    ft.Row([
                        self.twovol_initial_field,
                        self._create_browse_button(self.twovol_initial_field),
                    ]),
                    ft.Row([
                        self.twovol_realtime_field,
                        self._create_browse_button(self.twovol_realtime_field),
                    ]),
                    ft.Row([
                        self.feedback_source_field,
                        self._create_browse_button(self.feedback_source_field),
                    ]),
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    def _on_cancel(self, e):
        """Handle cancel button click."""
        logger.info("Study editor cancelled")
        if self.on_cancel_callback:
            self.on_cancel_callback()

    def _on_save(self, e):
        """Handle save button click."""
        if self.validation_errors:
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Please fix validation errors before saving"),
                open=True,
            )
            self.page.update()
            return

        # Update study config from form fields
        self._update_study_from_form()

        # Save
        try:
            path = self.config_manager.save_study(self.study_config, overwrite=True)
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Study saved: {self.study_config.metadata.name}"),
                open=True,
            )
            self.page.update()

            logger.info(f"Study saved: {path}")

            # Navigate back
            if self.on_save_callback:
                self.on_save_callback()

        except Exception as ex:
            logger.error(f"Failed to save study: {ex}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Error saving study: {ex}"),
                open=True,
            )
            self.page.update()

    def _on_save_as(self, e):
        """Handle save as button click."""
        logger.info("Save as clicked")

        # Create text field for new name
        new_name_field = ft.TextField(
            label="New study name",
            value=f"{self.study_config.metadata.name} (Copy)",
            autofocus=True,
        )

        # Create dialog first so it's accessible in nested functions
        dialog = ft.AlertDialog(
            title=ft.Text("Save Study As"),
            content=ft.Container(
                content=new_name_field,
                width=400,
            ),
        )

        def save_with_new_name(e):
            new_name = new_name_field.value.strip()

            if not new_name:
                new_name_field.error_text = "Study name is required"
                new_name_field.update()
                return

            # Check if name already exists
            existing_studies = self.config_manager.list_studies()
            if any(s.name == new_name for s in existing_studies):
                new_name_field.error_text = "A study with this name already exists"
                new_name_field.update()
                return

            # Update study config from form fields
            self._update_study_from_form()

            # Update metadata with new name
            self.study_config.metadata.name = new_name

            # Clear file_path so save_study generates new path from new name
            self.study_config.file_path = None

            # Save as new study
            try:
                path = self.config_manager.save_study(self.study_config, overwrite=False)
                self.page.pop_dialog()

                self.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Study saved as: {new_name}"),
                    open=True,
                )
                self.page.update()

                logger.info(f"Study saved as: {path}")

                # Navigate back
                if self.on_save_callback:
                    self.on_save_callback()

            except Exception as ex:
                logger.error(f"Failed to save study: {ex}")
                new_name_field.error_text = str(ex)
                new_name_field.update()

        def cancel_save_as(e):
            self.page.pop_dialog()

        # Add actions to dialog
        dialog.actions = [
            ft.TextButton("Cancel", on_click=cancel_save_as),
            ft.ElevatedButton("Save", on_click=save_with_new_name),
        ]

        self.page.show_dialog(dialog)

    def _on_start_session(self, e):
        """Handle start session button click."""
        logger.info("Start Session clicked")

        # Get example participant ID from current form value (not saved config)
        example_id = self.example_participant_id_field.value.strip() or "sub-001"

        # Create text field for participant ID
        participant_id_field = ft.TextField(
            label="Participant ID",
            hint_text=f"e.g., {example_id}",
            autofocus=True,
        )

        # Create dialog
        dialog = ft.AlertDialog(
            title=ft.Text("Start Participant Session"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            f"Study: {self.study_config.metadata.name}",
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
        )

        def start_session(e):
            participant_id = participant_id_field.value.strip()

            if not participant_id:
                participant_id_field.error_text = "Participant ID is required"
                participant_id_field.update()
                return

            self.page.pop_dialog()

            logger.info(f"Starting session for participant: {participant_id}")

            # Call the start session callback
            if self.on_start_session_callback:
                self.on_start_session_callback(participant_id)

        def cancel_start(e):
            self.page.pop_dialog()

        # Add actions to dialog
        dialog.actions = [
            ft.TextButton("Cancel", on_click=cancel_start),
            ft.FilledButton("Start", icon=ft.icons.Icons.PLAY_ARROW, on_click=start_session),
        ]

        self.page.show_dialog(dialog)

    def _update_study_from_form(self):
        """Update study config from form field values.

        Raises:
            ValidationError: If any field contains invalid data
        """
        # Update metadata
        self.study_config.metadata.name = self.name_field.value.strip()
        self.study_config.metadata.description = self.description_field.value.strip()

        # Update scanner config with validation
        config = self.study_config.session_config

        # Update example participant ID
        config["paths"]["example_participant_id"] = self.example_participant_id_field.value.strip() or "sub-001"

        try:
            config["scan"]["tr"] = float(self.tr_field.value)
            config["scan"]["n_volumes"] = int(self.n_volumes_field.value)
            config["scan"]["n_baseline"] = int(self.n_baseline_field.value)
            config["scan"]["rest_max_runs"] = int(self.rest_max_runs_field.value)
            config["scan"]["collection_speed"] = float(self.collection_speed_field.value)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid scanner parameter: {e}")
            raise ValidationError(f"Invalid scanner parameter: {e}")

        # Update network config with validation
        try:
            config["mask"]["n_yeo"] = int(self.n_yeo_radio.value)
            config["mask"]["smooth_fwhm"] = float(self.smooth_fwhm_field.value)
            config["mask"]["highpass"] = float(self.highpass_field.value)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid network parameter: {e}")
            raise ValidationError(f"Invalid network parameter: {e}")

        # Update selected networks
        selected_networks = [
            checkbox.label for checkbox in self.yeo_checkboxes.controls
            if checkbox.value
        ]
        config["mask"]["yeo_targets"] = selected_networks
        config["mask"]["mask_names"] = selected_networks

        # Update task config with validation
        task_type = self.task_type_dropdown.value
        config["task"]["task_type"] = task_type

        try:
            config["task"]["max_feedback_runs"] = int(self.max_feedback_runs_field.value)

            # Update task-specific parameters dynamically
            metadata = get_task_metadata(task_type)

            # Initialize task config if missing
            if task_type not in config["task"]:
                config["task"][task_type] = {}

            # Update each parameter from dynamic fields
            for param in metadata.parameters:
                field = self.task_param_fields.get(param.key)
                if field is None:
                    continue

                if param.type == "bool":
                    config["task"][task_type][param.key] = field.value
                elif param.type == "int":
                    config["task"][task_type][param.key] = int(field.value)
                elif param.type == "float":
                    config["task"][task_type][param.key] = float(field.value)
                elif param.type == "str":
                    config["task"][task_type][param.key] = field.value
                elif param.type == "list[str]":
                    # Parse comma-separated string to list
                    config["task"][task_type][param.key] = [
                        s.strip() for s in field.value.split(",") if s.strip()
                    ]

            # Validate task config
            metadata.validate_config(config["task"][task_type])

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid task parameter: {e}")
            raise ValidationError(f"Invalid task parameter: {e}")

        # Update paths
        config["paths"]["base_dir"] = self.base_dir_field.value.strip()

        # Update simulator source paths
        config["scan"]["rest_source_dicoms"] = [
            self.rest_source_1_field.value.strip(),
            self.rest_source_2_field.value.strip(),
        ]
        config["scan"]["twovol_initial_dicoms"] = self.twovol_initial_field.value.strip()
        config["scan"]["twovol_realtime_dicoms"] = self.twovol_realtime_field.value.strip()
        config["scan"]["feedback_dicoms"] = self.feedback_source_field.value.strip()
