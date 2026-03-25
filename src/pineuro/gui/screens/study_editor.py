"""Study Editor screen for creating and editing study configurations.

This module provides a tabbed interface for configuring all aspects of a pineuro study.
Localizer and feedback tabs are created dynamically based on user-specified counts.
"""

import contextlib
import logging

import flet as ft

from pineuro.tasks.task_registry import get_task_metadata, list_available_tasks

from ..config_manager import ConfigManager
from ..models import StudyConfig, StudyMetadata
from ._feedback_tab import FeedbackTabBuilder
from ._localizer_tab import LocalizerTabBuilder

logger = logging.getLogger(__name__)


class StudyEditorScreen:
    """Study editor screen with dynamic tabbed parameter sections."""

    FIELD_W = 400
    PAIR_W = 195

    @staticmethod
    def _f(control):
        """Wrap a control in a Row so Column doesn't stretch it."""
        return ft.Row([control])

    def __init__(
        self,
        page: ft.Page,
        config_manager: ConfigManager,
        study_config: StudyConfig | None = None,
        on_cancel=None,
        on_save=None,
        on_start_session=None,
    ):
        self.page = page
        self.config_manager = config_manager
        self.is_new = study_config is None
        self.on_cancel_callback = on_cancel
        self.on_save_callback = on_save
        self.on_start_session_callback = on_start_session

        self.study_config: StudyConfig = study_config if study_config is not None else self._create_default_config()
        self.validation_errors: dict[str, str] = {}

        # Initialize localizer and feedback configs from session_config
        self._init_localizer_configs()
        self._init_feedback_configs()

        # Create Study tab and Paths tab fields (non-dynamic)
        self._create_study_fields()
        self._create_paths_fields()

    # ------------------------------------------------------------------
    # Config initialization
    # ------------------------------------------------------------------

    def _init_localizer_configs(self):
        """Initialize localizer config list from session_config."""
        config = self.study_config.session_config
        self._localizer_configs = config.get("localizers", [])

    def _init_feedback_configs(self):
        """Initialize feedback run config list from session_config."""
        config = self.study_config.session_config
        self._feedback_configs = config.get("feedback_runs", [])

    # ------------------------------------------------------------------
    # Default config
    # ------------------------------------------------------------------

    def _create_default_config(self) -> StudyConfig:
        metadata = StudyMetadata(name="New Study", description="")
        session_config = {
            "paths": {
                "participant_id": "",
                "base_dir": "/data/pineuro",
                "example_participant_id": "sub-001",
            },
            "scan": {
                "localizer_tr": 1.2,
                "localizer_n_volumes": 200,
                "localizer_n_runs": 2,
                "localizer_task_type": "rest",
                "feedback_tr": 1.2,
                "feedback_n_volumes": 150,
                "reference_initial_enabled": True,
                "reference_initial_n_volumes": 2,
                "reference_realtime_enabled": True,
                "reference_realtime_n_volumes": 2,
                "reference_initial_dicoms": "",
                "reference_realtime_dicoms": "",
                "localizer_source_dicoms": ["", ""],
                "feedback_source_dicoms": ["", ""],
                "transfer_source_dicoms": ["", ""],
                "simulator_source_mapping": "",
                "simulator_enabled": True,
                "collection_speed": 10.0,
                "feedback_simulator_speed": 1.0,
                "scanner_dicom_dir": "/BOLD_DICOM",
                "scanner_ip": "",
            },
            "mask": {
                "n_yeo": 17,
                "network_atlas": "yeo_17",
                "yeo_targets": ["ContA", "DefaultA"],
                "n_components": 20,
                "smooth_fwhm": 5.0,
                "highpass": 0.01,
                "lowpass": None,
                "mask_names": ["ContA", "DefaultA"],
                "threshold": 2000,
                "threshold_method": "top_n",
                "atlas_name": None,
                "atlas_mask": None,
                "atlas_source": None,
                "atlas_regions": None,
                "cluster_start_threshold": 2.0,
                "cluster_step": 0.1,
                "cluster_return_all": False,
                "min_cluster_size": 50,
                "max_total_voxels": None,
            },
            "feedback": {
                "n_baseline": 25,
                "n_volumes": 150,
                "use_efficiency_weights": True,
                "fd_threshold": 0.5,
                "dvars_threshold": 1.5,
                "tsnr_threshold": 20,
                "motion_correction": "siemens_moco",
                "include_drift": True,
                "include_motion_regressors": True,
                "include_wm_regressor": False,
                "include_csf_regressor": False,
                "smooth_fwhm": None,
                "differential_pair": None,
                "feedback_interval": None,
                "connectivity_pairs": None,
                "connectivity_window": 15,
                "feedback_mode": "activation",
                "connectivity_target": "increase",
            },
            "localizer_task": {},
            "task": {
                "task_type": "ball",
                "max_feedback_runs": 1,
            },
            "sham": {
                "enabled": False,
                "blind_mode": False,
                "delivery_method": "alternative_masks",
                "sham_mask_mode": "hemisphere_split",
                "hemisphere_network": "SomMotA",
                "target_hemisphere": "right",
                "alternative_mask_targets": ["SomMotA", "DefaultA"],
                "randomization_file": "",
            },
            "display": {
                "control_monitor": "left",
                "presenter_monitor": "right",
            },
        }
        return StudyConfig(metadata=metadata, session_config=session_config)

    # ------------------------------------------------------------------
    # Study tab fields
    # ------------------------------------------------------------------

    def _create_study_fields(self):
        """Create form fields for the Study tab."""
        config = self.study_config.session_config

        self.name_field = ft.TextField(
            width=600,
            label="Study Name",
            value=self.study_config.metadata.name,
            hint_text="Unique name shown in the study manager",
            on_change=self._on_name_change,
        )
        self.description_field = ft.TextField(
            width=600,
            label="Description",
            value=self.study_config.metadata.description,
            hint_text="Brief summary shown on the study card",
            multiline=True,
            min_lines=2,
            max_lines=4,
        )
        self.example_participant_id_field = ft.TextField(
            width=600,
            label="Example Participant ID",
            value=config["paths"].get("example_participant_id", "sub-001"),
            hint_text="Shown as hint when starting a session (e.g., sub-001)",
        )

        # Multi-visit
        self.multi_visit_enabled_checkbox = ft.Checkbox(
            label="Enable Multiple Visits",
            value=config.get("study", {}).get("multi_visit_enabled", False),
        )

        # Simulator mode
        self.simulator_enabled_checkbox = ft.Checkbox(
            label="Simulator Mode",
            value=config["scan"].get("simulator_enabled", True),
            on_change=self._on_simulator_mode_changed,
        )
        self.collection_speed_field = ft.TextField(
            width=290,
            label="Localizer speed (× realtime)",
            value=str(config["scan"].get("collection_speed", 10.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Speed for localizer/setup runs",
        )
        self.feedback_simulator_speed_field = ft.TextField(
            width=290,
            label="Feedback speed (× realtime)",
            value=str(config["scan"].get("feedback_simulator_speed", 1.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Speed for feedback runs",
        )

        # Scanner settings
        self.scanner_dicom_dir_field = ft.TextField(
            width=540,
            label="Scanner DICOM Directory",
            value=config["scan"].get("scanner_dicom_dir", "/BOLD_DICOM"),
            hint_text="Directory where scanner exports DICOMs",
        )
        self.scanner_ip_field = ft.TextField(
            width=290,
            label="Scanner IP Address",
            value=config["scan"].get("scanner_ip", ""),
            hint_text="e.g., 192.168.2.1",
        )
        self.scanner_setup_guide_btn = ft.OutlinedButton(
            "Setup Guide",
            icon=ft.Icons.HELP_OUTLINE,
            on_click=lambda e: self._show_scanner_setup_guide(),
        )

        # Reference scan settings (session-level)
        self.reference_initial_enabled_checkbox = ft.Checkbox(
            label="Collect initial reference scan (before localizers)",
            value=config["scan"].get("reference_initial_enabled", True),
            on_change=self._on_reference_scan_changed,
        )
        self.reference_initial_n_volumes_field = ft.TextField(
            width=160,
            label="Volumes",
            value=str(config["scan"].get("reference_initial_n_volumes", 2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=config["scan"].get("reference_initial_enabled", True),
        )
        self.reference_realtime_enabled_checkbox = ft.Checkbox(
            label="Collect real-time reference scan (before feedback)",
            value=config["scan"].get("reference_realtime_enabled", True),
            on_change=self._on_reference_scan_changed,
        )
        self.reference_realtime_n_volumes_field = ft.TextField(
            width=160,
            label="Volumes",
            value=str(config["scan"].get("reference_realtime_n_volumes", 2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=config["scan"].get("reference_realtime_enabled", True),
        )

        # Display settings
        display_config = config.get("display", {})
        self.control_monitor_dropdown = ft.Dropdown(
            width=290,
            label="Control Monitor (GUI)",
            options=[
                ft.dropdown.Option(key="left", text="Left"),
                ft.dropdown.Option(key="right", text="Right"),
            ],
            value=display_config.get("control_monitor", "left"),
        )
        self.presenter_monitor_dropdown = ft.Dropdown(
            width=290,
            label="Presenter Monitor (Task)",
            options=[
                ft.dropdown.Option(key="left", text="Left"),
                ft.dropdown.Option(key="right", text="Right"),
            ],
            value=display_config.get("presenter_monitor", "right"),
        )

        # Type dropdown options (lazily computed — avoid triggering heavy task module imports)
        self._localizer_type_options_cache: list[ft.dropdown.Option] | None = None
        self._feedback_type_options_cache: list[ft.dropdown.Option] | None = None

    @property
    def _localizer_type_options(self) -> list[ft.dropdown.Option]:
        if self._localizer_type_options_cache is None:
            opts = [ft.dropdown.Option(key="rest", text="Rest (ICA Network Extraction)")]
            available_loc_tasks = list_available_tasks(role="localizer")
            opts.extend(
                ft.dropdown.Option(key=task_id, text=f"{meta.display_name} (GLM Contrast)")
                for task_id, meta in available_loc_tasks.items()
                if meta.default_contrasts
            )
            self._localizer_type_options_cache = opts
        return self._localizer_type_options_cache

    @property
    def _feedback_type_options(self) -> list[ft.dropdown.Option]:
        if self._feedback_type_options_cache is None:
            feedback_tasks = list_available_tasks(role="feedback")
            self._feedback_type_options_cache = [
                ft.dropdown.Option(key=task_id, text=meta.display_name) for task_id, meta in feedback_tasks.items()
            ]
        return self._feedback_type_options_cache

    def _create_paths_fields(self):
        """Create form fields for the Paths tab."""
        config = self.study_config.session_config

        self.base_dir_field = ft.TextField(
            width=540,
            label="Base data directory",
            value=config["paths"]["base_dir"],
            hint_text="Root directory for participant data (e.g., /data/pineuro)",
        )

        # Simulator source mode toggle
        has_mapping = bool(config["scan"].get("simulator_source_mapping", "").strip())
        self.simulator_source_mode_radio = ft.RadioGroup(
            content=ft.Row(
                [
                    ft.Radio(value="individual", label="Same paths for all participants"),
                    ft.Radio(value="mapping", label="Per-subject mapping file"),
                ]
            ),
            value="mapping" if has_mapping else "individual",
            on_change=self._on_simulator_source_mode_changed,
        )

        self.simulator_source_mapping_field = ft.TextField(
            width=540,
            label="Mapping File",
            value=config["scan"].get("simulator_source_mapping", ""),
            hint_text="CSV/TSV file mapping subjects to DICOM paths",
        )

    # ------------------------------------------------------------------
    # Tab builders management
    # ------------------------------------------------------------------

    def _create_tab_builders(self):
        """Create LocalizerTabBuilder and FeedbackTabBuilder instances."""
        # Create localizer builders
        self._localizer_builders: list[LocalizerTabBuilder] = []
        for i, cfg in enumerate(self._localizer_configs):
            builder = LocalizerTabBuilder(
                index=i,
                config=cfg,
                page=self.page,
                on_change=self._on_localizer_changed,
                create_browse_button=self._create_browse_button,
                create_file_browse_button=self._create_file_browse_button,
            )
            self._localizer_builders.append(builder)

        # Aggregate mask names from all localizers
        all_masks = self._aggregate_mask_names()

        # Create feedback builders
        self._feedback_builders: list[FeedbackTabBuilder] = []
        for i, cfg in enumerate(self._feedback_configs):
            builder = FeedbackTabBuilder(
                index=i,
                config=cfg,
                page=self.page,
                on_change=self._on_feedback_changed,
                available_mask_names=all_masks,
                create_browse_button=self._create_browse_button,
                create_file_browse_button=self._create_file_browse_button,
            )
            self._feedback_builders.append(builder)

    def _aggregate_mask_names(self) -> list[str]:
        """Aggregate mask names from all localizer builders."""
        names: list[str] = []
        for builder in self._localizer_builders:
            names.extend(builder.get_mask_names())
        return names

    def _on_localizer_changed(self):
        """Called when any localizer tab changes networks/type."""
        all_masks = self._aggregate_mask_names()
        for fb in self._feedback_builders:
            fb.update_available_masks(all_masks)
        self._sync_paths_tab_visibility()

    def _on_feedback_changed(self):
        """Called when any feedback tab changes."""
        self._sync_paths_tab_visibility()

    def _sync_paths_tab_visibility(self):
        """Update visibility of Paths tab simulator fields based on Study tab checkboxes."""
        # Initial reference visibility
        ref_row = getattr(self, "_sim_loc_ref_row", None)
        if ref_row is not None:
            enabled = self.reference_initial_enabled_checkbox.value
            ref_row.visible = enabled
            ref_row.update()
            ref_label = getattr(self, "_sim_loc_ref_label", None)
            if ref_label is not None:
                ref_label.visible = enabled
                ref_label.update()

        # Realtime reference visibility
        rt_ref_row = getattr(self, "_sim_fb_ref_row", None)
        if rt_ref_row is not None:
            enabled = self.reference_realtime_enabled_checkbox.value
            rt_ref_row.visible = enabled
            rt_ref_row.update()
            rt_ref_label = getattr(self, "_sim_fb_ref_label", None)
            if rt_ref_label is not None:
                rt_ref_label.visible = enabled
                rt_ref_label.update()

    # ------------------------------------------------------------------
    # Count control handlers
    # ------------------------------------------------------------------

    def _on_add_localizer(self, e):
        """Add a new localizer tab."""
        self._collect_all_configs()
        self._localizer_configs.append(
            {
                "type": "rest",
                "tr": 1.2,
                "n_volumes": 200,
                "n_runs": 2,
                "network_atlas": "yeo_17",
                "n_yeo": 17,
                "yeo_targets": ["DefaultA", "ContA"],
                "n_components": 20,
                "smooth_fwhm": 5.0,
                "highpass": 0.01,
                "threshold": 2000,
                "threshold_method": "top_n",
                "source_dicoms": [],
            }
        )
        self._rebuild_tabs()

    def _on_remove_localizer_at(self, index: int):
        """Remove localizer at given index."""
        self._collect_all_configs()
        self._localizer_configs.pop(index)
        self._rebuild_tabs()

    def _on_localizer_type_changed(self, index: int, e):
        """Handle localizer type change from Study tab overview."""
        self._collect_all_configs()
        new_type = self._loc_type_dropdowns[index].value or "rest"
        old_type = self._localizer_configs[index].get("type", "rest")

        if new_type == old_type:
            return

        # Reset type-specific defaults when switching between rest and task
        cfg = self._localizer_configs[index]
        cfg["type"] = new_type
        if new_type == "rest" and old_type != "rest":
            # Switching to rest: set ICA defaults
            cfg.setdefault("network_atlas", "yeo_17")
            cfg.setdefault("n_components", 20)
            cfg.setdefault("smooth_fwhm", 5.0)
            cfg.setdefault("highpass", 0.01)
            cfg.setdefault("yeo_targets", ["DefaultA", "ContA"])
        elif new_type != "rest":
            # Switching to task: set contrast defaults from task metadata
            try:
                meta = get_task_metadata(new_type)
                if meta.default_contrasts:
                    cfg["contrasts"] = dict(meta.default_contrasts)
            except (KeyError, ValueError):
                pass
            cfg["include_reverse_contrasts"] = cfg.get("include_reverse_contrasts", False)

        self._rebuild_tabs()

    def _on_add_feedback(self, e):
        """Add a new feedback task tab."""
        self._collect_all_configs()
        template = {"task_type": "ball", "tr": 1.2, "n_volumes": 150, "n_baseline": 25, "max_runs": 1}
        self._feedback_configs.append(template)
        self._rebuild_tabs()

    def _on_remove_feedback_at(self, index: int):
        """Remove feedback task at given index."""
        self._collect_all_configs()
        self._feedback_configs.pop(index)
        self._rebuild_tabs()

    def _on_feedback_type_changed(self, index: int, e):
        """Handle feedback task type change from Study tab overview."""
        self._collect_all_configs()
        new_type = self._fb_type_dropdowns[index].value
        old_type = self._feedback_configs[index].get("task_type", "ball")

        if new_type == old_type:
            return

        cfg = self._feedback_configs[index]
        cfg["task_type"] = new_type
        # Clear old task-specific params since they're for the old task type
        cfg.pop("task_params", None)
        self._rebuild_tabs()

    def _collect_all_configs(self):
        """Collect current field values from all tab builders and Paths tab."""
        if hasattr(self, "_localizer_builders"):
            self._localizer_configs = [b.collect_config() for b in self._localizer_builders]
        if hasattr(self, "_feedback_builders"):
            self._feedback_configs = [b.collect_config() for b in self._feedback_builders]
        # Collect simulator paths from Paths tab into configs
        self._collect_simulator_paths()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ft.Control:
        """Build the study editor UI."""
        # Create tab builders
        self._create_tab_builders()

        # Build tabs
        self._build_tab_controls()

        # Action buttons
        action_buttons = ft.Row(
            [
                ft.TextButton("Cancel", on_click=self._on_cancel),
                ft.Row(
                    [
                        ft.ElevatedButton("Save", on_click=self._on_save),
                        ft.ElevatedButton("Save As", on_click=self._on_save_as),
                        ft.FilledButton(
                            "Start Session",
                            icon=ft.Icons.PLAY_ARROW,
                            on_click=self._on_start_session,
                            disabled=self.is_new,
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
                # Tabs
                ft.Container(
                    content=self.tabs_widget,
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

    def _get_localizer_tab_label(self, index: int) -> str:
        """Get tab label for a localizer, including short type name."""
        cfg = self._localizer_configs[index]
        loc_type = cfg.get("type", "rest")
        if loc_type == "rest":
            short = "Rest"
        else:
            try:
                meta = get_task_metadata(loc_type)
                short = meta.display_name
            except (KeyError, ValueError):
                short = loc_type
        n = len(self._localizer_configs)
        prefix = f"Localizer {index + 1}" if n > 1 else "Localizer"
        return f"{prefix} ({short})"

    def _get_feedback_tab_label(self, index: int) -> str:
        """Get tab label for a feedback task, including short type name."""
        cfg = self._feedback_configs[index]
        task_type = cfg.get("task_type", "ball")
        try:
            meta = get_task_metadata(task_type)
            short = meta.display_name
        except (KeyError, ValueError):
            short = task_type
        n = len(self._feedback_configs)
        prefix = f"Feedback {index + 1}" if n > 1 else "Feedback"
        return f"{prefix} ({short})"

    def _build_tab_controls(self):
        """Build the TabBar + TabBarView with dynamic tabs."""
        n_loc = len(self._localizer_configs)
        n_fb = len(self._feedback_configs)

        # Tab labels
        tabs = [ft.Tab(label="Study")]
        for i in range(n_loc):
            tabs.append(ft.Tab(label=self._get_localizer_tab_label(i)))
        for i in range(n_fb):
            tabs.append(ft.Tab(label=self._get_feedback_tab_label(i)))
        tabs.append(ft.Tab(label="Paths"))

        self.tab_bar = ft.TabBar(tabs=tabs)

        # Tab content
        content = [self._build_study_tab()]
        for builder in self._localizer_builders:
            content.append(builder.build())
        for builder in self._feedback_builders:
            content.append(builder.build())
        content.append(self._build_paths_tab())

        self.tab_view = ft.TabBarView(controls=content, expand=True)

        self.tabs_content = ft.Column(
            [self.tab_bar, self.tab_view],
            spacing=0,
            expand=True,
        )

        self.tabs_widget = ft.Tabs(
            content=self.tabs_content,
            length=len(tabs),
        )

    def _rebuild_tabs(self):
        """Rebuild all tabs after count or type change."""
        # Recreate builders from current configs
        self._create_tab_builders()

        n_loc = len(self._localizer_configs)
        n_fb = len(self._feedback_configs)

        # Rebuild tab labels
        tabs = [ft.Tab(label="Study")]
        for i in range(n_loc):
            tabs.append(ft.Tab(label=self._get_localizer_tab_label(i)))
        for i in range(n_fb):
            tabs.append(ft.Tab(label=self._get_feedback_tab_label(i)))
        tabs.append(ft.Tab(label="Paths"))

        self.tab_bar.tabs = tabs

        # Rebuild content
        content = [self._build_study_tab()]
        for builder in self._localizer_builders:
            content.append(builder.build())
        for builder in self._feedback_builders:
            content.append(builder.build())
        content.append(self._build_paths_tab())

        self.tab_view.controls = content

        # Update length
        self.tabs_widget.length = len(tabs)

        with contextlib.suppress(Exception):
            self.tab_bar.update()
            self.tab_view.update()
            self.page.update()

    # ------------------------------------------------------------------
    # Study tab
    # ------------------------------------------------------------------

    def _build_study_tab(self) -> ft.Container:
        """Build the study information tab."""
        self.scanner_dicom_dir_browse_btn = self._create_browse_button(self.scanner_dicom_dir_field)

        is_simulator = self.simulator_enabled_checkbox.value
        self.collection_speed_field.visible = is_simulator
        self.feedback_simulator_speed_field.visible = is_simulator
        self.scanner_dicom_dir_field.visible = not is_simulator
        self.scanner_dicom_dir_browse_btn.visible = not is_simulator
        self.scanner_ip_field.visible = not is_simulator
        self.scanner_setup_guide_btn.visible = not is_simulator

        self.simulator_instruction_text = ft.Text(
            "Simulator replays volumes from source directories at configurable speed for offline testing."
            if is_simulator
            else "Scanner DICOM directory will be monitored for real-time acquisition.",
            size=11,
            color=ft.Colors.GREY_600,
            italic=True,
        )

        # --- Experiment Design: Localizer Tasks ---
        loc_rows: list[ft.Control] = []
        self._loc_type_dropdowns: list[ft.Dropdown] = []
        for i, cfg in enumerate(self._localizer_configs):
            dd = ft.Dropdown(
                width=560,
                label=f"Localizer {i + 1}",
                options=[ft.dropdown.Option(key=o.key, text=o.text) for o in self._localizer_type_options],
                value=cfg.get("type", "rest"),
                on_select=lambda e, idx=i: self._on_localizer_type_changed(idx, e),
            )
            self._loc_type_dropdowns.append(dd)
            remove_btn = ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip="Remove this localizer",
                on_click=lambda e, idx=i: self._on_remove_localizer_at(idx),
            )
            loc_rows.append(ft.Row([dd, remove_btn], vertical_alignment=ft.CrossAxisAlignment.CENTER))

        add_loc_btn = ft.TextButton(
            "Add Localizer Task",
            icon=ft.Icons.ADD,
            on_click=self._on_add_localizer,
        )

        # --- Experiment Design: Feedback Tasks ---
        fb_rows: list[ft.Control] = []
        self._fb_type_dropdowns: list[ft.Dropdown] = []
        for i, cfg in enumerate(self._feedback_configs):
            dd = ft.Dropdown(
                width=560,
                label=f"Feedback {i + 1}",
                options=[ft.dropdown.Option(key=o.key, text=o.text) for o in self._feedback_type_options],
                value=cfg.get("task_type", "ball"),
                on_select=lambda e, idx=i: self._on_feedback_type_changed(idx, e),
            )
            self._fb_type_dropdowns.append(dd)
            remove_btn = ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip="Remove this feedback task",
                on_click=lambda e, idx=i: self._on_remove_feedback_at(idx),
            )
            fb_rows.append(ft.Row([dd, remove_btn], vertical_alignment=ft.CrossAxisAlignment.CENTER))

        add_fb_btn = ft.TextButton(
            "Add Feedback Task",
            icon=ft.Icons.ADD,
            on_click=self._on_add_feedback,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Study Information", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Study metadata shown in the GUI and used for organizing session data.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self._f(self.name_field),
                    self.description_field,
                    self.example_participant_id_field,
                    ft.Divider(),
                    ft.Text("Experiment Design", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Define localizer and feedback tasks. Each has its own configuration tab.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Text("Localizer Tasks", size=14, weight=ft.FontWeight.W_500),
                    *loc_rows,
                    add_loc_btn,
                    ft.Text("Feedback Tasks", size=14, weight=ft.FontWeight.W_500),
                    *fb_rows,
                    add_fb_btn,
                    ft.Divider(),
                    self.multi_visit_enabled_checkbox,
                    ft.Text(
                        "When enabled, participants can have multiple visits (ses-01, ses-02, etc.).",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Divider(),
                    ft.Text("Reference Scans", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Short scans used for spatial registration. Can be re-collected at any time during the session.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Row(
                        [
                            ft.Container(self.reference_initial_enabled_checkbox, width=420),
                            self.reference_initial_n_volumes_field,
                        ]
                    ),
                    ft.Row(
                        [
                            ft.Container(self.reference_realtime_enabled_checkbox, width=420),
                            self.reference_realtime_n_volumes_field,
                        ]
                    ),
                    ft.Divider(),
                    ft.Text("Acquisition Mode", size=16, weight=ft.FontWeight.BOLD),
                    self.simulator_enabled_checkbox,
                    ft.Row([self.collection_speed_field, self.feedback_simulator_speed_field]),
                    ft.Row([self.scanner_dicom_dir_field, self.scanner_dicom_dir_browse_btn]),
                    ft.Row(
                        [self.scanner_ip_field, self.scanner_setup_guide_btn],
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.simulator_instruction_text,
                    ft.Divider(),
                    ft.Text("Display Settings", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row([self.control_monitor_dropdown, self.presenter_monitor_dropdown]),
                    self._build_monitor_preview(),
                    ft.Text(
                        "Configure which monitor shows the control GUI vs the participant task display.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    # ------------------------------------------------------------------
    # Paths tab
    # ------------------------------------------------------------------

    def _build_paths_tab(self) -> ft.Container:
        """Build the file paths tab with simulator source path management."""
        is_simulator = self.simulator_enabled_checkbox.value
        is_mapping = self.simulator_source_mode_radio.value == "mapping"

        self.simulator_source_mapping_row = ft.Row(
            [
                self.simulator_source_mapping_field,
                self._create_file_browse_button(
                    self.simulator_source_mapping_field,
                    filetypes=[("CSV/TSV files", "*.csv *.tsv"), ("All files", "*.*")],
                ),
            ]
        )
        self.simulator_mapping_container = ft.Column(
            [
                ft.Text(
                    "Columns: subject, rest1, rest2, ..., reference_realtime, feedback_run-01, ...",
                    size=11,
                    color=ft.Colors.GREY_600,
                ),
                self.simulator_source_mapping_row,
            ],
            visible=is_mapping,
        )

        # Build individual simulator source path sections
        individual_controls = self._build_individual_simulator_paths()
        self.simulator_individual_container = ft.Column(
            individual_controls,
            visible=not is_mapping,
        )

        self.simulator_paths_container = ft.Column(
            [
                ft.Divider(),
                ft.Text("Simulator Source Paths", size=16, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Configure DICOM source directories for offline testing with the simulator",
                    size=11,
                    color=ft.Colors.GREY_600,
                    italic=True,
                ),
                self.simulator_source_mode_radio,
                self.simulator_mapping_container,
                self.simulator_individual_container,
            ],
            visible=is_simulator,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Output Directory", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "All processed data, masks, and feedback output will be saved under this directory.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Row([self.base_dir_field, self._create_browse_button(self.base_dir_field)]),
                    self.simulator_paths_container,
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    # ------------------------------------------------------------------
    # Simulator source paths (Paths tab)
    # ------------------------------------------------------------------

    def _build_individual_simulator_paths(self) -> list[ft.Control]:
        """Build individual simulator source path fields for all localizers and feedback runs."""
        controls: list[ft.Control] = []

        # Store fields for later collection
        self._sim_loc_source_fields: list[list[ft.TextField]] = []
        self._sim_loc_ref_field: ft.TextField | None = None
        self._sim_fb_source_fields: list[ft.TextField] = []
        self._sim_fb_transfer_fields: list[ft.TextField] = []
        self._sim_fb_ref_field: ft.TextField | None = None

        # --- Localizer sources ---
        for loc_idx, cfg in enumerate(self._localizer_configs):
            loc_type = cfg.get("type", "rest")
            if loc_type == "rest":
                label = f"Localizer {loc_idx + 1} (Rest)"
            else:
                try:
                    meta = get_task_metadata(loc_type)
                    label = f"Localizer {loc_idx + 1} ({meta.display_name})"
                except (KeyError, ValueError):
                    label = f"Localizer {loc_idx + 1} ({loc_type})"

            controls.append(ft.Text(label, size=14, weight=ft.FontWeight.W_500))

            source_dicoms = cfg.get("source_dicoms", [])
            if isinstance(source_dicoms, str):
                source_dicoms = [source_dicoms]

            try:
                n_runs = int(cfg.get("n_runs", 2))
            except (ValueError, TypeError):
                n_runs = 2

            loc_fields: list[ft.TextField] = []
            for run_idx in range(n_runs):
                val = source_dicoms[run_idx] if run_idx < len(source_dicoms) else ""
                field = ft.TextField(
                    width=540,
                    label=f"Run {run_idx + 1} source DICOMs",
                    value=val,
                    hint_text="Path to DICOM directory",
                )
                loc_fields.append(field)
                browse = self._create_browse_button(field)
                controls.append(ft.Row([field, browse]))

            self._sim_loc_source_fields.append(loc_fields)
            controls.append(ft.Divider())

        # --- Feedback sources ---
        for fb_idx, cfg in enumerate(self._feedback_configs):
            task_type = cfg.get("task_type", "ball")
            try:
                meta = get_task_metadata(task_type)
                label = f"Feedback {fb_idx + 1} ({meta.display_name})"
            except (KeyError, ValueError):
                label = f"Feedback {fb_idx + 1} ({task_type})"

            controls.append(ft.Text(label, size=14, weight=ft.FontWeight.W_500))

            # Feedback run source
            fb_field = ft.TextField(
                width=540,
                label="Feedback run source DICOMs",
                value=cfg.get("source_dicoms", ""),
                hint_text="Path to DICOM directory",
            )
            self._sim_fb_source_fields.append(fb_field)
            browse = self._create_browse_button(fb_field)
            controls.append(ft.Row([fb_field, browse]))

            # Transfer run source (only if transfers enabled)
            use_transfers = cfg.get("use_transfer_runs", False)
            tf_field = ft.TextField(
                width=540,
                label="Transfer run source DICOMs",
                value=cfg.get("transfer_source_dicoms", ""),
                hint_text="Path to DICOM directory",
                visible=use_transfers,
            )
            self._sim_fb_transfer_fields.append(tf_field)
            tf_browse = self._create_browse_button(tf_field)
            tf_row = ft.Row([tf_field, tf_browse], visible=use_transfers)
            controls.append(tf_row)

            controls.append(ft.Divider())

        # --- Reference scans (session-level) ---
        scan_config = self.study_config.session_config.get("scan", {})

        # Initial reference
        init_enabled = self.reference_initial_enabled_checkbox.value
        init_val = scan_config.get("reference_initial_dicoms", "")
        self._sim_loc_ref_label = ft.Text("Initial Reference Scan", size=14, weight=ft.FontWeight.W_500)
        self._sim_loc_ref_label.visible = init_enabled
        controls.append(self._sim_loc_ref_label)
        self._sim_loc_ref_field = ft.TextField(
            width=540,
            label="Initial Reference Scan source DICOMs",
            value=init_val,
            hint_text="Path to DICOM directory",
        )
        browse = self._create_browse_button(self._sim_loc_ref_field)
        self._sim_loc_ref_row = ft.Row(
            [self._sim_loc_ref_field, browse],
            visible=init_enabled,
        )
        controls.append(self._sim_loc_ref_row)

        # Real-time reference
        rt_enabled = self.reference_realtime_enabled_checkbox.value
        rt_val = scan_config.get("reference_realtime_dicoms", "")
        self._sim_fb_ref_label = ft.Text("Real-Time Reference Scan", size=14, weight=ft.FontWeight.W_500)
        self._sim_fb_ref_label.visible = rt_enabled
        controls.append(self._sim_fb_ref_label)
        self._sim_fb_ref_field = ft.TextField(
            width=540,
            label="Real-Time Reference Scan source DICOMs",
            value=rt_val,
            hint_text="Path to DICOM directory",
        )
        browse = self._create_browse_button(self._sim_fb_ref_field)
        self._sim_fb_ref_row = ft.Row(
            [self._sim_fb_ref_field, browse],
            visible=rt_enabled,
        )
        controls.append(self._sim_fb_ref_row)

        return controls

    def _collect_simulator_paths(self, config: dict | None = None):
        """Collect simulator source paths from Paths tab fields into configs.

        Reference scan DICOM paths are written to config["scan"] (session-level).
        Per-run source DICOMs are written to localizer/feedback config dicts.
        """
        # Localizer sources
        for loc_idx, loc_fields in enumerate(getattr(self, "_sim_loc_source_fields", [])):
            if loc_idx < len(self._localizer_configs):
                self._localizer_configs[loc_idx]["source_dicoms"] = [f.value.strip() for f in loc_fields]

        # Feedback sources
        for fb_idx, fb_field in enumerate(getattr(self, "_sim_fb_source_fields", [])):
            if fb_idx < len(self._feedback_configs):
                self._feedback_configs[fb_idx]["source_dicoms"] = fb_field.value.strip()

        # Feedback transfer sources
        for fb_idx, tf_field in enumerate(getattr(self, "_sim_fb_transfer_fields", [])):
            if fb_idx < len(self._feedback_configs):
                self._feedback_configs[fb_idx]["transfer_source_dicoms"] = tf_field.value.strip()

        # Reference scan DICOM paths (session-level)
        if config is not None:
            _loc_ref = getattr(self, "_sim_loc_ref_field", None)
            if _loc_ref is not None:
                config["scan"]["reference_initial_dicoms"] = _loc_ref.value.strip()
            _fb_ref = getattr(self, "_sim_fb_ref_field", None)
            if _fb_ref is not None:
                config["scan"]["reference_realtime_dicoms"] = _fb_ref.value.strip()

    # ------------------------------------------------------------------
    # Event handlers (Study tab)
    # ------------------------------------------------------------------

    def _on_reference_scan_changed(self, e):
        """Handle reference scan enable/disable toggle."""
        self.reference_initial_n_volumes_field.visible = self.reference_initial_enabled_checkbox.value
        self.reference_realtime_n_volumes_field.visible = self.reference_realtime_enabled_checkbox.value
        self._sync_paths_tab_visibility()
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_name_change(self, e):
        if not self.name_field.value.strip():
            self.name_field.error_text = "Study name is required"
            self.validation_errors["name"] = "Study name is required"
        else:
            self.name_field.error_text = None
            self.validation_errors.pop("name", None)
        self.name_field.update()

    def _on_simulator_mode_changed(self, e):
        is_simulator = self.simulator_enabled_checkbox.value
        self.collection_speed_field.visible = is_simulator
        self.feedback_simulator_speed_field.visible = is_simulator
        self.scanner_dicom_dir_field.visible = not is_simulator
        if hasattr(self, "scanner_dicom_dir_browse_btn"):
            self.scanner_dicom_dir_browse_btn.visible = not is_simulator
        self.scanner_ip_field.visible = not is_simulator
        self.scanner_setup_guide_btn.visible = not is_simulator

        if hasattr(self, "simulator_instruction_text"):
            self.simulator_instruction_text.value = (
                "Simulator replays volumes from source directories at configurable speed for offline testing."
                if is_simulator
                else "Scanner DICOM directory will be monitored for real-time acquisition."
            )
        if hasattr(self, "simulator_paths_container"):
            self.simulator_paths_container.visible = is_simulator

        self.page.update()

    def _on_simulator_source_mode_changed(self, e):
        mode = self.simulator_source_mode_radio.value
        if hasattr(self, "simulator_mapping_container"):
            self.simulator_mapping_container.visible = mode == "mapping"
        if hasattr(self, "simulator_individual_container"):
            self.simulator_individual_container.visible = mode != "mapping"
        self.page.update()

    # ------------------------------------------------------------------
    # Browse button factories
    # ------------------------------------------------------------------

    def _create_browse_button(self, field: ft.TextField) -> ft.IconButton:
        """Create a browse button for directory selection."""

        def pick_directory(e):
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                logger.error("tkinter is not available")
                return

            try:
                from pineuro.gui.dispatch import get_dispatcher

                def _show_dialog():
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    result = filedialog.askdirectory(
                        title=f"Select {field.label}", initialdir=field.value if field.value else None
                    )
                    root.destroy()
                    return result

                path = get_dispatcher().submit(_show_dialog).result()
                if path:
                    field.value = path
                    field.update()
            except Exception as ex:
                logger.error(f"Error opening directory picker: {ex}")

        return ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Browse for folder",
            on_click=pick_directory,
        )

    def _create_file_browse_button(
        self,
        field: ft.TextField,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> ft.IconButton:
        """Create a browse button for file selection."""
        if filetypes is None:
            filetypes = [("All Files", "*.*")]

        def pick_file(e):
            try:
                import os
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                logger.error("tkinter is not available")
                return

            try:
                from pineuro.gui.dispatch import get_dispatcher

                def _show_dialog():
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    initial_dir = None
                    if field.value:
                        if os.path.isfile(field.value):
                            initial_dir = os.path.dirname(field.value)
                        elif os.path.isdir(field.value):
                            initial_dir = field.value
                    result = filedialog.askopenfilename(
                        title=f"Select {field.label}",
                        initialdir=initial_dir,
                        filetypes=filetypes,
                    )
                    root.destroy()
                    return result

                path = get_dispatcher().submit(_show_dialog).result()
                if path:
                    field.value = path
                    field.update()
            except Exception as ex:
                logger.error(f"Error opening file picker: {ex}")

        return ft.IconButton(
            icon=ft.Icons.FILE_OPEN,
            tooltip="Browse for file",
            on_click=pick_file,
        )

    # ------------------------------------------------------------------
    # Monitor preview
    # ------------------------------------------------------------------

    def _build_monitor_preview(self) -> ft.Container:
        try:
            from pineuro.display import get_monitors

            monitors = get_monitors()
            if not monitors:
                return ft.Container(
                    content=ft.Text("No monitors detected", size=12, color=ft.Colors.ORANGE_700),
                    padding=10,
                )

            monitor_texts = []
            for m in monitors:
                position = "Left" if m.index == 0 else "Right"
                monitor_texts.append(
                    ft.Text(f"{position}: {m.name} ({m.width}x{m.height})", size=12, color=ft.Colors.GREY_700)
                )

            return ft.Container(
                content=ft.Column(
                    [ft.Text("Detected monitors:", size=12, weight=ft.FontWeight.BOLD), *monitor_texts],
                    spacing=2,
                ),
                bgcolor=ft.Colors.GREY_100,
                padding=10,
                border_radius=5,
            )
        except Exception as e:
            logger.warning(f"Failed to detect monitors: {e}")
            return ft.Container(
                content=ft.Text(f"Monitor detection unavailable: {e}", size=12, color=ft.Colors.ORANGE_700),
                padding=10,
            )

    def _show_scanner_setup_guide(self):
        """Show platform-specific SMB share setup instructions dialog."""
        from pineuro.scanner_setup import get_current_platform_key, get_setup_instructions

        watch_dir = self.scanner_dicom_dir_field.value.strip() or "/BOLD_DICOM"
        scanner_ip = self.scanner_ip_field.value.strip()

        instructions = get_setup_instructions(watch_dir, scanner_ip)
        platform_key = get_current_platform_key()
        initial_index = {"windows": 0, "macos": 1, "linux": 2}.get(platform_key, 0)

        def _make_content(text: str) -> ft.Container:
            return ft.Container(
                content=ft.Column(
                    [ft.Text(text, font_family="monospace", selectable=True, size=12)],
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=10,
            )

        tabs = ft.Tabs(
            length=3,
            selected_index=initial_index,
            content=ft.Column(
                [
                    ft.TabBar(
                        tabs=[ft.Tab(label="Windows"), ft.Tab(label="macOS"), ft.Tab(label="Linux")],
                    ),
                    ft.TabBarView(
                        controls=[
                            _make_content(instructions["windows"]),
                            _make_content(instructions["macos"]),
                            _make_content(instructions["linux"]),
                        ],
                        expand=True,
                    ),
                ],
                expand=True,
            ),
            expand=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Scanner Setup Guide"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text("SMB Share Setup", size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(content=tabs, height=300),
                        ft.Divider(),
                        ft.Text("Scanner Console Configuration", size=14, weight=ft.FontWeight.BOLD),
                        ft.Text(instructions["scanner_notes"], size=12),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=700,
                height=550,
            ),
            actions=[ft.TextButton("Close", on_click=lambda e: _close())],
        )

        def _close():
            dialog.open = False
            self.page.update()

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    # ------------------------------------------------------------------
    # Save / Cancel / Start
    # ------------------------------------------------------------------

    def _on_cancel(self, e):
        logger.info("Study editor cancelled")
        if self.on_cancel_callback:
            self.on_cancel_callback()

    def _on_save(self, e):
        if self.validation_errors:
            _snack = ft.SnackBar(ft.Text("Please fix validation errors before saving"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        self._update_study_from_form()

        # Validate: at least one localizer or feedback run required
        config = self.study_config.session_config
        has_localizers = len(config.get("localizers", [])) > 0
        has_feedback = len(config.get("feedback_runs", [])) > 0
        if not has_localizers and not has_feedback:
            _snack = ft.SnackBar(
                ft.Text("At least one localizer or feedback task is required"),
                bgcolor=ft.Colors.RED_200,
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        try:
            path = self.config_manager.save_study(self.study_config, overwrite=True)
            _snack = ft.SnackBar(ft.Text(f"Study saved: {self.study_config.metadata.name}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            logger.info(f"Study saved: {path}")
            if self.on_save_callback:
                self.on_save_callback()
        except Exception as ex:
            logger.error(f"Failed to save study: {ex}")
            _snack = ft.SnackBar(ft.Text(f"Error saving study: {ex}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

    def _on_save_as(self, e):
        new_name_field = ft.TextField(
            label="New study name",
            value=f"{self.study_config.metadata.name} (Copy)",
            autofocus=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Save Study As"),
            content=ft.Container(content=new_name_field, width=400),
        )

        def save_with_new_name(e):
            new_name = new_name_field.value.strip()
            if not new_name:
                new_name_field.error_text = "Study name is required"
                new_name_field.update()
                return

            existing = self.config_manager.list_studies()
            if any(s.name == new_name for s in existing):
                new_name_field.error_text = "A study with this name already exists"
                new_name_field.update()
                return

            self._update_study_from_form()
            self.study_config.metadata.name = new_name
            self.study_config.file_path = None

            try:
                path = self.config_manager.save_study(self.study_config, overwrite=False)
                self.page.pop_dialog()
                _snack = ft.SnackBar(ft.Text(f"Study saved as: {new_name}"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                logger.info(f"Study saved as: {path}")
                if self.on_save_callback:
                    self.on_save_callback()
            except Exception as ex:
                logger.error(f"Failed to save study: {ex}")
                new_name_field.error_text = str(ex)
                new_name_field.update()

        dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.page.pop_dialog()),
            ft.ElevatedButton("Save", on_click=save_with_new_name),
        ]
        self.page.show_dialog(dialog)

    def _on_start_session(self, e):
        from pathlib import Path

        from ..components.visit_selection_dialog import VisitSelectionDialog

        example_id = self.example_participant_id_field.value.strip() or "sub-001"

        participant_id_field = ft.TextField(
            label="Participant ID",
            hint_text=f"e.g., {example_id}",
            autofocus=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Start Participant Session"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(f"Study: {self.study_config.metadata.name}", size=14, weight=ft.FontWeight.BOLD),
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

            self._update_study_from_form()
            self.page.pop_dialog()
            logger.info(f"Starting session for participant: {participant_id}")

            multi_visit_enabled = self.multi_visit_enabled_checkbox.value

            if multi_visit_enabled:
                base_dir = Path(self.study_config.session_config["paths"]["base_dir"])

                def on_visit_selected(visit_id, visit_config):
                    if self.on_start_session_callback:
                        self.on_start_session_callback(participant_id, visit_id, visit_config)

                visit_dialog = VisitSelectionDialog(
                    page=self.page,
                    participant_id=participant_id,
                    base_dir=base_dir,
                    on_select=on_visit_selected,
                )
                visit_dialog.show()
            else:
                if self.on_start_session_callback:
                    self.on_start_session_callback(participant_id, None, None)

        dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.page.pop_dialog()),
            ft.FilledButton("Start", icon=ft.Icons.PLAY_ARROW, on_click=start_session),
        ]
        self.page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # Save logic
    # ------------------------------------------------------------------

    def _update_study_from_form(self):
        """Update study config from form field values."""
        self.study_config.metadata.name = self.name_field.value.strip()
        self.study_config.metadata.description = self.description_field.value.strip()

        config = self.study_config.session_config

        # Paths
        config["paths"]["example_participant_id"] = self.example_participant_id_field.value.strip() or "sub-001"
        config["paths"]["base_dir"] = self.base_dir_field.value.strip() or "/data/pineuro"

        # Multi-visit
        if "study" not in config:
            config["study"] = {}
        config["study"]["multi_visit_enabled"] = self.multi_visit_enabled_checkbox.value

        # Reference scans (session-level, from Study tab)
        config["scan"]["reference_initial_enabled"] = self.reference_initial_enabled_checkbox.value
        try:
            config["scan"]["reference_initial_n_volumes"] = int(self.reference_initial_n_volumes_field.value or 2)
        except (ValueError, TypeError):
            config["scan"]["reference_initial_n_volumes"] = 2
        config["scan"]["reference_realtime_enabled"] = self.reference_realtime_enabled_checkbox.value
        try:
            config["scan"]["reference_realtime_n_volumes"] = int(self.reference_realtime_n_volumes_field.value or 2)
        except (ValueError, TypeError):
            config["scan"]["reference_realtime_n_volumes"] = 2

        # Acquisition mode
        config["scan"]["simulator_enabled"] = self.simulator_enabled_checkbox.value
        try:
            config["scan"]["collection_speed"] = float(self.collection_speed_field.value or 10.0)
        except (ValueError, TypeError):
            config["scan"]["collection_speed"] = 10.0
        try:
            config["scan"]["feedback_simulator_speed"] = float(self.feedback_simulator_speed_field.value or 1.0)
        except (ValueError, TypeError):
            config["scan"]["feedback_simulator_speed"] = 1.0
        config["scan"]["scanner_dicom_dir"] = self.scanner_dicom_dir_field.value.strip() or "/BOLD_DICOM"
        config["scan"]["scanner_ip"] = self.scanner_ip_field.value.strip()

        # Simulator mapping
        if self.simulator_source_mode_radio.value == "mapping":
            config["scan"]["simulator_source_mapping"] = self.simulator_source_mapping_field.value.strip()
        else:
            config["scan"]["simulator_source_mapping"] = ""

        # Display
        if "display" not in config:
            config["display"] = {}
        config["display"]["control_monitor"] = self.control_monitor_dropdown.value or "left"
        config["display"]["presenter_monitor"] = self.presenter_monitor_dropdown.value or "right"

        # --- Collect from tab builders ---

        # Localizers — collect from builders, then inject Paths tab values
        localizer_configs = [b.collect_config() for b in self._localizer_builders]
        self._localizer_configs = localizer_configs

        # Backward compat: write flat keys from first localizer
        if not localizer_configs:
            config["scan"]["localizer_n_runs"] = 0
        if localizer_configs:
            first = localizer_configs[0]
            config["scan"]["localizer_tr"] = first.get("tr", 1.2)
            config["scan"]["localizer_n_volumes"] = first.get("n_volumes", 200)
            config["scan"]["localizer_n_runs"] = first.get("n_runs", 2)
            config["scan"]["localizer_task_type"] = first.get("type", "rest")

            # Flat mask keys from first localizer
            config["mask"]["localizer_type"] = "rest" if first.get("type") == "rest" else "task"
            if first.get("type") == "rest":
                config["mask"]["network_atlas"] = first.get("network_atlas", "yeo_17")
                config["mask"]["n_yeo"] = first.get("n_yeo", 17)
                config["mask"]["n_components"] = first.get("n_components", 20)
                config["mask"]["smooth_fwhm"] = first.get("smooth_fwhm", 5.0)
                config["mask"]["highpass"] = first.get("highpass", 0.01)
            config["mask"]["threshold"] = first.get("threshold", 2000)
            config["mask"]["threshold_method"] = first.get("threshold_method", "top_n")
            config["mask"]["include_reverse_contrasts"] = first.get("include_reverse_contrasts", False)
            if first.get("threshold_method") == "cluster":
                config["mask"]["atlas_source"] = first.get("atlas_source")
                config["mask"]["atlas_regions"] = first.get("atlas_regions")
                config["mask"]["cluster_start_threshold"] = first.get("cluster_start_threshold", 2.0)
                config["mask"]["cluster_step"] = first.get("cluster_step", 0.1)
                config["mask"]["cluster_return_all"] = first.get("cluster_return_all", False)
            elif first.get("threshold_method") == "cluster_extent":
                config["mask"]["cluster_start_threshold"] = first.get("cluster_start_threshold", 2.0)
                config["mask"]["min_cluster_size"] = first.get("min_cluster_size", 50)
                config["mask"]["max_total_voxels"] = first.get("max_total_voxels")
                if first.get("atlas_source"):
                    config["mask"]["atlas_source"] = first.get("atlas_source")

            # Localizer source DICOMs (flat)
            config["scan"]["localizer_source_dicoms"] = first.get("source_dicoms", [])

        # Aggregate mask names from all localizers
        all_mask_names: list[str] = []
        for loc_cfg in localizer_configs:
            if loc_cfg.get("type") == "rest":
                all_mask_names.extend(loc_cfg.get("yeo_targets", []))
            else:
                contrasts = loc_cfg.get("contrasts", {})
                all_mask_names.extend(contrasts.keys())
                if loc_cfg.get("include_reverse_contrasts"):
                    all_mask_names.extend(f"{k}_reverse" for k in contrasts)
        config["mask"]["yeo_targets"] = all_mask_names
        config["mask"]["mask_names"] = all_mask_names

        # Feedback runs — collect from builders, then inject Paths tab values
        feedback_configs = [b.collect_config() for b in self._feedback_builders]
        self._feedback_configs = feedback_configs

        # Now inject Paths tab simulator paths into the collected configs
        self._collect_simulator_paths(config)

        config["localizers"] = self._localizer_configs
        config["feedback_runs"] = self._feedback_configs

        # Backward compat: write flat keys from first feedback run
        if not feedback_configs:
            config["task"]["max_feedback_runs"] = 0
        if feedback_configs:
            first_fb = feedback_configs[0]
            config["scan"]["feedback_tr"] = first_fb.get("tr", 1.2)
            config["scan"]["feedback_n_volumes"] = first_fb.get("n_volumes", 150)

            # Flat feedback config
            config["feedback"]["n_baseline"] = first_fb.get("n_baseline", 25)
            config["feedback"]["n_volumes"] = first_fb.get("n_volumes", 150)
            config["feedback"]["fd_threshold"] = first_fb.get("fd_threshold", 0.5)
            config["feedback"]["dvars_threshold"] = first_fb.get("dvars_threshold", 1.5)
            config["feedback"]["tsnr_threshold"] = first_fb.get("tsnr_threshold", 20)
            config["feedback"]["motion_correction"] = first_fb.get("motion_correction", "siemens_moco")
            config["feedback"]["include_drift"] = first_fb.get("include_drift", True)
            config["feedback"]["include_motion_regressors"] = first_fb.get("include_motion_regressors", True)
            config["feedback"]["include_wm_regressor"] = first_fb.get("include_wm_regressor", False)
            config["feedback"]["include_csf_regressor"] = first_fb.get("include_csf_regressor", False)
            config["feedback"]["smooth_fwhm"] = first_fb.get("smooth_fwhm")
            config["feedback"]["feedback_mode"] = first_fb.get("feedback_mode", "activation")
            config["feedback"]["differential_pair"] = first_fb.get("differential_pair")
            config["feedback"]["connectivity_pairs"] = first_fb.get("connectivity_pairs")
            config["feedback"]["connectivity_window"] = first_fb.get("connectivity_window", 15)
            config["feedback"]["connectivity_target"] = first_fb.get("connectivity_target", "increase")
            config["feedback"]["feedback_interval"] = first_fb.get("feedback_interval")
            # Remove legacy fields
            config["feedback"].pop("moco_enabled", None)
            config["feedback"].pop("nuisance_regression", None)

            # Flat task config
            task_type = first_fb.get("task_type", "ball")
            config["task"]["task_type"] = task_type
            # Total runs = sum of max_runs across all feedback tasks
            config["task"]["max_feedback_runs"] = sum(fb_cfg.get("max_runs", 1) for fb_cfg in feedback_configs)
            task_params = first_fb.get("task_params", {})
            if task_type not in config["task"]:
                config["task"][task_type] = {}
            config["task"][task_type].update(task_params)
            config["task"][task_type]["n_baseline"] = first_fb.get("n_baseline", 25)

            # Flat sham config
            sham = first_fb.get("sham", {})
            if "sham" not in config:
                config["sham"] = {}
            config["sham"].update(sham)
            config["sham"].pop("replay_source_dir", None)

            # Flat DICOM sources
            config["scan"]["feedback_source_dicoms"] = [fb_cfg.get("source_dicoms", "") for fb_cfg in feedback_configs]
            config["scan"]["transfer_source_dicoms"] = [
                fb_cfg.get("transfer_source_dicoms", "") for fb_cfg in feedback_configs[:2]
            ]

        # Remove legacy single-path
        config["scan"].pop("feedback_dicoms", None)

        # Localizer task config (flat, from first localizer if task-based)
        if localizer_configs:
            first_loc = localizer_configs[0]
            if first_loc.get("type") != "rest":
                loc_type = first_loc["type"]
                if "localizer_task" not in config:
                    config["localizer_task"] = {}
                if loc_type not in config["localizer_task"]:
                    config["localizer_task"][loc_type] = {}
                config["localizer_task"][loc_type].update(first_loc.get("task_params", {}))
                # Store contrasts in mask config
                config["mask"]["contrasts"] = first_loc.get("contrasts", {})

        # Network assignment for first feedback task
        if feedback_configs and localizer_configs:
            task_type = feedback_configs[0].get("task_type", "ball")
            try:
                task_meta = get_task_metadata(task_type)

                if task_type not in config["task"]:
                    config["task"][task_type] = {}

                # Use explicit network_assignment from feedback tab if available
                explicit_assignment = feedback_configs[0].get("network_assignment")
                if explicit_assignment:
                    config["task"][task_type].update(explicit_assignment)
                else:
                    # Fall back to inferring from differential pair / yeo targets
                    first_loc = localizer_configs[0]
                    is_task_loc = first_loc.get("type") != "rest"
                    if is_task_loc:
                        dropdowns_values = feedback_configs[0].get("differential_pair") or []
                    else:
                        dropdowns_values = first_loc.get("yeo_targets", [])

                    for i, role in enumerate(task_meta.network_roles):
                        if i < len(dropdowns_values) and dropdowns_values[i]:
                            config["task"][task_type][role.key] = dropdowns_values[i]
            except (ValueError, AttributeError):
                pass
