"""Study Editor screen for creating and editing study configurations.

This module provides a tabbed interface for configuring all aspects of a pineuro study:
- Metadata and scanner parameters
- Network selection and preprocessing
- Task-specific parameters (Ball task, future tasks)
- File paths for simulator mode

The editor supports both creating new studies and editing existing ones, with
real-time validation and user-friendly file browsing via tkinter dialogs.
"""

import contextlib
import logging

import flet as ft

from pineuro.tasks import TaskParameter, get_task_metadata, list_available_tasks

from ..config_manager import ConfigManager
from ..models import StudyConfig, StudyMetadata
from ..utils import (
    ValidationError,
    validate_positive_int,
    validate_positive_number,
)

logger = logging.getLogger(__name__)


class StudyEditorScreen:
    """Study editor screen with tabbed parameter sections."""

    # Uniform widths for visual consistency across all tabs
    FIELD_W = 400  # standalone fields (text, dropdown, number)
    PAIR_W = 195  # side-by-side paired fields (two per row)

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
        self.is_new = study_config is None
        self.on_cancel_callback = on_cancel
        self.on_save_callback = on_save
        self.on_start_session_callback = on_start_session

        # Create default config if new study
        self.study_config: StudyConfig = study_config if study_config is not None else self._create_default_config()

        # Validation state
        self.validation_errors: dict[str, str] = {}

        # Dynamic task parameter fields (populated based on task type)
        self.task_param_fields: dict[str, ft.Control] = {}
        # Dynamic localizer task parameter fields
        self.localizer_task_param_fields: dict[str, ft.Control] = {}

        # Create form fields
        self._create_form_fields()

    def _create_default_config(self) -> StudyConfig:
        """Create a default study configuration.

        Returns:
            Default StudyConfig
        """
        metadata = StudyMetadata(name="New Study", description="")

        session_config = {
            "paths": {
                "participant_id": "",
                "base_dir": "/data/pineuro",
                "example_participant_id": "sub-001",
            },
            "scan": {
                # Localizer parameters
                "localizer_tr": 1.2,
                "localizer_n_volumes": 200,
                "localizer_n_runs": 2,
                "localizer_task_type": "rest",
                # Feedback parameters
                "feedback_tr": 1.2,
                "feedback_n_volumes": 150,
                # Reference scan parameters
                "reference_initial_enabled": True,
                "reference_initial_n_volumes": 2,
                "reference_realtime_enabled": True,
                "reference_realtime_n_volumes": 2,
                "reference_initial_dicoms": "",
                "reference_realtime_dicoms": "",
                # Simulator/acquisition
                "localizer_source_dicoms": ["", ""],
                "feedback_source_dicoms": ["", ""],  # Per-run feedback DICOM sources
                "transfer_source_dicoms": ["", ""],  # Per-run transfer DICOM sources (transfer_1, transfer_2)
                "simulator_source_mapping": "",  # Path to CSV/TSV mapping subjects to DICOM paths
                "simulator_enabled": True,
                "collection_speed": 10.0,  # Speed for localizer/collection
                "feedback_simulator_speed": 1.0,  # Speed for feedback runs
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
            "localizer_task": {
                # Localizer task parameters (when not rest)
                "ball": {
                    "use_transfer_runs": False,
                    "initial_scale_factor": 10.0,
                    "trigger_keys": ["t", "+", "5", "equal"],
                    "left_button": "1",
                    "right_button": "2",
                    "enter_button": "0",
                    "min_hits_threshold": 3,
                    "max_hits_threshold": 5,
                    "scale_adjust_factor": 0.25,
                },
            },
            "task": {
                "task_type": "ball",  # Feedback task type identifier
                "max_feedback_runs": 2,
                # Task-specific params populated from TaskMetadata defaults
            },
            "sham": {
                "enabled": False,
                "blind_mode": False,  # Hide network names in GUI (show Target/Non-Target)
                "delivery_method": "alternative_masks",
                "sham_mask_mode": "hemisphere_split",
                "hemisphere_network": "SomMotA",
                "target_hemisphere": "right",
                "alternative_mask_targets": ["SomMotA", "DefaultA"],
                "randomization_file": "",
            },
            "display": {
                "control_monitor": "left",  # Where GUI opens (left/right)
                "presenter_monitor": "right",  # Where PsychoPy tasks open (left/right)
            },
        }

        return StudyConfig(metadata=metadata, session_config=session_config)

    def _create_form_fields(self):
        """Create all form field controls."""
        config = self.study_config.session_config

        # Metadata fields
        self.name_field = ft.TextField(
            width=600,
            label="Study Name",
            value=self.study_config.metadata.name,
            on_change=self._on_name_change,
        )

        self.description_field = ft.TextField(
            width=600,
            label="Description",
            value=self.study_config.metadata.description,
            multiline=True,
            min_lines=2,
            max_lines=4,
        )

        self.example_participant_id_field = ft.TextField(
            width=600,
            label="Example Participant ID",
            value=config["paths"].get("example_participant_id", "sub-001"),
            hint_text="Shown as hint when starting a session (e.g., sub-001, sub-mindbpd001)",
        )

        # Multi-visit mode toggle (Study tab)
        self.multi_visit_enabled_checkbox = ft.Checkbox(
            label="Enable Multiple Visits",
            value=config.get("study", {}).get("multi_visit_enabled", False),
        )

        # Simulator mode toggle (Study tab)
        self.simulator_enabled_checkbox = ft.Checkbox(
            label="Simulator Mode",
            value=config["scan"].get("simulator_enabled", True),
            on_change=self._on_simulator_mode_changed,
        )

        self.collection_speed_field = ft.TextField(
            width=290,
            label="Localizer speed (× realtime)",
            value=str(config["scan"].get("collection_speed", config["scan"].get("simulator_speed", 10.0))),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Speed for localizer/setup runs",
        )

        self.feedback_simulator_speed_field = ft.TextField(
            width=290,
            label="Feedback speed (× realtime)",
            value=str(config["scan"].get("feedback_simulator_speed", config["scan"].get("simulator_speed", 10.0))),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Speed for feedback runs",
        )

        # Scanner DICOM directory for real-time mode
        self.scanner_dicom_dir_field = ft.TextField(
            width=540,
            label="Scanner DICOM Directory",
            value=config["scan"].get("scanner_dicom_dir", "/BOLD_DICOM"),
            hint_text="Directory where scanner exports DICOMs (e.g., /BOLD_DICOM)",
        )

        # Scanner IP address for connection testing
        self.scanner_ip_field = ft.TextField(
            width=290,
            label="Scanner IP Address",
            value=config["scan"].get("scanner_ip", ""),
            hint_text="e.g., 192.168.2.1",
        )

        # Setup guide button
        self.scanner_setup_guide_btn = ft.OutlinedButton(
            "Setup Guide",
            icon=ft.Icons.HELP_OUTLINE,
            on_click=lambda e: self._show_scanner_setup_guide(),
        )

        # --- Localizer tab fields ---
        # Localizer task type (rest or registered task)
        available_tasks = list_available_tasks(role="localizer")
        localizer_task_options = [ft.dropdown.Option(key="rest", text="Rest (ICA Network Extraction)")]
        localizer_task_options.extend(
            [
                ft.dropdown.Option(key=task_id, text=f"{metadata.display_name} (GLM Contrast)")
                for task_id, metadata in available_tasks.items()
                if metadata.default_contrasts
            ]
        )
        self.localizer_task_type_dropdown = ft.Dropdown(
            width=600,
            label="Localizer Type",
            options=localizer_task_options,
            value=config["scan"].get("localizer_task_type", "rest"),
            on_select=self._on_localizer_task_type_changed,
        )

        # Show GLM contrast info for the selected task
        self.contrast_info_text = ft.Text(
            value=self._get_contrast_info_text(config["scan"].get("localizer_task_type", "rest")),
            size=12,
            color=ft.Colors.BLUE_700,
            italic=True,
        )

        # Reverse contrasts checkbox (task-based localizers only)
        self.include_reverse_contrasts_checkbox = ft.Checkbox(
            label="Include reverse contrasts (enables dual-network from single-contrast localizer)",
            value=config.get("mask", {}).get("include_reverse_contrasts", False),
            visible=config["scan"].get("localizer_task_type", "rest") != "rest",
            on_change=self._on_include_reverse_contrasts_changed,
        )

        self.localizer_tr_field = ft.TextField(
            width=600,
            label="TR (seconds)",
            value=str(config["scan"].get("localizer_tr", 1.2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("localizer_tr", self.localizer_tr_field, validate_positive_number),
        )

        self.localizer_n_volumes_field = ft.TextField(
            width=600,
            label="Volumes per localizer run",
            value=str(config["scan"].get("localizer_n_volumes", 200)),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field(
                "localizer_n_volumes", self.localizer_n_volumes_field, validate_positive_int
            ),
        )

        self.localizer_n_runs_field = ft.TextField(
            width=600,
            label="Number of localizer runs",
            value=str(config["scan"].get("localizer_n_runs", 2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=self._on_localizer_n_runs_changed,
        )

        # Reference Initial settings (for Localizer tab)
        self.reference_initial_enabled_checkbox = ft.Checkbox(
            label="Enable Initial Reference Scan",
            value=config["scan"].get("reference_initial_enabled", True),
            on_change=self._on_reference_initial_enabled_changed,
        )

        self.reference_initial_n_volumes_field = ft.TextField(
            width=600,
            label="Number of volumes",
            value=str(config["scan"].get("reference_initial_n_volumes", 2)),
            visible=config["scan"].get("reference_initial_enabled", True),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field(
                "reference_initial_n_volumes", self.reference_initial_n_volumes_field, validate_positive_int
            ),
        )

        # Reference Realtime settings (for Feedback tab)
        self.reference_realtime_enabled_checkbox = ft.Checkbox(
            label="Enable Real-Time Reference Scan",
            value=config["scan"].get("reference_realtime_enabled", True),
            on_change=self._on_reference_realtime_enabled_changed,
        )

        self.reference_realtime_n_volumes_field = ft.TextField(
            width=600,
            label="Number of volumes",
            value=str(config["scan"].get("reference_realtime_n_volumes", 2)),
            visible=config["scan"].get("reference_realtime_enabled", True),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field(
                "reference_realtime_n_volumes", self.reference_realtime_n_volumes_field, validate_positive_int
            ),
        )

        # Create localizer task parameter fields
        self._create_localizer_task_parameter_fields()

        # --- Feedback tab fields ---
        self.feedback_tr_field = ft.TextField(
            width=600,
            label="TR (seconds)",
            value=str(config["scan"].get("feedback_tr", 1.2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("feedback_tr", self.feedback_tr_field, validate_positive_number),
        )

        self.feedback_n_volumes_field = ft.TextField(
            width=600,
            label="Volumes per feedback run",
            value=str(config["scan"].get("feedback_n_volumes", 150)),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field(
                "feedback_n_volumes", self.feedback_n_volumes_field, validate_positive_int
            ),
        )

        # Network atlas dropdown (replaces n_yeo_radio)
        from pineuro.mask_extraction import ATLAS_SOURCES

        self.network_atlas_dropdown = ft.Dropdown(
            width=600,
            label="Network atlas",
            options=[
                ft.dropdown.Option(key=k, text=v["display_name"])
                for k, v in ATLAS_SOURCES.items()
                if v["type"] == "network"
            ],
            value=config["mask"].get("network_atlas", "yeo_17"),
            on_select=self._on_network_atlas_changed,
        )

        # Network selection dropdowns
        # Get current targets from config
        yeo_targets = config["mask"].get("yeo_targets", ["DefaultA", "ContA"])
        first_network = yeo_targets[0] if len(yeo_targets) > 0 else None
        second_network = yeo_targets[1] if len(yeo_targets) > 1 else None

        # Derive n_networks from the selected atlas
        current_atlas = config["mask"].get("network_atlas", "yeo_17")
        current_n_networks = ATLAS_SOURCES.get(current_atlas, {}).get("n_networks", 17)

        self.first_network_dropdown = ft.Dropdown(
            width=290,
            label="First Network",
            options=self._get_yeo_network_options(current_n_networks),
            value=first_network,
        )

        self.second_network_dropdown = ft.Dropdown(
            width=290,
            label="Second Network",
            options=self._get_yeo_network_options(current_n_networks),
            value=second_network,
        )

        self.smooth_fwhm_field = ft.TextField(
            width=600,
            label="Spatial smoothing (mm FWHM)",
            value=str(config["mask"]["smooth_fwhm"]),
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        self.highpass_field = ft.TextField(
            width=600,
            label="Highpass filter (Hz)",
            value=str(config["mask"]["highpass"]),
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        self.n_components_field = ft.TextField(
            width=600,
            label="ICA components",
            value=str(config["mask"].get("n_components", 20)),
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_components", self.n_components_field, validate_positive_int),
        )

        self.threshold_field = ft.TextField(
            width=600,
            label="Voxels per mask",
            value=str(config["mask"].get("threshold", 2000)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of voxels to keep per extracted mask",
            on_change=lambda _: self._validate_field("threshold", self.threshold_field, validate_positive_int),
        )

        # --- Threshold method (top_n vs cluster) ---
        self.threshold_method_dropdown = ft.Dropdown(
            width=600,
            label="Thresholding method",
            options=[
                ft.dropdown.Option(key="top_n", text="Top N voxels (default)"),
                ft.dropdown.Option(key="cluster", text="Atlas-guided cluster"),
            ],
            value=config["mask"].get("threshold_method", "top_n"),
            on_select=self._on_threshold_method_changed,
        )

        # Two-tier atlas selection for cluster thresholding
        # First tier: atlas source (Harvard-Oxford, AAL, Schaefer, Gray Matter)
        cluster_atlas_options = [
            ft.dropdown.Option(key=k, text=v["display_name"])
            for k, v in ATLAS_SOURCES.items()
            if v["type"] in ("anatomical", "tissue") or k.startswith("schaefer")
        ]
        current_atlas_source = config["mask"].get("atlas_source") or "harvard_oxford"
        self.atlas_source_dropdown = ft.Dropdown(
            width=600,
            label="Atlas",
            options=cluster_atlas_options,
            value=current_atlas_source,
            on_select=self._on_atlas_source_changed,
        )

        # Second tier: region selection (populated on demand from atlas)
        current_regions = config["mask"].get("atlas_regions") or []
        self.atlas_region_dropdown = ft.Dropdown(
            width=600,
            label="Region",
            options=[],
            value=current_regions[0] if current_regions else None,
        )

        self.cluster_start_threshold_field = ft.TextField(
            width=290,
            label="Start threshold",
            value=str(config["mask"].get("cluster_start_threshold", 2.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Initial stat threshold for cluster search",
        )

        self.cluster_step_field = ft.TextField(
            width=290,
            label="Step size",
            value=str(config["mask"].get("cluster_step", 0.1)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Threshold increment per iteration",
        )

        self.cluster_return_all_checkbox = ft.Checkbox(
            label="Return all clusters (use for whole-brain atlases like gray matter)",
            value=config["mask"].get("cluster_return_all", False),
        )

        # Cluster settings section (shown/hidden based on threshold_method)
        self.cluster_settings_section = ft.Column(
            [
                self.atlas_source_dropdown,
                self.atlas_region_dropdown,
                ft.Row([self.cluster_start_threshold_field, self.cluster_step_field]),
                self.cluster_return_all_checkbox,
            ],
            visible=config["mask"].get("threshold_method", "top_n") == "cluster",
        )

        # Feedback task type selection (populated from registry)
        feedback_tasks = list_available_tasks(role="feedback")
        self.task_type_dropdown = ft.Dropdown(
            width=600,
            label="Feedback Task Type",
            options=[
                ft.dropdown.Option(
                    key=task_id,
                    text=metadata.display_name,
                )
                for task_id, metadata in feedback_tasks.items()
            ],
            value=config["task"].get("task_type", "ball"),
            on_select=self._on_task_type_changed,
        )

        self.max_feedback_runs_field = ft.TextField(
            width=600,
            label="Number of feedback runs",
            value=str(config["task"]["max_feedback_runs"]),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Total feedback runs (transfer runs are optional)",
            on_change=self._on_max_feedback_runs_changed,
        )

        # Baseline volumes - task-specific but shown in Session Structure
        # Get from task config (ball.n_baseline) or feedback config as fallback
        task_type = config["task"].get("task_type", "ball")
        n_baseline_default = (
            config["task"].get(task_type, {}).get("n_baseline", config.get("feedback", {}).get("n_baseline", 25))
        )
        self.n_baseline_field = ft.TextField(
            width=600,
            label="Baseline volumes",
            value=str(n_baseline_default),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of baseline volumes at the start of each feedback run",
            on_change=lambda _: self._validate_field("n_baseline", self.n_baseline_field, validate_positive_int),
        )

        # FD threshold for motion monitoring
        self.fd_threshold_field = ft.TextField(
            width=600,
            label="FD threshold (mm)",
            value=str(config.get("feedback", {}).get("fd_threshold", 0.5)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Frames exceeding this will be flagged in real time",
            on_change=lambda _: self._validate_field("fd_threshold", self.fd_threshold_field, validate_positive_number),
        )

        # DVARS threshold for signal quality monitoring
        self.dvars_threshold_field = ft.TextField(
            width=600,
            label="DVARS threshold (standardized)",
            value=str(config.get("feedback", {}).get("dvars_threshold", 1.5)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Standardized DVARS above this will be flagged in real time",
            on_change=lambda _: self._validate_field(
                "dvars_threshold", self.dvars_threshold_field, validate_positive_number
            ),
        )

        # tSNR threshold for signal quality monitoring
        self.tsnr_threshold_field = ft.TextField(
            width=600,
            label="tSNR threshold",
            value=str(config.get("feedback", {}).get("tsnr_threshold", 20)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="tSNR below this will be flagged in real time",
            on_change=lambda _: self._validate_field(
                "tsnr_threshold", self.tsnr_threshold_field, validate_positive_number
            ),
        )

        # Motion correction mode
        _mc_value = config.get("feedback", {}).get("motion_correction", "siemens_moco")
        # Backward compat: migrate old moco_enabled boolean
        if "moco_enabled" in config.get("feedback", {}) and "motion_correction" not in config.get("feedback", {}):
            _mc_value = "siemens_moco" if config["feedback"]["moco_enabled"] else "software"
        self.motion_correction_dropdown = ft.Dropdown(
            label="Motion correction",
            width=600,
            value=_mc_value,
            options=[
                ft.dropdown.Option(key="siemens_moco", text="Siemens MoCo"),
                ft.dropdown.Option(key="software", text="Software (ANTsPy)"),
                ft.dropdown.Option(key="none", text="None"),
            ],
            tooltip="How volumes are motion-corrected during real-time feedback.",
            on_select=self._on_motion_correction_changed,
        )

        # GLM nuisance regressor checkboxes
        _fb_cfg = config.get("feedback", {})
        # Backward compat: migrate old nuisance_regression boolean
        _old_nuisance = _fb_cfg.get("nuisance_regression", False)
        self.drift_checkbox = ft.Checkbox(
            label="Linear drift",
            value=_fb_cfg.get("include_drift", True),
            tooltip="Include linear drift regressor in GLM nuisance model.",
        )
        self.motion_regressors_checkbox = ft.Checkbox(
            label="Motion parameters (6 DOF)",
            value=_fb_cfg.get("include_motion_regressors", True),
            tooltip="Include 6 differential motion parameters in GLM. Requires motion correction.",
            disabled=_mc_value == "none",
        )
        self.wm_regressor_checkbox = ft.Checkbox(
            label="White matter mean",
            value=_fb_cfg.get("include_wm_regressor", _old_nuisance),
            tooltip="Include WM mean signal as GLM nuisance regressor. Mask from MNI priors.",
        )
        self.csf_regressor_checkbox = ft.Checkbox(
            label="CSF mean",
            value=_fb_cfg.get("include_csf_regressor", _old_nuisance),
            tooltip="Include CSF mean signal as GLM nuisance regressor. Mask from MNI priors.",
        )

        # Real-time spatial smoothing (checkbox + conditional FWHM field)
        rt_smooth_val = config.get("feedback", {}).get("smooth_fwhm")
        self.rt_smooth_enabled_checkbox = ft.Checkbox(
            label="Enable real-time spatial smoothing",
            value=rt_smooth_val is not None,
            on_change=self._on_rt_smooth_enabled_changed,
        )
        self.rt_smooth_fwhm_field = ft.TextField(
            width=600,
            label="Smoothing FWHM (mm)",
            value=str(rt_smooth_val) if rt_smooth_val is not None else "6.0",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Gaussian kernel size in mm (e.g., 5.0 or 6.0)",
            visible=rt_smooth_val is not None,
        )

        # Feedback signal mode (dropdown: activation vs connectivity)
        mask_names_for_diff = config.get("mask", {}).get("mask_names", [])
        diff_options = [ft.dropdown.Option(key=name, text=name) for name in mask_names_for_diff]
        has_multi_mask = len(mask_names_for_diff) >= 2
        feedback_mode_val = config.get("feedback", {}).get("feedback_mode", "activation")

        self.feedback_mode_dropdown = ft.Dropdown(
            width=600,
            label="Feedback signal",
            options=[
                ft.dropdown.Option("activation", "Activation (differential)"),
                ft.dropdown.Option("connectivity", "Connectivity (sliding-window correlation)"),
            ],
            value="activation" if feedback_mode_val == "activation_single" else feedback_mode_val,
            on_select=self._on_feedback_mode_changed,
        )

        # Differential pair (shown in activation mode)
        diff_pair = config.get("feedback", {}).get("differential_pair")
        self.differential_positive_dropdown = ft.Dropdown(
            width=250,
            label="Positive mask",
            options=diff_options,
            value=diff_pair[0]
            if diff_pair and len(diff_pair) == 2
            else (mask_names_for_diff[0] if mask_names_for_diff else None),
        )
        self.differential_negative_dropdown = ft.Dropdown(
            width=250,
            label="Negative mask",
            options=diff_options,
            value=diff_pair[1]
            if diff_pair and len(diff_pair) == 2
            else (mask_names_for_diff[1] if has_multi_mask else None),
        )
        self.differential_row = ft.Column(
            [
                ft.Text("Differential Signal", size=14, weight=ft.FontWeight.W_500),
                ft.Row(
                    [
                        self.differential_positive_dropdown,
                        ft.Text("minus", size=14),
                        self.differential_negative_dropdown,
                    ]
                ),
            ],
            visible=has_multi_mask and feedback_mode_val == "activation",
        )

        # Connectivity config (shown in connectivity mode)
        conn_config = config.get("feedback", {})
        conn_pairs = conn_config.get("connectivity_pairs")
        conn_window = conn_config.get("connectivity_window", 15)
        is_conn_mode = feedback_mode_val == "connectivity"

        conn_pair_options = [ft.dropdown.Option(key=name, text=name) for name in mask_names_for_diff]
        conn_pair_a = (
            conn_pairs[0][0]
            if conn_pairs and len(conn_pairs) > 0 and len(conn_pairs[0]) == 2
            else (mask_names_for_diff[0] if mask_names_for_diff else None)
        )
        conn_pair_b = (
            conn_pairs[0][1]
            if conn_pairs and len(conn_pairs) > 0 and len(conn_pairs[0]) == 2
            else (mask_names_for_diff[1] if has_multi_mask else None)
        )
        self.connectivity_pair_a_dropdown = ft.Dropdown(
            width=250,
            label="ROI A",
            options=conn_pair_options,
            value=conn_pair_a,
        )
        self.connectivity_pair_b_dropdown = ft.Dropdown(
            width=250,
            label="ROI B",
            options=conn_pair_options,
            value=conn_pair_b,
        )
        self.connectivity_window_field = ft.TextField(
            width=600,
            label="Sliding window size (volumes)",
            value=str(conn_window),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of volumes in correlation window (e.g., 10-30)",
        )
        self.connectivity_target_dropdown = ft.Dropdown(
            width=600,
            label="Target direction",
            options=[
                ft.dropdown.Option("increase", "Increase correlation"),
                ft.dropdown.Option("decrease", "Decrease correlation (anti-correlation)"),
            ],
            value=conn_config.get("connectivity_target", "increase"),
        )
        self.connectivity_section = ft.Column(
            [
                ft.Text("Connectivity Pair", size=14, weight=ft.FontWeight.W_500),
                ft.Row(
                    [
                        self.connectivity_pair_a_dropdown,
                        ft.Text("corr", size=14),
                        self.connectivity_pair_b_dropdown,
                    ]
                ),
                self.connectivity_window_field,
                self.connectivity_target_dropdown,
            ],
            visible=is_conn_mode,
        )

        # Intermittent feedback mode (checkbox + interval field)
        interval_val = config.get("feedback", {}).get("feedback_interval")
        self.intermittent_enabled_checkbox = ft.Checkbox(
            label="Enable intermittent feedback",
            value=interval_val is not None,
            on_change=self._on_intermittent_enabled_changed,
        )
        self.feedback_interval_field = ft.TextField(
            width=600,
            label="Feedback interval (volumes)",
            value=str(interval_val) if interval_val is not None else "5",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Deliver mean activation every N volumes after baseline",
            visible=interval_val is not None,
        )

        # Create dynamic task parameter fields for current feedback task type
        self._create_task_parameter_fields()

        # Paths fields
        self.base_dir_field = ft.TextField(
            width=540,
            label="Base data directory",
            value=config["paths"]["base_dir"],
        )

        # Simulator source paths - create dynamically based on n_runs
        localizer_dicoms = config["scan"].get("localizer_source_dicoms", ["", ""])
        # Ensure we have enough entries for max possible runs
        while len(localizer_dicoms) < 10:
            localizer_dicoms.append("")

        # Create localizer source fields (up to 10, visibility controlled dynamically)
        self.localizer_source_fields = []
        for i in range(10):
            field = ft.TextField(
                width=540,
                label=f"Localizer run {i + 1} source DICOMs",
                value=localizer_dicoms[i] if i < len(localizer_dicoms) else "",
                hint_text="Path to DICOM directory for simulator (optional)",
            )
            self.localizer_source_fields.append(field)

        self.reference_initial_field = ft.TextField(
            width=540,
            label="Initial Reference Scan source DICOMs",
            value=config["scan"].get("reference_initial_dicoms", ""),
            hint_text="Path to DICOM directory for simulator (optional)",
        )

        self.reference_realtime_field = ft.TextField(
            width=540,
            label="Real-Time Reference Scan source DICOMs",
            value=config["scan"].get("reference_realtime_dicoms", ""),
            hint_text="Path to DICOM directory for simulator (optional)",
        )

        # Feedback source paths - create dynamically based on max_feedback_runs
        feedback_dicoms = config["scan"].get("feedback_source_dicoms", ["", ""])
        # Handle legacy single-path format
        if isinstance(feedback_dicoms, str):
            feedback_dicoms = [feedback_dicoms, feedback_dicoms]
        # Ensure we have enough entries for max possible runs (10)
        while len(feedback_dicoms) < 10:
            feedback_dicoms.append("")

        # Create feedback source fields (up to 10, visibility controlled dynamically)
        self.feedback_source_fields = []
        for i in range(10):
            field = ft.TextField(
                width=540,
                label=f"Feedback run {i + 1} source DICOMs",
                value=feedback_dicoms[i] if i < len(feedback_dicoms) else "",
                hint_text="Path to DICOM directory for simulator (optional)",
            )
            self.feedback_source_fields.append(field)

        # Transfer source paths - create 2 fields (transfer_1 and transfer_2)
        transfer_dicoms = config["scan"].get("transfer_source_dicoms", ["", ""])
        # Handle legacy single-path format
        if isinstance(transfer_dicoms, str):
            transfer_dicoms = [transfer_dicoms, transfer_dicoms]
        # Ensure we have 2 entries
        while len(transfer_dicoms) < 2:
            transfer_dicoms.append("")

        # Create transfer source fields (always 2: transfer_1 and transfer_2)
        self.transfer_source_fields = []
        for i in range(2):
            field = ft.TextField(
                width=540,
                label=f"Transfer run {i + 1} source DICOMs",
                value=transfer_dicoms[i] if i < len(transfer_dicoms) else "",
                hint_text="Path to DICOM directory for simulator (optional)",
            )
            self.transfer_source_fields.append(field)

        # Simulator source mode toggle
        # Determine initial mode based on whether mapping file is set
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

        # Simulator source mapping file (for per-subject DICOM paths)
        self.simulator_source_mapping_field = ft.TextField(
            width=540,
            label="Mapping File",
            value=config["scan"].get("simulator_source_mapping", ""),
            hint_text="CSV/TSV file mapping subjects to DICOM paths",
        )

        # --- Sham/Control tab fields ---
        sham_config = config.get("sham", {})

        self.sham_enabled_checkbox = ft.Checkbox(
            label="Enable Sham/Control Conditions",
            value=sham_config.get("enabled", False),
            on_change=self._on_sham_enabled_changed,
        )

        self.blind_mode_checkbox = ft.Checkbox(
            label="Enable Blind Mode",
            value=sham_config.get("blind_mode", False),
        )

        self.sham_delivery_method_radio = ft.RadioGroup(
            content=ft.Column(
                [
                    ft.Radio(value="alternative_masks", label="Alternative Masks"),
                    ft.Radio(value="replay", label="Replay Previous Participant"),
                ]
            ),
            value=sham_config.get("delivery_method", "alternative_masks"),
            on_change=self._on_sham_method_changed,
        )

        # Sham mask mode radio (hemisphere_split vs two_networks)
        self.sham_mask_mode_radio = ft.RadioGroup(
            content=ft.Column(
                [
                    ft.Radio(value="hemisphere_split", label="Hemisphere Split (single network)"),
                    ft.Radio(value="two_networks", label="Two Distinct Networks"),
                ]
            ),
            value=sham_config.get("sham_mask_mode", "hemisphere_split"),
            on_change=self._on_sham_mask_mode_changed,
        )

        # Hemisphere split controls
        self.sham_hemisphere_network_dropdown = ft.Dropdown(
            width=600,
            label="Network to Split",
            options=self._get_yeo_network_options(config["mask"]["n_yeo"]),
            value=sham_config.get("hemisphere_network", "SomMotA"),
        )
        self.sham_target_hemisphere_radio = ft.RadioGroup(
            content=ft.Row(
                [
                    ft.Radio(value="right", label="Right (target)"),
                    ft.Radio(value="left", label="Left (target)"),
                ]
            ),
            value=sham_config.get("target_hemisphere", "right"),
        )

        # Sham alternative network dropdowns (for two_networks mode)
        sham_targets = sham_config.get("alternative_mask_targets", ["SomMotA", "DefaultA"])
        sham_first = sham_targets[0] if len(sham_targets) > 0 else None
        sham_second = sham_targets[1] if len(sham_targets) > 1 else None

        self.sham_first_network_dropdown = ft.Dropdown(
            width=290,
            label="First Sham Network",
            options=self._get_yeo_network_options(config["mask"]["n_yeo"]),
            value=sham_first,
        )
        self.sham_second_network_dropdown = ft.Dropdown(
            width=290,
            label="Second Sham Network",
            options=self._get_yeo_network_options(config["mask"]["n_yeo"]),
            value=sham_second,
        )

        self.sham_randomization_file_field = ft.TextField(
            width=540,
            label="Randomization File",
            value=sham_config.get("randomization_file", ""),
            hint_text="TSV file with subject_id, condition, matched_participant columns",
        )

        # --- Display settings ---
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

    def _build_monitor_preview(self) -> ft.Container:
        """Build a preview showing detected monitors."""
        try:
            from pineuro.display import get_monitors

            monitors = get_monitors()

            if not monitors:
                return ft.Container(
                    content=ft.Text(
                        "No monitors detected",
                        size=12,
                        color=ft.Colors.ORANGE_700,
                    ),
                    padding=10,
                )

            # Build monitor info text
            monitor_texts: list[ft.Control] = []
            for m in monitors:
                position = "Left" if m.index == 0 else "Right"
                monitor_texts.append(
                    ft.Text(
                        f"{position}: {m.name} ({m.width}x{m.height})",
                        size=12,
                        color=ft.Colors.GREY_700,
                    )
                )

            return ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Detected monitors:", size=12, weight=ft.FontWeight.BOLD),
                    ]
                    + monitor_texts,
                    spacing=2,
                ),
                bgcolor=ft.Colors.GREY_100,
                padding=10,
                border_radius=5,
            )
        except Exception as e:
            logger.warning(f"Failed to detect monitors: {e}")
            return ft.Container(
                content=ft.Text(
                    f"Monitor detection unavailable: {e}",
                    size=12,
                    color=ft.Colors.ORANGE_700,
                ),
                padding=10,
            )

    def _on_include_reverse_contrasts_changed(self, e):
        """Handle include reverse contrasts checkbox change."""
        enabled = self.include_reverse_contrasts_checkbox.value

        # Store in config
        self.study_config.session_config.setdefault("mask", {})["include_reverse_contrasts"] = enabled

        # Refresh feedback dropdowns with updated contrast key options
        self._update_feedback_dropdowns_for_localizer()

    def _get_yeo_network_options(self, n_networks: int) -> list:
        """Get network options for dropdowns based on network count.

        Args:
            n_networks: 7 or 17

        Returns:
            List of dropdown options
        """
        from ..utils import YEO_7_LABELS, YEO_17_LABELS

        labels = YEO_7_LABELS if n_networks == 7 else YEO_17_LABELS
        return [ft.dropdown.Option(key=label, text=label) for label in labels]

    def _update_feedback_dropdowns_for_localizer(self):
        """Update feedback dropdowns with contrast key options for task-based localizers.

        When the localizer is task-based, repopulate all network/mask dropdowns
        with contrast key names (plus _reverse suffixes when enabled) instead
        of Yeo network labels.
        """
        import contextlib

        config = self.study_config.session_config
        localizer_task_type = config.get("scan", {}).get("localizer_task_type", "rest")
        if localizer_task_type == "rest":
            return

        # Build contrast key options from task metadata
        contrasts = config.get("mask", {}).get("contrasts", {})
        if not contrasts:
            from pineuro.tasks.task_registry import get_task_metadata

            try:
                metadata = get_task_metadata(localizer_task_type)
                contrasts = metadata.default_contrasts or {}
            except (KeyError, ValueError):
                return

        names = list(contrasts.keys())
        if config.get("mask", {}).get("include_reverse_contrasts", False):
            names = names + [f"{n}_reverse" for n in names]

        new_options = [ft.dropdown.Option(key=n, text=n) for n in names]

        # Update all feedback-related dropdowns
        for dropdown in [
            self.first_network_dropdown,
            self.second_network_dropdown,
            self.sham_first_network_dropdown,
            self.sham_second_network_dropdown,
            self.sham_hemisphere_network_dropdown,
            self.differential_positive_dropdown,
            self.differential_negative_dropdown,
            self.connectivity_pair_a_dropdown,
            self.connectivity_pair_b_dropdown,
        ]:
            old_value = dropdown.value
            dropdown.options = new_options
            valid_keys = [opt.key for opt in new_options]
            if old_value not in valid_keys:
                dropdown.value = None

        with contextlib.suppress(Exception):
            self.page.update()

    def _on_network_atlas_changed(self, e):
        """Handle network atlas change - update all network dropdowns."""
        from pineuro.mask_extraction import ATLAS_SOURCES

        atlas_key = self.network_atlas_dropdown.value
        n_networks = ATLAS_SOURCES.get(atlas_key, {}).get("n_networks", 17)
        new_options = self._get_yeo_network_options(n_networks)

        # Update all network dropdowns with new options
        for dropdown in [
            self.first_network_dropdown,
            self.second_network_dropdown,
            self.sham_first_network_dropdown,
            self.sham_second_network_dropdown,
            self.sham_hemisphere_network_dropdown,
            self.differential_positive_dropdown,
            self.differential_negative_dropdown,
            self.connectivity_pair_a_dropdown,
            self.connectivity_pair_b_dropdown,
        ]:
            old_value = dropdown.value
            dropdown.options = new_options
            valid_keys = [opt.key for opt in new_options]
            if old_value not in valid_keys:
                dropdown.value = None

        # Update feedback mode dropdown based on mask count
        n_networks = len(new_options)
        if n_networks < 2:
            self.feedback_mode_dropdown.options = [
                ft.dropdown.Option("activation", "Activation (differential)"),
            ]
            if self.feedback_mode_dropdown.value not in ("activation",):
                self.feedback_mode_dropdown.value = "activation"
            self.differential_row.visible = False
            self.connectivity_section.visible = False
        else:
            self.feedback_mode_dropdown.options = [
                ft.dropdown.Option("activation", "Activation (differential)"),
                ft.dropdown.Option("connectivity", "Connectivity (sliding-window correlation)"),
            ]

        self.page.update()

    def _on_atlas_source_changed(self, e):
        """Handle cluster atlas source change - populate region dropdown on demand."""
        import threading

        atlas_source = self.atlas_source_dropdown.value
        if not atlas_source:
            return

        # Gray matter uses whole-brain — no region selection needed
        if atlas_source == "gray_matter":
            self.atlas_region_dropdown.options = []
            self.atlas_region_dropdown.value = None
            self.atlas_region_dropdown.visible = False
            with contextlib.suppress(Exception):
                self.page.update()
            return

        self.atlas_region_dropdown.visible = True

        # Show loading state
        self.atlas_region_dropdown.options = [ft.dropdown.Option(key="__loading__", text="Loading regions...")]
        self.atlas_region_dropdown.value = "__loading__"
        with contextlib.suppress(Exception):
            self.page.update()

        def _fetch_and_update():
            try:
                from pineuro.mask_extraction import get_atlas_labels

                labels = get_atlas_labels(atlas_source)
                self.atlas_region_dropdown.options = [ft.dropdown.Option(key=lbl, text=lbl) for lbl in labels]
                self.atlas_region_dropdown.value = labels[0] if labels else None
            except Exception as ex:
                logger.warning(f"Failed to load atlas labels for {atlas_source}: {ex}")
                self.atlas_region_dropdown.options = []
                self.atlas_region_dropdown.value = None
            with contextlib.suppress(Exception):
                self.page.update()

        # Fetch labels in background thread to avoid blocking UI
        threading.Thread(target=_fetch_and_update, daemon=True).start()

    def _validate_field(self, field_name: str, field: ft.TextField, validator):
        """Validate a form field and update UI.

        Args:
            field_name: Field identifier
            field: TextField control
            validator: Validation function
        """
        try:
            validator(field.value, field_name)
            field.error_text = None  # type: ignore[attr-defined]
            if field_name in self.validation_errors:
                del self.validation_errors[field_name]
        except ValidationError as e:
            field.error_text = str(e)  # type: ignore[attr-defined]
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
                        tabs=[
                            ft.Tab(label="Windows"),
                            ft.Tab(label="macOS"),
                            ft.Tab(label="Linux"),
                        ],
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

    def _on_simulator_mode_changed(self, e):
        """Handle simulator mode checkbox change."""
        is_simulator = self.simulator_enabled_checkbox.value

        # Update visibility of simulator-specific fields
        self.collection_speed_field.visible = is_simulator
        self.feedback_simulator_speed_field.visible = is_simulator
        self.scanner_dicom_dir_field.visible = not is_simulator
        self.scanner_dicom_dir_browse_btn.visible = not is_simulator
        self.scanner_ip_field.visible = not is_simulator
        self.scanner_setup_guide_btn.visible = not is_simulator

        # Update simulator instruction text
        if hasattr(self, "simulator_instruction_text"):
            if is_simulator:
                self.simulator_instruction_text.value = (
                    "Simulator replays volumes from source directories at configurable speed for offline testing."
                )
            else:
                self.simulator_instruction_text.value = (
                    "Scanner DICOM directory will be monitored for real-time acquisition."
                )

        # Update simulator paths container visibility
        if hasattr(self, "simulator_paths_container"):
            self.simulator_paths_container.visible = is_simulator

        self.page.update()

    def _on_simulator_source_mode_changed(self, e):
        """Handle simulator source mode radio change."""
        self._update_simulator_source_visibility()

    def _update_simulator_source_visibility(self):
        """Update visibility of simulator source sections based on selected mode."""
        mode = self.simulator_source_mode_radio.value

        if hasattr(self, "simulator_mapping_container"):
            self.simulator_mapping_container.visible = mode == "mapping"
        if hasattr(self, "simulator_individual_container"):
            self.simulator_individual_container.visible = mode == "individual"

        self.page.update()

    def _on_motion_correction_changed(self, e):
        """Disable motion regressors checkbox when motion correction is 'none'."""
        is_none = self.motion_correction_dropdown.value == "none"
        self.motion_regressors_checkbox.disabled = is_none
        if is_none:
            self.motion_regressors_checkbox.value = False
        self.page.update()

    def _on_rt_smooth_enabled_changed(self, e):
        """Toggle visibility of the smoothing FWHM field."""
        self.rt_smooth_fwhm_field.visible = self.rt_smooth_enabled_checkbox.value
        self.page.update()

    def _on_intermittent_enabled_changed(self, e):
        """Toggle visibility of the intermittent feedback interval field."""
        self.feedback_interval_field.visible = self.intermittent_enabled_checkbox.value
        self.page.update()

    def _on_feedback_mode_changed(self, e):
        """Toggle visibility of activation vs connectivity feedback fields."""
        mode = self.feedback_mode_dropdown.value
        self.differential_row.visible = mode == "activation"
        self.connectivity_section.visible = mode == "connectivity"
        self._update_sham_visibility()
        self.page.update()

    def _on_sham_enabled_changed(self, e):
        """Handle sham enabled checkbox change."""
        self._update_sham_visibility()

    def _on_sham_method_changed(self, e):
        """Handle sham delivery method change."""
        self._update_sham_visibility()

    def _on_sham_mask_mode_changed(self, e):
        """Handle sham mask mode change (hemisphere_split vs two_networks)."""
        self._update_sham_visibility()

    def _on_generate_randomization(self, e):
        """Open dialog to generate a randomization file."""
        prefix_field = ft.TextField(
            label="Participant ID prefix",
            value="sub-",
            width=200,
            hint_text="e.g. sub- → sub-01, sub-rtchr0 → sub-rtchr001",
        )
        n_participants_field = ft.TextField(
            label="Number of participants",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        n_sham_field = ft.TextField(
            label="Number of sham",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        method_dropdown = ft.Dropdown(
            label="Method",
            width=200,
            options=[
                ft.dropdown.Option(key="simple", text="Simple"),
                ft.dropdown.Option(key="replay", text="Replay"),
            ],
            value="simple",
        )
        seed_field = ft.TextField(
            label="Seed (optional)",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Generate Randomization File"),
            content=ft.Container(
                content=ft.Column(
                    [
                        prefix_field,
                        n_participants_field,
                        n_sham_field,
                        method_dropdown,
                        seed_field,
                    ],
                    tight=True,
                    spacing=10,
                ),
                width=300,
            ),
        )

        def generate(e):
            # Validate
            prefix = prefix_field.value.strip()
            if not prefix:
                prefix_field.error_text = "Required"
                prefix_field.update()
                return

            try:
                n_total = int(n_participants_field.value)
                if n_total < 2:
                    raise ValueError
            except (TypeError, ValueError):
                n_participants_field.error_text = "Must be an integer >= 2"
                n_participants_field.update()
                return
            n_participants_field.error_text = None

            try:
                n_sham = int(n_sham_field.value)
                if n_sham < 1 or n_sham >= n_total:
                    raise ValueError
            except (TypeError, ValueError):
                n_sham_field.error_text = (
                    f"Must be between 1 and {n_total - 1}" if "n_total" in dir() else "Must be a positive integer"
                )
                n_sham_field.update()
                return
            n_sham_field.error_text = None

            seed_val = None
            if seed_field.value and seed_field.value.strip():
                try:
                    seed_val = int(seed_field.value)
                except ValueError:
                    seed_field.error_text = "Must be an integer"
                    seed_field.update()
                    return

            # Ask user where to save
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                logger.error("tkinter is not available for file dialog")
                return

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            save_path = filedialog.asksaveasfilename(
                title="Save randomization file",
                defaultextension=".tsv",
                filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv")],
            )
            root.destroy()

            if not save_path:
                return  # User cancelled

            # Generate
            from pineuro.sham import generate_randomization

            participant_ids = [f"{prefix}{i:02d}" for i in range(1, n_total + 1)]
            method = method_dropdown.value

            generate_randomization(
                participant_ids=participant_ids,
                n_sham=n_sham,
                method=method,
                output_path=save_path,
                seed=seed_val,
            )

            # Populate file field and close dialog
            self.sham_randomization_file_field.value = save_path
            self.sham_randomization_file_field.update()
            self.page.pop_dialog()

            _snack = ft.SnackBar(ft.Text(f"Randomization file saved: {save_path}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        def cancel(e):
            self.page.pop_dialog()

        dialog.actions = [
            ft.TextButton("Cancel", on_click=cancel),
            ft.ElevatedButton("Generate", on_click=generate),
        ]
        self.page.show_dialog(dialog)

    def _update_sham_visibility(self):
        """Update visibility of sham sections based on enabled state and delivery method."""
        is_enabled = self.sham_enabled_checkbox.value
        method = self.sham_delivery_method_radio.value
        mask_mode = self.sham_mask_mode_radio.value
        is_connectivity = (self.feedback_mode_dropdown.value or "activation") == "connectivity"

        # Update visibility of sham configuration fields
        if hasattr(self, "sham_config_container"):
            self.sham_config_container.visible = is_enabled
        if hasattr(self, "sham_alternative_section"):
            self.sham_alternative_section.visible = is_enabled and (method == "alternative_masks")
        if hasattr(self, "sham_hemisphere_split_section"):
            self.sham_hemisphere_split_section.visible = (
                is_enabled and method == "alternative_masks" and mask_mode == "hemisphere_split"
            )
        if hasattr(self, "sham_two_networks_section"):
            self.sham_two_networks_section.visible = (
                is_enabled and method == "alternative_masks" and mask_mode == "two_networks"
            )
        if hasattr(self, "sham_replay_section"):
            self.sham_replay_section.visible = is_enabled and (method == "replay")
        if hasattr(self, "sham_randomization_section"):
            self.sham_randomization_section.visible = is_enabled

        # Target hemisphere is irrelevant for connectivity (correlation is symmetric)
        if hasattr(self, "sham_target_hemisphere_container"):
            self.sham_target_hemisphere_container.visible = not is_connectivity

        # Update description text based on feedback mode
        if hasattr(self, "sham_hemisphere_description"):
            if is_connectivity:
                self.sham_hemisphere_description.value = (
                    "Split the selected network into left and right hemispheres. "
                    "Compute correlation between hemispheres for sham participants."
                )
            else:
                self.sham_hemisphere_description.value = (
                    "Split the selected network into left and right hemispheres. "
                    "One hemisphere is target, the other is non-target."
                )

        self.page.update()

    def _create_localizer_task_parameter_fields(self):
        """Create form fields for localizer task type's parameters (when not rest)."""
        localizer_task_type = self.localizer_task_type_dropdown.value

        self.localizer_task_param_fields.clear()

        if localizer_task_type == "rest":
            return  # No task parameters for rest

        try:
            metadata = get_task_metadata(localizer_task_type)
        except ValueError:
            return

        config = self.study_config.session_config
        localizer_task_config = config.get("localizer_task", {}).get(localizer_task_type, {})

        for param in metadata.parameters:
            field = self._create_field_for_parameter(param, localizer_task_config)
            self.localizer_task_param_fields[param.key] = field

    def _create_task_parameter_fields(self):
        """Create form fields for current feedback task type's parameters."""
        # Read from config (source of truth) rather than dropdown
        config = self.study_config.session_config
        task_type = config["task"].get("task_type", "ball")
        metadata = get_task_metadata(task_type)

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
            # Add special change handler for use_transfer_runs
            on_change = None
            if param.key == "use_transfer_runs":
                on_change = self._on_use_transfer_runs_changed
            return ft.Checkbox(
                label=param.label,
                value=current_value,
                on_change=on_change,
            )

        elif param.type in ["float", "int"]:
            return ft.TextField(
                width=600,
                label=param.label,
                value=str(current_value),
                keyboard_type=ft.KeyboardType.NUMBER,
                hint_text=param.hint,
            )

        elif param.type == "dir":
            field = ft.TextField(
                width=600,
                label=param.label,
                value=current_value,
                hint_text=param.hint,
            )
            return ft.Row([field, self._create_browse_button(field)])

        elif param.type == "str":
            if param.choices:
                return ft.Dropdown(
                    width=600,
                    label=param.label,
                    value=current_value,
                    hint_text=param.hint,
                    options=[ft.dropdown.Option(c) for c in param.choices],
                )
            return ft.TextField(
                width=600,
                label=param.label,
                value=current_value,
                hint_text=param.hint,
            )

        elif param.type == "list[str]":
            # Convert list to comma-separated string
            value_str = ", ".join(current_value) if isinstance(current_value, list) else current_value
            return ft.TextField(
                width=600,
                label=param.label,
                value=value_str,
                hint_text=f"{param.hint} (comma-separated)" if param.hint else "Comma-separated values",
            )

        # Fallback for unsupported parameter types
        return ft.TextField(
            width=600,
            label=param.label,
            value=str(current_value) if current_value is not None else "",
            hint_text=param.hint,
        )

    def _on_task_type_changed(self, e):
        """Handle feedback task type dropdown change.

        Updates network dropdown visibility and task-specific parameters
        without rebuilding tabs (which would corrupt field values).
        """
        task_type = e.control.value if e and hasattr(e, "control") else self.task_type_dropdown.value
        logger.info(f"Feedback task type changed to: {task_type}")

        # Store task type in config as source of truth
        self.study_config.session_config["task"]["task_type"] = task_type

        # Get task metadata to determine n_networks
        try:
            task_metadata = get_task_metadata(task_type)
            n_networks = task_metadata.n_networks
        except ValueError:
            task_metadata = None
            n_networks = 2

        # Update network dropdown visibility in Localizer tab
        if hasattr(self, "second_network_dropdown"):
            self.second_network_dropdown.visible = n_networks >= 2

        # Update differential row visibility (only for 2+ masks)
        if hasattr(self, "differential_row"):
            self.differential_row.visible = n_networks >= 2

        # Update target networks label in Localizer tab
        if hasattr(self, "target_networks_label"):
            self.target_networks_label.value = f"Target Networks (select {n_networks})"

        # Update task info text in Study tab
        if hasattr(self, "task_info_text"):
            task_info = f"Requires {n_networks} {'network' if n_networks == 1 else 'networks'}"
            self.task_info_text.value = task_info

        # Rebuild network dropdowns row based on n_networks
        if hasattr(self, "network_dropdowns_row"):
            network_dropdowns = [self.first_network_dropdown]
            if n_networks >= 2:
                network_dropdowns.append(self.second_network_dropdown)
            self.network_dropdowns_row.controls.clear()
            self.network_dropdowns_row.controls.extend(network_dropdowns)
            with contextlib.suppress(Exception):
                self.network_dropdowns_row.update()

        # Recreate task parameter fields for new task type
        self._create_task_parameter_fields()

        # Update task-specific section content in Feedback tab
        if hasattr(self, "task_specific_section"):
            task_specific_controls = self._build_task_specific_params()
            self.task_specific_section.content = ft.Column(task_specific_controls)
            # Update visibility based on whether task has parameters
            has_parameters = task_metadata and len(task_metadata.parameters) > 0
            self.task_specific_section.visible = has_parameters

        # Show snackbar feedback
        if task_metadata:
            msg = f"Task: {task_metadata.display_name} (requires {n_networks} {'network' if n_networks == 1 else 'networks'})"
        else:
            msg = f"Task changed to {task_type}"

        _snack = ft.SnackBar(ft.Text(msg))
        _snack.open = True
        self.page.overlay.append(_snack)
        self.page.update()

    def _get_contrast_info_text(self, task_type: str) -> str:
        """Get contrast info string for the selected localizer task."""
        if task_type == "rest":
            return ""
        try:
            from pineuro.tasks.task_registry import get_task_metadata

            metadata = get_task_metadata(task_type)
            if metadata.default_contrasts:
                parts = [f"{name}: {expr}" for name, expr in metadata.default_contrasts.items()]
                return f"GLM contrast: {', '.join(parts)}"
        except KeyError:
            pass
        return ""

    def _on_localizer_task_type_changed(self, e):
        """Handle localizer task type dropdown change.

        Updates network section visibility without rebuilding tabs
        (which would corrupt field values).
        """
        localizer_task_type = (
            e.control.value if e and hasattr(e, "control") else self.localizer_task_type_dropdown.value
        )
        logger.info(f"Localizer task type changed to: {localizer_task_type}")

        # Store in config as source of truth
        self.study_config.session_config["scan"]["localizer_task_type"] = localizer_task_type

        # Update mask.localizer_type to match (used by participant_workflow and pipeline_runner)
        is_rest = localizer_task_type == "rest"
        self.study_config.session_config["mask"]["localizer_type"] = "rest" if is_rest else "task"

        # Toggle network extraction section visibility
        if hasattr(self, "networks_section"):
            self.networks_section.visible = is_rest
            with contextlib.suppress(Exception):
                self.networks_section.update()

        # Update contrast info
        if hasattr(self, "contrast_info_text"):
            self.contrast_info_text.value = self._get_contrast_info_text(localizer_task_type)
            with contextlib.suppress(Exception):
                self.contrast_info_text.update()

        # Toggle reverse contrasts checkbox visibility (task-based only)
        if hasattr(self, "include_reverse_contrasts_checkbox"):
            self.include_reverse_contrasts_checkbox.visible = not is_rest
            with contextlib.suppress(Exception):
                self.include_reverse_contrasts_checkbox.update()

        # Refresh feedback dropdowns with contrast keys when task-based
        self._update_feedback_dropdowns_for_localizer()

        # Recreate localizer task parameter fields
        self._create_localizer_task_parameter_fields()

        # Update localizer task parameters section
        if hasattr(self, "localizer_task_params_section"):
            task_params = self._build_localizer_task_specific_params()
            self.localizer_task_params_section.controls.clear()
            self.localizer_task_params_section.controls.extend(task_params)
            with contextlib.suppress(Exception):
                self.localizer_task_params_section.update()

        # Show snackbar feedback
        msg = (
            "Localizer: Rest (ICA-based network extraction)"
            if is_rest
            else f"Localizer: {localizer_task_type} (GLM-based)"
        )
        _snack = ft.SnackBar(ft.Text(msg))
        _snack.open = True
        self.page.overlay.append(_snack)
        self.page.update()

    def _on_threshold_method_changed(self, e):
        """Handle threshold method dropdown change."""
        method = e.control.value if e and hasattr(e, "control") else self.threshold_method_dropdown.value
        is_cluster = method == "cluster"
        if hasattr(self, "cluster_settings_section"):
            self.cluster_settings_section.visible = is_cluster
            with contextlib.suppress(Exception):
                self.cluster_settings_section.update()

    def _on_localizer_n_runs_changed(self, e):
        """Handle localizer n_runs field change - update paths tab visibility."""
        # Validate the field first
        self._validate_field("localizer_n_runs", self.localizer_n_runs_field, validate_positive_int)

        # Update localizer source field visibility if paths tab has been built
        if hasattr(self, "localizer_source_rows"):
            try:
                n_runs = int(self.localizer_n_runs_field.value)
            except (ValueError, TypeError):
                n_runs = 2

            for i, row in enumerate(self.localizer_source_rows):
                row.visible = i < n_runs
                row.update()

    def _on_reference_initial_enabled_changed(self, e):
        """Handle reference initial enable checkbox change - update paths tab."""
        enabled = self.reference_initial_enabled_checkbox.value

        # Update n_volumes field visibility in localizer tab
        self.reference_initial_n_volumes_field.visible = enabled
        self.reference_initial_n_volumes_field.update()

        # Update paths tab row visibility if it has been built
        if hasattr(self, "reference_initial_row"):
            self.reference_initial_row.visible = enabled
            self.reference_initial_row.update()

    def _on_reference_realtime_enabled_changed(self, e):
        """Handle reference realtime enable checkbox change - update paths tab and note."""
        enabled = self.reference_realtime_enabled_checkbox.value

        # Update n_volumes field visibility in feedback tab
        self.reference_realtime_n_volumes_field.visible = enabled
        self.reference_realtime_n_volumes_field.update()

        # Update paths tab row visibility if it has been built
        if hasattr(self, "reference_realtime_row"):
            self.reference_realtime_row.visible = enabled
            self.reference_realtime_row.update()

        # Show/hide note when disabled
        if hasattr(self, "reference_realtime_note"):
            self.reference_realtime_note.visible = not enabled
            self.reference_realtime_note.update()

    def _on_max_feedback_runs_changed(self, e):
        """Handle max_feedback_runs field change - update paths tab visibility."""
        # Validate the field first
        self._validate_field("max_feedback_runs", self.max_feedback_runs_field, validate_positive_int)

        # Update feedback source field visibility if paths tab has been built
        if hasattr(self, "feedback_source_rows"):
            try:
                n_runs = int(self.max_feedback_runs_field.value)
            except (ValueError, TypeError):
                n_runs = 2

            for i, row in enumerate(self.feedback_source_rows):
                row.visible = i < n_runs
                row.update()

    def _on_use_transfer_runs_changed(self, e):
        """Handle use_transfer_runs checkbox change - update paths tab visibility."""
        enabled = e.control.value if e and hasattr(e, "control") else False

        # Update transfer source rows and header visibility if paths tab has been built
        if hasattr(self, "transfer_source_rows"):
            for row in self.transfer_source_rows:
                row.visible = enabled
                row.update()

        if hasattr(self, "transfer_section_header"):
            self.transfer_section_header.visible = enabled
            self.transfer_section_header.update()

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
                _snack = ft.SnackBar(ft.Text("File browser requires tkinter. Install: sudo apt-get install python3-tk"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                return

            try:
                # Create a temporary tkinter root window
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.attributes("-topmost", True)  # Bring to front

                # Open directory picker
                path = filedialog.askdirectory(
                    title=f"Select {field.label}", initialdir=field.value if field.value else None
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
                _snack = ft.SnackBar(ft.Text(f"Error opening file browser: {ex}"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()

        return ft.IconButton(
            icon=ft.icons.Icons.FOLDER_OPEN,
            tooltip="Browse for folder",
            on_click=pick_directory,
        )

    def _create_file_browse_button(
        self,
        field: ft.TextField,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> ft.IconButton:
        """Create a browse button for file selection.

        Args:
            field: TextField to update with selected file path
            filetypes: List of (description, pattern) tuples for file filter.
                       Example: [("NIfTI Files", "*.nii"), ("All Files", "*.*")]

        Returns:
            IconButton that opens file picker
        """
        if filetypes is None:
            filetypes = [("All Files", "*.*")]

        def pick_file(e):
            """Handle file picker click event."""
            logger.info(f"File picker clicked for field: {field.label}")

            # Use tkinter file dialog as a workaround for Flet FilePicker issues on Linux
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                logger.error("tkinter is not available")
                _snack = ft.SnackBar(ft.Text("File browser requires tkinter. Install: sudo apt-get install python3-tk"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                return

            try:
                # Create a temporary tkinter root window
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                root.attributes("-topmost", True)  # Bring to front

                # Get initial directory from current value
                import os

                initial_dir = None
                if field.value:
                    if os.path.isfile(field.value):
                        initial_dir = os.path.dirname(field.value)
                    elif os.path.isdir(field.value):
                        initial_dir = field.value

                # Open file picker
                path = filedialog.askopenfilename(
                    title=f"Select {field.label}",
                    initialdir=initial_dir,
                    filetypes=filetypes,
                )

                # Clean up
                root.destroy()

                logger.info(f"Selected file: {path}")
                if path:
                    field.value = path
                    field.update()

            except Exception as ex:
                logger.error(f"Error opening file picker: {ex}")
                # Show error to user
                _snack = ft.SnackBar(ft.Text(f"Error opening file browser: {ex}"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()

        return ft.IconButton(
            icon=ft.icons.Icons.FILE_OPEN,
            tooltip="Browse for file",
            on_click=pick_file,
        )

    def build(self) -> ft.Control:
        """Build the study editor UI.

        Returns:
            Flet control for the study editor
        """
        # Create tab headers (Networks and Sham integrated into Feedback tab)
        self.tab_bar = ft.TabBar(
            tabs=[
                ft.Tab(label="Study"),
                ft.Tab(label="Localizer"),
                ft.Tab(label="Feedback"),
                ft.Tab(label="Paths"),
            ],
        )

        # Create tab content panels - store reference for dynamic rebuilding
        self.tab_view = ft.TabBarView(
            controls=[
                self._build_study_tab(),
                self._build_localizer_tab(),
                self._build_feedback_tab(),
                self._build_paths_tab(),
            ],
            expand=True,
        )

        # Wrap TabBar and TabBarView in a Column for Tabs
        # Store reference for dynamic tab rebuilding when task type changes
        self.tabs_content = ft.Column(
            [self.tab_bar, self.tab_view],
            spacing=0,
            expand=True,
        )

        # Create Tabs control
        tabs = ft.Tabs(
            content=self.tabs_content,
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

        # Populate feedback dropdowns with contrast keys for task-based localizers
        self._update_feedback_dropdowns_for_localizer()

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

    def _build_study_tab(self) -> ft.Container:
        """Build the study information, experiment design, and acquisition mode tab.

        Task selection is placed here (early) so subsequent tabs can be built
        based on the selected task's requirements (e.g., n_networks).
        """
        # Create browse button for scanner DICOM directory
        self.scanner_dicom_dir_browse_btn = self._create_browse_button(self.scanner_dicom_dir_field)

        # Set initial visibility based on simulator mode
        is_simulator = self.simulator_enabled_checkbox.value
        self.collection_speed_field.visible = is_simulator
        self.feedback_simulator_speed_field.visible = is_simulator
        self.scanner_dicom_dir_field.visible = not is_simulator
        self.scanner_dicom_dir_browse_btn.visible = not is_simulator
        self.scanner_ip_field.visible = not is_simulator
        self.scanner_setup_guide_btn.visible = not is_simulator

        # Get task info for display
        task_type = self.task_type_dropdown.value
        try:
            task_metadata = get_task_metadata(task_type)
            task_info = (
                f"Requires {task_metadata.n_networks} {'network' if task_metadata.n_networks == 1 else 'networks'}"
            )
        except ValueError:
            task_info = ""

        # Create task info text control (will be updated dynamically)
        self.task_info_text = ft.Text(
            task_info,
            size=11,
            color=ft.Colors.GREY_600,
            italic=True,
        )

        # Create simulator mode instruction text (will be updated dynamically)
        self.simulator_instruction_text = ft.Text(
            "Simulator replays volumes from source directories at configurable speed for offline testing."
            if is_simulator
            else "Scanner DICOM directory will be monitored for real-time acquisition.",
            size=11,
            color=ft.Colors.GREY_600,
            italic=True,
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
                    ft.Text("Localizer Task", size=14, weight=ft.FontWeight.BOLD),
                    self.localizer_task_type_dropdown,
                    self.contrast_info_text,
                    ft.Text(
                        "Rest uses ICA for network extraction. Task-based localizers use GLM contrasts.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Text("Feedback Task", size=14, weight=ft.FontWeight.BOLD),
                    self._f(self.task_type_dropdown),
                    self.task_info_text,
                    ft.Divider(),
                    self.multi_visit_enabled_checkbox,
                    ft.Text(
                        "When enabled, participants can have multiple visits (ses-01, ses-02, etc.). "
                        "Masks from the first visit can be reused in subsequent visits.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Divider(),
                    ft.Text("Acquisition Mode", size=16, weight=ft.FontWeight.BOLD),
                    self.simulator_enabled_checkbox,
                    ft.Row(
                        [
                            self.collection_speed_field,
                            self.feedback_simulator_speed_field,
                        ]
                    ),
                    ft.Row(
                        [
                            self.scanner_dicom_dir_field,
                            self.scanner_dicom_dir_browse_btn,
                        ]
                    ),
                    ft.Row(
                        [
                            self.scanner_ip_field,
                            self.scanner_setup_guide_btn,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.simulator_instruction_text,
                    ft.Divider(),
                    ft.Text("Display Settings", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        [
                            self.control_monitor_dropdown,
                            self.presenter_monitor_dropdown,
                        ]
                    ),
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

    def _build_localizer_tab(self) -> ft.Container:
        """Build the localizer parameters tab with network configuration.

        Network extraction settings are here because they are part of the
        localizer analysis phase. The number of network dropdowns shown
        depends on the task selected in the Study tab (n_networks).
        """
        # Build localizer task-specific parameters section
        localizer_task_params = self._build_localizer_task_specific_params()

        # Get task metadata to determine n_networks
        # Read from config (source of truth) rather than dropdown
        task_type = self.study_config.session_config["task"].get("task_type", "ball")
        try:
            task_metadata = get_task_metadata(task_type)
            n_networks = task_metadata.n_networks
        except ValueError:
            n_networks = 2  # Default fallback

        # Only show network section for rest localizer (ICA-based)
        is_rest_localizer = self.localizer_task_type_dropdown.value == "rest"

        # Store the target networks label so it can be updated dynamically
        self.target_networks_label = ft.Text(
            f"Target Networks (select {n_networks})",
            size=14,
            weight=ft.FontWeight.BOLD,
        )

        # Set second network dropdown visibility based on n_networks
        self.second_network_dropdown.visible = n_networks >= 2

        # Build network dropdowns row dynamically based on n_networks
        network_dropdowns = [self.first_network_dropdown]
        if n_networks >= 2:
            network_dropdowns.append(self.second_network_dropdown)

        self.network_dropdowns_row = ft.Row(network_dropdowns)

        # Network Configuration section (only visible when localizer is rest)
        self.networks_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Network Extraction", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "ICA-based network extraction from resting-state localizer",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Divider(),
                    ft.Text("Network Atlas", size=14, weight=ft.FontWeight.BOLD),
                    self.network_atlas_dropdown,
                    ft.Divider(),
                    self.target_networks_label,
                    ft.Text(
                        "Select target networks for neurofeedback training.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self.network_dropdowns_row,
                    ft.Divider(),
                    ft.Text("ICA Settings", size=14, weight=ft.FontWeight.BOLD),
                    self.n_components_field,
                    self.smooth_fwhm_field,
                    self.highpass_field,
                    ft.Divider(),
                ]
            ),
            visible=is_rest_localizer,
        )

        # Task-specific parameters section (wrapped in container for dynamic updates)
        self.localizer_task_params_section = ft.Column(
            localizer_task_params,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Scan Parameters", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Configure the localizer scan that identifies participant-specific brain networks.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self._f(self.localizer_tr_field),
                    self.localizer_n_volumes_field,
                    self.localizer_n_runs_field,
                    ft.Divider(),
                    self.networks_section,
                    self.localizer_task_params_section,
                    self.include_reverse_contrasts_checkbox,
                    ft.Text("Mask Extraction", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Configure how network masks are extracted from the stat maps.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self.threshold_field,
                    self.threshold_method_dropdown,
                    self.cluster_settings_section,
                    ft.Divider(),
                    ft.Text("Initial Reference Scan", size=16, weight=ft.FontWeight.BOLD),
                    self.reference_initial_enabled_checkbox,
                    self.reference_initial_n_volumes_field,
                    ft.Text(
                        "Collects a quick reference scan before localizer runs for positioning verification.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Divider(),
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    def _build_localizer_task_specific_params(self) -> list:
        """Build localizer task-specific parameter controls.

        Returns:
            List of Flet controls for localizer task parameters
        """
        localizer_task_type = self.localizer_task_type_dropdown.value

        if localizer_task_type == "rest":
            return []  # No task parameters for rest

        try:
            metadata = get_task_metadata(localizer_task_type)
        except ValueError:
            return [
                ft.Text(
                    f"Unknown task type: {localizer_task_type}",
                    size=14,
                    color=ft.Colors.ORANGE_700,
                ),
            ]

        controls = [
            ft.Text(
                f"{metadata.display_name} Localizer Parameters",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            ft.Text(
                "Task-specific parameters for the localizer. Configure stimulus paths and task settings.",
                size=11,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            ft.Divider(),
        ]

        # Add each parameter field
        for param in metadata.parameters:
            field = self.localizer_task_param_fields.get(param.key)
            if field is not None:
                controls.append(field)

                if param.hint and param.type != "bool":
                    controls.append(
                        ft.Text(
                            param.hint,
                            size=11,
                            color=ft.Colors.GREY_600,
                            italic=True,
                        )
                    )

        return controls

    def _build_feedback_tab(self) -> ft.Container:
        """Build the feedback session parameters tab.

        Task selection is in the Study tab. Network config is in the Localizer tab.
        This tab focuses on feedback session runtime settings.

        Order follows workflow sequence:
        1. Scan Parameters
        2. Session Structure
        3. Real-Time Reference Scan
        4. Task-Specific Parameters
        5. Sham/Control Conditions
        """
        # Scanner parameters for feedback
        scanner_section = [
            ft.Text("Scan Parameters", size=16, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Configure feedback run parameters (must match scanner sequence settings).",
                size=11,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            self._f(self.feedback_tr_field),
            self.feedback_n_volumes_field,
            ft.Divider(),
        ]

        # Get task metadata to determine whether task has parameters
        # Read from config (source of truth) rather than dropdown
        task_type = self.study_config.session_config["task"].get("task_type", "ball")
        try:
            task_metadata = get_task_metadata(task_type)
            has_parameters = len(task_metadata.parameters) > 0
        except ValueError:
            has_parameters = True

        # Session structure - generic for all tasks
        self.session_structure_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Session Structure", size=16, weight=ft.FontWeight.BOLD),
                    self.max_feedback_runs_field,
                    self.n_baseline_field,
                    self._f(self.fd_threshold_field),
                    self._f(self.dvars_threshold_field),
                    self._f(self.tsnr_threshold_field),
                    self.motion_correction_dropdown,
                    ft.Text(
                        "Siemens MoCo: parse from scanner JSON (~0ms). "
                        "Software: ANTsPy registration (~400-500ms/vol). "
                        "None: no motion correction.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Text("GLM Nuisance Regressors", size=14, weight=ft.FontWeight.W_500),
                    ft.Text(
                        "Select which regressors to include in the real-time GLM (intercept is always included).",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self.drift_checkbox,
                    self.motion_regressors_checkbox,
                    self.wm_regressor_checkbox,
                    self.csf_regressor_checkbox,
                    self.rt_smooth_enabled_checkbox,
                    self.rt_smooth_fwhm_field,
                    self.feedback_mode_dropdown,
                    self.differential_row,
                    self.connectivity_section,
                    self.intermittent_enabled_checkbox,
                    self.feedback_interval_field,
                    ft.Divider(),
                ]
            ),
            visible=True,  # Always visible for all tasks
        )

        # Reference Realtime section
        # Create note for when reference scan is disabled
        self.reference_realtime_note = ft.Container(
            content=ft.Text(
                "When disabled, feedback uses masks from subject boldref space (no day-of registration).",
                size=12,
            ),
            bgcolor=ft.Colors.AMBER_100,
            padding=10,
            border_radius=5,
            visible=not self.reference_realtime_enabled_checkbox.value,
        )

        reference_section = [
            ft.Text("Real-Time Reference Scan", size=16, weight=ft.FontWeight.BOLD),
            self.reference_realtime_enabled_checkbox,
            self.reference_realtime_n_volumes_field,
            self.reference_realtime_note,
            ft.Text(
                "Collects a quick reference scan before feedback runs for day-of registration.",
                size=11,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            ft.Divider(),
        ]

        # Task-specific parameters (wrapped in container for visibility control)
        # Visible for any task that has configurable parameters
        task_specific_controls = self._build_task_specific_params()
        self.task_specific_section = ft.Container(
            content=ft.Column(task_specific_controls),
            visible=has_parameters,
        )

        # Sham section (integrated from sham tab)
        sham_section = self._build_sham_section()

        return ft.Container(
            content=ft.Column(
                scanner_section
                + [self.session_structure_section]
                + reference_section
                + [self.task_specific_section]
                + [sham_section],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
        )

    def _build_sham_section(self) -> ft.Container:
        """Build the sham/control condition section for feedback tab."""
        is_enabled = self.sham_enabled_checkbox.value
        method = self.sham_delivery_method_radio.value
        mask_mode = self.sham_mask_mode_radio.value

        # Hemisphere split section (single network split into left/right)
        feedback_mode_val = self.feedback_mode_dropdown.value or "activation"
        is_connectivity = feedback_mode_val == "connectivity"

        # Target hemisphere is irrelevant for connectivity (correlation is symmetric)
        self.sham_target_hemisphere_container = ft.Column(
            [
                ft.Text("Target Hemisphere:", size=12),
                self.sham_target_hemisphere_radio,
            ],
            visible=not is_connectivity,
        )

        self.sham_hemisphere_description = ft.Text(
            "Split the selected network into left and right hemispheres. "
            + (
                "Compute correlation between hemispheres for sham participants."
                if is_connectivity
                else "One hemisphere is target, the other is non-target."
            ),
            size=11,
            color=ft.Colors.GREY_600,
            italic=True,
        )

        self.sham_hemisphere_split_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Hemisphere Split", size=13, weight=ft.FontWeight.BOLD),
                    self.sham_hemisphere_network_dropdown,
                    self.sham_target_hemisphere_container,
                    self.sham_hemisphere_description,
                ]
            ),
            visible=is_enabled and method == "alternative_masks" and mask_mode == "hemisphere_split",
        )

        # Two networks section (legacy: use two distinct networks)
        self.sham_two_networks_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Alternative Networks", size=13, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        [
                            self.sham_first_network_dropdown,
                            self.sham_second_network_dropdown,
                        ]
                    ),
                    ft.Text(
                        "Use two distinct networks for sham (e.g., motor networks instead of default mode)",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ]
            ),
            visible=is_enabled and method == "alternative_masks" and mask_mode == "two_networks",
        )

        # Alternative masks section (contains mask mode radio and subsections)
        self.sham_alternative_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Alternative Mask Mode", size=14, weight=ft.FontWeight.BOLD),
                    self.sham_mask_mode_radio,
                    self.sham_hemisphere_split_section,
                    self.sham_two_networks_section,
                ]
            ),
            visible=is_enabled and (method == "alternative_masks"),
        )

        # Replay info section (replay uses base_dir + randomization file)
        self.sham_replay_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Replay Configuration", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Replay mode uses the matched_participant column from the randomization file "
                        "to find feedbacklog.tsv files in the study's base data directory.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Text(
                        "Ensure the matched participant's feedback session has completed before "
                        "running the sham participant.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ]
            ),
            visible=is_enabled and (method == "replay"),
        )

        # Configuration container (visible when sham enabled)
        self.sham_config_container = ft.Column(
            [
                ft.Divider(),
                ft.Text("Delivery Method", size=14, weight=ft.FontWeight.BOLD),
                self.sham_delivery_method_radio,
                ft.Divider(),
            ],
            visible=is_enabled,
        )

        # Randomization section (visible when sham enabled)
        self.sham_randomization_section = ft.Column(
            [
                ft.Divider(),
                ft.Text("Randomization", size=14, weight=ft.FontWeight.BOLD),
                ft.Row(
                    [
                        self.sham_randomization_file_field,
                        self._create_file_browse_button(
                            self.sham_randomization_file_field,
                            filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv"), ("All files", "*.*")],
                        ),
                        ft.IconButton(
                            icon=ft.Icons.SHUFFLE,
                            tooltip="Generate randomization file",
                            on_click=self._on_generate_randomization,
                        ),
                    ]
                ),
                ft.Text(
                    "TSV file with columns: subject_id, condition (real/sham), "
                    "matched_participant (for replay), matched_visit",
                    size=11,
                    color=ft.Colors.GREY_600,
                    italic=True,
                ),
            ],
            visible=is_enabled,
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Sham/Control Conditions", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Configure real/sham neurofeedback for controlled studies. "
                        "All masks are extracted for all participants to maintain blinding.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self.sham_enabled_checkbox,
                    self.blind_mode_checkbox,
                    ft.Text(
                        "Shows Target/Non-Target instead of network names in the task display.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self.sham_config_container,
                    self.sham_alternative_section,
                    self.sham_replay_section,
                    self.sham_randomization_section,
                ]
            ),
        )

    def _build_task_specific_params(self) -> list:
        """Build task-specific parameter controls dynamically from task metadata.

        Returns:
            List of Flet controls for the current task type
        """
        # Read from config (source of truth) rather than dropdown
        task_type = self.study_config.session_config["task"].get("task_type", "ball")

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
        is_simulator = self.simulator_enabled_checkbox.value

        # Get current number of localizer runs
        try:
            n_localizer_runs = int(self.localizer_n_runs_field.value)
        except (ValueError, TypeError):
            n_localizer_runs = 2

        # Get current number of feedback runs
        try:
            n_feedback_runs = int(self.max_feedback_runs_field.value)
        except (ValueError, TypeError):
            n_feedback_runs = 2

        # Check if transfer runs are enabled
        config = self.study_config.session_config
        task_type = config["task"].get("task_type", "ball")
        use_transfer_runs = config["task"].get(task_type, {}).get("use_transfer_runs", True)

        # Create localizer source rows with visibility based on n_runs
        self.localizer_source_rows = []
        for i, field in enumerate(self.localizer_source_fields):
            row = ft.Row(
                [field, self._create_browse_button(field)],
                visible=(i < n_localizer_runs),
            )
            self.localizer_source_rows.append(row)

        # Create reference rows with visibility based on enable checkboxes
        self.reference_initial_row = ft.Row(
            [self.reference_initial_field, self._create_browse_button(self.reference_initial_field)],
            visible=self.reference_initial_enabled_checkbox.value,
        )

        self.reference_realtime_row = ft.Row(
            [self.reference_realtime_field, self._create_browse_button(self.reference_realtime_field)],
            visible=self.reference_realtime_enabled_checkbox.value,
        )

        # Create transfer source rows (visibility based on use_transfer_runs)
        self.transfer_source_rows = []
        for i, field in enumerate(self.transfer_source_fields):
            row = ft.Row(
                [field, self._create_browse_button(field)],
                visible=use_transfer_runs,
            )
            self.transfer_source_rows.append(row)

        # Create feedback source rows with visibility based on n_feedback_runs
        self.feedback_source_rows = []
        for i, field in enumerate(self.feedback_source_fields):
            row = ft.Row(
                [field, self._create_browse_button(field)],
                visible=(i < n_feedback_runs),
            )
            self.feedback_source_rows.append(row)

        # Build simulator paths content
        is_mapping_mode = self.simulator_source_mode_radio.value == "mapping"

        simulator_header: list[ft.Control] = [
            ft.Divider(),
            ft.Text("Simulator Source Paths", size=16, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Configure DICOM source directories for offline testing with the simulator",
                size=11,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            self.simulator_source_mode_radio,
        ]

        # Per-subject mapping container
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
            visible=is_mapping_mode,
        )

        # Individual paths container
        individual_content: list[ft.Control] = []

        # Localizer section
        individual_content.append(ft.Text("Localizer Runs", size=14, weight=ft.FontWeight.BOLD))
        individual_content.extend(self.localizer_source_rows)

        # Reference section
        individual_content.append(ft.Text("Reference Scans", size=14, weight=ft.FontWeight.BOLD))
        individual_content.append(self.reference_initial_row)
        individual_content.append(self.reference_realtime_row)

        # Transfer section
        self.transfer_section_header = ft.Text(
            "Transfer Runs",
            size=14,
            weight=ft.FontWeight.BOLD,
            visible=use_transfer_runs,
        )
        individual_content.append(self.transfer_section_header)
        individual_content.extend(self.transfer_source_rows)

        # Feedback section
        individual_content.append(ft.Text("Feedback Runs", size=14, weight=ft.FontWeight.BOLD))
        individual_content.extend(self.feedback_source_rows)

        self.simulator_individual_container = ft.Column(
            individual_content,
            visible=not is_mapping_mode,
        )

        self.simulator_paths_container = ft.Column(
            simulator_header + [self.simulator_mapping_container, self.simulator_individual_container],
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
                    ft.Row(
                        [
                            self.base_dir_field,
                            self._create_browse_button(self.base_dir_field),
                        ]
                    ),
                    self.simulator_paths_container,
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
            _snack = ft.SnackBar(ft.Text("Please fix validation errors before saving"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Update study config from form fields
        self._update_study_from_form()

        # Save
        try:
            path = self.config_manager.save_study(self.study_config, overwrite=True)
            _snack = ft.SnackBar(ft.Text(f"Study saved: {self.study_config.metadata.name}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

            logger.info(f"Study saved: {path}")

            # Navigate back
            if self.on_save_callback:
                self.on_save_callback()

        except Exception as ex:
            logger.error(f"Failed to save study: {ex}")
            _snack = ft.SnackBar(ft.Text(f"Error saving study: {ex}"))
            _snack.open = True
            self.page.overlay.append(_snack)
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

                _snack = ft.SnackBar(ft.Text(f"Study saved as: {new_name}"))
                _snack.open = True
                self.page.overlay.append(_snack)
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
        from pathlib import Path

        from ..components.visit_selection_dialog import VisitSelectionDialog

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

            # Sync form values to study_config before starting session
            # This ensures the participant workflow gets current form values
            self._update_study_from_form()

            self.page.pop_dialog()

            logger.info(f"Starting session for participant: {participant_id}")

            # Check if multi-visit mode is enabled (use checkbox value, not saved config)
            multi_visit_enabled = self.multi_visit_enabled_checkbox.value

            if multi_visit_enabled:
                # Get base directory from config
                base_dir = Path(self.study_config.session_config["paths"]["base_dir"])

                # Show visit selection dialog
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
                # Single-visit mode - go directly to workflow
                if self.on_start_session_callback:
                    self.on_start_session_callback(participant_id, None, None)

        def cancel_start(e):
            self.page.pop_dialog()

        # Add actions to dialog
        dialog.actions = [
            ft.TextButton("Cancel", on_click=cancel_start),
            ft.FilledButton("Start", icon=ft.Icons.PLAY_ARROW, on_click=start_session),
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

        # Update study design settings
        if "study" not in config:
            config["study"] = {}
        config["study"]["multi_visit_enabled"] = self.multi_visit_enabled_checkbox.value

        try:
            # Localizer parameters
            config["scan"]["localizer_tr"] = float(self.localizer_tr_field.value)
            config["scan"]["localizer_n_volumes"] = int(self.localizer_n_volumes_field.value)
            config["scan"]["localizer_n_runs"] = int(self.localizer_n_runs_field.value)
            config["scan"]["localizer_task_type"] = self.localizer_task_type_dropdown.value

            # Sync mask.localizer_type (used by participant_workflow and pipeline_runner)
            localizer_task_type = self.localizer_task_type_dropdown.value
            config["mask"]["localizer_type"] = "rest" if localizer_task_type == "rest" else "task"

            # Reference scan parameters
            config["scan"]["reference_initial_enabled"] = self.reference_initial_enabled_checkbox.value
            config["scan"]["reference_initial_n_volumes"] = int(self.reference_initial_n_volumes_field.value)
            config["scan"]["reference_realtime_enabled"] = self.reference_realtime_enabled_checkbox.value
            config["scan"]["reference_realtime_n_volumes"] = int(self.reference_realtime_n_volumes_field.value)

            # Feedback parameters
            config["scan"]["feedback_tr"] = float(self.feedback_tr_field.value)
            config["scan"]["feedback_n_volumes"] = int(self.feedback_n_volumes_field.value)

            # Acquisition mode
            config["scan"]["simulator_enabled"] = self.simulator_enabled_checkbox.value
            config["scan"]["collection_speed"] = float(self.collection_speed_field.value)
            config["scan"]["feedback_simulator_speed"] = float(self.feedback_simulator_speed_field.value)
            config["scan"]["scanner_dicom_dir"] = self.scanner_dicom_dir_field.value.strip()
            config["scan"]["scanner_ip"] = self.scanner_ip_field.value.strip()
            # Only save mapping if in mapping mode
            if self.simulator_source_mode_radio.value == "mapping":
                config["scan"]["simulator_source_mapping"] = self.simulator_source_mapping_field.value.strip()
            else:
                config["scan"]["simulator_source_mapping"] = ""
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid scanner parameter: {e}")
            raise ValidationError(f"Invalid scanner parameter: {e}") from e

        # Update network config with validation
        try:
            # Network atlas
            from pineuro.mask_extraction import ATLAS_SOURCES

            network_atlas_val = self.network_atlas_dropdown.value or "yeo_17"
            n_networks = ATLAS_SOURCES.get(network_atlas_val, {}).get("n_networks", 17)

            smooth_val = self.smooth_fwhm_field.value
            highpass_val = self.highpass_field.value
            n_components_val = self.n_components_field.value

            # Debug: log values if they look wrong
            if not smooth_val or smooth_val == "None":
                logger.warning(f"smooth_fwhm_field has invalid value: {smooth_val!r}, using default 6.0")
                smooth_val = "6.0"
            if not highpass_val or highpass_val == "None":
                logger.warning(f"highpass_field has invalid value: {highpass_val!r}, using default 0.01")
                highpass_val = "0.01"
            if not n_components_val or n_components_val == "None":
                logger.warning(f"n_components_field has invalid value: {n_components_val!r}, using default 30")
                n_components_val = "30"

            threshold_val = self.threshold_field.value
            if not threshold_val or threshold_val == "None":
                logger.warning(f"threshold_field has invalid value: {threshold_val!r}, using default 2000")
                threshold_val = "2000"

            config["mask"]["network_atlas"] = network_atlas_val
            config["mask"]["n_yeo"] = n_networks
            config["mask"]["smooth_fwhm"] = float(smooth_val)
            config["mask"]["highpass"] = float(highpass_val)
            config["mask"]["n_components"] = int(n_components_val)
            config["mask"]["threshold"] = int(threshold_val)

            # Cluster-based thresholding settings
            threshold_method = self.threshold_method_dropdown.value or "top_n"
            config["mask"]["threshold_method"] = threshold_method
            if threshold_method == "cluster":
                atlas_source = self.atlas_source_dropdown.value
                atlas_region = self.atlas_region_dropdown.value
                config["mask"]["atlas_source"] = atlas_source
                config["mask"]["atlas_regions"] = (
                    [atlas_region] if atlas_region and atlas_region != "__loading__" else None
                )
                config["mask"]["atlas_name"] = None
                config["mask"]["atlas_mask"] = None
                config["mask"]["cluster_start_threshold"] = float(self.cluster_start_threshold_field.value or 2.0)
                config["mask"]["cluster_step"] = float(self.cluster_step_field.value or 0.1)
                config["mask"]["cluster_return_all"] = bool(self.cluster_return_all_checkbox.value)
            else:
                config["mask"]["atlas_source"] = None
                config["mask"]["atlas_regions"] = None
                config["mask"]["atlas_name"] = None
                config["mask"]["atlas_mask"] = None
                config["mask"]["cluster_return_all"] = False
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid network parameter: {e}")
            raise ValidationError(f"Invalid network parameter: {e}") from e

        # Update selected networks from dropdowns
        # Only save as many networks as the task requires (n_networks)
        task_type = self.task_type_dropdown.value
        try:
            task_meta = get_task_metadata(task_type)
            n_networks = task_meta.n_networks
        except ValueError:
            n_networks = 2  # Default fallback

        is_task_localizer = config.get("mask", {}).get("localizer_type", "rest") == "task"
        if is_task_localizer:
            # For task-based localizers, mask_names are derived from contrast keys
            contrasts = config.get("mask", {}).get("contrasts", {})
            if not contrasts:
                localizer_task_type = config.get("scan", {}).get("localizer_task_type")
                if localizer_task_type:
                    try:
                        loc_meta = get_task_metadata(localizer_task_type)
                        contrasts = loc_meta.default_contrasts or {}
                    except (KeyError, ValueError):
                        pass
            selected_networks = list(contrasts.keys())
            if config.get("mask", {}).get("include_reverse_contrasts", False):
                selected_networks = selected_networks + [f"{n}_reverse" for n in selected_networks]
        else:
            selected_networks = []
            if self.first_network_dropdown.value:
                selected_networks.append(self.first_network_dropdown.value)
            if n_networks >= 2 and self.second_network_dropdown.value:
                selected_networks.append(self.second_network_dropdown.value)
        config["mask"]["yeo_targets"] = selected_networks
        config["mask"]["mask_names"] = selected_networks

        # Update task config with validation
        task_type = self.task_type_dropdown.value
        config["task"]["task_type"] = task_type

        try:
            config["task"]["max_feedback_runs"] = int(self.max_feedback_runs_field.value)

            # Save n_baseline, n_volumes, and fd_threshold to feedback config (used by pipeline)
            n_baseline = int(self.n_baseline_field.value)
            config["feedback"]["n_baseline"] = n_baseline
            # Sync n_volumes from scan config to feedback config
            config["feedback"]["n_volumes"] = int(self.feedback_n_volumes_field.value)
            # FD threshold for motion monitoring
            config["feedback"]["fd_threshold"] = float(self.fd_threshold_field.value)
            config["feedback"]["dvars_threshold"] = float(self.dvars_threshold_field.value)
            config["feedback"]["tsnr_threshold"] = float(self.tsnr_threshold_field.value)
            # Motion correction mode and GLM nuisance regressors
            config["feedback"]["motion_correction"] = self.motion_correction_dropdown.value
            config["feedback"]["include_drift"] = self.drift_checkbox.value
            config["feedback"]["include_motion_regressors"] = self.motion_regressors_checkbox.value
            config["feedback"]["include_wm_regressor"] = self.wm_regressor_checkbox.value
            config["feedback"]["include_csf_regressor"] = self.csf_regressor_checkbox.value
            config["feedback"].pop("moco_enabled", None)
            config["feedback"].pop("nuisance_regression", None)
            # Real-time spatial smoothing
            if self.rt_smooth_enabled_checkbox.value:
                rt_smooth_val = self.rt_smooth_fwhm_field.value.strip()
                config["feedback"]["smooth_fwhm"] = float(rt_smooth_val) if rt_smooth_val else 6.0
            else:
                config["feedback"]["smooth_fwhm"] = None
            # Differential feedback pair (auto-enabled when 2+ masks)
            pos = self.differential_positive_dropdown.value
            neg = self.differential_negative_dropdown.value
            if pos and neg and pos != neg:
                config["feedback"]["differential_pair"] = [pos, neg]
            else:
                config["feedback"]["differential_pair"] = None
            # Intermittent feedback interval
            if self.intermittent_enabled_checkbox.value:
                interval_str = self.feedback_interval_field.value.strip()
                config["feedback"]["feedback_interval"] = int(interval_str) if interval_str else 5
            else:
                config["feedback"]["feedback_interval"] = None
            # Feedback mode and connectivity
            feedback_mode = self.feedback_mode_dropdown.value or "activation"
            config["feedback"]["feedback_mode"] = feedback_mode
            config["feedback"]["connectivity_target"] = self.connectivity_target_dropdown.value or "increase"
            if feedback_mode == "connectivity":
                a = self.connectivity_pair_a_dropdown.value
                b = self.connectivity_pair_b_dropdown.value
                if a and b and a != b:
                    config["feedback"]["connectivity_pairs"] = [[a, b]]
                else:
                    config["feedback"]["connectivity_pairs"] = None
                window_str = self.connectivity_window_field.value.strip()
                config["feedback"]["connectivity_window"] = int(window_str) if window_str else 15
                config["feedback"]["differential_pair"] = None
            else:
                config["feedback"]["connectivity_pairs"] = None
                config["feedback"]["connectivity_window"] = 15
            # Also save to task-specific config for reference
            if task_type not in config["task"]:
                config["task"][task_type] = {}
            config["task"][task_type]["n_baseline"] = n_baseline

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

                # For "dir" type, the field is a Row containing [TextField, IconButton]
                if param.type == "dir":
                    text_field = field.controls[0] if isinstance(field, ft.Row) else field
                    config["task"][task_type][param.key] = text_field.value
                elif param.type == "bool":
                    config["task"][task_type][param.key] = field.value
                elif param.type == "int":
                    config["task"][task_type][param.key] = int(field.value)
                elif param.type == "float":
                    config["task"][task_type][param.key] = float(field.value)
                elif param.type == "str":
                    config["task"][task_type][param.key] = field.value
                elif param.type == "list[str]":
                    # Parse comma-separated string to list
                    config["task"][task_type][param.key] = [s.strip() for s in field.value.split(",") if s.strip()]

            # Validate task config
            metadata.validate_config(config["task"][task_type])

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid task parameter: {e}")
            raise ValidationError(f"Invalid task parameter: {e}") from e

        # Save reverse contrast setting
        config["mask"]["include_reverse_contrasts"] = self.include_reverse_contrasts_checkbox.value

        # Save network assignment for current task type using its network_roles
        task_type = config["task"].get("task_type", "ball")
        if task_type not in config["task"]:
            config["task"][task_type] = {}
        try:
            task_meta = get_task_metadata(task_type)
            is_task_localizer = config.get("mask", {}).get("localizer_type", "rest") == "task"
            if is_task_localizer:
                # For task-based localizers, derive roles from differential pair
                dropdowns = [self.differential_positive_dropdown, self.differential_negative_dropdown]
            else:
                dropdowns = [self.first_network_dropdown, self.second_network_dropdown]
            for i, role in enumerate(task_meta.network_roles):
                if i < len(dropdowns) and dropdowns[i].value:
                    config["task"][task_type][role.key] = dropdowns[i].value
        except (ValueError, AttributeError):
            pass

        # Update localizer task config (if not rest)
        localizer_task_type = self.localizer_task_type_dropdown.value
        if localizer_task_type != "rest":
            try:
                metadata = get_task_metadata(localizer_task_type)

                # Initialize localizer_task config if missing
                if "localizer_task" not in config:
                    config["localizer_task"] = {}
                if localizer_task_type not in config["localizer_task"]:
                    config["localizer_task"][localizer_task_type] = {}

                # Update each parameter from dynamic fields
                for param in metadata.parameters:
                    field = self.localizer_task_param_fields.get(param.key)
                    if field is None:
                        continue

                    # For "dir" type, the field is a Row containing [TextField, IconButton]
                    if param.type == "dir":
                        text_field = field.controls[0] if isinstance(field, ft.Row) else field
                        config["localizer_task"][localizer_task_type][param.key] = text_field.value
                    elif param.type == "bool":
                        config["localizer_task"][localizer_task_type][param.key] = field.value
                    elif param.type == "int":
                        config["localizer_task"][localizer_task_type][param.key] = int(field.value)
                    elif param.type == "float":
                        config["localizer_task"][localizer_task_type][param.key] = float(field.value)
                    elif param.type == "str":
                        config["localizer_task"][localizer_task_type][param.key] = field.value
                    elif param.type == "list[str]":
                        config["localizer_task"][localizer_task_type][param.key] = [
                            s.strip() for s in field.value.split(",") if s.strip()
                        ]

            except (ValueError, TypeError) as e:
                logger.error(f"Invalid localizer task parameter: {e}")
                raise ValidationError(f"Invalid localizer task parameter: {e}") from e

        # Update paths
        config["paths"]["base_dir"] = self.base_dir_field.value.strip()

        # Update simulator source paths (dynamic based on n_runs)
        try:
            n_localizer_runs = int(self.localizer_n_runs_field.value)
        except (ValueError, TypeError):
            n_localizer_runs = 2
        config["scan"]["localizer_source_dicoms"] = [
            field.value.strip() for field in self.localizer_source_fields[:n_localizer_runs]
        ]
        config["scan"]["reference_initial_dicoms"] = self.reference_initial_field.value.strip()
        config["scan"]["reference_realtime_dicoms"] = self.reference_realtime_field.value.strip()

        # Update feedback source paths (per-run)
        try:
            n_feedback_runs = int(self.max_feedback_runs_field.value)
        except (ValueError, TypeError):
            n_feedback_runs = 2
        config["scan"]["feedback_source_dicoms"] = [
            field.value.strip() for field in self.feedback_source_fields[:n_feedback_runs]
        ]

        # Update transfer source paths (always 2)
        config["scan"]["transfer_source_dicoms"] = [field.value.strip() for field in self.transfer_source_fields[:2]]

        # Remove legacy single-path feedback_dicoms if present
        config["scan"].pop("feedback_dicoms", None)

        # Update sham/control configuration
        if "sham" not in config:
            config["sham"] = {}
        config["sham"]["enabled"] = self.sham_enabled_checkbox.value
        config["sham"]["blind_mode"] = self.blind_mode_checkbox.value
        config["sham"]["delivery_method"] = self.sham_delivery_method_radio.value
        # Save sham mask mode and related settings
        config["sham"]["sham_mask_mode"] = self.sham_mask_mode_radio.value
        config["sham"]["hemisphere_network"] = self.sham_hemisphere_network_dropdown.value or "SomMotA"
        config["sham"]["target_hemisphere"] = self.sham_target_hemisphere_radio.value or "right"
        # Save sham alternative networks from dropdowns (for two_networks mode)
        sham_targets = []
        if self.sham_first_network_dropdown.value:
            sham_targets.append(self.sham_first_network_dropdown.value)
        if self.sham_second_network_dropdown.value:
            sham_targets.append(self.sham_second_network_dropdown.value)
        config["sham"]["alternative_mask_targets"] = sham_targets
        config["sham"]["randomization_file"] = self.sham_randomization_file_field.value.strip()
        # Remove legacy replay_source_dir if present (now uses base_dir automatically)
        config["sham"].pop("replay_source_dir", None)

        # Update display configuration
        if "display" not in config:
            config["display"] = {}
        config["display"]["control_monitor"] = self.control_monitor_dropdown.value or "left"
        config["display"]["presenter_monitor"] = self.presenter_monitor_dropdown.value or "right"
