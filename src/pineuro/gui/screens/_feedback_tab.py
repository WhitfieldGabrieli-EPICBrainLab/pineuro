"""Feedback tab builder for the study editor.

Each instance manages the full form fields and layout for a single feedback
run tab (Feedback 1, Feedback 2, etc.).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import flet as ft

from pineuro.tasks.task_registry import TaskParameter, get_task_metadata

from ..utils import (
    ValidationError,
    validate_positive_int,
    validate_positive_number,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

FIELD_W = 400
PAIR_W = 195


def _f(control: ft.Control) -> ft.Row:
    """Wrap a control in a Row so Column doesn't stretch it."""
    return ft.Row([control])


class FeedbackTabBuilder:
    """Builds and manages form fields for a single feedback run tab."""

    def __init__(
        self,
        index: int,
        config: dict[str, Any],
        page: ft.Page,
        *,
        on_change: Callable[[], None] | None = None,
        available_mask_names: list[str] | None = None,
        create_browse_button: Callable[[ft.TextField], ft.IconButton] | None = None,
        create_file_browse_button: Callable[..., ft.IconButton] | None = None,
    ):
        """Initialize the feedback tab builder.

        Parameters
        ----------
        index : int
            0-based feedback run index
        config : dict
            Feedback run config dict with all per-run settings.
        page : ft.Page
            Flet page for updates
        on_change : callable, optional
            Called when feedback config changes
        available_mask_names : list[str], optional
            Mask names available from all localizers
        create_browse_button : callable, optional
            Factory for directory browse buttons
        create_file_browse_button : callable, optional
            Factory for file browse buttons
        """
        self.index = index
        self.config = dict(config)
        self.page = page
        self._on_change = on_change
        self._available_mask_names = available_mask_names or []
        self._create_browse_button = create_browse_button
        self._create_file_browse_button = create_file_browse_button

        self.validation_errors: dict[str, str] = {}
        self.task_param_fields: dict[str, Any] = {}

        self._create_fields()

    # ------------------------------------------------------------------
    # Field creation
    # ------------------------------------------------------------------

    def _create_fields(self):
        """Create all form field controls for this feedback run."""
        c = self.config
        fb = c.get("feedback", c)  # support both flat and nested format

        # Task type (read-only — set in Study tab)
        self._task_type = c.get("task_type", "ball")
        try:
            task_meta = get_task_metadata(self._task_type)
            type_label = task_meta.display_name
            self._task_n_networks = task_meta.n_networks
            self._task_network_roles = task_meta.network_roles
        except (KeyError, ValueError):
            type_label = self._task_type
            self._task_n_networks = 2  # assume multi-network if unknown
            self._task_network_roles = []
        self.type_display_text = ft.Text(
            type_label,
            size=14,
            weight=ft.FontWeight.W_500,
            color=ft.Colors.BLUE_700,
        )

        # Number of runs for this feedback task
        self.max_runs_field = ft.TextField(
            width=600,
            label="Number of runs",
            value=str(c.get("max_runs", 1)),
            hint_text="How many times this feedback task is repeated per session",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("max_runs", self.max_runs_field, validate_positive_int),
        )

        # Scan params
        self.tr_field = ft.TextField(
            width=600,
            label="TR (seconds)",
            value=str(c.get("tr", fb.get("feedback_tr", 1.2))),
            hint_text="Repetition time of the feedback sequence (e.g. 1.2)",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("tr", self.tr_field, validate_positive_number),
        )

        self.n_volumes_field = ft.TextField(
            width=600,
            label="Volumes per feedback run",
            value=str(c.get("n_volumes", fb.get("feedback_n_volumes", 150))),
            hint_text="Volumes acquired per feedback run",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_volumes", self.n_volumes_field, validate_positive_int),
        )

        # Session structure
        n_baseline_default = c.get("n_baseline", fb.get("n_baseline", 25))

        self.n_baseline_field = ft.TextField(
            width=600,
            label="Baseline volumes",
            value=str(n_baseline_default),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of baseline volumes at the start of each feedback run",
            on_change=lambda _: self._validate_field("n_baseline", self.n_baseline_field, validate_positive_int),
        )

        # Quality thresholds
        self.fd_threshold_field = ft.TextField(
            width=600,
            label="FD threshold (mm)",
            value=str(c.get("fd_threshold", fb.get("fd_threshold", 0.5))),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Frames exceeding this are flagged in real time (default: 0.5 mm)",
            on_change=lambda _: self._validate_field("fd_threshold", self.fd_threshold_field, validate_positive_number),
        )

        self.dvars_threshold_field = ft.TextField(
            width=600,
            label="DVARS threshold (standardized)",
            value=str(c.get("dvars_threshold", fb.get("dvars_threshold", 1.5))),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Standardized DVARS above this are flagged in real time (default: 1.5)",
            on_change=lambda _: self._validate_field(
                "dvars_threshold", self.dvars_threshold_field, validate_positive_number
            ),
        )

        self.tsnr_threshold_field = ft.TextField(
            width=600,
            label="tSNR threshold",
            value=str(c.get("tsnr_threshold", fb.get("tsnr_threshold", 20))),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="tSNR below this is flagged in real time (default: 20)",
            on_change=lambda _: self._validate_field(
                "tsnr_threshold", self.tsnr_threshold_field, validate_positive_number
            ),
        )

        # Motion correction
        mc_value = c.get("motion_correction", fb.get("motion_correction", "siemens_moco"))
        # Backward compat
        if "moco_enabled" in c and "motion_correction" not in c:
            mc_value = "siemens_moco" if c["moco_enabled"] else "software"

        self.motion_correction_dropdown = ft.Dropdown(
            label="Motion correction",
            width=600,
            value=mc_value,
            options=[
                ft.dropdown.Option(key="siemens_moco", text="Siemens MoCo"),
                ft.dropdown.Option(key="software", text="Software (ANTsPy)"),
                ft.dropdown.Option(key="none", text="None"),
            ],
            tooltip="How volumes are motion-corrected during real-time feedback.",
            on_select=self._on_motion_correction_changed,
        )

        # GLM nuisance regressors
        self.drift_checkbox = ft.Checkbox(
            label="Linear drift",
            value=c.get("include_drift", fb.get("include_drift", True)),
            tooltip="Include linear drift regressor in GLM nuisance model.",
        )
        self.motion_regressors_checkbox = ft.Checkbox(
            label="Motion parameters (6 DOF)",
            value=c.get("include_motion_regressors", fb.get("include_motion_regressors", True)),
            tooltip="Include 6 differential motion parameters in GLM. Requires motion correction.",
            disabled=mc_value == "none",
        )
        self.wm_regressor_checkbox = ft.Checkbox(
            label="White matter mean",
            value=c.get("include_wm_regressor", fb.get("include_wm_regressor", False)),
            tooltip="Include WM mean signal as GLM nuisance regressor.",
        )
        self.csf_regressor_checkbox = ft.Checkbox(
            label="CSF mean",
            value=c.get("include_csf_regressor", fb.get("include_csf_regressor", False)),
            tooltip="Include CSF mean signal as GLM nuisance regressor.",
        )

        # Real-time smoothing
        rt_smooth_val = c.get("smooth_fwhm", fb.get("smooth_fwhm"))
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
            hint_text="Gaussian kernel size in mm",
            visible=rt_smooth_val is not None,
        )

        # Feedback mode — only relevant for multi-network tasks (n_networks >= 2)
        mask_options = [ft.dropdown.Option(key=n, text=n) for n in self._available_mask_names]
        has_multi_mask = len(self._available_mask_names) >= 2
        needs_multi_network = self._task_n_networks >= 2
        feedback_mode_val = c.get("feedback_mode", fb.get("feedback_mode", "activation"))

        self.feedback_mode_dropdown = ft.Dropdown(
            width=600,
            label="Feedback signal",
            options=[
                ft.dropdown.Option("activation", "Activation (differential)"),
                ft.dropdown.Option("connectivity", "Connectivity (sliding-window correlation)"),
            ],
            value="activation" if feedback_mode_val == "activation_single" else feedback_mode_val,
            on_select=self._on_feedback_mode_changed,
            visible=needs_multi_network,
        )

        # Single-mask selector (for single-network tasks like thermometer, experience sampling)
        single_mask = c.get("target_mask", fb.get("target_mask"))
        if not single_mask and self._available_mask_names:
            single_mask = self._available_mask_names[0]
        self.single_mask_dropdown = ft.Dropdown(
            width=600,
            label="Target mask",
            options=mask_options,
            value=single_mask,
            visible=(self._task_n_networks == 1),
        )

        # Extra network role dropdowns (for tasks with 3+ networks, e.g. trigger network)
        # Roles beyond the last 2 (which are handled by the differential pair) get
        # their own dropdown.
        self._extra_role_dropdowns: list[tuple[str, ft.Dropdown]] = []
        n_extra = max(0, self._task_n_networks - 2)
        network_assignment = c.get("network_assignment", {})
        for role_idx in range(n_extra):
            if role_idx < len(self._task_network_roles):
                role = self._task_network_roles[role_idx]
                saved = network_assignment.get(role.key)
                if not saved and self._available_mask_names:
                    saved = self._available_mask_names[0]
                dd = ft.Dropdown(
                    width=600,
                    label=role.label,
                    options=mask_options,
                    value=saved,
                    tooltip=role.description,
                )
                self._extra_role_dropdowns.append((role.key, dd))
        self._extra_roles_column = ft.Column(
            [dd for _, dd in self._extra_role_dropdowns],
            visible=len(self._extra_role_dropdowns) > 0,
        )

        # Differential pair
        diff_pair = c.get("differential_pair", fb.get("differential_pair"))
        self.differential_positive_dropdown = ft.Dropdown(
            width=250,
            label="Positive mask",
            options=mask_options,
            value=diff_pair[0]
            if diff_pair and len(diff_pair) == 2
            else (self._available_mask_names[0] if self._available_mask_names else None),
        )
        self.differential_negative_dropdown = ft.Dropdown(
            width=250,
            label="Negative mask",
            options=mask_options,
            value=diff_pair[1]
            if diff_pair and len(diff_pair) == 2
            else (self._available_mask_names[1] if has_multi_mask else None),
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
            visible=needs_multi_network and has_multi_mask and feedback_mode_val == "activation",
        )

        # Connectivity config
        conn_pairs = c.get("connectivity_pairs", fb.get("connectivity_pairs"))
        conn_window = c.get("connectivity_window", fb.get("connectivity_window", 15))
        is_conn = feedback_mode_val == "connectivity"

        conn_pair_a = (
            conn_pairs[0][0]
            if conn_pairs and len(conn_pairs) > 0 and len(conn_pairs[0]) == 2
            else (self._available_mask_names[0] if self._available_mask_names else None)
        )
        conn_pair_b = (
            conn_pairs[0][1]
            if conn_pairs and len(conn_pairs) > 0 and len(conn_pairs[0]) == 2
            else (self._available_mask_names[1] if has_multi_mask else None)
        )
        self.connectivity_pair_a_dropdown = ft.Dropdown(
            width=250,
            label="ROI A",
            options=mask_options,
            value=conn_pair_a,
        )
        self.connectivity_pair_b_dropdown = ft.Dropdown(
            width=250,
            label="ROI B",
            options=mask_options,
            value=conn_pair_b,
        )
        self.connectivity_window_field = ft.TextField(
            width=600,
            label="Sliding window size (volumes)",
            value=str(conn_window),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of volumes in correlation window",
        )
        self.connectivity_target_dropdown = ft.Dropdown(
            width=600,
            label="Target direction",
            options=[
                ft.dropdown.Option("increase", "Increase correlation"),
                ft.dropdown.Option("decrease", "Decrease correlation (anti-correlation)"),
            ],
            value=c.get("connectivity_target", fb.get("connectivity_target", "increase")),
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
            visible=needs_multi_network and is_conn,
        )

        # Intermittent feedback
        interval_val = c.get("feedback_interval", fb.get("feedback_interval"))
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

        # Sham config
        sham_config = c.get("sham", {})
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

        # Sham network dropdowns
        from ..utils import YEO_17_LABELS

        yeo_options = [ft.dropdown.Option(key=label, text=label) for label in YEO_17_LABELS]

        self.sham_hemisphere_network_dropdown = ft.Dropdown(
            width=600,
            label="Network to Split",
            options=yeo_options,
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

        sham_targets = sham_config.get("alternative_mask_targets", ["SomMotA", "DefaultA"])
        self.sham_first_network_dropdown = ft.Dropdown(
            width=290,
            label="First Sham Network",
            options=yeo_options,
            value=sham_targets[0] if len(sham_targets) > 0 else None,
        )
        self.sham_second_network_dropdown = ft.Dropdown(
            width=290,
            label="Second Sham Network",
            options=yeo_options,
            value=sham_targets[1] if len(sham_targets) > 1 else None,
        )
        self.sham_randomization_file_field = ft.TextField(
            width=540,
            label="Randomization File",
            value=sham_config.get("randomization_file", ""),
            hint_text="TSV file with subject_id, condition, matched_participant columns",
        )

        # Create task parameter fields
        self._create_task_parameter_fields()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ft.Container:
        """Build the tab content."""
        task_type = self._task_type or "ball"
        try:
            task_metadata = get_task_metadata(task_type)
            has_parameters = len(task_metadata.parameters) > 0
        except ValueError:
            has_parameters = True

        # Scanner parameters section
        scanner_section = [
            ft.Text("Task Type", size=16, weight=ft.FontWeight.BOLD),
            self.type_display_text,
            ft.Divider(),
            ft.Text("Scan Parameters", size=16, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Configure feedback run parameters (must match scanner sequence settings).",
                size=11,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            self.max_runs_field,
            _f(self.tr_field),
            self.n_volumes_field,
            ft.Divider(),
        ]

        # Session structure
        self.session_structure_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Real-Time Processing", size=16, weight=ft.FontWeight.BOLD),
                    self.n_baseline_field,
                    _f(self.fd_threshold_field),
                    _f(self.dvars_threshold_field),
                    _f(self.tsnr_threshold_field),
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
                    ft.Divider(),
                ]
            ),
        )

        # Feedback signal
        self.feedback_signal_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Feedback Signal", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Configure which brain signal drives the feedback display.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    self.single_mask_dropdown,
                    self._extra_roles_column,
                    self.feedback_mode_dropdown,
                    self.differential_row,
                    self.connectivity_section,
                    self.intermittent_enabled_checkbox,
                    self.feedback_interval_field,
                    ft.Divider(),
                ]
            ),
        )

        # Task-specific parameters
        task_specific_controls = self._build_task_specific_params()
        self.task_specific_section = ft.Container(
            content=ft.Column(task_specific_controls),
            visible=has_parameters,
        )

        # Sham section
        sham_section = self._build_sham_section()

        # Assemble
        all_controls = scanner_section + [
            self.session_structure_section,
            self.feedback_signal_section,
            self.task_specific_section,
            sham_section,
        ]

        return ft.Container(
            content=ft.Column(all_controls, scroll=ft.ScrollMode.AUTO),
            padding=20,
        )

    # ------------------------------------------------------------------
    # Collect config
    # ------------------------------------------------------------------

    def collect_config(self) -> dict[str, Any]:
        """Gather current field values into a config dict."""
        task_type = self._task_type or "ball"

        cfg: dict[str, Any] = {
            "task_type": task_type,
            "max_runs": int(self.max_runs_field.value or 1),
            "tr": float(self.tr_field.value or 1.2),
            "n_volumes": int(self.n_volumes_field.value or 150),
            "n_baseline": int(self.n_baseline_field.value or 25),
            "fd_threshold": float(self.fd_threshold_field.value or 0.5),
            "dvars_threshold": float(self.dvars_threshold_field.value or 1.5),
            "tsnr_threshold": float(self.tsnr_threshold_field.value or 20),
            "motion_correction": self.motion_correction_dropdown.value or "siemens_moco",
            "include_drift": self.drift_checkbox.value,
            "include_motion_regressors": self.motion_regressors_checkbox.value,
            "include_wm_regressor": self.wm_regressor_checkbox.value,
            "include_csf_regressor": self.csf_regressor_checkbox.value,
        }

        # Smoothing
        if self.rt_smooth_enabled_checkbox.value:
            val = self.rt_smooth_fwhm_field.value.strip()
            cfg["smooth_fwhm"] = float(val) if val else 6.0
        else:
            cfg["smooth_fwhm"] = None

        # Target mask (single-network tasks)
        if self._task_n_networks == 1 and self.single_mask_dropdown.value:
            cfg["target_mask"] = self.single_mask_dropdown.value

        # Feedback mode
        feedback_mode = self.feedback_mode_dropdown.value or "activation"
        cfg["feedback_mode"] = feedback_mode

        # Differential pair
        pos = self.differential_positive_dropdown.value
        neg = self.differential_negative_dropdown.value
        if pos and neg and pos != neg:
            cfg["differential_pair"] = [pos, neg]
        else:
            cfg["differential_pair"] = None

        # Network assignment (for tasks with 3+ networks)
        if self._extra_role_dropdowns:
            assignment: dict[str, str] = {}
            for role_key, dd in self._extra_role_dropdowns:
                if dd.value:
                    assignment[role_key] = dd.value
            # Also include the differential pair roles
            diff_roles = self._task_network_roles[len(self._extra_role_dropdowns) :]
            if len(diff_roles) >= 2 and pos and neg:
                assignment[diff_roles[0].key] = pos
                assignment[diff_roles[1].key] = neg
            cfg["network_assignment"] = assignment

        # Connectivity
        cfg["connectivity_target"] = self.connectivity_target_dropdown.value or "increase"
        if feedback_mode == "connectivity":
            a = self.connectivity_pair_a_dropdown.value
            b = self.connectivity_pair_b_dropdown.value
            if a and b and a != b:
                cfg["connectivity_pairs"] = [[a, b]]
            else:
                cfg["connectivity_pairs"] = None
            window_str = self.connectivity_window_field.value.strip()
            cfg["connectivity_window"] = int(window_str) if window_str else 15
        else:
            cfg["connectivity_pairs"] = None
            cfg["connectivity_window"] = 15

        # Intermittent feedback
        if self.intermittent_enabled_checkbox.value:
            interval_str = self.feedback_interval_field.value.strip()
            cfg["feedback_interval"] = int(interval_str) if interval_str else 5
        else:
            cfg["feedback_interval"] = None

        # Task-specific params
        task_params = self._collect_task_params()
        cfg["task_params"] = task_params

        # Copy use_transfer_runs to top level (keep in task_params for round-trip)
        cfg["use_transfer_runs"] = task_params.get("use_transfer_runs", False)

        # Sham config
        cfg["sham"] = {
            "enabled": self.sham_enabled_checkbox.value,
            "blind_mode": self.blind_mode_checkbox.value,
            "delivery_method": self.sham_delivery_method_radio.value,
            "sham_mask_mode": self.sham_mask_mode_radio.value,
            "hemisphere_network": self.sham_hemisphere_network_dropdown.value or "SomMotA",
            "target_hemisphere": self.sham_target_hemisphere_radio.value or "right",
            "alternative_mask_targets": [
                v
                for v in [
                    self.sham_first_network_dropdown.value,
                    self.sham_second_network_dropdown.value,
                ]
                if v
            ],
            "randomization_file": self.sham_randomization_file_field.value.strip(),
        }

        # Preserve existing simulator source paths (managed by Paths tab)
        for key in ("source_dicoms", "transfer_source_dicoms"):
            if key in self.config:
                cfg[key] = self.config[key]

        return cfg

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_available_masks(self, mask_names: list[str]):
        """Update available mask names in feedback dropdowns."""
        self._available_mask_names = mask_names
        new_options = [ft.dropdown.Option(key=n, text=n) for n in mask_names]
        has_multi = len(mask_names) >= 2

        extra_dds = [dd for _, dd in self._extra_role_dropdowns]
        for dropdown in [
            self.single_mask_dropdown,
            self.differential_positive_dropdown,
            self.differential_negative_dropdown,
            self.connectivity_pair_a_dropdown,
            self.connectivity_pair_b_dropdown,
            *extra_dds,
        ]:
            old = dropdown.value
            dropdown.options = new_options
            valid = [o.key for o in new_options]
            if old not in valid:
                dropdown.value = None

        # Update differential row visibility (only for multi-network tasks)
        needs_multi = self._task_n_networks >= 2
        mode = self.feedback_mode_dropdown.value
        self.feedback_mode_dropdown.visible = needs_multi
        self.differential_row.visible = needs_multi and has_multi and mode == "activation"
        self.connectivity_section.visible = needs_multi and mode == "connectivity"

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_motion_correction_changed(self, e):
        """Disable motion regressors when motion correction is none."""
        is_none = self.motion_correction_dropdown.value == "none"
        self.motion_regressors_checkbox.disabled = is_none
        if is_none:
            self.motion_regressors_checkbox.value = False
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_rt_smooth_enabled_changed(self, e):
        """Toggle smoothing FWHM field."""
        self.rt_smooth_fwhm_field.visible = self.rt_smooth_enabled_checkbox.value
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_intermittent_enabled_changed(self, e):
        """Toggle intermittent feedback interval field."""
        self.feedback_interval_field.visible = self.intermittent_enabled_checkbox.value
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_feedback_mode_changed(self, e):
        """Toggle activation vs connectivity fields."""
        mode = self.feedback_mode_dropdown.value
        has_multi = len(self._available_mask_names) >= 2
        needs_multi = self._task_n_networks >= 2
        self.differential_row.visible = needs_multi and mode == "activation" and has_multi
        self.connectivity_section.visible = needs_multi and mode == "connectivity"
        self._update_sham_visibility()
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_sham_enabled_changed(self, e):
        self._update_sham_visibility()

    def _on_sham_method_changed(self, e):
        self._update_sham_visibility()

    def _on_sham_mask_mode_changed(self, e):
        self._update_sham_visibility()

    def _update_sham_visibility(self):
        """Update sham section visibility."""
        is_enabled = self.sham_enabled_checkbox.value
        method = self.sham_delivery_method_radio.value
        mask_mode = self.sham_mask_mode_radio.value
        is_conn = (self.feedback_mode_dropdown.value or "activation") == "connectivity"

        if hasattr(self, "sham_config_container"):
            self.sham_config_container.visible = is_enabled
        if hasattr(self, "sham_alternative_section"):
            self.sham_alternative_section.visible = is_enabled and method == "alternative_masks"
        if hasattr(self, "sham_hemisphere_split_section"):
            self.sham_hemisphere_split_section.visible = (
                is_enabled and method == "alternative_masks" and mask_mode == "hemisphere_split"
            )
        if hasattr(self, "sham_two_networks_section"):
            self.sham_two_networks_section.visible = (
                is_enabled and method == "alternative_masks" and mask_mode == "two_networks"
            )
        if hasattr(self, "sham_replay_section"):
            self.sham_replay_section.visible = is_enabled and method == "replay"
        if hasattr(self, "sham_randomization_section"):
            self.sham_randomization_section.visible = is_enabled
        if hasattr(self, "sham_target_hemisphere_container"):
            self.sham_target_hemisphere_container.visible = not is_conn
        if hasattr(self, "sham_hemisphere_description"):
            if is_conn:
                self.sham_hemisphere_description.value = (
                    "Split the selected network into left and right hemispheres. "
                    "Compute correlation between hemispheres for sham participants."
                )
            else:
                self.sham_hemisphere_description.value = (
                    "Split the selected network into left and right hemispheres. "
                    "One hemisphere is target, the other is non-target."
                )

        with contextlib.suppress(Exception):
            self.page.update()

    # ------------------------------------------------------------------
    # Generate randomization dialog
    # ------------------------------------------------------------------

    def _on_generate_randomization(self, e):
        """Open dialog to generate a randomization file."""
        prefix_field = ft.TextField(
            label="Participant ID prefix", value="sub-", width=200,
            hint_text="e.g. sub- → sub-01, sub-rtchr0 → sub-rtchr001",
        )
        n_participants_field = ft.TextField(
            label="Number of participants", width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        n_sham_field = ft.TextField(
            label="Number of sham", width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        method_dropdown = ft.Dropdown(
            label="Method", width=200,
            options=[
                ft.dropdown.Option(key="simple", text="Simple"),
                ft.dropdown.Option(key="replay", text="Replay"),
            ],
            value="simple",
        )
        seed_field = ft.TextField(
            label="Seed (optional)", width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Generate Randomization File"),
            content=ft.Container(
                content=ft.Column([
                    prefix_field,
                    n_participants_field,
                    n_sham_field,
                    method_dropdown,
                    seed_field,
                ], tight=True, spacing=10),
                width=300,
            ),
        )

        def generate(e):
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
                    f"Must be between 1 and {n_total - 1}"
                    if "n_total" in dir()
                    else "Must be a positive integer"
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
                return

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

            self.sham_randomization_file_field.value = save_path
            self.sham_randomization_file_field.update()
            self.page.pop_dialog()

            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Randomization file saved: {save_path}"),
                open=True,
            )
            self.page.update()

        def cancel(e):
            self.page.pop_dialog()

        dialog.actions = [
            ft.TextButton("Cancel", on_click=cancel),
            ft.ElevatedButton("Generate", on_click=generate),
        ]
        self.page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # Sham section builder
    # ------------------------------------------------------------------

    def _build_sham_section(self) -> ft.Container:
        """Build the sham/control condition section."""
        is_enabled = self.sham_enabled_checkbox.value
        method = self.sham_delivery_method_radio.value
        mask_mode = self.sham_mask_mode_radio.value
        is_conn = (self.feedback_mode_dropdown.value or "activation") == "connectivity"

        self.sham_target_hemisphere_container = ft.Column(
            [
                ft.Text("Target Hemisphere:", size=12),
                self.sham_target_hemisphere_radio,
            ],
            visible=not is_conn,
        )

        self.sham_hemisphere_description = ft.Text(
            "Split the selected network into left and right hemispheres. "
            + (
                "Compute correlation between hemispheres for sham participants."
                if is_conn
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

        self.sham_two_networks_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Alternative Networks", size=13, weight=ft.FontWeight.BOLD),
                    ft.Row([self.sham_first_network_dropdown, self.sham_second_network_dropdown]),
                    ft.Text(
                        "Use two distinct networks for sham",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ]
            ),
            visible=is_enabled and method == "alternative_masks" and mask_mode == "two_networks",
        )

        self.sham_alternative_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Alternative Mask Mode", size=14, weight=ft.FontWeight.BOLD),
                    self.sham_mask_mode_radio,
                    self.sham_hemisphere_split_section,
                    self.sham_two_networks_section,
                ]
            ),
            visible=is_enabled and method == "alternative_masks",
        )

        self.sham_replay_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Replay Configuration", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Replay mode uses the matched_participant column from the randomization file.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                ]
            ),
            visible=is_enabled and method == "replay",
        )

        self.sham_config_container = ft.Column(
            [
                ft.Divider(),
                ft.Text("Delivery Method", size=14, weight=ft.FontWeight.BOLD),
                self.sham_delivery_method_radio,
                ft.Divider(),
            ],
            visible=is_enabled,
        )

        # Randomization file row with browse button
        rand_controls: list[ft.Control] = [self.sham_randomization_file_field]
        if self._create_file_browse_button:
            rand_controls.append(
                self._create_file_browse_button(
                    self.sham_randomization_file_field,
                    filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv"), ("All files", "*.*")],
                )
            )
        rand_controls.append(
            ft.IconButton(
                icon=ft.Icons.SHUFFLE,
                tooltip="Generate randomization file",
                on_click=self._on_generate_randomization,
            ),
        )

        self.sham_randomization_section = ft.Column(
            [
                ft.Divider(),
                ft.Text("Randomization", size=14, weight=ft.FontWeight.BOLD),
                ft.Row(rand_controls),
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
                        "Configure real/sham neurofeedback for controlled studies.",
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

    # ------------------------------------------------------------------
    # Task-specific params
    # ------------------------------------------------------------------

    def _create_task_parameter_fields(self):
        """Create form fields for current feedback task type."""
        task_type = self._task_type or "ball"
        self.task_param_fields.clear()

        try:
            meta = get_task_metadata(task_type)
        except ValueError:
            return

        task_config = self.config.get("task_params", {})
        for param in meta.parameters:
            field = self._create_field_for_parameter(param, task_config)
            self.task_param_fields[param.key] = field

    def _create_field_for_parameter(self, param: TaskParameter, task_config: dict) -> Any:
        """Create appropriate Flet control for a parameter."""
        current_value = task_config.get(param.key, param.default)

        if param.type == "bool":
            return ft.Checkbox(label=param.label, value=current_value)
        elif param.type in ["float", "int"]:
            return ft.TextField(
                width=600,
                label=param.label,
                value=str(current_value),
                keyboard_type=ft.KeyboardType.NUMBER,
                hint_text=param.hint,
            )
        elif param.type == "dir":
            field = ft.TextField(width=600, label=param.label, value=current_value, hint_text=param.hint)
            if self._create_browse_button:
                return ft.Row([field, self._create_browse_button(field)])
            return field
        elif param.type == "str":
            if param.choices:
                return ft.Dropdown(
                    width=600,
                    label=param.label,
                    value=current_value,
                    hint_text=param.hint,
                    options=[ft.dropdown.Option(c) for c in param.choices],
                )
            return ft.TextField(width=600, label=param.label, value=current_value, hint_text=param.hint)
        elif param.type == "list[str]":
            value_str = ", ".join(current_value) if isinstance(current_value, list) else current_value
            return ft.TextField(
                width=600,
                label=param.label,
                value=value_str,
                hint_text=f"{param.hint} (comma-separated)" if param.hint else "Comma-separated values",
            )
        return ft.TextField(
            width=600,
            label=param.label,
            value=str(current_value) if current_value is not None else "",
            hint_text=param.hint,
        )

    def _build_task_specific_params(self) -> list[ft.Control]:
        """Build task-specific parameter controls."""
        task_type = self._task_type or "ball"

        try:
            meta = get_task_metadata(task_type)
        except ValueError:
            return [ft.Text(f"Unknown task type: {task_type}", size=14, color=ft.Colors.ORANGE_700)]

        controls: list[ft.Control] = [
            ft.Text(f"{meta.display_name} Parameters", size=16, weight=ft.FontWeight.BOLD),
            ft.Text(meta.description, size=12, color=ft.Colors.GREY_600, italic=True),
            ft.Divider(),
        ]

        for param in meta.parameters:
            field = self.task_param_fields.get(param.key)
            if field is not None:
                controls.append(field)
                if param.hint and param.type != "bool":
                    controls.append(ft.Text(param.hint, size=11, color=ft.Colors.GREY_600, italic=True))

        return controls

    def _collect_task_params(self) -> dict[str, Any]:
        """Collect task-specific parameter values."""
        task_type = self._task_type or "ball"

        try:
            meta = get_task_metadata(task_type)
        except ValueError:
            return {}

        params: dict[str, Any] = {}
        for param in meta.parameters:
            field = self.task_param_fields.get(param.key)
            if field is None:
                continue
            if param.type == "dir":
                text_field: Any = field.controls[0] if isinstance(field, ft.Row) else field
                params[param.key] = text_field.value
            elif param.type == "bool":
                params[param.key] = field.value
            elif param.type == "int":
                params[param.key] = int(field.value)
            elif param.type == "float":
                params[param.key] = float(field.value)
            elif param.type == "str":
                params[param.key] = field.value
            elif param.type == "list[str]":
                params[param.key] = [s.strip() for s in field.value.split(",") if s.strip()]
        return params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_field(self, name: str, field: ft.TextField, validator):
        """Validate a form field."""
        try:
            validator(field.value, name)
            field.error = None
            self.validation_errors.pop(name, None)
        except ValidationError as e:
            field.error = str(e)
            self.validation_errors[name] = str(e)
        with contextlib.suppress(Exception):
            field.update()
