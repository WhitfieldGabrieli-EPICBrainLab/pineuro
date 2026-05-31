"""Feedback tab builder for the study editor.

Each instance manages the full form fields and layout for a single feedback
run tab (Feedback 1, Feedback 2, etc.).
"""

from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import flet as ft

from pineuro.tasks.task_registry import TaskParameter, get_task_metadata

from .. import styles as st
from ..utils import (
    ValidationError,
    validate_positive_int,
    validate_positive_number,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Mask dropdown options are session-qualified: a selection carries both the mask
# name and the session that produces it. The dropdown option key joins them with a
# unit-separator (won't appear in session ids or mask names); the bare name goes to
# ``network_assignment`` and the source (when not the current session) to
# ``mask_sources``.
_SRC_SEP = "␟"


def _mask_key(source: str, name: str) -> str:
    """Build a session-qualified dropdown option key. ``source`` is ``""`` for the
    current session or ``"ses-NN"`` for an earlier session."""
    return f"{source}{_SRC_SEP}{name}"


def _split_mask_key(key: str) -> tuple[str, str]:
    """Inverse of :func:`_mask_key` → ``(source, name)``. A bare name (no separator,
    from older configs) parses as ``("", name)``."""
    if _SRC_SEP in key:
        source, name = key.split(_SRC_SEP, 1)
        return source, name
    return "", key


class FeedbackTabBuilder:
    """Builds and manages form fields for a single feedback run tab."""

    def __init__(
        self,
        index: int,
        config: dict[str, Any],
        page: ft.Page,
        *,
        on_change: Callable[[], None] | None = None,
        available_mask_options: list[tuple[str, str, str]] | None = None,
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
        available_mask_options : list[tuple[str, str, str]], optional
            Session-qualified masks available to this feedback run, as
            ``(source_session, name, label)`` triples. ``source_session`` is ``""``
            for masks produced fresh this session, or ``"ses-NN"`` for an earlier
            session; ``label`` is the dropdown display text (e.g. ``"ses-01 ·
            Trigger"``).
        create_browse_button : callable, optional
            Factory for directory browse buttons
        create_file_browse_button : callable, optional
            Factory for file browse buttons
        """
        self.index = index
        # Deep copy so editing this tab cannot mutate the caller's config
        # (nested dicts/lists would otherwise stay shared references).
        self.config = copy.deepcopy(config)
        self.page = page
        self._on_change = on_change
        self._available_mask_options: list[tuple[str, str, str]] = list(available_mask_options or [])
        self._create_browse_button = create_browse_button
        self._create_file_browse_button = create_file_browse_button

        self.validation_errors: dict[str, str] = {}
        self.task_param_fields: dict[str, Any] = {}

        self._create_fields()

    # ------------------------------------------------------------------
    # Session-qualified mask option helpers
    # ------------------------------------------------------------------

    @property
    def _available_mask_names(self) -> list[str]:
        """Bare mask names from the available options (may repeat across sessions)."""
        return [name for _, name, _ in self._available_mask_options]

    def _mask_dd_options(self) -> list[ft.dropdown.Option]:
        """Dropdown options for the available session-qualified masks."""
        return [
            ft.dropdown.Option(key=_mask_key(src, name), text=label)
            for src, name, label in self._available_mask_options
        ]

    def _mask_value_for(self, name: str | None) -> str | None:
        """The dropdown option key for an assigned mask ``name``, resolving its source
        session from ``self._mask_sources`` (empty source => current session). None
        when ``name`` is falsy."""
        if not name:
            return None
        return _mask_key(self._mask_sources.get(name, ""), name)

    # ------------------------------------------------------------------
    # Field creation
    # ------------------------------------------------------------------

    def _create_fields(self):
        """Create all form field controls for this feedback run."""
        c = self.config
        # Nested sub-blocks (default to empty when a run doesn't configure them).
        qc_cfg = c.get("qc", {})
        glm_cfg = c.get("glm", {})
        connectivity_cfg = c.get("connectivity", {})
        intermittent_cfg = c.get("intermittent", {})

        # Task type (read-only — set in Study tab)
        self._task_type = c.get("task", "ball")
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
            size=st.SUBSECTION,
            weight=ft.FontWeight.W_500,
            color=st.ACCENT,
        )

        # Number of runs for this feedback task
        self.max_runs_field = ft.TextField(
            label="Number of runs",
            value=str(c.get("max_runs", 1)),
            hint_text="How many times this feedback task is repeated per session",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("max_runs", self.max_runs_field, validate_positive_int),
        )

        # Scan params
        self.tr_field = ft.TextField(
            label="TR (seconds)",
            value=str(c.get("tr", 1.2)),
            hint_text="Repetition time of the feedback sequence (e.g. 1.2)",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("tr", self.tr_field, validate_positive_number),
        )

        self.n_volumes_field = ft.TextField(
            label="Volumes per feedback run",
            value=str(c.get("n_volumes", 150)),
            hint_text="Volumes acquired per feedback run",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_volumes", self.n_volumes_field, validate_positive_int),
        )

        # Session structure
        n_baseline_default = c.get("n_baseline", 25)

        self.n_baseline_field = ft.TextField(
            label="Baseline volumes",
            value=str(n_baseline_default),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of baseline volumes at the start of each feedback run",
            on_change=lambda _: self._validate_field("n_baseline", self.n_baseline_field, validate_positive_int),
        )

        # Quality thresholds (qc sub-block)
        self.fd_threshold_field = ft.TextField(
            label="FD threshold (mm)",
            value=str(qc_cfg.get("fd_threshold", 0.5)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Frames exceeding this are flagged in real time (default: 0.5 mm)",
            on_change=lambda _: self._validate_field("fd_threshold", self.fd_threshold_field, validate_positive_number),
        )

        self.dvars_threshold_field = ft.TextField(
            label="DVARS threshold (standardized)",
            value=str(qc_cfg.get("dvars_threshold", 1.5)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Standardized DVARS above this are flagged in real time (default: 1.5)",
            on_change=lambda _: self._validate_field(
                "dvars_threshold", self.dvars_threshold_field, validate_positive_number
            ),
        )

        self.tsnr_threshold_field = ft.TextField(
            label="tSNR threshold",
            value=str(qc_cfg.get("tsnr_threshold", 20)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="tSNR below this is flagged in real time (default: 20)",
            on_change=lambda _: self._validate_field(
                "tsnr_threshold", self.tsnr_threshold_field, validate_positive_number
            ),
        )

        # Motion correction (glm sub-block)
        mc_value = glm_cfg.get("motion_correction", "siemens_moco")

        self.motion_correction_dropdown = ft.Dropdown(
            label="Motion correction",
            value=mc_value,
            options=[
                ft.dropdown.Option(key="siemens_moco", text="Siemens MoCo"),
                ft.dropdown.Option(key="software", text="Software (ANTsPy)"),
                ft.dropdown.Option(key="none", text="None"),
            ],
            tooltip="How volumes are motion-corrected during real-time feedback.",
            on_select=self._on_motion_correction_changed,
        )

        # GLM nuisance regressors (glm sub-block)
        self.drift_checkbox = ft.Checkbox(
            label="Linear drift",
            value=glm_cfg.get("include_drift", True),
            tooltip="Include linear drift regressor in GLM nuisance model.",
        )
        self.motion_regressors_checkbox = ft.Checkbox(
            label="Motion parameters (6 DOF)",
            value=glm_cfg.get("include_motion_regressors", True),
            tooltip="Include 6 differential motion parameters in GLM. Requires motion correction.",
            disabled=mc_value == "none",
        )
        self.wm_regressor_checkbox = ft.Checkbox(
            label="White matter mean",
            value=glm_cfg.get("include_wm_regressor", False),
            tooltip="Include WM mean signal as GLM nuisance regressor.",
        )
        self.csf_regressor_checkbox = ft.Checkbox(
            label="CSF mean",
            value=glm_cfg.get("include_csf_regressor", False),
            tooltip="Include CSF mean signal as GLM nuisance regressor.",
        )

        # Real-time smoothing (glm sub-block)
        rt_smooth_val = glm_cfg.get("smooth_fwhm")
        self.rt_smooth_enabled_checkbox = ft.Checkbox(
            label="Enable real-time spatial smoothing",
            value=rt_smooth_val is not None,
            on_change=self._on_rt_smooth_enabled_changed,
        )
        self.rt_smooth_fwhm_field = ft.TextField(
            label="Smoothing FWHM (mm)",
            value=str(rt_smooth_val) if rt_smooth_val is not None else "6.0",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Gaussian kernel size in mm",
            visible=rt_smooth_val is not None,
        )

        # Feedback mode — only relevant for multi-network tasks (n_networks >= 2)
        self._mask_sources = dict(c.get("mask_sources", {}) or {})
        mask_options = self._mask_dd_options()
        has_multi_mask = len(self._available_mask_options) >= 2
        needs_multi_network = self._task_n_networks >= 2
        feedback_mode_val = c.get("feedback_mode", "activation")

        self.feedback_mode_dropdown = ft.Dropdown(
            label="Feedback signal",
            options=[
                ft.dropdown.Option("activation", "Activation (differential)"),
                ft.dropdown.Option("connectivity", "Connectivity (sliding-window correlation)"),
            ],
            value="activation" if feedback_mode_val == "activation_single" else feedback_mode_val,
            on_select=self._on_feedback_mode_changed,
            visible=needs_multi_network,
        )

        # ROI combine method — how per-voxel activations are aggregated per mask.
        self.roi_combine_dropdown = ft.Dropdown(
            label="ROI combine method",
            options=[
                ft.dropdown.Option("inverse_sd", "Inverse-SD weighting (Hinds 2011)"),
                ft.dropdown.Option("mean", "Unweighted mean"),
            ],
            value=c.get("roi_combine_method", "inverse_sd"),
        )

        # network_assignment is the single source of truth for role-to-mask mapping.
        network_assignment = c.get("network_assignment", {}) or {}

        # Single-mask selector (for single-network tasks like thermometer, experience sampling).
        # Use the single role key (typically "target_mask") from TaskMetadata.
        single_role_key = (
            self._task_network_roles[0].key
            if self._task_n_networks == 1 and self._task_network_roles
            else "target_mask"
        )
        single_value = self._mask_value_for(network_assignment.get(single_role_key))
        if single_value is None and mask_options:
            single_value = mask_options[0].key
        single_mask_label = (
            self._task_network_roles[0].label
            if self._task_n_networks == 1 and self._task_network_roles
            else "Target mask"
        )
        self.single_mask_dropdown = ft.Dropdown(
            label=single_mask_label,
            options=mask_options,
            value=single_value,
            visible=(self._task_n_networks == 1),
        )

        # Optional network-role dropdowns. Each role declared optional in the
        # task metadata (e.g. the RIT triggered ball's movement / top / bottom
        # networks) gets its own mask dropdown with a leading blank option so
        # "unassigned" is expressible. ``visible_when`` gates a dropdown to a
        # task-parameter value (e.g. show movement_network only in single mode);
        # required roles are handled by the single-mask dropdown or the
        # differential pair.
        self._extra_role_dropdowns: list[tuple[str, ft.Dropdown]] = []
        for role in self._task_network_roles:
            if not getattr(role, "optional", False):
                continue
            if getattr(role, "default_from", None):
                blank_label = f"(use {self._role_label(role.default_from)})"
            else:
                blank_label = "(unassigned)"
            dd = ft.Dropdown(
                label=role.label,
                options=[ft.dropdown.Option(key="", text=blank_label), *mask_options],
                value=self._mask_value_for(network_assignment.get(role.key)) or "",
                tooltip=role.description,
                visible=self._role_visible(role),
            )
            self._extra_role_dropdowns.append((role.key, dd))
        self._extra_roles_column = ft.Column(
            [st.field_row(dd) for _, dd in self._extra_role_dropdowns],
            visible=len(self._extra_role_dropdowns) > 0,
            spacing=st.GAP_MD,
        )

        # Differential pair — for 2-network tasks, the positive/negative dropdowns map
        # to network_roles[-2] (top/positive) and network_roles[-1] (bottom/negative).
        diff_roles = self._task_network_roles[len(self._extra_role_dropdowns) :]
        diff_pos_key = diff_roles[0].key if len(diff_roles) >= 2 else None
        diff_neg_key = diff_roles[1].key if len(diff_roles) >= 2 else None
        diff_pos_val = self._mask_value_for(network_assignment.get(diff_pos_key)) if diff_pos_key else None
        diff_neg_val = self._mask_value_for(network_assignment.get(diff_neg_key)) if diff_neg_key else None
        self.differential_positive_dropdown = ft.Dropdown(
            label="Positive mask",
            options=mask_options,
            value=diff_pos_val or (mask_options[0].key if mask_options else None),
        )
        self.differential_negative_dropdown = ft.Dropdown(
            label="Negative mask",
            options=mask_options,
            value=diff_neg_val or (mask_options[1].key if has_multi_mask else None),
        )
        self.differential_row = ft.Column(
            [
                st.subsection("Differential Signal"),
                ft.Row(
                    [
                        st.fill_dropdown(self.differential_positive_dropdown),
                        ft.Text("minus", size=st.BODY),
                        st.fill_dropdown(self.differential_negative_dropdown),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=st.GAP_SM,
                ),
            ],
            visible=needs_multi_network and has_multi_mask and feedback_mode_val == "activation",
        )

        # Connectivity config (connectivity sub-block)
        conn_pairs = connectivity_cfg.get("pairs")
        conn_window = connectivity_cfg.get("window", 15)
        is_conn = feedback_mode_val == "connectivity"

        if conn_pairs and len(conn_pairs) > 0 and len(conn_pairs[0]) == 2:
            conn_pair_a = self._mask_value_for(conn_pairs[0][0])
            conn_pair_b = self._mask_value_for(conn_pairs[0][1])
        else:
            conn_pair_a = mask_options[0].key if mask_options else None
            conn_pair_b = mask_options[1].key if has_multi_mask else None
        self.connectivity_pair_a_dropdown = ft.Dropdown(
            label="ROI A",
            options=mask_options,
            value=conn_pair_a,
        )
        self.connectivity_pair_b_dropdown = ft.Dropdown(
            label="ROI B",
            options=mask_options,
            value=conn_pair_b,
        )
        self.connectivity_window_field = ft.TextField(
            label="Sliding window size (volumes)",
            value=str(conn_window),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of volumes in correlation window",
        )
        self.connectivity_target_dropdown = ft.Dropdown(
            label="Target direction",
            options=[
                ft.dropdown.Option("increase", "Increase correlation"),
                ft.dropdown.Option("decrease", "Decrease correlation (anti-correlation)"),
            ],
            value=connectivity_cfg.get("target", "increase"),
        )
        self.connectivity_section = ft.Column(
            [
                st.subsection("Connectivity Pair"),
                ft.Row(
                    [
                        st.fill_dropdown(self.connectivity_pair_a_dropdown),
                        ft.Text("corr", size=st.BODY),
                        st.fill_dropdown(self.connectivity_pair_b_dropdown),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=st.GAP_SM,
                ),
                st.field_row(self.connectivity_window_field),
                st.field_row(self.connectivity_target_dropdown),
            ],
            visible=needs_multi_network and is_conn,
            spacing=st.GAP_MD,
        )

        # Intermittent feedback (intermittent sub-block)
        interval_val = intermittent_cfg.get("interval")
        self.intermittent_enabled_checkbox = ft.Checkbox(
            label="Enable intermittent feedback",
            value=interval_val is not None,
            on_change=self._on_intermittent_enabled_changed,
        )
        self.feedback_interval_field = ft.TextField(
            label="Feedback interval (volumes)",
            value=str(interval_val) if interval_val is not None else "5",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Deliver mean activation every N volumes after baseline",
            visible=interval_val is not None,
        )

        # Per-run sham mask construction. Whether sham runs at all, the randomization
        # assignment, and blind mode are study-wide (set in the Study tab); these settings
        # only define HOW this task's sham masks are built.
        sham_config = c.get("sham", {})
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

        sham_targets = sham_config.get("alternative_mask_targets", ["SomMotA", "SomMotB"])
        self.sham_first_network_dropdown = ft.Dropdown(
            label="First Sham Network",
            options=yeo_options,
            value=sham_targets[0] if len(sham_targets) > 0 else None,
        )
        self.sham_second_network_dropdown = ft.Dropdown(
            label="Second Sham Network",
            options=yeo_options,
            value=sham_targets[1] if len(sham_targets) > 1 else None,
        )

        # Create task parameter fields
        self._create_task_parameter_fields()
        # Wire task-params that gate optional-role visibility (e.g. movement_mode).
        self._wire_role_gates()

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
            st.section("Task Type"),
            self.type_display_text,
            ft.Divider(),
            st.section("Scan Parameters"),
            st.caption("Configure feedback run parameters (must match scanner sequence settings)."),
            st.field_row(self.max_runs_field),
            st.field_row(self.tr_field),
            st.field_row(self.n_volumes_field),
            ft.Divider(),
        ]

        # Session structure
        self.session_structure_section = ft.Container(
            content=ft.Column(
                [
                    st.section("Real-Time Processing"),
                    st.field_row(self.n_baseline_field),
                    st.field_row(self.fd_threshold_field),
                    st.field_row(self.dvars_threshold_field),
                    st.field_row(self.tsnr_threshold_field),
                    st.field_row(self.motion_correction_dropdown),
                    st.caption(
                        "Siemens MoCo: parse from scanner JSON (~0ms). "
                        "Software: ANTsPy registration (~400-500ms/vol). "
                        "None: no motion correction.",
                    ),
                    st.subsection("GLM Nuisance Regressors"),
                    st.caption(
                        "Select which regressors to include in the real-time GLM (intercept is always included)."
                    ),
                    self.drift_checkbox,
                    self.motion_regressors_checkbox,
                    self.wm_regressor_checkbox,
                    self.csf_regressor_checkbox,
                    self.rt_smooth_enabled_checkbox,
                    st.field_row(self.rt_smooth_fwhm_field),
                    ft.Divider(),
                ],
                spacing=st.GAP_MD,
            ),
        )

        # Feedback signal
        self.feedback_signal_section = ft.Container(
            content=ft.Column(
                [
                    st.section("Feedback Signal"),
                    st.caption("Configure which brain signal drives the feedback display."),
                    st.field_row(self.single_mask_dropdown),
                    self._extra_roles_column,
                    st.field_row(self.feedback_mode_dropdown),
                    st.field_row(self.roi_combine_dropdown),
                    self.differential_row,
                    self.connectivity_section,
                    self.intermittent_enabled_checkbox,
                    st.field_row(self.feedback_interval_field),
                    ft.Divider(),
                ],
                spacing=st.GAP_MD,
            ),
        )

        # Task-specific parameters
        task_specific_controls = self._build_task_specific_params()
        self.task_specific_section = ft.Container(
            content=ft.Column(task_specific_controls, spacing=st.GAP_MD),
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
            content=ft.Column(all_controls, scroll=ft.ScrollMode.AUTO, spacing=st.GAP_MD),
            padding=st.PAD,
        )

    # ------------------------------------------------------------------
    # Collect config
    # ------------------------------------------------------------------

    def collect_config(self) -> dict[str, Any]:
        """Gather current field values into a config dict (new nested schema)."""
        task_type = self._task_type or "ball"

        # Smoothing value computed first so it can be placed in the glm sub-block
        if self.rt_smooth_enabled_checkbox.value:
            _v = self.rt_smooth_fwhm_field.value.strip()
            smooth_fwhm: float | None = float(_v) if _v else 6.0
        else:
            smooth_fwhm = None

        cfg: dict[str, Any] = {
            "task": task_type,
            "mask_source": self.config.get("mask_source", ""),
            "max_runs": int(self.max_runs_field.value or 1),
            "tr": float(self.tr_field.value or 1.2),
            "n_volumes": int(self.n_volumes_field.value or 150),
            "n_baseline": int(self.n_baseline_field.value or 25),
            "qc": {
                "fd_threshold": float(self.fd_threshold_field.value or 0.5),
                "dvars_threshold": float(self.dvars_threshold_field.value or 1.5),
                "tsnr_threshold": float(self.tsnr_threshold_field.value or 20),
            },
            "glm": {
                "motion_correction": self.motion_correction_dropdown.value or "siemens_moco",
                "include_drift": self.drift_checkbox.value,
                "include_motion_regressors": self.motion_regressors_checkbox.value,
                "include_wm_regressor": self.wm_regressor_checkbox.value,
                "include_csf_regressor": self.csf_regressor_checkbox.value,
                "smooth_fwhm": smooth_fwhm,
            },
        }

        # Feedback mode
        feedback_mode = self.feedback_mode_dropdown.value or "activation"
        cfg["feedback_mode"] = feedback_mode

        # ROI combine method
        cfg["roi_combine_method"] = self.roi_combine_dropdown.value or "inverse_sd"

        # Network assignment — single source of truth for role-to-mask mapping.
        # Always emit it (no more 3+ network gating; differential_pair / target_mask
        # / top_network / bottom_network are derived from this map at runtime).
        # Dropdown values are session-qualified keys: split into a bare mask name
        # (-> network_assignment) and its source session (-> mask_sources, only when
        # an earlier session supplies the mask).
        assignment: dict[str, str] = {}
        mask_sources: dict[str, str] = {}

        def _record(role_key: str, dd_value: str | None) -> None:
            if not dd_value:
                return
            source, name = _split_mask_key(dd_value)
            if not name:
                return
            assignment[role_key] = name
            if source:
                mask_sources[name] = source

        pos = self.differential_positive_dropdown.value
        neg = self.differential_negative_dropdown.value
        for role_key, dd in self._extra_role_dropdowns:
            # Only persist a non-blank, currently-visible optional role. This
            # drops the blank "(unassigned)" sentinel and avoids saving stale
            # dual-mode masks while in single mode (and vice versa).
            if dd.value and dd.visible:
                _record(role_key, dd.value)
        diff_roles = self._task_network_roles[len(self._extra_role_dropdowns) :]
        if len(diff_roles) >= 2 and pos and neg:
            _record(diff_roles[0].key, pos)
            _record(diff_roles[1].key, neg)
        elif self._task_n_networks == 1 and self.single_mask_dropdown.value:
            single_role_key = self._task_network_roles[0].key if self._task_network_roles else "target_mask"
            _record(single_role_key, self.single_mask_dropdown.value)
        cfg["network_assignment"] = assignment

        # Connectivity (sub-block)
        connectivity_block: dict[str, Any] = {"target": self.connectivity_target_dropdown.value or "increase"}
        if feedback_mode == "connectivity":
            a_src, a = _split_mask_key(self.connectivity_pair_a_dropdown.value or "")
            b_src, b = _split_mask_key(self.connectivity_pair_b_dropdown.value or "")
            if a and b and a != b:
                connectivity_block["pairs"] = [[a, b]]
                if a_src:
                    mask_sources[a] = a_src
                if b_src:
                    mask_sources[b] = b_src
            else:
                connectivity_block["pairs"] = None
            window_str = self.connectivity_window_field.value.strip()
            connectivity_block["window"] = int(window_str) if window_str else 15
        else:
            connectivity_block["pairs"] = None
            connectivity_block["window"] = 15
        cfg["connectivity"] = connectivity_block

        # Cross-session mask sources (name -> "ses-NN"); empty for an all-fresh run.
        cfg["mask_sources"] = mask_sources

        # Intermittent feedback (sub-block)
        if self.intermittent_enabled_checkbox.value:
            interval_str = self.feedback_interval_field.value.strip()
            cfg["intermittent"] = {"interval": int(interval_str) if interval_str else 5}
        else:
            cfg["intermittent"] = {"interval": None}

        # Task-specific params
        task_params = self._collect_task_params()
        cfg["task_params"] = task_params

        # Copy use_transfer_runs to top level (keep in task_params for round-trip)
        cfg["use_transfer_runs"] = task_params.get("use_transfer_runs", False)

        # Per-run sham mask construction (study-wide gating/randomization lives in the Study tab).
        cfg["sham"] = {
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
        }

        # Preserve existing simulator source paths (managed by Paths tab)
        for key in ("source_dicoms", "transfer_source_dicoms"):
            if key in self.config:
                cfg[key] = self.config[key]

        return cfg

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_available_masks(self, mask_options: list[tuple[str, str, str]]):
        """Update the session-qualified masks available in the feedback dropdowns.

        ``mask_options`` is a list of ``(source_session, name, label)`` triples, as
        passed to the constructor.
        """
        self._available_mask_options = list(mask_options or [])
        new_options = self._mask_dd_options()
        has_multi = len(self._available_mask_options) >= 2

        for dropdown in [
            self.single_mask_dropdown,
            self.differential_positive_dropdown,
            self.differential_negative_dropdown,
            self.connectivity_pair_a_dropdown,
            self.connectivity_pair_b_dropdown,
        ]:
            old = dropdown.value
            dropdown.options = new_options
            valid = [o.key for o in new_options]
            if old not in valid:
                dropdown.value = None

        # Optional-role dropdowns keep their leading blank "(unassigned)" option.
        for _, dropdown in self._extra_role_dropdowns:
            old = dropdown.value
            blank = dropdown.options[0] if dropdown.options else ft.dropdown.Option(key="", text="(unassigned)")
            dropdown.options = [blank, *new_options]
            valid = [o.key for o in dropdown.options]
            if old not in valid:
                dropdown.value = ""

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

    def _on_sham_method_changed(self, e):
        self._update_sham_visibility()

    def _on_sham_mask_mode_changed(self, e):
        self._update_sham_visibility()

    def _update_sham_visibility(self):
        """Update sham construction sub-section visibility (keyed off delivery/mask mode).

        Whether sham runs is study-wide, so the construction controls are always shown;
        only the delivery-method- and mask-mode-specific sub-sections toggle.
        """
        method = self.sham_delivery_method_radio.value
        mask_mode = self.sham_mask_mode_radio.value
        is_conn = (self.feedback_mode_dropdown.value or "activation") == "connectivity"

        if hasattr(self, "sham_alternative_section"):
            self.sham_alternative_section.visible = method == "alternative_masks"
        if hasattr(self, "sham_hemisphere_split_section"):
            self.sham_hemisphere_split_section.visible = (
                method == "alternative_masks" and mask_mode == "hemisphere_split"
            )
        if hasattr(self, "sham_two_networks_section"):
            self.sham_two_networks_section.visible = method == "alternative_masks" and mask_mode == "two_networks"
        if hasattr(self, "sham_replay_section"):
            self.sham_replay_section.visible = method == "replay"
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
    # Sham section builder
    # ------------------------------------------------------------------

    def _build_sham_section(self) -> ft.Container:
        """Build the per-run sham mask-construction section.

        Whether sham runs, the participant randomization, and blind mode are study-wide
        (Study tab); these controls only define HOW this task's sham masks are built.
        """
        method = self.sham_delivery_method_radio.value
        mask_mode = self.sham_mask_mode_radio.value
        is_conn = (self.feedback_mode_dropdown.value or "activation") == "connectivity"

        self.sham_target_hemisphere_container = ft.Column(
            [
                ft.Text("Target Hemisphere:", size=st.CAPTION),
                self.sham_target_hemisphere_radio,
            ],
            visible=not is_conn,
        )

        self.sham_hemisphere_description = st.caption(
            "Split the selected network into left and right hemispheres. "
            + (
                "Compute correlation between hemispheres for sham participants."
                if is_conn
                else "One hemisphere is target, the other is non-target."
            ),
        )

        self.sham_hemisphere_split_section = ft.Container(
            content=ft.Column(
                [
                    st.subsection("Hemisphere Split"),
                    st.field_row(self.sham_hemisphere_network_dropdown),
                    self.sham_target_hemisphere_container,
                    self.sham_hemisphere_description,
                ],
                spacing=st.GAP_MD,
            ),
            visible=method == "alternative_masks" and mask_mode == "hemisphere_split",
        )

        self.sham_two_networks_section = ft.Container(
            content=ft.Column(
                [
                    st.subsection("Alternative Networks"),
                    st.field_pair(self.sham_first_network_dropdown, self.sham_second_network_dropdown),
                    st.caption("Use two distinct networks for sham"),
                ],
                spacing=st.GAP_MD,
            ),
            visible=method == "alternative_masks" and mask_mode == "two_networks",
        )

        self.sham_alternative_section = ft.Container(
            content=ft.Column(
                [
                    st.subsection("Alternative Mask Mode"),
                    self.sham_mask_mode_radio,
                    self.sham_hemisphere_split_section,
                    self.sham_two_networks_section,
                ],
                spacing=st.GAP_MD,
            ),
            visible=method == "alternative_masks",
        )

        self.sham_replay_section = ft.Container(
            content=ft.Column(
                [
                    st.subsection("Replay Configuration"),
                    st.caption("Replay mode uses the matched_participant column from the study randomization file."),
                ],
                spacing=st.GAP_MD,
            ),
            visible=method == "replay",
        )

        return ft.Container(
            content=ft.Column(
                [
                    st.section("Sham Mask Construction"),
                    st.caption(
                        "How this task builds sham masks. Whether sham runs, the participant "
                        "randomization, and blind mode are set study-wide in the Study tab."
                    ),
                    st.subsection("Delivery Method"),
                    self.sham_delivery_method_radio,
                    self.sham_alternative_section,
                    self.sham_replay_section,
                ],
                spacing=st.GAP_MD,
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
                label=param.label,
                value=str(current_value),
                keyboard_type=ft.KeyboardType.NUMBER,
                hint_text=param.hint,
            )
        elif param.type == "dir":
            field = ft.TextField(label=param.label, value=current_value, hint_text=param.hint)
            if self._create_browse_button:
                return st.browse_row(field, self._create_browse_button(field))
            return field
        elif param.type == "str":
            if param.choices:
                return ft.Dropdown(
                    label=param.label,
                    value=current_value,
                    hint_text=param.hint,
                    options=[ft.dropdown.Option(c) for c in param.choices],
                )
            return ft.TextField(label=param.label, value=current_value, hint_text=param.hint)
        elif param.type == "list[str]":
            value_str = ", ".join(current_value) if isinstance(current_value, list) else current_value
            return ft.TextField(
                label=param.label,
                value=value_str,
                hint_text=f"{param.hint} (comma-separated)" if param.hint else "Comma-separated values",
            )
        return ft.TextField(
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
            return [ft.Text(f"Unknown task type: {task_type}", size=st.SUBSECTION, color=st.WARNING)]

        controls: list[ft.Control] = [
            st.section(f"{meta.display_name} Parameters"),
            st.caption(meta.description),
            ft.Divider(),
        ]

        for param in meta.parameters:
            field = self.task_param_fields.get(param.key)
            if field is not None:
                # Checkboxes and field+browse rows keep their own layout; size
                # plain inputs to fill the form width.
                controls.append(field if isinstance(field, (ft.Row, ft.Checkbox)) else st.field_row(field))
                if param.hint and param.type != "bool":
                    controls.append(st.caption(param.hint))

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

    def _role_label(self, role_key: str | None) -> str:
        """Display label for a network role by key (falls back to the key)."""
        for r in self._task_network_roles:
            if r.key == role_key:
                return str(r.label)
        return role_key or ""

    def _current_param_value(self, param_key: str) -> Any:
        """Current value of a task parameter — live field, else saved config, else metadata default."""
        field = self.task_param_fields.get(param_key)
        if field is not None and getattr(field, "value", None) is not None:
            return field.value
        task_config = self.config.get("task_params", {})
        if param_key in task_config:
            return task_config[param_key]
        try:
            meta = get_task_metadata(self._task_type or "ball")
            for p in meta.parameters:
                if p.key == param_key:
                    return p.default
        except ValueError:
            pass
        return None

    def _role_visible(self, role: Any) -> bool:
        """Whether an optional role's dropdown should show, per its visible_when gate."""
        gate = getattr(role, "visible_when", None)
        if not gate:
            return True
        param_key, expected = gate
        return bool(self._current_param_value(param_key) == expected)

    def _wire_role_gates(self) -> None:
        """Attach change handlers to task-params that gate optional-role visibility."""
        gated_param_keys = {r.visible_when[0] for r in self._task_network_roles if getattr(r, "visible_when", None)}
        for param_key in gated_param_keys:
            field = self.task_param_fields.get(param_key)
            if isinstance(field, ft.Dropdown):
                field.on_select = self._on_role_gate_param_changed

    def _on_role_gate_param_changed(self, e) -> None:
        """A gating task-param changed — re-evaluate optional-role visibility."""
        self._apply_role_gate_visibility()
        with contextlib.suppress(Exception):
            self.page.update()

    def _apply_role_gate_visibility(self) -> None:
        """Show/hide each optional-role dropdown per its current visible_when gate."""
        for role_key, dd in self._extra_role_dropdowns:
            role = next((r for r in self._task_network_roles if r.key == role_key), None)
            if role is not None:
                dd.visible = self._role_visible(role)

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
