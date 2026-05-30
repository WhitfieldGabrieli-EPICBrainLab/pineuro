"""Localizer tab builder for the study editor.

Each instance manages the full form fields and layout for a single localizer
tab (Localizer 1, Localizer 2, etc.).
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
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


class LocalizerTabBuilder:
    """Builds and manages form fields for a single localizer tab."""

    def __init__(
        self,
        index: int,
        config: dict[str, Any],
        page: ft.Page,
        *,
        on_change: Callable[[], None] | None = None,
        create_browse_button: Callable[[ft.TextField], ft.IconButton] | None = None,
        create_file_browse_button: Callable[..., ft.IconButton] | None = None,
    ):
        """Initialize the localizer tab builder.

        Parameters
        ----------
        index : int
            0-based localizer index
        config : dict
            Localizer config dict with keys: type, tr, n_volumes, n_runs,
            plus network/ICA/mask extraction settings.
        page : ft.Page
            Flet page for updates
        on_change : callable, optional
            Called when localizer config changes (e.g., network selection)
        create_browse_button : callable, optional
            Factory for directory browse buttons
        create_file_browse_button : callable, optional
            Factory for file browse buttons
        """
        self.index = index
        # Deep copy so editing this tab cannot mutate the caller's config
        # (nested dicts/lists would otherwise stay shared references).
        self.config = copy.deepcopy(config)  # working copy
        self.page = page
        self._on_change = on_change
        self._create_browse_button = create_browse_button
        self._create_file_browse_button = create_file_browse_button

        self.validation_errors: dict[str, str] = {}
        self.task_param_fields: dict[str, Any] = {}

        self._create_fields()

    # ------------------------------------------------------------------
    # Field creation
    # ------------------------------------------------------------------

    def _create_fields(self):
        """Create all form field controls for this localizer."""
        c = self.config
        loc_type = c.get("type", "rest")
        self._loc_type = loc_type

        # Type display (read-only — type is set in Study tab)
        if loc_type == "rest":
            type_label = "Rest (ICA Network Extraction)"
        else:
            try:
                meta = get_task_metadata(loc_type)
                type_label = f"{meta.display_name} (GLM Contrast)"
            except (KeyError, ValueError):
                type_label = loc_type
        self.type_display_text = ft.Text(
            type_label,
            size=14,
            weight=ft.FontWeight.W_500,
            color=st.ACCENT,
        )

        # Contrast info text
        self.contrast_info_text = ft.Text(
            value=self._get_contrast_info_text(loc_type),
            size=12,
            color=st.ACCENT,
            italic=True,
        )

        # Scan params
        self.tr_field = ft.TextField(
            label="TR (seconds)",
            value=str(c.get("tr", 1.2)),
            hint_text="Repetition time of the localizer sequence (e.g. 1.2)",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("tr", self.tr_field, validate_positive_number),
        )

        self.n_volumes_field = ft.TextField(
            label="Volumes per localizer run",
            value=str(c.get("n_volumes", 250)),
            hint_text="Volumes acquired per localizer run",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_volumes", self.n_volumes_field, validate_positive_int),
        )

        # Informational label shown in place of n_volumes_field for tasks
        # that declare their own duration via TaskMetadata.compute_max_run_seconds.
        # Visibility is set in _configure_computed_n_volumes() below.
        self.n_volumes_info_text = ft.Text(
            "",
            size=12,
            color=st.ACCENT,
            italic=True,
            visible=False,
        )

        self.n_runs_field = ft.TextField(
            label="Number of localizer runs",
            value=str(c.get("n_runs", 2)),
            hint_text="Number of localizer runs",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=self._on_n_runs_changed,
        )

        # Motion correction for QC monitoring
        mc_value = c.get("motion_correction", "siemens_moco")
        self.motion_correction_dropdown = ft.Dropdown(
            label="Motion correction",
            value=mc_value,
            options=[
                ft.dropdown.Option(key="siemens_moco", text="Siemens MoCo"),
                ft.dropdown.Option(key="software", text="Software (ANTsPy)"),
                ft.dropdown.Option(key="none", text="None"),
            ],
            tooltip="How motion parameters are extracted during localizer collection for real-time QC monitoring.",
        )

        # Reverse contrasts checkbox (task-based only)
        self.include_reverse_contrasts_checkbox = ft.Checkbox(
            label="Include reverse contrasts (enables dual-network from single-contrast localizer)",
            value=c.get("include_reverse_contrasts", False),
            visible=loc_type != "rest",
            on_change=self._on_reverse_contrasts_changed,
        )

        # Network atlas (extraction sub-block)
        from pineuro.mask_extraction import ATLAS_SOURCES

        extraction_cfg = c.get("extraction", {}) or {}
        network_atlas = extraction_cfg.get("network_atlas", "yeo_17")
        self.network_atlas_dropdown = ft.Dropdown(
            label="Network atlas",
            options=[
                ft.dropdown.Option(key=k, text=v["display_name"])
                for k, v in ATLAS_SOURCES.items()
                if v["type"] == "network"
            ],
            value=network_atlas,
            on_select=self._on_network_atlas_changed,
        )

        # Network selection dropdowns (up to MAX_NETWORKS)
        MAX_NETWORKS = 5
        networks = c.get("networks", ["DefaultA", "ContA"])
        self._n_target_networks = len(networks) if networks else 2

        current_n = ATLAS_SOURCES.get(network_atlas, {}).get("n_networks", 17)
        net_options = self._get_yeo_network_options(current_n)

        self.n_target_networks_dropdown = ft.Dropdown(
            label="Number of networks",
            options=[ft.dropdown.Option(str(i), str(i)) for i in range(1, MAX_NETWORKS + 1)],
            value=str(self._n_target_networks),
            on_select=self._on_n_target_networks_changed,
        )

        self._network_dropdowns: list[ft.Dropdown] = []
        for i in range(MAX_NETWORKS):
            val = networks[i] if i < len(networks) else None
            dd = ft.Dropdown(
                width=290,
                label=f"Network {i + 1}",
                options=[ft.dropdown.Option(key=o.key, text=o.text) for o in net_options],
                value=val,
                visible=(i < self._n_target_networks),
            )
            self._network_dropdowns.append(dd)

        # ICA settings (rest only) — read from extraction sub-block
        self.n_components_field = ft.TextField(
            label="ICA components",
            value=str(extraction_cfg.get("n_components", 20)),
            hint_text="Number of ICA components for decomposition",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_components", self.n_components_field, validate_positive_int),
        )

        self.smooth_fwhm_field = ft.TextField(
            label="Spatial smoothing (mm FWHM)",
            value=str(extraction_cfg.get("smooth_fwhm", 5.0)),
            hint_text="Gaussian kernel size in mm",
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        self.highpass_field = ft.TextField(
            label="Highpass filter (Hz)",
            value=str(extraction_cfg.get("highpass", 0.01)),
            hint_text="Cutoff frequency for highpass filter",
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        # Mask extraction (extraction sub-block)
        def _ex(key: str, default: Any = None) -> Any:
            v = extraction_cfg.get(key, default)
            return v if v is not None else default

        self.threshold_field = ft.TextField(
            label="Voxels per mask",
            value=str(_ex("threshold", 2000)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of voxels to keep per extracted mask",
            on_change=lambda _: self._validate_field("threshold", self.threshold_field, validate_positive_int),
        )

        threshold_method = _ex("threshold_method", "top_n")
        self.threshold_method_dropdown = ft.Dropdown(
            label="Thresholding method",
            options=[
                ft.dropdown.Option(key="top_n", text="Top N voxels"),
                ft.dropdown.Option(key="cluster", text="Atlas-guided cluster"),
                ft.dropdown.Option(key="cluster_extent", text="Cluster extent thresholding"),
            ],
            value=threshold_method,
            on_select=self._on_threshold_method_changed,
        )

        # Cluster settings
        cluster_atlas_options = [
            ft.dropdown.Option(key=k, text=v["display_name"])
            for k, v in ATLAS_SOURCES.items()
            if v["type"] in ("anatomical", "tissue") or k.startswith("schaefer")
        ]
        atlas_source = _ex("atlas_source", "harvard_oxford")
        self.atlas_source_dropdown = ft.Dropdown(
            label="Atlas",
            options=cluster_atlas_options,
            value=atlas_source,
            on_select=self._on_atlas_source_changed,
        )

        current_regions = _ex("atlas_regions") or []
        self.atlas_region_dropdown = ft.Dropdown(
            label="Region",
            options=[],
            value=current_regions[0] if current_regions else None,
        )

        self.cluster_start_threshold_field = ft.TextField(
            width=290,
            label="Start threshold",
            value=str(_ex("cluster_start_threshold", 2.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Initial stat threshold for cluster search",
        )

        self.cluster_step_field = ft.TextField(
            width=290,
            label="Step size",
            value=str(_ex("cluster_step", 0.1)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Threshold increment per iteration",
        )

        self.cluster_return_all_checkbox = ft.Checkbox(
            label="Return all clusters (use for whole-brain atlases like gray matter)",
            value=_ex("cluster_return_all", False),
        )

        self.cluster_settings_section = ft.Column(
            [
                st.field_row(self.atlas_source_dropdown),
                st.field_row(self.atlas_region_dropdown),
                st.field_pair(self.cluster_start_threshold_field, self.cluster_step_field),
                self.cluster_return_all_checkbox,
            ],
            visible=threshold_method == "cluster",
            spacing=st.GAP_MD,
        )

        # Cluster-extent settings
        self.extent_stat_threshold_field = ft.TextField(
            width=290,
            label="Stat threshold",
            value=str(_ex("cluster_start_threshold", 2.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Fixed stat threshold (e.g. 2.0 for t-stat)",
        )

        self.min_cluster_size_field = ft.TextField(
            width=290,
            label="Min cluster size (voxels)",
            value=str(_ex("min_cluster_size", 50)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Clusters smaller than this are discarded",
        )

        max_total = _ex("max_total_voxels")
        self.max_total_voxels_field = ft.TextField(
            width=290,
            label="Max total voxels (optional)",
            value=str(max_total) if max_total else "",
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Safety cap on total mask size (leave empty for no cap)",
        )

        self.extent_grey_matter_checkbox = ft.Checkbox(
            label="Restrict to grey matter",
            value=atlas_source == "gray_matter" and threshold_method == "cluster_extent",
        )

        self.cluster_extent_settings_section = ft.Column(
            [
                st.field_pair(self.extent_stat_threshold_field, self.min_cluster_size_field),
                st.field_row(self.max_total_voxels_field),
                self.extent_grey_matter_checkbox,
            ],
            visible=threshold_method == "cluster_extent",
            spacing=st.GAP_MD,
        )

        # Create localizer task parameter fields
        self._create_task_parameter_fields()

        # If this localizer's task declares its own duration, hide the
        # editable n_volumes field, show the computed value, and wire
        # live refresh on TR / task-param changes.
        self._configure_computed_n_volumes()

        # Manual mask import fields (used when type == "manual")
        self._create_manual_mask_fields()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ft.Container:
        """Build the tab content."""
        is_rest = self._loc_type == "rest"
        is_manual = self._loc_type == "manual"

        # Network dropdowns row (shows only visible ones)
        visible_dropdowns: list[ft.Control] = [dd for dd in self._network_dropdowns if dd.visible]
        self.network_dropdowns_row = ft.Row(visible_dropdowns, wrap=True, spacing=st.GAP_SM, run_spacing=st.GAP_SM)

        # Target Networks section (rest-only for ICA)
        self.target_networks_section = ft.Container(
            content=ft.Column(
                [
                    st.section("Target Networks"),
                    st.caption("Select target networks for neurofeedback training."),
                    ft.Divider(),
                    st.subsection("Network Atlas"),
                    st.field_row(self.network_atlas_dropdown),
                    ft.Divider(),
                    st.field_row(self.n_target_networks_dropdown),
                    self.network_dropdowns_row,
                    ft.Divider(),
                ],
                spacing=st.GAP_MD,
            ),
            visible=is_rest,
        )

        # ICA Settings section (rest only)
        self.networks_section = ft.Container(
            content=ft.Column(
                [
                    st.section("ICA Settings"),
                    st.caption("ICA decomposition parameters for resting-state localizer"),
                    ft.Divider(),
                    st.field_row(self.n_components_field),
                    st.field_row(self.smooth_fwhm_field),
                    st.field_row(self.highpass_field),
                    ft.Divider(),
                ],
                spacing=st.GAP_MD,
            ),
            visible=is_rest,
        )

        # Task-specific parameters section
        task_params = self._build_task_specific_params()
        self.task_params_section = ft.Column(task_params, spacing=st.GAP_MD)

        # Manual mask import section
        self.manual_section = self._build_manual_section()

        # Assemble — a manual localizer imports pre-built masks and has no
        # scan, ICA/GLM extraction, or task parameters.
        if is_manual:
            controls: list[ft.Control] = [
                st.section("Localizer Type"),
                self.type_display_text,
                ft.Divider(),
                self.manual_section,
            ]
        else:
            controls = [
                st.section("Localizer Type"),
                self.type_display_text,
                self.contrast_info_text,
                ft.Divider(),
                st.section("Scan Parameters"),
                st.caption("Configure the localizer scan that identifies participant-specific brain networks."),
                st.field_row(self.tr_field),
                st.field_row(self.n_volumes_field),
                self.n_volumes_info_text,
                st.field_row(self.n_runs_field),
                st.field_row(self.motion_correction_dropdown),
                ft.Divider(),
                self.target_networks_section,
                self.networks_section,
                self.task_params_section,
                self.include_reverse_contrasts_checkbox,
                st.section("Mask Extraction"),
                st.caption("Configure how network masks are extracted from the stat maps."),
                st.field_row(self.threshold_field),
                st.field_row(self.threshold_method_dropdown),
                self.cluster_settings_section,
                self.cluster_extent_settings_section,
            ]

        return ft.Container(
            content=ft.Column(controls, scroll=ft.ScrollMode.AUTO, spacing=st.GAP_MD),
            padding=st.PAD,
        )

    # ------------------------------------------------------------------
    # Collect config
    # ------------------------------------------------------------------

    def collect_config(self) -> dict[str, Any]:
        """Gather current field values into a config dict (new nested schema)."""
        loc_type = self._loc_type or "rest"

        if loc_type == "manual":
            return self._collect_manual_config()

        is_rest = loc_type == "rest"

        extraction: dict[str, Any] = {
            "threshold": int(self.threshold_field.value or 2000),
            "threshold_method": self.threshold_method_dropdown.value or "top_n",
        }

        cfg: dict[str, Any] = {
            "type": loc_type,
            "id": self.config.get("id", loc_type),
            "tr": float(self.tr_field.value or 1.2),
            "n_volumes": int(self.n_volumes_field.value or 250),
            "n_runs": int(self.n_runs_field.value or 2),
            "motion_correction": self.motion_correction_dropdown.value or "siemens_moco",
        }

        # Cluster settings — go into extraction sub-block
        if extraction["threshold_method"] == "cluster":
            extraction["atlas_source"] = self.atlas_source_dropdown.value
            region = self.atlas_region_dropdown.value
            extraction["atlas_regions"] = [region] if region and region != "__loading__" else None
            extraction["cluster_start_threshold"] = float(self.cluster_start_threshold_field.value or 2.0)
            extraction["cluster_step"] = float(self.cluster_step_field.value or 0.1)
            extraction["cluster_return_all"] = bool(self.cluster_return_all_checkbox.value)
        elif extraction["threshold_method"] == "cluster_extent":
            extraction["cluster_start_threshold"] = float(self.extent_stat_threshold_field.value or 2.0)
            extraction["min_cluster_size"] = int(self.min_cluster_size_field.value or 50)
            max_val = self.max_total_voxels_field.value
            extraction["max_total_voxels"] = int(max_val) if max_val else None
            if self.extent_grey_matter_checkbox.value:
                extraction["atlas_source"] = "gray_matter"

        if is_rest:
            network_atlas = self.network_atlas_dropdown.value or "yeo_17"
            extraction["network_atlas"] = network_atlas
            smooth_val = self.smooth_fwhm_field.value
            extraction["smooth_fwhm"] = float(smooth_val) if smooth_val and smooth_val != "None" else 5.0
            highpass_val = self.highpass_field.value
            extraction["highpass"] = float(highpass_val) if highpass_val and highpass_val != "None" else 0.01
            extraction["lowpass"] = None
            n_comp_val = self.n_components_field.value
            extraction["n_components"] = int(n_comp_val) if n_comp_val and n_comp_val != "None" else 30

            # Network targets
            targets = []
            for i, dd in enumerate(self._network_dropdowns):
                if i >= self._n_target_networks:
                    break
                if dd.value:
                    targets.append(dd.value)
            cfg["networks"] = targets
        else:
            # Task-based localizer
            cfg["include_reverse_contrasts"] = self.include_reverse_contrasts_checkbox.value
            # Get contrasts from task metadata
            contrasts = self.config.get("contrasts", {})
            if not contrasts:
                try:
                    meta = get_task_metadata(loc_type)
                    contrasts = dict(meta.default_contrasts) if meta.default_contrasts else {}
                except (KeyError, ValueError):
                    pass
            cfg["contrasts"] = contrasts
            cfg["stat_type"] = self.config.get("stat_type", "t")
            cfg["hrf_model"] = self.config.get("hrf_model", "spm")
            cfg["drift_model"] = self.config.get("drift_model", "cosine")

            # Task-specific parameters
            task_params = self._collect_task_params()
            if task_params:
                cfg["task_params"] = task_params

        cfg["extraction"] = extraction

        # Preserve existing simulator source paths (managed by Paths tab)
        if "source_dicoms" in self.config:
            cfg["source_dicoms"] = self.config["source_dicoms"]

        return cfg

    def get_mask_names(self) -> list[str]:
        """Return the mask names this localizer will produce."""
        loc_type = self._loc_type or "rest"
        if loc_type == "manual":
            return [name for row in self._manual_mask_rows if (name := (row["name"].value or "").strip())]
        if loc_type == "rest":
            names = []
            for i, dd in enumerate(self._network_dropdowns):
                if i >= self._n_target_networks:
                    break
                if dd.value:
                    names.append(dd.value)
            return names
        else:
            # Task-based: contrast keys
            contrasts = self.config.get("contrasts", {})
            if not contrasts:
                try:
                    meta = get_task_metadata(loc_type)
                    contrasts = meta.default_contrasts or {}
                except (KeyError, ValueError):
                    pass
            names = list(contrasts.keys())
            if self.include_reverse_contrasts_checkbox.value:
                names = names + [f"{n}_reverse" for n in names]
            return names

    # ------------------------------------------------------------------
    # Manual mask import
    # ------------------------------------------------------------------

    def _create_manual_mask_fields(self) -> None:
        """Create the controls backing the Manual Mask Import section.

        Only participant-agnostic data lives in the study config: the mask
        space and the network names. Each participant's actual mask /
        reference EPI / T1 files are selected later in the participant
        workflow.
        """
        c = self.config

        self.manual_mask_space_dropdown = ft.Dropdown(
            label="Mask space",
            value=c.get("manual_mask_space", "native"),
            options=[
                ft.dropdown.Option(key="native", text="Native (EPI / functional space)"),
                ft.dropdown.Option(key="anat", text="Anatomical (T1 space)"),
                ft.dropdown.Option(key="mni", text="MNI152 standard space"),
            ],
        )

        # One row per network name. Seed from saved config.
        self._manual_mask_rows: list[dict[str, Any]] = []
        for name in c.get("manual_mask_names", []) or []:
            self._manual_mask_rows.append(self._make_manual_mask_row(str(name)))
        if not self._manual_mask_rows:
            self._manual_mask_rows.append(self._make_manual_mask_row(""))

        self.manual_masks_column = ft.Column([row["row"] for row in self._manual_mask_rows])

    def _make_manual_mask_row(self, name: str) -> dict[str, Any]:
        """Build one network-name row (name field + remove button)."""
        name_field = ft.TextField(
            label="Network name",
            value=name,
            hint_text="e.g. DefaultA",
            on_change=lambda _: self._notify_change(),
        )
        row_data: dict[str, Any] = {"name": name_field}
        row_data["row"] = st.browse_row(
            name_field,
            ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                tooltip="Remove this network",
                on_click=lambda _, rd=row_data: self._on_remove_mask_row(rd),
            ),
        )
        return row_data

    def _build_manual_section(self) -> ft.Container:
        """Build the Manual Mask Import section container."""
        return ft.Container(
            content=ft.Column(
                [
                    st.section("Manual Mask Import"),
                    st.caption(
                        "Use pre-built masks instead of extracting them. The mask files, "
                        "reference EPI, and T1 differ per participant — they are selected in "
                        "each participant's workflow at the Import Masks step. Here you only "
                        "set the mask space and the network names.",
                    ),
                    ft.Divider(),
                    st.field_row(self.manual_mask_space_dropdown),
                    st.subsection("Network names"),
                    st.caption("Each network becomes a feedback mask; the name wires it to feedback runs."),
                    self.manual_masks_column,
                    ft.TextButton("Add network", icon=ft.Icons.ADD, on_click=self._on_add_mask_row),
                ],
                spacing=st.GAP_MD,
            ),
        )

    def _collect_manual_config(self) -> dict[str, Any]:
        """Gather manual-mask study-config values (participant-agnostic only)."""
        names = [n for row in self._manual_mask_rows if (n := (row["name"].value or "").strip())]
        return {
            "type": "manual",
            "id": self.config.get("id", "manual"),
            "manual_mask_space": self.manual_mask_space_dropdown.value or "native",
            "manual_mask_names": names,
        }

    def _notify_change(self) -> None:
        """Fire the on_change callback if one is registered."""
        if self._on_change:
            self._on_change()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_reverse_contrasts_changed(self, e):
        """Handle reverse contrasts checkbox change."""
        if self._on_change:
            self._on_change()

    def _on_n_target_networks_changed(self, e):
        """Handle change in number of target networks."""
        try:
            n = int(self.n_target_networks_dropdown.value or 2)
        except (ValueError, TypeError):
            n = 2
        self._n_target_networks = n

        # Update dropdown visibility
        for i, dd in enumerate(self._network_dropdowns):
            dd.visible = i < n

        # Rebuild the row
        if hasattr(self, "network_dropdowns_row"):
            self.network_dropdowns_row.controls.clear()
            self.network_dropdowns_row.controls.extend(dd for dd in self._network_dropdowns if dd.visible)

        if self._on_change:
            self._on_change()

        with contextlib.suppress(Exception):
            self.page.update()

    def _on_network_atlas_changed(self, e):
        """Handle network atlas change."""
        from pineuro.mask_extraction import ATLAS_SOURCES

        atlas_key = self.network_atlas_dropdown.value
        n = ATLAS_SOURCES.get(atlas_key, {}).get("n_networks", 17)
        new_options = self._get_yeo_network_options(n)

        for dropdown in self._network_dropdowns:
            old = dropdown.value
            dropdown.options = [ft.dropdown.Option(key=o.key, text=o.text) for o in new_options]
            valid = [o.key for o in new_options]
            if old not in valid:
                dropdown.value = None

        if self._on_change:
            self._on_change()

        with contextlib.suppress(Exception):
            self.page.update()

    def _on_threshold_method_changed(self, e):
        """Handle threshold method change."""
        method = self.threshold_method_dropdown.value
        self.cluster_settings_section.visible = method == "cluster"
        self.cluster_extent_settings_section.visible = method == "cluster_extent"
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_atlas_source_changed(self, e):
        """Handle cluster atlas source change."""
        import threading

        atlas_source = self.atlas_source_dropdown.value
        if not atlas_source:
            return

        if atlas_source == "gray_matter":
            self.atlas_region_dropdown.options = []
            self.atlas_region_dropdown.value = None
            self.atlas_region_dropdown.visible = False
            with contextlib.suppress(Exception):
                self.page.update()
            return

        self.atlas_region_dropdown.visible = True
        self.atlas_region_dropdown.options = [ft.dropdown.Option(key="__loading__", text="Loading regions...")]
        self.atlas_region_dropdown.value = "__loading__"
        with contextlib.suppress(Exception):
            self.page.update()

        def _fetch():
            try:
                from pineuro.mask_extraction import get_atlas_labels

                labels = get_atlas_labels(atlas_source)
                self.atlas_region_dropdown.options = [ft.dropdown.Option(key=label, text=label) for label in labels]
                self.atlas_region_dropdown.value = labels[0] if labels else None
            except Exception as ex:
                logger.warning(f"Failed to load atlas labels for {atlas_source}: {ex}")
                self.atlas_region_dropdown.options = []
                self.atlas_region_dropdown.value = None

            async def _apply():
                with contextlib.suppress(Exception):
                    self.page.update()

            with contextlib.suppress(Exception):
                self.page.run_task(_apply)

        threading.Thread(target=_fetch, daemon=True).start()

    def _on_n_runs_changed(self, e):
        """Handle n_runs field change."""
        self._validate_field("n_runs", self.n_runs_field, validate_positive_int)
        if self._on_change:
            self._on_change()

    def _on_add_mask_row(self, e):
        """Append a blank network-name row."""
        row_data = self._make_manual_mask_row("")
        self._manual_mask_rows.append(row_data)
        self.manual_masks_column.controls.append(row_data["row"])
        with contextlib.suppress(Exception):
            self.manual_masks_column.update()
        self._notify_change()

    def _on_remove_mask_row(self, row_data: dict[str, Any]):
        """Remove a network-name row, keeping at least one."""
        if row_data in self._manual_mask_rows:
            self._manual_mask_rows.remove(row_data)
        if row_data["row"] in self.manual_masks_column.controls:
            self.manual_masks_column.controls.remove(row_data["row"])
        if not self._manual_mask_rows:
            new_row = self._make_manual_mask_row("")
            self._manual_mask_rows.append(new_row)
            self.manual_masks_column.controls.append(new_row["row"])
        with contextlib.suppress(Exception):
            self.manual_masks_column.update()
        self._notify_change()

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

    def _get_yeo_network_options(self, n_networks: int) -> list:
        """Get network options for dropdowns."""
        from ..utils import YEO_7_LABELS, YEO_17_LABELS

        labels = YEO_7_LABELS if n_networks == 7 else YEO_17_LABELS
        return [ft.dropdown.Option(key=label, text=label) for label in labels]

    def _get_contrast_info_text(self, task_type: str) -> str:
        """Get contrast info string for the selected localizer task."""
        if task_type == "rest":
            return ""
        try:
            meta = get_task_metadata(task_type)
            if meta.default_contrasts:
                parts = [f"{name}: {expr}" for name, expr in meta.default_contrasts.items()]
                return f"GLM contrast: {', '.join(parts)}"
        except (KeyError, ValueError):
            pass
        return ""

    def _create_task_parameter_fields(self):
        """Create form fields for localizer task parameters (non-rest)."""
        loc_type = self._loc_type
        self.task_param_fields.clear()

        if loc_type == "rest":
            return

        try:
            meta = get_task_metadata(loc_type)
        except ValueError:
            return

        task_config = self.config.get("task_params", {})
        for param in meta.parameters:
            field = self._create_field_for_parameter(param, task_config)
            self.task_param_fields[param.key] = field

    def _configure_computed_n_volumes(self):
        """Hide n_volumes_field and show a computed-value label when the
        selected task declares its own duration via
        ``TaskMetadata.compute_max_run_seconds``.

        Also wires refresh handlers onto ``tr_field`` and every task-param
        field so the displayed value updates live as the user edits.
        """
        loc_type = self._loc_type
        if loc_type == "rest":
            return
        try:
            meta = get_task_metadata(loc_type)
        except (KeyError, ValueError):
            return
        if meta.compute_max_run_seconds is None:
            return

        self.n_volumes_field.visible = False
        self.n_volumes_info_text.visible = True
        self._refresh_n_volumes_info()

        # Wrap the existing on_change so we both run validation/save and
        # refresh the computed display. Same pattern for every task-param
        # field so a change to e.g. rating_timeout updates the display.
        self._wrap_on_change_for_refresh(self.tr_field)
        for field in self.task_param_fields.values():
            target = field.controls[0] if isinstance(field, ft.Row) else field
            self._wrap_on_change_for_refresh(target)

    def _wrap_on_change_for_refresh(self, control: Any) -> None:
        """Append a call to ``_refresh_n_volumes_info`` to a control's on_change."""
        existing = getattr(control, "on_change", None)

        def wrapped(e, _existing=existing):
            if _existing is not None:
                _existing(e)
            self._refresh_n_volumes_info()

        with contextlib.suppress(AttributeError):
            control.on_change = wrapped

    def _refresh_n_volumes_info(self) -> None:
        """Recompute the displayed n_volumes from current TR and task params."""
        if not self.n_volumes_info_text.visible:
            return
        try:
            meta = get_task_metadata(self._loc_type)
        except (KeyError, ValueError):
            return
        if meta.compute_max_run_seconds is None:
            return

        try:
            tr = float(self.tr_field.value)
        except (TypeError, ValueError):
            tr = 0.0
        if tr <= 0:
            self.n_volumes_info_text.value = "Auto-computed: (set a valid TR)"
            with contextlib.suppress(Exception):
                self.page.update()
            return

        try:
            params = self._collect_task_params()
        except (TypeError, ValueError):
            params = self.config.get("task_params", {})

        duration = meta.compute_max_run_seconds(params)
        n_volumes = math.ceil(duration / tr)
        self.n_volumes_info_text.value = (
            f"Auto-computed: {n_volumes} volumes (worst-case run "
            f"~{duration:.0f}s at TR={tr}s; collection stops early "
            f"when the task finishes)"
        )
        with contextlib.suppress(Exception):
            self.page.update()

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
        loc_type = self._loc_type
        if loc_type == "rest":
            return []

        try:
            meta = get_task_metadata(loc_type)
        except ValueError:
            return []

        if not meta.parameters:
            return []

        controls: list[ft.Control] = [
            st.section(f"{meta.display_name} Parameters"),
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

        controls.append(ft.Divider())
        return controls

    def _collect_task_params(self) -> dict[str, Any]:
        """Collect task-specific parameter values."""
        loc_type = self._loc_type
        if loc_type == "rest":
            return {}

        try:
            meta = get_task_metadata(loc_type)
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
