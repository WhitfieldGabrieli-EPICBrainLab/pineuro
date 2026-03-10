"""Localizer tab builder for the study editor.

Each instance manages the full form fields and layout for a single localizer
tab (Localizer 1, Localizer 2, etc.).
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

# Uniform widths matching StudyEditorScreen
FIELD_W = 400
PAIR_W = 195


def _f(control: ft.Control) -> ft.Row:
    """Wrap a control in a Row so Column doesn't stretch it."""
    return ft.Row([control])


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
        self.config = dict(config)  # working copy
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
            color=ft.Colors.BLUE_700,
        )

        # Contrast info text
        self.contrast_info_text = ft.Text(
            value=self._get_contrast_info_text(loc_type),
            size=12,
            color=ft.Colors.BLUE_700,
            italic=True,
        )

        # Scan params
        self.tr_field = ft.TextField(
            width=600,
            label="TR (seconds)",
            value=str(c.get("tr", 1.2)),
            hint_text="Repetition time of the localizer sequence (e.g. 1.2)",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("tr", self.tr_field, validate_positive_number),
        )

        self.n_volumes_field = ft.TextField(
            width=600,
            label="Volumes per localizer run",
            value=str(c.get("n_volumes", 200)),
            hint_text="Volumes acquired per localizer run",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_volumes", self.n_volumes_field, validate_positive_int),
        )

        self.n_runs_field = ft.TextField(
            width=600,
            label="Number of localizer runs",
            value=str(c.get("n_runs", 2)),
            hint_text="Number of localizer runs",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=self._on_n_runs_changed,
        )

        # Reverse contrasts checkbox (task-based only)
        self.include_reverse_contrasts_checkbox = ft.Checkbox(
            label="Include reverse contrasts (enables dual-network from single-contrast localizer)",
            value=c.get("include_reverse_contrasts", False),
            visible=loc_type != "rest",
            on_change=self._on_reverse_contrasts_changed,
        )

        # Network atlas
        from pineuro.mask_extraction import ATLAS_SOURCES

        network_atlas = c.get("network_atlas", "yeo_17")
        self.network_atlas_dropdown = ft.Dropdown(
            width=600,
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
        yeo_targets = c.get("yeo_targets", ["DefaultA", "ContA"])
        self._n_target_networks = len(yeo_targets) if yeo_targets else 2

        current_n = ATLAS_SOURCES.get(network_atlas, {}).get("n_networks", 17)
        net_options = self._get_yeo_network_options(current_n)

        self.n_target_networks_dropdown = ft.Dropdown(
            width=600,
            label="Number of networks",
            options=[ft.dropdown.Option(str(i), str(i)) for i in range(1, MAX_NETWORKS + 1)],
            value=str(self._n_target_networks),
            on_select=self._on_n_target_networks_changed,
        )

        self._network_dropdowns: list[ft.Dropdown] = []
        for i in range(MAX_NETWORKS):
            val = yeo_targets[i] if i < len(yeo_targets) else None
            dd = ft.Dropdown(
                width=290,
                label=f"Network {i + 1}",
                options=[ft.dropdown.Option(key=o.key, text=o.text) for o in net_options],
                value=val,
                visible=(i < self._n_target_networks),
            )
            self._network_dropdowns.append(dd)

        # ICA settings (rest only)
        self.n_components_field = ft.TextField(
            width=600,
            label="ICA components",
            value=str(c.get("n_components", 20)),
            hint_text="Number of ICA components for decomposition",
            keyboard_type=ft.KeyboardType.NUMBER,
            on_change=lambda _: self._validate_field("n_components", self.n_components_field, validate_positive_int),
        )

        self.smooth_fwhm_field = ft.TextField(
            width=600,
            label="Spatial smoothing (mm FWHM)",
            value=str(c.get("smooth_fwhm", 5.0)),
            hint_text="Gaussian kernel size in mm",
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        self.highpass_field = ft.TextField(
            width=600,
            label="Highpass filter (Hz)",
            value=str(c.get("highpass", 0.01)),
            hint_text="Cutoff frequency for highpass filter",
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        # Mask extraction
        self.threshold_field = ft.TextField(
            width=600,
            label="Voxels per mask",
            value=str(c.get("threshold", 2000)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Number of voxels to keep per extracted mask",
            on_change=lambda _: self._validate_field("threshold", self.threshold_field, validate_positive_int),
        )

        self.threshold_method_dropdown = ft.Dropdown(
            width=600,
            label="Thresholding method",
            options=[
                ft.dropdown.Option(key="top_n", text="Top N voxels"),
                ft.dropdown.Option(key="cluster", text="Atlas-guided cluster"),
            ],
            value=c.get("threshold_method", "top_n"),
            on_select=self._on_threshold_method_changed,
        )

        # Cluster settings
        cluster_atlas_options = [
            ft.dropdown.Option(key=k, text=v["display_name"])
            for k, v in ATLAS_SOURCES.items()
            if v["type"] in ("anatomical", "tissue") or k.startswith("schaefer")
        ]
        self.atlas_source_dropdown = ft.Dropdown(
            width=600,
            label="Atlas",
            options=cluster_atlas_options,
            value=c.get("atlas_source", "harvard_oxford"),
            on_select=self._on_atlas_source_changed,
        )

        current_regions = c.get("atlas_regions") or []
        self.atlas_region_dropdown = ft.Dropdown(
            width=600,
            label="Region",
            options=[],
            value=current_regions[0] if current_regions else None,
        )

        self.cluster_start_threshold_field = ft.TextField(
            width=290,
            label="Start threshold",
            value=str(c.get("cluster_start_threshold", 2.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Initial stat threshold for cluster search",
        )

        self.cluster_step_field = ft.TextField(
            width=290,
            label="Step size",
            value=str(c.get("cluster_step", 0.1)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Threshold increment per iteration",
        )

        self.cluster_return_all_checkbox = ft.Checkbox(
            label="Return all clusters (use for whole-brain atlases like gray matter)",
            value=c.get("cluster_return_all", False),
        )

        self.cluster_settings_section = ft.Column(
            [
                self.atlas_source_dropdown,
                self.atlas_region_dropdown,
                ft.Row([self.cluster_start_threshold_field, self.cluster_step_field]),
                self.cluster_return_all_checkbox,
            ],
            visible=c.get("threshold_method", "top_n") == "cluster",
        )

        # Create localizer task parameter fields
        self._create_task_parameter_fields()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ft.Container:
        """Build the tab content."""
        is_rest = self._loc_type == "rest"

        # Network dropdowns row (shows only visible ones)
        visible_dropdowns: list[ft.Control] = [dd for dd in self._network_dropdowns if dd.visible]
        self.network_dropdowns_row = ft.Row(visible_dropdowns, wrap=True)

        # Target Networks section (rest-only for ICA)
        self.target_networks_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Target Networks", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Select target networks for neurofeedback training.",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Divider(),
                    ft.Text("Network Atlas", size=14, weight=ft.FontWeight.BOLD),
                    self.network_atlas_dropdown,
                    ft.Divider(),
                    self.n_target_networks_dropdown,
                    self.network_dropdowns_row,
                    ft.Divider(),
                ]
            ),
            visible=is_rest,
        )

        # ICA Settings section (rest only)
        self.networks_section = ft.Container(
            content=ft.Column(
                [
                    ft.Text("ICA Settings", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "ICA decomposition parameters for resting-state localizer",
                        size=11,
                        color=ft.Colors.GREY_600,
                        italic=True,
                    ),
                    ft.Divider(),
                    self.n_components_field,
                    self.smooth_fwhm_field,
                    self.highpass_field,
                    ft.Divider(),
                ]
            ),
            visible=is_rest,
        )

        # Task-specific parameters section
        task_params = self._build_task_specific_params()
        self.task_params_section = ft.Column(task_params)

        # Assemble
        controls: list[ft.Control] = [
            ft.Text("Localizer Type", size=16, weight=ft.FontWeight.BOLD),
            self.type_display_text,
            self.contrast_info_text,
            ft.Divider(),
            ft.Text("Scan Parameters", size=16, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Configure the localizer scan that identifies participant-specific brain networks.",
                size=11,
                color=ft.Colors.GREY_600,
                italic=True,
            ),
            _f(self.tr_field),
            self.n_volumes_field,
            self.n_runs_field,
            ft.Divider(),
            self.target_networks_section,
            self.networks_section,
            self.task_params_section,
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
        ]

        return ft.Container(
            content=ft.Column(controls, scroll=ft.ScrollMode.AUTO),
            padding=20,
        )

    # ------------------------------------------------------------------
    # Collect config
    # ------------------------------------------------------------------

    def collect_config(self) -> dict[str, Any]:
        """Gather current field values into a config dict."""
        from pineuro.mask_extraction import ATLAS_SOURCES

        loc_type = self._loc_type or "rest"
        is_rest = loc_type == "rest"

        cfg: dict[str, Any] = {
            "type": loc_type,
            "tr": float(self.tr_field.value or 1.2),
            "n_volumes": int(self.n_volumes_field.value or 200),
            "n_runs": int(self.n_runs_field.value or 2),
            "threshold": int(self.threshold_field.value or 2000),
            "threshold_method": self.threshold_method_dropdown.value or "top_n",
        }

        # Cluster settings
        if cfg["threshold_method"] == "cluster":
            cfg["atlas_source"] = self.atlas_source_dropdown.value
            region = self.atlas_region_dropdown.value
            cfg["atlas_regions"] = [region] if region and region != "__loading__" else None
            cfg["cluster_start_threshold"] = float(self.cluster_start_threshold_field.value or 2.0)
            cfg["cluster_step"] = float(self.cluster_step_field.value or 0.1)
            cfg["cluster_return_all"] = bool(self.cluster_return_all_checkbox.value)

        if is_rest:
            network_atlas = self.network_atlas_dropdown.value or "yeo_17"
            cfg["network_atlas"] = network_atlas
            cfg["n_yeo"] = ATLAS_SOURCES.get(network_atlas, {}).get("n_networks", 17)
            smooth_val = self.smooth_fwhm_field.value
            cfg["smooth_fwhm"] = float(smooth_val) if smooth_val and smooth_val != "None" else 5.0
            highpass_val = self.highpass_field.value
            cfg["highpass"] = float(highpass_val) if highpass_val and highpass_val != "None" else 0.01
            cfg["lowpass"] = None
            n_comp_val = self.n_components_field.value
            cfg["n_components"] = int(n_comp_val) if n_comp_val and n_comp_val != "None" else 20

            # Network targets
            targets = []
            for i, dd in enumerate(self._network_dropdowns):
                if i >= self._n_target_networks:
                    break
                if dd.value:
                    targets.append(dd.value)
            cfg["yeo_targets"] = targets
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

        # Preserve existing simulator source paths (managed by Paths tab)
        if "source_dicoms" in self.config:
            cfg["source_dicoms"] = self.config["source_dicoms"]

        return cfg

    def get_mask_names(self) -> list[str]:
        """Return the mask names this localizer will produce."""
        loc_type = self._loc_type or "rest"
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
            ft.Text(f"{meta.display_name} Parameters", size=16, weight=ft.FontWeight.BOLD),
            ft.Divider(),
        ]

        for param in meta.parameters:
            field = self.task_param_fields.get(param.key)
            if field is not None:
                controls.append(field)
                if param.hint and param.type != "bool":
                    controls.append(ft.Text(param.hint, size=11, color=ft.Colors.GREY_600, italic=True))

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
