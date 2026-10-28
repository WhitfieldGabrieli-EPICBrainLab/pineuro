"""Controls for changing a task (GLM) mask at review: re-threshold, fallback, replace.

UI and formatting only. The participant workflow screen supplies the callbacks
that preview and apply a change through
:func:`pineuro.mask_extraction.update_task_mask`. Free of heavy imports, so it
is safe to import at the GUI's top level.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import flet as ft

from .. import styles as st

# Editable re-threshold fields per method: (parameter, label, kind). The label's
# "{stat}" is the stat type (t or z). "optional_int" may be left empty (no cap).
THRESHOLD_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    "top_n": [("threshold", "Voxels to keep", "int")],
    "cluster": [
        ("threshold", "Target voxels", "int"),
        ("cluster_start_threshold", "Start threshold ({stat})", "float"),
    ],
    "cluster_extent": [
        ("cluster_start_threshold", "Cluster-forming threshold ({stat} ≥)", "float"),
        ("min_cluster_size", "Min cluster size (voxels)", "int"),
        ("max_total_voxels", "Max total voxels (optional)", "optional_int"),
    ],
}

_PROVENANCE = {
    "operator_rethreshold": "Re-thresholded at review",
    "fallback_configured": "Study fallback mask",
    "operator_replace": "Replaced at review",
    "atlas_roi": "Search atlas region",
}


def mask_lock_reason(completed_step_ids: Iterable[str]) -> str | None:
    """Why masks can no longer be changed, or None while they still can.

    Once a feedback or transfer run has completed, changing the masks would
    leave that run's data without a record of the mask it used.
    """
    if any(step_id.startswith(("feedback_", "transfer_")) for step_id in completed_step_ids):
        return "Masks are locked: a feedback run in this session has already been completed with them."
    return None


def fallback_label(spec: dict[str, str]) -> str:
    """Short label for a configured fallback mask (``extraction.fallback_masks`` value)."""
    if "file" in spec:
        return Path(spec["file"]).name
    from pineuro.atlas_registry import ATLAS_SOURCES

    display = ATLAS_SOURCES.get(spec["atlas"], {}).get("display_name", spec["atlas"])
    return f"{display} {spec['network']}"


def _threshold_text(params: dict[str, Any], stat_type: str) -> str:
    method = params.get("threshold_method")
    if method == "cluster_extent":
        text = f"{stat_type} ≥ {params['cluster_start_threshold']}, min {params['min_cluster_size']} voxels"
        return text + (f", cap {params['max_total_voxels']}" if params.get("max_total_voxels") else "")
    if method == "cluster":
        return f"target {params['threshold']} voxels, start {stat_type} ≥ {params['cluster_start_threshold']}"
    return f"top {params.get('threshold')} voxels"


def provenance_label(selection: dict[str, Any]) -> str | None:
    """Badge text for a mask an operator changed at review, or None for the study's own mask."""
    reason = selection.get("selection_reason", "auto")
    if reason == "auto":
        return None
    label = str(_PROVENANCE.get(reason, reason))
    if reason == "operator_rethreshold" and selection.get("threshold_params"):
        return f"{label} ({_threshold_text(selection['threshold_params'], selection.get('stat_type', 't'))})"
    if selection.get("source"):
        return f"{label}: {selection['source']}"
    return label


def preview_summary(selection: dict[str, Any]) -> str:
    """One-line summary of a (previewed) mask: voxels, clusters, largest cluster, outcome."""
    n_voxels = selection.get("n_voxels_thresholded", 0)
    parts = [f"{n_voxels} voxels"]
    table = selection.get("cluster_table") or []
    n_clusters = selection.get("n_clusters_found")
    if n_voxels > 0:
        n = n_clusters if n_clusters is not None else len(table)
        parts.append(f"{n} cluster{'s' if n != 1 else ''}")
        if table:
            parts.append(f"largest {table[0]['n_voxels']}")
    elif selection.get("n_clusters_input"):
        parts.append(
            f"{selection['n_clusters_input']} clusters below the minimum size "
            f"(largest {selection.get('largest_cluster_input')})"
        )
    if selection.get("cluster_outcome"):
        parts.append(str(selection["cluster_outcome"]).replace("_", " "))
    return " · ".join(parts)


class TaskMaskAdjustPanel(ft.Container):
    """Inline panel to re-threshold one task mask with the localizer's own method.

    ``on_preview`` and ``on_apply`` receive a ``{"kind": "rethreshold", ...}``
    override; ``on_reset`` restores the study's parameters; ``on_discard`` drops a
    shown preview (the caller marks one with :meth:`set_preview_pending`).
    """

    def __init__(
        self,
        threshold_method: str,
        values: dict[str, Any],
        *,
        stat_type: str = "t",
        override_active: bool = False,
        locked_reason: str | None = None,
        on_preview: Callable[[dict[str, Any]], None],
        on_apply: Callable[[dict[str, Any]], None],
        on_reset: Callable[[], None],
        on_discard: Callable[[], None] | None = None,
    ):
        self._fields: dict[str, tuple[ft.TextField, str]] = {}
        for key, label, kind in THRESHOLD_FIELDS.get(threshold_method, []):
            value = values.get(key)
            self._fields[key] = (
                ft.TextField(
                    label=label.format(stat=stat_type),
                    value="" if value is None else str(value),
                    keyboard_type=ft.KeyboardType.NUMBER,
                ),
                kind,
            )
        self.result_text = ft.Text("", size=st.CAPTION, color=st.MUTED)
        self.error_text = ft.Text("", size=st.CAPTION, color=st.ERROR, visible=False)
        self.apply_button = ft.FilledButton(
            "Apply",
            icon=ft.Icons.CHECK,
            disabled=locked_reason is not None,
            tooltip=locked_reason,
            on_click=lambda _e: self._emit(on_apply),
        )
        self.discard_button = ft.TextButton(
            "Discard preview",
            icon=ft.Icons.UNDO,
            visible=False,
            on_click=lambda _e: on_discard() if on_discard else None,
        )
        self.reset_button = ft.TextButton(
            "Reset to study parameters",
            icon=ft.Icons.RESTORE,
            visible=override_active,
            disabled=locked_reason is not None,
            on_click=lambda _e: on_reset(),
        )
        super().__init__(
            content=ft.Column(
                [
                    ft.Text(
                        "Preview shows the mask these values give, from the saved GLM map, without "
                        "changing it. Apply replaces the mask; Discard preview keeps the current one.",
                        size=st.CAPTION,
                        color=st.MUTED,
                    ),
                    *[st.field_row(field) for field, _kind in self._fields.values()],
                    ft.Row(
                        [
                            ft.OutlinedButton(
                                "Preview",
                                icon=ft.Icons.VISIBILITY,
                                on_click=lambda _e: self._emit(on_preview),
                            ),
                            self.apply_button,
                            self.discard_button,
                            self.reset_button,
                        ],
                        spacing=st.GAP_SM,
                        wrap=True,
                    ),
                    self.result_text,
                    self.error_text,
                ],
                spacing=st.GAP_SM,
            ),
            visible=False,
            padding=st.GAP_MD,
            border_radius=6,
            border=ft.Border.all(1, st.PANEL_BORDER),
        )

    def build_override(self) -> dict[str, Any]:
        """The re-threshold override for the current field values; ValueError if one is invalid."""
        values: dict[str, Any] = {}
        for key, (field, kind) in self._fields.items():
            raw = (field.value or "").strip()
            if kind == "optional_int" and not raw:
                values[key] = None
                continue
            try:
                number: float = int(raw) if kind in ("int", "optional_int") else float(raw)
            except ValueError:
                raise ValueError(f"{field.label} must be a {'whole ' if kind != 'float' else ''}number") from None
            if number <= 0:
                raise ValueError(f"{field.label} must be positive")
            values[key] = number
        return {"kind": "rethreshold", **values}

    def set_preview_pending(self, pending: bool) -> None:
        """Show the Discard preview button while a preview is shown but not applied."""
        self.discard_button.visible = pending
        with contextlib.suppress(Exception):
            self.update()

    def show_result(self, text: str) -> None:
        self.result_text.value = text
        self.error_text.visible = False
        with contextlib.suppress(Exception):
            self.update()

    def show_error(self, text: str) -> None:
        self.error_text.value = text
        self.error_text.visible = True
        with contextlib.suppress(Exception):
            self.update()

    def _emit(self, callback: Callable[[dict[str, Any]], None]) -> None:
        try:
            override = self.build_override()
        except ValueError as ex:
            self.show_error(str(ex))
            return
        self.error_text.visible = False
        callback(override)


def build_replace_dialog(
    mask_name: str,
    *,
    on_choose: Callable[[dict[str, Any]], None],
    on_cancel: Callable[[], None],
    initial_path: Path | None = None,
) -> ft.AlertDialog:
    """Dialog to replace one task mask with an atlas network, an MNI-space mask file,
    or a mask file in the participant's own space (e.g. an edited copy of the mask).

    ``on_choose`` receives the ``mask_overrides`` entry for the choice;
    ``initial_path`` seeds the file browser.
    """
    from pineuro.atlas_registry import ATLAS_SOURCES

    from ..utils import NIFTI_FILETYPES, YEO_7_LABELS, YEO_17_LABELS, pick_file

    def _network_options(atlas: str) -> list[ft.dropdown.Option]:
        labels = YEO_7_LABELS if ATLAS_SOURCES.get(atlas, {}).get("n_networks") == 7 else YEO_17_LABELS
        return [ft.dropdown.Option(key=label, text=label) for label in labels]

    source = ft.RadioGroup(
        value="atlas",
        content=ft.Column(
            [
                ft.Radio(value="atlas", label="Atlas network (warped from MNI space)"),
                ft.Radio(value="mni_file", label="Mask file in MNI space (warped to this participant)"),
                ft.Radio(
                    value="native_file",
                    label="Mask file in this participant's space (e.g. an edited copy of this mask)",
                ),
            ],
            spacing=0,
        ),
    )
    atlas_dropdown = ft.Dropdown(
        label="Atlas",
        value="yeo_17",
        options=[
            ft.dropdown.Option(key=k, text=v["display_name"])
            for k, v in ATLAS_SOURCES.items()
            if v["type"] == "network"
        ],
    )
    network_dropdown = ft.Dropdown(label="Network", options=_network_options("yeo_17"))
    file_field = ft.TextField(label="Mask file")
    error_text = ft.Text("", size=st.CAPTION, color=st.ERROR, visible=False)
    atlas_row = st.field_pair(atlas_dropdown, network_dropdown)
    file_row = st.browse_row(
        file_field, ft.IconButton(icon=ft.Icons.FILE_OPEN, tooltip="Browse", on_click=lambda e: _browse(e))
    )

    def _show_source_fields() -> None:
        # Only the selected source's fields are shown
        atlas_row.visible = source.value == "atlas"
        file_row.visible = source.value in ("mni_file", "native_file")
        file_field.hint_text = (
            "NIfTI mask on the localizer's BOLD reference grid"
            if source.value == "native_file"
            else "NIfTI mask in MNI152 space"
        )
        error_text.visible = False

    def _on_source(_e: Any) -> None:
        _show_source_fields()
        with contextlib.suppress(Exception):
            dialog.update()

    def _browse(_e: Any) -> None:
        seed = file_field.value or (str(initial_path) if initial_path else None)
        path = pick_file(title="Select mask file", filetypes=NIFTI_FILETYPES, initial_path=seed)
        if path:
            file_field.value = path
            with contextlib.suppress(Exception):
                dialog.update()

    def _on_atlas(_e: Any) -> None:
        network_dropdown.options = _network_options(atlas_dropdown.value or "yeo_17")
        network_dropdown.value = None
        with contextlib.suppress(Exception):
            network_dropdown.update()

    def _on_use(_e: Any) -> None:
        if source.value in ("mni_file", "native_file"):
            path = (file_field.value or "").strip()
            if not path:
                error_text.value, error_text.visible = "Choose a mask file.", True
            else:
                on_choose({"kind": source.value, "file": path, "selection_reason": "operator_replace"})
                return
        elif not network_dropdown.value:
            error_text.value, error_text.visible = "Choose a network.", True
        else:
            on_choose(
                {
                    "kind": "atlas_network",
                    "atlas": atlas_dropdown.value,
                    "network": network_dropdown.value,
                    "selection_reason": "operator_replace",
                }
            )
            return
        with contextlib.suppress(Exception):
            error_text.update()

    atlas_dropdown.on_select = _on_atlas
    source.on_change = _on_source
    _show_source_fields()
    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text(f"Replace {mask_name}"),
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "The replacement is warped to this participant's space and saved under "
                        f"the mask name {mask_name}, for this participant only.",
                        size=st.BODY,
                        color=st.MUTED,
                    ),
                    source,
                    atlas_row,
                    file_row,
                    error_text,
                ],
                spacing=st.GAP_MD,
                tight=True,
            ),
            width=st.DIALOG_MD,
        ),
        actions=[
            ft.TextButton("Cancel", on_click=lambda _e: on_cancel()),
            ft.FilledButton("Use This Mask", icon=ft.Icons.CHECK, on_click=_on_use),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    return dialog
