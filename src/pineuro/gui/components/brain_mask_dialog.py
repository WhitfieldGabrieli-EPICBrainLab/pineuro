"""Dialog to review a localizer's brain mask and adjust how much non-brain tissue it excludes.

UI only. The participant workflow screen supplies the callbacks that preview a
mask (:func:`pineuro.preprocessing.preview_brain_mask`) and commit it
(:func:`pineuro.preprocessing.recompute_brain_mask`). Free of heavy imports.
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import Callable
from pathlib import Path

import flet as ft

from .. import styles as st
from .base_dialog import BaseDialog

# Cleanup radii offered (0 disables the cleanup; the pipeline default is 2)
CLEANUP_LEVELS = range(5)


def parse_brain_mask_values(fraction_text: str | None, cleanup_value: str | None) -> tuple[float, int]:
    """The (threshold fraction, cleanup) the operator entered; ValueError if invalid."""
    try:
        fraction = float((fraction_text or "").strip())
    except ValueError:
        raise ValueError("Intensity threshold must be a number") from None
    if fraction <= 0:
        raise ValueError("Intensity threshold must be positive")
    try:
        cleanup = int(cleanup_value or "")
    except ValueError:
        raise ValueError("Choose a cleanup level") from None
    return fraction, cleanup


class BrainMaskReviewDialog(BaseDialog):
    """Show a brain mask's outline on the mean EPI, with the two settings that shape it.

    ``on_preview(fraction, cleanup)`` should compute a preview and call
    :meth:`set_preview`; ``on_accept(fraction, cleanup)`` commits the values
    (the dialog closes first).
    """

    def __init__(
        self,
        page: ft.Page,
        *,
        title: str,
        image_path: Path,
        voxel_text: str,
        current: tuple[float, int],
        study_defaults: tuple[float, int],
        adjusted: bool,
        locked_reason: str | None = None,
        on_preview: Callable[[float, int], None],
        on_accept: Callable[[float, int], None],
    ) -> None:
        super().__init__(page)
        self._title = title
        self._current = current
        self._study_defaults = study_defaults
        self._adjusted = adjusted
        self._locked_reason = locked_reason
        self._on_preview = on_preview
        self._on_accept = on_accept

        self.image = ft.Image(src=self._data_uri(image_path), fit=ft.BoxFit.CONTAIN, height=st.PREVIEW_H_LG + 60)
        self.voxel_text = ft.Text(voxel_text, size=st.BODY, weight=ft.FontWeight.W_500)
        self.status_text = ft.Text("", size=st.CAPTION, color=st.MUTED)
        self.error_text = ft.Text("", size=st.CAPTION, color=st.ERROR, visible=False)
        self.fraction_field = ft.TextField(
            label="Intensity threshold (× image mean)",
            value=str(current[0]),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text=f"Study default {study_defaults[0]}",
        )
        self.cleanup_dropdown = ft.Dropdown(
            label="Cleanup (voxels)",
            value=str(current[1]),
            options=[ft.dropdown.Option(key=str(n), text=str(n)) for n in CLEANUP_LEVELS],
        )
        self.accept_button = ft.FilledButton("Accept", icon=ft.Icons.CHECK, on_click=self._accept)

    @staticmethod
    def _data_uri(path: Path) -> str:
        return f"data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode('utf-8')}"

    def values(self) -> tuple[float, int]:
        """The values currently entered; ValueError if invalid."""
        return parse_brain_mask_values(self.fraction_field.value, self.cleanup_dropdown.value)

    def show(self) -> None:
        source = "Adjusted for this participant" if self._adjusted else "Study default"
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.PSYCHOLOGY, color=st.ACCENT),
                    ft.Text(self._title, expand=True),
                    ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close", on_click=lambda _e: self._close_dialog()),
                ]
            ),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "The red outline is the brain mask on the unmasked mean EPI. Raise the "
                            "intensity threshold to exclude more dim non-brain tissue; raise cleanup to "
                            "detach tissue joined to the brain by a narrow connection. Lower either if "
                            "the mask cuts off brain.",
                            size=st.BODY,
                            color=st.MUTED,
                        ),
                        ft.Container(
                            content=self.image,
                            border=ft.Border.all(1, st.PANEL_BORDER),
                            border_radius=5,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        ),
                        self.voxel_text,
                        ft.Text(
                            f"Current: {self._current[0]} × mean, cleanup {self._current[1]} ({source})",
                            size=st.CAPTION,
                            color=st.MUTED,
                        ),
                        st.field_pair(self.fraction_field, self.cleanup_dropdown),
                        ft.Row(
                            [
                                ft.OutlinedButton("Recompute", icon=ft.Icons.REFRESH, on_click=self._preview),
                                ft.TextButton("Reset to study defaults", icon=ft.Icons.RESTORE, on_click=self._reset),
                            ],
                            spacing=st.GAP_SM,
                        ),
                        self.status_text,
                        self.error_text,
                        ft.Text(
                            "Accepting changed values recomputes the brain mask and BOLD reference, "
                            "and re-runs mask extraction and coregistration for this localizer."
                            + (f" {self._locked_reason}" if self._locked_reason else ""),
                            size=st.CAPTION,
                            color=st.WARNING_FG if self._locked_reason else st.MUTED,
                        ),
                    ],
                    spacing=st.GAP_MD,
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=st.DIALOG_XL,
                height=640,
            ),
            actions=[
                ft.OutlinedButton("Close", on_click=lambda _e: self._close_dialog()),
                self.accept_button,
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self._open_dialog(dialog)

    def set_preview(self, image_path: Path, voxel_text: str) -> None:
        """Show a recomputed preview."""
        self.image.src = self._data_uri(image_path)
        self.voxel_text.value = voxel_text
        self.status_text.value = "Preview (not applied yet)"
        self.error_text.visible = False
        self._refresh()

    def set_status(self, text: str) -> None:
        self.status_text.value = text
        self._refresh()

    def show_error(self, text: str) -> None:
        self.error_text.value = text
        self.error_text.visible = True
        self._refresh()

    def _refresh(self) -> None:
        with contextlib.suppress(Exception):
            self.page.update()

    def _preview(self, _e: object) -> None:
        try:
            fraction, cleanup = self.values()
        except ValueError as ex:
            self.show_error(str(ex))
            return
        self.error_text.visible = False
        self.set_status("Computing preview…")
        self._on_preview(fraction, cleanup)

    def _reset(self, _e: object) -> None:
        self.fraction_field.value = str(self._study_defaults[0])
        self.cleanup_dropdown.value = str(self._study_defaults[1])
        self._refresh()

    def _accept(self, _e: object) -> None:
        try:
            fraction, cleanup = self.values()
        except ValueError as ex:
            self.show_error(str(ex))
            return
        if (fraction, cleanup) != self._current and self._locked_reason:
            self.show_error(self._locked_reason)
            return
        self._close_dialog()
        self._on_accept(fraction, cleanup)
