"""Screenshot-based task mirror component for GUI.

This module provides a component that displays screenshots from the PsychoPy
window, providing a true mirror of what participants see during tasks.

The TaskMirror receives base64-encoded PNG screenshots from tasks and
displays them in the GUI.
"""

import contextlib
import logging
from typing import Any

import flet as ft

from pineuro.gui import styles as st

logger = logging.getLogger(__name__)


class TaskMirror:
    """Task visualization that displays PsychoPy screenshots.

    This component receives screenshot data from tasks and displays them,
    providing a researcher-facing mirror of what the participant sees.

    Parameters
    ----------
    width : float, optional
        Display width in pixels (default: 480)
    height : float, optional
        Display height in pixels (default: 270)

    Examples
    --------
    >>> mirror = TaskMirror(width=480, height=270)
    >>> container = mirror.build()  # Add to Flet page
    >>> # Later, when task emits screenshot:
    >>> mirror.update_state({
    ...     'type': 'screenshot',
    ...     'image': '<base64 PNG data>'
    ... })
    """

    def __init__(self, width: float | None = st.TASK_MIRROR_W, height: float | None = st.TASK_MIRROR_H):
        """Initialize the task mirror.

        Parameters
        ----------
        width : float or None
            Display width in pixels, or None for responsive sizing
        height : float or None
            Display height in pixels, or None for responsive sizing
        """
        self.width = width
        self.height = height
        self._image: ft.Image | None = None
        self._container: ft.Container | None = None

    def build(self) -> ft.Container:
        """Build the image container.

        Returns
        -------
        ft.Container
            Container with the image display, ready to add to a Flet page
        """
        # 1x1 black pixel PNG as data URI placeholder
        placeholder = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        self._image = ft.Image(
            src=placeholder,
            width=self.width,
            height=self.height,
            fit=ft.BoxFit.CONTAIN,
        )
        self._container = ft.Container(
            content=self._image,
            bgcolor=ft.Colors.BLACK,
            border=ft.Border.all(1, st.PANEL_BORDER),
            border_radius=5,
            width=self.width,
            height=self.height,
        )
        return self._container

    def update_state(self, state: dict[str, Any]) -> None:
        """Update with new visual state from task.

        Parameters
        ----------
        state : dict
            Visual state dictionary containing:
            - 'type': 'screenshot'
            - 'image': data URI (e.g., 'data:image/jpeg;base64,...') or raw base64
        """
        if state.get("type") == "screenshot" and self._image is not None:
            img_data = state.get("image", "")
            if img_data:
                # Use data URI directly if provided, otherwise wrap as PNG
                if img_data.startswith("data:"):
                    self._image.src = img_data
                else:
                    self._image.src = f"data:image/png;base64,{img_data}"
                # Image update can fail during shutdown; safe to ignore.
                with contextlib.suppress(Exception):
                    self._image.update()

    def reset(self) -> None:
        """Reset to empty state."""
        if self._image is not None:
            self._image.src = ""
            with contextlib.suppress(Exception):
                self._image.update()
