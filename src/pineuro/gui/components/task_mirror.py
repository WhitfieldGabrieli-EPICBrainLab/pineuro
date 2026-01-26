"""Screenshot-based task mirror component for GUI.

This module provides a component that displays screenshots from the PsychoPy
window, providing a true mirror of what participants see during tasks.

The TaskMirror receives base64-encoded PNG screenshots from tasks and
displays them in the GUI.
"""

import logging
from typing import Any, Dict, Optional

import flet as ft

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

    def __init__(self, width: float = 480, height: float = 270):
        """Initialize the task mirror.

        Parameters
        ----------
        width : float
            Display width in pixels
        height : float
            Display height in pixels
        """
        self.width = width
        self.height = height
        self._image: Optional[ft.Image] = None
        self._container: Optional[ft.Container] = None

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
            fit="contain",
        )
        self._container = ft.Container(
            content=self._image,
            bgcolor=ft.Colors.BLACK,
            border=ft.border.all(1, ft.Colors.GREY_400),
            border_radius=5,
            width=self.width,
            height=self.height,
        )
        return self._container

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update with new visual state from task.

        Parameters
        ----------
        state : dict
            Visual state dictionary containing:
            - 'type': 'screenshot'
            - 'image': data URI (e.g., 'data:image/jpeg;base64,...') or raw base64
        """
        if state.get('type') == 'screenshot' and self._image is not None:
            img_data = state.get('image', '')
            if img_data:
                # Use data URI directly if provided, otherwise wrap as PNG
                if img_data.startswith('data:'):
                    self._image.src = img_data
                else:
                    self._image.src = f"data:image/png;base64,{img_data}"
                try:
                    self._image.update()
                except Exception as e:
                    logger.debug(f"Image update failed: {e}")

    def reset(self) -> None:
        """Reset to empty state."""
        if self._image is not None:
            self._image.src_base64 = None
            try:
                self._image.update()
            except Exception:
                pass
