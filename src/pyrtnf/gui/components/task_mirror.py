"""Generic task mirror component for GUI.

This module provides a Canvas-based component that renders any task's
visual state using drawable primitives. Tasks describe their visual
representation via get_mirror_state(), and this component renders
those primitives generically.

Supported primitive types:
- circle: x, y, radius, fill or stroke, stroke_width
- line: x1, y1, x2, y2, stroke, stroke_width
- rect: x, y, width, height, fill or stroke, stroke_width
- text: x, y, content, size, color

All coordinates are normalized 0-1 (x: left-right, y: top-bottom).
"""

import logging
from typing import Any, Dict, List, Optional

import flet as ft
import flet.canvas as cv

logger = logging.getLogger(__name__)


class TaskMirror:
    """Generic task visualization that renders primitive shapes.

    This component receives visual state dictionaries from tasks and
    renders them on a Flet Canvas. It provides a researcher-facing
    mirror of what the participant sees on the PsychoPy display.

    Parameters
    ----------
    width : float, optional
        Canvas width in pixels (default: 200)
    height : float, optional
        Canvas height in pixels (default: 300)

    Examples
    --------
    >>> mirror = TaskMirror(width=200, height=300)
    >>> container = mirror.build()  # Add to Flet page
    >>> # Later, when task emits state:
    >>> mirror.update_state({
    ...     'background': 'black',
    ...     'shapes': [
    ...         {'type': 'circle', 'x': 0.5, 'y': 0.5, 'radius': 0.1, 'fill': 'white'}
    ...     ]
    ... })
    """

    def __init__(self, width: float = 200, height: float = 300):
        """Initialize the task mirror.

        Parameters
        ----------
        width : float
            Canvas width in pixels
        height : float
            Canvas height in pixels
        """
        self.width = width
        self.height = height
        self.canvas: Optional[cv.Canvas] = None
        self._container: Optional[ft.Container] = None
        self._current_state: Dict[str, Any] = {'background': 'black', 'shapes': []}

    def build(self) -> ft.Container:
        """Build the canvas container.

        Returns
        -------
        ft.Container
            Container with the canvas, ready to add to a Flet page
        """
        self.canvas = cv.Canvas(
            shapes=self._render_shapes(),
            width=self.width,
            height=self.height,
        )
        self._container = ft.Container(
            content=self.canvas,
            border=ft.border.all(1, ft.Colors.GREY_400),
            border_radius=5,
        )
        return self._container

    def update_state(self, state: Dict[str, Any]) -> None:
        """Update with new visual state from task.

        Parameters
        ----------
        state : dict
            Visual state dictionary from task.get_mirror_state()
        """
        self._current_state = state
        if self._container:
            # Create a fresh canvas with the new shapes
            self.canvas = cv.Canvas(
                shapes=self._render_shapes(),
                width=self.width,
                height=self.height,
            )
            # Replace the container's content with the new canvas
            self._container.content = self.canvas
            try:
                self._container.update()
            except Exception as e:
                logger.debug(f"Container update failed: {e}")

    def _render_shapes(self) -> List[cv.Shape]:
        """Convert primitive state to Flet canvas shapes.

        Returns
        -------
        List[cv.Shape]
            List of Flet canvas shapes to render
        """
        shapes = []
        state = self._current_state

        # Background
        bg_color = state.get('background', 'black')
        shapes.append(cv.Rect(
            x=0, y=0, width=self.width, height=self.height,
            paint=ft.Paint(style=ft.PaintingStyle.FILL, color=bg_color)
        ))

        # Render each primitive
        for prim in state.get('shapes', []):
            rendered = self._render_primitive(prim)
            if rendered:
                if isinstance(rendered, list):
                    shapes.extend(rendered)
                else:
                    shapes.append(rendered)

        return shapes

    def _render_primitive(self, prim: dict) -> Optional[cv.Shape]:
        """Render a single primitive to canvas shape.

        Parameters
        ----------
        prim : dict
            Primitive dictionary with 'type' and type-specific properties

        Returns
        -------
        cv.Shape or None
            Rendered canvas shape, or None if type is unknown
        """
        ptype = prim.get('type')

        if ptype == 'circle':
            return self._render_circle(prim)
        elif ptype == 'line':
            return self._render_line(prim)
        elif ptype == 'rect':
            return self._render_rect(prim)
        elif ptype == 'text':
            return self._render_text(prim)
        else:
            logger.debug(f"Unknown primitive type: {ptype}")
            return None

    def _render_circle(self, prim: dict) -> cv.Circle:
        """Render a circle primitive.

        Parameters
        ----------
        prim : dict
            Circle primitive with x, y, radius, and fill or stroke

        Returns
        -------
        cv.Circle
            Rendered circle shape
        """
        x = prim['x'] * self.width
        y = prim['y'] * self.height
        # Use minimum dimension to keep circles proportional
        r = prim['radius'] * min(self.width, self.height)

        if 'fill' in prim:
            return cv.Circle(
                x=x, y=y, radius=r,
                paint=ft.Paint(
                    style=ft.PaintingStyle.FILL,
                    color=prim['fill']
                )
            )
        else:
            return cv.Circle(
                x=x, y=y, radius=r,
                paint=ft.Paint(
                    style=ft.PaintingStyle.STROKE,
                    color=prim.get('stroke', 'white'),
                    stroke_width=prim.get('stroke_width', 2)
                )
            )

    def _render_line(self, prim: dict) -> cv.Line:
        """Render a line primitive.

        Parameters
        ----------
        prim : dict
            Line primitive with x1, y1, x2, y2, stroke, stroke_width

        Returns
        -------
        cv.Line
            Rendered line shape
        """
        return cv.Line(
            x1=prim['x1'] * self.width,
            y1=prim['y1'] * self.height,
            x2=prim['x2'] * self.width,
            y2=prim['y2'] * self.height,
            paint=ft.Paint(
                style=ft.PaintingStyle.STROKE,
                color=prim.get('stroke', 'white'),
                stroke_width=prim.get('stroke_width', 2)
            )
        )

    def _render_rect(self, prim: dict) -> cv.Rect:
        """Render a rectangle primitive.

        Parameters
        ----------
        prim : dict
            Rectangle primitive with x, y, width, height, and fill or stroke

        Returns
        -------
        cv.Rect
            Rendered rectangle shape
        """
        x = prim['x'] * self.width
        y = prim['y'] * self.height
        w = prim['width'] * self.width
        h = prim['height'] * self.height

        if 'fill' in prim:
            style = ft.PaintingStyle.FILL
            color = prim['fill']
        else:
            style = ft.PaintingStyle.STROKE
            color = prim.get('stroke', 'white')

        return cv.Rect(
            x=x, y=y, width=w, height=h,
            paint=ft.Paint(
                style=style,
                color=color,
                stroke_width=prim.get('stroke_width', 2)
            )
        )

    def _render_text(self, prim: dict) -> cv.Text:
        """Render a text primitive.

        Parameters
        ----------
        prim : dict
            Text primitive with x, y, content, size, color

        Returns
        -------
        cv.Text
            Rendered text shape
        """
        x = prim['x'] * self.width
        y = prim['y'] * self.height

        return cv.Text(
            x=x, y=y,
            value=prim['content'],
            style=ft.TextStyle(
                size=prim.get('size', 12),
                foreground=ft.Paint(color=prim.get('color', 'white'))
            )
        )

    def reset(self) -> None:
        """Reset to empty state."""
        self._current_state = {'background': 'black', 'shapes': []}
        if self.canvas:
            self.canvas.shapes = self._render_shapes()
            try:
                self.canvas.update()
            except Exception:
                pass
