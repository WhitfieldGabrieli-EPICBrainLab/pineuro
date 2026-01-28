"""
UI helper functions for neurofeedback tasks.

This module provides reusable visual elements and utilities to reduce
boilerplate when creating custom tasks. All functions work with PsychoPy
visual elements.

Example
-------
```python
from pineuro.tasks.ui_helpers import create_fixation_cross, display_text

class MyTask(BaseTask):
    def setup(self):
        self.win = visual.Window([800, 600], color='black')
        self.fixation = create_fixation_cross(self.win)

    def update(self, data):
        if data['is_baseline']:
            self.fixation.draw()
        else:
            # Draw task stimuli
            ...
        self.win.flip()
```
"""

import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


def create_fixation_cross(
    win,
    size: float = 0.05,
    color: str = 'white',
    line_width: float = 3,
    pos: Tuple[float, float] = (0, 0),
):
    """Create a fixation cross stimulus.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window to create stimulus for
    size : float
        Size of the cross (half-width/height, default: 0.05)
    color : str
        Color of the cross (default: 'white')
    line_width : float
        Width of the cross lines (default: 3)
    pos : tuple of float
        Position (x, y) of the cross center (default: (0, 0))

    Returns
    -------
    psychopy.visual.ShapeStim
        A cross-shaped stimulus that can be drawn with .draw()
    """
    from psychopy import visual

    vertices = [
        (-size, 0), (size, 0),  # Horizontal line
        (0, 0),  # Return to center
        (0, -size), (0, size),  # Vertical line
    ]

    return visual.ShapeStim(
        win,
        vertices=vertices,
        closeShape=False,
        lineColor=color,
        lineWidth=line_width,
        pos=pos,
    )


def create_progress_bar(
    win,
    width: float = 0.8,
    height: float = 0.05,
    pos: Tuple[float, float] = (0, -0.4),
    fill_color: str = 'green',
    outline_color: str = 'white',
    line_width: float = 2,
):
    """Create a progress bar with outline and fill.

    Returns a tuple of (outline_rect, fill_rect). Update the fill by
    setting fill_rect.width and adjusting fill_rect.pos to keep it
    left-aligned within the outline.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window
    width : float
        Total width of the progress bar (default: 0.8)
    height : float
        Height of the progress bar (default: 0.05)
    pos : tuple of float
        Position of the bar center (default: (0, -0.4))
    fill_color : str
        Color of the fill (default: 'green')
    outline_color : str
        Color of the outline (default: 'white')
    line_width : float
        Width of the outline (default: 2)

    Returns
    -------
    tuple of (Rect, Rect)
        (outline, fill) rectangles
    """
    from psychopy import visual

    outline = visual.Rect(
        win,
        width=width,
        height=height,
        pos=pos,
        lineColor=outline_color,
        lineWidth=line_width,
        fillColor=None,
    )

    fill = visual.Rect(
        win,
        width=0,  # Start empty
        height=height - 0.01,  # Slightly smaller than outline
        pos=(pos[0] - width / 2, pos[1]),  # Left-aligned
        fillColor=fill_color,
        lineColor=None,
    )

    return outline, fill


def update_progress_bar(
    fill_rect,
    progress: float,
    total_width: float,
    left_x: float,
):
    """Update a progress bar fill to show current progress.

    Parameters
    ----------
    fill_rect : psychopy.visual.Rect
        The fill rectangle from create_progress_bar()
    progress : float
        Progress value from 0.0 to 1.0
    total_width : float
        Total width of the progress bar
    left_x : float
        X position of the left edge of the bar
    """
    progress = max(0.0, min(1.0, progress))
    fill_width = progress * total_width
    fill_rect.width = fill_width
    fill_rect.pos = (left_x + fill_width / 2, fill_rect.pos[1])


def display_text(
    win,
    text: str,
    duration: Optional[float] = None,
    wait_key: bool = False,
    wait_keys: list = None,
    pos: Tuple[float, float] = (0, 0),
    height: float = 0.05,
    color: str = 'white',
    wrap_width: float = 1.5,
) -> Optional[str]:
    """Display text on screen, optionally for a duration or until keypress.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window
    text : str
        Text to display
    duration : float, optional
        Display duration in seconds. If None and wait_key is False, just draws
        (caller must flip). If set, displays for that duration.
    wait_key : bool
        If True, wait for any key press (default: False)
    wait_keys : list of str, optional
        Specific keys to wait for (default: any key)
    pos : tuple of float
        Position of text (default: (0, 0))
    height : float
        Text height (default: 0.05)
    color : str
        Text color (default: 'white')
    wrap_width : float
        Width for text wrapping (default: 1.5)

    Returns
    -------
    str or None
        Key pressed if wait_key is True, otherwise None
    """
    from psychopy import visual, event, core

    text_stim = visual.TextStim(
        win,
        text=text,
        pos=pos,
        height=height,
        color=color,
        wrapWidth=wrap_width,
    )

    text_stim.draw()
    win.flip()

    if duration is not None:
        core.wait(duration)
        return None

    if wait_key:
        event.clearEvents('keyboard')
        while True:
            keys = event.getKeys(keyList=wait_keys)
            if keys:
                return keys[0]
            core.wait(0.01)

    return None


def create_circle(
    win,
    radius: float = 0.1,
    pos: Tuple[float, float] = (0, 0),
    fill_color: str = 'white',
    line_color: Optional[str] = None,
    line_width: float = 0,
):
    """Create a circle stimulus.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window
    radius : float
        Circle radius (default: 0.1)
    pos : tuple of float
        Position (x, y) of the circle center (default: (0, 0))
    fill_color : str
        Fill color (default: 'white')
    line_color : str, optional
        Outline color (default: None, no outline)
    line_width : float
        Outline width (default: 0)

    Returns
    -------
    psychopy.visual.Circle
        A circle stimulus
    """
    from psychopy import visual

    return visual.Circle(
        win,
        radius=radius,
        pos=pos,
        fillColor=fill_color,
        lineColor=line_color,
        lineWidth=line_width,
    )


def create_target_zone(
    win,
    y_center: float,
    height: float,
    width: float = 0.2,
    color: str = 'yellow',
    line_width: float = 2,
    opacity: float = 0.5,
):
    """Create a horizontal target zone indicator.

    Useful for thermometer-style tasks where users aim for a target level.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window
    y_center : float
        Y position of the zone center
    height : float
        Height of the target zone
    width : float
        Width of the zone (default: 0.2)
    color : str
        Color of the zone outline (default: 'yellow')
    line_width : float
        Width of the outline (default: 2)
    opacity : float
        Opacity of the zone (default: 0.5)

    Returns
    -------
    psychopy.visual.Rect
        A rectangle representing the target zone
    """
    from psychopy import visual

    return visual.Rect(
        win,
        width=width,
        height=height,
        pos=(0, y_center),
        fillColor=None,
        lineColor=color,
        lineWidth=line_width,
        opacity=opacity,
    )


def wait_with_escape(
    duration: float,
    check_stop_callback=None,
) -> bool:
    """Wait for a duration, checking for escape key or stop signal.

    Parameters
    ----------
    duration : float
        Duration to wait in seconds
    check_stop_callback : callable, optional
        Function that returns True if task should stop

    Returns
    -------
    bool
        True if completed without interruption, False if escaped/stopped

    Raises
    ------
    KeyboardInterrupt
        If escape is pressed or stop signal received
    """
    from psychopy import event, core

    clock = core.Clock()
    event.clearEvents('keyboard')

    while clock.getTime() < duration:
        if check_stop_callback and check_stop_callback():
            raise KeyboardInterrupt("Stopped by user")

        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            raise KeyboardInterrupt("Escaped by user")

        core.wait(0.01)

    return True
