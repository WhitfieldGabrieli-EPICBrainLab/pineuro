"""Base class for Flet modal dialogs.

All dialogs in the pineuro GUI follow the same lifecycle:
1. Instantiate with ``page`` and callback(s).
2. Call ``show()`` to open the ``ft.AlertDialog``.
3. Internally call ``_close_dialog()`` before invoking any callback.

This base class extracts that shared lifecycle so subclasses only need to
implement ``show()`` (build content + call ``_open_dialog()``) and call
``_close_dialog()`` before invoking their callbacks.
"""

import logging

import flet as ft

logger = logging.getLogger(__name__)


class BaseDialog:
    """Minimal base class for Flet AlertDialog wrappers.

    Subclasses should:
    - Accept ``page`` as the first ``__init__`` parameter and call
      ``super().__init__(page)``.
    - Implement ``show()`` to build an ``ft.AlertDialog`` and call
      ``self._open_dialog(dialog)``.
    - Call ``self._close_dialog()`` before invoking any confirm/cancel
      callback to ensure the dialog is dismissed first.

    Attributes
    ----------
    page : ft.Page
        The Flet page used for overlay management.
    _dialog : ft.AlertDialog or None
        The currently open dialog, or None if not shown yet.
    """

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._dialog: ft.AlertDialog | None = None

    def _open_dialog(self, dialog: ft.AlertDialog) -> None:
        """Attach *dialog* to the page overlay and open it.

        Parameters
        ----------
        dialog : ft.AlertDialog
            The dialog to display.
        """
        self._dialog = dialog
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _close_dialog(self) -> None:
        """Close the currently open dialog (if any)."""
        if self._dialog is not None:
            self._dialog.open = False
            self.page.update()
