"""Lifecycle tests for the ``BaseDialog`` AlertDialog wrapper.

``BaseDialog`` extracts the shared open/close lifecycle for Flet modal dialogs.
Subclasses build an ``ft.AlertDialog`` and call ``_open_dialog`` / ``_close_dialog``.
These tests exercise that lifecycle against the ``mock_flet_page`` fixture and assert
observable state transitions: overlay membership, ``dialog.open``, and ``page.update``
call counts.
"""

import flet as ft

from pineuro.gui.components.base_dialog import BaseDialog


class _ConcreteDialog(BaseDialog):
    """Minimal subclass that builds and opens a real ``ft.AlertDialog``."""

    def show(self) -> ft.AlertDialog:
        dialog = ft.AlertDialog(content=ft.Text("hello"))
        self._open_dialog(dialog)
        return dialog


def test_init_starts_with_no_dialog(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)
    assert dlg.page is mock_flet_page
    assert dlg._dialog is None


def test_open_appends_dialog_to_overlay(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)
    dialog = dlg.show()
    assert dialog in mock_flet_page.overlay
    assert mock_flet_page.overlay == [dialog]


def test_open_sets_open_true_and_records_handle(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)
    dialog = dlg.show()
    assert dialog.open is True
    assert dlg._dialog is dialog


def test_open_triggers_page_update(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)
    dlg.show()
    mock_flet_page.update.assert_called_once()


def test_close_sets_open_false(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)
    dialog = dlg.show()
    assert dialog.open is True

    dlg._close_dialog()
    assert dialog.open is False


def test_close_triggers_additional_page_update(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)
    dlg.show()
    assert mock_flet_page.update.call_count == 1

    dlg._close_dialog()
    assert mock_flet_page.update.call_count == 2


def test_close_keeps_dialog_in_overlay(mock_flet_page):
    # Closing only flips ``open``; the wrapper does not remove the control.
    dlg = _ConcreteDialog(mock_flet_page)
    dialog = dlg.show()
    dlg._close_dialog()
    assert dialog in mock_flet_page.overlay


def test_close_without_open_is_noop(mock_flet_page):
    # No dialog has been shown, so ``_dialog`` is None and close must not update.
    dlg = _ConcreteDialog(mock_flet_page)
    dlg._close_dialog()
    mock_flet_page.update.assert_not_called()
    assert dlg._dialog is None


def test_full_open_close_open_cycle(mock_flet_page):
    dlg = _ConcreteDialog(mock_flet_page)

    first = dlg.show()
    assert first.open is True
    dlg._close_dialog()
    assert first.open is False

    # Reopening builds a fresh dialog and replaces the handle.
    second = dlg.show()
    assert second is not first
    assert second.open is True
    assert dlg._dialog is second
    # Each open appends; the overlay accumulates both controls.
    assert mock_flet_page.overlay == [first, second]
    # 2 opens + 1 close = 3 page updates.
    assert mock_flet_page.update.call_count == 3
