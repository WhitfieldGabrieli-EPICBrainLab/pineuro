"""Construction smoke tests for migrated dialogs with no other coverage.

These build the dialog content trees (via ``__new__`` + minimal stubs, so no real
``page`` / config / disk is needed) purely to exercise the style-token call sites
at runtime — a guard against a token being used where its value type doesn't fit
the Flet kwarg. They assert the builder returns a control and does not raise.
"""

from pathlib import Path

import flet as ft


class TestUnifiedAttemptDialog:
    def test_build_content_constructs(self):
        from pineuro.gui.components.unified_attempt_dialog import (
            AttemptInfo,
            RunGroup,
            UnifiedAttemptDialog,
        )

        dlg = UnifiedAttemptDialog.__new__(UnifiedAttemptDialog)
        dlg._checkboxes = {}
        dlg.run_groups = [
            RunGroup(
                run_id=1,
                run_name="Run 1",
                attempts=[
                    AttemptInfo(
                        attempt_num=1, directory=Path("a"), n_volumes=200, expected_volumes=200, is_complete=True
                    ),
                    AttemptInfo(
                        attempt_num=2, directory=Path("b"), n_volumes=50, expected_volumes=200, is_complete=False
                    ),
                    AttemptInfo(
                        attempt_num=3, directory=Path("c"), n_volumes=0, expected_volumes=200, is_complete=False
                    ),
                ],
            )
        ]

        content = dlg._build_content()
        assert isinstance(content, ft.Control)


class TestQCPlotViewer:
    def test_build_content_constructs(self, tmp_path):
        from pineuro.gui.components.qc_plot_viewer import QCPlotViewer

        png = tmp_path / "plot.png"
        png.write_bytes(b"\x89PNG\r\n")  # _build_content only checks .exists()

        viewer = QCPlotViewer.__new__(QCPlotViewer)
        viewer.plots = [("Brain Mask Coverage", png), ("Missing", tmp_path / "nope.png")]

        content = viewer._build_content()
        assert isinstance(content, ft.Control)
