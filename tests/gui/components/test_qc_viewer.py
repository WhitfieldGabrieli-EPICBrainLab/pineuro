"""Tests for pineuro.gui.components.qc_viewer.open_qc_report.

Covers the branch behavior of opening a QC HTML report in a browser:
- missing report path returns False and warns (no browser call)
- existing report path returns True and opens the file:// URL
- Linux + explicit screen routes through the screen-aware helper
  (pineuro.qc._open_html_in_browser); every other branch falls back to
  webbrowser.open
- exceptions are swallowed and surface as a False return

Heavy/external side effects (the browser launch, the screen-aware helper,
and platform detection) are mocked; the function under test is not.
"""

from unittest.mock import patch

import pytest

from pineuro.gui.components.qc_viewer import open_qc_report


@pytest.fixture
def report(tmp_path):
    """A real on-disk HTML report file."""
    path = tmp_path / "sub-01_qc.html"
    path.write_text("<html><body>QC</body></html>", encoding="utf-8")
    return path


# =============================================================================
# Missing report
# =============================================================================


class TestMissingReport:
    def test_returns_false_when_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist.html"

        with patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb:
            result = open_qc_report(missing)

        assert result is False
        # Must not attempt to open anything when the file is absent.
        mock_wb.open.assert_not_called()

    def test_logs_warning_when_missing(self, tmp_path, caplog):
        missing = tmp_path / "nope.html"

        with (
            caplog.at_level("WARNING", logger="pineuro.gui.components.qc_viewer"),
            patch("pineuro.gui.components.qc_viewer.webbrowser"),
        ):
            open_qc_report(missing)

        assert any("not found" in rec.message for rec in caplog.records)
        assert any(str(missing) in rec.message for rec in caplog.records)


# =============================================================================
# Existing report — default (non-Linux / no screen) path
# =============================================================================


class TestExistingReportFallback:
    def test_returns_true_and_opens_url(self, report):
        with (
            patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb,
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Windows"),
        ):
            result = open_qc_report(report)

        assert result is True
        mock_wb.open.assert_called_once()
        # The URL opened is the resolved file:// URI of the report.
        opened_url = mock_wb.open.call_args.args[0]
        assert opened_url == report.resolve().as_uri()
        assert opened_url.startswith("file:")

    def test_screen_aware_helper_not_used_on_non_linux(self, report):
        # Even when a screen is requested, a non-Linux platform must fall back.
        with (
            patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb,
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Darwin"),
            patch("pineuro.qc._open_html_in_browser") as mock_helper,
        ):
            result = open_qc_report(report, screen=1)

        assert result is True
        mock_wb.open.assert_called_once()
        mock_helper.assert_not_called()

    def test_linux_without_screen_uses_webbrowser(self, report):
        # Linux but screen is None → no screen-aware routing.
        with (
            patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb,
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Linux"),
            patch("pineuro.qc._open_html_in_browser") as mock_helper,
        ):
            result = open_qc_report(report, screen=None)

        assert result is True
        mock_wb.open.assert_called_once()
        mock_helper.assert_not_called()


# =============================================================================
# Existing report — Linux screen-aware path
# =============================================================================


class TestLinuxScreenAware:
    def test_linux_with_screen_uses_helper(self, report):
        with (
            patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb,
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Linux"),
            patch("pineuro.qc._open_html_in_browser") as mock_helper,
        ):
            result = open_qc_report(report, screen=2)

        assert result is True
        # Screen-aware helper handles the open; webbrowser fallback is skipped.
        mock_helper.assert_called_once_with(report.resolve())
        mock_wb.open.assert_not_called()

    def test_linux_screen_zero_still_uses_helper(self, report):
        # screen=0 is "not None" → the screen-aware branch is taken.
        with (
            patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb,
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Linux"),
            patch("pineuro.qc._open_html_in_browser") as mock_helper,
        ):
            result = open_qc_report(report, screen=0)

        assert result is True
        mock_helper.assert_called_once_with(report.resolve())
        mock_wb.open.assert_not_called()

    def test_falls_back_to_webbrowser_when_helper_import_fails(self, report):
        # If the qc helper can't be imported, the Linux branch degrades to
        # webbrowser.open rather than failing.
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "pineuro.qc":
                raise ImportError("simulated missing qc helper")
            return real_import(name, *args, **kwargs)

        with (
            patch("pineuro.gui.components.qc_viewer.webbrowser") as mock_wb,
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Linux"),
            patch("builtins.__import__", side_effect=fake_import),
        ):
            result = open_qc_report(report, screen=1)

        assert result is True
        mock_wb.open.assert_called_once()
        assert mock_wb.open.call_args.args[0] == report.resolve().as_uri()


# =============================================================================
# Exception handling
# =============================================================================


class TestExceptionHandling:
    def test_returns_false_when_browser_raises(self, report, caplog):
        with (
            patch(
                "pineuro.gui.components.qc_viewer.webbrowser.open",
                side_effect=RuntimeError("no display"),
            ),
            patch("pineuro.gui.components.qc_viewer.platform.system", return_value="Windows"),
            caplog.at_level("ERROR", logger="pineuro.gui.components.qc_viewer"),
        ):
            result = open_qc_report(report)

        assert result is False
        assert any("Failed to open QC report" in rec.message for rec in caplog.records)


# =============================================================================
# Import sanity — webbrowser/platform are the real stdlib modules
# =============================================================================


class TestModuleWiring:
    def test_module_uses_stdlib_modules(self):
        import platform as stdlib_platform
        import webbrowser as stdlib_webbrowser

        from pineuro.gui.components import qc_viewer

        # Guards the patch targets above: if the module ever stops importing
        # these by name, the branch tests would silently no-op.
        assert qc_viewer.webbrowser is stdlib_webbrowser
        assert qc_viewer.platform is stdlib_platform
