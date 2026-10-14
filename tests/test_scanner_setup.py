"""Tests for pineuro.scanner_setup."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pineuro.scanner_setup import (
    ConnectionTestResult,
    _parse_ping_latency,
    check_watch_directory,
    get_current_platform_key,
    get_setup_instructions,
    ping_scanner,
    test_scanner_connection,
    watch_dir_account_issue,
)


class TestParsePingLatency:
    def test_windows_format(self):
        assert _parse_ping_latency("Reply from 192.168.2.1: bytes=32 time=1ms TTL=64") == 1.0

    def test_linux_format(self):
        assert _parse_ping_latency("64 bytes from 192.168.2.1: icmp_seq=1 ttl=64 time=0.523 ms") == 0.523

    def test_less_than(self):
        assert _parse_ping_latency("Reply from 192.168.2.1: bytes=32 time<1ms TTL=128") == 1.0

    def test_no_match(self):
        assert _parse_ping_latency("Request timed out.") is None

    def test_empty_string(self):
        assert _parse_ping_latency("") is None


class TestCheckWatchDirectory:
    def test_nonexistent(self, tmp_path):
        exists, writable, empty, count, detail = check_watch_directory(tmp_path / "nonexistent")
        assert exists is False
        assert writable is False
        assert "does not exist" in detail

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        exists, writable, empty, count, detail = check_watch_directory(d)
        assert exists is True
        assert writable is True
        assert empty is True
        assert count == 0
        assert "ready for new session" in detail

    def test_nonempty_dir(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        (d / "IM-0001.dcm").touch()
        (d / "IM-0002.dcm").touch()
        exists, writable, empty, count, detail = check_watch_directory(d)
        assert exists is True
        assert writable is True
        assert empty is False
        assert count == 2
        assert "may need clearing" in detail

    def test_path_is_file(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.touch()
        exists, writable, empty, count, detail = check_watch_directory(f)
        assert exists is False
        assert "not a directory" in detail

    def test_string_path(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        exists, writable, empty, count, detail = check_watch_directory(str(d))
        assert exists is True

    def test_subfolders_reported(self, tmp_path):
        """Siemens-style subfolders are reported in the detail string."""
        d = tmp_path / "watch"
        d.mkdir()
        (d / "seq_f1").mkdir()
        (d / "seq_f2").mkdir()
        exists, writable, empty, count, detail = check_watch_directory(d)
        assert exists is True
        assert "2 subfolder(s)" in detail
        assert "previous sequences" in detail

    def test_subfolders_with_top_level_files(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        (d / "seq_f1").mkdir()
        (d / "stray.dcm").touch()
        exists, writable, empty, count, detail = check_watch_directory(d)
        assert "1 subfolder(s)" in detail
        assert "1 top-level file(s)" in detail

    def test_inaccessible_folder_is_reported_not_raised(self, tmp_path, monkeypatch):
        # e.g. the share lives in another account's home folder: stat itself is denied
        watch = tmp_path / "rt-mgh" / "BOLD_DICOM"
        real_exists = Path.exists

        def exists(self):
            if self == watch:
                raise PermissionError("denied")
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", exists)
        found, writable, _, _, detail = check_watch_directory(watch)

        assert (found, writable) == (False, False)
        assert "permission denied" in detail
        assert "account the scanner writes as" in detail


class TestWatchDirAccountIssue:
    """pineuro must run as the account the scanner writes as (the BOLD Dot Add-In's User Name)."""

    NAMES = {501: "rt-mgh", 502: "jiahe"}

    @staticmethod
    def _fake_stat(table):
        """``os.stat`` stand-in: maps a path's name to (uid, mtime), or to an exception to raise."""

        def _stat(path):
            entry = table.get(Path(path).name)
            if entry is None:
                raise FileNotFoundError(path)
            if isinstance(entry, Exception):
                raise entry
            uid, mtime = entry
            return SimpleNamespace(st_uid=uid, st_mtime=mtime)

        return _stat

    def test_newest_sequence_folder_decides_the_account(self, tmp_path, monkeypatch):
        watch = tmp_path / "BOLD_DICOM"
        (watch / "seq_old").mkdir(parents=True)
        (watch / "seq_new").mkdir()
        monkeypatch.setattr("pineuro.scanner_setup._account_name", self.NAMES.get)
        stat = self._fake_stat({"seq_old": (502, 1.0), "seq_new": (501, 2.0), "BOLD_DICOM": (502, 0.0)})

        issue = watch_dir_account_issue(watch, current_uid=502, stat=stat)

        assert "belongs to the account 'rt-mgh'" in issue
        assert "running as 'jiahe'" in issue

    def test_no_issue_when_running_as_the_scanner_account(self, tmp_path):
        watch = tmp_path / "BOLD_DICOM"
        watch.mkdir()
        stat = self._fake_stat({"BOLD_DICOM": (501, 0.0)})
        assert watch_dir_account_issue(watch, current_uid=501, stat=stat) == ""

    def test_folder_hidden_in_another_account_names_that_account(self, tmp_path, monkeypatch):
        watch = tmp_path / "rt-mgh" / "BOLD_DICOM"
        monkeypatch.setattr("pineuro.scanner_setup._account_name", self.NAMES.get)
        stat = self._fake_stat({"BOLD_DICOM": PermissionError("denied"), "rt-mgh": (501, 0.0)})

        issue = watch_dir_account_issue(watch, current_uid=502, stat=stat)

        assert "belongs to the account 'rt-mgh'" in issue

    def test_root_owned_or_missing_folder_says_nothing(self, tmp_path):
        watch = tmp_path / "BOLD_DICOM"
        watch.mkdir()
        assert watch_dir_account_issue(watch, current_uid=501, stat=self._fake_stat({"BOLD_DICOM": (0, 0.0)})) == ""
        assert watch_dir_account_issue(tmp_path / "missing", current_uid=501, stat=self._fake_stat({})) == ""

    def test_not_checked_on_windows(self, tmp_path, monkeypatch):
        # Windows has no POSIX account ids, and its ownership works differently
        monkeypatch.setattr(sys, "platform", "win32")
        assert watch_dir_account_issue(tmp_path) == ""


class TestPingScanner:
    @patch("pineuro.scanner_setup.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="time=1.5 ms")
        ok, latency, detail = ping_scanner("192.168.2.1")
        assert ok is True
        assert latency == 1.5
        assert "responded" in detail

    @patch("pineuro.scanner_setup.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="Request timed out.")
        ok, latency, detail = ping_scanner("192.168.2.1")
        assert ok is False
        assert latency is None
        assert "did not respond" in detail

    @patch("pineuro.scanner_setup.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ping", timeout=3)
        ok, latency, detail = ping_scanner("192.168.2.1")
        assert ok is False
        assert "timed out" in detail

    def test_empty_ip(self):
        ok, latency, detail = ping_scanner("")
        assert ok is False
        assert "No scanner IP" in detail

    def test_none_ip(self):
        ok, latency, detail = ping_scanner(None)
        assert ok is False

    def test_whitespace_ip(self):
        ok, latency, detail = ping_scanner("   ")
        assert ok is False


class TestConnectionTestResult:
    def test_overall_ok_all_pass(self):
        r = ConnectionTestResult(ping_ok=True, dir_exists=True, dir_writable=True)
        assert r.overall_ok is True

    def test_overall_ok_ping_skipped(self):
        r = ConnectionTestResult(ping_ok=None, dir_exists=True, dir_writable=True)
        assert r.overall_ok is True

    def test_overall_fail_ping_failed(self):
        r = ConnectionTestResult(ping_ok=False, dir_exists=True, dir_writable=True)
        assert r.overall_ok is False

    def test_overall_fail_no_dir(self):
        r = ConnectionTestResult(ping_ok=True, dir_exists=False, dir_writable=False)
        assert r.overall_ok is False

    def test_overall_fail_not_writable(self):
        r = ConnectionTestResult(ping_ok=True, dir_exists=True, dir_writable=False)
        assert r.overall_ok is False

    def test_overall_fail_wrong_account(self):
        r = ConnectionTestResult(ping_ok=True, dir_exists=True, dir_writable=True, account_detail="wrong account")
        assert r.overall_ok is False

    def test_defaults(self):
        r = ConnectionTestResult()
        assert r.ping_ok is None
        assert r.dir_exists is False
        assert r.overall_ok is False


class TestTestScannerConnection:
    def test_with_ip_and_existing_dir(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        with patch("pineuro.scanner_setup.ping_scanner") as mock_ping:
            mock_ping.return_value = (True, 1.0, "responded")
            result = test_scanner_connection("192.168.2.1", d)
        assert result.ping_ok is True
        assert result.dir_exists is True
        assert result.dir_writable is True
        assert result.overall_ok is True

    def test_no_ip(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        result = test_scanner_connection("", d)
        assert result.ping_ok is None
        assert result.ping_detail == "Scanner IP not configured (ping skipped)"
        assert result.dir_exists is True

    def test_none_ip(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir()
        result = test_scanner_connection(None, d)
        assert result.ping_ok is None

    def test_nonexistent_dir(self):
        with patch("pineuro.scanner_setup.ping_scanner") as mock_ping:
            mock_ping.return_value = (True, 1.0, "responded")
            result = test_scanner_connection("192.168.2.1", "/nonexistent/path")
        assert result.ping_ok is True
        assert result.dir_exists is False
        assert result.overall_ok is False


class TestGetSetupInstructions:
    def test_has_all_platforms(self):
        result = get_setup_instructions("/BOLD_DICOM", "192.168.2.1")
        assert "windows" in result
        assert "macos" in result
        assert "linux" in result
        assert "scanner_notes" in result

    def test_linux_has_samba_config(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "[BOLD_DICOM]" in result["linux"]
        assert "smb.conf" in result["linux"]
        assert "smbpasswd" in result["linux"]
        assert "smbclient" in result["linux"]
        assert "smbd nmbd" in result["linux"]
        assert "valid users" in result["linux"]

    def test_windows_has_sharing(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "Share this folder" in result["windows"]
        assert "Advanced Sharing" in result["windows"]

    def test_macos_has_sharing(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "File Sharing" in result["macos"]
        assert "SMB" in result["macos"]
        assert "smbutil" in result["macos"]
        assert "Firewall" in result["macos"]

    def test_scanner_notes_has_dot_addin_fields(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "Auto Export" in result["scanner_notes"]
        assert "Target Host" in result["scanner_notes"]
        assert "Target Directory" in result["scanner_notes"]

    def test_scanner_notes_has_mac_net_use(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "net use" in result["scanner_notes"]

    def test_scanner_notes_mentions_ge_unsupported(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "GE" in result["scanner_notes"]
        assert "not supported" in result["scanner_notes"].lower()

    def test_scanner_notes_mentions_philips(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "Philips" in result["scanner_notes"]

    def test_platforms_have_local_user_step(self):
        result = get_setup_instructions("/BOLD_DICOM")
        assert "local user" in result["windows"].lower()
        assert "local user" in result["macos"].lower()

    def test_custom_watch_dir(self):
        result = get_setup_instructions("/custom/dicom/path")
        assert "/custom/dicom/path" in result["linux"]
        assert "/custom/dicom/path" in result["windows"]

    def test_scanner_ip_in_notes(self):
        result = get_setup_instructions("/BOLD_DICOM", "10.0.0.5")
        assert "10.0.0.5" in result["scanner_notes"]


class TestGetCurrentPlatformKey:
    @patch("pineuro.scanner_setup.sys")
    def test_windows(self, mock_sys):
        mock_sys.platform = "win32"
        assert get_current_platform_key() == "windows"

    @patch("pineuro.scanner_setup.sys")
    def test_macos(self, mock_sys):
        mock_sys.platform = "darwin"
        assert get_current_platform_key() == "macos"

    @patch("pineuro.scanner_setup.sys")
    def test_linux(self, mock_sys):
        mock_sys.platform = "linux"
        assert get_current_platform_key() == "linux"
