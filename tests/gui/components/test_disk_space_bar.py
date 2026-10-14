"""Tests for pineuro.gui.components.disk_space_bar (the CONN-style free-space readout)."""

import time
from pathlib import Path
from unittest.mock import MagicMock

from pineuro.gui.components import disk_space_bar as dsb
from pineuro.storage import DiskSpace


def _fake_disk_space(table: dict[Path, DiskSpace]):
    return lambda path: table.get(Path(path))


class TestSummarize:
    def test_reports_each_volume_once(self, monkeypatch):
        root = Path("/data/study")
        dicom_dir = Path("/data/BOLD_DICOM")  # same volume as the data root
        table = {
            root: DiskSpace(path=root, free=212_400_000_000, total=494_400_000_000, volume=1),
            dicom_dir: DiskSpace(path=dicom_dir, free=212_400_000_000, total=494_400_000_000, volume=1),
        }
        monkeypatch.setattr(dsb, "disk_space", _fake_disk_space(table))

        text, warning = dsb.summarize([root, dicom_dir])

        assert text.count("free of") == 1
        assert "212.4 GB free of 494.4 GB" in text
        assert warning is False

    def test_warns_below_one_sessions_projected_need(self, monkeypatch):
        root = Path("/data/study")
        table = {root: DiskSpace(path=root, free=3_200_000_000, total=494_400_000_000, volume=1)}
        monkeypatch.setattr(dsb, "disk_space", _fake_disk_space(table))

        text, warning = dsb.summarize([root], projected=6_100_000_000)

        assert warning is True
        assert "largest previous session" in text

    def test_unavailable_when_nothing_can_be_measured(self, monkeypatch):
        monkeypatch.setattr(dsb, "disk_space", lambda path: None)
        assert dsb.summarize([Path("/nowhere")]) == ("Free disk space: unavailable", False)


class TestDiskSpaceBar:
    def test_watch_publishes_to_the_page(self, tmp_path):
        page = MagicMock()
        bar = dsb.DiskSpaceBar(page)
        bar.start()
        try:
            bar.watch([tmp_path])
            deadline = time.time() + 5.0
            while time.time() < deadline and not page.run_task.called:
                time.sleep(0.01)
            assert page.run_task.called
        finally:
            bar.stop()
