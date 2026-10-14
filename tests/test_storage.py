"""Tests for pineuro.storage: the disk-space checks behind the GUI's free-space bar."""

from pathlib import Path

from pineuro.storage import (
    DiskSpace,
    directory_size,
    disk_space,
    existing_parent,
    format_bytes,
    largest_session_size,
    low_space_reason,
)


def _write(path: Path, n_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * n_bytes)


class TestExistingParent:
    def test_existing_path_is_returned(self, tmp_path):
        assert existing_parent(tmp_path) == tmp_path

    def test_missing_path_resolves_to_nearest_ancestor(self, tmp_path):
        assert existing_parent(tmp_path / "new_study" / "sub-01") == tmp_path


class TestDiskSpace:
    def test_measures_a_not_yet_created_data_root_at_its_ancestor(self, tmp_path):
        space = disk_space(tmp_path / "not" / "created")
        assert space is not None
        assert space.path == tmp_path
        assert 0 <= space.free <= space.total


class TestFormatBytes:
    def test_decimal_units(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(999) == "999 B"
        assert format_bytes(1500) == "1.5 KB"
        assert format_bytes(212_400_000_000) == "212.4 GB"
        assert format_bytes(1_400_000_000_000) == "1.4 TB"


class TestSessionSizes:
    def test_directory_size_sums_nested_files(self, tmp_path):
        _write(tmp_path / "a.dcm", 100)
        _write(tmp_path / "collection" / "run-01" / "b.dcm", 250)
        assert directory_size(tmp_path) == 350

    def test_largest_session_excludes_the_current_one(self, tmp_path):
        _write(tmp_path / "sub-01" / "ses-01" / "x.dcm", 300)
        _write(tmp_path / "sub-02" / "ses-01" / "x.dcm", 500)
        current = tmp_path / "sub-03" / "ses-01"
        _write(current / "x.dcm", 9_000)
        assert largest_session_size(tmp_path, exclude=current) == 500

    def test_non_session_folders_in_the_study_root_are_ignored(self, tmp_path):
        # e.g. simulator DICOMs or a keywords folder kept next to the participants
        _write(tmp_path / "test_dicoms" / "x.dcm", 9_000)
        _write(tmp_path / "RIT_Localizer" / "Keywords" / "k.json", 9_000)
        assert largest_session_size(tmp_path) is None

    def test_no_previous_session(self, tmp_path):
        current = tmp_path / "sub-01" / "ses-01"
        _write(current / "x.dcm", 100)
        assert largest_session_size(tmp_path, exclude=current) is None


class TestLowSpaceReason:
    @staticmethod
    def _space(free: int) -> DiskSpace:
        return DiskSpace(path=Path("/data"), free=free, total=10 * free + 1, volume=1)

    def test_flags_less_free_space_than_one_session(self):
        reason = low_space_reason(self._space(free=1_000_000_000), projected=6_100_000_000)
        assert reason is not None
        assert "6.1 GB" in reason

    def test_silent_with_enough_space_or_no_projection(self):
        assert low_space_reason(self._space(free=7_000_000_000), projected=6_100_000_000) is None
        assert low_space_reason(self._space(free=1), projected=None) is None
