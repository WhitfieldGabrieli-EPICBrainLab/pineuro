"""Tests for pineuro.gui.config_manager.ConfigManager."""

import pytest

from pineuro.gui.config_manager import ConfigManager
from pineuro.gui.models import StudyConfig, StudyMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_study(name="Test Study", description="A test study", session_config=None):
    if session_config is None:
        session_config = {"scan": {"tr": 1.5, "n_volumes": 200}}
    return StudyConfig(
        metadata=StudyMetadata(name=name, description=description),
        session_config=session_config,
    )


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(studies_dir=tmp_path / "studies")


# =============================================================================
# _next_unique_name
# =============================================================================


class TestNextUniqueName:
    def test_base_not_in_existing_returns_base(self, manager):
        assert manager._next_unique_name("Alpha", set()) == "Alpha"

    def test_base_in_existing_appends_2(self, manager):
        assert manager._next_unique_name("Alpha", {"Alpha"}) == "Alpha_2"

    def test_skips_taken_suffixes(self, manager):
        existing = {"Alpha", "Alpha_2", "Alpha_3"}
        assert manager._next_unique_name("Alpha", existing) == "Alpha_4"

    def test_large_gap_still_works(self, manager):
        existing = {"Alpha", "Alpha_2"}
        assert manager._next_unique_name("Alpha", existing) == "Alpha_3"

    def test_base_with_space(self, manager):
        result = manager._next_unique_name("My Study (Copy)", {"My Study (Copy)"})
        assert result == "My Study (Copy)_2"


# =============================================================================
# CRUD: save / load / delete / list
# =============================================================================


class TestCRUD:
    def test_save_and_load_roundtrip(self, manager):
        study = make_study()
        manager.save_study(study)
        loaded = manager.load_study("Test Study")
        assert loaded is not None
        assert loaded.metadata.name == "Test Study"
        assert loaded.session_config == study.session_config

    def test_load_nonexistent_returns_none(self, manager):
        assert manager.load_study("Nonexistent Study") is None

    def test_save_overwrites_by_default(self, manager):
        study = make_study()
        manager.save_study(study)
        study.metadata.description = "Updated description"
        manager.save_study(study)
        loaded = manager.load_study("Test Study")
        assert loaded.metadata.description == "Updated description"

    def test_delete_existing_returns_true(self, manager):
        study = make_study()
        manager.save_study(study)
        assert manager.delete_study("Test Study") is True
        assert manager.load_study("Test Study") is None

    def test_delete_nonexistent_returns_false(self, manager):
        assert manager.delete_study("Nonexistent") is False

    def test_list_studies_returns_metadata(self, manager):
        manager.save_study(make_study("Study A"))
        manager.save_study(make_study("Study B"))
        names = {m.name for m in manager.list_studies()}
        assert names == {"Study A", "Study B"}

    def test_list_studies_sorted_newest_first(self, manager):
        import time

        manager.save_study(make_study("Older"))
        time.sleep(0.02)
        manager.save_study(make_study("Newer"))
        names = [m.name for m in manager.list_studies()]
        assert names.index("Newer") < names.index("Older")

    def test_list_studies_empty_dir(self, manager):
        assert manager.list_studies() == []


# =============================================================================
# Name sanitization
# =============================================================================


class TestNameSanitization:
    def test_spaces_converted_to_underscores(self, manager):
        study = make_study("My Study Name")
        path = manager.save_study(study)
        assert path.name == "My_Study_Name.yaml"
        assert path.exists()

    def test_forward_slash_replaced(self, manager):
        study = make_study("Study/Name")
        path = manager.save_study(study)
        assert "/" not in path.name

    def test_roundtrip_preserves_original_name(self, manager):
        study = make_study("My Study Name")
        manager.save_study(study)
        loaded = manager.load_study("My Study Name")
        assert loaded is not None
        assert loaded.metadata.name == "My Study Name"


# =============================================================================
# duplicate_study
# =============================================================================


class TestDuplicateStudy:
    def test_creates_copy_suffix(self, manager):
        manager.save_study(make_study("Original"))
        copy = manager.duplicate_study("Original")
        assert copy is not None
        assert copy.metadata.name == "Original (Copy)"

    def test_nonexistent_source_returns_none(self, manager):
        assert manager.duplicate_study("Ghost") is None

    def test_duplicate_twice_avoids_collision(self, manager):
        manager.save_study(make_study("Original"))
        manager.duplicate_study("Original")  # → "Original (Copy)"
        copy2 = manager.duplicate_study("Original")  # → "Original (Copy)_2"
        assert copy2.metadata.name == "Original (Copy)_2"

    def test_preserves_session_config(self, manager):
        session = {"scan": {"tr": 2.0, "n_volumes": 100}}
        manager.save_study(make_study(session_config=session))
        copy = manager.duplicate_study("Test Study")
        assert copy.session_config == session

    def test_custom_new_name(self, manager):
        manager.save_study(make_study("Original"))
        copy = manager.duplicate_study("Original", new_name="My Clone")
        assert copy.metadata.name == "My Clone"

    def test_duplicate_persisted_on_disk(self, manager):
        manager.save_study(make_study("Original"))
        manager.duplicate_study("Original")
        loaded = manager.load_study("Original (Copy)")
        assert loaded is not None


# =============================================================================
# export_study / import_study
# =============================================================================


class TestExportImport:
    def test_export_creates_file(self, manager, tmp_path):
        manager.save_study(make_study())
        dest = tmp_path / "exported.yaml"
        result = manager.export_study("Test Study", dest)
        assert result is True
        assert dest.exists()

    def test_export_nonexistent_returns_false(self, manager, tmp_path):
        dest = tmp_path / "exported.yaml"
        result = manager.export_study("Ghost", dest)
        assert result is False
        assert not dest.exists()

    def test_import_roundtrip(self, manager, tmp_path):
        session = {"scan": {"tr": 1.2, "n_volumes": 150}}
        manager.save_study(make_study(session_config=session))
        dest = tmp_path / "exported.yaml"
        manager.export_study("Test Study", dest)

        manager2 = ConfigManager(studies_dir=tmp_path / "studies2")
        imported = manager2.import_study(dest)
        assert imported is not None
        assert imported.metadata.name == "Test Study"
        assert imported.session_config == session

    def test_import_persists_to_disk(self, manager, tmp_path):
        manager.save_study(make_study())
        dest = tmp_path / "exported.yaml"
        manager.export_study("Test Study", dest)

        manager2 = ConfigManager(studies_dir=tmp_path / "studies2")
        manager2.import_study(dest)
        assert manager2.load_study("Test Study") is not None

    def test_import_collision_renames(self, manager, tmp_path):
        manager.save_study(make_study())
        dest = tmp_path / "exported.yaml"
        manager.export_study("Test Study", dest)

        # Import into same manager — "Test Study" already exists
        imported = manager.import_study(dest)
        assert imported.metadata.name == "Test Study_2"

    def test_import_collision_both_coexist(self, manager, tmp_path):
        manager.save_study(make_study())
        dest = tmp_path / "exported.yaml"
        manager.export_study("Test Study", dest)

        manager.import_study(dest)
        assert manager.load_study("Test Study") is not None
        assert manager.load_study("Test Study_2") is not None


# =============================================================================
# Version history
# =============================================================================


class TestVersionHistory:
    def test_first_save_creates_no_snapshot(self, manager):
        """First save has nothing to snapshot (no previous file)."""
        study = make_study()
        manager.save_study(study)
        assert manager.list_versions("Test Study") == []

    def test_second_save_creates_snapshot(self, manager):
        """Saving after a change snapshots the previous version."""
        study = make_study()
        manager.save_study(study)
        study.metadata.description = "Updated"
        manager.save_study(study)
        versions = manager.list_versions("Test Study")
        assert len(versions) == 1
        assert versions[0]["version"] == 1

    def test_multiple_saves_increment_version(self, manager):
        """Multiple saves produce incrementing version numbers."""
        study = make_study()
        manager.save_study(study)
        study.metadata.description = "v2"
        manager.save_study(study)
        study.metadata.description = "v3"
        manager.save_study(study)
        versions = manager.list_versions("Test Study")
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2

    def test_restore_creates_new_snapshot(self, manager):
        """Restoring a version snapshots the current state first."""
        study = make_study()
        manager.save_study(study)
        study.metadata.description = "Changed"
        manager.save_study(study)
        versions = manager.list_versions("Test Study")
        assert len(versions) == 1

        manager.restore_version("Test Study", versions[0]["path"])
        assert len(manager.list_versions("Test Study")) == 2

    def test_restore_loads_old_content(self, manager):
        """Restoring actually brings back old content."""
        study = make_study(description="Original desc")
        manager.save_study(study)
        study.metadata.description = "New desc"
        manager.save_study(study)

        versions = manager.list_versions("Test Study")
        manager.restore_version("Test Study", versions[0]["path"])

        loaded = manager.load_study("Test Study")
        assert loaded.metadata.description == "Original desc"

    def test_delete_study_cleans_versions(self, manager):
        """Deleting a study removes its version history."""
        study = make_study()
        manager.save_study(study)
        study.metadata.description = "v2"
        manager.save_study(study)
        assert len(manager.list_versions("Test Study")) == 1

        manager.delete_study("Test Study")
        assert manager.list_versions("Test Study") == []

    def test_list_versions_nonexistent_study(self, manager):
        """Listing versions for a nonexistent study returns empty list."""
        assert manager.list_versions("Ghost") == []

    def test_restore_nonexistent_version_raises(self, manager):
        """Restoring from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            manager.restore_version("Test Study", manager.studies_dir / "fake.yaml")

    def test_version_files_are_valid_yaml(self, manager):
        """Version snapshot files are loadable study configs."""
        study = make_study()
        manager.save_study(study)
        study.metadata.description = "Updated"
        manager.save_study(study)

        versions = manager.list_versions("Test Study")
        loaded = manager.load_study_from_file(versions[0]["path"])
        assert loaded.metadata.name == "Test Study"
        assert loaded.metadata.description == "A test study"

    def test_versions_sorted_oldest_first(self, manager):
        """Versions are returned sorted oldest to newest."""
        study = make_study()
        manager.save_study(study)
        for i in range(3):
            study.metadata.description = f"edit {i}"
            manager.save_study(study)

        versions = manager.list_versions("Test Study")
        assert len(versions) == 3
        assert versions[0]["version"] < versions[1]["version"] < versions[2]["version"]


# =============================================================================
# Rename via save (regression: blank study after Copy + rename + Save)
# =============================================================================


class TestRenameViaSave:
    def test_rename_via_save_moves_file(self, manager):
        study = make_study("Original")
        old_path = manager.save_study(study)
        assert old_path.exists()

        study.metadata.name = "Renamed"
        new_path = manager.save_study(study)

        assert new_path.name == "Renamed.yaml"
        assert new_path.exists()
        assert not old_path.exists()

        loaded = manager.load_study("Renamed")
        assert loaded is not None
        assert loaded.metadata.name == "Renamed"
        assert loaded.session_config == study.session_config

    def test_rename_then_load_by_new_name_returns_full_config(self, manager):
        session = {"scan": {"tr": 1.5, "n_volumes": 200}, "feedback": {"n_baseline": 25}}
        study = make_study("Original (Copy)", session_config=session)
        manager.save_study(study)

        study.metadata.name = "My New Study"
        manager.save_study(study)

        loaded = manager.load_study("My New Study")
        assert loaded is not None
        assert loaded.session_config == session

    def test_rename_collision_rejected_overwrite_true(self, manager):
        manager.save_study(make_study("Foo"))
        bar = make_study("Bar")
        manager.save_study(bar)

        bar.metadata.name = "Foo"
        with pytest.raises(FileExistsError):
            manager.save_study(bar, overwrite=True)

        assert manager.load_study("Foo") is not None
        assert manager.load_study("Bar") is not None

    def test_rename_collision_rejected_overwrite_false(self, manager):
        manager.save_study(make_study("Foo"))
        bar = make_study("Bar")
        manager.save_study(bar)

        bar.metadata.name = "Foo"
        with pytest.raises(FileExistsError):
            manager.save_study(bar, overwrite=False)

        assert manager.load_study("Foo") is not None
        assert manager.load_study("Bar") is not None

    def test_rename_moves_versions_dir(self, manager):
        study = make_study("Original")
        manager.save_study(study)
        study.metadata.description = "edit 1"
        manager.save_study(study)
        assert len(manager.list_versions("Original")) == 1

        old_versions_dir = manager.studies_dir / ".versions" / "Original"
        assert old_versions_dir.exists()

        study.metadata.name = "Renamed"
        manager.save_study(study)

        new_versions_dir = manager.studies_dir / ".versions" / "Renamed"
        assert new_versions_dir.exists()
        assert not old_versions_dir.exists()
        assert len(manager.list_versions("Renamed")) >= 1

    def test_rename_same_name_no_op(self, manager):
        """Saving without renaming hits the fast path and doesn't error."""
        study = make_study("Stable")
        path = manager.save_study(study)
        study.metadata.description = "Updated"
        path2 = manager.save_study(study)
        assert path == path2
        assert manager.load_study("Stable").metadata.description == "Updated"
