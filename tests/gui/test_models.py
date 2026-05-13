"""Tests for pineuro.gui.models — StudyMetadata and StudyConfig."""

import time

import pytest

from pineuro.gui.models import StudyConfig, StudyMetadata

# =============================================================================
# StudyMetadata
# =============================================================================


class TestStudyMetadata:
    def test_roundtrip(self):
        meta = StudyMetadata(name="Alpha", description="Desc")
        d = meta.to_dict()
        restored = StudyMetadata.from_dict(d)
        assert restored.name == "Alpha"
        assert restored.description == "Desc"
        assert restored.created == meta.created
        assert restored.modified == meta.modified

    def test_from_dict_missing_keys_use_defaults(self):
        meta = StudyMetadata.from_dict({})
        assert meta.name == ""
        assert meta.description == ""
        assert meta.created  # non-empty timestamp
        assert meta.modified  # non-empty timestamp

    def test_update_modified_changes_timestamp(self):
        meta = StudyMetadata(name="X")
        old = meta.modified
        time.sleep(0.02)
        meta.update_modified()
        assert meta.modified != old

    def test_update_modified_does_not_change_created(self):
        meta = StudyMetadata(name="X")
        created = meta.created
        meta.update_modified()
        assert meta.created == created

    def test_to_dict_contains_all_keys(self):
        meta = StudyMetadata(name="Study", description="About")
        d = meta.to_dict()
        assert set(d.keys()) == {"name", "description", "created", "modified"}

    def test_from_dict_partial_keys(self):
        meta = StudyMetadata.from_dict({"name": "Partial"})
        assert meta.name == "Partial"
        assert meta.description == ""


# =============================================================================
# StudyConfig
# =============================================================================


class TestStudyConfig:
    def _make_config(self, name="Test", session_config=None):
        if session_config is None:
            session_config = {
                "paths": {"base_dir": "/data", "participant_id": ""},
                "study": {"multi_visit_enabled": False},
                "localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA", "ContA"]}],
                "feedback_runs": [{"task": "ball", "mask_source": "rest"}],
            }
        return StudyConfig(
            metadata=StudyMetadata(name=name),
            session_config=session_config,
        )

    def test_to_dict_contains_metadata_key(self):
        config = self._make_config()
        d = config.to_dict()
        assert "_metadata" in d
        assert d["_metadata"]["name"] == "Test"

    def test_to_dict_contains_session_config_keys(self):
        config = self._make_config()
        d = config.to_dict()
        assert "localizers" in d
        assert d["localizers"][0]["type"] == "rest"

    def test_from_dict_roundtrip(self):
        config = self._make_config()
        d = config.to_dict()
        restored = StudyConfig.from_dict(d)
        assert restored.metadata.name == "Test"
        assert restored.session_config == config.session_config

    def test_from_dict_does_not_mutate_input(self):
        """from_dict must not pop/delete the _metadata key from the caller's dict."""
        d = {"_metadata": {"name": "Alpha"}, "study": {"multi_visit_enabled": False}}
        original_keys = set(d.keys())
        StudyConfig.from_dict(d)
        assert set(d.keys()) == original_keys

    def test_from_dict_missing_metadata(self):
        d = {"study": {"multi_visit_enabled": True}}
        config = StudyConfig.from_dict(d)
        assert config.metadata.name == ""
        assert config.session_config == {"study": {"multi_visit_enabled": True}}

    def test_from_dict_rejects_legacy_top_level_keys(self):
        for legacy_key in ("mask", "feedback", "task", "localizer_task", "scan"):
            with pytest.raises(ValueError, match="outdated schema"):
                StudyConfig.from_dict({"_metadata": {"name": "X"}, legacy_key: {}})

    def test_from_dict_file_path_kwarg(self, tmp_path):
        config = self._make_config()
        d = config.to_dict()
        p = tmp_path / "study.yaml"
        restored = StudyConfig.from_dict(d, file_path=p)
        assert restored.file_path == p

    def test_file_path_not_serialized(self, tmp_path):
        """file_path is runtime state and must not appear in to_dict output."""
        config = self._make_config()
        config.file_path = tmp_path / "study.yaml"
        d = config.to_dict()
        assert "file_path" not in d

    def test_session_config_excludes_metadata_key(self):
        d = {"_metadata": {"name": "X"}, "study": {"multi_visit_enabled": False}}
        config = StudyConfig.from_dict(d)
        assert "_metadata" not in config.session_config
        assert config.session_config == {"study": {"multi_visit_enabled": False}}

    def test_empty_session_config(self):
        config = StudyConfig(
            metadata=StudyMetadata(name="Empty"),
            session_config={},
        )
        d = config.to_dict()
        assert "_metadata" in d
        assert len([k for k in d if k != "_metadata"]) == 0
