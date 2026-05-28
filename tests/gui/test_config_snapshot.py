"""Tests for pineuro.gui.config_snapshot — snapshot save/load/compare/archive."""

from pineuro.gui.config_snapshot import (
    SNAPSHOT_FILENAME,
    ConfigDifference,
    archive_config_snapshot,
    compare_protocol_configs,
    format_differences,
    load_config_snapshot,
    save_config_snapshot,
)
from pineuro.gui.models import StudyConfig, StudyMetadata


def _make_study_config(**overrides) -> StudyConfig:
    """Create a minimal StudyConfig (new schema) for testing."""
    session_config = {
        "paths": {"base_dir": "/tmp/data", "participant_id": ""},
        "simulator": {"enabled": True, "collection_speed": 1.0, "feedback_speed": 1.0},
        "scanner": {"dicom_dir": "/BOLD_DICOM", "ip": "192.168.1.1"},
        "reference": {
            "initial_enabled": True,
            "initial_n_volumes": 2,
            "realtime_enabled": True,
            "realtime_n_volumes": 2,
        },
        "sham": {"enabled": False},
        "display": {"control_monitor": "left"},
        "localizers": [
            {
                "id": "rest",
                "type": "rest",
                "tr": 1.2,
                "n_volumes": 150,
                "n_runs": 2,
                "networks": ["ContA", "DefaultA"],
                "extraction": {
                    "n_yeo": 7,
                    "network_atlas": "yeo_7",
                    "n_components": 15,
                    "threshold": 2000,
                },
            }
        ],
        "feedback_runs": [
            {
                "task": "ball",
                "mask_source": "rest",
                "max_runs": 2,
                "tr": 1.0,
                "n_volumes": 200,
                "n_baseline": 25,
                "qc": {"fd_threshold": 0.5},
                "glm": {"motion_correction": "siemens_moco", "smooth_fwhm": 6.0},
            }
        ],
    }
    session_config.update(overrides)
    return StudyConfig(
        metadata=StudyMetadata(name="Test Study"),
        session_config=session_config,
    )


# =============================================================================
# save / load round-trip
# =============================================================================


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        config = _make_study_config()
        save_config_snapshot(config, tmp_path, "sub-001")

        loaded = load_config_snapshot(tmp_path)
        assert loaded is not None
        assert loaded["_metadata"]["name"] == "Test Study"
        assert loaded["localizers"][0]["tr"] == 1.2
        assert loaded["feedback_runs"][0]["n_baseline"] == 25

    def test_snapshot_metadata_present(self, tmp_path):
        config = _make_study_config()
        save_config_snapshot(config, tmp_path, "sub-001", session_id="ses-01")

        loaded = load_config_snapshot(tmp_path)
        snap = loaded["_snapshot"]
        assert snap["participant_id"] == "sub-001"
        assert snap["session_id"] == "ses-01"
        assert "saved_at" in snap
        assert "pineuro_version" in snap

    def test_snapshot_no_session(self, tmp_path):
        config = _make_study_config()
        save_config_snapshot(config, tmp_path, "sub-002")

        loaded = load_config_snapshot(tmp_path)
        assert loaded["_snapshot"]["session_id"] is None

    def test_load_missing_returns_none(self, tmp_path):
        assert load_config_snapshot(tmp_path) is None

    def test_file_is_valid_yaml(self, tmp_path):
        config = _make_study_config()
        save_config_snapshot(config, tmp_path, "sub-001")

        import yaml

        with open(tmp_path / SNAPSHOT_FILENAME) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "_metadata" in data


# =============================================================================
# compare_protocol_configs
# =============================================================================


class TestCompare:
    def test_identical_configs_no_differences(self):
        config = _make_study_config()
        d = config.to_dict()
        assert compare_protocol_configs(d, d) == []

    def test_detects_changed_value(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        current["feedback_runs"][0]["n_baseline"] = 30

        diffs = compare_protocol_configs(saved, current)
        paths = [d.field_path for d in diffs]
        assert "feedback_runs.0.n_baseline" in paths

        diff = next(d for d in diffs if d.field_path == "feedback_runs.0.n_baseline")
        assert diff.saved_value == 25
        assert diff.current_value == 30

    def test_ignores_path_changes(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        current["paths"]["base_dir"] = "/completely/different/path"

        assert compare_protocol_configs(saved, current) == []

    def test_ignores_metadata_changes(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        current["_metadata"]["modified"] = "2099-01-01T00:00:00"

        assert compare_protocol_configs(saved, current) == []

    def test_ignores_simulator_keys(self):
        """simulator block is not in PROTOCOL_KEYS — environment-specific, not protocol."""
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        current["simulator"]["enabled"] = False
        current["scanner"]["ip"] = "10.0.0.1"

        assert compare_protocol_configs(saved, current) == []

    def test_detects_localizer_protocol_change(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        current["localizers"][0]["tr"] = 2.0

        diffs = compare_protocol_configs(saved, current)
        assert len(diffs) == 1
        assert diffs[0].field_path == "localizers.0.tr"
        assert diffs[0].saved_value == 1.2
        assert diffs[0].current_value == 2.0

    def test_detects_added_field(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        current["feedback_runs"][0]["new_param"] = 42

        diffs = compare_protocol_configs(saved, current)
        paths = [d.field_path for d in diffs]
        assert "feedback_runs.0.new_param" in paths

    def test_detects_removed_field(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        del current["feedback_runs"][0]["glm"]["smooth_fwhm"]

        diffs = compare_protocol_configs(saved, current)
        paths = [d.field_path for d in diffs]
        assert "feedback_runs.0.glm.smooth_fwhm" in paths

    def test_handles_missing_section(self):
        saved = _make_study_config().to_dict()
        current = _make_study_config().to_dict()
        del current["sham"]

        diffs = compare_protocol_configs(saved, current)
        paths = [d.field_path for d in diffs]
        assert "sham.enabled" in paths

    def test_list_order_matters(self):
        saved = _make_study_config().to_dict()
        saved["localizers"][0]["networks"] = ["ContA", "DefaultA"]
        current = _make_study_config().to_dict()
        current["localizers"][0]["networks"] = ["DefaultA", "ContA"]

        diffs = compare_protocol_configs(saved, current)
        paths = [d.field_path for d in diffs]
        assert "localizers.0.networks" in paths

    def test_nested_dict_change(self):
        saved = _make_study_config().to_dict()
        saved["feedback_runs"][0]["task_params"] = {"speed": 1.0, "size": 50}
        current = _make_study_config().to_dict()
        current["feedback_runs"][0]["task_params"] = {"speed": 2.0, "size": 50}

        diffs = compare_protocol_configs(saved, current)
        paths = [d.field_path for d in diffs]
        assert "feedback_runs.0.task_params.speed" in paths


# =============================================================================
# archive
# =============================================================================


class TestArchive:
    def test_archive_renames_file(self, tmp_path):
        config = _make_study_config()
        save_config_snapshot(config, tmp_path, "sub-001")

        archived = archive_config_snapshot(tmp_path)
        assert archived is not None
        assert archived.exists()
        assert not (tmp_path / SNAPSHOT_FILENAME).exists()
        assert archived.name.startswith("study_config.")
        assert archived.suffix == ".yaml"

    def test_archive_missing_returns_none(self, tmp_path):
        assert archive_config_snapshot(tmp_path) is None

    def test_archive_preserves_content(self, tmp_path):
        config = _make_study_config()
        save_config_snapshot(config, tmp_path, "sub-001")

        original = load_config_snapshot(tmp_path)
        archived_path = archive_config_snapshot(tmp_path)

        import yaml

        with open(archived_path) as f:
            archived = yaml.safe_load(f)

        assert original == archived


# =============================================================================
# format_differences
# =============================================================================


class TestFormat:
    def test_empty_list(self):
        assert format_differences([]) == "No differences."

    def test_single_difference(self):
        diffs = [ConfigDifference("feedback_runs.0.n_baseline", 25, 30)]
        text = format_differences(diffs)
        assert "feedback_runs.0.n_baseline" in text
        assert "25" in text
        assert "30" in text


# =============================================================================
# ConfigDifference
# =============================================================================


class TestConfigDifference:
    def test_str(self):
        d = ConfigDifference("localizers.0.extraction.threshold", 2000, 3000)
        assert "localizers.0.extraction.threshold" in str(d)
        assert "2000" in str(d)
        assert "3000" in str(d)
