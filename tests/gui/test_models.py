"""Tests for pineuro.gui.models — StudyMetadata and StudyConfig."""

import time

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
        d = {"_metadata": {"name": "Alpha"}, "simulator": {"enabled": True}}
        original_keys = set(d.keys())
        StudyConfig.from_dict(d)
        assert set(d.keys()) == original_keys

    def test_from_dict_missing_metadata(self):
        d = {"simulator": {"enabled": True}}
        config = StudyConfig.from_dict(d)
        assert config.metadata.name == ""
        assert config.session_config == {"simulator": {"enabled": True}}

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
        d = {"_metadata": {"name": "X"}, "simulator": {"enabled": False}}
        config = StudyConfig.from_dict(d)
        assert "_metadata" not in config.session_config
        assert config.session_config == {"simulator": {"enabled": False}}

    def test_empty_session_config(self):
        config = StudyConfig(
            metadata=StudyMetadata(name="Empty"),
            session_config={},
        )
        d = config.to_dict()
        assert "_metadata" in d
        assert len([k for k in d if k != "_metadata"]) == 0


# =============================================================================
# Per-session protocol: session_count / build_session_config
# =============================================================================


class TestBuildSessionConfig:
    def _study(self) -> StudyConfig:
        session_config = {
            "paths": {"base_dir": "/data/study", "participant_id": "", "example_participant_id": "sub-001"},
            "reference": {"initial_enabled": True, "realtime_enabled": True},
            "sessions": [
                {
                    "notes": "baseline",
                    "localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA", "ContA"]}],
                    "feedback_runs": [{"task": "ball", "mask_source": "rest"}],
                },
                {
                    "notes": "transfer",
                    "localizers": [
                        {
                            "id": "rest",
                            "type": "rest",
                            "networks": ["DefaultA", "ContA"],
                            "reuse_from_session": "ses-01",
                        }
                    ],
                    "feedback_runs": [{"task": "thermometer", "mask_source": "rest"}],
                },
            ],
        }
        return StudyConfig(metadata=StudyMetadata(name="Multi"), session_config=session_config)

    def test_session_count(self):
        assert self._study().session_count == 2
        assert StudyConfig(metadata=StudyMetadata(name="X"), session_config={}).session_count == 0

    def test_build_lifts_protocol_and_attaches_session_block(self):
        cfg = self._study().build_session_config(1, "sub-07", "ses-02")
        assert cfg["localizers"][0]["reuse_from_session"] == "ses-01"
        assert cfg["feedback_runs"][0]["task"] == "thermometer"
        assert cfg["session"] == {"session_id": "ses-02", "session_number": 2, "notes": "transfer"}
        # Study-wide blocks are carried over.
        assert cfg["reference"]["realtime_enabled"] is True
        # Per-session protocol is not left at the top of the *study* config.
        assert "sessions" not in cfg

    def test_build_rewrites_base_dir_to_participant_session(self):
        cfg = self._study().build_session_config(0, "sub-07", "ses-01")
        assert cfg["paths"]["base_dir"].replace("\\", "/").endswith("study/sub-07/ses-01")

    def test_build_is_idempotent_on_base_dir(self):
        study = self._study()
        once = study.build_session_config(0, "sub-07", "ses-01")["paths"]["base_dir"]
        # The study config itself is not mutated, so a second build matches.
        twice = study.build_session_config(0, "sub-07", "ses-01")["paths"]["base_dir"]
        assert once == twice

    def test_build_does_not_mutate_study(self):
        study = self._study()
        study.build_session_config(0, "sub-07", "ses-01")
        assert study.session_config["paths"]["base_dir"] == "/data/study"
        assert "localizers" not in study.session_config

    def test_build_rejects_out_of_range(self):
        import pytest

        with pytest.raises(IndexError):
            self._study().build_session_config(5, "sub-07", "ses-06")

    def test_build_rejects_no_sessions(self):
        import pytest

        study = StudyConfig(metadata=StudyMetadata(name="X"), session_config={"paths": {}})
        with pytest.raises(ValueError, match="sessions"):
            study.build_session_config(0, "sub-01", "ses-01")


# =============================================================================
# Cross-session mask reuse: synthesis during build_session_config
# =============================================================================


class TestBuildSessionConfigReuse:
    def _study(self) -> StudyConfig:
        session_config = {
            "paths": {"base_dir": "/data/study", "participant_id": ""},
            "reference": {"initial_enabled": True, "realtime_enabled": True},
            "sessions": [
                {
                    "notes": "localizer + ES",
                    "localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b", "Movement": "c>d"}}],
                    "feedback_runs": [
                        {"task": "ritexperiencesampling", "network_assignment": {"target_network": "Trigger"}}
                    ],
                },
                {
                    "notes": "feedback only — reuse ses-01 masks",
                    "localizers": [],
                    "feedback_runs": [
                        {
                            "task": "rittriggeredball",
                            "network_assignment": {"trigger_network": "Trigger", "movement_network": "Movement"},
                            "mask_sources": {"Trigger": "ses-01", "Movement": "ses-01"},
                        }
                    ],
                },
            ],
        }
        return StudyConfig(metadata=StudyMetadata(name="Reuse"), session_config=session_config)

    def test_synthesizes_reused_localizer_for_feedback_only_session(self):
        cfg = self._study().build_session_config(1, "sub-07", "ses-02")
        locs = cfg["localizers"]
        assert len(locs) == 1
        assert locs[0]["reuse_from_session"] == "ses-01"
        assert locs[0]["type"] == "rit"
        assert locs[0]["id"] == "rit"
        assert set(locs[0]["contrasts"]) == {"Trigger", "Movement"}
        # mask_source rewritten to the producing (synthesized) localizer id.
        assert cfg["feedback_runs"][0]["mask_source"] == "rit"

    def test_feedback_only_session_is_all_reused(self):
        cfg = self._study().build_session_config(1, "sub-07", "ses-02")
        assert cfg["localizers"]
        assert all(loc.get("reuse_from_session") for loc in cfg["localizers"])

    def test_no_synthesis_for_fresh_session(self):
        cfg = self._study().build_session_config(0, "sub-07", "ses-01")
        assert len(cfg["localizers"]) == 1
        assert not cfg["localizers"][0].get("reuse_from_session")
        # mask_source rewritten to the fresh producer of the referenced mask.
        assert cfg["feedback_runs"][0]["mask_source"] == "rit"

    def test_tissue_required_stamped_false_without_nuisance(self):
        cfg = self._study().build_session_config(0, "sub-07", "ses-01")
        assert cfg["_tissue_required"] is False

    def test_tissue_required_stamped_cross_session(self):
        # ses-02 (reusing ses-01) needs WM regressors -> ses-01 must extract tissue.
        study = self._study()
        study.session_config["sessions"][1]["feedback_runs"][0]["glm"] = {"include_wm_regressor": True}
        assert study.build_session_config(0, "sub-07", "ses-01")["_tissue_required"] is True
        assert study.build_session_config(1, "sub-07", "ses-02")["_tissue_required"] is True

    def test_two_prior_sources_two_synthesized_entries(self):
        session_config = {
            "paths": {"base_dir": "/data/study", "participant_id": ""},
            "sessions": [
                {"localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}], "feedback_runs": []},
                {"localizers": [{"id": "rest", "type": "rest", "networks": ["Movement"]}], "feedback_runs": []},
                {
                    "localizers": [],
                    "feedback_runs": [
                        {
                            "task": "rittriggeredball",
                            "network_assignment": {"trigger_network": "Trigger", "movement_network": "Movement"},
                            "mask_sources": {"Trigger": "ses-01", "Movement": "ses-02"},
                        }
                    ],
                },
            ],
        }
        study = StudyConfig(metadata=StudyMetadata(name="Two"), session_config=session_config)
        cfg = study.build_session_config(2, "sub-07", "ses-03")
        assert len(cfg["localizers"]) == 2
        assert {loc["reuse_from_session"] for loc in cfg["localizers"]} == {"ses-01", "ses-02"}

    def test_synthesized_config_passes_session_config_validation(self):
        from pineuro.session import SessionConfig

        cfg = self._study().build_session_config(1, "sub-07", "ses-02")
        # mask_source resolves to the synthesized localizer id, so no ValueError.
        sc = SessionConfig._from_dict(cfg)
        assert sc.localizers[0].reuse_from_session == "ses-01"
        assert sc.tissue_required is False
