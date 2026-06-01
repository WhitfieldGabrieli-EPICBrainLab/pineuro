"""Tests for pineuro.gui.utils validation functions and native pickers."""

import subprocess
import sys
import types

import pytest

from pineuro.gui.utils import (
    ValidationError,
    build_feedback_steps,
    feedback_step_for,
    frc_needs_tissue,
    frc_referenced_masks,
    has_fresh_localizer,
    localizer_is_reused,
    localizer_mask_names,
    localizer_producing,
    pick_directory,
    pick_file,
    pick_save_file,
    resolve_use_transfer_runs,
    reused_localizer_indices,
    ses_num,
    session_reuse_sources,
    sham_hemisphere_network_for_localizer,
    sham_networks_for_localizer,
    study_sham_construction,
    synthesize_reused_localizers,
    tissue_required_in_session,
    validate_participant_id,
    validate_positive_int,
    validate_positive_number,
    validate_session_config,
)

# =============================================================================
# Per-localizer reuse helpers
# =============================================================================


class TestLocalizerReuseHelpers:
    def test_mask_names_rest(self):
        assert localizer_mask_names({"type": "rest", "networks": ["DefaultA", "ContA"]}) == ["DefaultA", "ContA"]

    def test_mask_names_task_with_reverse(self):
        loc = {"type": "rit", "contrasts": {"self": "self>other"}, "include_reverse_contrasts": True}
        assert localizer_mask_names(loc) == ["self", "self_reverse"]

    def test_mask_names_manual(self):
        assert localizer_mask_names({"type": "manual", "manual_mask_names": ["L_amyg", "R_amyg"]}) == [
            "L_amyg",
            "R_amyg",
        ]

    def test_is_reused(self):
        assert localizer_is_reused({"reuse_from_session": "ses-01"}) is True
        assert localizer_is_reused({}) is False
        assert localizer_is_reused({"reuse_from_session": None}) is False

    def test_reused_indices_and_fresh(self):
        config = {
            "localizers": [
                {"type": "rest", "reuse_from_session": "ses-01"},
                {"type": "rit", "contrasts": {"x": "x>0"}},
            ]
        }
        assert reused_localizer_indices(config) == [0]
        assert has_fresh_localizer(config) is True

    def test_all_reused(self):
        config = {"localizers": [{"type": "rest", "reuse_from_session": "ses-01"}]}
        assert reused_localizer_indices(config) == [0]
        assert has_fresh_localizer(config) is False


# =============================================================================
# Cross-session mask reuse helpers (synthesis, source resolution, tissue)
# =============================================================================


class TestCrossSessionReuseHelpers:
    def test_ses_num(self):
        assert ses_num("ses-01") == 1
        assert ses_num("ses-12") == 12
        assert ses_num("session-1") is None
        assert ses_num("") is None

    def test_frc_referenced_masks_activation(self):
        frc = {"feedback_mode": "activation", "network_assignment": {"top": "DefaultA", "bottom": "ContA", "x": ""}}
        assert frc_referenced_masks(frc) == {"DefaultA", "ContA"}

    def test_frc_referenced_masks_connectivity(self):
        frc = {"feedback_mode": "connectivity", "connectivity": {"pairs": [["A", "B"], ["C", ""]]}}
        assert frc_referenced_masks(frc) == {"A", "B", "C"}

    def test_frc_needs_tissue(self):
        assert frc_needs_tissue({"glm": {"include_wm_regressor": True}}) is True
        assert frc_needs_tissue({"glm": {"include_csf_regressor": True}}) is True
        assert frc_needs_tissue({"glm": {}}) is False
        assert frc_needs_tissue({}) is False

    def test_localizer_producing_skips_reused(self):
        sess = {
            "localizers": [
                {"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}},
                {"id": "rest", "type": "rest", "networks": ["DefaultA"], "reuse_from_session": "ses-01"},
            ]
        }
        assert localizer_producing(sess, "Trigger")["id"] == "rit"
        # DefaultA only appears on a *reused* localizer -> not a fresh producer here.
        assert localizer_producing(sess, "DefaultA") is None

    def test_synthesize_rest_source(self):
        sessions = [
            {"localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA", "ContA"]}], "feedback_runs": []},
            {
                "localizers": [],
                "feedback_runs": [
                    {
                        "network_assignment": {"top": "DefaultA", "bottom": "ContA"},
                        "mask_sources": {"DefaultA": "ses-01", "ContA": "ses-01"},
                    }
                ],
            },
        ]
        out = synthesize_reused_localizers(sessions, sessions[1]["feedback_runs"])
        assert len(out) == 1
        entry = out[0]
        assert entry["reuse_from_session"] == "ses-01"
        assert entry["type"] == "rest"
        assert entry["id"] == "rest"
        assert sorted(entry["networks"]) == ["ContA", "DefaultA"]

    def test_synthesize_task_source_restricts_contrasts(self):
        sessions = [
            {
                "localizers": [
                    {"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b", "Movement": "c>d", "Other": "e>f"}}
                ],
                "feedback_runs": [],
            },
            {
                "localizers": [],
                "feedback_runs": [
                    {
                        "network_assignment": {"trigger_network": "Trigger", "movement_network": "Movement"},
                        "mask_sources": {"Trigger": "ses-01", "Movement": "ses-01"},
                    }
                ],
            },
        ]
        out = synthesize_reused_localizers(sessions, sessions[1]["feedback_runs"])
        assert len(out) == 1
        assert out[0]["type"] == "rit"
        assert set(out[0]["contrasts"]) == {"Trigger", "Movement"}  # "Other" excluded

    def test_synthesize_two_sources_two_entries(self):
        sessions = [
            {"localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}], "feedback_runs": []},
            {"localizers": [{"id": "rest", "type": "rest", "networks": ["Movement"]}], "feedback_runs": []},
            {
                "localizers": [],
                "feedback_runs": [
                    {
                        "network_assignment": {"t": "Trigger", "m": "Movement"},
                        "mask_sources": {"Trigger": "ses-01", "Movement": "ses-02"},
                    }
                ],
            },
        ]
        out = synthesize_reused_localizers(sessions, sessions[2]["feedback_runs"])
        assert len(out) == 2
        assert {e["reuse_from_session"] for e in out} == {"ses-01", "ses-02"}

    def test_synthesize_reuse_chain_resolves_to_producer(self):
        # ses-02 itself reuses ses-01's Trigger (feedback-only). ses-03 also sources
        # Trigger; the producer is ses-01 (the fresh producer), never ses-02.
        sessions = [
            {"localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}], "feedback_runs": []},
            {
                "localizers": [],
                "feedback_runs": [{"network_assignment": {"t": "Trigger"}, "mask_sources": {"Trigger": "ses-01"}}],
            },
            {
                "localizers": [],
                "feedback_runs": [{"network_assignment": {"t": "Trigger"}, "mask_sources": {"Trigger": "ses-01"}}],
            },
        ]
        out = synthesize_reused_localizers(sessions, sessions[2]["feedback_runs"])
        assert len(out) == 1
        assert out[0]["reuse_from_session"] == "ses-01"

    def test_synthesize_no_sources(self):
        sessions = [
            {
                "localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA"]}],
                "feedback_runs": [{"network_assignment": {"t": "DefaultA"}}],
            }
        ]
        assert synthesize_reused_localizers(sessions, sessions[0]["feedback_runs"]) == []

    def test_session_reuse_sources(self):
        sessions = [
            {"localizers": [], "feedback_runs": []},
            {"localizers": [], "feedback_runs": [{"mask_sources": {"X": "ses-01", "Y": "ses-01"}}]},
        ]
        assert session_reuse_sources(sessions, 1) == {"ses-01"}
        assert session_reuse_sources(sessions, 0) == set()

    def test_tissue_required_own_feedback(self):
        sessions = [
            {
                "localizers": [{"type": "rest", "networks": ["X"]}],
                "feedback_runs": [{"glm": {"include_wm_regressor": True}}],
            }
        ]
        assert tissue_required_in_session(sessions, 0) is True

    def test_tissue_required_later_reusing_session(self):
        sessions = [
            {
                "localizers": [{"id": "rest", "type": "rest", "networks": ["X"]}],
                "feedback_runs": [{"network_assignment": {"t": "X"}}],
            },
            {
                "localizers": [],
                "feedback_runs": [
                    {
                        "network_assignment": {"t": "X"},
                        "mask_sources": {"X": "ses-01"},
                        "glm": {"include_csf_regressor": True},
                    }
                ],
            },
        ]
        # ses-01's own feedback needs no tissue, but ses-02 (reusing it) does.
        assert tissue_required_in_session(sessions, 0) is True
        assert tissue_required_in_session(sessions, 1) is True

    def test_tissue_not_required(self):
        sessions = [{"localizers": [{"type": "rest", "networks": ["X"]}], "feedback_runs": [{"glm": {}}]}]
        assert tissue_required_in_session(sessions, 0) is False


# =============================================================================
# validate_participant_id
# =============================================================================


class TestValidateParticipantId:
    @pytest.mark.parametrize("value", ["sub01", "sub-01", "001", "A", "P001"])
    def test_valid_ids_accepted(self, value):
        ok, msg = validate_participant_id(value)
        assert ok
        assert msg is None  # valid ids carry no error message

    @pytest.mark.parametrize(
        ("value", "label"),
        [
            ("", "empty"),
            ("-sub01", "leading hyphen"),
            ("my id", "space"),
            ("sub01!", "exclamation"),
            ("sub_01", "underscore"),
            ("sub.01", "dot"),
            ("sub@01", "at-symbol"),
        ],
    )
    def test_invalid_ids_rejected_with_message(self, value, label):
        ok, msg = validate_participant_id(value)
        assert not ok, f"{label} id {value!r} should be rejected"
        assert isinstance(msg, str) and msg  # a non-empty explanation


# =============================================================================
# validate_positive_number
# =============================================================================


class TestValidatePositiveNumber:
    def test_valid_float_string(self):
        assert validate_positive_number("1.5", "TR") == pytest.approx(1.5)

    def test_valid_integer_string(self):
        assert validate_positive_number("2", "TR") == pytest.approx(2.0)

    def test_valid_float_value(self):
        assert validate_positive_number(1.2, "TR") == pytest.approx(1.2)

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_number("0", "TR")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_number("-1.5", "TR")

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError, match="number"):
            validate_positive_number("abc", "TR")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_positive_number(None, "TR")

    def test_error_includes_name(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_number("abc", "MyField")
        assert "MyField" in str(exc_info.value)


# =============================================================================
# validate_positive_int
# =============================================================================


class TestValidatePositiveInt:
    def test_valid_int_string(self):
        assert validate_positive_int("5", "n_volumes") == 5

    def test_valid_int_value(self):
        assert validate_positive_int(10, "n_volumes") == 10

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_int("0", "n_volumes")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_int("-3", "n_volumes")

    def test_non_integer_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            validate_positive_int("abc", "n_volumes")

    def test_float_string_truncated(self):
        # int("1.5") raises ValueError → ValidationError
        with pytest.raises(ValidationError):
            validate_positive_int("1.5", "n_volumes")


# =============================================================================
# validate_session_config
# =============================================================================


class TestValidateSessionConfig:
    def _valid_config(self):
        return {
            "localizers": [
                {
                    "id": "rest",
                    "type": "rest",
                    "tr": 1.5,
                    "n_volumes": 200,
                    "n_runs": 2,
                    "networks": ["Default", "Cont"],
                    "extraction": {
                        "network_atlas": "yeo_7",
                        "smooth_fwhm": 5.0,
                        "highpass": 0.01,
                    },
                }
            ],
            "feedback_runs": [
                {
                    "task": "ball",
                    "mask_source": "rest",
                    "n_baseline": 10,
                    "n_volumes": 200,
                }
            ],
        }

    def test_valid_config_passes(self):
        ok, err = validate_session_config(self._valid_config())
        assert ok is True
        assert err is None

    def test_invalid_tr_fails(self):
        config = self._valid_config()
        config["localizers"][0]["tr"] = 0
        ok, err = validate_session_config(config)
        assert ok is False
        assert err is not None

    def test_negative_tr_fails(self):
        config = self._valid_config()
        config["localizers"][0]["tr"] = -1.0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_n_volumes_fails(self):
        config = self._valid_config()
        config["localizers"][0]["n_volumes"] = -1
        ok, err = validate_session_config(config)
        assert ok is False

    def test_zero_n_runs_fails(self):
        config = self._valid_config()
        config["localizers"][0]["n_runs"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_single_run_passes(self):
        config = self._valid_config()
        config["localizers"][0]["n_runs"] = 1
        ok, err = validate_session_config(config)
        assert ok is True
        assert err is None

    def test_invalid_network_atlas_fails(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["network_atlas"] = "yeo_99"  # not a known atlas
        ok, err = validate_session_config(config)
        assert ok is False

    def test_valid_network_atlas_17(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["network_atlas"] = "yeo_17"
        ok, err = validate_session_config(config)
        assert ok is True

    def test_empty_networks_fails(self):
        config = self._valid_config()
        config["localizers"][0]["networks"] = []
        ok, err = validate_session_config(config)
        assert ok is False

    def test_single_network_fails(self):
        config = self._valid_config()
        config["localizers"][0]["networks"] = ["Default"]
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_smooth_fwhm_fails(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["smooth_fwhm"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_highpass_fails(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["highpass"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_feedback_n_baseline_fails(self):
        config = self._valid_config()
        config["feedback_runs"][0]["n_baseline"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_missing_feedback_runs_fails(self):
        config = self._valid_config()
        config["feedback_runs"] = []
        ok, err = validate_session_config(config)
        assert ok is False

    def test_error_message_is_string(self):
        config = self._valid_config()
        config["localizers"][0]["tr"] = 0
        ok, err = validate_session_config(config)
        assert isinstance(err, str)
        assert len(err) > 0


# =============================================================================
# resolve_use_transfer_runs
# =============================================================================


class TestResolveUseTransferRuns:
    def test_explicit_value_wins(self):
        # An explicitly-set value is honored regardless of the task's metadata default.
        assert resolve_use_transfer_runs({"task": "gauge", "use_transfer_runs": True}) is True
        assert resolve_use_transfer_runs({"task": "ball", "use_transfer_runs": False}) is False

    def test_ball_defaults_true(self):
        # ball declares a use_transfer_runs TaskParameter default of True.
        assert resolve_use_transfer_runs({"task": "ball"}) is True

    def test_gauge_and_thermometer_default_false(self):
        assert resolve_use_transfer_runs({"task": "gauge"}) is False
        assert resolve_use_transfer_runs({"task": "thermometer"}) is False

    def test_unknown_or_undeclared_task_defaults_false(self):
        assert resolve_use_transfer_runs({"task": "not-a-real-task"}) is False
        assert resolve_use_transfer_runs({}) is False


# =============================================================================
# Sham construction helpers
# =============================================================================


class TestShamNetworksForLocalizer:
    def _loc(self):
        return {"id": "rest", "type": "rest", "networks": ["DefaultA", "ContA"]}

    def test_disabled_returns_empty(self):
        config = {"sham": {"enabled": False}, "feedback_runs": [{"mask_source": "rest", "sham": {}}]}
        assert sham_networks_for_localizer(config, self._loc()) == []

    def test_hemisphere_split_adds_network(self):
        config = {
            "sham": {"enabled": True},
            "feedback_runs": [
                {"mask_source": "rest", "sham": {"sham_mask_mode": "hemisphere_split", "hemisphere_network": "SomMotA"}}
            ],
        }
        assert sham_networks_for_localizer(config, self._loc()) == ["SomMotA"]
        assert sham_hemisphere_network_for_localizer(config, self._loc()) == "SomMotA"

    def test_two_networks_unions_targets(self):
        config = {
            "sham": {"enabled": True},
            "feedback_runs": [
                {
                    "mask_source": "rest",
                    "sham": {"sham_mask_mode": "two_networks", "alternative_mask_targets": ["VisA", "VisB"]},
                }
            ],
        }
        assert sham_networks_for_localizer(config, self._loc()) == ["VisA", "VisB"]
        # two_networks mode splits no single network into hemispheres.
        assert sham_hemisphere_network_for_localizer(config, self._loc()) is None

    def test_only_matches_referencing_runs(self):
        config = {
            "sham": {"enabled": True},
            "feedback_runs": [
                {"mask_source": "other", "sham": {"sham_mask_mode": "hemisphere_split", "hemisphere_network": "X"}}
            ],
        }
        assert sham_networks_for_localizer(config, self._loc()) == []


class TestStudyShamConstruction:
    def test_empty_when_no_runs(self):
        assert study_sham_construction({"feedback_runs": []}) == {}

    def test_returns_first_alternative_mask_run(self):
        config = {
            "feedback_runs": [
                {"sham": {"delivery_method": "replay"}},
                {"sham": {"delivery_method": "alternative_masks", "hemisphere_network": "SomMotA"}},
            ]
        }
        assert study_sham_construction(config)["hemisphere_network"] == "SomMotA"


# =============================================================================
# Native file/folder pickers: pick_file / pick_directory / pick_save_file
# =============================================================================


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["osascript"], returncode=returncode, stdout=stdout, stderr=stderr)


def _force_macos(monkeypatch, *, osascript="/usr/bin/osascript"):
    """Pretend we are on macOS with (or without) osascript available."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("shutil.which", lambda name: osascript)


def _install_stub_tkinter(monkeypatch):
    """Make ``import tkinter`` succeed without a real Tk install (headless CI)."""
    tk_mod = types.ModuleType("tkinter")

    class _Root:
        def withdraw(self):
            pass

        def attributes(self, *a, **k):
            pass

        def destroy(self):
            pass

    tk_mod.Tk = _Root
    fd_mod = types.ModuleType("tkinter.filedialog")
    fd_mod.askopenfilename = lambda **k: ""
    fd_mod.askdirectory = lambda **k: ""
    fd_mod.asksaveasfilename = lambda **k: ""
    tk_mod.filedialog = fd_mod
    monkeypatch.setitem(sys.modules, "tkinter", tk_mod)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", fd_mod)


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeDispatcher:
    """Returns a canned value without ever running the dialog callback."""

    def __init__(self, value):
        self._value = value

    def submit(self, fn, *args, **kwargs):
        return _FakeFuture(self._value)


class TestPickFileMacOS:
    def test_returns_selected_path(self, monkeypatch):
        _force_macos(monkeypatch)
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _completed(stdout="/Users/x/file.nii\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = pick_file(title="Pick a file")
        assert result == "/Users/x/file.nii"
        # osascript was invoked with the title passed as an argv argument
        assert captured["cmd"][0] == "/usr/bin/osascript"
        assert "Pick a file" in captured["cmd"]

    def test_cancel_returns_none(self, monkeypatch):
        _force_macos(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(returncode=1, stderr="User canceled. (-128)"))
        assert pick_file(title="Pick") is None

    def test_missing_osascript_returns_none(self, monkeypatch):
        _force_macos(monkeypatch, osascript=None)

        def boom(*a, **k):
            raise AssertionError("subprocess.run must not run without osascript")

        monkeypatch.setattr(subprocess, "run", boom)
        assert pick_file(title="Pick") is None


class TestPickDirectoryMacOS:
    def test_returns_selected_dir(self, monkeypatch):
        _force_macos(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="/Users/x/data/\n"))
        assert pick_directory(title="Pick a folder") == "/Users/x/data"


class TestPickSaveFileMacOS:
    def test_returns_selected_path(self, monkeypatch):
        _force_macos(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="/Users/x/out.tsv\n"))
        assert pick_save_file(title="Save") == "/Users/x/out.tsv"

    def test_appends_default_extension_when_missing(self, monkeypatch):
        _force_macos(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="/Users/x/rand\n"))
        assert pick_save_file(title="Save", default_extension=".tsv") == "/Users/x/rand.tsv"

    def test_keeps_existing_extension(self, monkeypatch):
        _force_macos(monkeypatch)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="/Users/x/rand.csv\n"))
        assert pick_save_file(title="Save", default_extension=".tsv") == "/Users/x/rand.csv"

    def test_default_name_passed_as_argv(self, monkeypatch):
        _force_macos(monkeypatch)
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _completed(stdout="/Users/x/out.tsv\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        pick_save_file(title="Save", default_name="study.yaml")
        assert "study.yaml" in captured["cmd"]


class TestPickersNonMacOS:
    def test_pick_file_uses_dispatcher_not_osascript(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        _install_stub_tkinter(monkeypatch)
        monkeypatch.setattr("pineuro.gui.dispatch.get_dispatcher", lambda: _FakeDispatcher("C:/x/f.yaml"))

        def boom(*a, **k):
            raise AssertionError("subprocess.run (osascript) must not run off macOS")

        monkeypatch.setattr(subprocess, "run", boom)
        assert pick_file(title="Pick") == "C:/x/f.yaml"

    def test_pick_directory_uses_dispatcher(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        _install_stub_tkinter(monkeypatch)
        monkeypatch.setattr("pineuro.gui.dispatch.get_dispatcher", lambda: _FakeDispatcher("/home/x/data"))
        assert pick_directory(title="Pick") == "/home/x/data"

    def test_pick_save_file_uses_dispatcher(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        _install_stub_tkinter(monkeypatch)
        monkeypatch.setattr("pineuro.gui.dispatch.get_dispatcher", lambda: _FakeDispatcher("C:/x/out.tsv"))
        assert pick_save_file(title="Save", default_extension=".tsv") == "C:/x/out.tsv"


# =============================================================================
# build_feedback_steps / feedback_step_for
# =============================================================================


class TestBuildFeedbackSteps:
    """Per-task BIDS entities and per-task run numbering for feedback/transfer steps."""

    # ball x2 with pre/post transfers, then gauge x1 (no transfers).
    CONFIG = {
        "feedback_runs": [
            {"task": "ball", "max_runs": 2, "use_transfer_runs": True},
            {"task": "gauge", "max_runs": 1},
        ]
    }

    def test_ordering_and_step_ids(self):
        steps = build_feedback_steps(self.CONFIG)
        assert [s.step_id for s in steps] == [
            "transfer_1",
            "feedback_1",
            "feedback_2",
            "transfer_2",
            "feedback_3",
        ]

    def test_task_entities_use_real_name_with_transfer_suffix(self):
        steps = build_feedback_steps(self.CONFIG)
        assert [s.task_entity for s in steps] == [
            "balltransfer",
            "ball",
            "ball",
            "balltransfer",
            "gauge",
        ]

    def test_run_numbers_restart_per_task_entity(self):
        steps = build_feedback_steps(self.CONFIG)
        # ball feedback: 1, 2 ; balltransfer: 1, 2 ; gauge feedback: 1
        by_step = {s.step_id: s.run_number for s in steps}
        assert by_step == {
            "transfer_1": 1,
            "feedback_1": 1,
            "feedback_2": 2,
            "transfer_2": 2,
            "feedback_3": 1,
        }

    def test_kind_tracks_feedback_vs_transfer(self):
        steps = {s.step_id: s.kind for s in build_feedback_steps(self.CONFIG)}
        assert steps["feedback_1"] == "feedback"
        assert steps["transfer_1"] == "transfer"

    def test_is_last_feedback_marks_final_feedback_of_each_block(self):
        # True on the last feedback step of each block (ball: feedback_2; gauge: feedback_3),
        # False on earlier feedback steps and on every transfer step. Gates
        # "last_feedback_only" post-run questions per task in the GUI path.
        by_step = {s.step_id: s.is_last_feedback for s in build_feedback_steps(self.CONFIG)}
        assert by_step == {
            "transfer_1": False,
            "feedback_1": False,
            "feedback_2": True,
            "transfer_2": False,
            "feedback_3": True,
        }

    def test_feedback_step_for_lookup(self):
        step = feedback_step_for(self.CONFIG, "feedback_3")
        assert step is not None
        assert step.task_id == "gauge"
        assert step.task_entity == "gauge"
        assert step.run_number == 1

    def test_feedback_step_for_unknown_returns_none(self):
        assert feedback_step_for(self.CONFIG, "feedback_99") is None

    def test_empty_config(self):
        assert build_feedback_steps({}) == []
