"""Tests for pineuro.script_generator — mask extraction script generation."""

import sys

import pytest

from pineuro.script_generator import generate_mask_extraction_script, save_script


def _rest_params(tmp_path, **over):
    p = {
        "participant_id": "sub-01",
        "session_id": "ses-01",
        "root_dir": str(tmp_path / "masks"),
        "input_file": str(tmp_path / "preproc.nii.gz"),
        "mask_file": str(tmp_path / "mask.nii.gz"),
        "ref_file": str(tmp_path / "boldref.nii.gz"),
        "networks": ["DefaultA", "ContA"],
        "network_atlas": "yeo_17",
        "n_components": 30,
        "threshold": 2000,
        "smooth_fwhm": None,
        "highpass": None,
        "lowpass": None,
        "overwrite": True,
        "qc_enabled": True,
    }
    p.update(over)
    return p


def _task_params(tmp_path, **over):
    p = {
        "participant_id": "sub-01",
        "session_id": "ses-01",
        "root_dir": str(tmp_path / "masks"),
        "input_file": str(tmp_path / "preproc.nii.gz"),
        "mask_file": str(tmp_path / "mask.nii.gz"),
        "events": str(tmp_path / "events.tsv"),
        "contrasts": {"motor": "task - rest"},
        "tr": 2.0,
        "threshold": 2000,
        "threshold_method": "top_n",
        "overwrite": True,
        "qc_enabled": True,
    }
    p.update(over)
    return p


def _mni_params(tmp_path, **over):
    p = {
        "participant_id": "sub-01",
        "session_id": "ses-01",
        "root_dir": str(tmp_path / "masks"),
        "ref_file": str(tmp_path / "boldref.nii.gz"),
        "networks": ["DefaultA"],
        "network_atlas": "yeo_17",
        "threshold": 2000,
        "overwrite": True,
        "qc_enabled": True,
    }
    p.update(over)
    return p


# =============================================================================
# generate_mask_extraction_script — rest / ICA
# =============================================================================


class TestRestScript:
    def test_returns_valid_python(self, tmp_path):
        script = generate_mask_extraction_script("rest", _rest_params(tmp_path))
        assert isinstance(script, str)
        compile(script, "<test>", "exec")

    def test_routes_through_dispatcher(self, tmp_path):
        script = generate_mask_extraction_script("rest", _rest_params(tmp_path))
        assert "run_mask_extraction_from_config" in script
        assert 'MODE = "rest"' in script

    def test_contains_parameters(self, tmp_path):
        script = generate_mask_extraction_script(
            "rest",
            _rest_params(tmp_path, participant_id="sub-42", n_components=25, threshold=1500),
        )
        assert "sub-42" in script
        assert "'n_components': 25" in script
        assert "'threshold': 1500" in script

    def test_carries_network_atlas(self, tmp_path):
        """The Schaefer/non-Yeo atlas is now reproducible (was dropped before)."""
        script = generate_mask_extraction_script("rest", _rest_params(tmp_path, network_atlas="schaefer_400_17net"))
        assert "schaefer_400_17net" in script

    def test_qc_disabled_in_recorded_call(self, tmp_path):
        """Rest regenerates QC itself, so the recorded call disables built-in QC."""
        script = generate_mask_extraction_script("rest", _rest_params(tmp_path, qc_enabled=True))
        assert "'qc_enabled': False" in script

    def test_no_selections_generates_placeholder(self, tmp_path):
        script = generate_mask_extraction_script("rest", _rest_params(tmp_path), selections=None)
        assert "No manual mask selections" in script

    def test_with_manual_selections(self, tmp_path):
        from pineuro.mask_extraction import MaskSelection

        sel = MaskSelection(
            network_name="DefaultA",
            source_type="ica",
            component_index=5,
            correlation=0.85,
            selection_reason="manual",
        )
        script = generate_mask_extraction_script("rest", _rest_params(tmp_path), selections={"DefaultA": sel})
        compile(script, "<test>", "exec")
        assert "STEP 2" in script
        assert "manual" in script


# =============================================================================
# generate_mask_extraction_script — task / GLM
# =============================================================================


class TestTaskScript:
    def test_returns_valid_python(self, tmp_path):
        script = generate_mask_extraction_script("task", _task_params(tmp_path))
        assert isinstance(script, str)
        compile(script, "<test>", "exec")
        assert 'MODE = "task"' in script

    def test_contains_contrasts(self, tmp_path):
        script = generate_mask_extraction_script(
            "task",
            _task_params(tmp_path, contrasts={"motor": "task - rest", "visual": "visual - baseline"}),
        )
        assert "'motor'" in script
        assert "task - rest" in script
        assert "visual - baseline" in script

    def test_cluster_extent_method_and_atlas_source(self, tmp_path):
        """cluster_extent (the newest method) + atlas_source/atlas_regions propagate."""
        script = generate_mask_extraction_script(
            "task",
            _task_params(
                tmp_path,
                threshold_method="cluster_extent",
                atlas_source="harvard_oxford",
                atlas_regions=["Superior Temporal Gyrus"],
                min_cluster_size=40,
            ),
        )
        compile(script, "<test>", "exec")
        assert "cluster_extent" in script
        assert "harvard_oxford" in script
        assert "Superior Temporal Gyrus" in script
        assert "'min_cluster_size': 40" in script


# =============================================================================
# generate_mask_extraction_script — mni / atlas fallback
# =============================================================================


class TestMniScript:
    def test_returns_valid_python(self, tmp_path):
        script = generate_mask_extraction_script("mni", _mni_params(tmp_path))
        compile(script, "<test>", "exec")
        assert 'MODE = "mni"' in script
        assert "run_mask_extraction_from_config" in script

    def test_carries_network_atlas(self, tmp_path):
        script = generate_mask_extraction_script("mni", _mni_params(tmp_path, network_atlas="yeo_7"))
        assert "yeo_7" in script


def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown mask extraction mode"):
        generate_mask_extraction_script("bogus", _mni_params(tmp_path))


# =============================================================================
# run_mask_extraction_from_config — dispatch
# =============================================================================


class TestDispatch:
    def test_rest_dispatch_forwards_runtime_args(self, monkeypatch):
        import pineuro.mask_extraction as me

        seen = {}

        def fake_rest(**kwargs):
            seen.update(kwargs)
            return ["m1"]

        monkeypatch.setattr(me, "mask_extraction_rest", fake_rest)
        out = me.run_mask_extraction_from_config(
            "rest", {"participant_id": "s"}, progress_callback="CB", stop_event="EVT"
        )
        assert out == ["m1"]
        assert seen["participant_id"] == "s"
        assert seen["progress_callback"] == "CB"
        assert seen["stop_event"] == "EVT"

    def test_task_dispatch_does_not_forward_stop_event(self, monkeypatch):
        import pineuro.mask_extraction as me

        seen = {}

        def fake_task(**kwargs):
            seen.update(kwargs)
            return ["t1"]

        monkeypatch.setattr(me, "mask_extraction_task", fake_task)
        out = me.run_mask_extraction_from_config("task", {"contrasts": {}}, stop_event="EVT")
        assert out == ["t1"]
        assert "stop_event" not in seen
        assert seen["progress_callback"] is None

    def test_mni_dispatch(self, monkeypatch):
        import pineuro.mask_extraction as me

        seen = {}

        def fake_mni(**kwargs):
            seen.update(kwargs)
            return ["a1"]

        monkeypatch.setattr(me, "mask_extraction_mni", fake_mni)
        out = me.run_mask_extraction_from_config("mni", {"ref_file": "r"})
        assert out == ["a1"]
        assert seen["ref_file"] == "r"

    def test_unknown_mode_raises(self):
        from pineuro.mask_extraction import run_mask_extraction_from_config

        with pytest.raises(ValueError, match="Unknown mask extraction mode"):
            run_mask_extraction_from_config("bogus", {})


# =============================================================================
# save_script
# =============================================================================


class TestSaveScript:
    def test_saves_to_file(self, tmp_path):
        script = "#!/usr/bin/env python3\nprint('hello')\n"
        path = save_script(script, tmp_path / "test_script.py")
        assert path.exists()
        assert path.read_text() == script

    def test_creates_parent_dirs(self, tmp_path):
        path = save_script("print('hi')", tmp_path / "sub" / "dir" / "script.py")
        assert path.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not applicable on Windows")
    def test_executable_on_unix(self, tmp_path):
        path = save_script("print('hi')", tmp_path / "script.py")
        assert path.stat().st_mode & 0o100  # owner execute bit
