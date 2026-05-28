"""Tests for pineuro.script_generator — mask extraction script generation."""

import sys

import pytest

from pineuro.script_generator import (
    generate_mask_extraction_script,
    generate_task_mask_extraction_script,
    save_script,
)

# =============================================================================
# generate_mask_extraction_script — rest/ICA pipeline
# =============================================================================


class TestGenerateMaskExtractionScript:
    def test_returns_valid_python(self, tmp_path):
        script = generate_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path / "masks",
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
        )
        assert isinstance(script, str)
        # Should be valid Python syntax
        compile(script, "<test>", "exec")

    def test_contains_participant_id(self, tmp_path):
        script = generate_mask_extraction_script(
            participant_id="sub-42",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
        )
        assert "sub-42" in script

    def test_contains_parameters(self, tmp_path):
        script = generate_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
            n_components=25,
            threshold=1500,
            n_yeo=7,
            yeo_targets=["DefaultA", "VisualA"],
        )
        assert "n_components = 25" in script
        assert "threshold = 1500" in script
        assert "n_yeo = 7" in script
        assert "'DefaultA'" in script
        assert "'VisualA'" in script

    def test_default_yeo_targets(self, tmp_path):
        script = generate_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
        )
        assert "'DefaultA'" in script
        assert "'ContA'" in script

    def test_smoothing_and_filtering(self, tmp_path):
        script = generate_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
            smooth_fwhm=6.0,
            highpass=0.01,
            lowpass=0.1,
        )
        assert "smooth_fwhm = 6.0" in script
        assert "highpass = 0.01" in script
        assert "lowpass = 0.1" in script

    def test_no_selections_generates_placeholder(self, tmp_path):
        script = generate_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
            selections=None,
        )
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
        script = generate_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            ref_file=tmp_path / "boldref.nii.gz",
            selections={"DefaultA": sel},
        )
        # Manual selections should be in commented-out block
        assert "STEP 2" in script
        assert "manual" in script


# =============================================================================
# generate_task_mask_extraction_script — GLM pipeline
# =============================================================================


class TestGenerateTaskMaskExtractionScript:
    def test_returns_valid_python(self, tmp_path):
        script = generate_task_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            events_file=tmp_path / "events.tsv",
            contrasts={"motor": "task - rest"},
            tr=2.0,
        )
        assert isinstance(script, str)
        compile(script, "<test>", "exec")

    def test_contains_contrasts(self, tmp_path):
        script = generate_task_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            events_file=tmp_path / "events.tsv",
            contrasts={"motor": "task - rest", "visual": "visual - baseline"},
            tr=1.5,
        )
        assert "'motor'" in script
        assert "'visual'" in script

    def test_cluster_method_params(self, tmp_path):
        script = generate_task_mask_extraction_script(
            participant_id="sub-01",
            session_id="ses-01",
            mask_extraction_dir=tmp_path,
            input_file=tmp_path / "preproc.nii.gz",
            mask_file=tmp_path / "mask.nii.gz",
            events_file=tmp_path / "events.tsv",
            contrasts={"motor": "task - rest"},
            tr=2.0,
            threshold_method="cluster",
            atlas_name="harvard_oxford",
            cluster_start_threshold=3.0,
            cluster_step=0.2,
        )
        assert 'threshold_method = "cluster"' in script
        assert "'harvard_oxford'" in script


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
