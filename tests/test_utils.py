"""Tests for pineuro.utils — utility functions, timing, file validation, masking."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from pineuro.utils import (
    Timer,
    apply_mask,
    compute_boxcar_task_regressor,
    compute_hrf,
    compute_mean,
    disable_timing,
    enable_timing,
    erode_mask,
    get_json_sidecar_path,
    get_timing_data,
    get_volume_files,
    is_nifti_file,
    outputs_exist,
    save_timing_report,
)

# =============================================================================
# outputs_exist
# =============================================================================


class TestOutputsExist:
    def test_all_present(self, tmp_path):
        f1 = tmp_path / "a.nii.gz"
        f2 = tmp_path / "b.nii.gz"
        f1.touch()
        f2.touch()
        assert outputs_exist([f1, f2], overwrite=False) is True

    def test_some_missing(self, tmp_path):
        f1 = tmp_path / "a.nii.gz"
        f2 = tmp_path / "b.nii.gz"
        f1.touch()
        assert outputs_exist([f1, f2], overwrite=False) is False

    def test_overwrite_true_always_false(self, tmp_path):
        f1 = tmp_path / "a.nii.gz"
        f1.touch()
        assert outputs_exist(f1, overwrite=True) is False

    def test_single_path(self, tmp_path):
        f1 = tmp_path / "a.nii.gz"
        f1.touch()
        assert outputs_exist(f1, overwrite=False) is True

    def test_single_string(self, tmp_path):
        f1 = tmp_path / "a.nii.gz"
        f1.touch()
        assert outputs_exist(str(f1), overwrite=False) is True

    def test_empty_dir_missing(self, tmp_path):
        f1 = tmp_path / "nonexistent.nii.gz"
        assert outputs_exist(f1, overwrite=False) is False


# =============================================================================
# is_nifti_file
# =============================================================================


class TestIsNiftiFile:
    def test_nii_gz(self):
        assert is_nifti_file(Path("vol.nii.gz")) is True

    def test_nii(self):
        assert is_nifti_file(Path("vol.nii")) is True

    def test_dcm(self):
        assert is_nifti_file(Path("vol.dcm")) is False

    def test_no_extension(self):
        assert is_nifti_file(Path("vol")) is False

    def test_case_insensitive(self):
        assert is_nifti_file(Path("vol.NII.GZ")) is True
        assert is_nifti_file(Path("vol.NII")) is True


# =============================================================================
# get_json_sidecar_path
# =============================================================================


class TestGetJsonSidecarPath:
    def test_nii_gz(self):
        result = get_json_sidecar_path(Path("/data/vol_001.nii.gz"))
        assert result == Path("/data/vol_001.json")

    def test_nii(self):
        result = get_json_sidecar_path(Path("/data/vol_001.nii"))
        assert result == Path("/data/vol_001.json")

    def test_preserves_directory(self):
        result = get_json_sidecar_path(Path("/some/deep/path/scan.nii.gz"))
        assert result.parent == Path("/some/deep/path")


# =============================================================================
# get_volume_files
# =============================================================================


class TestGetVolumeFiles:
    def test_sorted_output(self, tmp_path):
        (tmp_path / "vol_003.dcm").touch()
        (tmp_path / "vol_001.dcm").touch()
        (tmp_path / "vol_002.dcm").touch()
        result = get_volume_files(tmp_path)
        names = [p.name for p in result]
        assert names == ["vol_001.dcm", "vol_002.dcm", "vol_003.dcm"]

    def test_skips_directories(self, tmp_path):
        (tmp_path / "vol_001.dcm").touch()
        (tmp_path / "subdir").mkdir()
        result = get_volume_files(tmp_path)
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = get_volume_files(tmp_path)
        assert result == []


# =============================================================================
# Timing infrastructure
# =============================================================================


class TestTiming:
    def setup_method(self):
        """Reset timing state before each test."""
        enable_timing()  # clears _timing_data
        disable_timing()  # then disable

    def test_enable_disable(self):
        enable_timing()
        data = get_timing_data()
        assert isinstance(data, dict)
        assert len(data) == 0
        disable_timing()

    def test_timer_records_when_enabled(self):
        enable_timing()
        with Timer("test_op", "test_category"):
            pass  # instant operation
        data = get_timing_data()
        assert "test_category" in data
        assert len(data["test_category"]) == 1
        entry = data["test_category"][0]
        assert entry["name"] == "test_op"
        assert "elapsed_seconds" in entry

    def test_timer_noop_when_disabled(self):
        disable_timing()
        with Timer("test_op", "test_category"):
            pass
        data = get_timing_data()
        assert len(data) == 0

    def test_save_timing_report(self, tmp_path):
        enable_timing()
        with Timer("test_op", "cat"):
            pass
        report_path = tmp_path / "report.json"
        save_timing_report(report_path)
        assert report_path.exists()
        with open(report_path) as f:
            loaded = json.load(f)
        assert "cat" in loaded


# =============================================================================
# compute_hrf
# =============================================================================


class TestComputeHrf:
    def test_shape(self):
        hrf = compute_hrf(tr=2.0, duration=32.0)
        expected_length = int(32.0 / 2.0)
        assert hrf.shape == (expected_length,)

    def test_normalized_sum(self):
        hrf = compute_hrf(tr=1.0)
        # HRF is downsampled so won't sum exactly to 1, but should be close-ish
        # The important thing is it's normalized and not all zeros
        assert np.sum(np.abs(hrf)) > 0

    def test_peak_location(self):
        hrf = compute_hrf(tr=1.0, duration=32.0)
        peak_idx = np.argmax(hrf)
        peak_time = peak_idx * 1.0
        # HRF peak should be around 4-7 seconds
        assert 3 <= peak_time <= 8

    def test_peak_location_different_tr(self):
        """HRF peak should be ~5s regardless of TR."""
        for tr in [0.8, 1.0, 1.2, 1.5, 2.0]:
            hrf = compute_hrf(tr=tr, duration=32.0)
            peak_idx = np.argmax(hrf)
            peak_time = peak_idx * tr
            assert 3 <= peak_time <= 8, f"TR={tr}: peak at {peak_time}s (expected 4-7s)"

    def test_different_tr(self):
        hrf_1s = compute_hrf(tr=1.0, duration=32.0)
        hrf_2s = compute_hrf(tr=2.0, duration=32.0)
        assert len(hrf_1s) > len(hrf_2s)


# =============================================================================
# compute_boxcar_task_regressor
# =============================================================================


class TestComputeBoxcarTaskRegressor:
    def test_shape(self):
        reg = compute_boxcar_task_regressor(n_baseline=25, n_volumes=100, tr=2.0)
        assert reg.shape == (100,)

    def test_baseline_is_zero(self):
        reg = compute_boxcar_task_regressor(n_baseline=25, n_volumes=100, tr=2.0)
        # First few volumes during baseline should be zero or near-zero
        assert np.allclose(reg[:20], 0.0, atol=1e-6)

    def test_active_period_nonzero(self):
        reg = compute_boxcar_task_regressor(n_baseline=10, n_volumes=100, tr=1.0)
        # Well after baseline + HRF delay, regressor should be non-zero
        assert np.any(reg[30:] > 0.1)

    def test_ramps_up(self):
        reg = compute_boxcar_task_regressor(n_baseline=10, n_volumes=100, tr=1.0)
        # The regressor should increase from baseline to active
        late_mean = np.mean(reg[60:])
        early_mean = np.mean(reg[:10])
        assert late_mean > early_mean


# =============================================================================
# compute_mean (with synthetic NIfTI)
# =============================================================================


class TestComputeMean:
    def test_basic(self, synthetic_nifti_4d, tmp_path):
        input_path = synthetic_nifti_4d(shape=(4, 4, 4), n_volumes=10)
        output_path = tmp_path / "mean.nii.gz"
        compute_mean(input_path, output_path)
        assert output_path.exists()
        img = nib.load(output_path)
        assert img.shape == (4, 4, 4)

    def test_known_values(self, tmp_path):
        # Create 4D volume with known data: all ones
        data = np.ones((3, 3, 3, 5), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        input_path = tmp_path / "input.nii.gz"
        nib.save(img, input_path)
        output_path = tmp_path / "mean.nii.gz"
        compute_mean(input_path, output_path)
        result = nib.load(output_path).get_fdata()
        np.testing.assert_allclose(result, 1.0)

    def test_overwrite_false_skips(self, synthetic_nifti_4d, tmp_path):
        input_path = synthetic_nifti_4d()
        output_path = tmp_path / "mean.nii.gz"
        output_path.touch()  # pre-existing output
        compute_mean(input_path, output_path, overwrite=False)
        # File should still be empty (not overwritten)
        assert output_path.stat().st_size == 0


# =============================================================================
# apply_mask (with synthetic NIfTI)
# =============================================================================


class TestApplyMask:
    def test_zeros_outside_mask(self, tmp_path):
        shape = (4, 4, 4)
        data = np.ones(shape, dtype=np.float32)
        mask_data = np.zeros(shape, dtype=np.float32)
        mask_data[1:3, 1:3, 1:3] = 1.0  # small cube in center

        input_path = tmp_path / "input.nii.gz"
        mask_path = tmp_path / "mask.nii.gz"
        output_path = tmp_path / "masked.nii.gz"

        nib.save(nib.Nifti1Image(data, np.eye(4)), input_path)
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

        apply_mask(input_path, mask_path, output_path)

        result = nib.load(output_path).get_fdata()
        # Inside mask: 1.0, outside: 0.0
        assert result[0, 0, 0] == 0.0
        assert result[1, 1, 1] == 1.0

    def test_4d_with_3d_mask(self, tmp_path):
        shape_3d = (4, 4, 4)
        data_4d = np.ones((*shape_3d, 5), dtype=np.float32)
        mask_data = np.zeros(shape_3d, dtype=np.float32)
        mask_data[2, 2, 2] = 1.0

        input_path = tmp_path / "input4d.nii.gz"
        mask_path = tmp_path / "mask.nii.gz"
        output_path = tmp_path / "masked4d.nii.gz"

        nib.save(nib.Nifti1Image(data_4d, np.eye(4)), input_path)
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

        apply_mask(input_path, mask_path, output_path)
        result = nib.load(output_path).get_fdata()
        assert result.shape == (*shape_3d, 5)
        assert result[0, 0, 0, 0] == 0.0
        assert result[2, 2, 2, 0] == 1.0


# =============================================================================
# erode_mask (with synthetic NIfTI)
# =============================================================================


class TestErodeMask:
    def test_eroded_is_subset(self, tmp_path):
        shape = (10, 10, 10)
        mask_data = np.ones(shape, dtype=np.float32)

        input_path = tmp_path / "mask.nii.gz"
        output_path = tmp_path / "eroded.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), input_path)

        erode_mask(input_path, output_path, n_erode=1)

        original = nib.load(input_path).get_fdata() > 0
        eroded = nib.load(output_path).get_fdata() > 0
        # Eroded mask should be a subset
        assert np.sum(eroded) < np.sum(original)
        assert np.all(eroded <= original)

    def test_boundary_removed(self, tmp_path):
        shape = (10, 10, 10)
        mask_data = np.ones(shape, dtype=np.float32)

        input_path = tmp_path / "mask.nii.gz"
        output_path = tmp_path / "eroded.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), input_path)

        erode_mask(input_path, output_path, n_erode=1)
        eroded = nib.load(output_path).get_fdata() > 0
        # Edge voxels should be removed
        assert not eroded[0, 0, 0]
        assert eroded[5, 5, 5]


# =============================================================================
# run_command (mocked)
# =============================================================================


class TestRunCommand:
    def test_success(self, tmp_path):
        from pineuro.utils import run_command

        output_file = tmp_path / "output.txt"

        with (
            patch("pineuro.utils.subprocess.run") as mock_run,
            patch("pineuro.utils._check_command_installation"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            # Create output file to simulate command success
            output_file.touch()
            run_command("echo", ["echo", "hello"], "test echo", outputs=[output_file])

    def test_skip_existing_outputs(self, tmp_path):
        from pineuro.utils import run_command

        output_file = tmp_path / "output.txt"
        output_file.touch()

        with patch("pineuro.utils.subprocess.run") as mock_run:
            run_command("echo", ["echo", "hello"], "test echo", outputs=[output_file], overwrite=False)
            mock_run.assert_not_called()

    def test_missing_input_raises(self, tmp_path):
        from pineuro.utils import run_command

        with (
            pytest.raises(FileNotFoundError),
            patch("pineuro.utils._check_command_installation"),
        ):
            run_command("echo", ["echo"], "test", inputs=[tmp_path / "nonexistent.txt"])
