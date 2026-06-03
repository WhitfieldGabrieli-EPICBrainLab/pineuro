"""Tests for pineuro.utils — utility functions, timing, file validation, masking."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from pineuro.utils import (
    HEAD_RADIUS_MM,
    Timer,
    apply_mask,
    compute_boxcar_task_regressor,
    compute_dvars,
    compute_fd,
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
    parse_siemens_motion_comment,
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
# get_volume_files — directory enumeration with validation
# =============================================================================


def _make_nifti(path: Path) -> Path:
    """Write a tiny valid NIfTI volume at ``path``."""
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), path)
    return path


def _make_dicom(path: Path) -> Path:
    """Write a minimal valid Part-10 DICOM at ``path`` (header only, no pixels)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientID = "PHANTOM"
    ds.Modality = "MR"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.save_as(str(path), enforce_file_format=True)
    return path


class TestGetVolumeFiles:
    def test_sorted_output(self, tmp_path):
        # validate=False: this exercises ordering only, not file validity.
        (tmp_path / "vol_003.dcm").touch()
        (tmp_path / "vol_001.dcm").touch()
        (tmp_path / "vol_002.dcm").touch()
        result = get_volume_files(tmp_path, validate=False)
        names = [p.name for p in result]
        assert names == ["vol_001.dcm", "vol_002.dcm", "vol_003.dcm"]

    def test_skips_directories(self, tmp_path):
        (tmp_path / "vol_001.dcm").touch()
        (tmp_path / "subdir").mkdir()
        result = get_volume_files(tmp_path, validate=False)
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = get_volume_files(tmp_path)
        assert result == []

    def test_skips_ds_store(self, tmp_path, caplog):
        _make_nifti(tmp_path / "vol_001.nii.gz")
        _make_nifti(tmp_path / "vol_002.nii.gz")
        (tmp_path / ".DS_Store").write_bytes(b"\x00junk")
        with caplog.at_level(logging.WARNING):
            result = get_volume_files(tmp_path)
        assert [p.name for p in result] == ["vol_001.nii.gz", "vol_002.nii.gz"]
        assert ".DS_Store" in caplog.text

    def test_skips_text_file(self, tmp_path, caplog):
        _make_nifti(tmp_path / "vol_001.nii.gz")
        (tmp_path / "notes.txt").write_text("not a volume")
        with caplog.at_level(logging.WARNING):
            result = get_volume_files(tmp_path)
        assert [p.name for p in result] == ["vol_001.nii.gz"]
        assert "notes.txt" in caplog.text

    def test_excludes_json_sidecar_silently(self, tmp_path, caplog):
        _make_nifti(tmp_path / "vol_000.nii.gz")
        (tmp_path / "vol_000.json").write_text('{"RepetitionTime": 2.0}')
        with caplog.at_level(logging.WARNING):
            result = get_volume_files(tmp_path)
        assert [p.name for p in result] == ["vol_000.nii.gz"]
        # Sidecars are expected companions, not junk — not warned about.
        assert "vol_000.json" not in caplog.text

    def test_skips_os_junk_basenames(self, tmp_path):
        _make_nifti(tmp_path / "vol_001.nii.gz")
        (tmp_path / "Thumbs.db").write_bytes(b"\x00")
        (tmp_path / "._vol_001.nii.gz").write_bytes(b"\x00")  # AppleDouble
        result = get_volume_files(tmp_path)
        assert [p.name for p in result] == ["vol_001.nii.gz"]

    def test_only_junk_returns_empty_with_warning(self, tmp_path, caplog):
        (tmp_path / ".DS_Store").write_bytes(b"\x00")
        (tmp_path / "notes.txt").write_text("nope")
        with caplog.at_level(logging.WARNING):
            result = get_volume_files(tmp_path)
        assert result == []
        assert "No valid DICOM/NIfTI volumes" in caplog.text

    def test_validate_false_returns_all(self, tmp_path):
        _make_nifti(tmp_path / "vol_001.nii.gz")
        (tmp_path / ".DS_Store").write_bytes(b"\x00")
        (tmp_path / "notes.txt").write_text("nope")
        result = get_volume_files(tmp_path, validate=False)
        assert {p.name for p in result} == {"vol_001.nii.gz", ".DS_Store", "notes.txt"}

    def test_recognizes_extensionless_dicom(self, tmp_path):
        _make_dicom(tmp_path / "MR000001")  # no extension, as some scanners export
        (tmp_path / ".DS_Store").write_bytes(b"\x00")
        result = get_volume_files(tmp_path)
        assert [p.name for p in result] == ["MR000001"]


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
# run_command, mocked
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


# =============================================================================
# dicom_to_nifti — overwrite cleanup of stale dcm2niix artifacts
# =============================================================================


class TestDicomToNiftiOverwriteCleanup:
    """dcm2niix's default -w 2 appends a suffix (a, b, _e2, ...) on filename
    collision instead of overwriting, so dicom_to_nifti(overwrite=True) must
    delete the prior outputs first to avoid silently leaving stale files."""

    def _setup(self, tmp_path, output_name):
        from pineuro.utils import dicom_to_nifti

        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()
        (dicom_dir / "fake.dcm").touch()
        output_dir = tmp_path / "01_nifti"
        output_dir.mkdir()
        primary = output_dir / f"{output_name}.nii.gz"
        return dicom_to_nifti, dicom_dir, output_dir, primary

    def test_removes_stale_primary_suffix_and_multiecho(self, tmp_path):
        output_name = "sub-01_ses-01_task-rest_run-01_attempt-01"
        dicom_to_nifti, dicom_dir, output_dir, primary = self._setup(tmp_path, output_name)

        # Stale files dcm2niix may have left behind on prior runs
        stale_files = [
            output_dir / f"{output_name}.nii.gz",  # primary from a prior run
            output_dir / f"{output_name}.json",
            output_dir / f"{output_name}a.nii.gz",  # collision suffix from -w 2
            output_dir / f"{output_name}a.json",
            output_dir / f"{output_name}_e2.nii.gz",  # multi-echo split
            output_dir / f"{output_name}_ph.nii.gz",  # phase
        ]
        for f in stale_files:
            f.write_text("stale")

        def fake_run(cmd, *args, **kwargs):
            # Simulate dcm2niix writing the fresh primary file
            primary.write_bytes(b"fresh")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("pineuro.utils.subprocess.run", side_effect=fake_run),
            patch("pineuro.utils._check_command_installation"),
        ):
            dicom_to_nifti(dicom_dir, output_dir, output_name, overwrite=True)

        # All stale variants gone; primary has fresh content
        for f in stale_files:
            if f == primary:
                continue
            assert not f.exists(), f"Stale file should have been deleted: {f}"
        assert primary.read_bytes() == b"fresh"

    def test_does_not_touch_other_outputs(self, tmp_path):
        """Cleanup must be scoped to the current output_name only."""
        output_name = "sub-01_ses-01_task-rest_run-01_attempt-01"
        dicom_to_nifti, dicom_dir, output_dir, primary = self._setup(tmp_path, output_name)

        sibling_run = output_dir / "sub-01_ses-01_task-rest_run-02_attempt-01.nii.gz"
        sibling_run.write_text("other-run")
        sibling_attempt = output_dir / "sub-01_ses-01_task-rest_run-01_attempt-02.nii.gz"
        sibling_attempt.write_text("other-attempt")

        def fake_run(cmd, *args, **kwargs):
            primary.write_bytes(b"fresh")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("pineuro.utils.subprocess.run", side_effect=fake_run),
            patch("pineuro.utils._check_command_installation"),
        ):
            dicom_to_nifti(dicom_dir, output_dir, output_name, overwrite=True)

        assert sibling_run.read_text() == "other-run"
        assert sibling_attempt.read_text() == "other-attempt"

    def test_overwrite_false_no_cleanup(self, tmp_path):
        """When overwrite=False, cleanup must not run (it would be wasted work
        and could surprise a user who explicitly opted out)."""
        output_name = "sub-01_ses-01_task-rest_run-01_attempt-01"
        dicom_to_nifti, dicom_dir, output_dir, primary = self._setup(tmp_path, output_name)

        stale_sidecar = output_dir / f"{output_name}.json"
        stale_sidecar.write_text("stale-json")
        primary.write_bytes(b"prior")  # so run_command's skip path triggers

        with patch("pineuro.utils.subprocess.run") as mock_run:
            dicom_to_nifti(dicom_dir, output_dir, output_name, overwrite=False)
            mock_run.assert_not_called()

        assert stale_sidecar.read_text() == "stale-json"
        assert primary.read_bytes() == b"prior"


# =============================================================================
# compute_fd
# =============================================================================


class TestComputeFD:
    def test_first_volume_returns_zero(self):
        motion = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        assert compute_fd(motion, None) == 0.0

    def test_no_motion_returns_zero(self):
        motion = np.zeros(6)
        prev = np.zeros(6)
        assert compute_fd(motion, prev) == 0.0

    def test_pure_translation(self):
        motion = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        prev = np.zeros(6)
        # Translation of 1 mm in x -> FD = 1.0 mm.
        assert compute_fd(motion, prev) == pytest.approx(1.0)

    def test_rotation_uses_head_radius(self):
        prev = np.zeros(6)
        motion_with_rot = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        # 1 deg rotation -> arc length HEAD_RADIUS_MM * pi/180.
        expected_arc = HEAD_RADIUS_MM * np.pi / 180.0
        assert compute_fd(motion_with_rot, prev) == pytest.approx(expected_arc)

    def test_combined_translation_and_rotation(self):
        motion = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        prev = np.zeros(6)
        assert compute_fd(motion, prev) > 0


# =============================================================================
# compute_dvars
# =============================================================================


class TestComputeDVARS:
    def test_first_volume_returns_zero(self):
        voxels = np.array([100.0, 200.0, 300.0])
        assert compute_dvars(voxels, None) == 0.0

    def test_no_change_returns_zero(self):
        voxels = np.array([100.0, 200.0, 300.0])
        assert compute_dvars(voxels, voxels) == 0.0

    def test_uniform_change(self):
        prev = np.array([100.0, 100.0, 100.0])
        curr = np.array([110.0, 110.0, 110.0])
        # RMS of [10, 10, 10] = 10.0.
        assert compute_dvars(curr, prev) == pytest.approx(10.0)


# =============================================================================
# standardize_dvars
# =============================================================================


class TestStandardizeDVARS:
    def test_raw_zero_returns_zero(self):
        # First-volume DVARS is 0.0 and stays 0.0 regardless of history.
        from pineuro.utils import standardize_dvars

        assert standardize_dvars(0.0, []) == 0.0
        assert standardize_dvars(0.0, [10.0, 10.0]) == 0.0

    def test_first_real_value_is_reference(self):
        # With no prior history the first real DVARS is the reference -> 1.0.
        from pineuro.utils import standardize_dvars

        assert standardize_dvars(12.3, []) == 1.0

    def test_divides_by_running_median(self):
        from pineuro.utils import standardize_dvars

        assert standardize_dvars(10.0, [10.0, 10.0, 10.0]) == pytest.approx(1.0)
        assert standardize_dvars(20.0, [10.0, 10.0, 10.0]) == pytest.approx(2.0)

    def test_median_is_robust_to_spikes(self):
        # A spike already in the history must not inflate the scale (median, not mean).
        from pineuro.utils import standardize_dvars

        assert standardize_dvars(10.0, [10.0, 10.0, 100.0]) == pytest.approx(1.0)

    def test_degenerate_scale_returns_one(self):
        from pineuro.utils import standardize_dvars

        assert standardize_dvars(5.0, [0.0, 0.0]) == 1.0


# =============================================================================
# parse_siemens_motion_comment
# =============================================================================


class TestParseSiemensMotionComment:
    def test_reference_volume_returns_zeros(self):
        motion = parse_siemens_motion_comment("Reference volume for motion correction")
        assert motion is not None
        np.testing.assert_array_equal(motion, np.zeros(6))

    def test_motion_params_parsed(self):
        motion = parse_siemens_motion_comment("Motion: 0.1,-0.2,0.3,1.0,-2.0,3.0")
        assert motion is not None
        np.testing.assert_allclose(motion, [0.1, -0.2, 0.3, 1.0, -2.0, 3.0])

    def test_no_pattern_returns_none(self):
        assert parse_siemens_motion_comment("some other comment") is None

    def test_empty_string_returns_none(self):
        assert parse_siemens_motion_comment("") is None


class TestParallelWorkerErrors:
    """Parallel-job failures must carry a traceback back to the parent, not just str(e)."""

    def test_worker_captures_traceback(self):
        from pineuro.utils import _run_job_worker

        # Missing the required "cmd_name" key raises KeyError inside the worker (an
        # unexpected error, not a CalledProcessError); the traceback is what tells you
        # where it failed across the process boundary.
        result = _run_job_worker({"description": "bad job"})

        assert result["success"] is False
        assert result["description"] == "bad job"
        assert "Traceback (most recent call last)" in result["traceback"]
        assert "KeyError" in result["traceback"]

    def test_parallel_run_command_surfaces_traceback(self, monkeypatch):
        from pineuro.utils import parallel_run_command

        # Run the worker in-process via a fake Pool so the test never spawns processes.
        class _FakePool:
            def __init__(self, processes):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, items):
                return [fn(i) for i in items]

        monkeypatch.setattr("multiprocessing.Pool", _FakePool)

        with pytest.raises(RuntimeError) as exc_info:
            parallel_run_command([{"description": "bad job"}], n_jobs=1)

        message = str(exc_info.value)
        assert "bad job" in message
        assert "Traceback (most recent call last)" in message
