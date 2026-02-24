"""Tests for pineuro.preprocessing — pipeline steps and volume handling."""

import threading
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pineuro.preprocessing import collect_volumes, concat, spatial_smoothing, temporal_filtering

# =============================================================================
# concat
# =============================================================================


class TestConcat:
    def test_two_volumes(self, tmp_path):
        shape = (4, 4, 4)
        affine = np.eye(4)

        data1 = np.ones((*shape, 5), dtype=np.float32)
        data2 = np.ones((*shape, 3), dtype=np.float32) * 2.0

        f1 = tmp_path / "vol1.nii.gz"
        f2 = tmp_path / "vol2.nii.gz"
        output = tmp_path / "concat.nii.gz"

        nib.save(nib.Nifti1Image(data1, affine), f1)
        nib.save(nib.Nifti1Image(data2, affine), f2)

        concat([f1, f2], output)

        result = nib.load(output)
        assert result.shape == (*shape, 8)
        result_data = result.get_fdata()
        np.testing.assert_allclose(result_data[..., :5], 1.0)
        np.testing.assert_allclose(result_data[..., 5:], 2.0)

    def test_single_volume(self, tmp_path):
        data = np.ones((4, 4, 4, 10), dtype=np.float32)
        f1 = tmp_path / "vol.nii.gz"
        output = tmp_path / "concat.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), f1)

        concat([f1], output)

        result = nib.load(output)
        assert result.shape == (4, 4, 4, 10)

    def test_overwrite_false_skips(self, tmp_path):
        data = np.ones((4, 4, 4, 5), dtype=np.float32)
        f1 = tmp_path / "vol.nii.gz"
        output = tmp_path / "concat.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), f1)
        output.touch()  # pre-existing empty file

        concat([f1], output, overwrite=False)
        assert output.stat().st_size == 0  # not overwritten


# =============================================================================
# spatial_smoothing
# =============================================================================


class TestSpatialSmoothing:
    def test_output_exists(self, synthetic_nifti_4d, tmp_path):
        input_path = synthetic_nifti_4d(shape=(8, 8, 8), n_volumes=5)
        output_path = tmp_path / "smoothed.nii.gz"

        spatial_smoothing(input_path, output_path, fwhm=5.0)

        assert output_path.exists()
        result = nib.load(output_path)
        assert result.shape == (8, 8, 8, 5)

    def test_smoothing_changes_values(self, synthetic_nifti_4d, tmp_path):
        input_path = synthetic_nifti_4d(shape=(8, 8, 8), n_volumes=5)
        output_path = tmp_path / "smoothed.nii.gz"

        spatial_smoothing(input_path, output_path, fwhm=5.0)

        original = nib.load(input_path).get_fdata()
        smoothed = nib.load(output_path).get_fdata()
        # Smoothing should change the values
        assert not np.allclose(original, smoothed)

    def test_overwrite_false_skips(self, synthetic_nifti_4d, tmp_path):
        input_path = synthetic_nifti_4d()
        output_path = tmp_path / "smoothed.nii.gz"
        output_path.touch()

        spatial_smoothing(input_path, output_path, overwrite=False)
        assert output_path.stat().st_size == 0


# =============================================================================
# temporal_filtering
# =============================================================================


class TestTemporalFiltering:
    def test_highpass_removes_low_freq(self, tmp_path):
        """High-pass filter should attenuate slow drifts."""
        n_volumes = 100
        tr = 2.0
        shape = (3, 3, 3)

        # Create synthetic data: slow drift + fast signal
        t = np.arange(n_volumes) * tr
        slow_drift = np.sin(2 * np.pi * t / 200)  # very slow: period=200s
        fast_signal = np.sin(2 * np.pi * t / 10)  # fast: period=10s

        data = np.zeros((*shape, n_volumes), dtype=np.float32)
        data[1, 1, 1, :] = slow_drift + fast_signal

        input_path = tmp_path / "input.nii.gz"
        output_path = tmp_path / "filtered.nii.gz"

        img = nib.Nifti1Image(data, np.eye(4))
        img.header["pixdim"][4] = tr
        nib.save(img, input_path)

        temporal_filtering(input_path, output_path, highpass=0.01)  # cutoff=0.01 Hz

        result = nib.load(output_path).get_fdata()
        filtered_voxel = result[1, 1, 1, :]

        # Filter should have removed most of the drift
        assert np.var(filtered_voxel) < np.var(slow_drift + fast_signal)

    def test_no_filter_passthrough(self, synthetic_nifti_4d, tmp_path):
        """No highpass/lowpass should not change data."""
        input_path = synthetic_nifti_4d(shape=(3, 3, 3), n_volumes=20)
        output_path = tmp_path / "filtered.nii.gz"

        temporal_filtering(input_path, output_path, highpass=None, lowpass=None)
        assert output_path.exists()


# =============================================================================
# motion_correction (mocked ANTsPy)
# =============================================================================


class TestMotionCorrection:
    def test_overwrite_false_skips(self, synthetic_nifti_4d, tmp_path):
        """motion_correction should skip when output exists and overwrite=False."""
        from pineuro.preprocessing import motion_correction

        input_path = synthetic_nifti_4d()
        output_path = tmp_path / "moco.nii.gz"
        par_path = Path(str(output_path) + ".par")
        # Create both expected outputs (output + .par)
        output_path.touch()
        par_path.touch()

        # Should return early without importing ants
        motion_correction(input_path, output_path, overwrite=False)
        assert output_path.exists()


# =============================================================================
# brain_extraction (mocked ANTsPy)
# =============================================================================


class TestBrainExtraction:
    def test_overwrite_false_skips(self, synthetic_nifti_3d, tmp_path):
        """brain_extraction should skip when outputs exist and overwrite=False."""
        from pineuro.preprocessing import brain_extraction

        input_path = synthetic_nifti_3d(shape=(8, 8, 8))
        output_dir = tmp_path / "brain_extraction"
        output_dir.mkdir()

        # output_name = input_file.with_suffix("").stem → "vol3d"
        output_name = input_path.with_suffix("").stem
        brain_file = output_dir / f"{output_name}_brain.nii.gz"
        mask_file = output_dir / f"{output_name}_brain_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), brain_file)
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), mask_file)

        # Should return early without importing ants
        result = brain_extraction(input_path, output_dir, overwrite=False)
        assert result == (brain_file, mask_file)


# =============================================================================
# preprocessing pipeline (mocked sub-steps)
# =============================================================================


class TestPreprocessingPipeline:
    def test_validates_input_dicoms(self, tmp_path):
        """preprocessing should raise FileNotFoundError for missing input_dicoms."""
        from pineuro.preprocessing import preprocessing

        root_dir = tmp_path / "study"
        root_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Folder not found"):
            preprocessing("sub-01", "rest", root_dir, tmp_path / "nonexistent_dicoms")

    def test_validates_empty_dicom_dir(self, tmp_path):
        """preprocessing should raise FileNotFoundError for empty input_dicoms dir."""
        from pineuro.preprocessing import preprocessing

        root_dir = tmp_path / "study"
        root_dir.mkdir()
        empty_dicoms = tmp_path / "dicoms"
        empty_dicoms.mkdir()

        with pytest.raises(FileNotFoundError, match="Directory is empty"):
            preprocessing("sub-01", "rest", root_dir, empty_dicoms)


# =============================================================================
# collect_volumes — recursive subfolder watching
# =============================================================================


def _make_nifti_volume(path: Path, shape=(4, 4, 4)) -> None:
    """Create a minimal valid NIfTI volume file."""
    data = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(path))


class TestCollectVolumesRecursive:
    """Tests for recursive subfolder watching in collect_volumes."""

    def test_flat_directory_still_works(self, tmp_path):
        """Flat directory (GE/simulator style) works as before."""
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()
        output_dir = tmp_path / "output"
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(3):
            _make_nifti_volume(source_dir / f"vol_{i:04d}.nii.gz")

        result = collect_volumes(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=3,
            simulator_source_dir=source_dir,
            simulator_speed=100.0,
            timeout=30.0,
        )
        collected = sorted(result.iterdir())
        assert len([f for f in collected if f.is_file()]) >= 3

    def test_subfolder_files_detected(self, tmp_path):
        """Files created in a new subfolder are detected and collected."""
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()
        output_dir = tmp_path / "output"

        def write_to_subfolder():
            """Simulate scanner creating a subfolder and writing NIfTIs."""
            time.sleep(0.5)  # Let watchdog start
            subdir = watch_dir / "seq_f1"
            subdir.mkdir()
            for i in range(3):
                time.sleep(0.1)
                _make_nifti_volume(subdir / f"vol_{i:04d}.nii.gz")

        writer = threading.Thread(target=write_to_subfolder)
        writer.start()

        result = collect_volumes(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=3,
            timeout=15.0,
        )
        writer.join()
        collected = [f for f in result.iterdir() if f.is_file()]
        assert len(collected) >= 3

    def test_preexisting_subfolder_files_skipped(self, tmp_path):
        """Files in pre-existing subfolders are not collected (already-seen)."""
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()
        output_dir = tmp_path / "output"

        # Pre-existing subfolder from a previous sequence
        old_subdir = watch_dir / "old_rest_seq"
        old_subdir.mkdir()
        for i in range(5):
            _make_nifti_volume(old_subdir / f"vol_{i:04d}.nii.gz")

        def write_new_subfolder():
            """Simulate new scanner sequence after a delay."""
            time.sleep(0.5)
            new_subdir = watch_dir / "new_feedback_seq"
            new_subdir.mkdir()
            for i in range(3):
                time.sleep(0.1)
                _make_nifti_volume(new_subdir / f"vol_{i:04d}.nii.gz")

        writer = threading.Thread(target=write_new_subfolder)
        writer.start()

        result = collect_volumes(
            watch_dir=watch_dir,
            output_dir=output_dir,
            n_volumes=3,
            clear_watch_dir=False,  # Don't clear — old files should be skipped via pre-cataloging
            timeout=15.0,
        )
        writer.join()

        # Should have collected exactly 3 (from new subfolder), not 8 (5 old + 3 new)
        collected = [f for f in result.iterdir() if f.is_file()]
        assert len(collected) == 3
