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
    def test_smoothing_changes_values_and_preserves_shape(self, synthetic_nifti_4d, tmp_path):
        input_path = synthetic_nifti_4d(shape=(8, 8, 8), n_volumes=5)
        output_path = tmp_path / "smoothed.nii.gz"

        spatial_smoothing(input_path, output_path, fwhm=5.0)

        assert nib.load(output_path).shape == (8, 8, 8, 5)
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
        """No highpass/lowpass should leave the data numerically unchanged."""
        input_path = synthetic_nifti_4d(shape=(3, 3, 3), n_volumes=20)
        output_path = tmp_path / "filtered.nii.gz"

        temporal_filtering(input_path, output_path, highpass=None, lowpass=None)

        original = nib.load(input_path).get_fdata()
        filtered = nib.load(output_path).get_fdata()
        np.testing.assert_allclose(filtered, original, rtol=0, atol=1e-6)


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
            preprocessing("sub-01", "ses-01", "rest", root_dir, tmp_path / "nonexistent_dicoms")

    def test_validates_empty_dicom_dir(self, tmp_path):
        """preprocessing should raise FileNotFoundError for empty input_dicoms dir."""
        from pineuro.preprocessing import preprocessing

        root_dir = tmp_path / "study"
        root_dir.mkdir()
        empty_dicoms = tmp_path / "dicoms"
        empty_dicoms.mkdir()

        with pytest.raises(FileNotFoundError, match="Directory is empty"):
            preprocessing("sub-01", "ses-01", "rest", root_dir, empty_dicoms)


# =============================================================================
# preprocessing — run/attempt-aware intermediate file naming
# =============================================================================


class TestParseRunAttemptFromPath:
    def test_matches_collection_folder(self):
        from pineuro.preprocessing import _parse_run_attempt_from_path

        assert _parse_run_attempt_from_path(Path("/x/y/task-rest_run-01_attempt-02")) == (1, 2)

    def test_matches_with_extra_path_components(self):
        from pineuro.preprocessing import _parse_run_attempt_from_path

        assert _parse_run_attempt_from_path(Path("/study/collection/task-rest_run-02_attempt-03/dcm")) == (2, 3)

    def test_no_match_returns_nones(self):
        from pineuro.preprocessing import _parse_run_attempt_from_path

        assert _parse_run_attempt_from_path(Path("/external/source_dicoms/run1")) == (None, None)


class TestPreprocessingAttemptAwareNaming:
    """preprocessing() must thread (run, attempt) into intermediate filenames so that
    re-preprocessing a different attempt does not collide, and multi-attempt-per-run
    selections ({1: [2, 3]}) keep their run identity."""

    def _make_dicom_dirs(self, tmp_path, names):
        dirs = []
        for n in names:
            d = tmp_path / "collection" / n
            d.mkdir(parents=True)
            (d / "fake.dcm").touch()
            dirs.append(d)
        return dirs

    def _patch_steps(self, monkeypatch, nifti_dir):
        """Replace heavy steps with thin stubs that just record call paths and
        materialize empty output files so existence checks pass downstream."""

        call_log: dict[str, list[Path]] = {
            "dicom_to_nifti": [],
            "motion_correction": [],
            "compute_mean": [],
            "brain_extraction": [],
            "compute_xfm": [],
            "apply_xfm": [],
            "concat": [],
            "erode_mask": [],
            "apply_mask": [],
        }

        def fake_dicom_to_nifti(dicom_dir, output_dir, output_name, overwrite=False, **_kw):
            out = Path(output_dir) / f"{output_name}.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"x")
            call_log["dicom_to_nifti"].append(out)
            return out

        def fake_motion_correction(input_file, output_file, **_kw):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"x")
            Path(str(output_file) + ".par").write_text("0 0 0 0 0 0\n")
            call_log["motion_correction"].append(Path(output_file))

        def fake_compute_mean(input_file, output_file, **_kw):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"x")
            call_log["compute_mean"].append(Path(output_file))

        def fake_brain_extraction(input_file, output_dir, **_kw):
            stem = Path(input_file).with_suffix("").stem
            brain = Path(output_dir) / f"{stem}_brain.nii.gz"
            mask = Path(output_dir) / f"{stem}_brain_mask.nii.gz"
            brain.write_bytes(b"x")
            mask.write_bytes(b"x")
            call_log["brain_extraction"].append(mask)
            return brain, mask

        def fake_compute_xfm(moving, fixed, output_prefix, **_kw):
            Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
            xfm = Path(str(output_prefix) + "0GenericAffine.mat")
            xfm.write_text("xfm")
            call_log["compute_xfm"].append(Path(output_prefix))
            return None, [xfm]

        def fake_apply_xfm(moving, fixed, transform_files, output_file, **_kw):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"x")
            call_log["apply_xfm"].append(Path(output_file))

        def fake_concat(input_files, output_file, **_kw):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"x")
            call_log["concat"].append(Path(output_file))

        def fake_erode_mask(mask, output, **_kw):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_bytes(b"x")
            call_log["erode_mask"].append(Path(output))

        def fake_apply_mask(input_file, mask_file, output_file, **_kw):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"x")
            call_log["apply_mask"].append(Path(output_file))

        import pineuro.preprocessing as pp

        monkeypatch.setattr(pp, "dicom_to_nifti", fake_dicom_to_nifti)
        monkeypatch.setattr(pp, "motion_correction", fake_motion_correction)
        monkeypatch.setattr(pp, "compute_mean", fake_compute_mean)
        monkeypatch.setattr(pp, "brain_extraction", fake_brain_extraction)
        monkeypatch.setattr(pp, "compute_xfm", fake_compute_xfm)
        monkeypatch.setattr(pp, "apply_xfm", fake_apply_xfm)
        monkeypatch.setattr(pp, "concat", fake_concat)
        monkeypatch.setattr(pp, "erode_mask", fake_erode_mask)
        monkeypatch.setattr(pp, "apply_mask", fake_apply_mask)

        return call_log

    def test_intermediates_carry_attempt_when_parsed_from_folder(self, tmp_path, monkeypatch):
        from pineuro.preprocessing import preprocessing

        dirs = self._make_dicom_dirs(tmp_path, ["task-rest_run-01_attempt-03", "task-rest_run-02_attempt-01"])
        root_dir = tmp_path / "preproc"
        root_dir.mkdir()
        log = self._patch_steps(monkeypatch, root_dir / "01_nifti")

        preprocessing(
            "sub-01",
            "ses-01",
            "rest",
            root_dir,
            [str(d) for d in dirs],
            n_jobs=1,  # force sequential path
            qc_enabled=False,
        )

        nifti_names = sorted(p.name for p in log["dicom_to_nifti"])
        assert nifti_names == [
            "sub-01_ses-01_task-rest_run-01_attempt-03.nii.gz",
            "sub-01_ses-01_task-rest_run-02_attempt-01.nii.gz",
        ]
        mcf_names = sorted(p.name for p in log["motion_correction"])
        assert mcf_names == [
            "sub-01_ses-01_task-rest_run-01_attempt-03_mcf.nii.gz",
            "sub-01_ses-01_task-rest_run-02_attempt-01_mcf.nii.gz",
        ]
        mean_names = sorted(p.name for p in log["compute_mean"])
        assert mean_names == [
            "sub-01_ses-01_task-rest_run-01_attempt-03_mean.nii.gz",
            "sub-01_ses-01_task-rest_run-02_attempt-01_mean.nii.gz",
        ]

    def test_multi_attempt_per_run_keeps_run_identity(self, tmp_path, monkeypatch):
        """{1: [2, 3]} selection: two inputs at run-01 with different attempts
        must produce run-01_attempt-02 and run-01_attempt-03, NOT run-01/run-02."""
        from pineuro.preprocessing import preprocessing

        dirs = self._make_dicom_dirs(tmp_path, ["task-rest_run-01_attempt-02", "task-rest_run-01_attempt-03"])
        root_dir = tmp_path / "preproc"
        root_dir.mkdir()
        log = self._patch_steps(monkeypatch, root_dir / "01_nifti")

        preprocessing(
            "sub-01",
            "ses-01",
            "rest",
            root_dir,
            [str(d) for d in dirs],
            n_jobs=1,
            qc_enabled=False,
            attempt_per_input=[2, 3],
            run_per_input=[1, 1],
        )

        nifti_names = sorted(p.name for p in log["dicom_to_nifti"])
        assert nifti_names == [
            "sub-01_ses-01_task-rest_run-01_attempt-02.nii.gz",
            "sub-01_ses-01_task-rest_run-01_attempt-03.nii.gz",
        ]
        # No "run-02" should appear in any per-input artifact
        all_names = [
            p.name for key in ("dicom_to_nifti", "motion_correction", "compute_mean", "apply_xfm") for p in log[key]
        ]
        assert not any("run-02" in n for n in all_names), all_names

    def test_external_paths_fall_back_to_enumeration(self, tmp_path, monkeypatch):
        """Folders that don't match the run-NN_attempt-NN convention default to
        attempt=1 and run=enumerate_index+1 (covers skip_collection paths)."""
        from pineuro.preprocessing import preprocessing

        dirs = self._make_dicom_dirs(tmp_path, ["raw_run_one", "raw_run_two"])
        root_dir = tmp_path / "preproc"
        root_dir.mkdir()
        log = self._patch_steps(monkeypatch, root_dir / "01_nifti")

        preprocessing(
            "sub-01",
            "ses-01",
            "rest",
            root_dir,
            [str(d) for d in dirs],
            n_jobs=1,
            qc_enabled=False,
        )

        nifti_names = sorted(p.name for p in log["dicom_to_nifti"])
        assert nifti_names == [
            "sub-01_ses-01_task-rest_run-01_attempt-01.nii.gz",
            "sub-01_ses-01_task-rest_run-02_attempt-01.nii.gz",
        ]

    def test_explicit_lengths_must_match(self, tmp_path, monkeypatch):
        from pineuro.preprocessing import preprocessing

        dirs = self._make_dicom_dirs(tmp_path, ["task-rest_run-01_attempt-01"])
        root_dir = tmp_path / "preproc"
        root_dir.mkdir()
        self._patch_steps(monkeypatch, root_dir / "01_nifti")

        with pytest.raises(ValueError, match="attempt_per_input has 2 entries"):
            preprocessing(
                "sub-01",
                "ses-01",
                "rest",
                root_dir,
                [str(d) for d in dirs],
                n_jobs=1,
                qc_enabled=False,
                attempt_per_input=[1, 2],
            )


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
        )
        writer.join()

        # Should have collected exactly 3 (from new subfolder), not 8 (5 old + 3 new)
        collected = [f for f in result.iterdir() if f.is_file()]
        assert len(collected) == 3
