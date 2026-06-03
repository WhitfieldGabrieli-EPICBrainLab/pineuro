"""Tests for pineuro.collection_qc — real-time QC monitoring during collection."""

import numpy as np

from pineuro.collection_qc import CollectionQCMonitor, _extract_motion_from_dicom_header

# FD and DVARS math is exercised in tests/test_utils.py (the shared
# utils.compute_fd / compute_dvars that collection_qc now delegates to).

# =============================================================================
# _extract_motion_from_dicom_header
# =============================================================================


class TestExtractMotionFromDicomHeader:
    def test_no_image_comments(self):
        from unittest.mock import MagicMock

        dcm = MagicMock(spec=[])  # No attributes
        assert _extract_motion_from_dicom_header(dcm) is None

    def test_reference_volume(self):
        from unittest.mock import MagicMock

        dcm = MagicMock()
        dcm.ImageComments = "Reference volume for motion correction"
        result = _extract_motion_from_dicom_header(dcm)
        assert result is not None
        np.testing.assert_array_equal(result, np.zeros(6))

    def test_motion_params_parsed(self):
        from unittest.mock import MagicMock

        dcm = MagicMock()
        dcm.ImageComments = "Motion: 0.1,-0.2,0.3,1.0,-2.0,3.0"
        result = _extract_motion_from_dicom_header(dcm)
        assert result is not None
        np.testing.assert_allclose(result, [0.1, -0.2, 0.3, 1.0, -2.0, 3.0])

    def test_no_motion_pattern(self):
        from unittest.mock import MagicMock

        dcm = MagicMock()
        dcm.ImageComments = "Some other comment"
        assert _extract_motion_from_dicom_header(dcm) is None


# =============================================================================
# CollectionQCMonitor
# =============================================================================


class TestCollectionQCMonitor:
    def test_tsv_header_written(self, tmp_path):
        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path)
        monitor._ensure_csv()
        monitor.close()

        content = log_path.read_text()
        assert "volume" in content
        assert "fd" in content
        assert "dvars" in content
        assert "tsnr" in content

    def test_process_nifti_volume(self, tmp_path):
        import nibabel as nib

        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path, motion_monitoring="none")

        # Create a simple NIfTI volume
        vol_path = tmp_path / "vol_001.nii.gz"
        data = np.random.default_rng(42).standard_normal((4, 4, 4)).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), vol_path)

        result = monitor.process_volume(vol_path, volume_num=1)
        monitor.close()

        assert "fd" in result
        assert "dvars" in result
        assert "tsnr" in result
        assert result["fd"] == 0.0  # No motion correction
        assert result["dvars"] == 0.0  # First volume

    def test_dvars_nonzero_after_two_volumes(self, tmp_path):
        import nibabel as nib

        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path, motion_monitoring="none")

        rng = np.random.default_rng(42)
        for i in range(2):
            vol_path = tmp_path / f"vol_{i:03d}.nii.gz"
            data = rng.standard_normal((4, 4, 4)).astype(np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), vol_path)
            result = monitor.process_volume(vol_path, volume_num=i + 1)

        monitor.close()
        assert result["dvars"] > 0  # Should be nonzero for second volume

    def test_callback_invoked(self, tmp_path):
        import nibabel as nib

        callback_args = []

        def on_qc(vol, fd, dvars, tsnr, slice_img):
            callback_args.append((vol, fd, dvars, tsnr))

        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path, motion_monitoring="none", on_qc_data=on_qc)

        vol_path = tmp_path / "vol_001.nii.gz"
        data = np.ones((4, 4, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), vol_path)

        monitor.process_volume(vol_path, volume_num=1)
        monitor.close()

        assert len(callback_args) == 1
        assert callback_args[0][0] == 1  # volume_num

    def test_tsnr_increases_with_volumes(self, tmp_path):
        import nibabel as nib

        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path, motion_monitoring="none")

        rng = np.random.default_rng(42)
        tsnr_vals = []
        for i in range(5):
            vol_path = tmp_path / f"vol_{i:03d}.nii.gz"
            # Stable signal + small noise → high tSNR
            data = (100.0 + rng.standard_normal((4, 4, 4)) * 1.0).astype(np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), vol_path)
            result = monitor.process_volume(vol_path, volume_num=i + 1)
            tsnr_vals.append(result["tsnr"])

        monitor.close()
        # tSNR should become positive after first 2 volumes
        assert tsnr_vals[-1] > 0

    def test_close_idempotent(self, tmp_path):
        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path)
        monitor._ensure_csv()
        monitor.close()
        monitor.close()  # Should not raise

    def test_brain_voxels_gate_excludes_background(self, tmp_path):
        # tSNR/DVARS must measure tissue, not a FOV-weighted blend with air. The intensity
        # gate keeps the bright (brain) voxels and drops the ~0 background.
        monitor = CollectionQCMonitor(tmp_path / "log.tsv")
        voxels = np.concatenate([np.full(200, 1.0), np.full(200, 1000.0)])
        brain = monitor._brain_voxels(voxels)
        assert len(brain) == 200
        assert np.all(brain > 100.0)
        # The mask is fixed after the first volume — a later volume reuses the same indices
        # (required so Welford accumulates a consistent voxel set).
        voxels2 = np.concatenate([np.full(200, 5.0), np.full(200, 900.0)])
        brain2 = monitor._brain_voxels(voxels2)
        assert np.all(brain2 == 900.0)

    def test_brain_voxels_falls_back_when_too_few(self, tmp_path):
        # Degenerate/tiny volume: gate would keep <100 voxels, so use all of them.
        monitor = CollectionQCMonitor(tmp_path / "log.tsv")
        voxels = np.array([0.0, 0.0, 1000.0, 1000.0])  # gate keeps only 2 (<100)
        assert len(monitor._brain_voxels(voxels)) == 4

    def test_tsnr_held_until_two_volumes(self, tmp_path):
        import nibabel as nib

        monitor = CollectionQCMonitor(tmp_path / "log.tsv", motion_monitoring="none")
        rng = np.random.default_rng(1)
        shape = (6, 6, 6)

        def _save(name, data):
            p = tmp_path / name
            nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), p)
            return p

        r1 = monitor.process_volume(_save("a.nii.gz", 1000.0 + rng.standard_normal(shape) * 2.0), 1)
        assert r1["tsnr"] == 0.0  # one volume — below the 2-volume variance floor
        r2 = monitor.process_volume(_save("b.nii.gz", 1000.0 + rng.standard_normal(shape) * 2.0), 2)
        assert r2["tsnr"] > 0  # two volumes — reported

    def test_motion_spike_is_censored_from_tsnr(self, tmp_path):
        import nibabel as nib

        monitor = CollectionQCMonitor(tmp_path / "log.tsv", motion_monitoring="none", dvars_threshold=1.5)
        rng = np.random.default_rng(0)
        shape = (6, 6, 6)

        def _save(name, data):
            p = tmp_path / name
            nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), p)
            return p

        # Six stable volumes accumulate a clean tSNR estimate.
        tsnr_before = 0.0
        for i in range(6):
            data = 1000.0 + rng.standard_normal(shape) * 2.0
            tsnr_before = monitor.process_volume(_save(f"v{i:03d}.nii.gz", data), i + 1)["tsnr"]
        assert monitor._welford_count == 6

        # A large intensity jump -> standardized DVARS >> 1.5 -> volume is censored: the
        # running tSNR is held and the Welford state is untouched (no permanent dent).
        spike = 1000.0 + rng.standard_normal(shape) * 2.0 + 200.0
        r_spike = monitor.process_volume(_save("spike.nii.gz", spike), 7)
        assert r_spike["dvars"] > 1.5
        assert r_spike["tsnr"] == tsnr_before
        assert monitor._welford_count == 6  # spike did not enter the variance estimate
