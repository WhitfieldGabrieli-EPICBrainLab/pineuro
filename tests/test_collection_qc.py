"""Tests for pineuro.collection_qc — real-time QC monitoring during collection."""

import numpy as np
import pytest

from pineuro.collection_qc import (
    HEAD_RADIUS_MM,
    CollectionQCMonitor,
    _compute_dvars,
    _compute_fd,
    _extract_motion_from_dicom_header,
)

# =============================================================================
# _compute_fd
# =============================================================================


class TestComputeFD:
    def test_first_volume_returns_zero(self):
        motion = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        assert _compute_fd(motion, None) == 0.0

    def test_no_motion_returns_zero(self):
        motion = np.zeros(6)
        prev = np.zeros(6)
        assert _compute_fd(motion, prev) == 0.0

    def test_pure_translation(self):
        motion = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        prev = np.zeros(6)
        fd = _compute_fd(motion, prev)
        # Translation of 1mm in x → FD = 1.0mm
        assert fd == pytest.approx(1.0)

    def test_rotation_uses_head_radius(self):
        # 1 degree rotation in x
        prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        motion_with_rot = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        fd = _compute_fd(motion_with_rot, prev)
        # FD should include arc-length conversion: 1deg * HEAD_RADIUS_MM * pi/180
        expected_arc = HEAD_RADIUS_MM * np.pi / 180.0
        assert fd == pytest.approx(expected_arc)

    def test_combined_translation_and_rotation(self):
        motion = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        prev = np.zeros(6)
        fd = _compute_fd(motion, prev)
        assert fd > 0


# =============================================================================
# _compute_dvars
# =============================================================================


class TestComputeDVARS:
    def test_first_volume_returns_zero(self):
        voxels = np.array([100.0, 200.0, 300.0])
        assert _compute_dvars(voxels, None) == 0.0

    def test_no_change_returns_zero(self):
        voxels = np.array([100.0, 200.0, 300.0])
        assert _compute_dvars(voxels, voxels) == 0.0

    def test_uniform_change(self):
        prev = np.array([100.0, 100.0, 100.0])
        curr = np.array([110.0, 110.0, 110.0])
        dvars = _compute_dvars(curr, prev)
        # RMS of [10, 10, 10] = 10.0
        assert dvars == pytest.approx(10.0)


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
    def test_construction(self, tmp_path):
        log_path = tmp_path / "collectionlog.tsv"
        monitor = CollectionQCMonitor(log_path)
        assert monitor.log_path == log_path
        monitor.close()

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
        monitor = CollectionQCMonitor(log_path, motion_correction="none")

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
        monitor = CollectionQCMonitor(log_path, motion_correction="none")

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
        monitor = CollectionQCMonitor(log_path, motion_correction="none", on_qc_data=on_qc)

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
        monitor = CollectionQCMonitor(log_path, motion_correction="none")

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
