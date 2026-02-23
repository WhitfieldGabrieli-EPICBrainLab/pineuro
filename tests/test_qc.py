"""Tests for pineuro.qc — QC report generation and plotting utilities."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing


import nibabel as nib
import numpy as np

# =============================================================================
# summarize_mask
# =============================================================================


class TestSummarizeMask:
    def test_known_mask(self, tmp_path):
        from pineuro.qc import summarize_mask

        shape = (10, 10, 10)
        mask_data = np.zeros(shape, dtype=np.float32)
        mask_data[3:7, 3:7, 3:7] = 1.0  # 4x4x4 = 64 voxels

        mask_path = tmp_path / "mask.nii.gz"
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2mm voxels
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        nib.save(nib.Nifti1Image(mask_data, affine), mask_path)

        summary = summarize_mask(mask_path, "TestMask")
        assert summary.name == "TestMask"
        assert summary.voxels == 64
        # Volume should be 64 voxels * 8mm³/voxel = 512mm³ = 0.512 mL
        assert abs(summary.volume_ml - 0.512) < 0.01

    def test_empty_mask(self, tmp_path):
        from pineuro.qc import summarize_mask

        mask_data = np.zeros((5, 5, 5), dtype=np.float32)
        mask_path = tmp_path / "empty_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

        summary = summarize_mask(mask_path, "Empty")
        assert summary.voxels == 0


# =============================================================================
# Plot functions (verify file creation, not visual correctness)
# =============================================================================


class TestPlotFunctions:
    def test_plot_motion_timeseries(self, sample_motion_par, tmp_path):
        from pineuro.qc import plot_motion_timeseries

        par_path = sample_motion_par(n_volumes=50)
        output = tmp_path / "motion.png"

        plot_motion_timeseries(par_path, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_motion_framewise_displacement(self, sample_motion_par, tmp_path):
        from pineuro.qc import plot_motion_framewise_displacement

        par_path = sample_motion_par(n_volumes=50)
        output = tmp_path / "fd.png"

        plot_motion_framewise_displacement(par_path, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_mask_overlay_tripanel(self, tmp_path):
        from pineuro.qc import plot_mask_overlay_tripanel

        shape = (20, 20, 20)
        ref_data = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
        mask_data = np.zeros(shape, dtype=np.float32)
        mask_data[8:12, 8:12, 8:12] = 1.0

        ref_path = tmp_path / "ref.nii.gz"
        mask_path = tmp_path / "mask.nii.gz"
        output = tmp_path / "overlay.png"

        nib.save(nib.Nifti1Image(ref_data, np.eye(4)), ref_path)
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

        slices = {"axial": [10], "sagittal": [10], "coronal": [10]}
        plot_mask_overlay_tripanel(mask_path, ref_path, output, slices=slices)
        assert output.exists()

    def test_plot_activation_timeseries(self, sample_feedbacklog, tmp_path):
        from pineuro.qc import plot_activation_timeseries

        log_path = sample_feedbacklog(n_volumes=50, mask_names=["DMN"])
        output = tmp_path / "activation.png"

        # Signature: (feedback_log_path, out_path, n_baseline=None, title=None)
        plot_activation_timeseries(log_path, output)
        assert output.exists()

    def test_plot_fd_timeseries(self, sample_feedbacklog, tmp_path):
        from pineuro.qc import plot_fd_timeseries

        log_path = sample_feedbacklog(n_volumes=50)
        output = tmp_path / "fd_feedback.png"

        plot_fd_timeseries(log_path, output)
        assert output.exists()
