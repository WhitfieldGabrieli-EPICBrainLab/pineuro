"""Tests for pineuro.mask_extraction — ICA, GLM, atlas, and manual mask workflows."""

from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

from pineuro.mask_extraction import (
    MaskSelection,
    load_mask_selections,
    save_mask_selections,
)

# =============================================================================
# MaskSelection dataclass
# =============================================================================


class TestMaskSelection:
    def test_construction(self):
        sel = MaskSelection(
            network_name="DefaultA",
            source_type="ica",
            component_index=5,
            correlation=0.85,
            selection_reason="auto",
        )
        assert sel.network_name == "DefaultA"
        assert sel.component_index == 5
        assert sel.correlation == 0.85

    def test_optional_fields(self):
        sel = MaskSelection(
            network_name="ContA",
            source_type="ica_combined",
            component_index=3,
            correlation=0.7,
            selection_reason="manual_combined",
            secondary_component_index=8,
            secondary_correlation=0.6,
        )
        assert sel.secondary_component_index == 8
        assert sel.secondary_correlation == 0.6


# =============================================================================
# save/load mask selections roundtrip
# =============================================================================


class TestMaskSelectionsIO:
    def test_roundtrip(self, tmp_path):
        selections = {
            "DefaultA": MaskSelection(
                network_name="DefaultA",
                source_type="ica",
                component_index=5,
                correlation=0.85,
                selection_reason="auto",
            ),
            "ContA": MaskSelection(
                network_name="ContA",
                source_type="ica",
                component_index=12,
                correlation=0.72,
                selection_reason="auto",
            ),
        }

        save_mask_selections(tmp_path, selections)
        loaded = load_mask_selections(tmp_path)

        assert set(loaded.keys()) == {"DefaultA", "ContA"}
        assert loaded["DefaultA"].component_index == 5
        assert loaded["ContA"].correlation == pytest.approx(0.72)

    def test_empty_selections(self, tmp_path):
        save_mask_selections(tmp_path, {})
        loaded = load_mask_selections(tmp_path)
        assert loaded == {}


# =============================================================================
# spatial_correlation
# =============================================================================


class TestSpatialCorrelation:
    def test_identity_correlation(self, tmp_path):
        """Identical component and parcellation should have high correlation."""
        from pineuro.mask_extraction import spatial_correlation

        shape = (10, 10, 10)
        affine = np.eye(4)

        # Create a component map
        comp_data = np.zeros((*shape, 2), dtype=np.float32)
        comp_data[2:5, 2:5, 2:5, 0] = 1.0  # component 0 in a cube
        comp_data[6:9, 6:9, 6:9, 1] = 1.0  # component 1 in another cube
        comp_file = tmp_path / "components.nii.gz"
        nib.save(nib.Nifti1Image(comp_data, affine), comp_file)

        # Parcellation that matches component 0
        parc_data = np.zeros(shape, dtype=np.float32)
        parc_data[2:5, 2:5, 2:5] = 1  # label 1 matches component 0
        parc_data[6:9, 6:9, 6:9] = 2  # label 2 matches component 1
        parc_file = tmp_path / "parcellation.nii.gz"
        nib.save(nib.Nifti1Image(parc_data, affine), parc_file)

        # Mask covering the whole brain
        mask_data = np.ones(shape, dtype=np.float32)
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask_data, affine), mask_file)

        output_file = tmp_path / "corr.tsv"
        result = spatial_correlation(
            comp_file,
            parc_file,
            mask_file,
            labels=[1, 2],
            output_file=output_file,
        )

        assert result.shape[0] == 2  # 2 components
        assert result.shape[1] == 2  # 2 labels
        # Component 0 should correlate best with label 1
        assert result.iloc[0, 0] > result.iloc[0, 1]


# =============================================================================
# cluster_threshold_stat_map
# =============================================================================


class TestClusterThreshold:
    def test_basic_thresholding(self):
        from pineuro.mask_extraction import cluster_threshold_stat_map

        # Create stat map with one clear cluster
        stat = np.zeros((20, 20, 20), dtype=np.float64)
        stat[8:12, 8:12, 8:12] = 5.0  # strong cluster of 64 voxels

        # Atlas ROI covering the cluster
        atlas = np.zeros((20, 20, 20), dtype=np.float64)
        atlas[5:15, 5:15, 5:15] = 1.0

        mask, meta = cluster_threshold_stat_map(stat, atlas, target_n_voxels=50, start_threshold=1.0)

        assert mask.sum() > 0
        assert mask.sum() <= 64  # Can't exceed cluster size

    def test_empty_stat_map(self):
        from pineuro.mask_extraction import cluster_threshold_stat_map

        stat = np.zeros((10, 10, 10), dtype=np.float64)
        atlas = np.ones((10, 10, 10), dtype=np.float64)

        mask, meta = cluster_threshold_stat_map(stat, atlas, target_n_voxels=100)
        assert mask.sum() == 0

    def test_return_all_clusters(self):
        """With return_all_clusters=True, all disconnected clusters should be returned."""
        from pineuro.mask_extraction import cluster_threshold_stat_map

        # Two clearly separated clusters (gap of ~16 voxels between them)
        # Each: 4 * 4 * 4 = 64 voxels
        stat = np.zeros((30, 10, 10), dtype=np.float64)
        stat[1:5, 1:5, 1:5] = 5.0  # cluster A: 64 voxels
        stat[25:29, 1:5, 1:5] = 5.0  # cluster B: 64 voxels, far from A

        # Whole-brain atlas covering both
        atlas = np.ones((30, 10, 10), dtype=np.float64)

        # Single-cluster mode (default) — should return only the largest (either A or B)
        mask_single, _ = cluster_threshold_stat_map(stat, atlas, target_n_voxels=70, start_threshold=1.0)
        # Each cluster is 64 voxels; single mode should return exactly one
        assert mask_single.sum() <= 64

        # All-clusters mode — should return both clusters combined
        mask_all, meta_all = cluster_threshold_stat_map(
            stat, atlas, target_n_voxels=130, start_threshold=1.0, return_all_clusters=True
        )
        assert mask_all.sum() == 128  # both clusters (64 + 64)
        assert meta_all["n_clusters_at_final"] == 2

    def test_return_all_clusters_respects_target(self):
        """With return_all_clusters=True the threshold is raised until total fits target."""
        from pineuro.mask_extraction import cluster_threshold_stat_map

        stat = np.zeros((30, 10, 10), dtype=np.float64)
        stat[2:8, 4:6, 4:6] = 5.0  # cluster A: 48 voxels
        stat[22:28, 4:6, 4:6] = 5.0  # cluster B: 48 voxels

        atlas = np.ones((30, 10, 10), dtype=np.float64)

        mask, meta = cluster_threshold_stat_map(
            stat, atlas, target_n_voxels=80, start_threshold=1.0, return_all_clusters=True
        )
        # Total of both clusters (96 voxels) exceeds 80, so threshold should be raised
        # until some voxels are trimmed from the edges; result must be ≤ 80
        assert mask.sum() <= 80


# =============================================================================
# run_ica (mocked nilearn)
# =============================================================================


class TestRunIca:
    def test_calls_canica(self, synthetic_nifti_4d, sample_mask, tmp_path):
        from pineuro.mask_extraction import run_ica

        input_path = synthetic_nifti_4d(shape=(8, 8, 8), n_volumes=30)
        mask_path = sample_mask(shape=(8, 8, 8))
        output_dir = tmp_path / "ica"
        output_dir.mkdir()

        mock_components = np.random.default_rng(42).standard_normal((5, 8, 8, 8)).astype(np.float32)

        # CanICA is imported inside the function, so patch at the source module
        with patch("nilearn.decomposition.CanICA") as mock_canica_cls:
            mock_canica = MagicMock()
            mock_canica_cls.return_value = mock_canica
            mock_canica.components_img_ = nib.Nifti1Image(mock_components, np.eye(4))

            components_file, metadata_file = run_ica(input_path, mask_path, output_dir, "test", n_components=5)

            assert mock_canica.fit.called
            assert components_file.exists()


# =============================================================================
# coregister_reference (mocked ANTsPy)
# =============================================================================


class TestCoregisterReference:
    def test_calls_registration(self, synthetic_nifti_3d, sample_mask, tmp_path):
        from pineuro.mask_extraction import coregister_reference

        # coregister_reference expects input_dicom to be a directory with files
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()
        (dicom_dir / "fake.dcm").touch()

        # Create a fake NIfTI that dicom_to_nifti would produce
        fake_nifti = tmp_path / "reference.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4)), fake_nifti)

        preproc_ref = synthetic_nifti_3d(shape=(8, 8, 8), name="preproc_ref.nii.gz")
        mask_file = sample_mask(shape=(8, 8, 8), name="network_mask.nii.gz")
        root_dir = tmp_path / "study"
        root_dir.mkdir(parents=True)

        with (
            patch("pineuro.mask_extraction.dicom_to_nifti", return_value=fake_nifti) as mock_dcm,
            patch("pineuro.mask_extraction.compute_xfm") as mock_compute,
            patch("pineuro.mask_extraction.apply_xfm"),
        ):
            mock_compute.return_value = (tmp_path / "warped.nii.gz", ["transform.mat"])

            coregister_reference(
                "sub-01",
                root_dir,
                dicom_dir,
                [mask_file],
                preproc_ref,
            )

            assert mock_dcm.called
            assert mock_compute.called


# =============================================================================
# fetch_atlas_mask (mocked nilearn)
# =============================================================================


class TestFetchAtlasMask:
    def test_skips_when_exists(self, tmp_path):
        """fetch_atlas_mask should return early when output already exists."""
        from pineuro.mask_extraction import fetch_atlas_mask

        output_file = tmp_path / "atlas_mask.nii.gz"
        # Create a pre-existing file
        mask_data = np.ones((10, 10, 10), dtype=np.uint8)
        nib.save(nib.Nifti1Image(mask_data, np.eye(4)), output_file)

        # Should return immediately without needing nilearn
        result = fetch_atlas_mask(output_file, atlas_source="harvard_oxford", regions=["Region1"])
        assert result == output_file
