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
# save/load mask selections roundtrip
#
# MaskSelection is a plain dataclass; its field round-trip (including the optional
# secondary_* fields) is exercised by TestMaskSelectionsIO below, so no separate
# attribute-storage test is kept.
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
            # A combined selection exercises the optional secondary_* fields through
            # real save/load serialization (the field round-trip, not attr storage).
            "ContA": MaskSelection(
                network_name="ContA",
                source_type="ica_combined",
                component_index=12,
                correlation=0.72,
                selection_reason="manual_combined",
                secondary_component_index=8,
                secondary_correlation=0.6,
            ),
        }

        save_mask_selections(tmp_path, selections)
        loaded = load_mask_selections(tmp_path)

        assert set(loaded.keys()) == {"DefaultA", "ContA"}
        assert loaded["DefaultA"].component_index == 5
        assert loaded["ContA"].correlation == pytest.approx(0.72)
        assert loaded["ContA"].secondary_component_index == 8
        assert loaded["ContA"].secondary_correlation == pytest.approx(0.6)

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
# cluster_extent_threshold
# =============================================================================


class TestClusterExtentThreshold:
    def test_basic_two_clusters(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        # Two clusters, both above min_cluster_size
        stat = np.zeros((20, 20, 20), dtype=np.float64)
        stat[2:6, 2:6, 2:6] = 4.0  # cluster A: 64 voxels
        stat[14:18, 14:18, 14:18] = 3.0  # cluster B: 64 voxels

        mask, info = cluster_extent_threshold(stat, stat_threshold=2.0, min_cluster_size=10)

        assert info["outcome"] == "success"
        assert info["n_clusters_surviving"] == 2
        assert mask.sum() == 128  # both clusters kept

    def test_filters_small_clusters(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.zeros((20, 20, 20), dtype=np.float64)
        stat[2:6, 2:6, 2:6] = 4.0  # 64 voxels — survives
        stat[14:16, 14:16, 14:16] = 3.0  # 8 voxels — too small

        mask, info = cluster_extent_threshold(stat, stat_threshold=2.0, min_cluster_size=20)

        assert info["outcome"] == "success"
        assert info["n_clusters_surviving"] == 1
        assert mask.sum() == 64

    def test_max_total_voxels_cap(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.zeros((30, 10, 10), dtype=np.float64)
        stat[1:5, 1:5, 1:5] = 5.0  # 64 voxels
        stat[10:14, 1:5, 1:5] = 4.0  # 64 voxels
        stat[20:24, 1:5, 1:5] = 3.0  # 64 voxels

        mask, info = cluster_extent_threshold(
            stat,
            stat_threshold=2.0,
            min_cluster_size=10,
            max_total_voxels=100,
        )

        assert info["outcome"] == "capped"
        assert info["capped"] is True
        # Should keep only the largest cluster (64 voxels fits in 100)
        assert mask.sum() <= 100

    def test_atlas_intersection(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.zeros((20, 20, 20), dtype=np.float64)
        stat[2:8, 2:8, 2:8] = 4.0  # 216 voxels

        # Atlas covers only part of the cluster
        atlas = np.zeros((20, 20, 20), dtype=np.float64)
        atlas[4:8, 4:8, 4:8] = 1.0  # 64 voxels overlap

        mask, info = cluster_extent_threshold(
            stat,
            stat_threshold=2.0,
            min_cluster_size=10,
            atlas_data=atlas,
        )

        assert info["outcome"] == "success"
        assert mask.sum() == 64  # only the intersection
        assert info["atlas_voxel_count"] == 64

    def test_no_suprathreshold_voxels(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.ones((10, 10, 10), dtype=np.float64)  # all values = 1.0

        mask, info = cluster_extent_threshold(stat, stat_threshold=5.0, min_cluster_size=10)

        assert mask.sum() == 0
        assert info["outcome"] == "no_suprathreshold_voxels"

    def test_no_clusters_survive_filtering(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        # Many small clusters, all below min_cluster_size
        stat = np.zeros((20, 20, 20), dtype=np.float64)
        stat[1, 1, 1] = 5.0  # 1-voxel "cluster"
        stat[10, 10, 10] = 5.0  # another 1-voxel "cluster"

        mask, info = cluster_extent_threshold(stat, stat_threshold=2.0, min_cluster_size=10)

        assert mask.sum() == 0
        assert info["outcome"] == "no_clusters_survive"
        assert info["n_clusters_input"] == 2

    def test_shape_mismatch_raises(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.zeros((10, 10, 10), dtype=np.float64)
        atlas = np.zeros((5, 5, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="Shape mismatch"):
            cluster_extent_threshold(stat, atlas_data=atlas)


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
                "ses-01",
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


# =============================================================================
# import_manual_masks (user-supplied masks)
# =============================================================================


class TestImportManualMasks:
    def test_native_import(self, sample_mask, synthetic_nifti_3d, tmp_path):
        """Native-space masks are binarised and copied to 03_masks/ with a manual selections JSON."""
        import json

        from pineuro.mask_extraction import import_manual_masks

        m1 = sample_mask(shape=(8, 8, 8), name="dmn.nii.gz")
        m2 = sample_mask(shape=(8, 8, 8), name="fpn.nii.gz")
        ref = synthetic_nifti_3d(shape=(8, 8, 8), name="ref.nii.gz")
        root = tmp_path / "mask_extraction"

        out = import_manual_masks(
            participant_id="sub-01",
            session_id="ses-01",
            root_dir=root,
            mask_paths=[m1, m2],
            mask_names=["DMN", "FPN"],
            ref_file=ref,
            mask_space="native",
        )

        assert len(out) == 2
        for p in out:
            assert p.exists()
            assert p.parent == root / "03_masks"
            data = np.asarray(nib.loadsave.load(p).get_fdata())
            assert set(np.unique(data)).issubset({0.0, 1.0})

        sel = json.loads((root / "03_masks" / "mask_selections.json").read_text())
        assert sel["type"] == "manual"
        # Per-mask dicts live at the top level so qc.py / the review dialog detect them
        assert sel["DMN"]["source_type"] == "manual_import"
        assert sel["FPN"]["source_type"] == "manual_import"

    def test_native_resamples_on_shape_mismatch(self, sample_mask, synthetic_nifti_3d, tmp_path):
        """A native mask whose grid differs from the reference is resampled to the reference grid."""
        from pineuro.mask_extraction import import_manual_masks

        mask = sample_mask(shape=(6, 6, 6), name="roi.nii.gz")
        ref = synthetic_nifti_3d(shape=(8, 8, 8), name="ref.nii.gz")

        out = import_manual_masks(
            participant_id="sub-01",
            session_id="ses-01",
            root_dir=tmp_path / "mask_extraction",
            mask_paths=[mask],
            mask_names=["ROI"],
            ref_file=ref,
            mask_space="native",
        )

        assert nib.loadsave.load(out[0]).shape[:3] == (8, 8, 8)

    def test_empty_paths_raises(self, synthetic_nifti_3d, tmp_path):
        from pineuro.mask_extraction import import_manual_masks

        ref = synthetic_nifti_3d(shape=(6, 6, 6), name="ref.nii.gz")
        with pytest.raises(ValueError, match="at least one"):
            import_manual_masks("sub-01", "ses-01", tmp_path, [], [], ref, mask_space="native")

    def test_duplicate_names_raises(self, sample_mask, synthetic_nifti_3d, tmp_path):
        from pineuro.mask_extraction import import_manual_masks

        m1 = sample_mask(shape=(6, 6, 6), name="a.nii.gz")
        m2 = sample_mask(shape=(6, 6, 6), name="b.nii.gz")
        ref = synthetic_nifti_3d(shape=(6, 6, 6), name="ref.nii.gz")
        with pytest.raises(ValueError, match="unique"):
            import_manual_masks("sub-01", "ses-01", tmp_path, [m1, m2], ["X", "X"], ref, mask_space="native")

    def test_anat_requires_anat_file(self, sample_mask, synthetic_nifti_3d, tmp_path):
        from pineuro.mask_extraction import import_manual_masks

        mask = sample_mask(shape=(6, 6, 6), name="a.nii.gz")
        ref = synthetic_nifti_3d(shape=(6, 6, 6), name="ref.nii.gz")
        with pytest.raises(ValueError, match="anat_file"):
            import_manual_masks("sub-01", "ses-01", tmp_path, [mask], ["A"], ref, mask_space="anat")

    def test_qc_report_invoked_when_enabled(self, sample_mask, synthetic_nifti_3d, tmp_path):
        """qc_enabled routes through generate_mask_extraction_report with the imported mask names."""
        from pineuro.mask_extraction import import_manual_masks

        mask = sample_mask(shape=(8, 8, 8), name="a.nii.gz")
        ref = synthetic_nifti_3d(shape=(8, 8, 8), name="ref.nii.gz")
        root = tmp_path / "mask_extraction"

        with patch("pineuro.qc.generate_mask_extraction_report") as mock_qc:
            import_manual_masks("sub-01", "ses-01", root, [mask], ["A"], ref, mask_space="native", qc_enabled=True)

        mock_qc.assert_called_once()
        kwargs = mock_qc.call_args.kwargs
        assert kwargs["participant_id"] == "sub-01"
        assert kwargs["network_names"] == ["A"]

    def test_qc_failure_does_not_abort_import(self, sample_mask, synthetic_nifti_3d, tmp_path):
        """A QC-report exception is swallowed — the import still returns its masks."""
        from pineuro.mask_extraction import import_manual_masks

        mask = sample_mask(shape=(8, 8, 8), name="a.nii.gz")
        ref = synthetic_nifti_3d(shape=(8, 8, 8), name="ref.nii.gz")

        with patch("pineuro.qc.generate_mask_extraction_report", side_effect=RuntimeError("boom")):
            out = import_manual_masks(
                "sub-01",
                "ses-01",
                tmp_path / "mask_extraction",
                [mask],
                ["A"],
                ref,
                mask_space="native",
                qc_enabled=True,
            )

        assert len(out) == 1 and out[0].exists()

    def test_anat_import_applies_registration(self, sample_mask, synthetic_nifti_3d, tmp_path):
        """Anatomical-space masks are registered T1->EPI via ANTs before binarising."""
        from pineuro.mask_extraction import import_manual_masks

        mask = sample_mask(shape=(8, 8, 8), name="roi.nii.gz")
        ref = synthetic_nifti_3d(shape=(8, 8, 8), name="ref.nii.gz")
        anat = synthetic_nifti_3d(shape=(8, 8, 8), name="t1.nii.gz")

        def _fake_apply_xfm(*, moving, fixed, transform_files, output_file, **kwargs):
            nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), output_file)

        with (
            patch(
                "pineuro.mask_extraction.compute_xfm", return_value=(tmp_path / "warped.nii.gz", ["xfm.mat"])
            ) as mock_xfm,
            patch("pineuro.mask_extraction.apply_xfm", side_effect=_fake_apply_xfm) as mock_apply,
        ):
            out = import_manual_masks(
                "sub-01",
                "ses-01",
                tmp_path / "mask_extraction",
                [mask],
                ["ROI"],
                ref,
                mask_space="anat",
                anat_file=anat,
            )

        assert mock_xfm.called
        assert mock_apply.called
        assert out[0].exists()
