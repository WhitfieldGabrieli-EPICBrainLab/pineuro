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
        # Top-N by stat: exactly the cap, drawn from the highest-stat voxels —
        # all 64 at 5.0 plus 36 of the 64 at 4.0; none of the 3.0 cluster.
        assert mask.sum() == 100
        assert mask[1:5, 1:5, 1:5].sum() == 64  # whole 5.0 cluster kept
        assert mask[20:24, 1:5, 1:5].sum() == 0  # weakest cluster dropped

    def test_cap_single_cluster_larger_than_cap(self):
        """Regression: a single cluster larger than the cap must NOT empty the
        mask. The buggy greedy logic broke on the first (largest) cluster and
        returned zero voxels while reporting outcome='capped'."""
        from pineuro.mask_extraction import cluster_extent_threshold

        # One contiguous cluster of 1000 voxels with distinct, graded stat values
        stat = np.arange(1000, dtype=np.float64).reshape(10, 10, 10) / 100.0 + 3.0

        mask, info = cluster_extent_threshold(
            stat,
            stat_threshold=2.0,
            min_cluster_size=10,
            max_total_voxels=200,
        )

        assert info["outcome"] == "capped"
        assert info["capped"] is True
        assert mask.sum() == 200  # exactly the cap, NOT empty
        # The kept voxels are the 200 highest-stat ones
        kept_min = stat[mask.astype(bool)].min()
        assert kept_min == np.partition(stat.ravel(), -200)[-200]

    def test_cap_disabled_keeps_all_surviving(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.zeros((10, 10, 10), dtype=np.float64)
        stat[2:6, 2:6, 2:6] = 4.0  # 64 voxels

        # Cap larger than surviving voxels => no capping
        mask, info = cluster_extent_threshold(
            stat,
            stat_threshold=2.0,
            min_cluster_size=10,
            max_total_voxels=1000,
        )

        assert info["outcome"] == "success"
        assert info["capped"] is False
        assert mask.sum() == 64

    def test_cap_non_positive_raises(self):
        from pineuro.mask_extraction import cluster_extent_threshold

        stat = np.zeros((10, 10, 10), dtype=np.float64)
        stat[2:6, 2:6, 2:6] = 4.0

        with pytest.raises(ValueError, match="max_total_voxels must be a positive integer"):
            cluster_extent_threshold(stat, max_total_voxels=0)

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


# =============================================================================
# mask_extraction_task thresholding on cached GLM stat maps
#
# The GLM fit is skipped when every 01_glm stat map exists, so these tests
# drive the thresholding step directly from a synthetic stat map: a 4x4x4 blob
# at t=5 (64 voxels), a 2x2x2 blob at t=3 (8 voxels), and weak t=0.5 voxels.
# =============================================================================

_PREFIX = "sub-01_ses-01"


def _task_tree(tmp_path):
    """Build a mask-extraction root with a cached stat map; return the kwargs for mask_extraction_task."""
    import pandas as pd

    shape = (12, 12, 12)
    affine = np.eye(4)
    stat = np.zeros(shape)
    stat[1:5, 1:5, 1:5] = 5.0  # big blob: 64 voxels
    stat[8:10, 8:10, 8:10] = 3.0  # small blob: 8 voxels
    stat[0, 11, 0] = 0.5
    stat[11, 0, 11] = 0.5

    root = tmp_path / "mask_extraction"
    (root / "01_glm").mkdir(parents=True)
    nib.save(nib.Nifti1Image(stat, affine), root / "01_glm" / f"{_PREFIX}_net_tmap.nii.gz")

    func = tmp_path / "func.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((*shape, 3)), affine), func)
    brain = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), brain)
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), atlas)

    params = {
        "participant_id": "sub-01",
        "session_id": "ses-01",
        "root_dir": root,
        "input_file": func,
        "mask_file": brain,
        "events": pd.DataFrame({"onset": [0.0, 10.0], "duration": [10.0, 10.0], "trial_type": ["a", "b"]}),
        "contrasts": {"net": "a - b"},
        "tr": 1.0,
        "qc_enabled": False,
    }
    return params, stat, atlas


def _thr(root):
    return np.asarray(nib.load(root / "03_masks" / f"{_PREFIX}_net_thr.nii.gz").get_fdata())


def _selection(root):
    import json

    return json.loads((root / "03_masks" / "mask_selections.json").read_text())["selections"]["net"]


class TestMaskExtractionTaskThresholding:
    def test_top_n_keeps_highest_voxels(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, threshold=10, threshold_method="top_n")

        mask = _thr(params["root_dir"])
        assert mask.sum() == 10
        assert np.all(stat[mask > 0] == 5.0)
        sel = _selection(params["root_dir"])
        assert sel["threshold_method"] == "top_n"
        assert sel["n_voxels_thresholded"] == 10
        assert sel["n_voxels_above_zero"] == 64 + 8 + 2
        assert sel["peak_stat"] == 5.0

    def test_cluster_extent_drops_small_clusters(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        mask_extraction_task(
            **params, threshold_method="cluster_extent", cluster_start_threshold=2.0, min_cluster_size=20
        )

        mask = _thr(params["root_dir"])
        assert mask.sum() == 64
        assert np.all(stat[mask > 0] == 5.0)
        sel = _selection(params["root_dir"])
        assert sel["cluster_outcome"] == "success"
        assert sel["n_clusters_found"] == 1
        assert sel["final_cluster_threshold"] == 2.0
        assert sel["min_cluster_size"] == 20

    def test_cluster_extent_no_cluster_survives(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, threshold_method="cluster_extent", min_cluster_size=100)

        assert _thr(params["root_dir"]).sum() == 0
        sel = _selection(params["root_dir"])
        assert sel["cluster_outcome"] == "no_clusters_survive"
        assert sel["n_voxels_thresholded"] == 0
        assert sel["n_clusters_found"] == 0

    def test_cluster_extent_cap_keeps_top_voxels(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, threshold_method="cluster_extent", min_cluster_size=5, max_total_voxels=64)

        mask = _thr(params["root_dir"])
        assert mask.sum() == 64
        assert np.all(stat[mask > 0] == 5.0)  # the cap keeps the highest-stat voxels
        assert _selection(params["root_dir"])["cluster_outcome"] == "capped"

    def test_cluster_returns_largest_cluster_within_atlas(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, atlas = _task_tree(tmp_path)
        mask_extraction_task(**params, threshold=100, threshold_method="cluster", atlas_mask=atlas)

        mask = _thr(params["root_dir"])
        assert mask.sum() == 64
        assert np.all(stat[mask > 0] == 5.0)
        sel = _selection(params["root_dir"])
        assert sel["threshold_method"] == "cluster"
        assert sel["atlas_used"] == str(atlas)
        # A user atlas keeps its file name (extension included) in the saved QC copy
        assert (params["root_dir"] / "03_masks" / f"{_PREFIX}_atlas_atlas.nii.gz.nii.gz").exists()

    def test_cluster_extent_records_clusters_before_size_filter(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, threshold_method="cluster_extent", min_cluster_size=100)

        sel = _selection(params["root_dir"])
        assert sel["n_clusters_input"] == 2  # both blobs pass t >= 2.0
        assert sel["largest_cluster_input"] == 64
        assert sel["threshold_params"] == {
            "threshold_method": "cluster_extent",
            "cluster_start_threshold": 2.0,
            "min_cluster_size": 100,
            "max_total_voxels": None,
        }
        assert sel["selection_reason"] == "auto"


def _write_like(stat_shape):
    """apply_xfm stand-in that writes an empty image at ``output_file``."""

    def _write(output_file, **_kw):
        nib.save(nib.Nifti1Image(np.zeros(stat_shape), np.eye(4)), output_file)

    return _write


def _harvard_oxford(stat_shape):
    return MagicMock(maps=nib.Nifti1Image(np.zeros(stat_shape), np.eye(4)), labels=["Background"])


class TestTaskMaskOverrides:
    def test_rethreshold_override_applies_and_is_recorded(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(
            **params,
            threshold_method="cluster_extent",
            min_cluster_size=100,
            mask_overrides={"net": {"kind": "rethreshold", "min_cluster_size": 5}},
        )

        assert _thr(params["root_dir"]).sum() == 64 + 8
        sel = _selection(params["root_dir"])
        assert sel["selection_reason"] == "operator_rethreshold"
        assert sel["min_cluster_size"] == 5
        assert sel["threshold_params"]["min_cluster_size"] == 5

    def test_skip_existing_keeps_recorded_override(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        kwargs = {"threshold_method": "cluster_extent", "min_cluster_size": 100}
        override = {"net": {"kind": "rethreshold", "min_cluster_size": 5}}
        mask_extraction_task(**params, **kwargs, mask_overrides=override)
        mask_extraction_task(**params, **kwargs)  # overwrite=False: the mask on disk is kept

        assert _thr(params["root_dir"]).sum() == 72
        assert _selection(params["root_dir"])["selection_reason"] == "operator_rethreshold"

    def test_unknown_mask_override_is_ignored(self, tmp_path, caplog):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(
            **params,
            threshold_method="cluster_extent",
            min_cluster_size=20,
            mask_overrides={"gone": {"kind": "rethreshold", "min_cluster_size": 5}},
        )

        assert _thr(params["root_dir"]).sum() == 64
        assert any("unknown mask" in m for m in caplog.messages)

    @pytest.mark.parametrize(
        "override",
        [
            {"kind": "shrink"},
            {"kind": "rethreshold"},
            {"kind": "rethreshold", "threshold": 10},  # not a cluster_extent field
            {"kind": "atlas_network", "atlas": "harvard_oxford", "network": "DefaultA"},
            {"kind": "atlas_network", "atlas": "yeo_17", "network": "Nope"},
            {"kind": "mni_file"},
        ],
    )
    def test_invalid_override_raises(self, tmp_path, override):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        with pytest.raises(ValueError):
            mask_extraction_task(**params, threshold_method="cluster_extent", mask_overrides={"net": override})

    def test_mni_replacement_requires_ref_file(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        with pytest.raises(ValueError, match="ref_file"):
            mask_extraction_task(
                **params,
                threshold_method="cluster_extent",
                mask_overrides={"net": {"kind": "atlas_network", "atlas": "yeo_17", "network": "DefaultA"}},
            )

    def test_atlas_roi_override_uses_search_atlas(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, atlas = _task_tree(tmp_path)
        mask_extraction_task(
            **params,
            threshold=100,
            threshold_method="cluster",
            atlas_mask=atlas,
            mask_overrides={"net": {"kind": "atlas_roi"}},
        )

        assert _thr(params["root_dir"]).sum() == stat.size  # the atlas covers the whole grid
        sel = _selection(params["root_dir"])
        assert sel["source_type"] == "atlas_roi"
        assert sel["selection_reason"] == "atlas_roi"

    def test_atlas_network_override_warps_network(self, tmp_path):
        from pineuro.mask_extraction import get_yeo_labels, mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        ref = tmp_path / "boldref.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(stat.shape), np.eye(4)), ref)
        # A subject-space Yeo-17 parcellation with DefaultA in one slab
        parcellation = np.zeros(stat.shape)
        parcellation[6:8, :, :] = get_yeo_labels(n_networks=17).index("DefaultA") + 1
        warped = tmp_path / "yeo_subject.nii.gz"
        nib.save(nib.Nifti1Image(parcellation, np.eye(4)), warped)

        with (
            patch("pineuro.mask_extraction._register_atlas_to_subject", return_value=(warped, [], [])) as reg,
            patch("pineuro.mask_extraction.compute_xfm_to_mni", return_value=(ref, [], [])),
            patch("pineuro.mask_extraction.apply_xfm", side_effect=_write_like(stat.shape)),
            patch("nilearn.datasets.fetch_atlas_harvard_oxford", return_value=_harvard_oxford(stat.shape)),
        ):
            mask_extraction_task(
                **params,
                ref_file=ref,
                threshold_method="cluster_extent",
                mask_overrides={"net": {"kind": "atlas_network", "atlas": "yeo_17", "network": "DefaultA"}},
            )

        assert reg.call_args.kwargs["xfm_prefix"].name == f"{_PREFIX}_to_MNI_boldref"
        assert _thr(params["root_dir"]).sum() == 2 * 12 * 12
        sel = _selection(params["root_dir"])
        assert sel["source_type"] == "atlas_network"
        assert sel["source"] == "yeo_17:DefaultA"
        assert sel["selection_reason"] == "operator_replace"

    def test_mni_file_override_warps_file(self, tmp_path):
        from pineuro.mask_extraction import update_task_mask

        params, stat, _ = _task_tree(tmp_path)
        ref = tmp_path / "boldref.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(stat.shape), np.eye(4)), ref)
        mni_mask = tmp_path / "group_mask_mni.nii.gz"
        data = np.zeros(stat.shape)
        data[0:3, 0:3, 0:3] = 1
        nib.save(nib.Nifti1Image(data, np.eye(4)), mni_mask)

        def _copy_moving(moving, output_file, **_kw):
            nib.save(nib.load(moving), output_file)

        with (
            patch("pineuro.mask_extraction.compute_xfm_to_mni", return_value=(ref, [], ["inv"])),
            patch("pineuro.mask_extraction.apply_xfm", side_effect=_copy_moving),
        ):
            sel = update_task_mask(
                {**params, "ref_file": ref, "threshold_method": "cluster_extent"},
                "net",
                {"kind": "mni_file", "file": str(mni_mask), "selection_reason": "fallback_configured"},
            )

        assert _thr(params["root_dir"]).sum() == 27
        assert sel.source_type == "mni_file"
        assert sel.selection_reason == "fallback_configured"
        assert _selection(params["root_dir"])["source"] == str(mni_mask)


class TestParticipantSpaceReplacement:
    def test_edited_mask_is_used_as_is_without_ref_file(self, tmp_path):
        """An edited copy on the participant's grid replaces the mask; no MNI warp, so no ref_file."""
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        edited = np.zeros(stat.shape)
        edited[1:5, 1:5, 1:3] = 1  # half of the t=5 blob, the rest deleted by hand
        edited_file = tmp_path / "net_edited.nii.gz"
        nib.save(nib.Nifti1Image(edited, np.eye(4)), edited_file)

        mask_extraction_task(
            **params,
            threshold_method="cluster_extent",
            mask_overrides={"net": {"kind": "native_file", "file": str(edited_file)}},
        )

        np.testing.assert_array_equal(_thr(params["root_dir"]), edited)
        sel = _selection(params["root_dir"])
        assert sel["source_type"] == "native_file"
        assert sel["source"] == str(edited_file)
        assert sel["selection_reason"] == "operator_replace"

    def test_mask_on_another_grid_is_refused(self, tmp_path):
        from pineuro.mask_extraction import update_task_mask

        params, _, _ = _task_tree(tmp_path)
        other = tmp_path / "other_grid.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), other)

        with pytest.raises(ValueError, match="BOLD reference grid"):
            update_task_mask(
                {**params, "threshold_method": "cluster_extent"}, "net", {"kind": "native_file", "file": str(other)}
            )


class TestUpdateTaskMask:
    KWARGS = {"threshold_method": "cluster_extent", "min_cluster_size": 20}

    def test_restoring_study_parameters_matches_pipeline(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task, update_task_mask

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, **self.KWARGS)
        pipeline_mask, pipeline_sel = _thr(params["root_dir"]), _selection(params["root_dir"])

        update_task_mask({**params, **self.KWARGS}, "net", {"kind": "rethreshold", "min_cluster_size": 5})
        assert _thr(params["root_dir"]).sum() == 72
        update_task_mask({**params, **self.KWARGS}, "net", None)

        np.testing.assert_array_equal(_thr(params["root_dir"]), pipeline_mask)
        assert _selection(params["root_dir"]) == pipeline_sel

    def test_preview_writes_nothing_and_matches_update(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task, preview_task_mask, update_task_mask

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, **self.KWARGS)
        masks_dir = params["root_dir"] / "03_masks"
        before = {f.name: f.read_bytes() for f in masks_dir.iterdir()}
        override = {"kind": "rethreshold", "cluster_start_threshold": 4.0}

        mask_img, sel = preview_task_mask({**params, **self.KWARGS}, "net", override)

        assert {f.name: f.read_bytes() for f in masks_dir.iterdir()} == before
        assert sel.n_voxels_thresholded == 64  # only the t=5 blob passes t >= 4
        assert sel.n_clusters_found == 1
        assert sel.cluster_table is not None and sel.cluster_table[0]["n_voxels"] == 64
        update_task_mask({**params, **self.KWARGS}, "net", override)
        np.testing.assert_array_equal(np.asarray(mask_img.dataobj), _thr(params["root_dir"]))

    def test_missing_stat_map_raises(self, tmp_path):
        from pineuro.mask_extraction import update_task_mask

        params, _, _ = _task_tree(tmp_path)
        (params["root_dir"] / "01_glm" / f"{_PREFIX}_net_tmap.nii.gz").unlink()
        with pytest.raises(FileNotFoundError, match="run mask extraction first"):
            update_task_mask({**params, **self.KWARGS}, "net", None)


class TestExcludeClusters:
    """_task_tree's stat map has a 64-voxel t=5 blob and a smaller t=3 blob."""

    KWARGS = {"threshold_method": "cluster_extent", "min_cluster_size": 5}

    @staticmethod
    def _peak(stat, value):
        return [int(v) for v in np.argwhere(stat == value)[0]]

    def test_removes_only_the_chosen_cluster(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        peak = self._peak(stat, 3.0)
        mask_extraction_task(
            **params, **self.KWARGS, mask_overrides={"net": {"kind": "exclude_clusters", "peaks": [peak]}}
        )

        mask = _thr(params["root_dir"])
        assert mask.sum() == 64
        assert np.all(stat[mask > 0] == 5.0)
        sel = _selection(params["root_dir"])
        assert sel["selection_reason"] == "operator_exclude_clusters"
        assert sel["n_voxels_thresholded"] == 64
        assert sel["n_clusters_found"] == 1
        assert sel["excluded_clusters"] == [{"n_voxels": 8, "peak_stat": 3.0, "peak_ijk": peak}]
        assert [c["n_voxels"] for c in sel["cluster_table"]] == [64]

    def test_threshold_values_apply_before_the_exclusion(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        override = {"kind": "exclude_clusters", "peaks": [self._peak(stat, 5.0)], "min_cluster_size": 5}
        mask_extraction_task(
            **params, threshold_method="cluster_extent", min_cluster_size=100, mask_overrides={"net": override}
        )

        mask = _thr(params["root_dir"])
        assert mask.sum() == 8  # the study's minimum size would have emptied the mask
        assert np.all(stat[mask > 0] == 3.0)
        assert _selection(params["root_dir"])["threshold_params"]["min_cluster_size"] == 5

    def test_excluded_voxels_are_not_refilled_to_the_cap(self, tmp_path):
        from pineuro.mask_extraction import preview_task_mask

        params, stat, _ = _task_tree(tmp_path)
        run_params = {**params, **self.KWARGS, "max_total_voxels": 64}
        override = {"kind": "exclude_clusters", "peaks": [self._peak(stat, 5.0)]}
        _img, sel = preview_task_mask(run_params, "net", override)
        assert sel.n_voxels_thresholded == 0  # the cap kept only the t=5 blob, which is now gone

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("top_n", {"threshold": 72}),
            ("cluster", {"threshold": 100, "cluster_return_all": True}),
            ("cluster_extent", {"min_cluster_size": 5}),
        ],
    )
    def test_review_and_replay_agree_for_every_method(self, tmp_path, method, kwargs):
        from pineuro.mask_extraction import mask_extraction_task, update_task_mask

        params, stat, atlas = _task_tree(tmp_path)
        run = {**params, "threshold_method": method, **kwargs}
        if method == "cluster":
            run["atlas_mask"] = atlas
        mask_extraction_task(**run)
        before = _thr(params["root_dir"]).copy()
        peak = self._peak(stat, 5.0)
        override = {"kind": "exclude_clusters", "peaks": [peak]}

        sel = update_task_mask(run, "net", override)
        reviewed = _thr(params["root_dir"]).copy()
        assert sel.selection_reason == "operator_exclude_clusters"
        assert reviewed[tuple(peak)] == 0
        assert reviewed.sum() == before.sum() - sel.excluded_clusters[0]["n_voxels"]

        (params["root_dir"] / "03_masks" / f"{_PREFIX}_net_thr.nii.gz").unlink()  # replay from the cached GLM
        mask_extraction_task(**run, mask_overrides={"net": override})
        np.testing.assert_array_equal(_thr(params["root_dir"]), reviewed)

    def test_peak_outside_the_mask_warns_and_keeps_the_mask(self, tmp_path, caplog):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        mask_extraction_task(
            **params,
            **self.KWARGS,
            mask_overrides={"net": {"kind": "exclude_clusters", "peaks": [[11, 0, 11], [99, 0, 0]]}},
        )

        assert _thr(params["root_dir"]).sum() == 72
        assert _selection(params["root_dir"])["excluded_clusters"] == []
        assert sum("not in 'net'" in m for m in caplog.messages) == 2

    def test_two_peaks_in_one_cluster_remove_it_once(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        peaks = [[int(v) for v in ijk] for ijk in np.argwhere(stat == 3.0)[:2]]
        mask_extraction_task(
            **params, **self.KWARGS, mask_overrides={"net": {"kind": "exclude_clusters", "peaks": peaks}}
        )

        assert len(_selection(params["root_dir"])["excluded_clusters"]) == 1

    @pytest.mark.parametrize(
        "override",
        [
            {"kind": "exclude_clusters"},
            {"kind": "exclude_clusters", "peaks": []},
            {"kind": "exclude_clusters", "peaks": [[1, 2]]},
            {"kind": "exclude_clusters", "peaks": [[1.5, 2, 3]]},
            {"kind": "exclude_clusters", "peaks": [[1, 2, 3]], "threshold": 10},  # not a cluster_extent field
        ],
    )
    def test_invalid_override_raises(self, tmp_path, override):
        from pineuro.mask_extraction import mask_extraction_task

        params, _, _ = _task_tree(tmp_path)
        with pytest.raises(ValueError, match="Cluster exclusion"):
            mask_extraction_task(**params, **self.KWARGS, mask_overrides={"net": override})

    def test_selection_written_before_this_field_still_loads(self, tmp_path):
        import json

        from pineuro.mask_extraction import load_task_mask_selections

        (tmp_path / "mask_selections.json").write_text(
            json.dumps({"type": "task", "selections": {"net": {"mask_name": "net"}}})
        )
        assert load_task_mask_selections(tmp_path)["net"].excluded_clusters is None

    def test_single_cluster_image(self, tmp_path):
        from pineuro.mask_extraction import mask_extraction_task, single_cluster_image

        params, stat, _ = _task_tree(tmp_path)
        mask_extraction_task(**params, **self.KWARGS)
        thr_file = params["root_dir"] / "03_masks" / f"{_PREFIX}_net_thr.nii.gz"

        img, n_voxels = single_cluster_image(thr_file, self._peak(stat, 3.0))
        assert n_voxels == 8
        assert np.all(stat[np.asarray(img.dataobj) > 0] == 3.0)
        with pytest.raises(ValueError, match="not in"):
            single_cluster_image(thr_file, [11, 0, 11])


class TestMaskOverridesFile:
    def test_round_trip_and_empty_removes(self, tmp_path):
        from pineuro.mask_extraction import load_mask_overrides, save_mask_overrides

        assert load_mask_overrides(tmp_path) == {}
        overrides = {"net": {"kind": "atlas_network", "atlas": "yeo_17", "network": "DefaultA"}}
        path = save_mask_overrides(tmp_path, overrides)
        assert load_mask_overrides(tmp_path) == overrides

        save_mask_overrides(tmp_path, {})
        assert not path.exists()


class TestHarvardOxfordTransform:
    def test_overwrite_recomputes_labelling_transform(self, tmp_path):
        """Regression: the labelling transform was always reused, so a new boldref left it stale."""
        import pandas as pd

        from pineuro.mask_extraction import mask_extraction_task

        params, stat, _ = _task_tree(tmp_path)
        ref = tmp_path / "boldref.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(stat.shape), np.eye(4)), ref)

        class _FakeGLM:
            def __init__(self, **_kw):
                pass

            def fit(self, *_a, **_kw):
                self.design_matrices_ = [pd.DataFrame({"a": [0.0, 0.0, 0.0]})]
                return self

            def compute_contrast(self, *_a, **_kw):
                return nib.Nifti1Image(stat, np.eye(4))

        with (
            patch("pineuro.mask_extraction.compute_xfm_to_mni", return_value=(ref, [], [])) as xfm,
            patch("pineuro.mask_extraction.apply_xfm", side_effect=_write_like(stat.shape)),
            patch("nilearn.datasets.fetch_atlas_harvard_oxford", return_value=_harvard_oxford(stat.shape)),
            patch("nilearn.glm.first_level.FirstLevelModel", _FakeGLM),
        ):
            mask_extraction_task(**params, ref_file=ref, threshold_method="cluster_extent", overwrite=True)

        assert xfm.call_count == 1
        assert xfm.call_args.kwargs["overwrite"] is True
