"""Validation -- cluster_threshold_stat_map iterative thresholding.

cluster_threshold_stat_map iteratively raises a stat threshold inside an atlas
mask until the largest contiguous cluster fits within ``target_n_voxels``.

The function is fully deterministic: for any synthetic stat map and atlas
combination, the output binary mask is computable by hand from the threshold
sweep. The validation cases below construct stat maps where the expected
final cluster shape and threshold can be reasoned about exactly:

- A constant-stat region inside the atlas yields a known cluster size at
  the chosen threshold.
- A two-peak region with one large cluster and one small one tests that
  the iterative search converges on the cluster that fits the budget.
- An empty atlas / empty stat map raises the documented outcome codes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pineuro.mask_extraction import cluster_threshold_stat_map

pytestmark = pytest.mark.validation


def test_shape_mismatch_raises():
    """Stat and atlas shapes must match; otherwise ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        cluster_threshold_stat_map(
            stat_data=np.zeros((10, 10, 10)),
            atlas_data=np.zeros((10, 10, 8)),
            target_n_voxels=10,
        )


def test_empty_atlas_returns_zero_mask():
    """An empty atlas yields an all-zero mask with outcome 'empty_atlas'."""
    mask, info = cluster_threshold_stat_map(
        stat_data=np.ones((10, 10, 10)) * 5.0,
        atlas_data=np.zeros((10, 10, 10), dtype=np.int32),
        target_n_voxels=100,
    )
    assert mask.sum() == 0  # nothing was selected
    assert info["outcome"] == "empty_atlas"
    assert info["atlas_voxel_count"] == 0


def test_no_clusters_found_when_stat_below_threshold():
    """A stat map entirely below the start_threshold yields 'no_clusters_found'."""
    stat = np.zeros((10, 10, 10), dtype=np.float64)
    atlas = np.ones((10, 10, 10), dtype=np.int32)
    mask, info = cluster_threshold_stat_map(
        stat_data=stat,
        atlas_data=atlas,
        target_n_voxels=100,
        start_threshold=1.0,
    )
    assert mask.sum() == 0
    assert info["outcome"] == "no_clusters_found"


def test_already_under_target_returns_immediately():
    """If the initial cluster already fits the target, outcome is 'already_under_target'."""
    stat = np.zeros((10, 10, 10), dtype=np.float64)
    stat[3:5, 3:5, 3:5] = 5.0  # a 2x2x2 cluster of value 5 -> 8 voxels
    atlas = np.ones((10, 10, 10), dtype=np.int32)

    mask, info = cluster_threshold_stat_map(
        stat_data=stat,
        atlas_data=atlas,
        target_n_voxels=10,  # 8 voxels fit comfortably in 10
        start_threshold=1.0,
    )
    assert info["outcome"] == "already_under_target"
    assert info["cluster_size"] == 8  # the 2x2x2 cube has exactly 8 voxels
    assert int(mask.sum()) == 8


def test_iterative_threshold_finds_fit_cluster():
    """A graded stat map iteratively converges on a smaller cluster.

    Stat map: a 4x4x4 cube of value 1, with a smaller 2x2x2 inner cube of value 3.
    At threshold=1.0 the cluster is 64 voxels (too large); raise to >1.0 and the
    smaller 8-voxel cluster remains. With target_n_voxels=10, the loop converges
    on the inner 8-voxel cluster.
    """
    stat = np.zeros((10, 10, 10), dtype=np.float64)
    stat[3:7, 3:7, 3:7] = 1.0  # outer cube: 64 voxels at threshold 1.0
    stat[4:6, 4:6, 4:6] = 3.0  # inner cube: 8 voxels at threshold 2.0+
    atlas = np.ones((10, 10, 10), dtype=np.int32)

    mask, info = cluster_threshold_stat_map(
        stat_data=stat,
        atlas_data=atlas,
        target_n_voxels=10,
        start_threshold=0.5,  # start below 1.0 so we sweep through both regimes
        step=0.5,
    )
    assert info["outcome"] == "success"
    assert info["cluster_size"] == 8  # inner 2x2x2 cube has exactly 8 voxels
    # Mask exactly covers the inner cube.
    expected_mask = np.zeros_like(stat, dtype=np.uint8)
    expected_mask[4:6, 4:6, 4:6] = 1
    np.testing.assert_array_equal(mask, expected_mask)


def test_largest_cluster_selected_with_return_all_false():
    """With return_all_clusters=False (default), the LARGEST cluster is returned.

    Two disjoint clusters: a 27-voxel cube and an 8-voxel cube. Target=30 -> the
    27-voxel cluster fits but the smaller does not by itself. The single-largest
    contract picks the 27-voxel cluster only.
    """
    stat = np.zeros((10, 10, 10), dtype=np.float64)
    stat[1:4, 1:4, 1:4] = 5.0  # 3x3x3 = 27 voxels
    stat[6:8, 6:8, 6:8] = 5.0  # 2x2x2 = 8 voxels (disjoint, separated by gap)
    atlas = np.ones((10, 10, 10), dtype=np.int32)

    mask, info = cluster_threshold_stat_map(
        stat_data=stat,
        atlas_data=atlas,
        target_n_voxels=30,
        start_threshold=1.0,
    )
    assert info["cluster_size"] == 27  # the larger cluster
    assert int(mask.sum()) == 27
    # The mask covers the 3x3x3 cube exactly.
    expected_mask = np.zeros_like(stat, dtype=np.uint8)
    expected_mask[1:4, 1:4, 1:4] = 1
    np.testing.assert_array_equal(mask, expected_mask)


def test_all_clusters_returned_with_return_all_true():
    """With return_all_clusters=True, all suprathreshold clusters are unioned."""
    stat = np.zeros((10, 10, 10), dtype=np.float64)
    stat[1:4, 1:4, 1:4] = 5.0  # 27 voxels
    stat[6:8, 6:8, 6:8] = 5.0  # 8 voxels
    atlas = np.ones((10, 10, 10), dtype=np.int32)

    mask, info = cluster_threshold_stat_map(
        stat_data=stat,
        atlas_data=atlas,
        target_n_voxels=50,  # 27+8=35 fits in 50
        start_threshold=1.0,
        return_all_clusters=True,
    )
    assert info["cluster_size"] == 35  # 27 + 8
    assert int(mask.sum()) == 35


def test_atlas_restricts_to_within_atlas():
    """Stat values outside the atlas are ignored even if they exceed threshold."""
    stat = np.zeros((10, 10, 10), dtype=np.float64)
    stat[1:4, 1:4, 1:4] = 5.0  # this cluster is INSIDE the atlas
    stat[6:9, 6:9, 6:9] = 5.0  # this cluster is OUTSIDE the atlas, must be ignored

    atlas = np.zeros((10, 10, 10), dtype=np.int32)
    atlas[0:5, 0:5, 0:5] = 1  # atlas covers only the lower octant

    mask, info = cluster_threshold_stat_map(
        stat_data=stat,
        atlas_data=atlas,
        target_n_voxels=50,
        start_threshold=1.0,
    )
    # Only the in-atlas cluster (27 voxels) is selected; the out-of-atlas
    # cluster is invisible to the algorithm because masked_stat zeroed it.
    assert info["cluster_size"] == 27
    assert int(mask.sum()) == 27
    # Mask does not contain any voxel outside the atlas.
    assert int(((mask > 0) & (atlas == 0)).sum()) == 0
