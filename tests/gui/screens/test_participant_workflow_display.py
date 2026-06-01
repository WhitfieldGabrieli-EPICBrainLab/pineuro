"""Tests for participant workflow pipeline-panel display helpers."""

import pytest

from pineuro.gui.screens.participant_workflow import _format_threshold_summary

pytestmark = pytest.mark.unit


def test_threshold_summary_top_n():
    assert _format_threshold_summary({"threshold_method": "top_n", "threshold": 2000}) == "Top 2000 voxels"


def test_threshold_summary_defaults_to_top_n():
    # No threshold_method => top_n with the default voxel count.
    assert _format_threshold_summary({}) == "Top 2000 voxels"


def test_threshold_summary_cluster():
    summary = _format_threshold_summary(
        {"threshold_method": "cluster", "threshold": 1500, "cluster_start_threshold": 2.5}
    )
    assert "Atlas-guided" in summary
    assert "target 1500 vox" in summary
    assert "2.5" in summary


def test_threshold_summary_cluster_extent_with_cap():
    summary = _format_threshold_summary(
        {
            "threshold_method": "cluster_extent",
            "threshold": 2000,  # unused by cluster_extent — must not appear
            "cluster_start_threshold": 1.0,
            "min_cluster_size": 10,
            "max_total_voxels": 2000,
        }
    )
    assert summary == "Cluster extent: stat ≥ 1.0, min 10 vox, cap 2000"


def test_threshold_summary_cluster_extent_no_cap():
    summary = _format_threshold_summary(
        {
            "threshold_method": "cluster_extent",
            "cluster_start_threshold": 2.0,
            "min_cluster_size": 50,
            "max_total_voxels": None,
        }
    )
    assert summary == "Cluster extent: stat ≥ 2.0, min 50 vox"
    assert "cap" not in summary
