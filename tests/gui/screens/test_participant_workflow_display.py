"""Tests for participant workflow display: pipeline-panel helpers and the live monitoring panel."""

from types import SimpleNamespace

from pineuro.gui.screens.participant_workflow import ParticipantWorkflowScreen, _format_threshold_summary


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


def test_localizer_brain_slice_keeps_last_frame_between_volumes():
    """Regression guard: Flet's default (``gapless_playback=False``) blanks the image on
    every new ``src``, so each volume's slice flashed against the white page.

    ``__init__`` builds the live layout, so the panel is built on a bare instance.
    """
    screen = ParticipantWorkflowScreen.__new__(ParticipantWorkflowScreen)
    screen.study_config = SimpleNamespace(session_config={"localizers": [{"type": "rest"}]})
    screen._build_localizer_monitoring_panel("localizer_0_run_1", n_volumes=10)
    assert screen._slice_image.gapless_playback is True
