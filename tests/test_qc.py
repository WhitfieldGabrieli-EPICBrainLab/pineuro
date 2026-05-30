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


# =============================================================================
# HTML template
# =============================================================================


class TestHtmlTemplate:
    def test_raw_html_section(self):
        from pineuro.qc import _generate_html_template

        raw = '<div class="viewer-section"><h3>Test</h3><p>Content</p></div>'
        html = _generate_html_template(
            title="Test",
            participant_id="sub-01",
            stage="Test",
            sections=[{"type": "raw_html", "content": raw}],
        )
        assert raw in html
        assert "viewer-section" in html


# =============================================================================
# Triggered-event durations (RIT experience sampling / triggered ball)
# =============================================================================


def _write_events_tsv(path, rows, columns=("onset", "duration", "trial_type", "volume", "activation")):
    lines = ["\t".join(columns)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "n/a")) for c in columns))
    path.write_text("\n".join(lines))
    return path


# baseline + per-volume activation rows (excluded) plus three episodes.
_SAMPLE_EVENT_ROWS = [
    {"onset": 0.0, "duration": 20.0, "trial_type": "baseline", "volume": 0},
    {"onset": 21.0, "duration": 1.0, "trial_type": "activation", "volume": 11, "activation": 0.5},
    {"onset": 23.0, "duration": 1.0, "trial_type": "activation", "volume": 12, "activation": 0.6},
    {"onset": 25.0, "duration": 1.0, "trial_type": "activation", "volume": 13, "activation": 0.7},
    {"onset": 27.0, "duration": 1.0, "trial_type": "activation", "volume": 14, "activation": 0.8},
    {"onset": 29.0, "duration": 1.0, "trial_type": "activation", "volume": 15, "activation": 0.4},
    {"onset": 22.0, "duration": 5.5, "trial_type": "prompt_triggered", "volume": 11},
    {"onset": 40.0, "duration": 8.0, "trial_type": "prompt_timeout", "volume": 20},
    {"onset": 55.0, "duration": 3.0, "trial_type": "prompt_triggered_incomplete", "volume": 28},
]


class TestSummarizeTriggeredEvents:
    def test_returns_only_episode_rows(self, tmp_path):
        from pineuro.qc import _summarize_triggered_events

        path = _write_events_tsv(tmp_path / "events.tsv", _SAMPLE_EVENT_ROWS)
        eps = _summarize_triggered_events(path, ("baseline", "activation"))

        assert [e["trial_type"] for e in eps] == [
            "prompt_triggered",
            "prompt_timeout",
            "prompt_triggered_incomplete",
        ]
        assert [e["episode"] for e in eps] == [1, 2, 3]

        first = eps[0]
        assert first["onset"] == 22.0
        assert first["volume"] == 11
        assert first["duration"] == 5.5
        # Start volume is the trigger volume; end maps onset+duration via the
        # per-volume activation rows and never precedes the start.
        assert first["start_vol"] == 11.0
        assert first["end_vol"] >= first["start_vol"]

    def test_missing_file_returns_empty(self, tmp_path):
        from pineuro.qc import _summarize_triggered_events

        assert _summarize_triggered_events(tmp_path / "nope.tsv", ("baseline", "activation")) == []

    def test_no_episode_rows_returns_empty(self, tmp_path):
        from pineuro.qc import _summarize_triggered_events

        rows = [r for r in _SAMPLE_EVENT_ROWS if r["trial_type"] in ("baseline", "activation")]
        path = _write_events_tsv(tmp_path / "events.tsv", rows)
        assert _summarize_triggered_events(path, ("baseline", "activation")) == []


class TestActivationOverlay:
    def test_plot_activation_timeseries_with_episodes(self, sample_feedbacklog, tmp_path):
        from pineuro.qc import plot_activation_timeseries

        log_path = sample_feedbacklog(n_volumes=50, mask_names=["DMN"])
        output = tmp_path / "activation_episodes.png"
        episodes = [
            {"trial_type": "prompt_triggered", "start_vol": 11.0, "end_vol": 14.0},
            {"trial_type": "prompt_timeout", "start_vol": 20.0, "end_vol": 24.0},
        ]
        plot_activation_timeseries(log_path, output, episodes=episodes)
        assert output.exists()
        assert output.stat().st_size > 0


class TestFeedbackReportTriggeredDurations:
    def _feedback_dir(self, tmp_path, sample_feedbacklog, task, with_events=True):
        prefix = f"sub-01_ses-01_task-{task}_run-01"
        sample_feedbacklog(n_volumes=50, mask_names=["DMN"], name=f"{prefix}_attempt-01_feedbacklog.tsv")
        if with_events:
            _write_events_tsv(tmp_path / f"{prefix}_events.tsv", _SAMPLE_EVENT_ROWS)
        return tmp_path

    def test_triggered_task_renders_durations(self, sample_feedbacklog, tmp_path):
        from pineuro.qc import generate_feedback_report

        feedback_dir = self._feedback_dir(tmp_path, sample_feedbacklog, "ritexperiencesampling")
        report = generate_feedback_report(
            feedback_dir=feedback_dir,
            subject="sub-01",
            session="ses-01",
            task="ritexperiencesampling",
            run=1,
            attempt=1,
            overwrite=True,
        )
        html = report.read_text()
        assert "Triggered Prompt Durations" in html
        assert "Duration (s)" in html
        assert "5.5 s" in html
        assert "n=3" in html

    def test_block_task_has_no_durations_section(self, sample_feedbacklog, tmp_path):
        from pineuro.qc import generate_feedback_report
        from pineuro.tasks.task_registry import get_task_metadata

        # Sanity: a block-design task does not declare triggered episodes, so the
        # section is gated out even though an events TSV is present.
        assert get_task_metadata("thermometer").triggered_event_report is None

        feedback_dir = self._feedback_dir(tmp_path, sample_feedbacklog, "thermometer")
        report = generate_feedback_report(
            feedback_dir=feedback_dir,
            subject="sub-01",
            session="ses-01",
            task="thermometer",
            run=1,
            attempt=1,
            overwrite=True,
        )
        html = report.read_text()
        assert "Duration (s)" not in html
        assert "Triggered Prompt Durations" not in html
