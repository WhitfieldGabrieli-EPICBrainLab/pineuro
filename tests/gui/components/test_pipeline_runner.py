"""Tests for pineuro.gui.components.pipeline_runner pure helpers."""

import pytest

from pineuro.gui.components.pipeline_runner import PipelineRunner, _select_sham_task_mask_names


class TestSelectShamTaskMaskNames:
    def test_hemisphere_split_full_pair(self):
        # hemisphere_split produces both hemispheres; both roles must be
        # driven by the sham pair, not fall back to the real non-target.
        task = _select_sham_task_mask_names(
            sham_mask_names=["SomMotA_right", "SomMotA_left"],
            real_mask_names=["SomMotA", "DefaultA"],
        )
        assert task == ["SomMotA_right", "SomMotA_left"]

    def test_two_networks_full_pair(self):
        task = _select_sham_task_mask_names(
            sham_mask_names=["ContA", "SalA"],
            real_mask_names=["SomMotA", "DefaultA"],
        )
        assert task == ["ContA", "SalA"]

    def test_single_sham_mask_swaps_target_only(self):
        # Degraded case (only one sham mask resolved, e.g. a missing hemisphere
        # mask): the target role is swapped to sham, the real non-target is kept.
        task = _select_sham_task_mask_names(
            sham_mask_names=["SalA"],
            real_mask_names=["SomMotA", "DefaultA"],
        )
        assert task == ["SalA", "DefaultA"]

    def test_more_than_two_sham_masks_truncates(self):
        task = _select_sham_task_mask_names(
            sham_mask_names=["A", "B", "C"],
            real_mask_names=["X", "Y"],
        )
        assert task == ["A", "B"]

    def test_returns_copy_not_alias(self):
        # Mutating the result must not clobber the caller's sham list.
        sham = ["SomMotA_right", "SomMotA_left"]
        task = _select_sham_task_mask_names(sham, ["SomMotA", "DefaultA"])
        task.append("extra")
        assert sham == ["SomMotA_right", "SomMotA_left"]


class TestRunManualMaskImport:
    def test_raises_when_not_configured(self, tmp_path):
        # A manual localizer whose per-participant files were never chosen
        # must fail with a clear, actionable message — not a cryptic error.
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.participant_id = "sub-01"
        runner.overwrite = False
        runner._emit_progress = lambda *a, **k: None  # type: ignore[method-assign]

        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "native"}
        with pytest.raises(ValueError, match="not configured"):
            runner._run_manual_mask_import(
                loc,
                base_dir=tmp_path,
                root_dir=tmp_path / "mask_extraction",
                qc_dir=tmp_path / "qc",
            )
