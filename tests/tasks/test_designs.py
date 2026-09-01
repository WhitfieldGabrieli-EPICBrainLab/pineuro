"""Tests for pineuro.tasks.designs — BlockDesign, NuisanceOnlyDesign, design_for_metadata.

These are lightweight structural checks (shape, baseline-zero, active-nonzero) that run
in the default suite. The rigorous HRF / independent-reference correctness checks for the
same design regressors live in tests/validation/test_numeric_core.py (opt-in `-m validation`).
"""

import numpy as np
import pytest


class TestBlockDesign:
    def test_compute_regressors_shape(self):
        from pineuro.tasks.designs import BlockDesign

        design = BlockDesign(n_baseline=10)
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        assert regressors.shape == (50, 1)

    def test_baseline_is_zero(self):
        from pineuro.tasks.designs import BlockDesign

        design = BlockDesign(n_baseline=10)
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        # First few volumes should be zero (baseline)
        np.testing.assert_allclose(regressors[:8, 0], 0.0, atol=1e-6)

    def test_active_is_nonzero(self):
        from pineuro.tasks.designs import BlockDesign

        design = BlockDesign(n_baseline=10)
        regressors = design.compute_regressors(n_volumes=100, tr=1.0)

        # Well after baseline + HRF delay, regressor should be non-zero
        assert np.any(regressors[30:, 0] > 0.1)


class TestNuisanceOnlyDesign:
    def test_empty_regressors(self):
        from pineuro.tasks.designs import NuisanceOnlyDesign

        design = NuisanceOnlyDesign()
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        assert regressors.shape == (50, 0)

    def test_n_regressors_is_zero(self):
        from pineuro.tasks.designs import NuisanceOnlyDesign

        design = NuisanceOnlyDesign()
        assert design.n_regressors == 0


def _metadata(design_type: str):
    """Minimal TaskMetadata carrying just the design_type under test."""
    from pineuro.tasks.task_registry import TaskMetadata

    return TaskMetadata(
        task_id="fake",
        display_name="Fake",
        description="Test double",
        parameters=[],
        design_type=design_type,
    )


class TestDesignForMetadata:
    """The single design_type -> TaskDesign mapping, shared by session.py and the GUI.

    Both runners dispatch through this, so a task cannot end up with a different GLM
    depending on which one launched it.
    """

    def test_block_maps_to_block_design(self):
        from pineuro.tasks.designs import BlockDesign, design_for_metadata

        design = design_for_metadata(_metadata("block"), n_baseline=10)

        assert isinstance(design, BlockDesign)
        assert design.n_baseline == 10

    def test_nuisance_only_maps_to_nuisance_only_design(self):
        from pineuro.tasks.designs import NuisanceOnlyDesign, design_for_metadata

        design = design_for_metadata(_metadata("nuisance_only"), n_baseline=10)

        assert isinstance(design, NuisanceOnlyDesign)

    def test_unknown_design_type_raises(self):
        """An unrecognized design_type is a programming error, not a silent default.

        Both dispatch sites previously fell back silently and *differently* — session.py
        to no task regressor, the GUI to a block design — so the same task got two
        different GLMs depending on how it was launched.
        """
        from pineuro.tasks.designs import design_for_metadata

        with pytest.raises(ValueError, match="event_related"):
            design_for_metadata(_metadata("event_related"), n_baseline=10)

    def test_registered_tasks_all_have_a_valid_design_type(self):
        """No shipped task can hit the raise above."""
        from pineuro.tasks.designs import DESIGN_TYPES
        from pineuro.tasks.task_registry import get_task_metadata, list_available_tasks

        for task_id in list_available_tasks():
            assert get_task_metadata(task_id).design_type in DESIGN_TYPES, task_id
