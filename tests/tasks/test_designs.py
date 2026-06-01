"""Tests for pineuro.tasks.designs — BlockDesign, NuisanceOnlyDesign, EventRelatedDesign.

These are lightweight structural checks (shape, baseline-zero, active-nonzero) that run
in the default suite. The rigorous HRF / independent-reference correctness checks for the
same design regressors live in tests/validation/test_numeric_core.py (opt-in `-m validation`).
"""

import numpy as np


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


class TestEventRelatedDesign:
    def test_compute_regressors_shape(self):
        from pineuro.tasks.designs import EventRelatedDesign

        design = EventRelatedDesign(
            event_onsets={"face": [10.0, 30.0, 50.0], "house": [20.0, 40.0, 60.0]},
            event_durations={"face": 5.0, "house": 5.0},
        )
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        assert regressors.shape[0] == 50
        assert regressors.shape[1] == 2  # 2 conditions
