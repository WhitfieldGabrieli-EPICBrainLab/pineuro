"""Tests for pineuro.tasks.ball — BallTask movement math and outlier rejection.

PsychoPy-dependent methods (setup, apply_movement, check_hits, run) are not
unit-tested; the focus is on calculate_ball_movement (pure-ish math + state)
and check_pause_movement (pure logic). Registry/metadata well-formedness is
covered generically in tests/tasks/test_task_registry.py.
"""

import numpy as np


def _stub_ball(signal_mode="activation", connectivity_target="increase"):
    """Construct a BallTask bypassing __init__ — only the attrs movement logic touches."""
    from pineuro.tasks.ball import BallTask

    task = BallTask.__new__(BallTask)
    task._signal_mode = signal_mode
    task._connectivity_target = connectivity_target
    task.mask_names = ["ContA", "DefaultA"]
    task.pda_outlier_threshold = 2.0
    task.network_directions = np.array([1, -1])  # ContA → up, DefaultA → down
    # Outputs (initialized by the methods under test)
    task.outlier_count = 0
    task._current_is_outlier = False
    task._last_connectivity = 0.0
    task.current_activity = 0.0
    task.current_direction = 0.0
    return task


# =============================================================================
# Activation-mode movement
# =============================================================================


class TestCalculateBallMovementActivation:
    def test_higher_first_mask_moves_up(self):
        task = _stub_ball()
        task.calculate_ball_movement({"ContA": 1.0, "DefaultA": 0.0})
        assert task.current_direction == 1.0  # network_directions[0]
        assert task.current_activity > 0.0
        assert task._current_is_outlier is False

    def test_higher_second_mask_moves_down(self):
        task = _stub_ball()
        task.calculate_ball_movement({"ContA": 0.0, "DefaultA": 1.5})
        assert task.current_direction == -1.0
        assert task.current_activity > 0.0

    def test_magnitude_from_absolute_difference(self):
        task = _stub_ball()
        task.calculate_ball_movement({"ContA": 1.0, "DefaultA": 0.0})
        # |1.0 - 0.0| / 10 = 0.1
        assert task.current_activity == 0.1

    def test_outlier_rejected_increments_count(self):
        task = _stub_ball()
        task.current_activity = 99.0  # pre-existing value
        task.current_direction = 1.0
        task.calculate_ball_movement({"ContA": 5.0, "DefaultA": 0.0})  # 5.0 > threshold 2.0
        assert task.outlier_count == 1
        assert task._current_is_outlier is True
        # Outlier path returns without overwriting current_activity/direction
        assert task.current_activity == 99.0
        assert task.current_direction == 1.0

    def test_balanced_activations_preserve_current(self):
        task = _stub_ball()
        task.current_activity = 0.5
        task.current_direction = -1.0
        # [0.5, -0.5] → mean == 0 → early return preserves state
        task.calculate_ball_movement({"ContA": 0.5, "DefaultA": -0.5})
        assert task.current_activity == 0.5
        assert task.current_direction == -1.0


# =============================================================================
# Connectivity-mode movement
# =============================================================================


class TestCalculateBallMovementConnectivity:
    def _activations(self, corr):
        return {"corr_ContA_DefaultA": corr}

    def test_positive_corr_increase_target_up(self):
        task = _stub_ball(signal_mode="connectivity", connectivity_target="increase")
        task.calculate_ball_movement(self._activations(0.5))
        assert task.current_direction == 1.0  # positive correlation → ball up
        assert task._last_connectivity == 0.5

    def test_negative_corr_increase_target_down(self):
        task = _stub_ball(signal_mode="connectivity", connectivity_target="increase")
        task.calculate_ball_movement(self._activations(-0.3))
        assert task.current_direction == -1.0

    def test_decrease_target_inverts_direction(self):
        task = _stub_ball(signal_mode="connectivity", connectivity_target="decrease")
        # Negative correlation should move ball UP when goal is to decrease connectivity
        task.calculate_ball_movement(self._activations(-0.4))
        assert task.current_direction == 1.0

    def test_outlier_rejected(self):
        task = _stub_ball(signal_mode="connectivity")
        task.calculate_ball_movement(self._activations(5.0))  # |5.0| > 2.0
        assert task.outlier_count == 1
        assert task._current_is_outlier is True


# =============================================================================
# Pause movement (state transition)
# =============================================================================


class TestCheckPauseMovement:
    def _stub_with_circles(self):
        task = _stub_ball()
        task.circle_positions = [1.0 / 3, -1.0 / 3]
        task.network_directions = np.array([1, -1])
        # check_pause_movement iterates task.circles for length but never reads them
        task.circles = [object(), object()]
        task.pause_movement = False
        task.ball_pos = [0.0, 0.0]
        return task

    def test_no_pause_when_ball_centered(self):
        task = self._stub_with_circles()
        task.check_pause_movement()
        assert task.pause_movement is False

    def test_pauses_when_ball_reaches_top_circle(self):
        task = self._stub_with_circles()
        task.ball_pos = [0.0, 1.0 / 3]  # at top circle center
        task.check_pause_movement()
        assert task.pause_movement is True

    def test_pauses_when_ball_reaches_bottom_circle(self):
        task = self._stub_with_circles()
        task.ball_pos = [0.0, -1.0 / 3]  # at bottom circle center
        task.check_pause_movement()
        assert task.pause_movement is True
