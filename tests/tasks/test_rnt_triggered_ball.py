"""Tests for pineuro.tasks.rnt_triggered_ball — metadata, trigger detection, TriggerState."""


class TestRNTTriggeredBallMetadata:
    def test_registered_in_registry(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "rnt_triggered_ball" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rnt_triggered_ball")
        assert meta.task_id == "rnt_triggered_ball"
        assert meta.n_networks == 3

    def test_has_network_roles(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rnt_triggered_ball")
        role_keys = [r.key for r in meta.network_roles]
        assert "trigger_network" in role_keys
        assert "top_network" in role_keys
        assert "bottom_network" in role_keys

    def test_task_parameters_defined(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rnt_triggered_ball")
        param_keys = {p.key for p in meta.parameters}
        assert "activation_contrast" in param_keys
        assert "consecutive_volumes" in param_keys
        assert "activation_trend" in param_keys
        assert "trend_method" in param_keys
        assert "min_prompt_interval" in param_keys
        assert "timeout_duration" in param_keys


def _make_task(**kwargs):
    from pineuro.tasks.rnt_triggered_ball import RNTTriggeredBallTask, TriggerState

    task = RNTTriggeredBallTask.__new__(RNTTriggeredBallTask)
    task.activation_contrast = kwargs.get("activation_contrast", "positive")
    task.consecutive_volumes = kwargs.get("consecutive_volumes", 3)
    task.activation_trend = kwargs.get("activation_trend", "increasing")
    task.trend_method = kwargs.get("trend_method", "strict")
    task.min_prompt_interval = kwargs.get("min_prompt_interval", 10.0)
    task.timeout_duration = kwargs.get("timeout_duration", 60.0)
    task._trigger_state = TriggerState()
    return task


class TestRNTTriggeredBallTriggerDetection:
    def _make_task(self, **kwargs):
        return _make_task(**kwargs)

    def test_cooldown_blocks_trigger(self):
        task = self._make_task(min_prompt_interval=10.0)
        task._trigger_state.last_prompt_time = 5.0
        # Current time 10.0, only 5s since last → cooldown
        result = task._check_trigger(10.0, 1.0)
        assert result is None

    def test_timeout_fires(self):
        task = self._make_task(timeout_duration=60.0, min_prompt_interval=10.0)
        task._trigger_state.last_prompt_time = 0.0
        # 70s since last prompt → timeout
        result = task._check_trigger(70.0, -1.0)
        assert result == "timeout"

    def test_threshold_not_met(self):
        task = self._make_task(activation_contrast="positive", min_prompt_interval=0.0)
        task._trigger_state.last_prompt_time = 40.0  # 10s ago — past cooldown, before timeout
        # Activation < 0 when expecting positive
        result = task._check_trigger(50.0, -0.5)
        assert result is None

    def test_trigger_fires_with_trend(self):
        from collections import deque

        task = self._make_task(
            consecutive_volumes=3,
            activation_trend="increasing",
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0  # past cooldown, before timeout
        task._trigger_state.activation_history = deque([0.1, 0.5, 1.0], maxlen=50)

        result = task._check_trigger(50.0, 1.0)
        assert result == "triggered"

    def test_no_trigger_without_enough_history(self):
        from collections import deque

        task = self._make_task(consecutive_volumes=3, min_prompt_interval=0.0)
        task._trigger_state.last_prompt_time = 40.0  # past cooldown, before timeout
        task._trigger_state.activation_history = deque([0.1, 0.5], maxlen=50)

        result = task._check_trigger(50.0, 1.0)
        assert result is None

    def test_negative_contrast(self):
        from collections import deque

        task = self._make_task(
            activation_contrast="negative",
            activation_trend="decreasing",
            consecutive_volumes=2,
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0  # past cooldown, before timeout
        task._trigger_state.activation_history = deque([-0.5, -1.0], maxlen=50)

        result = task._check_trigger(50.0, -1.0)
        assert result == "triggered"


class TestTriggerState:
    def test_defaults(self):
        from pineuro.tasks.rnt_triggered_ball import TriggerState

        state = TriggerState()
        assert state.n_episodes_triggered == 0
        assert state.n_episodes_timeout == 0
        assert state.last_processed_volume == 0
        assert len(state.activation_history) == 0


class TestLeastSquaresSlope:
    def test_empty_returns_zero(self):
        from pineuro.tasks.rnt_triggered_ball import _least_squares_slope

        assert _least_squares_slope([]) == 0.0

    def test_single_value_returns_zero(self):
        from pineuro.tasks.rnt_triggered_ball import _least_squares_slope

        assert _least_squares_slope([0.5]) == 0.0

    def test_monotonic_increase_positive_slope(self):
        from pineuro.tasks.rnt_triggered_ball import _least_squares_slope

        assert _least_squares_slope([0.0, 1.0, 2.0]) > 0

    def test_monotonic_decrease_negative_slope(self):
        from pineuro.tasks.rnt_triggered_ball import _least_squares_slope

        assert _least_squares_slope([2.0, 1.0, 0.0]) < 0

    def test_blip_does_not_flip_sign(self):
        # Net rise with one down-blip: slope still positive.
        from pineuro.tasks.rnt_triggered_ball import _least_squares_slope

        assert _least_squares_slope([0.1, 0.4, 0.3]) > 0


class TestRNTTriggeredBallSlopeMode:
    """Slope-mode trend detection in _check_trigger."""

    def test_strict_rejects_blip(self):
        # Anchor: current strict behavior is preserved.
        from collections import deque

        task = _make_task(
            trend_method="strict",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([0.1, 0.4, 0.3], maxlen=50)

        assert task._check_trigger(50.0, 0.3) is None

    def test_slope_accepts_blip(self):
        # Same data slope mode accepts because the overall trend is up.
        from collections import deque

        task = _make_task(
            trend_method="slope",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([0.1, 0.4, 0.3], maxlen=50)

        assert task._check_trigger(50.0, 0.3) == "triggered"

    def test_slope_rejects_when_last_value_below_threshold(self):
        # Slope is positive but last value is negative -> threshold gate blocks.
        from collections import deque

        task = _make_task(
            trend_method="slope",
            activation_contrast="positive",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([-0.5, -0.3, -0.1], maxlen=50)

        assert task._check_trigger(50.0, -0.1) is None

    def test_slope_negative_contrast_with_blip(self):
        from collections import deque

        task = _make_task(
            trend_method="slope",
            activation_contrast="negative",
            activation_trend="decreasing",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([-0.1, -0.4, -0.3], maxlen=50)

        assert task._check_trigger(50.0, -0.3) == "triggered"

    def test_slope_rejects_when_trend_wrong_direction(self):
        # Increasing trend configured, but the data trends down.
        from collections import deque

        task = _make_task(
            trend_method="slope",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([0.5, 0.4, 0.3], maxlen=50)

        # Latest value still positive so threshold passes, but slope is negative.
        assert task._check_trigger(50.0, 0.3) is None


class TestConstructorValidation:
    def test_invalid_trend_method_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rnt_triggered_ball import RNTTriggeredBallTask

        coord = mock_coordinator(mask_names=["RNT", "ContA", "DefaultA"])
        with pytest.raises(ValueError, match="trend_method"):
            RNTTriggeredBallTask(coord, trend_method="bogus")
