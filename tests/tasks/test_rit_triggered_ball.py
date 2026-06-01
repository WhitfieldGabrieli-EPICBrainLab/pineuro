"""Tests for pineuro.tasks.rit_triggered_ball — metadata, trigger detection, TriggerState."""


class TestRITTriggeredBallMetadata:
    def test_registered_in_registry(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "rittriggeredball" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rittriggeredball")
        assert meta.task_id == "rittriggeredball"
        # Only the trigger/movement network is mandatory; movement override and
        # the dual-mode networks are optional roles.
        assert meta.n_networks == 1

    def test_declares_triggered_event_report(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rittriggeredball")
        assert meta.triggered_event_report is not None
        assert meta.triggered_event_report.label == "Triggered Episode Durations"
        assert meta.triggered_event_report.episode_noun == "Episode"

    def test_has_network_roles(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rittriggeredball")
        roles = {r.key: r for r in meta.network_roles}
        assert "trigger_network" in roles
        assert "movement_network" in roles
        assert "top_network" in roles
        assert "bottom_network" in roles

        # The trigger network is required; the rest are optional.
        assert roles["trigger_network"].optional is False
        assert roles["movement_network"].optional is True
        assert roles["top_network"].optional is True
        assert roles["bottom_network"].optional is True

        # Single-mode movement defaults to the trigger network.
        assert roles["movement_network"].default_from == "trigger_network"

    def test_task_parameters_defined(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rittriggeredball")
        param_keys = {p.key for p in meta.parameters}
        assert "movement_mode" in param_keys
        assert "movement_direction" in param_keys
        assert "activation_contrast" in param_keys
        assert "consecutive_volumes" in param_keys
        assert "activation_trend" in param_keys
        assert "trend_method" in param_keys
        assert "min_prompt_interval" in param_keys
        assert "timeout_duration" in param_keys

    def test_movement_param_defaults(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rittriggeredball")
        params = {p.key: p for p in meta.parameters}
        assert params["movement_mode"].default == "single"
        assert params["movement_mode"].choices == ["single", "dual"]
        assert params["movement_direction"].default == "positive_down"
        assert params["movement_direction"].choices == ["positive_down", "positive_up"]


def _make_task(**kwargs):
    import numpy as np

    from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask, TriggerState

    task = RITTriggeredBallTask.__new__(RITTriggeredBallTask)
    task.activation_contrast = kwargs.get("activation_contrast", "positive")
    task.consecutive_volumes = kwargs.get("consecutive_volumes", 3)
    task.activation_trend = kwargs.get("activation_trend", "increasing")
    task.trend_method = kwargs.get("trend_method", "strict")
    task.min_prompt_interval = kwargs.get("min_prompt_interval", 10.0)
    task.timeout_duration = kwargs.get("timeout_duration", 60.0)
    task._trigger_state = TriggerState()

    # Movement state so _calculate_ball_movement can be exercised directly
    # (no PsychoPy / coordinator needed).
    movement_direction = kwargs.get("movement_direction", "positive_down")
    task.movement_mode = kwargs.get("movement_mode", "single")
    task.movement_direction = movement_direction
    task._direction_sign = -1.0 if movement_direction == "positive_down" else 1.0
    task._movement_mask = kwargs.get("movement_mask", "RIT")
    task._ball_mask_names = kwargs.get("ball_mask_names", [task._movement_mask])
    task._network_directions = np.array([1, -1])
    task.pda_outlier_threshold = kwargs.get("pda_outlier_threshold", 2.0)
    task._current_activity = 0.0
    task._current_direction = 0.0
    task._current_is_outlier = False
    return task


class TestRITTriggeredBallTriggerDetection:
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
        from pineuro.tasks.rit_triggered_ball import TriggerState

        state = TriggerState()
        assert state.n_episodes_triggered == 0
        assert state.n_episodes_timeout == 0
        assert state.last_processed_volume == 0
        assert len(state.activation_history) == 0


class TestLeastSquaresSlope:
    def test_empty_returns_zero(self):
        from pineuro.tasks.rit_triggered_ball import _least_squares_slope

        assert _least_squares_slope([]) == 0.0

    def test_single_value_returns_zero(self):
        from pineuro.tasks.rit_triggered_ball import _least_squares_slope

        assert _least_squares_slope([0.5]) == 0.0

    def test_monotonic_increase_positive_slope(self):
        from pineuro.tasks.rit_triggered_ball import _least_squares_slope

        assert _least_squares_slope([0.0, 1.0, 2.0]) > 0

    def test_monotonic_decrease_negative_slope(self):
        from pineuro.tasks.rit_triggered_ball import _least_squares_slope

        assert _least_squares_slope([2.0, 1.0, 0.0]) < 0

    def test_blip_does_not_flip_sign(self):
        # Net rise with one down-blip: slope still positive.
        from pineuro.tasks.rit_triggered_ball import _least_squares_slope

        assert _least_squares_slope([0.1, 0.4, 0.3]) > 0


class TestRITTriggeredBallSlopeMode:
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


class TestSingleMovement:
    """Single-network movement: magnitude from |z|, direction from signed activation."""

    def test_positive_down_positive_activation_moves_down(self):
        task = _make_task(movement_mode="single", movement_direction="positive_down", movement_mask="RIT")
        task._calculate_ball_movement({"RIT": 1.0})
        assert task._current_is_outlier is False
        assert task._current_direction == -1.0  # more rumination -> ball down
        assert task._current_activity == 0.1  # |1.0| / 10.0

    def test_positive_down_negative_activation_moves_up(self):
        # Main use case: less rumination (negative activation) sends the ball up.
        task = _make_task(movement_mode="single", movement_direction="positive_down", movement_mask="RIT")
        task._calculate_ball_movement({"RIT": -1.0})
        assert task._current_direction == 1.0
        assert task._current_activity == 0.1

    def test_positive_up_flips_direction(self):
        task = _make_task(movement_mode="single", movement_direction="positive_up", movement_mask="RIT")
        task._calculate_ball_movement({"RIT": 1.0})
        assert task._current_direction == 1.0

    def test_outlier_rejected(self):
        task = _make_task(movement_mode="single", movement_mask="RIT", pda_outlier_threshold=2.0)
        task._calculate_ball_movement({"RIT": 3.0})
        assert task._current_is_outlier is True
        assert task._current_activity == 0.0  # unchanged — movement skipped

    def test_movement_override_reads_other_network(self):
        # Movement mask differs from the trigger network.
        task = _make_task(movement_mode="single", movement_direction="positive_down", movement_mask="ContA")
        task._calculate_ball_movement({"RIT": 0.0, "ContA": -2.0})
        assert task._current_direction == 1.0  # negative ContA -> up
        assert task._current_activity == 0.2  # |-2.0| / 10.0


class TestDualMovement:
    """Dual-network movement (regression): magnitude from the activation difference."""

    def test_top_more_active_moves_up(self):
        task = _make_task(movement_mode="dual", ball_mask_names=["ContA", "DefaultA"])
        task._calculate_ball_movement({"ContA": 0.8, "DefaultA": 0.3})
        assert task._current_direction == 1  # network_directions[argmax] = +1
        assert task._current_activity == 0.05  # |0.8 - 0.3| / 10.0

    def test_bottom_more_active_moves_down(self):
        task = _make_task(movement_mode="dual", ball_mask_names=["ContA", "DefaultA"])
        task._calculate_ball_movement({"ContA": 0.3, "DefaultA": 0.8})
        assert task._current_direction == -1


class TestNetworkRoleDefaults:
    def test_defaults(self):
        from pineuro.tasks.task_registry import NetworkRole

        role = NetworkRole("x", "X")
        assert role.optional is False
        assert role.default_from is None
        assert role.visible_when is None


class TestConstructorValidation:
    def test_invalid_trend_method_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask

        coord = mock_coordinator(mask_names=["RIT", "ContA", "DefaultA"])
        with pytest.raises(ValueError, match="trend_method"):
            RITTriggeredBallTask(coord, trend_method="bogus")

    def test_single_mask_constructs(self, mock_coordinator, tmp_path):
        from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask

        coord = mock_coordinator(mask_names=["RIT"], session="ses-01", output_dir=str(tmp_path))
        task = RITTriggeredBallTask(coord, movement_mode="single")
        assert task.mask_name == "RIT"
        assert task._movement_mask == "RIT"  # defaults to the trigger network
        assert task._ball_mask_names == ["RIT"]

    def test_single_mask_movement_override(self, mock_coordinator, tmp_path):
        from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask

        coord = mock_coordinator(mask_names=["RIT", "ContA"], session="ses-01", output_dir=str(tmp_path))
        task = RITTriggeredBallTask(coord, mask_names=["RIT", "ContA"], movement_mode="single")
        assert task.mask_name == "RIT"
        assert task._movement_mask == "ContA"  # override

    def test_dual_mode_requires_three_masks(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask

        coord = mock_coordinator(mask_names=["RIT"])
        with pytest.raises(ValueError, match="dual movement mode requires 3"):
            RITTriggeredBallTask(coord, movement_mode="dual")

    def test_invalid_movement_mode_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask

        coord = mock_coordinator(mask_names=["RIT"])
        with pytest.raises(ValueError, match="movement_mode"):
            RITTriggeredBallTask(coord, movement_mode="bogus")

    def test_invalid_movement_direction_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_triggered_ball import RITTriggeredBallTask

        coord = mock_coordinator(mask_names=["RIT"])
        with pytest.raises(ValueError, match="movement_direction"):
            RITTriggeredBallTask(coord, movement_direction="bogus")


class TestRITTriggeredBallLogSuffix:
    """Per-volume log suffix built from the shared trigger evaluation.

    The suffix and the real trigger both route through _evaluate_trigger, so
    these assertions also confirm the log can't disagree with the decision.
    """

    def _suffix(self, task, current_time, activation):
        time_since_last = current_time - task._trigger_state.last_prompt_time
        history = list(task._trigger_state.activation_history)
        ev = task._evaluate_trigger(time_since_last, activation, history)
        return task._format_trigger_suffix(ev)

    def test_warmup_while_history_filling(self):
        from collections import deque

        task = _make_task(consecutive_volumes=3, min_prompt_interval=0.0)
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([0.1, 0.5], maxlen=50)

        suffix = self._suffix(task, 50.0, 0.5)
        assert suffix == " | warmup 2/3"
        assert suffix.isascii()

    def test_slope_trigger_shows_both_gates_and_verdict(self):
        from collections import deque

        task = _make_task(trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0)
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([0.1, 0.4, 0.9], maxlen=50)

        suffix = self._suffix(task, 50.0, 0.9)
        assert suffix.startswith(" | trend slope +")
        assert "need>0 ok" in suffix
        assert "thresh act>0 ok" in suffix
        assert suffix.endswith("=> TRIGGER")
        assert suffix.isascii()
        # Must match the authoritative decision.
        assert task._check_trigger(50.0, 0.9) == "triggered"

    def test_slope_blocked_trend_has_no_trailer(self):
        from collections import deque

        task = _make_task(trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0)
        task._trigger_state.last_prompt_time = 40.0
        task._trigger_state.activation_history = deque([0.9, 0.4, 0.3], maxlen=50)

        suffix = self._suffix(task, 50.0, 0.3)
        assert "need>0 no" in suffix
        assert "thresh act>0 ok" in suffix
        assert "=>" not in suffix
        assert task._check_trigger(50.0, 0.3) is None

    def test_cooldown_waiting_trailer_rounds_up(self):
        from collections import deque

        task = _make_task(trend_method="slope", consecutive_volumes=3, min_prompt_interval=10.0)
        task._trigger_state.last_prompt_time = 45.5  # 4.5s ago -> still in cooldown
        task._trigger_state.activation_history = deque([0.1, 0.4, 0.9], maxlen=50)

        suffix = self._suffix(task, 50.0, 0.9)
        assert suffix.endswith("=> waiting (cooldown 6s)")
        assert suffix.isascii()
        assert task._check_trigger(50.0, 0.9) is None

    def test_strict_shows_progress_fraction(self):
        from collections import deque

        task = _make_task(trend_method="strict", consecutive_volumes=3, min_prompt_interval=0.0)
        task._trigger_state.last_prompt_time = 40.0
        # Largest increasing run ending at current value is [0.4, 0.9] -> 2/3.
        task._trigger_state.activation_history = deque([0.9, 0.4, 0.9], maxlen=50)

        suffix = self._suffix(task, 50.0, 0.9)
        assert suffix == " | trend increasing 2/3 | thresh act>0 ok"
        assert "=>" not in suffix

    def test_timeout_trailer(self):
        from collections import deque

        task = _make_task(trend_method="slope", consecutive_volumes=3, min_prompt_interval=10.0, timeout_duration=60.0)
        task._trigger_state.last_prompt_time = 0.0
        task._trigger_state.activation_history = deque([0.1, 0.4, 0.9], maxlen=50)

        suffix = self._suffix(task, 70.0, 0.9)
        assert suffix.endswith("=> TIMEOUT (no trigger in 60s)")
        assert suffix.isascii()
        assert task._check_trigger(70.0, 0.9) == "timeout"
