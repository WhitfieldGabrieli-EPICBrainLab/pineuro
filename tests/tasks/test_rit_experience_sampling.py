"""Tests for pineuro.tasks.rit_experience_sampling — trigger logic and registration."""


# =============================================================================
# Registration
# =============================================================================


class TestRITExperienceSamplingRegistration:
    def test_task_registered_in_feedback_tasks(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks(role="feedback")
        assert "ritexperiencesampling" in tasks

    def test_task_metadata(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("ritexperiencesampling")
        assert meta.task_id == "ritexperiencesampling"
        assert meta.display_name == "RIT Experience Sampling"
        assert meta.n_networks == 1
        assert meta.design_type == "nuisance_only"
        assert "feedback" in meta.roles

    def test_declares_triggered_event_report(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("ritexperiencesampling")
        assert meta.triggered_event_report is not None
        assert meta.triggered_event_report.label == "Triggered Prompt Durations"
        assert meta.triggered_event_report.episode_noun == "Prompt"

    def test_task_parameters_defined(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("ritexperiencesampling")
        param_keys = {p.key for p in meta.parameters}
        assert "activation_contrast" in param_keys
        assert "consecutive_volumes" in param_keys
        assert "activation_trend" in param_keys
        assert "trend_method" in param_keys
        assert "min_prompt_interval" in param_keys
        assert "timeout_duration" in param_keys
        assert "rating_timeout" in param_keys

    def test_task_class_registered_in_factory(self):
        from pineuro.tasks.factory import get_registered_task_ids

        ids = get_registered_task_ids()
        assert "ritexperiencesampling" in ids

    def test_import_from_tasks_package(self):
        from pineuro.tasks import RITExperienceSamplingTask

        assert RITExperienceSamplingTask.task_id == "ritexperiencesampling"


# =============================================================================
# TriggerState
# =============================================================================


class TestTriggerState:
    def test_default_values(self):
        from pineuro.tasks.rit_experience_sampling import TriggerState

        state = TriggerState()
        assert state.last_prompt_time == -float("inf")
        assert state.n_prompts_triggered == 0
        assert state.n_prompts_timeout == 0
        assert state.last_processed_volume == 0
        assert len(state.activation_history) == 0

    def test_activation_history_maxlen(self):
        from pineuro.tasks.rit_experience_sampling import TriggerState

        state = TriggerState()
        for i in range(100):
            state.activation_history.append(float(i))
        assert len(state.activation_history) == 50  # maxlen=50


# =============================================================================
# Trigger detection (_check_trigger)
# =============================================================================


class TestTriggerDetection:
    """Test the trigger detection algorithm in isolation.

    We create an RITExperienceSamplingTask-like object with just enough state
    to test _check_trigger() without needing PsychoPy or a coordinator.
    """

    def _make_task_stub(self, **kwargs):
        """Create a minimal stub with trigger detection capability."""
        from pineuro.tasks.rit_experience_sampling import RITExperienceSamplingTask, TriggerState

        class TriggerStub:
            """Minimal stub mimicking RITExperienceSamplingTask for trigger testing."""

            pass

        stub = TriggerStub()
        stub.activation_contrast = kwargs.get("activation_contrast", "positive")
        stub.consecutive_volumes = kwargs.get("consecutive_volumes", 3)
        stub.activation_trend = kwargs.get("activation_trend", "increasing")
        stub.trend_method = kwargs.get("trend_method", "strict")
        stub.min_prompt_interval = kwargs.get("min_prompt_interval", 30.0)
        stub.timeout_duration = kwargs.get("timeout_duration", 60.0)
        stub.use_activation_condition = kwargs.get("use_activation_condition", True)
        stub.activation_threshold = kwargs.get("activation_threshold", 0.0)
        stub.use_trend_condition = kwargs.get("use_trend_condition", True)
        stub.slope_threshold = kwargs.get("slope_threshold", 0.0)
        stub.use_mean_condition = kwargs.get("use_mean_condition", False)
        stub.mean_threshold = kwargs.get("mean_threshold", 0.0)
        stub.use_count_condition = kwargs.get("use_count_condition", False)
        stub.count_min = kwargs.get("count_min", 3)
        stub.trigger_logic = kwargs.get("trigger_logic", "all")
        stub._trigger_state = TriggerState()

        # Bind the methods. _check_trigger delegates to _evaluate_trigger, and
        # the log suffix uses _format_trigger_suffix, so all three are needed.
        stub._check_trigger = RITExperienceSamplingTask._check_trigger.__get__(stub)
        stub._evaluate_trigger = RITExperienceSamplingTask._evaluate_trigger.__get__(stub)
        stub._format_trigger_suffix = RITExperienceSamplingTask._format_trigger_suffix.__get__(stub)
        return stub

    def test_cooldown_blocks_trigger(self):
        stub = self._make_task_stub(min_prompt_interval=30.0)
        stub._trigger_state.last_prompt_time = 10.0  # prompt at t=10

        # t=20, only 10s since last prompt (< 30s cooldown)
        result = stub._check_trigger(20.0, 0.5)
        assert result is None

    def test_timeout_fires_after_duration(self):
        stub = self._make_task_stub(min_prompt_interval=30.0, timeout_duration=60.0)
        stub._trigger_state.last_prompt_time = 0.0  # prompt at t=0

        # t=60, exactly timeout_duration since last prompt
        result = stub._check_trigger(60.0, 0.0)
        assert result == "timeout"

    def test_timeout_fires_when_past_duration(self):
        stub = self._make_task_stub(min_prompt_interval=30.0, timeout_duration=60.0)
        stub._trigger_state.last_prompt_time = 0.0

        # t=75, well past timeout
        result = stub._check_trigger(75.0, -1.0)
        assert result == "timeout"

    def test_threshold_above_blocks_when_below(self):
        stub = self._make_task_stub(activation_contrast="positive", consecutive_volumes=2)
        stub._trigger_state.last_prompt_time = 0.0

        # Activation below 0, so threshold not met
        # Add trending history
        stub._trigger_state.activation_history.extend([-0.3, -0.2, -0.1])
        result = stub._check_trigger(35.0, -0.1)
        assert result is None

    def test_threshold_below_blocks_when_above(self):
        stub = self._make_task_stub(activation_contrast="negative", consecutive_volumes=2)
        stub._trigger_state.last_prompt_time = 0.0

        # Activation above 0, so "below" threshold not met
        stub._trigger_state.activation_history.extend([0.1, 0.2, 0.3])
        result = stub._check_trigger(35.0, 0.3)
        assert result is None

    def test_insufficient_history_blocks_trigger(self):
        stub = self._make_task_stub(consecutive_volumes=3)
        stub._trigger_state.last_prompt_time = 0.0

        # Only 2 volumes in history (need 3)
        stub._trigger_state.activation_history.extend([0.1, 0.2])
        result = stub._check_trigger(35.0, 0.2)
        assert result is None

    def test_increasing_trend_triggers(self):
        stub = self._make_task_stub(
            activation_contrast="positive",
            consecutive_volumes=3,
            activation_trend="increasing",
        )
        stub._trigger_state.last_prompt_time = 0.0

        # 3 increasing volumes, activation above 0
        stub._trigger_state.activation_history.extend([0.1, 0.3, 0.5])
        result = stub._check_trigger(35.0, 0.5)
        assert result == "triggered"

    def test_decreasing_trend_triggers(self):
        stub = self._make_task_stub(
            activation_contrast="negative",
            consecutive_volumes=3,
            activation_trend="decreasing",
        )
        stub._trigger_state.last_prompt_time = 0.0

        # 3 decreasing volumes, activation below 0
        stub._trigger_state.activation_history.extend([-0.1, -0.3, -0.5])
        result = stub._check_trigger(35.0, -0.5)
        assert result == "triggered"

    def test_non_monotonic_trend_blocks(self):
        stub = self._make_task_stub(
            activation_contrast="positive",
            consecutive_volumes=3,
            activation_trend="increasing",
        )
        stub._trigger_state.last_prompt_time = 0.0

        # Not monotonically increasing: 0.1, 0.5, 0.3
        stub._trigger_state.activation_history.extend([0.1, 0.5, 0.3])
        result = stub._check_trigger(35.0, 0.3)
        assert result is None

    def test_flat_trend_blocks(self):
        stub = self._make_task_stub(
            activation_contrast="positive",
            consecutive_volumes=3,
            activation_trend="increasing",
        )
        stub._trigger_state.last_prompt_time = 0.0

        # Flat: 0.3, 0.3, 0.3
        stub._trigger_state.activation_history.extend([0.3, 0.3, 0.3])
        result = stub._check_trigger(35.0, 0.3)
        assert result is None

    def test_longer_history_uses_last_n(self):
        """Trigger checks only the last `consecutive_volumes` entries."""
        stub = self._make_task_stub(
            activation_contrast="positive",
            consecutive_volumes=3,
            activation_trend="increasing",
        )
        stub._trigger_state.last_prompt_time = 0.0

        # Long history with decreasing start but increasing tail
        stub._trigger_state.activation_history.extend([0.9, 0.5, 0.1, 0.2, 0.4])
        result = stub._check_trigger(35.0, 0.4)
        assert result == "triggered"

    def test_timeout_takes_priority_over_threshold_check(self):
        """Timeout fires even when threshold is not met."""
        stub = self._make_task_stub(
            activation_contrast="positive",
            timeout_duration=60.0,
        )
        stub._trigger_state.last_prompt_time = 0.0

        # Activation below 0, but timeout elapsed
        result = stub._check_trigger(65.0, -0.5)
        assert result == "timeout"

    def test_cooldown_blocks_timeout(self):
        """Cooldown period blocks even timeout triggers."""
        stub = self._make_task_stub(
            min_prompt_interval=30.0,
            timeout_duration=60.0,
        )
        stub._trigger_state.last_prompt_time = 10.0

        # t=25, only 15s since prompt (< 30s cooldown), even though
        # time_since_last would be big if last_prompt_time were earlier
        result = stub._check_trigger(25.0, 0.5)
        assert result is None

    def test_consecutive_volumes_1_triggers_immediately(self):
        """With consecutive_volumes=1, only threshold matters."""
        stub = self._make_task_stub(
            activation_contrast="positive",
            consecutive_volumes=1,
            activation_trend="increasing",
        )
        stub._trigger_state.last_prompt_time = 0.0

        # Single volume above threshold
        stub._trigger_state.activation_history.extend([0.5])
        # With consecutive_volumes=1, we need at least 1 entry but the
        # trend check on a single element has no pairs to compare, so
        # all() over empty range returns True
        result = stub._check_trigger(35.0, 0.5)
        assert result == "triggered"


# =============================================================================
# Slope-mode trend detection
# =============================================================================


class TestSlopeModeTriggerDetection:
    """Slope-mode coverage; shares the stub-builder shape with TestTriggerDetection."""

    def _make_task_stub(self, **kwargs):
        from pineuro.tasks.rit_experience_sampling import RITExperienceSamplingTask, TriggerState

        class TriggerStub:
            pass

        stub = TriggerStub()
        stub.activation_contrast = kwargs.get("activation_contrast", "positive")
        stub.consecutive_volumes = kwargs.get("consecutive_volumes", 3)
        stub.activation_trend = kwargs.get("activation_trend", "increasing")
        stub.trend_method = kwargs.get("trend_method", "strict")
        stub.min_prompt_interval = kwargs.get("min_prompt_interval", 30.0)
        stub.timeout_duration = kwargs.get("timeout_duration", 60.0)
        stub.use_activation_condition = kwargs.get("use_activation_condition", True)
        stub.activation_threshold = kwargs.get("activation_threshold", 0.0)
        stub.use_trend_condition = kwargs.get("use_trend_condition", True)
        stub.slope_threshold = kwargs.get("slope_threshold", 0.0)
        stub.use_mean_condition = kwargs.get("use_mean_condition", False)
        stub.mean_threshold = kwargs.get("mean_threshold", 0.0)
        stub.use_count_condition = kwargs.get("use_count_condition", False)
        stub.count_min = kwargs.get("count_min", 3)
        stub.trigger_logic = kwargs.get("trigger_logic", "all")
        stub._trigger_state = TriggerState()
        stub._check_trigger = RITExperienceSamplingTask._check_trigger.__get__(stub)
        stub._evaluate_trigger = RITExperienceSamplingTask._evaluate_trigger.__get__(stub)
        stub._format_trigger_suffix = RITExperienceSamplingTask._format_trigger_suffix.__get__(stub)
        return stub

    def test_strict_rejects_blip(self):
        # Anchor: current strict behavior is preserved under the new param.
        stub = self._make_task_stub(
            trend_method="strict",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        stub._trigger_state.last_prompt_time = 30.0
        stub._trigger_state.activation_history.extend([0.1, 0.4, 0.3])
        assert stub._check_trigger(40.0, 0.3) is None

    def test_slope_accepts_blip(self):
        stub = self._make_task_stub(
            trend_method="slope",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        stub._trigger_state.last_prompt_time = 30.0
        stub._trigger_state.activation_history.extend([0.1, 0.4, 0.3])
        assert stub._check_trigger(40.0, 0.3) == "triggered"

    def test_slope_rejects_when_last_value_below_threshold(self):
        stub = self._make_task_stub(
            trend_method="slope",
            activation_contrast="positive",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        stub._trigger_state.last_prompt_time = 30.0
        stub._trigger_state.activation_history.extend([-0.5, -0.3, -0.1])
        assert stub._check_trigger(40.0, -0.1) is None

    def test_slope_negative_contrast_with_blip(self):
        stub = self._make_task_stub(
            trend_method="slope",
            activation_contrast="negative",
            activation_trend="decreasing",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        stub._trigger_state.last_prompt_time = 30.0
        stub._trigger_state.activation_history.extend([-0.1, -0.4, -0.3])
        assert stub._check_trigger(40.0, -0.3) == "triggered"

    def test_slope_rejects_when_trend_wrong_direction(self):
        stub = self._make_task_stub(
            trend_method="slope",
            activation_contrast="positive",
            activation_trend="increasing",
            consecutive_volumes=3,
            min_prompt_interval=0.0,
        )
        stub._trigger_state.last_prompt_time = 30.0
        stub._trigger_state.activation_history.extend([0.5, 0.4, 0.3])
        # Threshold passes (last value > 0) but slope is negative.
        assert stub._check_trigger(40.0, 0.3) is None


# =============================================================================
# Constructor validation
# =============================================================================


class TestConstructorValidation:
    def test_invalid_trend_method_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_experience_sampling import RITExperienceSamplingTask

        coord = mock_coordinator(mask_names=["RIT"])
        with pytest.raises(ValueError, match="trend_method"):
            RITExperienceSamplingTask(coord, trend_method="bogus")


# =============================================================================
# Log suffix (_format_trigger_suffix)
# =============================================================================


class TestLogSuffix(TestTriggerDetection):
    """Per-volume log suffix built from the shared trigger evaluation.

    Inherits _make_task_stub from TestTriggerDetection. The suffix and the real
    trigger both route through _evaluate_trigger, so these also confirm the log
    can't disagree with the decision.
    """

    def _suffix(self, stub, current_time, activation):
        time_since_last = current_time - stub._trigger_state.last_prompt_time
        history = list(stub._trigger_state.activation_history)
        ev = stub._evaluate_trigger(time_since_last, activation, history)
        return stub._format_trigger_suffix(ev)

    def test_warmup_while_history_filling(self):
        stub = self._make_task_stub(consecutive_volumes=3, min_prompt_interval=0.0)
        stub._trigger_state.last_prompt_time = 0.0
        stub._trigger_state.activation_history.extend([0.1, 0.5])

        suffix = self._suffix(stub, 35.0, 0.5)
        assert suffix == " | warmup 2/3"
        assert suffix.isascii()

    def test_slope_trigger_shows_both_gates_and_verdict(self):
        stub = self._make_task_stub(trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0)
        stub._trigger_state.last_prompt_time = 0.0
        stub._trigger_state.activation_history.extend([0.1, 0.4, 0.9])

        suffix = self._suffix(stub, 35.0, 0.9)
        assert suffix.startswith(" | [all] ")
        assert "act>0 ok" in suffix
        assert "slope +" in suffix and ">0 ok" in suffix
        assert suffix.endswith("=> TRIGGER")
        assert suffix.isascii()
        assert stub._check_trigger(35.0, 0.9) == "triggered"

    def test_slope_blocked_trend_has_no_trailer(self):
        stub = self._make_task_stub(trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0)
        stub._trigger_state.last_prompt_time = 0.0
        stub._trigger_state.activation_history.extend([0.9, 0.4, 0.3])

        suffix = self._suffix(stub, 35.0, 0.3)
        assert "slope " in suffix and ">0 no" in suffix
        assert "act>0 ok" in suffix
        assert "=>" not in suffix
        assert stub._check_trigger(35.0, 0.3) is None

    def test_cooldown_waiting_trailer_rounds_up(self):
        stub = self._make_task_stub(trend_method="slope", consecutive_volumes=3, min_prompt_interval=30.0)
        stub._trigger_state.last_prompt_time = 10.5  # 4.5s ago at t=15 -> in cooldown
        stub._trigger_state.activation_history.extend([0.1, 0.4, 0.9])

        suffix = self._suffix(stub, 15.0, 0.9)
        assert suffix.endswith("=> waiting (cooldown 26s)")
        assert suffix.isascii()
        assert stub._check_trigger(15.0, 0.9) is None

    def test_strict_shows_progress_fraction(self):
        stub = self._make_task_stub(trend_method="strict", consecutive_volumes=3, min_prompt_interval=0.0)
        stub._trigger_state.last_prompt_time = 0.0
        # Largest increasing run ending at current value is [0.4, 0.9] -> 2/3.
        stub._trigger_state.activation_history.extend([0.9, 0.4, 0.9])

        suffix = self._suffix(stub, 35.0, 0.9)
        assert suffix == " | [all] act>0 ok | trend increasing 2/3 no"
        assert "=>" not in suffix

    def test_timeout_trailer(self):
        stub = self._make_task_stub(
            trend_method="slope", consecutive_volumes=3, min_prompt_interval=30.0, timeout_duration=60.0
        )
        stub._trigger_state.last_prompt_time = 0.0
        stub._trigger_state.activation_history.extend([0.1, 0.4, 0.9])

        suffix = self._suffix(stub, 70.0, 0.9)
        assert suffix.endswith("=> TIMEOUT (no trigger in 60s)")
        assert suffix.isascii()
        assert stub._check_trigger(70.0, 0.9) == "timeout"


class TestConfigurableTriggerConditions(TestTriggerDetection):
    """Configurable condition gates: thresholds, new metrics, and combine logic.

    Inherits _make_task_stub from TestTriggerDetection.
    """

    def _seed(self, stub, history, last_prompt_time=0.0):
        stub._trigger_state.last_prompt_time = last_prompt_time
        stub._trigger_state.activation_history.extend(history)

    def test_defaults_reproduce_activation_and_slope(self):
        stub = self._make_task_stub(trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0)
        self._seed(stub, [0.1, 0.4, 0.9])
        assert stub._check_trigger(35.0, 0.9) == "triggered"

    def test_activation_threshold_blocks_below(self):
        stub = self._make_task_stub(
            trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0, activation_threshold=1.5
        )
        self._seed(stub, [0.1, 0.4, 0.9])
        assert stub._check_trigger(35.0, 0.9) is None

    def test_activation_threshold_fires_above(self):
        stub = self._make_task_stub(
            trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0, activation_threshold=1.5
        )
        self._seed(stub, [1.0, 1.6, 2.0])
        assert stub._check_trigger(35.0, 2.0) == "triggered"

    def test_slope_threshold_requires_steeper_rise(self):
        stub = self._make_task_stub(
            trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0, slope_threshold=0.5
        )
        self._seed(stub, [0.8, 0.9, 1.0])
        assert stub._check_trigger(35.0, 1.0) is None

    def test_mean_condition_only(self):
        stub = self._make_task_stub(
            consecutive_volumes=3,
            min_prompt_interval=0.0,
            use_activation_condition=False,
            use_trend_condition=False,
            use_mean_condition=True,
            mean_threshold=1.0,
        )
        self._seed(stub, [1.2, 1.4, 1.6])
        assert stub._check_trigger(35.0, 1.6) == "triggered"

    def test_mean_condition_blocks_below(self):
        stub = self._make_task_stub(
            consecutive_volumes=3,
            min_prompt_interval=0.0,
            use_activation_condition=False,
            use_trend_condition=False,
            use_mean_condition=True,
            mean_threshold=1.0,
        )
        self._seed(stub, [0.2, 0.4, 0.6])
        assert stub._check_trigger(35.0, 0.6) is None

    def test_count_condition_only(self):
        stub = self._make_task_stub(
            consecutive_volumes=3,
            min_prompt_interval=0.0,
            use_activation_condition=False,
            use_trend_condition=False,
            use_count_condition=True,
            activation_threshold=1.0,
            count_min=2,
        )
        self._seed(stub, [1.5, 0.2, 1.5])
        assert stub._check_trigger(35.0, 1.5) == "triggered"
        stub2 = self._make_task_stub(
            consecutive_volumes=3,
            min_prompt_interval=0.0,
            use_activation_condition=False,
            use_trend_condition=False,
            use_count_condition=True,
            activation_threshold=1.0,
            count_min=2,
        )
        self._seed(stub2, [1.5, 0.2, 0.2])
        assert stub2._check_trigger(35.0, 0.2) is None

    def test_trigger_logic_any_fires_on_single_gate(self):
        stub = self._make_task_stub(
            trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0, trigger_logic="any"
        )
        self._seed(stub, [0.9, 0.5, 0.4])  # slope negative, current > 0
        assert stub._check_trigger(35.0, 0.4) == "triggered"

    def test_trigger_logic_all_requires_every_gate(self):
        stub = self._make_task_stub(
            trend_method="slope", consecutive_volumes=3, min_prompt_interval=0.0, trigger_logic="all"
        )
        self._seed(stub, [0.9, 0.5, 0.4])
        assert stub._check_trigger(35.0, 0.4) is None

    def test_activation_only_needs_no_warmup(self):
        stub = self._make_task_stub(
            consecutive_volumes=3,
            min_prompt_interval=0.0,
            use_activation_condition=True,
            use_trend_condition=False,
        )
        self._seed(stub, [0.5])
        assert stub._check_trigger(35.0, 0.5) == "triggered"


class TestTriggerConditionValidation:
    def test_no_condition_enabled_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_experience_sampling import RITExperienceSamplingTask

        coord = mock_coordinator(mask_names=["RIT"])
        with pytest.raises(ValueError, match="At least one trigger condition"):
            RITExperienceSamplingTask(
                coord,
                use_activation_condition=False,
                use_trend_condition=False,
                use_mean_condition=False,
                use_count_condition=False,
            )

    def test_invalid_trigger_logic_raises(self, mock_coordinator):
        import pytest

        from pineuro.tasks.rit_experience_sampling import RITExperienceSamplingTask

        coord = mock_coordinator(mask_names=["RIT"])
        with pytest.raises(ValueError, match="trigger_logic"):
            RITExperienceSamplingTask(coord, trigger_logic="both")

    def test_cross_validator_rejects_count_min_over_window(self):
        import pytest

        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("ritexperiencesampling")
        with pytest.raises(ValueError, match="Count Min"):
            meta.validate_config({"use_count_condition": True, "count_min": 5, "consecutive_volumes": 3})

    def test_cross_validator_rejects_no_condition(self):
        import pytest

        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("ritexperiencesampling")
        with pytest.raises(ValueError, match="At least one trigger condition"):
            meta.validate_config(
                {
                    "use_activation_condition": False,
                    "use_trend_condition": False,
                    "use_mean_condition": False,
                    "use_count_condition": False,
                }
            )


class TestEventsFileNaming:
    def test_events_prefix_carries_task_entity_and_attempt(self, mock_coordinator, tmp_path):
        from pineuro.tasks.rit_experience_sampling import RITExperienceSamplingTask

        coord = mock_coordinator(
            mask_names=["RIT"],
            session="ses-01",
            output_dir=str(tmp_path),
            task="ritexperiencesamplingtransfer",
            attempt=3,
        )
        task = RITExperienceSamplingTask(coord)

        # Same prefix as the feedback log: re-attempts and transfer runs get their own events file.
        assert task._bids_run_prefix == "sub-01_ses-01_task-ritexperiencesamplingtransfer_run-01_attempt-03"


# =============================================================================
# Instructions — key names
# =============================================================================


class TestInstructionKeyNames:
    LABELS = {"left_key_label": "index finger", "right_key_label": "middle finger", "confirm_key_label": "thumb"}

    @staticmethod
    def _make_task(mock_coordinator, tmp_path, config):
        from pineuro.tasks.factory import create_task

        coord = mock_coordinator(
            mask_names=["RIT"], session="ses-01", output_dir=str(tmp_path), task="ritexperiencesampling"
        )
        return create_task("ritexperiencesampling", coord, config, output_dir=str(tmp_path))

    def test_label_parameters_default_blank(self):
        from pineuro.tasks.task_registry import get_task_metadata

        params = {p.key: p for p in get_task_metadata("ritexperiencesampling").parameters}
        for key in self.LABELS:
            assert params[key].type == "str"
            assert params[key].default == ""

    def test_blank_labels_name_the_keys(self, mock_coordinator, tmp_path):
        task = self._make_task(mock_coordinator, tmp_path, {"left_key": "3"})
        assert task._key_names() == ("button 3", "button 2", "button 0")

    def test_study_labels_replace_key_names(self, mock_coordinator, tmp_path):
        task = self._make_task(mock_coordinator, tmp_path, self.LABELS)
        assert task._key_names() == ("index finger", "middle finger", "thumb")
