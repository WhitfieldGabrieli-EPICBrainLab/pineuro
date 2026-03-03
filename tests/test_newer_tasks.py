"""Tests for newer task types — RNT, gauge, rnt_triggered_ball, protocol."""

import json
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Protocol module — runtime_checkable protocols
# =============================================================================


class TestProtocols:
    def test_coordinator_protocol_importable(self):
        from pineuro.tasks.protocol import CoordinatorProtocol

        # Annotated attributes live in __annotations__, not as class attrs
        annotations = CoordinatorProtocol.__annotations__
        assert "subject" in annotations
        assert "n_volumes" in annotations

    def test_task_protocol_importable(self):
        from pineuro.tasks.protocol import TaskProtocol

        # Methods are defined on Protocol class, accessible via hasattr
        assert hasattr(TaskProtocol, "setup")
        assert hasattr(TaskProtocol, "update")
        assert hasattr(TaskProtocol, "run")
        assert hasattr(TaskProtocol, "cleanup")

    def test_coordinator_protocol_is_runtime_checkable(self):
        from pineuro.tasks.protocol import CoordinatorProtocol

        mock = MagicMock()
        mock.subject = "sub-01"
        mock.task = "ball"
        mock.run_number = 1
        mock.acq = None
        mock.n_volumes = 100
        mock.mask_names = ["DMN"]
        mock.is_complete = MagicMock(return_value=False)
        mock.check_error = MagicMock()
        mock.was_stopped = MagicMock(return_value=False)
        mock.should_show_questions = MagicMock(return_value=True)
        mock.send_trigger = MagicMock()

        assert isinstance(mock, CoordinatorProtocol)

    def test_task_protocol_is_runtime_checkable(self):
        from pineuro.tasks.protocol import TaskProtocol

        mock = MagicMock()
        mock.setup = MagicMock()
        mock.update = MagicMock()
        mock.run = MagicMock()
        mock.cleanup = MagicMock()
        mock.get_run_results = MagicMock(return_value={})
        mock.get_mirror_state = MagicMock(return_value={})
        mock.get_volume_log_suffix = MagicMock(return_value="")

        assert isinstance(mock, TaskProtocol)


# =============================================================================
# RNT task — metadata and keyword loading
# =============================================================================


class TestRNTTaskMetadata:
    def test_registered_in_registry(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "rnt" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rnt")
        assert meta.task_id == "rnt"
        assert "localizer" in meta.roles
        assert meta.n_networks == 0
        assert "rumination_network" in meta.default_contrasts

    def test_has_parameters(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rnt")
        param_keys = [p.key for p in meta.parameters]
        assert "keywords_dir" in param_keys
        assert "rumination_duration" in param_keys
        assert "n_trials_per_run" in param_keys


class TestRNTKeywordParsing:
    def test_parse_json_keywords(self, tmp_path):
        from pineuro.tasks.rnt import RNTTask

        # Create a JSON keywords file
        keywords = {
            "memories": [
                {"keyword": "loss", "pass": 1},
                {"keyword": "shame", "pass": 1},
                {"keyword": "anger", "pass": 1},
                {"keyword": "fear", "pass": 1},
                {"keyword": "lonely", "pass": 2},
                {"keyword": "rejected", "pass": 2},
                {"keyword": "hurt", "pass": 2},
                {"keyword": "sad", "pass": 2},
            ]
        }
        kw_file = tmp_path / "sub-01_rnt_keywords.json"
        kw_file.write_text(json.dumps(keywords))

        task = RNTTask.__new__(RNTTask)
        result = task._parse_keywords_file(kw_file)

        assert len(result) == 8
        assert result[0]["keyword"] == "loss"
        assert result[0]["pass"] == 1

    def test_parse_tsv_keywords(self, tmp_path):
        from pineuro.tasks.rnt import RNTTask

        content = "keyword\tpass\nloss\t1\nshame\t1\nanger\t1\nfear\t1\nlonely\t2\nrejected\t2\nhurt\t2\nsad\t2\n"
        kw_file = tmp_path / "keywords.tsv"
        kw_file.write_text(content)

        task = RNTTask.__new__(RNTTask)
        result = task._parse_keywords_file(kw_file)

        assert len(result) == 8
        assert result[4]["keyword"] == "lonely"
        assert result[4]["pass"] == 2

    def test_resolve_keywords_file_direct(self, tmp_path):
        from pineuro.tasks.rnt import RNTTask

        kw_file = tmp_path / "keywords.json"
        kw_file.write_text(json.dumps({"memories": [{"keyword": "test", "pass": 1}]}))

        task = RNTTask.__new__(RNTTask)
        task.keywords_dir = None
        task._keywords_file_override = kw_file
        task.participant_id = "sub-01"

        result = task._resolve_keywords_file()
        assert result == kw_file

    def test_resolve_keywords_file_missing_raises(self, tmp_path):
        from pineuro.tasks.rnt import RNTTask

        task = RNTTask.__new__(RNTTask)
        task.keywords_dir = None
        task._keywords_file_override = tmp_path / "nonexistent.json"
        task.participant_id = "sub-01"

        with pytest.raises(FileNotFoundError):
            task._resolve_keywords_file()

    def test_resolve_keywords_file_by_glob(self, tmp_path):
        from pineuro.tasks.rnt import RNTTask

        kw_dir = tmp_path / "keywords"
        kw_dir.mkdir()
        kw_file = kw_dir / "sub-01_rnt_keywords.json"
        kw_file.write_text(json.dumps({"memories": [{"keyword": "test", "pass": 1}]}))

        task = RNTTask.__new__(RNTTask)
        task.keywords_dir = kw_dir
        task._keywords_file_override = None
        task.participant_id = "sub-01"

        result = task._resolve_keywords_file()
        assert result == kw_file


# =============================================================================
# RNT prompts
# =============================================================================


class TestRNTPrompts:
    def test_memory_prompts_not_empty(self):
        from pineuro.tasks.rnt import MEMORY_PROMPTS

        assert len(MEMORY_PROMPTS) >= 4

    def test_rumination_prompts_not_empty(self):
        from pineuro.tasks.rnt import RUMINATION_PROMPTS

        assert len(RUMINATION_PROMPTS) >= 4

    def test_distraction_prompts_not_empty(self):
        from pineuro.tasks.rnt import DISTRACTION_PROMPTS

        assert len(DISTRACTION_PROMPTS) >= 4


# =============================================================================
# Gauge task — metadata and registration
# =============================================================================


class TestGaugeTaskMetadata:
    def test_registered_in_registry(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "gauge" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("gauge")
        assert meta.task_id == "gauge"
        assert meta.n_networks == 2

    def test_has_network_roles(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("gauge")
        role_keys = [r.key for r in meta.network_roles]
        assert "focus_network" in role_keys
        assert "wander_network" in role_keys

    def test_has_parameters(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("gauge")
        param_keys = [p.key for p in meta.parameters]
        assert "block_durations" in param_keys
        assert "score_factor" in param_keys
        assert "label_swap_after_block" in param_keys


class TestGaugeTaskScoreUpdate:
    def test_score_update_positive_differential(self):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task.mask_names = ["ContA", "DefaultA"]
        task.score = 90.0
        task.score_factor = 3.0
        task._direction_sign = 1.0

        # Focus > wander → score should increase
        task._update_score({"ContA": 2.0, "DefaultA": 0.5})
        assert task.score > 90.0

    def test_score_update_bounded(self):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task.mask_names = ["ContA", "DefaultA"]
        task.score = 179.0
        task.score_factor = 10.0
        task._direction_sign = 1.0

        task._update_score({"ContA": 100.0, "DefaultA": 0.0})
        assert task.score == 180.0  # Clamped

    def test_score_update_lower_bound(self):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task.mask_names = ["ContA", "DefaultA"]
        task.score = 1.0
        task.score_factor = 10.0
        task._direction_sign = 1.0

        task._update_score({"ContA": 0.0, "DefaultA": 100.0})
        assert task.score == 0.0  # Clamped

    def test_direction_sign_flips_movement(self):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task.mask_names = ["ContA", "DefaultA"]
        task.score = 90.0
        task.score_factor = 3.0

        # Same activations, opposite direction signs
        task._direction_sign = 1.0
        task._update_score({"ContA": 2.0, "DefaultA": 0.0})
        score_positive = task.score

        task.score = 90.0
        task._direction_sign = -1.0
        task._update_score({"ContA": 2.0, "DefaultA": 0.0})
        score_negative = task.score

        assert score_positive > 90.0
        assert score_negative < 90.0


# =============================================================================
# RNT Triggered Ball — metadata and trigger logic
# =============================================================================


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


class TestRNTTriggeredBallTriggerDetection:
    def _make_task(self, **kwargs):
        from pineuro.tasks.rnt_triggered_ball import RNTTriggeredBallTask, TriggerState

        task = RNTTriggeredBallTask.__new__(RNTTriggeredBallTask)
        task.activation_contrast = kwargs.get("activation_contrast", "positive")
        task.consecutive_volumes = kwargs.get("consecutive_volumes", 3)
        task.activation_trend = kwargs.get("activation_trend", "increasing")
        task.min_prompt_interval = kwargs.get("min_prompt_interval", 10.0)
        task.timeout_duration = kwargs.get("timeout_duration", 60.0)
        task._trigger_state = TriggerState()
        return task

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
