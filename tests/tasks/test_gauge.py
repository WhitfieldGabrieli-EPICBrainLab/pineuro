"""Tests for pineuro.tasks.gauge — GaugeTask metadata, score update, label swap, events."""

import csv
from types import SimpleNamespace


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


class TestGaugeLabelSwap:
    @staticmethod
    def _make_task(*, labels_swapped=False, direction_sign=1.0):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task._labels_swapped = labels_swapped
        task._direction_sign = direction_sign
        task._left_label = SimpleNamespace(text="Focused")
        task._right_label = SimpleNamespace(text="Wandering")
        return task

    def test_swap_toggles_state(self):
        task = self._make_task(labels_swapped=False)
        task._swap_label_positions()
        assert task._labels_swapped is True
        task._swap_label_positions()
        assert task._labels_swapped is False

    def test_swap_updates_label_text(self):
        task = self._make_task(labels_swapped=False)
        task._swap_label_positions()
        assert task._left_label.text == "Wandering"
        assert task._right_label.text == "Focused"
        # Swap back restores original labels
        task._swap_label_positions()
        assert task._left_label.text == "Focused"
        assert task._right_label.text == "Wandering"

    def test_swap_flips_direction_sign(self):
        task = self._make_task(direction_sign=1.0)
        task._swap_label_positions()
        assert task._direction_sign == -1.0
        task._swap_label_positions()
        assert task._direction_sign == 1.0

    def test_swap_effect_on_score_reverses_direction(self):
        """End-to-end: after a swap, the same activation profile moves the score the opposite way."""
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task.mask_names = ["ContA", "DefaultA"]
        task.score_factor = 3.0
        task._labels_swapped = False
        task._direction_sign = 1.0
        task._left_label = SimpleNamespace(text="Focused")
        task._right_label = SimpleNamespace(text="Wandering")

        task.score = 90.0
        task._update_score({"ContA": 2.0, "DefaultA": 0.0})
        before = task.score

        task._swap_label_positions()  # Flips direction
        task.score = 90.0
        task._update_score({"ContA": 2.0, "DefaultA": 0.0})
        after = task.score

        # Same input, opposite direction sign → score moves opposite way
        assert (before - 90.0) == -(after - 90.0)


class TestGaugeRecordEvent:
    @staticmethod
    def _make_task():
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task._events = []
        return task

    def test_basic_event_appended(self):
        task = self._make_task()
        task._record_event(onset=10.0, duration=4.0, trial_type="focus")
        assert task._events == [{"onset": 10.0, "duration": 4.0, "trial_type": "focus"}]

    def test_extras_merged(self):
        task = self._make_task()
        task._record_event(
            onset=0.0,
            duration=2.0,
            trial_type="button_press",
            response="1",
            response_time=0.42,
        )
        ev = task._events[0]
        assert ev["response"] == "1"
        assert ev["response_time"] == 0.42


class TestGaugeSaveEventsTsv:
    def test_writes_bids_columns_in_order(self, tmp_path):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task._events_path = tmp_path / "events.tsv"
        task._events = [
            {
                "onset": 0.0,
                "duration": 2.0,
                "trial_type": "button_press",
                "response": "1",
                "response_time": 0.42,
                "score": 95.0,
                "labels_swapped": 0,
            }
        ]
        task._save_events_tsv()

        with open(task._events_path, newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))

        assert rows[0] == [
            "onset",
            "duration",
            "trial_type",
            "response",
            "response_time",
            "score",
            "labels_swapped",
        ]
        assert rows[1][0] == "0.0"
        assert rows[1][2] == "button_press"

    def test_missing_columns_default_to_na(self, tmp_path):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task._events_path = tmp_path / "events.tsv"
        task._events = [{"onset": 0.0, "duration": 1.0, "trial_type": "rest"}]
        task._save_events_tsv()

        with open(task._events_path, newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        # Missing fields rendered as "n/a"
        assert "n/a" in rows[1]

    def test_no_events_no_file(self, tmp_path):
        from pineuro.tasks.gauge import GaugeTask

        task = GaugeTask.__new__(GaugeTask)
        task._events_path = tmp_path / "events.tsv"
        task._events = []
        task._save_events_tsv()
        assert not task._events_path.exists()
