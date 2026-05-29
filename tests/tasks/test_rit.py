"""Tests for pineuro.tasks.rit — RIT localizer task metadata, keyword loading, prompts, duration."""

import json

import pytest

# =============================================================================
# RIT task — metadata and registration
# =============================================================================


class TestRITTaskMetadata:
    def test_registered_in_registry(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "rit" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rit")
        assert meta.task_id == "rit"
        assert "localizer" in meta.roles
        assert meta.n_networks == 0
        assert "rumination_network" in meta.default_contrasts

    def test_has_parameters(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rit")
        param_keys = [p.key for p in meta.parameters]
        assert "keywords_dir" in param_keys
        assert "rumination_duration" in param_keys
        assert "n_trials_per_run" in param_keys


# =============================================================================
# RIT keyword parsing — file resolution and JSON parsing
# =============================================================================


class TestRITKeywordParsing:
    def test_parse_json_keywords(self, tmp_path):
        from pineuro.tasks.rit import RITTask

        keywords = {
            "memories": [
                {"keyword": "loss", "pass": 1, "memory_prompt": "Remember a time you lost something."},
                {"keyword": "shame", "pass": 1, "memory_prompt": "Remember a time you felt ashamed."},
                {"keyword": "anger", "pass": 1, "memory_prompt": "Remember a time you felt angry."},
                {"keyword": "fear", "pass": 1, "memory_prompt": "Remember a time you felt afraid."},
                {
                    "keyword": "lonely",
                    "pass": 2,
                    "memory_prompt": "Remember a time you felt lonely.",
                    "unpleasantness": 7,
                    "trial": 5,
                },
                {"keyword": "rejected", "pass": 2, "memory_prompt": "Remember a time you felt rejected."},
                {"keyword": "hurt", "pass": 2, "memory_prompt": "Remember a time your feelings were hurt."},
                {"keyword": "sad", "pass": 2, "memory_prompt": "Remember a time you felt sad."},
            ]
        }
        kw_file = tmp_path / "sub-01_rit_keywords.json"
        kw_file.write_text(json.dumps(keywords))

        task = RITTask.__new__(RITTask)
        result = task._parse_keywords_file(kw_file)

        assert len(result) == 8
        assert result[0]["keyword"] == "loss"
        assert result[0]["pass"] == 1
        assert result[0]["memory_prompt"] == "Remember a time you lost something."
        # Optional fields preserved as-is
        assert result[4]["unpleasantness"] == 7
        assert result[4]["trial"] == 5

    def test_parse_rejects_missing_memory_prompt(self, tmp_path):
        from pineuro.tasks.rit import RITTask

        keywords = {"memories": [{"keyword": "loss", "pass": 1}]}  # no memory_prompt
        kw_file = tmp_path / "sub-01_rit_keywords.json"
        kw_file.write_text(json.dumps(keywords))

        task = RITTask.__new__(RITTask)
        with pytest.raises(ValueError, match="memory_prompt"):
            task._parse_keywords_file(kw_file)

    def test_resolve_keywords_file_direct(self, tmp_path):
        from pineuro.tasks.rit import RITTask

        kw_file = tmp_path / "keywords.json"
        kw_file.write_text(json.dumps({"memories": [{"keyword": "test", "pass": 1, "memory_prompt": "Test prompt."}]}))

        task = RITTask.__new__(RITTask)
        task.keywords_dir = None
        task._keywords_file_override = kw_file
        task.participant_id = "sub-01"

        result = task._resolve_keywords_file()
        assert result == kw_file

    def test_resolve_keywords_file_missing_raises(self, tmp_path):
        from pineuro.tasks.rit import RITTask

        task = RITTask.__new__(RITTask)
        task.keywords_dir = None
        task._keywords_file_override = tmp_path / "nonexistent.json"
        task.participant_id = "sub-01"

        with pytest.raises(FileNotFoundError):
            task._resolve_keywords_file()

    def test_resolve_keywords_file_by_glob(self, tmp_path):
        from pineuro.tasks.rit import RITTask

        kw_dir = tmp_path / "keywords"
        kw_dir.mkdir()
        kw_file = kw_dir / "sub-01_rit_keywords.json"
        kw_file.write_text(json.dumps({"memories": [{"keyword": "test", "pass": 1, "memory_prompt": "Test prompt."}]}))

        task = RITTask.__new__(RITTask)
        task.keywords_dir = kw_dir
        task._keywords_file_override = None
        task.participant_id = "sub-01"

        result = task._resolve_keywords_file()
        assert result == kw_file

    def test_resolve_keywords_file_by_glob_keywords_first(self, tmp_path):
        """Glob should also find files where 'keywords' appears before the participant ID."""
        from pineuro.tasks.rit import RITTask

        kw_dir = tmp_path / "keywords"
        kw_dir.mkdir()
        kw_file = kw_dir / "RIT_keywords_sub-01.json"
        kw_file.write_text(json.dumps({"memories": [{"keyword": "test", "pass": 1, "memory_prompt": "Test prompt."}]}))

        task = RITTask.__new__(RITTask)
        task.keywords_dir = kw_dir
        task._keywords_file_override = None
        task.participant_id = "sub-01"

        result = task._resolve_keywords_file()
        assert result == kw_file


# =============================================================================
# RIT keyword loading — randomization behavior
# =============================================================================


class TestRITKeywordLoading:
    """Tests for _load_keywords pass-assignment and within-pass shuffles."""

    @staticmethod
    def _write_keywords(tmp_path):
        memories = [
            {"keyword": f"kw{i}_p{p}", "pass": p, "memory_prompt": f"Prompt {i} pass {p}"}
            for p in (1, 2)
            for i in range(1, 5)
        ]
        kw_file = tmp_path / "sub-01_rit_keywords.json"
        kw_file.write_text(json.dumps({"memories": memories}))
        return kw_file

    @staticmethod
    def _make_task(kw_file, participant_id, run):
        from pineuro.tasks.rit import RITTask

        task = RITTask.__new__(RITTask)
        task.keywords_dir = None
        task._keywords_file_override = kw_file
        task.participant_id = participant_id
        task.run_number = run
        task.n_trials_per_run = 4
        task._task_state = None  # mirrors TaskBase.__init__ (bypassed by __new__)
        return task

    def test_load_keywords_deterministic_per_participant(self, tmp_path):
        kw_file = self._write_keywords(tmp_path)
        task1 = self._make_task(kw_file, "sub-001", 1)
        task2 = self._make_task(kw_file, "sub-001", 1)
        task1._load_keywords()
        task2._load_keywords()
        assert [m["keyword"] for m in task1._memories] == [m["keyword"] for m in task2._memories]

    def test_load_keywords_all_memories_same_pass(self, tmp_path):
        kw_file = self._write_keywords(tmp_path)
        task = self._make_task(kw_file, "sub-001", 1)
        task._load_keywords()
        passes = {m["pass"] for m in task._memories}
        assert len(passes) == 1, f"run 1 mixed multiple passes: {passes}"
        assert len(task._memories) == 4

    def test_load_keywords_pass_assignment_varies_across_participants(self, tmp_path):
        """Across many participant IDs, run 1 should sometimes draw from pass 2."""
        kw_file = self._write_keywords(tmp_path)
        passes_for_run1 = set()
        for i in range(64):
            task = self._make_task(kw_file, f"sub-{i:03d}", 1)
            task._load_keywords()
            passes_for_run1.add(task._memories[0]["pass"])
            if len(passes_for_run1) == 2:
                break
        assert passes_for_run1 == {1, 2}, (
            f"Expected both pass values for run 1 across participants; saw {passes_for_run1}"
        )

    def test_load_keywords_within_pass_shuffles(self, tmp_path):
        """Memories within the chosen pass aren't always presented in file order."""
        kw_file = self._write_keywords(tmp_path)
        file_order_p1 = [f"kw{i}_p1" for i in range(1, 5)]
        file_order_p2 = [f"kw{i}_p2" for i in range(1, 5)]
        seen_shuffle = False
        for i in range(64):
            task = self._make_task(kw_file, f"sub-{i:03d}", 1)
            task._load_keywords()
            order = [m["keyword"] for m in task._memories]
            if order != file_order_p1 and order != file_order_p2:
                seen_shuffle = True
                break
        assert seen_shuffle, "Expected to observe at least one shuffled within-pass order across participants"


# =============================================================================
# RIT prompts
# =============================================================================


class TestRITPrompts:
    def test_rumination_prompts_not_empty(self):
        from pineuro.tasks.rit import RUMINATION_PROMPTS

        assert len(RUMINATION_PROMPTS) >= 4

    def test_distraction_prompts_not_empty(self):
        from pineuro.tasks.rit import DISTRACTION_PROMPTS

        assert len(DISTRACTION_PROMPTS) >= 4


# =============================================================================
# RIT volume computation — TaskMetadata.compute_max_run_seconds
# =============================================================================


class TestRITVolumeComputation:
    """Tests for the worst-case run-duration calculator on RIT."""

    def test_compute_max_run_seconds_defaults(self):
        from pineuro.tasks.rit import HEAD_TAIL_BUFFER_S, _rit_max_run_seconds

        # 4 trials * (5 + 25 + 60 + 20 + 15) + 10 buffer = 510s with defaults
        expected = 4 * (5.0 + 25.0 + 2 * 30.0 + 4 * 5.0 + 15.0) + HEAD_TAIL_BUFFER_S
        assert _rit_max_run_seconds({}) == pytest.approx(expected)

    def test_compute_max_run_seconds_scales_with_trials(self):
        from pineuro.tasks.rit import HEAD_TAIL_BUFFER_S, _rit_max_run_seconds

        d4 = _rit_max_run_seconds({"n_trials_per_run": 4})
        d8 = _rit_max_run_seconds({"n_trials_per_run": 8})
        # Doubling trials doubles the per-trial portion (buffer unchanged)
        per_trial_portion_4 = d4 - HEAD_TAIL_BUFFER_S
        per_trial_portion_8 = d8 - HEAD_TAIL_BUFFER_S
        assert per_trial_portion_8 == pytest.approx(2 * per_trial_portion_4)

    def test_compute_max_run_seconds_respects_rating_timeout(self):
        from pineuro.tasks.rit import _rit_max_run_seconds

        base = _rit_max_run_seconds({})  # rating_timeout=5
        bumped = _rit_max_run_seconds({"rating_timeout": 30.0})
        # 4 ratings per trial * 4 trials = 16 rating phases; each grows by 25s
        assert bumped - base == pytest.approx(16 * 25.0)

    def test_rit_metadata_exposes_callable(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rit")
        assert meta.compute_max_run_seconds is not None
        result = meta.compute_max_run_seconds({})
        assert isinstance(result, float)
        assert result > 0


# =============================================================================
# RIT task-state persistence — audit + read-on-retry
# =============================================================================


class TestRITTaskStatePersistence:
    """RIT records its per-run selection and restores it exactly on retry."""

    @staticmethod
    def _write_keywords(tmp_path):
        memories = [
            {"keyword": f"kw{i}_p{p}", "pass": p, "memory_prompt": f"Prompt {i} pass {p}"}
            for p in (1, 2)
            for i in range(1, 5)
        ]
        kw_file = tmp_path / "sub-01_rit_keywords.json"
        kw_file.write_text(json.dumps({"memories": memories}))
        return kw_file

    @staticmethod
    def _make_task(kw_file, participant_id, run, task_state):
        from pineuro.tasks.rit import RITTask

        task = RITTask.__new__(RITTask)
        task.keywords_dir = None
        task._keywords_file_override = kw_file
        task.participant_id = participant_id
        task.run_number = run
        task.n_trials_per_run = 4
        task._task_state = task_state
        return task

    def test_fresh_run_stages_snapshot(self, tmp_path):
        from pineuro.tasks.state import TaskState

        kw_file = self._write_keywords(tmp_path)
        state = TaskState.create_for_task(tmp_path / "ts", "rit")
        task = self._make_task(kw_file, "sub-001", 1, state)

        task._load_keywords()

        snap = task._task_state_snapshot
        assert snap["target_pass"] in (1, 2)
        assert snap["keywords"] == [m["keyword"] for m in task._memories]
        assert len(snap["keywords"]) == 4
        assert len(snap["rumination_prompts"]) == 4
        assert len(snap["distraction_prompts"]) == 4

    def test_restore_on_retry_reuses_stored_selection(self, tmp_path):
        from pineuro.tasks.state import TaskState

        kw_file = self._write_keywords(tmp_path)
        state = TaskState.create_for_task(tmp_path / "ts", "rit")
        # A cross-pass keyword set that a fresh compute can NEVER produce
        # (fresh always draws all 4 from a single pass) -- proves restore reads
        # the recorded selection rather than recomputing.
        stored_keywords = ["kw1_p1", "kw1_p2", "kw2_p1", "kw2_p2"]
        state.record_run(
            run_type="localizer",
            volumes_processed=0,
            attempt=1,
            slot_run_number=1,
            task_state_snapshot={
                "target_pass": 2,
                "keywords": stored_keywords,
                "rumination_prompts": ["r1", "r2", "r3", "r4"],
                "distraction_prompts": ["d1", "d2", "d3", "d4"],
            },
        )

        task = self._make_task(kw_file, "sub-001", 1, state)
        task._load_keywords()

        assert task._target_pass == 2
        assert [m["keyword"] for m in task._memories] == stored_keywords
        assert task._rumination_prompts == ["r1", "r2", "r3", "r4"]
        assert task._distraction_prompts == ["d1", "d2", "d3", "d4"]
        # Re-stages the same selection so the retry records identically.
        assert task._task_state_snapshot["keywords"] == stored_keywords

    def test_missing_keyword_falls_back_to_fresh(self, tmp_path, caplog):
        import logging

        from pineuro.tasks.state import TaskState

        kw_file = self._write_keywords(tmp_path)
        state = TaskState.create_for_task(tmp_path / "ts", "rit")
        state.record_run(
            run_type="localizer",
            volumes_processed=0,
            attempt=1,
            slot_run_number=1,
            task_state_snapshot={
                "target_pass": 1,
                "keywords": ["kw1_p1", "GONE_keyword", "kw2_p1", "kw3_p1"],
            },
        )

        task = self._make_task(kw_file, "sub-001", 1, state)
        with caplog.at_level(logging.WARNING, logger="pineuro.tasks.rit"):
            task._load_keywords()

        # Fell back to a fresh, valid single-pass selection.
        assert len(task._memories) == 4
        assert len({m["pass"] for m in task._memories}) == 1
        assert any("missing" in msg.lower() for msg in caplog.messages)
