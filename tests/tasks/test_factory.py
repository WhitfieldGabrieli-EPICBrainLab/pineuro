"""Tests for pineuro.tasks.factory — create_task, registered IDs, config handling."""

import pytest


class TestTaskFactory:
    def test_get_registered_task_ids(self):
        from pineuro.tasks.factory import get_registered_task_ids

        ids = get_registered_task_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_unknown_config_keys_logged(self, caplog):
        import logging
        from unittest.mock import MagicMock

        from pineuro.tasks import factory as factory_mod
        from pineuro.tasks.factory import get_registered_task_ids

        ids = get_registered_task_ids()
        if not ids:
            pytest.skip("No tasks registered")

        task_id = ids[0]

        # Substitute a mock task class so we don't need PsychoPy running
        original_classes = dict(factory_mod._TASK_CLASSES)
        factory_mod._TASK_CLASSES[task_id] = MagicMock(return_value=MagicMock())

        mock_coordinator = MagicMock()
        mock_coordinator.mask_names = ["ContA", "DefaultA"]

        try:
            with caplog.at_level(logging.WARNING, logger="pineuro.tasks.factory"):
                factory_mod.create_task(
                    task_id=task_id,
                    coordinator=mock_coordinator,
                    config={"mask_names": ["ContA", "DefaultA"], "typo_key_xyz": 999},
                )
            assert any("typo_key_xyz" in msg for msg in caplog.messages)
        finally:
            factory_mod._TASK_CLASSES.clear()
            factory_mod._TASK_CLASSES.update(original_classes)
