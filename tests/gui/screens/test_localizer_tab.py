"""Tests for pineuro.gui.screens._localizer_tab — auto-computed n_volumes display.

Covers the behaviour added by commit d4d964e: when the selected localizer task
exposes ``TaskMetadata.compute_max_run_seconds``, ``_refresh_n_volumes_info``
must update the displayed ceiling from the live TR + task params. Fixed-duration
tasks (e.g. fixation) must not touch the field.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _stub_builder(loc_type, *, tr="1.2", task_params=None, info_visible=True):
    """Construct a LocalizerTabBuilder bypassing __init__; only stubs what _refresh_n_volumes_info needs."""
    from pineuro.gui.screens._localizer_tab import LocalizerTabBuilder

    builder = LocalizerTabBuilder.__new__(LocalizerTabBuilder)
    builder._loc_type = loc_type
    builder.tr_field = SimpleNamespace(value=tr)
    builder.n_volumes_info_text = SimpleNamespace(visible=info_visible, value="")
    builder.config = {"task_params": task_params or {}}
    builder._collect_task_params = MagicMock(return_value=task_params or {})
    builder.page = MagicMock()
    return builder


class TestRefreshNVolumesInfoForVariableDurationTask:
    def test_writes_ceiling_message_for_rnt(self):
        # RNT exposes compute_max_run_seconds (worst-case duration depends on params).
        builder = _stub_builder("rnt", tr="1.5", task_params={"n_trials_per_run": 4})
        builder._refresh_n_volumes_info()
        # Default RNT 4-trial run ~670s @ TR=1.5 → ceil(447) volumes
        assert "Auto-computed:" in builder.n_volumes_info_text.value
        assert "volumes" in builder.n_volumes_info_text.value
        assert "TR=1.5" in builder.n_volumes_info_text.value
        builder.page.update.assert_called()

    def test_scales_inversely_with_tr(self):
        # Halving TR doubles the volume count for the same task duration.
        b1 = _stub_builder("rnt", tr="2.0", task_params={})
        b2 = _stub_builder("rnt", tr="1.0", task_params={})
        b1._refresh_n_volumes_info()
        b2._refresh_n_volumes_info()

        def _extract(msg):
            # Message: "Auto-computed: 335 volumes (worst-case run ..."
            return int(msg.split(":")[1].strip().split(" ")[0])

        v_tr_2 = _extract(b1.n_volumes_info_text.value)
        v_tr_1 = _extract(b2.n_volumes_info_text.value)
        # math.ceil(D/1.0) is roughly 2× math.ceil(D/2.0); allow ±1 for ceil rounding.
        assert v_tr_1 == pytest.approx(2 * v_tr_2, abs=1)

    def test_invalid_tr_shows_helper_message(self):
        builder = _stub_builder("rnt", tr="not-a-number")
        builder._refresh_n_volumes_info()
        assert "set a valid TR" in builder.n_volumes_info_text.value

    def test_zero_tr_shows_helper_message(self):
        builder = _stub_builder("rnt", tr="0")
        builder._refresh_n_volumes_info()
        assert "set a valid TR" in builder.n_volumes_info_text.value

    def test_negative_tr_shows_helper_message(self):
        builder = _stub_builder("rnt", tr="-1.0")
        builder._refresh_n_volumes_info()
        assert "set a valid TR" in builder.n_volumes_info_text.value


class TestRefreshNVolumesInfoForFixedDurationTask:
    def test_fixation_leaves_value_untouched(self):
        # Fixation does not declare compute_max_run_seconds → method should return without writing.
        builder = _stub_builder("fixation", tr="1.5")
        builder.n_volumes_info_text.value = "(unchanged)"
        builder._refresh_n_volumes_info()
        assert builder.n_volumes_info_text.value == "(unchanged)"


class TestRefreshNVolumesInfoEarlyReturns:
    def test_invisible_info_text_skips_recompute(self):
        # When the info text is hidden, the method should short-circuit before computing.
        builder = _stub_builder("rnt", tr="1.5", info_visible=False)
        builder.n_volumes_info_text.value = "(unchanged)"
        builder._refresh_n_volumes_info()
        assert builder.n_volumes_info_text.value == "(unchanged)"
        builder._collect_task_params.assert_not_called()

    def test_unknown_task_silent(self):
        builder = _stub_builder("definitely_not_a_real_task", tr="1.5")
        builder.n_volumes_info_text.value = "(unchanged)"
        builder._refresh_n_volumes_info()
        assert builder.n_volumes_info_text.value == "(unchanged)"
