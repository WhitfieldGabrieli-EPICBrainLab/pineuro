"""Tests for pineuro.gui.screens._feedback_tab — study-wide sham gating.

When sham is disabled study-wide (the Study tab's "Enable Sham/Control Conditions"
is off), the Sham Mask Construction section is hidden and ``collect_config`` must
not persist a per-run ``sham`` block. When sham is enabled — or when no provider is
supplied (back-compat default) — the section is shown and the block is emitted.
"""

from unittest.mock import MagicMock

import flet as ft

from pineuro.gui.screens._feedback_tab import FeedbackTabBuilder


def _feedback_builder(*, sham_enabled=None):
    """Construct a real ball-task FeedbackTabBuilder with a mock page."""
    return FeedbackTabBuilder(
        index=0,
        config={"task": "ball"},
        page=MagicMock(),
        create_browse_button=lambda *a, **k: ft.IconButton(),
        create_file_browse_button=lambda *a, **k: ft.IconButton(),
        sham_enabled=sham_enabled,
    )


class TestShamGatingCollectConfig:
    def test_omits_sham_when_disabled(self):
        out = _feedback_builder(sham_enabled=lambda: False).collect_config()
        assert "sham" not in out

    def test_includes_sham_when_enabled(self):
        out = _feedback_builder(sham_enabled=lambda: True).collect_config()
        assert "sham" in out
        assert out["sham"]["delivery_method"] in ("alternative_masks", "replay")

    def test_includes_sham_by_default(self):
        # No provider => treated as enabled (back-compat).
        out = _feedback_builder(sham_enabled=None).collect_config()
        assert "sham" in out


class TestShamSectionVisibility:
    def test_section_hidden_when_disabled(self):
        builder = _feedback_builder(sham_enabled=lambda: False)
        section = builder._build_sham_section()
        assert section is builder.sham_section
        assert section.visible is False

    def test_section_visible_when_enabled(self):
        builder = _feedback_builder(sham_enabled=lambda: True)
        section = builder._build_sham_section()
        assert section.visible is True
