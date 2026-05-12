"""Tests for pineuro.tasks.ui_helpers — pure logic only.

Most helpers in this module are thin PsychoPy wrappers (visual.Rect, visual.Circle,
event.getKeys, etc.) whose unit-test value is low. We cover the one helper with
non-trivial pure logic: update_progress_bar (clamp + position math).
"""

from types import SimpleNamespace

from pineuro.tasks.ui_helpers import update_progress_bar


class TestUpdateProgressBar:
    @staticmethod
    def _make_fill(initial_y=-0.4):
        return SimpleNamespace(width=0.0, pos=(0.0, initial_y))

    def test_zero_progress_zero_width(self):
        fill = self._make_fill()
        update_progress_bar(fill, progress=0.0, total_width=0.8, left_x=-0.4)
        assert fill.width == 0.0
        # Left-aligned: center x = left_x + width/2 = -0.4 + 0
        assert fill.pos == (-0.4, -0.4)

    def test_full_progress_full_width(self):
        fill = self._make_fill()
        update_progress_bar(fill, progress=1.0, total_width=0.8, left_x=-0.4)
        assert fill.width == 0.8
        # Center x = -0.4 + 0.8/2 = 0.0
        assert fill.pos == (0.0, -0.4)

    def test_half_progress(self):
        fill = self._make_fill()
        update_progress_bar(fill, progress=0.5, total_width=0.8, left_x=-0.4)
        assert fill.width == 0.4
        # Center x = -0.4 + 0.4/2 = -0.2
        assert fill.pos[0] == -0.2

    def test_overflow_clamped_to_one(self):
        fill = self._make_fill()
        update_progress_bar(fill, progress=2.5, total_width=0.8, left_x=-0.4)
        assert fill.width == 0.8  # clamped, not 2.0

    def test_negative_clamped_to_zero(self):
        fill = self._make_fill()
        update_progress_bar(fill, progress=-0.3, total_width=0.8, left_x=-0.4)
        assert fill.width == 0.0

    def test_y_position_preserved(self):
        fill = self._make_fill(initial_y=0.25)
        update_progress_bar(fill, progress=0.5, total_width=0.8, left_x=-0.4)
        assert fill.pos[1] == 0.25
