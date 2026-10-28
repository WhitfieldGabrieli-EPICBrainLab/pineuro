"""Tests for pineuro.gui.components.brain_mask_dialog."""

from unittest.mock import MagicMock

import pytest

from pineuro.gui.components.brain_mask_dialog import BrainMaskReviewDialog, parse_brain_mask_values


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "review.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def _dialog(png, **kw):
    calls: dict = {"preview": [], "accept": []}
    dialog = BrainMaskReviewDialog(
        MagicMock(),
        title="Review Brain Mask",
        image_path=png,
        voxel_text="Brain mask: 30000 voxels",
        current=(0.5, 2),
        study_defaults=(0.5, 2),
        adjusted=False,
        on_preview=lambda f, c: calls["preview"].append((f, c)),
        on_accept=lambda f, c: calls["accept"].append((f, c)),
        **kw,
    )
    dialog.show()
    return dialog, calls


class TestParseBrainMaskValues:
    def test_valid(self):
        assert parse_brain_mask_values("0.7", "3") == (0.7, 3)

    @pytest.mark.parametrize(("fraction", "cleanup"), [("", "2"), ("abc", "2"), ("0", "2"), ("0.5", None)])
    def test_invalid(self, fraction, cleanup):
        with pytest.raises(ValueError):
            parse_brain_mask_values(fraction, cleanup)


class TestBrainMaskReviewDialog:
    def test_preview_and_accept_pass_entered_values(self, png):
        dialog, calls = _dialog(png)
        dialog.fraction_field.value = "0.8"
        dialog.cleanup_dropdown.value = "4"
        dialog._preview(None)
        dialog._accept(None)
        assert calls == {"preview": [(0.8, 4)], "accept": [(0.8, 4)]}

    def test_invalid_value_shows_error(self, png):
        dialog, calls = _dialog(png)
        dialog.fraction_field.value = "-1"
        dialog._preview(None)
        assert calls["preview"] == []
        assert dialog.error_text.visible

    def test_reset_restores_study_defaults(self, png):
        dialog, _ = _dialog(png)
        dialog.fraction_field.value = "0.9"
        dialog.cleanup_dropdown.value = "0"
        dialog._reset(None)
        assert (dialog.fraction_field.value, dialog.cleanup_dropdown.value) == ("0.5", "2")

    def test_locked_masks_block_a_changed_mask_but_not_acceptance_as_is(self, png):
        dialog, calls = _dialog(png, locked_reason="Masks are locked")
        dialog.fraction_field.value = "0.8"
        dialog._accept(None)
        assert calls["accept"] == []
        assert dialog.error_text.visible
        dialog.fraction_field.value = "0.5"
        dialog._accept(None)
        assert calls["accept"] == [(0.5, 2)]

    def test_set_preview_updates_image_and_text(self, png):
        dialog, _ = _dialog(png)
        dialog.set_preview(png, "Preview: 25000 voxels")
        assert dialog.voxel_text.value == "Preview: 25000 voxels"
        assert dialog.status_text.value.startswith("Preview")
