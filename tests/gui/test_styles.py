"""Tests for the shared GUI style tokens (pineuro.gui.styles).

These guard the token vocabulary the rest of the GUI now depends on:

1. ``STEP_STATUS_COLORS`` must cover every ``StepStatus`` value. The map is keyed
   by the enum's *value strings* (not the enum) so ``styles`` stays flet-only and
   free of a ``screens -> styles`` import cycle; the trade-off is that a new enum
   member would otherwise silently ``KeyError`` at render time. This test makes
   that loud instead.
2. ``styles`` must stay flet-only — importing it must not drag in numpy / nibabel /
   the heavy ``pineuro`` pipeline modules (the GUI lazy-import rule).
"""

import subprocess
import sys

from pineuro.gui import styles as st


class TestStepStatusColors:
    def test_covers_every_step_status_value(self):
        from pineuro.gui.screens.participant_workflow import StepStatus

        for status in StepStatus:
            assert status.value in st.STEP_STATUS_COLORS, (
                f"StepStatus.{status.name} ({status.value!r}) is not in STEP_STATUS_COLORS"
            )

    def test_no_extra_keys(self):
        from pineuro.gui.screens.participant_workflow import StepStatus

        valid = {s.value for s in StepStatus}
        assert set(st.STEP_STATUS_COLORS) == valid

    def test_values_are_color_strings(self):
        # Flet color tokens are strings; a None or stray enum would break rendering.
        assert st.STEP_STATUS_COLORS
        assert all(isinstance(v, str) and v for v in st.STEP_STATUS_COLORS.values())


class TestStylesIsFletOnly:
    def test_import_pulls_no_heavy_modules(self):
        # Importing styles in a fresh interpreter must not load numpy/nibabel/scipy
        # or the heavy pineuro pipeline modules (GUI lazy-import rule).
        heavy = (
            "numpy",
            "nibabel",
            "scipy",
            "psychopy",
            "pineuro.feedback",
            "pineuro.preprocessing",
            "pineuro.mask_extraction",
        )
        code = (
            "import sys, pineuro.gui.styles\n"
            f"bad = [m for m in {heavy!r} if m in sys.modules]\n"
            "raise SystemExit('LOADED:' + ','.join(bad) if bad else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, f"styles pulled heavy modules: {result.stderr or result.stdout}"


class TestTokenVocabulary:
    def test_status_tokens_present(self):
        for name in ("SUCCESS", "WARNING", "ERROR", "INFO", "NEUTRAL"):
            assert isinstance(getattr(st, name), str)
            assert isinstance(getattr(st, f"{name}_BG"), str)

    def test_provenance_tokens_present(self):
        for name in ("MASK_TISSUE", "MASK_MANUAL"):
            assert isinstance(getattr(st, name), str)

    def test_icon_scale_is_ascending(self):
        assert st.ICON_SM < st.ICON_MD < st.ICON_LG < st.ICON_XL < st.ICON_XXL

    def test_spacing_scale_is_ascending(self):
        assert st.GAP_XS < st.GAP_SM < st.GAP_MD < st.GAP < st.GAP_LG
