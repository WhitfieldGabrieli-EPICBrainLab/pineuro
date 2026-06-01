"""Regression tests for the GUI lazy-import rule (see .claude/rules/architecture.md).

Importing a GUI screen/component module -- or ``pineuro.tasks.task_registry`` -- must NOT
pull in the heavy pineuro neuroimaging/data stack at import time. Those libraries cost
seconds to load and previously caused multi-second hitches on every screen transition.

Each module is imported in a FRESH interpreter (subprocess) so we measure its true
import-time footprint, uncontaminated by whatever pytest itself has already loaded.

Chart backend exception: ``flet_charts`` (which transitively imports matplotlib, which
imports numpy) is the GUI's real-time chart-rendering backend. It is genuinely needed by
the participant-workflow screen, which displays live activation/QC plots, so numpy +
matplotlib are tolerated there -- but the heavy pineuro modules (feedback, preprocessing,
mask_extraction, ...) and the dialog data libs (pandas, pydicom) must still stay out.
"""

import subprocess
import sys

import pytest

# Heavy libraries that must NEVER load from importing any GUI module. Their presence
# means a heavy pineuro module (feedback/preprocessing/mask_extraction/session/...) or a
# dialog data lib was imported eagerly instead of inside the method that needs it.
_NEVER = ("scipy", "nibabel", "nilearn", "ants", "psychopy", "pandas", "pydicom")

# numpy and matplotlib additionally must not load for modules that do not render charts.
# (On chart-bearing modules they arrive only via flet_charts -- see module docstring.)
_NO_CHART_BACKEND = _NEVER + ("numpy", "matplotlib")

# Modules that must stay completely free of heavy libs: the GUI entry point, the screens
# that don't render charts, the (now lazy) components package, the components that
# previously leaked numpy/pandas/pydicom, and the task registry the screens import from.
_STRICT_MODULES = [
    "pineuro.gui.app",
    "pineuro.gui.screens.study_editor",
    "pineuro.gui.screens._localizer_tab",
    "pineuro.gui.screens._feedback_tab",
    "pineuro.gui.components",
    "pineuro.gui.components.csv_poller",
    "pineuro.gui.components.feedback_summary",
    "pineuro.gui.components.collection_progress",
    "pineuro.tasks.task_registry",
]


def _heavy_libs_loaded(module: str, candidates: tuple[str, ...]) -> set[str]:
    """Import `module` in a fresh interpreter; return which `candidates` libs loaded."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"cand = {tuple(candidates)!r}\n"
        "loaded = [h for h in cand if any(m == h or m.startswith(h + '.') for m in sys.modules)]\n"
        "print('HEAVYLIBS:' + ','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # A cold subprocess importing a GUI module takes a few seconds; 60s is a
        # generous ceiling that still fails fast on a real module-level hang.
        timeout=60,
    )
    assert result.returncode == 0, f"importing {module} failed:\n{result.stdout}\n{result.stderr}"
    for line in result.stdout.splitlines():
        if line.startswith("HEAVYLIBS:"):
            payload = line[len("HEAVYLIBS:") :].strip()
            return set(payload.split(",")) if payload else set()
    raise AssertionError(f"sentinel not found in output for {module}:\n{result.stdout}\n{result.stderr}")


@pytest.mark.slow  # spawns a cold interpreter per module (~seconds each)
@pytest.mark.parametrize("module", _STRICT_MODULES)
def test_strict_modules_import_without_heavy_libs(module):
    leaked = _heavy_libs_loaded(module, _NO_CHART_BACKEND)
    assert not leaked, (
        f"Importing {module!r} eagerly loaded heavy libraries {sorted(leaked)}, "
        "violating the GUI lazy-import rule (.claude/rules/architecture.md). "
        "Move the offending import into the method/function that needs it."
    )


@pytest.mark.slow  # spawns a cold interpreter (~seconds)
def test_participant_workflow_defers_pineuro_heavy_libs():
    """The main workflow screen may load numpy+matplotlib via its chart backend, but must
    never eagerly import the heavy pineuro neuroimaging/data stack."""
    module = "pineuro.gui.screens.participant_workflow"
    leaked = _heavy_libs_loaded(module, _NEVER)
    assert not leaked, (
        f"Importing {module!r} eagerly loaded heavy libraries {sorted(leaked)}, "
        "violating the GUI lazy-import rule (.claude/rules/architecture.md). "
        "Move the offending import into the method/function that needs it."
    )


def test_tasks_lazy_names_resolve():
    """numpy-backed names deferred behind ``tasks.__getattr__`` still import correctly.

    These deliberately use the package-level import (not the submodule path) because that
    is exactly the lazy path under test.
    """
    from pineuro.tasks import (
        ActivationNormalizer,
        ActivationSmoother,
        BlockDesign,
        EventRelatedDesign,
        NuisanceOnlyDesign,
        TaskDesign,
    )

    for obj in (
        TaskDesign,
        BlockDesign,
        NuisanceOnlyDesign,
        EventRelatedDesign,
        ActivationSmoother,
        ActivationNormalizer,
    ):
        assert isinstance(obj, type)


def test_components_lazy_names_resolve():
    """Lazily re-exported component names (PEP 562) still resolve to the real objects."""
    from pineuro.gui.components import (
        ActivationPlot,
        CollectionProgressPanel,
        FeedbackMonitor,
        compute_feedback_summary,
    )

    assert isinstance(ActivationPlot, type)
    assert isinstance(CollectionProgressPanel, type)
    assert isinstance(FeedbackMonitor, type)
    assert callable(compute_feedback_summary)
