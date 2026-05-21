"""Fixtures and reporting hooks for the pineuro validation suite.

A session-wide :class:`ResultRecorder` collects every metric a test records via
the ``record`` fixture. At the end of a session that recorded anything, the
``pytest_sessionfinish`` hook writes ``_report/validation_metrics.json`` and
renders the markdown / HTML report.

This module also **pins the numerical environment** (see below). Several
validation metrics are asserted against recorded baselines (``baselines.py``);
those baselines are only reproducible if the iterative optimizers underneath them
run deterministically. Single-threading ITK and BLAS removes the float-addition
reorder that multithreaded reductions would otherwise introduce.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path

# Pin the environment BEFORE anything imports ANTs / nilearn / numpy's BLAS.
# conftest.py is imported by pytest during collection, before any test module
# (and ANTs is lazy-imported inside pineuro at call time), so this is early
# enough for ITK to read the thread count when it builds its threader pool.
for _var in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pytest

from .metrics import ResultRecorder

# Session-wide recorder: populated by the `record` fixture, drained by the
# pytest_sessionfinish hook below.
_RECORDER = ResultRecorder()
_REPORT_DIR = Path(__file__).parent / "_report"


def _capture_doc(request: pytest.FixtureRequest) -> str:
    """First paragraph of the current test's docstring, dedented and one-lined.

    ``request.node.obj`` is the test function; for a parametrized test every
    param case shares one function object, so the docstring is the same across
    cases — which is what the report wants (it describes the test, not the
    case). Missing docstrings yield ``""``.
    """
    fn = getattr(request.node, "obj", None)
    raw = inspect.getdoc(fn) if fn is not None else None  # getdoc dedents + handles None
    if not raw:
        return ""
    first_paragraph = raw.split("\n\n", 1)[0]
    return " ".join(first_paragraph.split())


@pytest.fixture
def record(request: pytest.FixtureRequest) -> Callable[..., float]:
    """Record a metric for the current test, feeding the validation report.

    Usage::

        def test_x(record):
            r = pearson_r(recovered, truth)
            assert record(
                "feedback", "signal_recovery", r, threshold=0.9,
                description="Pearson r of recovered vs. injected ROI signal; "
                            "above 0.9 confirms the signal is tracked.",
            ) > 0.9

    The recorded value is returned, so a metric can be recorded and asserted in
    a single expression. ``op`` is one of ``>``, ``>=``, ``<``, ``<=``, ``==``.
    The current test's docstring is captured automatically as section context;
    ``description`` explains, in plain English, what *this* metric means and why
    its threshold is the right bar — it is what makes the report self-explanatory.
    """
    test_doc = _capture_doc(request)

    def _record(
        stage: str,
        metric: str,
        value: float,
        *,
        threshold: float | None = None,
        op: str = ">",
        units: str = "",
        description: str = "",
    ) -> float:
        _RECORDER.add(
            test=request.node.nodeid,
            stage=stage,
            metric=metric,
            value=value,
            threshold=threshold,
            op=op,
            units=units,
            description=description,
            test_doc=test_doc,
        )
        return float(value)

    return _record


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the metrics JSON and render the report — only if a test recorded."""
    if len(_RECORDER) == 0:
        return
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = _RECORDER.to_list()
    (_REPORT_DIR / "validation_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    # Lazy import keeps matplotlib off the import path of a plain `pytest` run.
    from . import report

    report.generate(records, _REPORT_DIR)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print a one-line validation summary and point at the generated report.

    Silent on a plain ``pytest`` run (validation deselected), so it only speaks
    when the suite actually recorded metrics.
    """
    if len(_RECORDER) == 0:
        return
    records = _RECORDER.to_list()
    total = sum(1 for r in records if r["passed"] is not None)
    passed = sum(1 for r in records if r["passed"])
    n_tests = len({r["test"] for r in records})
    status = "ok" if passed == total else "FAILURES"
    terminalreporter.write_sep("=", "pineuro validation report")
    terminalreporter.write_line(f"  {passed}/{total} threshold checks passed across {n_tests} tests  [{status}]")
    terminalreporter.write_line(f"  report:  {_REPORT_DIR / 'validation_report.md'}")
    terminalreporter.write_line(f"  metrics: {_REPORT_DIR / 'validation_metrics.json'}")
