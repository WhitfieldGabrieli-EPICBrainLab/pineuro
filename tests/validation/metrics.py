"""Recovery metrics and the result recorder for the validation suite.

The metric helpers quantify how well pineuro recovered a phantom's ground
truth; :class:`ResultRecorder` collects every recorded number so the suite can
emit a human-readable report alongside the pass/fail assertions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Recovery metrics
# =============================================================================


def _flat(x: object) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).ravel()


def _zscore(x: object) -> NDArray[np.float64]:
    arr = _flat(x)
    sd = float(arr.std())
    centered = arr - arr.mean()
    result = centered / sd if sd > 0 else centered
    return np.asarray(result, dtype=np.float64)


def pearson_r(a: object, b: object) -> float:
    """Pearson correlation coefficient; ``nan`` if either input is constant."""
    av, bv = _flat(a), _flat(b)
    if av.std() == 0 or bv.std() == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def rmse_z(a: object, b: object) -> float:
    """Root-mean-square error between the z-scored inputs (scale-free)."""
    za, zb = _zscore(a), _zscore(b)
    return float(np.sqrt(np.mean((za - zb) ** 2)))


def dice(a: object, b: object) -> float:
    """Dice overlap of two binary masks; 1.0 if both are empty, 0.0 if one is."""
    am = np.asarray(a) > 0
    bm = np.asarray(b) > 0
    total = int(am.sum()) + int(bm.sum())
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(am, bm).sum() / total)


def relative_error(estimated: float, true: float) -> float:
    """``|estimated - true| / |true|``; 0.0 if both are zero, ``nan`` if only true is."""
    estimated, true = float(estimated), float(true)
    if true == 0.0:
        return 0.0 if estimated == 0.0 else float("nan")
    return abs(estimated - true) / abs(true)


def euclidean_offset(a: object, b: object) -> float:
    """Euclidean distance between two coordinate vectors."""
    return float(np.linalg.norm(_flat(a) - _flat(b)))


def best_match_correlation(estimated_maps: object, true_maps: object) -> NDArray[np.float64]:
    """For each true spatial map, the max ``|Pearson r|`` over all estimated maps.

    ``estimated_maps`` is ``(n_estimated, n_voxels)`` and ``true_maps`` is
    ``(n_true, n_voxels)``. ICA component order and sign are arbitrary, hence
    the best-match search and the absolute value.
    """
    estimated = np.asarray(estimated_maps, dtype=np.float64)
    true = np.asarray(true_maps, dtype=np.float64)
    out = np.zeros(true.shape[0], dtype=np.float64)
    for i, true_map in enumerate(true):
        out[i] = max(abs(pearson_r(true_map, est)) for est in estimated)
    return out


# =============================================================================
# Result recorder
# =============================================================================

_OPS = {
    ">": lambda v, t: v > t,
    ">=": lambda v, t: v >= t,
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
    "==": lambda v, t: v == t,  # exact equality — only for integer / exactly-produced values
}


@dataclass
class MetricRecord:
    """One recorded metric: a measured value plus its pass/fail verdict.

    ``description`` is a plain-English explanation of what *this metric* means
    and why its threshold is the right bar; ``test_doc`` is the first paragraph
    of the recording test's docstring, captured automatically. Both feed the
    human-readable report (see :mod:`report`).
    """

    test: str
    stage: str
    metric: str
    value: float
    op: str
    threshold: float | None
    passed: bool | None
    units: str
    description: str = ""
    test_doc: str = ""

    def to_dict(self) -> dict:
        return {
            "test": self.test,
            "stage": self.stage,
            "metric": self.metric,
            "value": self.value,
            "op": self.op,
            "threshold": self.threshold,
            "passed": self.passed,
            "units": self.units,
            "description": self.description,
            "test_doc": self.test_doc,
        }


class ResultRecorder:
    """Collects :class:`MetricRecord` entries across a validation session."""

    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def add(
        self,
        *,
        test: str,
        stage: str,
        metric: str,
        value: float,
        threshold: float | None = None,
        op: str = ">",
        units: str = "",
        description: str = "",
        test_doc: str = "",
    ) -> MetricRecord:
        """Record a metric. ``passed`` is computed when a ``threshold`` is given."""
        if op not in _OPS:
            raise ValueError(f"unknown comparison op {op!r}; expected one of {sorted(_OPS)}")
        numeric = float(value)
        passed: bool | None = None
        if threshold is not None and not np.isnan(numeric):
            passed = bool(_OPS[op](numeric, threshold))
        elif threshold is not None:
            passed = False  # a NaN metric never passes
        record = MetricRecord(test, stage, metric, numeric, op, threshold, passed, units, description, test_doc)
        self.records.append(record)
        return record

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

    def clear(self) -> None:
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)
