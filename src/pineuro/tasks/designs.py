"""Task design specifications for GLM modeling.

This module provides abstractions for defining task designs that can be
used with FeedbackEstimator. Designs specify how to compute task regressors
from timing information.

A task's ``TaskMetadata.design_type`` selects which design it gets;
``design_for_metadata`` is the single mapping from that field to a design, used
by both the session runner and the GUI pipeline runner. The two values below are
exhaustive. Event-related paradigms run under ``nuisance_only``: the online GLM
does not model event onsets, since the design matrix is built before the run
starts. Event-related *localizers* are modelled offline via
``mask_extraction_task``, which fits a nilearn first-level model from a BIDS
events file.

Classes
-------
TaskDesign : Abstract base class for task designs
BlockDesign : Block design with baseline and active periods
NuisanceOnlyDesign : No task regressors (resting state)

Functions
---------
design_for_metadata : Map a task's ``design_type`` to its TaskDesign

Examples
--------
>>> from pineuro.tasks.designs import BlockDesign
>>> design = BlockDesign(n_baseline=25)
>>> regressors = design.compute_regressors(n_volumes=150, tr=1.2)
>>> print(regressors.shape)  # (150, 1)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class TaskDesign(ABC):
    """Abstract base class for task designs.

    Subclasses implement specific experimental designs (block, nuisance-only)
    and provide methods to compute task regressors for GLM modeling.

    This abstraction allows FeedbackEstimator to work with different task
    structures without knowing the specific design details.
    """

    @abstractmethod
    def compute_regressors(
        self,
        n_volumes: int,
        tr: float,
    ) -> NDArray[np.float64]:
        """Compute task regressors for the design.

        Parameters
        ----------
        n_volumes : int
            Total number of volumes in the run
        tr : float
            Repetition time in seconds

        Returns
        -------
        NDArray, shape (n_volumes, n_regressors)
            Task regressor matrix. Each column is a separate regressor.
            Returns empty array with shape (n_volumes, 0) if no task regressors.
        """
        pass

    @property
    def n_regressors(self) -> int:
        """Number of task regressors this design produces.

        Returns
        -------
        int
            Number of columns in the regressor matrix
        """
        return 1  # Default for most designs


class BlockDesign(TaskDesign):
    """Block design with baseline and active periods.

    Creates an HRF-convolved boxcar regressor where:
    - Baseline period (volumes 0 to n_baseline-1): task = 0
    - Active period (volumes n_baseline onwards): task = 1 convolved with HRF

    This is the standard design for the ball neurofeedback task.

    Parameters
    ----------
    n_baseline : int
        Number of baseline volumes at the start of the run

    Examples
    --------
    >>> design = BlockDesign(n_baseline=25)
    >>> regressors = design.compute_regressors(n_volumes=150, tr=1.2)
    >>> # Regressor is 0 during baseline, ramps up after
    >>> print(regressors[0, 0])   # ~0.0 (baseline)
    >>> print(regressors[50, 0])  # ~1.0 (active, steady state)
    """

    def __init__(self, n_baseline: int):
        if n_baseline < 0:
            raise ValueError(f"n_baseline must be >= 0, got {n_baseline}")
        self.n_baseline = n_baseline

    def compute_regressors(
        self,
        n_volumes: int,
        tr: float,
    ) -> NDArray[np.float64]:
        """Compute HRF-convolved boxcar regressor.

        Parameters
        ----------
        n_volumes : int
            Total number of volumes
        tr : float
            Repetition time in seconds

        Returns
        -------
        NDArray, shape (n_volumes, 1)
            Single-column task regressor matrix
        """
        from pineuro.utils import compute_boxcar_task_regressor

        if self.n_baseline >= n_volumes:
            raise ValueError(
                f"n_baseline ({self.n_baseline}) must be < n_volumes ({n_volumes}); the task block would never start."
            )

        regressor = compute_boxcar_task_regressor(self.n_baseline, n_volumes, tr)
        return regressor.reshape(-1, 1)

    @property
    def n_regressors(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"BlockDesign(n_baseline={self.n_baseline})"


class NuisanceOnlyDesign(TaskDesign):
    """No task regressors - nuisance-only model for resting state.

    Use this design when there is no task structure and you only want
    to model nuisance regressors (motion, drift, etc.) in the GLM.

    Examples
    --------
    >>> design = NuisanceOnlyDesign()
    >>> regressors = design.compute_regressors(n_volumes=200, tr=1.2)
    >>> print(regressors.shape)  # (200, 0) - empty regressor matrix
    """

    def compute_regressors(
        self,
        n_volumes: int,
        tr: float,
    ) -> NDArray[np.float64]:
        """Return empty regressor array.

        Parameters
        ----------
        n_volumes : int
            Total number of volumes
        tr : float
            Repetition time in seconds (unused)

        Returns
        -------
        NDArray, shape (n_volumes, 0)
            Empty regressor matrix (no task columns)
        """
        return np.zeros((n_volumes, 0), dtype=np.float64)

    @property
    def n_regressors(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "NuisanceOnlyDesign()"


# Exhaustive mapping of TaskMetadata.design_type -> design. Both the session
# runner and the GUI pipeline runner dispatch through here, so a task cannot get
# a different GLM depending on which one launched it.
DESIGN_TYPES = ("block", "nuisance_only")


def design_for_metadata(metadata: Any, n_baseline: int) -> TaskDesign:
    """Map a task's ``design_type`` to the design that builds its regressors.

    Parameters
    ----------
    metadata : TaskMetadata
        Task metadata; only ``design_type`` is read.
    n_baseline : int
        Number of baseline volumes, used by ``block`` designs.

    Returns
    -------
    TaskDesign
        ``BlockDesign`` for ``"block"``, ``NuisanceOnlyDesign`` for
        ``"nuisance_only"``.

    Raises
    ------
    ValueError
        If ``design_type`` is not one of ``DESIGN_TYPES``. ``design_type`` is set
        in task code and never in user config, so an unrecognized value is a
        programming error -- it is raised rather than silently defaulted, which
        would otherwise fit a task with the wrong GLM.
    """
    design_type = metadata.design_type
    if design_type == "nuisance_only":
        return NuisanceOnlyDesign()
    if design_type == "block":
        return BlockDesign(n_baseline)
    raise ValueError(
        f"Unknown design_type {design_type!r} on task "
        f"{getattr(metadata, 'task_id', '<unknown>')!r}; expected one of {DESIGN_TYPES}. "
        "Event-related paradigms run under 'nuisance_only' -- the online GLM cannot "
        "model event onsets, which are not known when the design matrix is built."
    )
