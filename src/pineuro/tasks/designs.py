"""Task design specifications for GLM modeling.

This module provides abstractions for defining task designs that can be
used with FeedbackEstimator. Designs specify how to compute task regressors
from timing information.

Classes
-------
TaskDesign : Abstract base class for task designs
BlockDesign : Block design with baseline and active periods
NuisanceOnlyDesign : No task regressors (resting state)

Examples
--------
>>> from pineuro.tasks.designs import BlockDesign
>>> design = BlockDesign(n_baseline=25)
>>> regressors = design.compute_regressors(n_volumes=150, tr=1.2)
>>> print(regressors.shape)  # (150, 1)
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class TaskDesign(ABC):
    """Abstract base class for task designs.

    Subclasses implement specific experimental designs (block, event-related, etc.)
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


class EventRelatedDesign(TaskDesign):
    """Event-related design with discrete event onsets.

    Creates HRF-convolved regressors for events occurring at specific times.
    Supports multiple event types (conditions) with separate regressors.

    Parameters
    ----------
    event_onsets : dict[str, list[float]]
        Dictionary mapping condition names to lists of onset times (in seconds)
    event_durations : dict[str, float], optional
        Dictionary mapping condition names to event durations (in seconds).
        If not provided, events are modeled as impulses (duration=0).

    Examples
    --------
    >>> # Two conditions with events at different times
    >>> design = EventRelatedDesign(
    ...     event_onsets={
    ...         'target': [10.0, 30.0, 50.0],  # Target events at 10s, 30s, 50s
    ...         'distractor': [20.0, 40.0],    # Distractor events
    ...     },
    ...     event_durations={'target': 2.0, 'distractor': 1.0}
    ... )
    >>> regressors = design.compute_regressors(n_volumes=100, tr=1.2)
    >>> print(regressors.shape)  # (100, 2) - one column per condition
    """

    def __init__(
        self,
        event_onsets: dict[str, list[float]],
        event_durations: dict[str, float] | None = None,
    ):
        self.event_onsets = event_onsets
        self.event_durations = event_durations or {}
        self._condition_names = list(event_onsets.keys())

    def compute_regressors(
        self,
        n_volumes: int,
        tr: float,
    ) -> NDArray[np.float64]:
        """Compute HRF-convolved event regressors.

        Parameters
        ----------
        n_volumes : int
            Total number of volumes
        tr : float
            Repetition time in seconds

        Returns
        -------
        NDArray, shape (n_volumes, n_conditions)
            Regressor matrix with one column per condition
        """
        from pineuro.utils import compute_hrf

        n_conditions = len(self._condition_names)
        if n_conditions == 0:
            return np.zeros((n_volumes, 0), dtype=np.float64)

        # Compute HRF kernel
        hrf = compute_hrf(tr)

        # Total time in seconds
        total_time = n_volumes * tr

        # Build regressor for each condition
        regressors = np.zeros((n_volumes, n_conditions), dtype=np.float64)

        for i, condition in enumerate(self._condition_names):
            onsets = self.event_onsets[condition]
            duration = self.event_durations.get(condition, 0.0)

            # Create stimulus function at TR resolution
            stim = np.zeros(n_volumes, dtype=np.float64)

            for onset in onsets:
                if onset >= total_time:
                    continue

                # Convert onset to volume index
                start_vol = int(onset / tr)
                if duration > 0:
                    end_vol = min(int((onset + duration) / tr) + 1, n_volumes)
                    stim[start_vol:end_vol] = 1.0
                else:
                    # Impulse event
                    if start_vol < n_volumes:
                        stim[start_vol] = 1.0

            # Convolve with HRF
            convolved = np.convolve(stim, hrf, mode="full")[:n_volumes]

            # Normalize
            if np.max(np.abs(convolved)) > 0:
                convolved = convolved / np.max(np.abs(convolved))

            regressors[:, i] = convolved

        return regressors

    @property
    def n_regressors(self) -> int:
        return len(self._condition_names)

    @property
    def condition_names(self) -> list[str]:
        """Names of conditions in order."""
        return self._condition_names.copy()

    def __repr__(self) -> str:
        conditions = ", ".join(self._condition_names)
        return f"EventRelatedDesign(conditions=[{conditions}])"
