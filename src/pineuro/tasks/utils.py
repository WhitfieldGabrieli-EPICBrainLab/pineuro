"""
Utility functions for neurofeedback tasks.

Summary
-------
Helpers for smoothing, scaling, and BIDS filename construction in real-time
neurofeedback tasks.

Components
----------
- ActivationSmoother: Exponential moving average for jitter reduction
- ActivationNormalizer: Percentile-based scaling to [0, 1]
- make_bids_run_prefix: Build BIDS-compliant filename prefix
- is_activation_outlier: Detect outlier activations by magnitude

Workflow
--------
- Update smoother each frame to damp spikes
- Feed activations to normalizer to map to [0, 1] for display scaling
- Use make_bids_run_prefix() to build consistent BIDS filenames

Dependencies
------------
numpy, collections (stdlib)

Notes
-----
Smoother trades lag for stability via EMA; normalizer adapts scaling to the
recent distribution using percentiles. Both are lightweight and safe to use
in per-frame update loops.
"""

# Standard library imports
import logging
from collections import deque

# Third-party imports
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)


def make_bids_run_prefix(
    subject: str,
    task: str,
    run: int,
    acq: int | str | None = None,
) -> str:
    """Build a BIDS-compliant filename prefix for task output files.

    Parameters
    ----------
    subject : str
        Subject label (e.g., 'sub-01' or '01')
    task : str
        Task label (e.g., 'ball', 'thermometer')
    run : int
        Run number (formatted as zero-padded 2-digit string)
    acq : int or str or None, optional
        Acquisition label. If int, formatted as zero-padded 2-digit string.
        If None, omitted from prefix.

    Returns
    -------
    str
        Filename prefix like ``{subject}_task-{task}_run-01`` or
        ``{subject}_task-{task}_acq-01_run-01`` when acq is provided.

    Examples
    --------
    >>> make_bids_run_prefix("sub-01", "ball", run=1)
    'sub-01_task-ball_run-01'
    >>> make_bids_run_prefix("sub-01", "ball", run=2, acq=3)
    'sub-01_task-ball_acq-03_run-02'
    """
    if acq is not None:
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        return f"{subject}_task-{task}_acq-{acq_str}_run-{run:02d}"
    return f"{subject}_task-{task}_run-{run:02d}"


def is_activation_outlier(acts: list[float] | float, threshold: float) -> bool:
    """Return True if any activation magnitude exceeds the threshold.

    Parameters
    ----------
    acts : float or list of float
        Activation value(s) to test.
    threshold : float
        Maximum allowed magnitude. Values with abs(act) > threshold are
        considered outliers.

    Returns
    -------
    bool
        True if any value exceeds the threshold.
    """
    if isinstance(acts, (int, float)):
        return abs(acts) > threshold
    return bool(np.max(np.abs(acts)) > threshold)


class ActivationSmoother:
    """Exponential moving average smoother for activation values.

    Useful for reducing noise and jitter in real-time feedback displays.

    Parameters
    ----------
    alpha : float, default=0.3
        Smoothing factor (0=no update/max smoothing, 1=no smoothing/instant update)

    Examples
    --------
    >>> smoother = ActivationSmoother(alpha=0.3)
    >>> smoothed = smoother.update(0.5)
    >>> smoothed = smoother.update(0.7)  # Gradually approaches 0.7
    """

    def __init__(self, alpha: float = 0.3):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.value = 0.0

    def update(self, new_value: float) -> float:
        """Update with new value and return smoothed value.

        Parameters
        ----------
        new_value : float
            New activation value

        Returns
        -------
        float
            Smoothed activation value
        """
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset smoother to initial state."""
        self.value = 0.0


class ActivationNormalizer:
    """Normalize activation to [0, 1] range using percentile-based scaling.

    Tracks recent activation values and normalizes based on percentiles,
    providing robust scaling that adapts to the activation distribution.

    Parameters
    ----------
    window_size : int, default=50
        Number of recent values to use for computing percentiles
    lower_percentile : float, default=10
        Lower bound percentile (maps to 0)
    upper_percentile : float, default=90
        Upper bound percentile (maps to 1)

    Examples
    --------
    >>> normalizer = ActivationNormalizer(window_size=50)
    >>> norm_val = normalizer.update(0.5)  # Returns ~0.5 after warmup
    """

    def __init__(self, window_size: int = 50, lower_percentile: float = 10, upper_percentile: float = 90):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError("percentiles must satisfy: 0 <= lower < upper <= 100")

        self.window: deque[float] = deque(maxlen=window_size)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def update(self, value: float) -> float:
        """Add new value and return normalized value.

        Parameters
        ----------
        value : float
            New activation value

        Returns
        -------
        float
            Normalized value in [0, 1] range
        """
        self.window.append(value)

        # Return neutral value until we have enough data
        if len(self.window) < 3:
            return 0.5

        values = np.array(self.window)
        lower = np.percentile(values, self.lower_percentile)
        upper = np.percentile(values, self.upper_percentile)

        # Handle edge case where all values are identical
        if upper <= lower:
            return 0.5

        # Normalize to [0, 1] and clip
        normalized = (value - lower) / (upper - lower)
        return float(np.clip(normalized, 0.0, 1.0))

    def reset(self):
        """Reset normalizer to initial state."""
        self.window.clear()
