"""
Utility functions for neurofeedback tasks.

Provides helper classes for smoothing and normalizing activation values
in real-time neurofeedback displays.

Key Classes
-----------
ActivationSmoother : Exponential moving average smoother
ActivationNormalizer : Percentile-based normalizer to [0, 1] range

Dependencies
------------
- numpy
- collections (standard library)

Notes
-----
These utilities help reduce noise and jitter in visual feedback:

- ActivationSmoother uses exponential moving average (EMA) to smooth
  rapid fluctuations. Lower alpha = more smoothing but more lag.

- ActivationNormalizer tracks recent values and maps to [0, 1] using
  percentiles, providing robust scaling that adapts to the distribution.

"""

# Standard library imports
from collections import deque
import logging

# Third-party imports
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        window_size: int = 50,
        lower_percentile: float = 10,
        upper_percentile: float = 90
    ):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError(
                "percentiles must satisfy: 0 <= lower < upper <= 100"
            )

        self.window = deque(maxlen=window_size)
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
        return np.clip(normalized, 0.0, 1.0)

    def reset(self):
        """Reset normalizer to initial state."""
        self.window.clear()
