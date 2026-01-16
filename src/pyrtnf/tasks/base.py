"""
Base class for PsychoPy neurofeedback tasks.

Summary
-------
Thread-safe task scaffold that polls the coordinator for activations and drives
PsychoPy displays at a target refresh rate.

Components
----------
- BaseTask: Abstract base with setup/update/cleanup hooks

Workflow
--------
- setup(): create window/stimuli
- run(): poll coordinator at refresh_rate, call update() each frame
- cleanup(): release resources when coordinator signals completion

Dependencies
------------
threading (stdlib), PsychoPy (for concrete tasks)

Notes
-----
The run() loop calls setup(), then updates at refresh_rate Hz until the
coordinator reports completion, then calls cleanup(). Use this when building
PsychoPy tasks that need thread-safe access to live activations.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Optional
import logging
import threading
import time

# Configure module-level logger
logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Abstract base class for neurofeedback tasks.

    Subclasses must implement setup(), update(), and cleanup() methods.
    The task automatically handles thread-safe access to activation data
    from the NeurofeedbackCoordinator running in a background thread.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator object providing activation data
    mask_name : str, optional
        Name of mask to display. If None, uses first mask.
    refresh_rate : float, optional
        Display refresh rate in Hz (default: 60.0)

    Attributes
    ----------
    coordinator : NeurofeedbackCoordinator
        Reference to coordinator for polling latest activations
    mask_name : str
        Name of the mask being displayed
    refresh_rate : float
        Target refresh rate in Hz
    frame_duration : float
        Target duration per frame in seconds

    Examples
    --------
    >>> class MyTask(BaseTask):
    ...     def setup(self):
    ...         self.win = visual.Window([800, 600])
    ...
    ...     def update(self, data: dict):
    ...         # Draw based on data['activation']
    ...         pass
    ...
    ...     def cleanup(self):
    ...         self.win.close()
    """

    def __init__(
        self,
        coordinator,
        mask_name: Optional[str] = None,
        refresh_rate: float = 60.0
    ):
        self.coordinator = coordinator
        self.mask_name = mask_name or coordinator.mask_names[0]
        self.refresh_rate = refresh_rate
        self.frame_duration = 1.0 / refresh_rate

        # Thread-safe access to latest data
        self._lock = threading.Lock()
        self._latest_data = {
            'activation': 0.0,
            'volume': 0,
            'time': 0.0,
            'is_baseline': True,
        }

        self._running = False

    def _update_data(
        self,
        activations: dict,
        volume: int,
        elapsed: float,
        is_baseline: bool
    ):
        """Internal callback to update latest data (thread-safe).

        Called by coordinator after each volume is processed.
        """
        with self._lock:
            self._latest_data = {
                'activation': activations.get(self.mask_name, 0.0),
                'volume': volume,
                'time': elapsed,
                'is_baseline': is_baseline,
            }

    def get_latest_data(self) -> dict:
        """Get latest activation data (thread-safe).

        Returns
        -------
        dict
            Dictionary containing:
            - 'activation': float - current activation value
            - 'volume': int - current volume number
            - 'time': float - elapsed time in seconds
            - 'is_baseline': bool - whether in baseline period
        """
        with self._lock:
            return self._latest_data.copy()

    @abstractmethod
    def setup(self):
        """Initialize PsychoPy window and stimuli.

        Called once before run loop starts. Create window, visual stimuli,
        and any other resources needed for the task.
        """
        pass

    @abstractmethod
    def update(self, data: dict):
        """Update display based on latest activation data.

        Called every display frame with latest data. Update stimulus
        properties (position, color, size, etc.) based on activation.

        Parameters
        ----------
        data : dict
            Dictionary containing:
            - 'activation': float - current activation value
            - 'volume': int - current volume number
            - 'time': float - elapsed time in seconds
            - 'is_baseline': bool - whether in baseline period
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources.

        Called once after run loop ends. Close window and release resources.
        """
        pass

    def run(self):
        """Main task loop - runs in main thread for PsychoPy.

        Calls setup(), then loops calling update() at the target refresh rate
        until the session completes, then calls cleanup().
        """
        self._running = True
        self.setup()

        try:
            while self._running and not self.coordinator.is_complete():
                frame_start = time.time()

                # Check for errors from coordinator/simulator
                self.coordinator.check_error()

                # Get latest data and update display
                data = self.get_latest_data()
                self.update(data)

                # Frame timing
                elapsed = time.time() - frame_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()

    def stop(self):
        """Signal task to stop."""
        self._running = False
