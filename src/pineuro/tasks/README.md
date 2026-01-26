# PsychoPy Task Integration

This module provides tools for creating PsychoPy tasks that receive real-time fMRI activation feedback from the pyrtnf processing pipeline.

## Overview

The task framework handles:
- **Thread-safe communication** between feedback processing and PsychoPy display
- **Continuous refresh** at display rates (60+ Hz) while feedback arrives at TR intervals (~1-2 Hz)
- **Callback notifications** when new volumes are processed
- **Standard lifecycle** for task setup, update, and cleanup

## Quick Start

```python
from pyrtnf import run_feedback_session, BallTask
# Start feedback in background thread
session_info = run_feedback_session(
    mask_paths=['/path/to/cen_mask.nii.gz', '/path/to/dmn_mask.nii.gz'],
    subject='sub-01',
    task='rest',
    run=1,
    output_dir='/path/to/output',
    mask_names=['CEN', 'DMN'],
    n_baseline=25,
    n_volumes=150,
    run_in_background=True,  # Required for PsychoPy
)

coordinator = session_info['coordinator']
coordinator.wait_until_ready(timeout=5)
coordinator.send_trigger()  # In simulation mode this can be automatic

# Run task in main thread
task = BallTask(coordinator, mask_names=['CEN', 'DMN'])
coordinator._activation_callback = task._update_data

try:
    task.run()
finally:
    coordinator.stop()
    session_info['thread'].join(timeout=5)
```

## Creating Custom Tasks

Subclass `BaseTask` and implement three methods:

```python
from pyrtnf.tasks import BaseTask
from psychopy import visual

class MyTask(BaseTask):
    def setup(self):
        """Initialize PsychoPy window and stimuli."""
        self.win = visual.Window([800, 600])
        self.stim = visual.Circle(self.win, radius=0.1)

    def update(self, data: dict):
        """Update display based on latest activation.

        Called every frame (~60 Hz) with:
        - data['activation']: current activation value
        - data['volume']: volume number
        - data['time']: elapsed time
        - data['is_baseline']: baseline status
        """
        # Scale stimulus based on activation
        self.stim.radius = 0.1 + data['activation'] * 0.05
        self.stim.draw()
        self.win.flip()

    def cleanup(self):
        """Close window and release resources."""
        self.win.close()
```

## Architecture

### Threading Model

- **Coordinator thread** (background): Watches for DICOMs, processes volumes, updates shared state
- **Task thread** (main): Runs PsychoPy at display refresh rate, polls latest activation

PsychoPy **must** run in the main thread (OpenGL requirement), so the coordinator runs in background.

### Communication Pattern

Two complementary mechanisms:

1. **Callback**: Coordinator calls `task._update_data()` immediately after each volume
2. **Polling**: Task calls `task.get_latest_data()` continuously at display rate

This enables:
- Immediate response to new volumes (callback)
- Smooth animations between volumes (polling)

### Thread Safety

All shared state access protected by `threading.Lock` to prevent race conditions.

## Available Components

### BaseTask

Abstract base class providing:
- Thread-safe data access
- Standard lifecycle (setup, update, cleanup)
- Polling infrastructure
- Frame timing

### BallTask

Example implementation with:
- Dual-network feedback (e.g., CEN/DMN)
- Ball movement based on activation
- Hit detection with circle shrink
- Adaptive difficulty scaling; scale factors persist via SessionState

### ActivationSmoother

Exponential moving average for reducing noise:

```python
from pyrtnf.tasks import ActivationSmoother

smoother = ActivationSmoother(alpha=0.3)  # 0=max smoothing, 1=no smoothing
smoothed = smoother.update(new_activation)
```

### ActivationNormalizer

Percentile-based normalization to [0, 1]:

```python
from pyrtnf.tasks import ActivationNormalizer

normalizer = ActivationNormalizer(window_size=50, lower_percentile=10, upper_percentile=90)
normalized = normalizer.update(raw_activation)  # Returns value in [0, 1]
```

## Examples

See [examples/run_feedback_with_task.py](../../../examples/run_feedback_with_task.py) for complete working example with command-line interface.

## Tips

- **Always run coordinator in background**: Set `run_in_background=True`
- **Register callback**: `coordinator._activation_callback = task._update_data`
- **Wait for initialization**: Brief `time.sleep(2)` after starting coordinator
- **Graceful shutdown**: Always call `coordinator.stop()` and join thread
- **Escape key**: Standard convention to quit task (check in `update()`)

## Requirements

- pyrtnf (this package)
- psychopy
- numpy
- See `docs/references.md` for underlying method references.
