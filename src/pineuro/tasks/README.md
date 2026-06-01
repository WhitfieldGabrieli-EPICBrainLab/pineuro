# PsychoPy Task Integration

This module provides tools for creating PsychoPy tasks that receive real-time fMRI activation feedback from the pineuro processing pipeline.

## Overview

The task framework handles:
- **Thread-safe communication** between feedback processing and PsychoPy display
- **Continuous refresh** at display rates (60+ Hz) while feedback arrives at TR intervals (~1-2 Hz)
- **Callback notifications** when new volumes are processed
- **Standard lifecycle** for task setup, update, and cleanup

## Running a Task

Feedback tasks are normally run through the GUI or the session-orchestration helpers
(`pineuro.run_single_task` / `run_full_session`), which build the task, wire it to the
real-time coordinator, and manage the run lifecycle for you. The underlying pattern those
helpers follow is:

```python
from pineuro import run_feedback_task
from pineuro.tasks import create_task

# 1. Start the real-time feedback loop in a background thread.
#    Returns {"coordinator": ..., "thread": ...}.
session_info = run_feedback_task(
    mask_paths=["/path/cen_thr.nii.gz", "/path/dmn_thr.nii.gz"],
    subject="sub-01", session="ses-01", task="ball", run=1,
    output_dir="/data/sub-01/ses-01/feedback",
    collection_dir="/scanner/incoming",
    n_baseline=25, n_volumes=150, tr=2.0,
    mask_names=["CEN", "DMN"],
    # simulator_source_dir=..., simulator_speed=10.0,  # to replay DICOMs offline
)
coordinator = session_info["coordinator"]

# 2. Build the PsychoPy task from the registry and wire it to the coordinator.
task = create_task(
    task_id="ball",
    coordinator=coordinator,
    config={"mask_names": ["CEN", "DMN"], "feedback_mode": True},
)
coordinator._activation_callback = task._update_data

# 3. Run the task on the main thread (PsychoPy/OpenGL requirement).
try:
    task.run()
finally:
    coordinator.stop()
    session_info["thread"].join(timeout=5)
```

## Creating Custom Tasks

Tasks are registered through a decorator + metadata system. Define a `TaskMetadata`
describing the configurable parameters, then decorate a `BaseTask` subclass with
`@pineuro_task` (its `task_id` must match the metadata). Implement `setup()` and
`update()`; `cleanup()` is optional.

```python
from psychopy import visual

from pineuro.tasks.base import BaseTask
from pineuro.tasks.task_registry import TaskMetadata, TaskParameter, pineuro_task

MY_TASK_METADATA = TaskMetadata(
    task_id="my_task",
    display_name="My Task",
    description="Brief description of the task",
    parameters=[
        TaskParameter(key="speed", label="Speed", type="float",
                      default=1.0, min_value=0.1, max_value=10.0),
    ],
    n_networks=1,           # number of brain-network masks the task needs
    roles={"feedback"},     # "feedback", "localizer", or both
)

@pineuro_task(MY_TASK_METADATA)
class MyTask(BaseTask):
    task_id = "my_task"

    def setup(self):
        """Create the PsychoPy window and stimuli."""
        self.create_window(monitor_position="right")
        self.stim = visual.Circle(self.win, radius=0.1)

    def update(self, data: dict):
        """Update the display from the latest activation.

        Called every frame (~60 Hz) with:
        - data['activations']: all mask activations, keyed by mask name
        - data['volume']: volume number
        - data['time']: elapsed time
        - data['is_baseline']: baseline status
        """
        self.stim.radius = 0.1 + data["activations"][self.mask_name] * 0.05
        self.stim.draw()
        self.win.flip()

    def cleanup(self):
        """Close the window and release resources (optional)."""
        self.win.close()
```

The `@pineuro_task` decorator registers both the metadata and the class. To make a
**built-in** task discoverable, add its module to the import list in
`_load_builtin_tasks()` in `__init__.py`; **external plugins** are loaded at runtime via
`discover_tasks([Path(...)])`. For **localizer** tasks (stimulus presentation without
real-time feedback), subclass `BaseLocalizerTask` and implement `setup()` +
`run_paradigm()` instead. See the project `CONTRIBUTING.md` ("Adding a task") for the
full checklist.

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
- Adaptive difficulty scaling; scale factors persist via TaskState

### ActivationSmoother

Exponential moving average for reducing noise:

```python
from pineuro.tasks import ActivationSmoother

smoother = ActivationSmoother(alpha=0.3)  # 0=max smoothing, 1=no smoothing
smoothed = smoother.update(new_activation)
```

### ActivationNormalizer

Percentile-based normalization to [0, 1]:

```python
from pineuro.tasks import ActivationNormalizer

normalizer = ActivationNormalizer(window_size=50, lower_percentile=10, upper_percentile=90)
normalized = normalizer.update(raw_activation)  # Returns value in [0, 1]
```

## Tips

- **Escape key**: quit the task from `update()` (standard convention).
- **Smooth animation**: render every frame from the latest activation — new volumes arrive
  only every TR (~1-2 s), so interpolate/smooth between them.
- **Persist across runs**: use `TaskState` for values that should survive between runs
  (e.g. adaptive difficulty, hit counts).

## Requirements

- pineuro (this package)
- psychopy
- numpy
