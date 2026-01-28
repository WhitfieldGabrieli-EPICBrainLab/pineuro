# Creating Custom Neurofeedback Tasks

This guide explains how to create custom neurofeedback tasks that integrate with pineuro's framework. Custom tasks appear in the GUI dropdown and can be configured through the study editor.

## Quick Start

Here's a minimal task template:

```python
from pineuro.tasks import (
    BaseTask,
    TaskMetadata,
    TaskParameter,
    SessionStateField,
    PostRunQuestion,
    register_task,
    register_task_class,
)

class MyTask(BaseTask):
    """My custom neurofeedback task."""

    task_id = "my_task"  # Must match metadata task_id

    def __init__(self, coordinator, my_param: float = 1.0, **kwargs):
        self.coordinator = coordinator
        self.my_param = my_param
        self._running = False
        # ... initialize other attributes

    def setup(self):
        """Create PsychoPy window and stimuli."""
        from psychopy import visual
        self.win = visual.Window([800, 600])
        # ... create stimuli

    def update(self, data: dict):
        """Update display based on activation."""
        activation = data.get('activation', 0.0)
        # ... update display based on activation
        self.win.flip()

    def cleanup(self):
        """Clean up resources."""
        self.win.close()

    def run(self):
        """Main loop (can use default BaseTask.run())."""
        self._running = True
        self.setup()
        try:
            while self._running and not self.coordinator.is_complete():
                data = self.coordinator.get_latest_activations() or {}
                self.update(data)
        finally:
            self.cleanup()

# Define metadata
MY_TASK_METADATA = TaskMetadata(
    task_id="my_task",
    display_name="My Custom Task",
    description="A custom neurofeedback task",
    n_networks=1,  # How many masks this task requires
    parameters=[
        TaskParameter(
            key="my_param",
            label="My Parameter",
            type="float",
            default=1.0,
        ),
    ],
)

# Register (happens automatically when module is imported)
register_task(MY_TASK_METADATA)
register_task_class("my_task", MyTask)
```

## Step-by-Step Guide

### 1. Create Your Task Class

Your task class should extend `BaseTask` and implement these methods:

| Method | Required | Description |
|--------|----------|-------------|
| `__init__` | Yes | Initialize task with coordinator and parameters |
| `setup()` | Yes | Create PsychoPy window and stimuli |
| `update(data)` | Yes | Update display based on activation data |
| `cleanup()` | Yes | Close window and save state |
| `run()` | Optional | Main loop (default in BaseTask works for most cases) |

#### The `__init__` Method

```python
def __init__(
    self,
    coordinator,                        # Required: provides activation data
    mask_names: list[str] = None,       # Which masks to use
    my_param: float = 1.0,              # Your custom parameter
    feedback_mode: bool = True,         # Whether feedback is active
    session_state: "SessionState" = None,  # For state persistence
    **kwargs,                           # Accept extra kwargs for compatibility
):
    self.coordinator = coordinator
    self.mask_name = (mask_names or coordinator.mask_names)[0]
    self.my_param = my_param
    self.feedback_mode = feedback_mode
    self._session_state = session_state
    self._running = False
```

#### Getting Activation Data

In your `update()` method, get activation data from the coordinator:

```python
def update(self, data: dict):
    # Get latest activations
    activations = self.coordinator.get_latest_activations()

    if activations:
        # activations is a dict with:
        # - 'activations': dict of mask_name -> activation_value
        # - 'volume': current volume number
        # - 'time': elapsed time
        # - 'is_baseline': whether still in baseline period

        activation = activations['activations'].get(self.mask_name, 0.0)
        is_baseline = activations.get('is_baseline', True)

        if not is_baseline and self.feedback_mode:
            # Update display based on activation
            self.update_display(activation)
```

### 2. Define Task Metadata

The `TaskMetadata` class tells the GUI how to configure your task:

```python
MY_TASK_METADATA = TaskMetadata(
    task_id="my_task",           # Unique identifier (lowercase, underscores)
    display_name="My Task",      # Shown in GUI dropdown
    description="Description",   # Help text
    n_networks=1,                # Number of masks required
    parameters=[...],            # Configurable parameters
    session_state_fields=[...],  # Persistent state
    post_run_questions=[...],    # Questions after each run
    cross_parameter_validator=None,  # Optional validation function
)
```

#### Parameters (GUI Configuration)

Parameters appear in the study editor and are passed to your task's `__init__`:

```python
parameters=[
    TaskParameter(
        key="target_level",      # Parameter name (passed to __init__)
        label="Target Level",    # GUI label
        type="float",            # "float", "int", "bool", "str", "list[str]"
        default=0.5,             # Default value
        min_value=0.0,           # Optional: minimum (for numeric)
        max_value=1.0,           # Optional: maximum (for numeric)
        hint="Help text",        # Shown below field
    ),
]
```

#### Session State Fields (Persistence)

Session state persists across runs within a session:

```python
session_state_fields=[
    SessionStateField(
        key="total_score",       # Field name
        type="float",            # "float", "int", "bool", "list[float]", "list[int]"
        default=0.0,             # Initial value
        description="Cumulative score across runs",
    ),
]
```

Access in your task:

```python
# Read state
current_score = self._session_state.get("total_score", 0.0)

# Update state
self._session_state.set("total_score", current_score + run_score)
```

#### Post-Run Questions

Questions displayed after each run:

```python
post_run_questions=[
    PostRunQuestion(
        text="How focused did you feel?",
        left_label="Not at all",
        right_label="Extremely",
        condition="always",  # "always", "feedback_only", "last_feedback_only"
    ),
]
```

### 3. Register Your Task

Registration happens automatically when your module is imported:

```python
# At the end of your module
register_task(MY_TASK_METADATA)
register_task_class("my_task", MyTask)
```

To enable auto-registration, add your import to `pineuro/tasks/__init__.py`:

```python
from .my_task import MyTask  # Registers "my_task" task
```

### 4. Network Requirements

Set `n_networks` in your metadata to specify how many masks your task needs:

- `n_networks=1`: Single-network feedback (e.g., ThermometerTask)
- `n_networks=2`: Dual-network feedback (e.g., BallTask)
- `n_networks=N`: Multi-network feedback

The factory validates that enough masks are provided before creating your task.

## Example Tasks

### ThermometerTask (Single Network)

See `pineuro/tasks/thermometer.py` for a complete single-network example:
- Simple vertical bar that fills based on activation
- Target zone for participants to hit
- Tracks time spent in target zone

### BallTask (Dual Network)

See `pineuro/tasks/ball.py` for a complex dual-network example:
- Ball moves based on competition between two networks
- Adaptive difficulty (scale factor adjusts based on performance)
- Hit detection with shrinking targets
- Post-run questions

## Runtime Parameters

These parameters are provided by the framework, not defined in metadata:

| Parameter | Type | Description |
|-----------|------|-------------|
| `coordinator` | NeurofeedbackCoordinator | Provides activation data |
| `mask_names` | list[str] | Which masks to use |
| `feedback_mode` | bool | Whether feedback is active |
| `session_state` | SessionState | For state persistence |
| `presenter_monitor` | str | Which monitor to display on |

Accept these in your `__init__` even if you don't use them all:

```python
def __init__(
    self,
    coordinator,
    mask_names=None,
    feedback_mode=True,
    session_state=None,
    presenter_monitor="right",
    **kwargs,  # Catch any extras
):
```

## Best Practices

1. **Set `task_id` as class attribute** matching your metadata's `task_id`
2. **Accept `**kwargs`** in `__init__` for forward compatibility
3. **Check `is_baseline`** before updating display during baseline period
4. **Check `feedback_mode`** to handle transfer runs (display but don't move)
5. **Call `coordinator.check_error()`** in your main loop to propagate errors
6. **Use `coordinator.is_complete()`** to detect session end
7. **Save state in `cleanup()`** using `session_state.record_run()`

## Testing Your Task

1. **Unit test** your task with a mock coordinator
2. **Integration test** using the DICOM simulator:
   ```bash
   python -m pineuro.simulator --source /path/to/dicoms --target /tmp/incoming
   ```
3. **GUI test** by selecting your task in the study editor

## Troubleshooting

### Task not appearing in GUI
- Ensure your module is imported in `tasks/__init__.py`
- Check that both `register_task()` and `register_task_class()` are called
- Verify `task_id` matches between class attribute and metadata

### Parameters not saving
- Ensure parameter `key` matches your `__init__` argument name
- Check parameter `type` matches the expected Python type

### Session state not persisting
- Call `session_state.record_run()` in your `cleanup()` method
- Verify `session_state_fields` are defined in metadata
