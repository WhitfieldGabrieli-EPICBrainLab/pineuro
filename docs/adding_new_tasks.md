# Adding New Tasks to pineuro GUI

## Overview

pineuro supports pluggable task types through a registry system. To add a new task, you only need to:
1. Define task metadata with parameters
2. Register the task in the task registry
3. Implement the PsychoPy task class (optional if using existing task code)

The GUI will automatically discover and display your task's parameters without any GUI code changes.

## Quick Start Example

Here's how to add a simple "Rest" task with minimal parameters:

```python
# Create src/pineuro/tasks/rest.py

from .base import BaseTask
from .task_registry import TaskMetadata, TaskParameter, register_task

class RestTask(BaseTask):
    """Resting state task with fixation cross."""

    def setup(self):
        from psychopy import visual
        self.win = visual.Window(fullscr=True, color=self.config.get("background_color", "black"))
        self.cross = visual.ShapeStim(
            self.win,
            vertices='cross',
            fillColor=self.config.get("cross_color", "white"),
        )

    def update(self, data: dict):
        self.cross.draw()
        self.win.flip()

    def cleanup(self):
        self.win.close()


# Task metadata (at bottom of file)
REST_TASK_METADATA = TaskMetadata(
    task_id="rest",
    display_name="Resting State",
    description="Fixation cross for resting state acquisition",
    parameters=[
        TaskParameter(
            key="cross_color",
            label="Fixation Cross Color",
            type="str",
            default="white",
            hint="Color name or hex code",
        ),
        TaskParameter(
            key="background_color",
            label="Background Color",
            type="str",
            default="black",
            hint="Color name or hex code",
        ),
    ],
)

# Auto-register on import
register_task(REST_TASK_METADATA)
```

Then add to `src/pineuro/tasks/__init__.py`:
```python
from .rest import RestTask  # Registers "rest" task
```

That's it! The GUI will now show "Resting State" in the task type dropdown with these two parameters.

## Step-by-Step Guide

### 1. Create Your Task File

Create a new file `src/pineuro/tasks/nback.py`:

```python
from typing import Dict, Any
from .base import BaseTask
from .task_registry import TaskMetadata, TaskParameter, register_task

class NBackTask(BaseTask):
    """N-back working memory task."""

    def __init__(self, coordinator, config, **kwargs):
        super().__init__(coordinator, **kwargs)
        self.config = config
        self.n = config.get("n", 2)
        # ... initialize from config

    def setup(self):
        """Create PsychoPy window and stimuli."""
        from psychopy import visual
        self.win = visual.Window(fullscr=True)
        self.text_stim = visual.TextStim(self.win)
        # ... create more stimuli

    def update(self, data: dict):
        """Update display each frame."""
        # ... update stimuli based on activation
        self.text_stim.draw()
        self.win.flip()

    def cleanup(self):
        """Release resources."""
        self.win.close()


# Task metadata (at bottom of same file)
def _validate_nback(config: Dict[str, Any]) -> None:
    """Cross-parameter validator for N-back task."""
    n = config.get("n")
    if n is not None and n < 0:
        raise ValueError("N-back level cannot be negative")

NBACK_TASK_METADATA = TaskMetadata(
    task_id="nback",  # Unique identifier
    display_name="N-Back Task",  # Shown in dropdown
    description="Working memory task with N-back paradigm",
    parameters=[
        TaskParameter(
            key="n",
            label="N-Back Level",
            type="int",
            default=2,
            min_value=0,
            max_value=5,
            hint="Number of items to remember (0-5)",
        ),
        TaskParameter(
            key="block_duration",
            label="Block Duration (seconds)",
            type="float",
            default=30.0,
            min_value=10.0,
            max_value=120.0,
            hint="Duration of each task block",
        ),
        TaskParameter(
            key="enable_audio_cues",
            label="Enable Audio Cues",
            type="bool",
            default=True,
            hint="Play sound when stimulus appears",
        ),
        TaskParameter(
            key="stimulus_letters",
            label="Stimulus Letters",
            type="list[str]",
            default=["A", "B", "C", "D", "E"],
            hint="Letters to use as stimuli (comma-separated)",
        ),
    ],
    cross_parameter_validator=_validate_nback,  # Optional
)

# Auto-register on import
register_task(NBACK_TASK_METADATA)
```

### 2. Register in `__init__.py`

Add your task to `src/pineuro/tasks/__init__.py`:

```python
# Import tasks to trigger auto-registration
from .ball import BallTask  # Registers "ball" task
from .nback import NBackTask  # Registers "nback" task  <-- Add this line
```

That's it! Your task is now available in the GUI.

### 3. Parameter Types

The GUI automatically creates appropriate controls for each parameter type:

| Type | GUI Control | Example Use | Notes |
|------|-------------|-------------|-------|
| `bool` | Checkbox | Enable/disable features | Value is True/False |
| `int` | TextField (numeric) | Discrete values (levels, counts) | Supports min/max validation |
| `float` | TextField (numeric) | Continuous values (durations, scales) | Supports min/max validation |
| `str` | TextField | Text inputs (colors, button keys) | Free text |
| `list[str]` | TextField | Multiple values (trigger keys, stimuli) | Shown as comma-separated |

### 4. Validation

**Automatic validation:**
- Numeric parameters respect `min_value` and `max_value`
- Type conversions are validated (e.g., "abc" fails for int)

**Custom validation:**
```python
def custom_validator(value):
    if value not in ["red", "green", "blue"]:
        raise ValueError("Color must be red, green, or blue")

TaskParameter(
    key="color",
    label="Color",
    type="str",
    default="red",
    validator=custom_validator,
)
```

**Cross-parameter validation:**
```python
def validate_thresholds(config: Dict[str, Any]) -> None:
    min_val = config.get("min_threshold")
    max_val = config.get("max_threshold")
    if min_val is not None and max_val is not None:
        if min_val >= max_val:
            raise ValueError("Min threshold must be < max threshold")

TaskMetadata(
    task_id="mytask",
    ...,
    cross_parameter_validator=validate_thresholds,
)
```

### 5. Configuration Storage

Task configurations are stored in YAML with this structure:

```yaml
task:
  task_type: "nback"  # Current active task
  max_feedback_runs: 2  # Common parameter

  # Task-specific configs (all preserved)
  ball:
    initial_scale_factor: 10.0
    min_hits_threshold: 3
    max_hits_threshold: 5
    # ... other ball params

  nback:
    n: 2
    block_duration: 30.0
    enable_audio_cues: true
    stimulus_letters: ["A", "B", "C", "D", "E"]
```

**Key points:**
- All task configs are preserved when switching task types
- Only the active task (specified by `task_type`) is used during execution
- Easy to switch between tasks without losing configurations

### 6. Testing Your Task

1. **Test in GUI:**
   ```bash
   python -m pineuro.gui.app
   ```
   - Create or edit a study
   - Go to "Task Parameters" tab
   - Select your task from dropdown
   - Verify all parameters appear with correct controls
   - Save and reload to verify config persistence

2. **Test validation:**
   - Try invalid values (e.g., negative numbers, out-of-range)
   - Verify error messages appear
   - Test cross-parameter validation

3. **Test configuration:**
   - Save a study with your task configured
   - Check `~/.pineuro/studies/yourstudy.yaml`
   - Verify task config is correctly formatted

## Complete Example: Stroop Task

```python
# Create src/pineuro/tasks/stroop.py

from typing import Dict, Any
from .base import BaseTask
from .task_registry import TaskMetadata, TaskParameter, register_task

class StroopTask(BaseTask):
    """Stroop color-word interference task."""

    def __init__(self, coordinator, config, **kwargs):
        super().__init__(coordinator, **kwargs)
        self.config = config
        # ... initialize from config

    def setup(self):
        from psychopy import visual
        self.win = visual.Window(fullscr=True)
        # ... create stimuli

    def update(self, data: dict):
        # ... update based on activation
        self.win.flip()

    def cleanup(self):
        self.win.close()


# Task metadata
def _validate_stroop_timing(config: Dict[str, Any]) -> None:
    """Ensure total timing makes sense."""
    stim_duration = config.get("stimulus_duration", 0)
    isi = config.get("inter_stimulus_interval", 0)
    if stim_duration + isi < 0.5:
        raise ValueError("Total trial time too short (< 0.5s)")

STROOP_TASK_METADATA = TaskMetadata(
    task_id="stroop",
    display_name="Stroop Task",
    description="Color-word interference task with neurofeedback",
    parameters=[
        TaskParameter(
            key="stimulus_duration",
            label="Stimulus Duration (seconds)",
            type="float",
            default=2.0,
            min_value=0.5,
            max_value=5.0,
            hint="How long each word is displayed",
        ),
        TaskParameter(
            key="inter_stimulus_interval",
            label="Inter-Stimulus Interval (seconds)",
            type="float",
            default=1.0,
            min_value=0.0,
            max_value=3.0,
            hint="Pause between stimuli",
        ),
        TaskParameter(
            key="congruent_proportion",
            label="Congruent Trial Proportion",
            type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            hint="Proportion of congruent trials (0.0-1.0)",
        ),
        TaskParameter(
            key="colors",
            label="Colors",
            type="list[str]",
            default=["red", "green", "blue", "yellow"],
            hint="Color words to use (comma-separated)",
        ),
        TaskParameter(
            key="show_feedback_bar",
            label="Show Feedback Bar",
            type="bool",
            default=True,
            hint="Display real-time activation as progress bar",
        ),
    ],
    cross_parameter_validator=_validate_stroop_timing,
)

# Auto-register on import
register_task(STROOP_TASK_METADATA)
```

Then add to `src/pineuro/tasks/__init__.py`:
```python
from .stroop import StroopTask  # Registers "stroop" task
```

Now your Stroop task is available in the GUI with all parameters configurable!

## Best Practices

1. **Parameter Naming:**
   - Use descriptive keys that match SessionConfig structure
   - Use snake_case for keys (e.g., `initial_scale_factor`)
   - Prefix task-specific keys to avoid conflicts if needed

2. **Sensible Defaults:**
   - Provide working defaults for all parameters
   - Defaults should work well for most use cases
   - Document default rationale in hint text

3. **Validation:**
   - Add min/max values for all numeric parameters
   - Use cross-parameter validators for related parameters
   - Provide clear error messages

4. **Documentation:**
   - Write clear, concise labels
   - Provide helpful hints explaining what each parameter does
   - Include units in labels (e.g., "Duration (seconds)")

5. **Backward Compatibility:**
   - Don't change existing parameter keys
   - Add new parameters with sensible defaults
   - Mark deprecated parameters in description

## Troubleshooting

**Task doesn't appear in dropdown:**
- Verify `register_task()` is called
- Check for Python syntax errors in metadata definition
- Restart the GUI application

**Parameters don't save correctly:**
- Check parameter keys match between metadata and implementation
- Verify type conversions in task code (e.g., `int(config["n"])`)
- Inspect YAML file at `~/.pineuro/studies/yourstudy.yaml`

**Validation errors:**
- Check min/max values are reasonable
- Test cross-parameter validator independently
- Look for type mismatches (int vs float)

**GUI shows wrong parameter type:**
- Verify `type` field in TaskParameter
- Check for typos ("int" not "integer")
- Ensure type is one of: "float", "int", "bool", "str", "list[str]"

## Advanced: Dynamic Default Values

If you need default values that depend on other config:

```python
# In your task implementation, not in metadata
def get_default_config(session_config):
    """Generate defaults based on session config."""
    tr = session_config["scan"]["tr"]
    return {
        "stimulus_duration": tr * 0.8,  # 80% of TR
        "isi": tr * 0.2,
    }
```

Then in the GUI, merge these defaults when loading the study.

## Questions?

- Check existing tasks in `src/pineuro/tasks/task_registry.py` for examples
- See `TaskMetadata` and `TaskParameter` docstrings for API details
- Test with the ball task configuration to understand expected behavior
