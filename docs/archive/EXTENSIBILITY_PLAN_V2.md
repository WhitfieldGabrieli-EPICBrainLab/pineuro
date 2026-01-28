# pineuro Extensibility Plan v2: Task-Agnostic Framework

## Executive Summary

This plan makes pineuro truly task-agnostic so that custom neurofeedback tasks are "plug and play". Tasks designed in Python will automatically appear in the GUI dropdown with configurable parameters, session state persistence, and post-run questions.

**Key principles:**
- Tasks declare everything they need via `TaskMetadata`
- The framework handles instantiation, state persistence, and UI generation
- Existing BallTask continues to work unchanged

---

## Current State

### What Already Works Well

1. **BaseTask abstract class** ([base.py](../src/pineuro/tasks/base.py)) - Task-agnostic scaffold with `setup()`, `update()`, `cleanup()` hooks
2. **TaskMetadata + TaskParameter** ([task_registry.py](../src/pineuro/tasks/task_registry.py)) - Declarative parameter definitions
3. **create_task() factory** ([factory.py](../src/pineuro/tasks/factory.py)) - Metadata-driven task instantiation
4. **Study Editor GUI** - Dynamically discovers tasks via `list_available_tasks()` and builds UI from metadata

### What Needs Fixing

| Problem | Location | Impact |
|---------|----------|--------|
| Hardcoded BallTask instantiation | [pipeline_runner.py:867-899](../src/pineuro/gui/components/pipeline_runner.py) | Custom tasks can't run through GUI |
| Ball-specific SessionState fields | [session.py:59-99](../src/pineuro/tasks/session.py) | Custom tasks can't persist state |
| Hardcoded post-run questions | [ball.py:705-786](../src/pineuro/tasks/ball.py) | Custom tasks can't have questions |
| Assumes exactly 2 networks | [ball.py:193-201](../src/pineuro/tasks/ball.py) | Single-network tasks unsupported |

---

## Implementation Plan

### Phase 1: Extend TaskMetadata

**File: [task_registry.py](../src/pineuro/tasks/task_registry.py)**

Add new dataclasses for session state and questions:

```python
@dataclass
class SessionStateField:
    """Definition of a task-specific persistent state field."""
    key: str                          # Field name (e.g., "hit_counters")
    type: Literal["float", "int", "bool", "list[float]", "list[int]"]
    default: Any                      # Default value
    description: str = ""             # Documentation

@dataclass
class PostRunQuestion:
    """Definition of a post-run slider question."""
    text: str                         # Question text
    left_label: str = "Never"         # Left end of slider
    right_label: str = "Always"       # Right end of slider
    condition: Literal["always", "feedback_only", "last_feedback_only"] = "always"
```

Extend `TaskMetadata`:

```python
@dataclass
class TaskMetadata:
    task_id: str
    display_name: str
    description: str
    parameters: List[TaskParameter]
    cross_parameter_validator: Optional[Callable] = None
    # NEW FIELDS:
    n_networks: int = 1               # Number of masks required (1, 2, or N)
    session_state_fields: List[SessionStateField] = field(default_factory=list)
    post_run_questions: List[PostRunQuestion] = field(default_factory=list)
```

---

### Phase 2: Config-Driven SessionState

**File: [session.py](../src/pineuro/tasks/session.py)**

Refactor `SessionState` to store task-declared fields dynamically:

```python
@dataclass
class SessionState:
    """Generic session state that stores task-defined fields."""
    session_dir: Path
    task_id: str
    task_state: Dict[str, Any] = field(default_factory=dict)  # Dynamic fields
    completed_runs: list[RunInfo] = field(default_factory=list)

    @classmethod
    def create_for_task(cls, session_dir: Path, task_id: str) -> "SessionState":
        """Create session state with defaults from task metadata."""
        from .task_registry import get_task_metadata
        metadata = get_task_metadata(task_id)
        defaults = {f.key: f.default for f in metadata.session_state_fields}
        return cls(session_dir=session_dir, task_id=task_id, task_state=defaults)

    def get(self, key: str) -> Any:
        """Get a task state field."""
        return self.task_state.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a task state field."""
        self.task_state[key] = value
```

**Backwards compatibility** - Add property accessors for existing BallTask fields:

```python
    @property
    def scale_factor(self) -> float:
        return self.get("scale_factor") or 10.0

    @scale_factor.setter
    def scale_factor(self, value: float):
        self.set("scale_factor", value)

    @property
    def hit_counters(self) -> list[int]:
        return self.get("hit_counters") or [0, 0]

    # ... similar for circle_radii
```

---

### Phase 3: Use Factory in Pipeline Runner

**File: [pipeline_runner.py](../src/pineuro/gui/components/pipeline_runner.py)** (lines 867-899)

Replace hardcoded BallTask instantiation:

```python
# BEFORE (hardcoded):
from pineuro.tasks import BallTask
task = BallTask(
    coordinator=coordinator,
    mask_names=ball_mask_names,
    scale_factor=task_config.get("initial_scale_factor", 10.0),
    # ... many more hardcoded parameters
)

# AFTER (factory-based):
from pineuro.tasks import create_task

task_type = config["task"].get("task_type", "ball")
task_config = config["task"].get(task_type, {})

factory_config = {
    **task_config,
    "mask_names": mask_names,
    "feedback_mode": feedback_mode,
    "presenter_monitor": presenter_monitor,
}

task = create_task(
    task_id=task_type,
    coordinator=coordinator,
    config=factory_config,
    session_state=session_state,
    output_dir=str(output_dir),
)
```

---

### Phase 4: Update BallTask Metadata

**File: [ball.py](../src/pineuro/tasks/ball.py)** (around line 1304)

Add new metadata fields to `BALL_TASK_METADATA`:

```python
BALL_TASK_METADATA = TaskMetadata(
    task_id="ball",
    display_name="Ball Task",
    description="Dual-network neurofeedback with ball control",
    n_networks=2,
    parameters=[...],  # existing parameters unchanged
    session_state_fields=[
        SessionStateField(
            key="scale_factor",
            type="float",
            default=10.0,
            description="Adaptive difficulty scale factor",
        ),
        SessionStateField(
            key="hit_counters",
            type="list[int]",
            default=[0, 0],
            description="Cumulative hit counts per network",
        ),
        SessionStateField(
            key="circle_radii",
            type="list[float]",
            default=[0.15, 0.15],
            description="Current circle radii",
        ),
    ],
    post_run_questions=[
        PostRunQuestion(
            text="How often were you using the Mindful Describing practice?",
            left_label="Never",
            right_label="Always",
            condition="always",
        ),
        PostRunQuestion(
            text="How often did you check the position of the ball?",
            left_label="Never",
            right_label="All the time",
            condition="always",
        ),
        PostRunQuestion(
            text="How difficult was it to apply Mindful Describing?",
            left_label="Not at all",
            right_label="Very Difficult",
            condition="always",
        ),
        PostRunQuestion(
            text="How calm do you feel right now?",
            left_label="Not at all",
            right_label="Very calm",
            condition="always",
        ),
        PostRunQuestion(
            text="The ball seemed to move up while I was practicing Mindful Describing",
            left_label="False",
            right_label="True",
            condition="last_feedback_only",
        ),
        PostRunQuestion(
            text="The ball seemed to move down while my mind was wandering.",
            left_label="False",
            right_label="True",
            condition="last_feedback_only",
        ),
        PostRunQuestion(
            text="The ball seemed to move down while I was trying to control it.",
            left_label="False",
            right_label="True",
            condition="last_feedback_only",
        ),
    ],
    cross_parameter_validator=_validate_ball_hits,
)
```

---

### Phase 5: Move Question Logic to BaseTask

**File: [base.py](../src/pineuro/tasks/base.py)**

Add generic question display method that reads from metadata:

```python
def _show_post_run_questions(self) -> List[Dict[str, Any]]:
    """Display post-run questions from task metadata.

    Returns list of {question, response} dicts for logging.
    """
    from .task_registry import get_task_metadata

    metadata = get_task_metadata(self.task_id)
    responses = []

    for question in metadata.post_run_questions:
        # Check condition
        if question.condition == "last_feedback_only":
            if not (self._session_state and self._session_state.is_last_feedback_run):
                continue
        elif question.condition == "feedback_only":
            if not self.feedback_mode:
                continue

        # Display slider and collect response
        response = self._run_slider(
            question_text=question.text,
            left_label=question.left_label,
            right_label=question.right_label,
        )
        responses.append({"question": question.text, "response": response})

    return responses
```

Refactor BallTask's `_show_post_run_questions()` to call this base implementation.

---

### Phase 6: Network Count Validation

**File: [factory.py](../src/pineuro/tasks/factory.py)**

Add validation that coordinator has enough masks:

```python
def create_task(...) -> BaseTask:
    metadata = get_task_metadata(task_id)

    # Validate network count
    mask_names = config.get("mask_names", coordinator.mask_names)
    if len(mask_names) < metadata.n_networks:
        raise ValueError(
            f"Task '{task_id}' requires {metadata.n_networks} networks, "
            f"but only {len(mask_names)} provided"
        )

    # ... rest of creation logic
```

---

### Phase 7: Example Task + Documentation

**New file: [thermometer.py](../src/pineuro/tasks/thermometer.py)**

Create a simple single-network example:

```python
"""
Example single-network task demonstrating the task-agnostic framework.
"""
from psychopy import visual
from .base import BaseTask
from .task_registry import TaskMetadata, TaskParameter, SessionStateField, PostRunQuestion, register_task
from .factory import register_task_class


class ThermometerTask(BaseTask):
    """Single-network thermometer feedback task.

    Displays a vertical bar that fills based on activation level.
    """

    task_id = "thermometer"

    def __init__(self, coordinator, target_level: float = 0.5, **kwargs):
        super().__init__(coordinator, **kwargs)
        self.target_level = target_level

    def setup(self):
        self.win = visual.Window([800, 600], fullscr=False, color='black')
        self.bar = visual.Rect(self.win, width=0.2, height=0.0,
                               pos=(0, -0.4), fillColor='green')
        self.target_line = visual.Line(self.win, start=(-0.15, self.target_level - 0.4),
                                        end=(0.15, self.target_level - 0.4),
                                        lineColor='yellow', lineWidth=3)

    def update(self, data: dict):
        # Scale activation to bar height (0-0.8)
        height = min(0.8, max(0.0, data['activation'] * 0.8))
        self.bar.height = height
        self.bar.pos = (0, -0.4 + height/2)

        self.bar.draw()
        self.target_line.draw()
        self.win.flip()

    def cleanup(self):
        self.win.close()


THERMOMETER_METADATA = TaskMetadata(
    task_id="thermometer",
    display_name="Thermometer Task",
    description="Single-network feedback with vertical bar display",
    n_networks=1,
    parameters=[
        TaskParameter(
            key="target_level",
            label="Target Level",
            type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            hint="Target activation level (0-1)",
        ),
    ],
    session_state_fields=[
        SessionStateField(
            key="time_at_target",
            type="float",
            default=0.0,
            description="Cumulative time spent at target level",
        ),
    ],
    post_run_questions=[
        PostRunQuestion(
            text="How focused did you feel during the task?",
            left_label="Not at all",
            right_label="Extremely",
        ),
    ],
)

register_task(THERMOMETER_METADATA)
register_task_class("thermometer", ThermometerTask)
```

**New file: [CUSTOM_TASKS.md](CUSTOM_TASKS.md)**

Developer guide covering:
1. Minimal task template
2. TaskMetadata reference
3. Session state usage
4. Post-run questions
5. N-network examples
6. Testing your task

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/pineuro/tasks/task_registry.py` | Add `SessionStateField`, `PostRunQuestion`; extend `TaskMetadata` |
| `src/pineuro/tasks/session.py` | Make `SessionState` config-driven with backwards-compatible properties |
| `src/pineuro/tasks/factory.py` | Add network count validation |
| `src/pineuro/tasks/base.py` | Add `task_id` property; add generic `_show_post_run_questions()` |
| `src/pineuro/tasks/ball.py` | Update `BALL_TASK_METADATA` with new fields; refactor question display |
| `src/pineuro/gui/components/pipeline_runner.py` | Use `create_task()` factory instead of hardcoded BallTask |
| `src/pineuro/tasks/__init__.py` | Export new classes; optionally add auto-discovery |

## New Files

| File | Purpose |
|------|---------|
| `src/pineuro/tasks/thermometer.py` | Example single-network task |
| `docs/CUSTOM_TASKS.md` | Developer documentation |

---

## Verification Plan

1. **Regression test**: Run existing BallTask through GUI, verify unchanged behavior
2. **Factory integration**: Verify `create_task()` is called, task type comes from config
3. **Session state**: Verify state persists across runs for both BallTask and ThermometerTask
4. **Questions**: Verify questions display from metadata, conditions work correctly
5. **N-network support**: Verify ThermometerTask (1 network) works alongside BallTask (2 networks)
6. **Documentation test**: Follow CUSTOM_TASKS.md to create a new task from scratch

---

## Migration Notes

- **Backwards compatible**: Existing study configs continue to work
- **No breaking changes**: BallTask API unchanged
- **Gradual adoption**: Tasks can migrate to new patterns incrementally
