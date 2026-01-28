# pineuro Extensibility Plan v3: Completing Plug-and-Play

## Executive Summary

This plan completes the task-agnostic framework by implementing **Phase 5 from V2** (post-run questions in BaseTask). Currently, custom tasks must copy ~100 lines of slider code from BallTask. After this change, questions defined in `TaskMetadata.post_run_questions` will display automatically.

**Key deliverable**: Add `_run_slider()` and `_show_post_run_questions()` to BaseTask, with auto-show in `cleanup()`. Custom tasks just define questions in metadata - they display automatically.

---

## Current State

### What's Complete (from V2)

| Phase | Status | Implementation |
|-------|--------|----------------|
| 1. Extended TaskMetadata | ✅ | `n_networks`, `session_state_fields`, `post_run_questions` in [task_registry.py](../src/pineuro/tasks/task_registry.py) |
| 2. Config-driven SessionState | ✅ | `task_state` dict with `get()`/`set()` in [session.py](../src/pineuro/tasks/session.py) |
| 3. Factory in pipeline_runner | ✅ | `create_task()` called dynamically in [pipeline_runner.py](../src/pineuro/gui/components/pipeline_runner.py) |
| 4. BallTask metadata | ✅ | Full metadata with all fields in [ball.py](../src/pineuro/tasks/ball.py) |
| 5. Question logic in BaseTask | ❌ | **NOT IMPLEMENTED** |
| 6. Network count validation | ✅ | Validation in [factory.py](../src/pineuro/tasks/factory.py) |
| 7. Example + docs | ✅ | [thermometer.py](../src/pineuro/tasks/thermometer.py) + [CUSTOM_TASKS.md](CUSTOM_TASKS.md) |

### The Gap

**ThermometerTask defines questions but never shows them:**

```python
# thermometer.py lines 395-408 - defined but never displayed
post_run_questions=[
    PostRunQuestion(text="How focused did you feel during the task?", ...),
    PostRunQuestion(text="How well could you control the bar?", ...),
]
```

**BallTask hardcodes questions instead of reading from metadata:**

```python
# ball.py lines 742-764 - hardcoded text, not from metadata
self._run_slider(
    question_text='How often were you using the Mindful Describing practice?',
    left_label='Never',
    right_label='Always',
)
```

---

## Implementation Plan

### Phase 1: Add Shared Slider Infrastructure to BaseTask

**File**: [src/pineuro/tasks/base.py](../src/pineuro/tasks/base.py)

Add attributes that subclasses need:

```python
class BaseTask(ABC):
    task_id: str = "base"

    def __init__(self, coordinator, ...):
        # Existing init code...

        # Add these for question support
        self.feedback_mode: bool = True
        self._session_state: Optional["SessionState"] = None
        self._questions_file = None  # For logging responses
```

Add reusable slider method:

```python
def _run_slider(
    self,
    question_text: str,
    left_label: str = "Never",
    right_label: str = "Always",
    left_button: str = "1",
    right_button: str = "2",
    enter_button: str = "3",
) -> float:
    """Display a slider question and return the response (0-100).

    Creates PsychoPy slider UI, waits for response, logs to file if configured.

    Parameters
    ----------
    question_text : str
        The question to display
    left_label : str
        Label for left end of slider (0)
    right_label : str
        Label for right end of slider (100)
    left_button : str
        Key to move slider left
    right_button : str
        Key to move slider right
    enter_button : str
        Key to confirm selection

    Returns
    -------
    float
        Response value 0-100
    """
    from psychopy import visual, event, core

    # Create slider UI elements
    slider = visual.Slider(
        self.win,
        ticks=[0, 100],
        labels=[left_label, right_label],
        startValue=50,
        size=(0.8, 0.05),
        pos=(0, -0.1),
        style='rating',
        granularity=1,
    )

    question = visual.TextStim(
        self.win,
        text=question_text,
        pos=(0, 0.2),
        height=0.04,
        wrapWidth=1.5,
        color='white',
    )

    # Wait for response
    start_time = core.getTime()
    event.clearEvents('keyboard')

    while True:
        # Check for stop signal
        if hasattr(self, 'coordinator') and self.coordinator.was_stopped():
            raise KeyboardInterrupt("Stopped by user")

        keys = event.getKeys(keyList=[left_button, right_button, enter_button, 'escape'])

        for key in keys:
            if key == 'escape':
                raise KeyboardInterrupt("Escaped during question")
            elif key == left_button:
                slider.markerPos = max(0, slider.markerPos - 5)
            elif key == right_button:
                slider.markerPos = min(100, slider.markerPos + 5)
            elif key == enter_button:
                rt = core.getTime() - start_time
                rating = slider.markerPos

                # Log response if file configured
                if self._questions_file is not None:
                    import csv
                    writer = csv.writer(self._questions_file)
                    writer.writerow([
                        'feedback' if self.feedback_mode else 'transfer',
                        question_text,
                        rating,
                        f'{rt:.3f}',
                    ])
                    self._questions_file.flush()

                return rating

        # Draw
        question.draw()
        slider.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()
        core.wait(0.01)
```

---

### Phase 2: Add Generic Question Display to BaseTask

**File**: [src/pineuro/tasks/base.py](../src/pineuro/tasks/base.py)

```python
def _show_post_run_questions(
    self,
    left_button: str = "1",
    right_button: str = "2",
    enter_button: str = "3",
) -> list[dict]:
    """Display post-run questions defined in task metadata.

    Called automatically by cleanup(). Tasks with no questions defined
    in metadata will skip this (no-op).

    Reads questions from TaskMetadata.post_run_questions and displays
    each one based on its condition:
    - "always": Show for all runs
    - "feedback_only": Only show when feedback_mode is True
    - "last_feedback_only": Only show on final feedback run

    Parameters
    ----------
    left_button : str
        Key to move slider left
    right_button : str
        Key to move slider right
    enter_button : str
        Key to confirm selection

    Returns
    -------
    list of dict
        List of {"question": str, "response": float} for each answered question
    """
    metadata = self.get_task_metadata()
    if not metadata or not metadata.post_run_questions:
        return []  # No-op for tasks without questions

    responses = []

    for question in metadata.post_run_questions:
        # Check condition
        if question.condition == "feedback_only":
            if not self.feedback_mode:
                continue

        elif question.condition == "last_feedback_only":
            is_last = False
            if self._session_state is not None:
                is_last = self._session_state.is_last_feedback_run
            if not (self.feedback_mode and is_last):
                continue

        # Display slider and collect response
        response = self._run_slider(
            question_text=question.text,
            left_label=question.left_label,
            right_label=question.right_label,
            left_button=left_button,
            right_button=right_button,
            enter_button=enter_button,
        )

        responses.append({
            "question": question.text,
            "response": response,
        })

    return responses
```

### Phase 2b: Update BaseTask.cleanup() to Auto-Show Questions

Make `cleanup()` concrete (not abstract) so it auto-handles questions:

```python
def cleanup(self):
    """Clean up resources after task completes.

    Default implementation:
    1. Shows post-run questions (if defined in metadata)
    2. Closes the PsychoPy window (if exists)

    Subclasses can override to add custom cleanup logic, but should
    call super().cleanup() at the end to ensure questions are shown
    and window is closed.
    """
    # Show post-run questions (no-op if none defined)
    if hasattr(self, 'win') and self.win is not None:
        self._show_post_run_questions()

    # Close window
    if hasattr(self, 'win') and self.win is not None:
        self.win.close()
```

This means:
- **Tasks with questions**: Just define them in metadata, they appear automatically
- **Tasks without questions**: cleanup() is a no-op for questions, just closes window
- **Tasks needing custom cleanup**: Override cleanup() and call `super().cleanup()` at end

---

### Phase 3: Update ThermometerTask

**File**: [src/pineuro/tasks/thermometer.py](../src/pineuro/tasks/thermometer.py)

1. Fix inheritance (call `super().__init__()`):

```python
def __init__(self, coordinator, mask_names=None, target_level=0.5, ...):
    # Initialize base class properly
    super().__init__(coordinator, mask_name=mask_names[0] if mask_names else None)

    # Set attributes needed for questions
    self.feedback_mode = feedback_mode
    self._session_state = session_state

    # Task-specific params...
```

2. Simplify cleanup to use base class (questions auto-show now):

```python
def cleanup(self):
    """Clean up resources and save state."""
    # Save session state (task-specific)
    if self._session_state is not None:
        self._session_state.record_run(...)

    # Let base class handle questions + window close
    super().cleanup()
```

3. Add `bar_color` to metadata:

```python
TaskParameter(
    key="bar_color",
    label="Bar Color",
    type="str",
    default="green",
    hint="Color of the thermometer bar",
),
```

4. Remove duplicated `_capture_and_emit_screenshot()` (now inherited from base)

---

### Phase 4: Refactor BallTask to Use Base Methods

**File**: [src/pineuro/tasks/ball.py](../src/pineuro/tasks/ball.py)

BallTask has custom behavior (instruction screen before questions, configurable buttons), so it overrides `_show_post_run_questions()` but delegates the actual questions to base:

```python
def _show_post_run_questions(self) -> None:
    """Display post-run slider questions.

    Shows BallTask-specific instruction screen, then uses base class
    to display questions from metadata.
    """
    # Show instruction screen (BallTask-specific)
    self.slider_instruction_text.draw()
    self._capture_and_emit_screenshot(force=True)
    self.win.flip()

    # Wait for button press to continue
    event.clearEvents('keyboard')
    while True:
        if self.coordinator.was_stopped():
            raise KeyboardInterrupt("Stopped by user")
        keys = event.getKeys(keyList=[self.left_button, self.right_button, self.enter_button])
        if keys:
            break
        core.wait(0.01)

    # Delegate to base class for actual questions (reads from metadata)
    responses = super()._show_post_run_questions(
        left_button=self.left_button,
        right_button=self.right_button,
        enter_button=self.enter_button,
    )

    logger.info(f"Post-run questions complete: {len(responses)} responses")
```

BallTask's `cleanup()` can then rely on the auto-show behavior, which will call its overridden `_show_post_run_questions()`.

This removes ~50 lines of duplicated slider code while preserving:
- BallTask's custom instruction screen
- BallTask's configurable button mapping
- All question conditions (always, feedback_only, last_feedback_only)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/pineuro/tasks/base.py` | Add `_run_slider()`, `_show_post_run_questions()`, `feedback_mode`, `_session_state` |
| `src/pineuro/tasks/thermometer.py` | Fix inheritance, call `_show_post_run_questions()`, add `bar_color` param |
| `src/pineuro/tasks/ball.py` | Refactor to call `super()._show_post_run_questions()` |

---

## Verification Plan

1. **Unit test base methods**: Test `_run_slider()` with mock window
2. **ThermometerTask integration**: Run task, verify both questions appear
3. **BallTask regression**: Run task, verify all 7 questions work correctly
4. **Condition logic**: Test `feedback_only` and `last_feedback_only` conditions
5. **Custom task test**: Create minimal task with only metadata questions, verify they display

---

## Implementation Order

1. **base.py** - Add shared infrastructure (no breaking changes)
2. **thermometer.py** - Simpler task, validates base implementation works
3. **ball.py** - Refactor last, verify regression-free

---

## Migration Notes

- **Backwards compatible**: Existing BallTask behavior unchanged until refactored
- **Opt-in for existing tasks**: Tasks can call `_show_post_run_questions()` when ready
- **New tasks get it free**: Just define questions in metadata, framework handles display

---

## Additional Improvements for Better Plug-and-Play

The following are additional friction points identified that could further improve the developer experience. These are **optional enhancements** beyond the core Phase 5 fix.

### Improvement A: Extract `wait_for_trigger()` to BaseTask

**Problem**: Both ThermometerTask and BallTask implement their own trigger-waiting logic (~20-40 lines each).

**Solution**: Add to BaseTask:
```python
def wait_for_trigger(
    self,
    message: str = "Waiting for scanner...",
    trigger_keys: list[str] = None,
) -> None:
    """Wait for scanner trigger before starting task.

    Displays message, polls for trigger keys, handles escape.
    Calls coordinator.send_trigger() when received.
    """
    trigger_keys = trigger_keys or ["t", "+", "5", "equal"]

    waiting_text = visual.TextStim(self.win, text=message, ...)
    waiting_text.draw()
    self.win.flip()

    event.clearEvents('keyboard')
    while True:
        keys = event.getKeys(keyList=trigger_keys + ['escape'])
        if 'escape' in keys:
            raise KeyboardInterrupt("Escaped during trigger wait")
        if keys:
            break
        core.wait(0.01)

    self.coordinator.send_trigger()
```

**Savings**: ~20 lines per task

---

### Improvement B: Extract `show_instructions()` to BaseTask

**Problem**: BallTask has a 54-line `_show_instructions()` method. Other tasks would need similar.

**Solution**: Add to BaseTask:
```python
def show_instructions(
    self,
    slides: list[str],
    advance_keys: list[str] = None,
) -> None:
    """Display instruction slides, advancing on key press.

    Parameters
    ----------
    slides : list of str
        Text content for each slide
    advance_keys : list of str
        Keys that advance to next slide (default: any key)
    """
```

**Savings**: ~50 lines per task that needs instructions

---

### Improvement C: Automatic Session State Recording

**Problem**: Every task's `cleanup()` must call `session_state.record_run()` manually.

**Solution**: Add to BaseTask:
```python
def _record_run_completion(self) -> None:
    """Record run completion to session state.

    Subclasses should set self._task_state_snapshot dict before cleanup.
    """
    if self._session_state is not None:
        self._session_state.record_run(
            run_type="feedback" if self.feedback_mode else "transfer",
            volumes_processed=getattr(self, '_volumes_processed', 0),
            task_state_snapshot=getattr(self, '_task_state_snapshot', {}),
        )
```

Then BaseTask.cleanup() could call this automatically.

**Savings**: ~15 lines per task

---

### Improvement D: Decorator-Based Registration

**Problem**: Developers must call both `register_task()` and `register_task_class()` separately.

**Solution**: Provide a class decorator:
```python
@pineuro_task(THERMOMETER_METADATA)
class ThermometerTask(BaseTask):
    task_id = "thermometer"
    ...
```

The decorator would:
1. Verify `task_id` matches metadata
2. Call `register_task(metadata)`
3. Call `register_task_class(task_id, cls)`

**Savings**: 2 lines, but more importantly reduces chance of registration errors

---

### Improvement E: UI Helpers Module

**Problem**: Each task reimplements common visual elements (fixation cross, etc.)

**Solution**: Create `src/pineuro/tasks/ui_helpers.py`:
```python
def create_fixation_cross(win, size=0.05, color='white') -> visual.ShapeStim:
    """Create a fixation cross stimulus."""

def display_text(win, text: str, duration: float = None, wait_key: bool = False):
    """Display text, optionally for duration or until keypress."""
```

**Savings**: Variable, but reduces cognitive load significantly

---

### Improvement F: Dynamic Task Discovery (External Plugins)

**Problem**: Adding a new task requires modifying `__init__.py`. No way to load external plugins.

**Solution**: Add auto-discovery in `__init__.py`:
```python
def discover_tasks(plugin_dirs: list[Path] = None):
    """Discover and register tasks from plugin directories.

    Looks for Python modules containing TaskMetadata and BaseTask subclasses.
    """
```

This would enable:
- Research groups to share task packages
- Tasks developed outside pineuro to be loaded at runtime
- Configuration-based task loading

**Impact**: Opens ecosystem for community contributions

---

## Summary: Lines of Code for a New Task

| Component | Current | After V3 Core | After All Improvements |
|-----------|---------|---------------|------------------------|
| Window setup | 15 | 15 | 5 (with helper) |
| Trigger waiting | 20 | 20 | 0 (use base) |
| Instructions | 50 | 50 | 0 (use base) |
| Main run loop | 15 | 15 | 15 |
| Update logic | varies | varies | varies |
| Cleanup | 20 | 10 | 5 (auto-record) |
| Post-run questions | 100 | 0 | 0 |
| Metadata | 60 | 60 | 60 |
| Registration | 2 | 2 | 0 (decorator) |
| **Total boilerplate** | **~280** | **~170** | **~85** |

The V3 core fix (post-run questions) is the highest-impact single change. The additional improvements would further reduce boilerplate from ~170 lines to ~85 lines.

---

## Recommended Priority

1. **V3 Core (this plan)**: Post-run questions in BaseTask - **HIGH PRIORITY**
2. **Improvement A**: `wait_for_trigger()` - Medium priority, easy win
3. **Improvement C**: Auto session recording - Medium priority, reduces errors
4. **Improvement D**: Decorator registration - Low effort, nice-to-have
5. **Improvement B**: `show_instructions()` - Medium effort, useful for complex tasks
6. **Improvement E**: UI helpers - Higher effort, good for v4
7. **Improvement F**: Plugin discovery - Future enhancement
