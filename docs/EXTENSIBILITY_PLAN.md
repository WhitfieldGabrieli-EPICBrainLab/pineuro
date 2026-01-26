# pineuro Extensibility Review: Recommendations for "Plug and Go" Custom Tasks

## Executive Summary

pineuro has excellent infrastructure for extensibility that isn't fully utilized. The core feedback system is task-agnostic, and a sophisticated task registry/factory pattern exists but isn't wired into the GUI. The main gaps are:

1. **Pipeline runner hardcodes BallTask** instead of using the existing factory
2. **Documentation gap** for external developers creating custom tasks
3. **BallTask constructor signature incompatible** with generic factory pattern
4. **No auto-discovery** of user-added task files in the tasks folder

**Scope decisions**:
- Plugin model: Users drop `.py` files into `tasks/` folder (no entry points)
- Mask extraction: Document ICA-only limitation; GLM contrast is future work
- Atlases: Keep Yeo-only for now

---

## Priority 1: Critical Fixes (Blocking Custom Tasks)

### 1.1 Use Task Factory in Pipeline Runner

**File**: `src/pineuro/gui/components/pipeline_runner.py` (lines 670-689)

**Current** (hardcoded):
```python
from pineuro.tasks import BallTask
task = BallTask(
    coordinator=coordinator,
    mask_names=mask_names,
    scale_factor=task_config.get("initial_scale_factor", 10.0),
    ...
)
```

**Required** (factory-based):
```python
from pineuro.tasks import create_task
task = create_task(
    task_id=config["task"]["task_type"],
    coordinator=coordinator,
    config=task_config,
    session_state=session_state,
    output_dir=output_dir,
)
```

### 1.2 Align BallTask Constructor with Factory Interface

**File**: `src/pineuro/tasks/ball.py`

The factory calls `task_class(coordinator=..., session_state=..., output_dir=..., **filtered_config)`, but BallTask constructor uses `mask_names` which isn't in filtered_config (it comes from elsewhere).

**Required changes**:
- BallTask should accept `mask_names` as optional parameter with fallback to `coordinator.mask_names`
- Add `mask_names` to `BALL_TASK_METADATA.parameters` as `list[str]` type, OR
- Pass `mask_names` through `config` dict from pipeline_runner

### 1.3 BaseTask Constructor Signature Mismatch

**File**: `src/pineuro/tasks/base.py` (lines 82-87)

BaseTask `__init__` signature: `(coordinator, mask_name, refresh_rate)`
Factory passes: `(coordinator, session_state, output_dir, **config)`

**Required**: Update BaseTask or ensure all subclasses handle both interfaces.

---

## Priority 2: Documentation for External Developers

### 2.1 Create Task Development Guide

Create `docs/CUSTOM_TASKS.md` with:

1. **Minimal Task Template**:
```python
from pineuro.tasks import (
    BaseTask, TaskMetadata, TaskParameter,
    register_task, register_task_class
)

class MyTask(BaseTask):
    def __init__(self, coordinator, my_param: float = 1.0, **kwargs):
        super().__init__(coordinator, **kwargs)
        self.my_param = my_param

    def setup(self):
        from psychopy import visual
        self.win = visual.Window([800, 600], fullscreen=False)

    def update(self, data: dict):
        # data = {'activation': float, 'volume': int, 'time': float, 'is_baseline': bool}
        if not data['is_baseline']:
            # Update display based on data['activation']
            pass
        self.win.flip()

    def cleanup(self):
        self.win.close()

MY_TASK_METADATA = TaskMetadata(
    task_id="my_task",
    display_name="My Custom Task",
    description="Description for GUI",
    parameters=[
        TaskParameter(key="my_param", label="My Parameter", type="float", default=1.0),
    ],
)

register_task(MY_TASK_METADATA)
register_task_class("my_task", MyTask)
```

2. **Registration checklist**:
   - Define `TaskMetadata` with all GUI-configurable parameters
   - Call `register_task()` and `register_task_class()` at module bottom
   - Import in `tasks/__init__.py` to trigger auto-registration

3. **Data flow diagram**: coordinator → callback → task.update()

4. **Mirror state protocol** for GUI visualization

### 2.2 Create Localizer Development Guide

Create `docs/CUSTOM_LOCALIZERS.md` explaining:
- Current limitation: ICA-based mask extraction only
- How to define task designs via `BlockDesign`, `EventRelatedDesign`
- Future: GLM contrast-based mask extraction

---

## Priority 3: Auto-Discovery of Custom Tasks

### 3.1 Add Task Auto-Discovery in `tasks/__init__.py`

Users should be able to drop a `.py` file into `src/pineuro/tasks/` and have it auto-register.

**File**: `src/pineuro/tasks/__init__.py`

Add auto-discovery logic:
```python
import importlib
import pkgutil
from pathlib import Path

def _discover_tasks():
    """Auto-import all .py files in tasks/ to trigger registration."""
    tasks_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(tasks_dir)]):
        if module_info.name.startswith('_'):
            continue  # Skip __init__, __pycache__, etc.
        if module_info.name in ('base', 'factory', 'task_registry', 'session', 'utils', 'designs'):
            continue  # Skip infrastructure modules
        try:
            importlib.import_module(f'.{module_info.name}', package='pineuro.tasks')
        except Exception as e:
            logger.warning(f"Failed to load task module {module_info.name}: {e}")

_discover_tasks()
```

This replaces the explicit `from .ball import BallTask` with dynamic discovery, so users just drop files in.

### 3.2 Document the Conventions

New task files must:
1. Define a `TaskMetadata` and call `register_task()`
2. Define the task class and call `register_task_class()`
3. Both calls should be at module level (executed on import)

---

## Priority 4: Known Limitations Documentation

### 4.1 Document Mask Extraction Limitation

Update GUI message in `src/pineuro/gui/screens/study_editor.py` (line 695):
- Current: "GLM-based mask extraction coming soon"
- Keep this message but also document in `docs/CUSTOM_LOCALIZERS.md`

### 4.2 Document Network Count Assumptions

BallTask assumes exactly 2 networks. Single-network or 3+ network tasks require custom implementations. Document this clearly.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/pineuro/gui/components/pipeline_runner.py` | Use `create_task()` factory instead of hardcoded BallTask |
| `src/pineuro/tasks/ball.py` | Align constructor with factory interface (accept kwargs, handle mask_names) |
| `src/pineuro/tasks/base.py` | Accept `session_state`, `output_dir` in constructor for factory compatibility |
| `src/pineuro/tasks/factory.py` | Pass `mask_names` through config dict |
| `src/pineuro/tasks/__init__.py` | Add auto-discovery of task modules |
| `docs/CUSTOM_TASKS.md` | **New file** - task development guide |
| `docs/CUSTOM_LOCALIZERS.md` | **New file** - localizer limitations and future plans |

---

## Verification Plan

1. **Test factory integration**: Create minimal test task, verify it appears in GUI dropdown and runs correctly
2. **Test auto-discovery**: Drop a new `.py` file in `tasks/`, restart GUI, confirm task appears
3. **Test task parameters**: Ensure all `TaskMetadata.parameters` render correctly in study editor
4. **Test mirror callback**: Verify GUI mirror displays custom task shapes
5. **Documentation test**: Follow guide to create new task from scratch, confirm it works end-to-end
