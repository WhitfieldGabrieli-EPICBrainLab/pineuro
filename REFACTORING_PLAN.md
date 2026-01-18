# pyrtnf Package Review and Refactoring Plan

## Overview

A comprehensive review of the pyrtnf package identified several areas for improvement across cross-platform compatibility, code quality, and task flexibility. This plan organizes changes into small, testable increments with GUI testing between changes.

---

## Key Findings

### Current State
- **~12.5K lines** across core modules, GUI fully implemented
- Pipeline works end-to-end for ball task with dual-network feedback
- Good architecture with clear separation (preprocessing → mask extraction → feedback)

### Issues Identified

1. **Missing config module** - `tests/test.py` imports `pyrtnf.config` for display settings but the module doesn't exist
2. **Cross-platform issues** - PsychoPy/Xlib workarounds are Linux-specific; WSL2 display issues undocumented
3. **Code duplication** - Overwrite checks, progress callbacks, validation patterns repeated
4. **Task inflexibility** - BallTask hardcodes 2 networks; HRF functions are task-specific
5. **Large functions** - `preprocessing()` is 800+ lines, should be decomposed

---

## Implementation Plan

### Phase 1: Critical Missing Pieces (Changes 1-2)

#### Change 1.1: Create config.py Module
**Files**: Create `src/pyrtnf/config.py`, modify `src/pyrtnf/__init__.py`

Create centralized display and platform configuration:
```python
# Display settings (configurable before importing other modules)
QC_BROWSER_SCREEN: int = 0       # Screen for QC reports
PSYCHOPY_SCREEN: int = 1         # Screen for PsychoPy tasks

# Platform detection helpers
def is_wsl() -> bool
def is_wayland() -> bool
def get_platform() -> str  # Returns 'darwin', 'linux', 'wsl', 'windows'
```

**Test**: `python -c "import pyrtnf.config; print(pyrtnf.config.PSYCHOPY_SCREEN)"`

---

#### Change 1.2: Document Platform Requirements
**Files**: Create `docs/PLATFORM_REQUIREMENTS.md`

Document:
- FSL requirement (Linux/macOS only - no native Windows support)
- ANTsPy installation notes per platform
- WSL2 display setup and known issues
- Multi-monitor configuration guide
- Flet/XWayland compatibility notes

**Test**: Review document for completeness

---

### Phase 2: Cross-Platform Improvements (Changes 3-6)

#### Change 2.1: Integrate config.py into QC Browser
**Files**: Modify `src/pyrtnf/qc.py`

- Use `config.QC_BROWSER_SCREEN` instead of hardcoded screen
- Add Wayland detection before Xlib operations
- Graceful fallback if xrandr/wmctrl unavailable

**Test**: Run `python tests/test.py --setup`, verify QC opens on correct screen

---

#### Change 2.2: Integrate config.py into PsychoPy
**Files**: Modify `src/pyrtnf/tasks/ball.py`

- Use `config.PSYCHOPY_SCREEN` instead of hardcoded `screen=1`
- Check platform before using Xlib workaround
- Skip Xlib on WSL2/Wayland (log info message instead)

**Test**: Run `python tests/test.py`, verify PsychoPy opens on correct screen

---

#### Change 2.3: Add Platform Compatibility Checks
**Files**: Modify `src/pyrtnf/utils.py`

Add `check_platform_requirements()` function that returns warnings for:
- Missing FSL commands
- Missing dcm2niix
- WSL2 environment (multi-monitor may not work)
- Wayland session (some X11 features unavailable)

**Test**: `python -c "from pyrtnf.utils import check_platform_requirements; print(check_platform_requirements())"`

---

#### Change 2.4: Add Graceful Xlib Fallback
**Files**: Modify `src/pyrtnf/tasks/ball.py`

Update `_move_window_to_secondary()` to:
- Skip on WSL2 or Wayland (detected via config)
- Log info message instead of attempting Xlib operations
- Handle missing Xlib gracefully

**Test**: Run on WSL2, verify no Xlib errors

---

### Phase 3: Code Streamlining (Changes 7-12)

#### Change 3.1: Extract Overwrite-Check Utility
**Files**: Modify `src/pyrtnf/utils.py`

Add `outputs_exist(outputs, overwrite)` function to replace repeated pattern:
```python
if not overwrite and output_path.exists():
    logger.info("Skipping...")
    return
```

**Test**: Unit test the function

---

#### Change 3.2: Extract Progress Callback Pattern
**Files**: Modify `src/pyrtnf/utils.py`

Add `ProgressEmitter` class:
```python
class ProgressEmitter:
    def __init__(self, callback, logger):
        ...
    def emit(self, step: str, detail: str = ""):
        # Log and call callback
```

**Test**: Unit test, then update one function to use it

---

#### Change 3.3: Move HRF Functions to utils.py
**Files**: Modify `src/pyrtnf/utils.py`, `src/pyrtnf/tasks/ball.py`, `src/pyrtnf/__init__.py`

Move from `tasks/ball.py` to `utils.py`:
- `compute_hrf(tr, duration)` - SPM-style canonical HRF
- `compute_boxcar_task_regressor(n_baseline, n_volumes, tr)`

Export from package `__init__.py`.

**Test**: `python -c "from pyrtnf import compute_hrf; print(compute_hrf(1.2).shape)"`

---

#### Change 3.4: Decompose preprocessing() Function
**Files**: Modify `src/pyrtnf/preprocessing.py`

Extract into 4 helper functions:
- `_step_dicom_to_nifti(...)` - DICOM conversion
- `_step_motion_correction(...)` - mcflirt
- `_step_brain_extraction(...)` - bet + mean computation
- `_step_coregistration(...)` - multi-run alignment

Main `preprocessing()` calls these in sequence.

**Test**: `python tests/test.py --setup` produces identical output

---

#### Change 3.5: Create TaskDesign Abstraction
**Files**: Create `src/pyrtnf/tasks/designs.py`

Abstract base class for task designs:
```python
class TaskDesign(ABC):
    @abstractmethod
    def compute_regressors(self, n_volumes, tr) -> NDArray

class BlockDesign(TaskDesign):
    def __init__(self, n_baseline): ...

class NuisanceOnlyDesign(TaskDesign):
    # For resting-state (no task regressors)
```

**Test**: `python -c "from pyrtnf.tasks.designs import BlockDesign; print(BlockDesign(25))"`

---

#### Change 3.6: Update FeedbackEstimator for TaskDesign
**Files**: Modify `src/pyrtnf/feedback.py`

Accept `TaskDesign` objects in addition to raw arrays:
```python
def __init__(self, ..., task_regressors: NDArray | TaskDesign = None):
    if hasattr(task_regressors, 'compute_regressors'):
        task_regressors = task_regressors.compute_regressors(n_volumes, tr)
```

Backward compatible with existing code.

**Test**: Verify both old (array) and new (TaskDesign) interfaces work

---

### Phase 4: Task Flexibility (Changes 13-16)

#### Change 4.1: Support Variable Number of Networks
**Files**: Modify `src/pyrtnf/tasks/ball.py`

Change from hardcoded 2 networks to configurable 1-N:
- Add `n_networks` parameter (default=2)
- Dynamic circle positioning based on network count
- Update hit counters, circle radii to be lists of length N

**Test**: Create test with 1 network, verify it works

---

#### Change 4.2: Create Task Factory
**Files**: Create `src/pyrtnf/tasks/factory.py`

```python
def create_task(task_id, coordinator, config, session_state=None) -> BaseTask:
    """Create task from registry by ID."""
```

Uses task registry metadata to validate config.

**Test**: `python -c "from pyrtnf.tasks.factory import create_task"`

---

#### Change 4.3: Register Ball Task with Full Metadata
**Files**: Modify `src/pyrtnf/tasks/__init__.py`

Register ball task parameters:
- initial_scale_factor, n_networks, fullscreen
- min_hits_threshold, max_hits_threshold, scale_adjust_factor

**Test**: `python -c "from pyrtnf.tasks import get_task_metadata; print(get_task_metadata('ball'))"`

---

#### Change 4.4: Update GUI to Use Task Registry
**Files**: Modify `src/pyrtnf/gui/screens/study_editor.py`

Generate task parameter UI from registry metadata instead of hardcoding.

**Test**: Launch GUI, verify task parameters appear correctly

---

## Verification Plan

After each change:
1. Run `python -c "import pyrtnf"` - no import errors
2. Run `python tests/test.py --setup` - setup phase works
3. Launch GUI: `pyrtnf-gui` - GUI opens and functions

After Phase 2 complete:
- Test on WSL2 environment
- Test QC browser positioning

After Phase 4 complete:
- Test ball task with 1 network
- Test ball task with 3 networks
- Verify GUI shows correct task parameters

---

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `src/pyrtnf/config.py` | Display/platform configuration |
| `src/pyrtnf/tasks/designs.py` | Task design abstractions |
| `src/pyrtnf/tasks/factory.py` | Task creation from registry |
| `docs/PLATFORM_REQUIREMENTS.md` | Platform documentation |

### Modified Files
| File | Changes |
|------|---------|
| `src/pyrtnf/__init__.py` | Export config, HRF functions |
| `src/pyrtnf/utils.py` | Add utilities, platform checks, HRF |
| `src/pyrtnf/qc.py` | Use config for screen, Wayland detection |
| `src/pyrtnf/preprocessing.py` | Decompose into step functions |
| `src/pyrtnf/feedback.py` | Accept TaskDesign objects |
| `src/pyrtnf/tasks/ball.py` | Config integration, variable networks |
| `src/pyrtnf/tasks/__init__.py` | Register ball task metadata |
| `src/pyrtnf/gui/screens/study_editor.py` | Generate UI from registry |

---

## Incremental Testing Workflow

For each change:
1. Make the code change
2. Run import test
3. Run `python tests/test.py --setup`
4. Launch GUI and click through workflow
5. Commit if working
6. Proceed to next change
