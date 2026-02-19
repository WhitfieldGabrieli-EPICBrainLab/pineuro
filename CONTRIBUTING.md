# Contributing to pineuro

Thank you for your interest in contributing to pineuro! This guide covers development setup, code conventions, and how to extend the framework.

## Development Setup

```bash
# Clone and install in editable mode with dev tools
git clone https://github.com/marlvan/pineuro.git
cd pineuro
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -e ".[dev]"
```

All dependencies are pure pip — no external tools (FSL, FreeSurfer, etc.) are required. Verify the install by running:

```bash
pineuro-gui
```

## Code Style & Linting

pineuro uses **Ruff** for linting and formatting. Configuration lives in `pyproject.toml`.

```bash
ruff check .         # Lint
ruff check . --fix   # Lint with auto-fix
ruff format .        # Format
```

Key settings:

| Setting | Value |
|---------|-------|
| Line length | 120 |
| Target version | Python 3.10+ |
| Rules | E/W/F (pycodestyle/pyflakes), I (isort), UP (pyupgrade), B (bugbear), SIM (simplify) |

### Conventions

- **Type annotations** on all public functions. The codebase uses `Union`, `Optional`, `Path`, etc. from `typing`.
- **Docstrings** in NumPy style (Summary / Parameters / Returns / Raises).
- **`pathlib.Path`** for all file paths — never string concatenation.
- **`@dataclass`** for configuration and metadata objects.
- **`overwrite=False`** as the default for any function that writes output files.
- External commands (subprocess calls) route through `run_command()` in `utils.py`.

## Project Architecture

See `CLAUDE.md` for a quick reference of the project layout. Key points:

- **Numbered output directories**: `01_nifti/`, `02_motion_correction/`, `03_masks/`, etc.
- **Threading model**: Coordinator thread (background processing) + Task thread (PsychoPy in main thread, required by OpenGL).
- **Modules**: `preprocessing.py` (pipeline), `mask_extraction.py` (ICA/GLM masks), `feedback.py` (real-time GLM), `session.py` (orchestration), `tasks/` (PsychoPy framework), `gui/` (Flet app).

## Adding a New Task

pineuro has a decorator-based plugin system for PsychoPy tasks. See `src/pineuro/tasks/README.md` for the full task framework guide.

### Steps

1. **Create your module** at `src/pineuro/tasks/my_task.py`.

2. **Define metadata** describing your task's configurable parameters:

```python
from pineuro.tasks.task_registry import TaskMetadata, TaskParameter, pineuro_task

MY_TASK_METADATA = TaskMetadata(
    task_id="my_task",
    display_name="My Task",
    description="Brief description of the task",
    parameters=[
        TaskParameter(
            key="speed",
            label="Animation Speed",
            type="float",
            default=1.0,
            min_value=0.1,
            max_value=10.0,
            hint="Controls how fast the stimulus moves",
        ),
    ],
    n_networks=1,           # Number of brain network masks required
    roles={"feedback"},     # "feedback", "localizer", or both
)
```

3. **Implement the task class** using the `@pineuro_task` decorator:

```python
from pineuro.tasks.base import BaseTask

@pineuro_task(MY_TASK_METADATA)
class MyTask(BaseTask):
    task_id = "my_task"  # Must match metadata

    def __init__(self, coordinator, speed: float = 1.0, **kwargs):
        super().__init__(coordinator, **kwargs)
        self.speed = speed

    def setup(self):
        """Create PsychoPy window and stimuli."""
        self.create_window(monitor_position="right")
        # Initialize visual stimuli here

    def update(self, data: dict):
        """Called every frame (~60 Hz). Update display from activation data.

        data keys: 'activation', 'activations', 'volume', 'time', 'is_baseline'
        """
        activation = data["activation"]
        # Draw stimuli based on activation
        self.win.flip()
```

For localizer tasks (stimulus presentation without real-time feedback), subclass `BaseLocalizerTask` and implement `setup()` + `run_paradigm()` instead.

4. **Register** by adding an import to `src/pineuro/tasks/__init__.py`:

```python
from .my_task import MyTask  # Registers "my_task"
```

The `@pineuro_task` decorator handles all registration automatically. The GUI discovers the task's parameters from the metadata — no GUI code changes needed.

### External plugins

Tasks can also live outside the pineuro source tree. Use `discover_tasks()` to load them at runtime:

```python
from pineuro.tasks import discover_tasks
discover_tasks([Path("/path/to/my/tasks")])
```

## Adding a New Atlas

Atlases are registered in a lightweight metadata dict (`ATLAS_SOURCES`) that the GUI reads without importing heavy neuroimaging libraries. The actual fetching happens in `mask_extraction.py`.

### Steps

1. **Add metadata** to `ATLAS_SOURCES` in `src/pineuro/atlas_registry.py`:

```python
ATLAS_SOURCES: dict[str, dict[str, Any]] = {
    # ... existing atlases ...

    # Network atlas (for ICA-based mask extraction)
    "my_network_atlas": {
        "display_name": "My Network Atlas",
        "type": "network",
        "n_networks": 10,
    },

    # Anatomical atlas (for region-based mask extraction)
    "my_anat_atlas": {
        "display_name": "My Anatomical Atlas",
        "type": "anatomical",
    },
}
```

2. **Implement fetching** in `src/pineuro/mask_extraction.py`:
   - Add a branch to `get_atlas_labels()` to return region/network labels.
   - Update `mask_extraction_rest()` (network atlases) or `import_manual_masks()` (anatomical atlases) to handle the new atlas.

3. The GUI picks up the new atlas automatically from `ATLAS_SOURCES` — no GUI code changes needed.

## Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Keep PRs focused — one feature or fix per PR.
3. Run `ruff check .` and `ruff format .` before submitting.
4. Include a clear description of **what** changed and **why**.
5. Ensure cross-platform compatibility (Windows, macOS, Linux).

## Testing

There is no formal pytest suite yet (this is planned — see item 1.7 in `FEATURE_GAP_ANALYSIS.md`). In the meantime:

- **Manual testing**: Launch the GUI (`pineuro-gui`) and verify your changes work end-to-end.
- **Validation scripts**: For pipeline changes, run the comparison scripts in `tests/` against known reference outputs.

## Platform Notes

pineuro supports Windows, macOS, and Linux. When contributing:

- Use `pathlib.Path` for all file paths.
- Never assume Unix-only conventions (e.g., hardcoded `/` separators).
- All new dependencies must be pip-installable on all three platforms.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
