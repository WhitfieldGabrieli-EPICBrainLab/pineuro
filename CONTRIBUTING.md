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

Key points:

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
4. Run `pytest` and ensure all tests pass.
5. Include a clear description of **what** changed and **why**.
6. Add a changelog entry to the `[Unreleased]` section of `CHANGELOG.md` under the appropriate heading (`Added`, `Changed`, `Fixed`, `Removed`).
7. Ensure cross-platform compatibility (Windows, macOS, Linux).

All PRs are checked by CI (GitHub Actions). The following must pass before merge:

- **Lint**: `ruff check .` + `ruff format . --check`
- **Type check**: `mypy src/pineuro/`
- **Tests**: `pytest` on Ubuntu/Windows/macOS × Python 3.10/3.11
- **Validation**: `pytest -m validation` — scientific-correctness gate (synthetic data, real ANTs/nilearn)
- **Build**: package builds successfully

## Versioning

pineuro follows [Semantic Versioning](https://semver.org/). While at **0.x** (pre-1.0), the API is under active development and may change between minor versions without deprecation warnings. This is standard semver behavior for initial development.

- **0.x.y**: Breaking changes bump the minor version (x); bug fixes bump the patch version (y).
- **1.0.0 and beyond**: Breaking changes require a major version bump, with deprecation periods for removed API.

Version is tracked in three places — keep them in sync when bumping:

- `pyproject.toml` (`version = "..."`)
- `src/pineuro/__init__.py` (`__version__ = "..."`)
- `src/pineuro/gui/__init__.py` (`__version__ = "..."`)

## Testing

pineuro uses **pytest**. Tests live in `tests/` with this structure:

```
tests/
├── conftest.py                       # Shared fixtures (synthetic NIfTI, masks, etc.)
├── test_import.py                    # Smoke test
├── test_config.py                    # Platform detection
├── test_utils.py                     # Utilities, timing, file I/O, masking
├── test_feedback.py                  # IncrementalGLM, FeedbackEstimator
├── test_preprocessing.py             # Pipeline steps (concat, smoothing, filtering)
├── test_mask_extraction.py           # ICA, GLM, atlas workflows
├── test_session.py                   # Config dataclasses, session management
├── test_session_orchestration.py     # Multi-run/multi-session orchestration
├── test_sham.py                      # Randomization, replay providers
├── test_simulator.py                 # Volume simulator
├── test_tasks.py                     # Plugin registry, designs, session state
├── test_newer_tasks.py               # Gauge, RNT task variants
├── test_rnt_experience_sampling.py   # RNT experience sampling task
├── test_qc.py                        # QC reports and plotting
├── test_atlas_registry.py            # Atlas metadata registry
├── test_collection_qc.py             # Real-time collection QC
├── test_scanner_setup.py             # Scanner connectivity testing
├── test_script_generator.py          # Script generation
├── test_gui_config_manager.py        # GUI config persistence
├── test_gui_config_snapshot.py       # GUI config snapshots
├── test_gui_csv_poller.py            # GUI CSV polling component
├── test_gui_models.py                # GUI data models
└── test_gui_utils.py                 # GUI utilities
```

### Running tests

```bash
pytest                           # Unit tests only (default, CI-safe)
pytest -v                        # Verbose output
pytest tests/test_feedback.py    # Run a specific file
pytest -k "test_glm"            # Run tests matching a keyword
pytest -m validation             # Validation suite (scientific correctness)
```

### Markers

- `@pytest.mark.slow` — tests taking >10 seconds
- `@pytest.mark.validation` — scientific-correctness suite (see below)

### Writing tests

- Place unit tests in `tests/test_<module>.py`
- Import from submodules directly: `from pineuro.feedback import IncrementalGLM`
- Use conftest fixtures for synthetic NIfTI data (e.g., `synthetic_nifti_4d`, `sample_mask`)
- Mock heavy dependencies (ANTsPy, nilearn ICA) for unit tests; test them for real in `tests/validation/`
- Use `assert` freely (ruff S101 is ignored in test files)

### Validation suite

`tests/validation/` is a separate, correctness-focused suite: it runs synthetic data with known ground truth through the *real* pipeline (real ANTsPy/nilearn, no mocking) and asserts pineuro recovers it within documented tolerances. Opt-in locally (`pytest -m validation`); runs on every push in CI as a hard correctness gate. See `tests/validation/README.md` for details.

### Validation comparison scripts

The `analysis/` directory contains standalone validation scripts for comparing pineuro output against reference tools (fMRIPrep, MELODIC, MURFI). These are separate from pytest.

## Platform Notes

pineuro supports Windows, macOS, and Linux. When contributing:

- Use `pathlib.Path` for all file paths.
- Never assume Unix-only conventions (e.g., hardcoded `/` separators).
- All new dependencies must be pip-installable on all three platforms.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
