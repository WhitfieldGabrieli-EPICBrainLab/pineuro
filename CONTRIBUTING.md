# Contributing to pineuro

Thank you for your interest in contributing to pineuro! This guide covers how to set up, run, and submit changes, plus the conventions and extension points you'll need.

## Getting Started

Clone the repo and install it in editable mode with the dev tools. Use **Python 3.10 or 3.11** (`requires-python = ">=3.10,<3.12"`):

```bash
git clone https://github.com/marlvan/pineuro.git
cd pineuro
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

All dependencies are pure pip — no external tools (FSL, FreeSurfer, etc.) are required. The `[dev]` extra adds the toolchain used below: `ruff`, `mypy`, `pytest`, `build`, and `types-PyYAML`.

### Running the app

Two equivalent ways to launch the GUI:

```bash
pineuro-gui                  # console script (created by the install)
python -m pineuro.gui.app    # run the module directly — handy from an IDE/debugger
```

Because the install is **editable**, your code changes are picked up on the next run — only re-run `pip install` when dependencies change.

## Making a Change

### Branching

Always work on a branch off `main`; never commit to `main` directly. There's no enforced naming convention — `yourname/topic`, `fix/...`, or `feat/...` all work.

**Lab collaborators (push access to `marlvan/pineuro`)** — clone once (see [Getting Started](#getting-started)), then branch off an up-to-date `main`:

```bash
git switch main && git pull                 # sync the latest main
git switch -c yourname/short-description     # new branch
# ...commit your work...
git push -u origin yourname/short-description
```

**External contributors (no push access)** — work from a fork:

```bash
# Fork marlvan/pineuro on GitHub, then:
git clone https://github.com/<your-username>/pineuro.git
cd pineuro
git switch -c short-description
# ...commit your work...
git push origin short-description
```

### Before you push

Run the five checks CI gates on — a green run here means a green run on GitHub:

```bash
ruff check .              # lint (whole repo — matches CI)
ruff format . --check     # format check  (run `ruff format .` to apply fixes)
mypy src/pineuro/         # type check
pytest                    # unit tests (CI runs these on Python 3.10 & 3.11)
pytest -m validation      # scientific-correctness gate (~1 min, real ANTs/nilearn)
```

In Claude Code, the `/check` shortcut runs all five at once. CI additionally builds the package (`python -m build`); the source of truth for every job is `.github/workflows/ci.yml`.

### Opening a pull request

Push your branch, then open a **pull request (PR)** on GitHub — a request to merge your branch into `main`, where it's reviewed and checked by CI before merging. (Lab collaborators open the PR within `marlvan/pineuro`; external contributors open it from their fork → `marlvan/pineuro:main`.) CI runs on every push to the PR branch.

Before requesting review:

1. Keep the PR focused — one feature or fix per PR.
2. All five checks above pass.
3. Describe **what** changed and **why**.
4. Add a `CHANGELOG.md` entry under `[Unreleased]` (`Added`, `Changed`, `Fixed`, or `Removed`).
5. Confirm it works cross-platform (Windows, macOS, Linux).

## Code Conventions

Linting and formatting are handled by **Ruff** (config in `pyproject.toml`): line length 120, target Python 3.10, rule families E/W/F (pycodestyle/pyflakes), I (isort), UP (pyupgrade), B (bugbear), SIM (simplify). The `ruff` and `mypy` commands are in [Before you push](#before-you-push).

- **Type annotations** on all public functions.
- **Docstrings** in NumPy style (Summary / Parameters / Returns / Raises).
- **`pathlib.Path`** for all file paths — never string concatenation or hardcoded `/` separators (this keeps Windows/macOS/Linux working).
- **`@dataclass`** for configuration and metadata objects.
- **`overwrite=False`** as the default for any function that writes output files.
- External commands (subprocess calls) route through `run_command()` in `utils.py`.
- New dependencies must be pip-installable on all three platforms.

## Testing

pineuro uses **pytest**; tests live in `tests/`, one `test_<module>.py` per source module.

```bash
pytest                           # unit tests only (default, CI-safe)
pytest tests/test_feedback.py    # a single file
pytest -k "glm"                  # tests matching a keyword
pytest -m validation             # validation suite (scientific correctness)
```

Markers: `@pytest.mark.slow` (tests >10s) and `@pytest.mark.validation` (scientific-correctness suite).

When writing tests:

- Import from submodules directly: `from pineuro.feedback import IncrementalGLM`.
- Use the conftest fixtures for synthetic data (e.g. `synthetic_nifti_4d`, `sample_mask`).
- Mock heavy dependencies (ANTsPy, nilearn ICA) in unit tests; exercise them for real in `tests/validation/`.
- Use `assert` freely (ruff `S101` is ignored in test files).

**Validation suite** — `tests/validation/` runs synthetic data with known ground truth through the *real* pipeline (real ANTsPy/nilearn, no mocking) and asserts pineuro recovers it within documented tolerances. Opt-in locally (`pytest -m validation`); runs on every push in CI as a hard correctness gate. See `tests/validation/README.md`. (Separately, `analysis/` holds standalone scripts comparing pineuro against reference tools like fMRIPrep/MELODIC/MURFI — these are not part of pytest.)

## Extending pineuro

### Adding a task

pineuro has a decorator-based plugin system for PsychoPy tasks. See `src/pineuro/tasks/README.md` for the full framework guide; the steps below are the essentials.

1. **Create the module** at `src/pineuro/tasks/my_task.py`.

2. **Define metadata** describing the task's configurable parameters:

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
    n_networks=1,           # number of brain-network masks the task needs
    roles={"feedback"},     # "feedback", "localizer", or both
)
```

3. **Implement the class** with the `@pineuro_task` decorator — its `task_id` must match the metadata:

```python
from pineuro.tasks.base import BaseTask

@pineuro_task(MY_TASK_METADATA)
class MyTask(BaseTask):
    task_id = "my_task"

    def __init__(self, coordinator, speed: float = 1.0, **kwargs):
        super().__init__(coordinator, **kwargs)
        self.speed = speed

    def setup(self):
        """Create the PsychoPy window and stimuli."""
        self.create_window(monitor_position="right")

    def update(self, data: dict):
        """Called every frame (~60 Hz). Update the display from activation data.

        data keys: 'activations', 'volume', 'time', 'is_baseline'
        """
        activation = data["activations"][self.mask_name]
        self.win.flip()
```

For localizer tasks (stimulus presentation without real-time feedback), subclass `BaseLocalizerTask` and implement `setup()` + `run_paradigm()` instead.

4. **Make it discoverable.** Built-in tasks are imported by the list in `_load_builtin_tasks()` in `src/pineuro/tasks/__init__.py` (lazy-loaded so heavy imports stay off app startup). Add your module there:

```python
def _load_builtin_tasks():
    from . import (  # noqa: F401
        ball,
        gauge,
        my_task,   # <- add this
        # ...existing tasks...
    )
```

Importing the module runs the `@pineuro_task` decorator, which registers both the metadata and the class — the GUI then discovers the task and its parameters automatically, with no GUI code changes. Use `ball.py` or `gauge.py` as templates.

**External plugins** — tasks can also live outside the source tree; load them at runtime with `discover_tasks()`:

```python
from pineuro.tasks import discover_tasks
discover_tasks([Path("/path/to/my/tasks")])
```

### Adding an atlas

Atlases are declared in `ATLAS_SOURCES` in `src/pineuro/atlas_registry.py` — a lightweight metadata dict the GUI reads (for its dropdown) without importing heavy neuroimaging libraries.

1. **Add an entry** to `ATLAS_SOURCES`:

```python
ATLAS_SOURCES: dict[str, dict[str, Any]] = {
    # ... existing atlases ...
    "my_network_atlas": {
        "display_name": "My Network Atlas",
        "type": "network",      # "network", "anatomical", or "tissue"
        "n_networks": 10,       # network atlases only
        # "n_rois": 200,        # add for Schaefer-style parcellations
    },
    "my_anat_atlas": {
        "display_name": "My Anatomical Atlas",
        "type": "anatomical",
    },
}
```

2. **Wire up fetching** in `src/pineuro/mask_extraction.py`. Dispatch is by explicit `if/elif` on the atlas key (not auto-derived from the metadata), so add a branch for your atlas in:
   - `get_atlas_labels()` — returns the region/network labels (all atlases).
   - `_register_atlas_to_subject()` — network atlases (resting-state / ICA path).
   - `_fetch_atlas_and_resolve_regions()` — anatomical atlases (task / GLM thresholding path).

   Follow the existing Yeo / Schaefer / Harvard-Oxford branches as templates.

The GUI picks up the new atlas in its dropdown automatically from `ATLAS_SOURCES`, but the dispatch branches above are required — without them, mask extraction will fail for the new atlas.

## Versioning

pineuro follows [Semantic Versioning](https://semver.org/). While at **0.x** (pre-1.0), the API is under active development and may change between minor versions without deprecation warnings — standard semver for initial development.

- **0.x.y**: breaking changes bump the minor (x); bug fixes bump the patch (y).
- **1.0.0+**: breaking changes require a major bump, with deprecation periods for removed API.

`version` in `pyproject.toml` is the single source of truth for the package — `pineuro.__version__` is read from the installed metadata at runtime, so there's nothing to hand-edit there. The GUI app carries its own independent `__version__` in `src/pineuro/gui/__init__.py`; bump that only when the GUI app version changes.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
