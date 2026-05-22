"""Task registration system for pluggable neurofeedback tasks.

This module provides the infrastructure for adding new task types to pineuro
without modifying the GUI code. Tasks register themselves by defining metadata
that describes their configurable parameters.

Example:
    # Define task metadata
    ball_metadata = TaskMetadata(
        task_id="ball",
        display_name="Ball Task",
        description="Dual-network neurofeedback with ball control",
        parameters=[
            TaskParameter(
                key="initial_scale_factor",
                label="Initial Scale Factor",
                type="float",
                default=10.0,
                min_value=1.0,
                max_value=50.0,
                hint="Starting difficulty (larger = easier)",
            ),
        ],
    )

    # Register task
    register_task(ball_metadata)
"""

import fnmatch
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Task registry (task_id -> metadata)
_TASK_REGISTRY: dict[str, "TaskMetadata"] = {}

# Lazy loaders: callables that import task modules on first access
_lazy_loaders: list[Callable[[], None]] = []


def add_lazy_loader(loader: Callable[[], None]) -> None:
    """Register a callable to import task modules on first access."""
    _lazy_loaders.append(loader)


def _ensure_tasks_loaded() -> None:
    """Import task modules if not yet loaded."""
    while _lazy_loaders:
        _lazy_loaders.pop(0)()


class TaskConfigError(ValueError):
    """Raised when task configuration validation fails.

    Collects all validation errors found in the config so callers see the
    complete list of problems rather than just the first failure.

    Attributes
    ----------
    task_id : str
        The task identifier whose config failed validation.
    errors : list[str]
        All validation error messages found in the config.
    """

    def __init__(self, task_id: str, errors: list[str]) -> None:
        self.task_id = task_id
        self.errors = list(errors)
        detail = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Task '{task_id}' config has {len(errors)} error(s):\n{detail}")


@dataclass
class NetworkRole:
    """Description of a network slot required by a task.

    Allows tasks to declare what each of their required network masks
    represents, so the GUI can show appropriately labeled dropdowns
    instead of hardcoding "top_network" / "bottom_network".

    Attributes:
        key: Identifier used in network_assignment config (e.g., "top_network")
        label: Display label in GUI (e.g., "Top Network (upward)")
        description: Tooltip/help text explaining this network's role
    """

    key: str
    label: str
    description: str = ""


@dataclass
class TaskParameter:
    """Definition of a single task parameter for GUI configuration.

    Attributes:
        key: Configuration key (e.g., "initial_scale_factor")
        label: Display label in GUI (e.g., "Initial Scale Factor")
        type: Parameter type (determines GUI control type)
        default: Default value
        hint: Help text shown below field (optional)
        min_value: Minimum value for numeric types (optional)
        max_value: Maximum value for numeric types (optional)
        choices: Allowed values for str type (optional). If set, GUI renders a
            dropdown instead of a text field.
        validator: Custom validation function (optional)
            Should take value and raise ValueError if invalid
    """

    key: str
    label: str
    type: Literal["float", "int", "bool", "str", "dir", "list[str]"]
    default: Any
    hint: str = ""
    min_value: float | None = None
    max_value: float | None = None
    choices: list[str] | None = None
    validator: Callable[[Any], None] | None = None

    def validate(self, value: Any) -> None:
        """Validate a value for this parameter.

        Args:
            value: Value to validate

        Raises:
            ValueError: If validation fails
        """
        # Type-specific validation
        if self.type in ["int", "float"]:
            try:
                num = float(value) if self.type == "float" else int(value)
            except (ValueError, TypeError):
                raise ValueError(f"{self.label} must be a number") from None

            if self.min_value is not None and num < self.min_value:
                raise ValueError(f"{self.label} must be >= {self.min_value}")

            if self.max_value is not None and num > self.max_value:
                raise ValueError(f"{self.label} must be <= {self.max_value}")

        # Choices validation (for str type with constrained values)
        if self.choices is not None and value not in self.choices:
            allowed = ", ".join(repr(c) for c in self.choices)
            raise ValueError(f"{self.label} must be one of {allowed}")

        # Custom validation
        if self.validator is not None:
            self.validator(value)


@dataclass
class SessionStateField:
    """Definition of a task-specific persistent state field.

    Session state fields are persisted across runs and can be used to track
    cumulative statistics, adaptive difficulty parameters, or any other
    state that should survive between task runs.

    Attributes:
        key: Field name used in session state dictionary (e.g., "hit_counters")
        type: Data type for validation and serialization
        default: Default value when session state is first created
        description: Documentation for this field (optional)
    """

    key: str
    type: Literal["float", "int", "bool", "list[float]", "list[int]"]
    default: Any
    description: str = ""


@dataclass
class PostRunQuestion:
    """Definition of a post-run slider question.

    Questions are displayed after each run completes. They can be conditional
    based on run type (feedback vs transfer) or run position (last feedback only).

    Attributes:
        text: Question text displayed to participant
        left_label: Label at left end of slider (e.g., "Never", "False")
        right_label: Label at right end of slider (e.g., "Always", "True")
        condition: When to show this question:
            - "always": Show after every run
            - "feedback_only": Show only after feedback runs (not transfer)
            - "last_feedback_only": Show only after the final feedback run
    """

    text: str
    left_label: str = "Never"
    right_label: str = "Always"
    condition: Literal["always", "feedback_only", "last_feedback_only"] = "always"


@dataclass(frozen=True)
class LiveMetric:
    """Declarative live-monitoring metric, polled from the last row of psychopylog.tsv.

    Attributes:
        key: Stable identifier (e.g., "hits", "score"). Used as the widget key.
        label: UI prefix (e.g., "Hits"). Rendered before the value(s).
        column_pattern: Either a literal psychopylog column name (e.g., "score")
            or an fnmatch glob (e.g., "*_hits") matching multiple columns.
        value_format: Per-cell format string. {value} is the parsed cell
            (int/float/str). Example: "{value:.1f}".
        item_format: Used only when column_pattern matches more than one column.
            {name} is the column suffix (after stripping name_strip_suffix);
            {value} is the formatted cell. Example: "{name}: {value}".
        join_with: Separator joining multiple items (e.g., ", ").
        name_strip_suffix: Suffix removed from matched column names to derive
            {name} (e.g., "_hits" so "ContA_hits" becomes "ContA").
        remap_blind_names: When True, the GUI applies its blind-mode display
            name map to {name} (e.g., "ContA" -> "Network A").
    """

    key: str
    label: str
    column_pattern: str
    value_format: str = "{value}"
    item_format: str = "{name}: {value}"
    join_with: str = ", "
    name_strip_suffix: str = ""
    remap_blind_names: bool = False


def read_live_metrics(log_path: Path, metrics: list[LiveMetric]) -> dict[str, dict[str, str]]:
    """Read declared live metrics from the last row of a psychopylog.tsv.

    For each metric, every column matching its `column_pattern` (literal name
    or fnmatch glob) is parsed from the most recent row, formatted via
    `value_format`, and keyed by the column name with `name_strip_suffix`
    removed. Callers (live monitor, post-run summary, QC report) iterate the
    returned mapping and apply task-agnostic rendering rules.

    Args:
        log_path: Path to the psychopylog.tsv file.
        metrics: Declarations from `TaskMetadata.live_metrics`.

    Returns:
        `{metric.key: {item_name: formatted_value, ...}}`. Metrics with no
        matching columns are omitted; an unreadable, missing, or empty file
        returns `{}`.
    """
    if not metrics or not log_path.exists():
        return {}
    try:
        with open(log_path) as f:
            lines = f.readlines()
        if len(lines) < 2:  # Need header + at least one data row
            return {}
        header = lines[0].rstrip("\n").split("\t")
        last_row = lines[-1].rstrip("\n").split("\t")
        out: dict[str, dict[str, str]] = {}
        for m in metrics:
            cells: dict[str, str] = {}
            for i, col in enumerate(header):
                if i >= len(last_row):
                    continue
                if not (col == m.column_pattern or fnmatch.fnmatchcase(col, m.column_pattern)):
                    continue
                raw = last_row[i]
                value: Any
                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
                if m.name_strip_suffix and col.endswith(m.name_strip_suffix):
                    name = col[: -len(m.name_strip_suffix)]
                else:
                    name = col
                cells[name] = m.value_format.format(value=value)
            if cells:
                out[m.key] = cells
        return out
    except Exception:
        return {}


@dataclass
class TaskMetadata:
    """Metadata for a task type.

    Defines everything a task needs to integrate with the pineuro framework:
    parameters for GUI configuration, session state for persistence across runs,
    post-run questions for participant feedback, and network requirements.

    Attributes:
        task_id: Unique identifier (e.g., "ball", "thermometer")
        display_name: Human-readable name shown in GUI (e.g., "Ball Task")
        description: Brief description of the task
        parameters: List of configurable parameters for GUI
        cross_parameter_validator: Optional function to validate relationships
            between parameters (e.g., "min_hits < max_hits")
        n_networks: Number of brain networks (masks) required by this task.
            Default is 1. Ball task requires 2 (top/bottom networks).
        roles: Set of roles this task can fulfill. Valid roles are "feedback"
            (real-time neurofeedback display) and "localizer" (task localizer
            for mask extraction). Default is {"feedback"}.
        default_contrasts: Default GLM contrasts for localizer tasks. Maps
            mask name to contrast expression (e.g., {"STG_self": "self - other"}).
            Used to pre-populate the GUI; users can still edit them.
        network_roles: Descriptions of network slots required by this task.
            Allows the GUI to render labeled dropdowns for each network role
            instead of hardcoding task-specific network names.
        session_state_fields: List of fields to persist across runs.
            These are stored in session_state.json and available to the task.
        post_run_questions: List of slider questions to display after each run.
            Questions can be conditional on run type or position.
        design_type: GLM design type for the pipeline runner. Determines how
            task regressors are computed. Options:
            - "block": HRF-convolved boxcar (baseline → active). Default.
            - "nuisance_only": No task regressors (nuisance-only model).
        compute_max_run_seconds: Optional. Callable that, given the task's
            parameter dict, returns the worst-case duration of a single run
            in seconds. Used by the pipeline runner to override the
            user-configured n_volumes for variable-duration tasks, and by
            the GUI to hide the n_volumes field. Tasks with predictable,
            fixed-by-n_volumes duration leave this as None.
        live_metrics: Declarative live-monitoring metrics surfaced in the GUI
            participant workflow panel. Each entry maps psychopylog.tsv
            columns to a labeled display (e.g., "Hits: ContA: 5"). Tasks that
            do not write per-frame logs or have nothing meaningful to surface
            leave this empty -- the GUI hides the row entirely.
        localizer_params_from_scan: When True, the localizer's runtime
            parameters are derived from the scan parameters (n_volumes * tr)
            rather than read from a per-task ``localizer_task`` config section.
            Used by fixed-duration localizers such as the fixation task.
    """

    task_id: str
    display_name: str
    description: str
    parameters: list[TaskParameter]
    cross_parameter_validator: Callable[[dict[str, Any]], None] | None = None
    n_networks: int = 1
    roles: set[str] = field(default_factory=lambda: {"feedback"})
    default_contrasts: dict[str, str] = field(default_factory=dict)
    network_roles: list[NetworkRole] = field(default_factory=list)
    session_state_fields: list[SessionStateField] = field(default_factory=list)
    post_run_questions: list[PostRunQuestion] = field(default_factory=list)
    design_type: str = "block"
    compute_max_run_seconds: Callable[[dict[str, Any]], float] | None = None
    live_metrics: list[LiveMetric] = field(default_factory=list)
    localizer_params_from_scan: bool = False

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate a complete task configuration.

        Collects *all* validation errors before raising so callers see the
        full list of problems rather than just the first failure.

        Args:
            config: Task configuration dictionary

        Raises:
            TaskConfigError: If any parameter or cross-parameter validation
                fails, containing every error message found.
        """
        errors: list[str] = []

        # Validate individual parameters that are present in config
        for param in self.parameters:
            if param.key in config:
                try:
                    param.validate(config[param.key])
                except ValueError as exc:
                    errors.append(str(exc))

        # Cross-parameter validation
        if self.cross_parameter_validator is not None:
            try:
                self.cross_parameter_validator(config)
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            raise TaskConfigError(self.task_id, errors)


def register_task(metadata: TaskMetadata) -> None:
    """Register a task type.

    Args:
        metadata: TaskMetadata object describing the task
    """
    _TASK_REGISTRY[metadata.task_id] = metadata


def get_task_metadata(task_id: str) -> TaskMetadata:
    """Get metadata for a task type.

    Args:
        task_id: Task identifier (e.g., "ball")

    Returns:
        TaskMetadata object

    Raises:
        ValueError: If task_id is not registered
    """
    _ensure_tasks_loaded()
    if task_id not in _TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}")
    return _TASK_REGISTRY[task_id]


def list_available_tasks(role: str | None = None) -> dict[str, TaskMetadata]:
    """List registered tasks, optionally filtered by role.

    Args:
        role: If specified, only return tasks that have this role
            (e.g., "feedback" or "localizer"). If None, returns all tasks.

    Returns:
        Dictionary mapping task_id to TaskMetadata
    """
    _ensure_tasks_loaded()
    if role is None:
        return _TASK_REGISTRY.copy()
    return {task_id: metadata for task_id, metadata in _TASK_REGISTRY.items() if role in metadata.roles}


def pineuro_task(metadata: TaskMetadata):
    """Decorator for registering a task class with its metadata.

    Simplifies task registration by combining register_task() and
    register_task_class() into a single decorator. Also verifies that
    the class's task_id attribute matches the metadata.

    Parameters
    ----------
    metadata : TaskMetadata
        Metadata describing the task's parameters, questions, etc.

    Returns
    -------
    callable
        Class decorator that registers the task

    Raises
    ------
    ValueError
        If class.task_id doesn't match metadata.task_id

    Example
    -------
    ```python
    MY_TASK_METADATA = TaskMetadata(
        task_id="my_task",
        display_name="My Task",
        ...
    )

    @pineuro_task(MY_TASK_METADATA)
    class MyTask(BaseTask):
        task_id = "my_task"
        ...
    ```
    """

    def decorator(cls):
        # Verify task_id matches
        if hasattr(cls, "task_id") and cls.task_id != metadata.task_id:
            raise ValueError(
                f"Task class '{cls.__name__}' has task_id='{cls.task_id}' "
                f"but metadata has task_id='{metadata.task_id}'. They must match."
            )

        # Register metadata
        register_task(metadata)

        # Register class (import here to avoid circular imports)
        from .factory import register_task_class

        register_task_class(metadata.task_id, cls)

        return cls

    return decorator


def discover_tasks(package_name: str | None = None) -> int:
    """Discover and register tasks from installed entry points or a package.

    Third-party task packages can advertise themselves in their
    ``pyproject.toml``::

        [project.entry-points."pineuro.tasks"]
        my_task = "my_package.tasks"

    When pineuro starts, call ``discover_tasks()`` (no arguments) to import
    every package registered under the ``pineuro.tasks`` entry-point group.
    Each imported package is expected to trigger its own ``@pineuro_task``
    decorators, which registers the tasks automatically.

    Parameters
    ----------
    package_name : str, optional
        If provided, import this single package directly (useful for testing
        or manual loading).  If ``None``, discover all packages registered
        under the ``"pineuro.tasks"`` entry-point group.

    Returns
    -------
    int
        Number of packages successfully imported.

    Examples
    --------
    Auto-discover all installed task plugins:

    >>> from pineuro.tasks.task_registry import discover_tasks
    >>> n = discover_tasks()

    Load a specific package manually:

    >>> discover_tasks("my_nfb_tasks.tasks")
    1
    """
    import importlib
    from importlib.metadata import entry_points

    if package_name is not None:
        importlib.import_module(package_name)
        logger.info(f"Loaded task plugin from package: {package_name}")
        return 1

    eps = entry_points(group="pineuro.tasks")
    count = 0
    for ep in eps:
        try:
            importlib.import_module(ep.value)
            count += 1
            logger.info(f"Loaded task plugin '{ep.name}' from {ep.value}")
        except Exception as exc:
            logger.warning(f"Failed to load task plugin '{ep.name}' ({ep.value}): {exc}")
    return count
