"""Validation -- task plugin lifecycle (registration, lookup, parameter validation).

The task plugin system has a small surface area but several invariants that
downstream code depends on:

- ``register_task(metadata)`` stores a TaskMetadata by its task_id.
- ``get_task_metadata(task_id)`` returns the registered metadata; raises
  ``ValueError`` for unknown ids.
- ``list_available_tasks(role=...)`` returns ids whose roles set includes
  the requested role.
- ``TaskParameter.validate(value)`` enforces type, range, and choices.
- ``TaskMetadata.validate_config(config)`` aggregates per-parameter errors
  into a single TaskConfigError.

Every assertion below uses literal expected values derived from the registry's
documented contract, not from re-running the validation logic.
"""

from __future__ import annotations

import pytest

from pineuro.tasks.task_registry import (
    SessionStateField,
    TaskConfigError,
    TaskMetadata,
    TaskParameter,
    get_task_metadata,
    list_available_tasks,
    register_task,
)

pytestmark = pytest.mark.validation


def _make_metadata(task_id, **overrides):
    """Build a minimal TaskMetadata with a single int parameter."""
    defaults = {
        "task_id": task_id,
        "display_name": f"Test Task {task_id}",
        "description": "validation-suite test task",
        "parameters": [TaskParameter(key="amplitude", label="Amplitude", type="float", default=1.0)],
        "session_state_fields": [SessionStateField(key="hits", type="int", default=0)],
    }
    defaults.update(overrides)
    return TaskMetadata(**defaults)


# =============================================================================
# Registration round-trip
# =============================================================================


def test_register_and_lookup_round_trip():
    """A registered metadata is retrievable by task_id."""
    task_id = "_validation_register_lookup"
    md = _make_metadata(task_id)
    register_task(md)

    fetched = get_task_metadata(task_id)
    assert fetched is md  # exact-object identity, no copy


def test_lookup_unknown_raises():
    """get_task_metadata of an unregistered id raises ValueError."""
    with pytest.raises(ValueError, match="Unknown task"):
        get_task_metadata("_validation_definitely_not_registered_task_id")


def test_list_available_includes_registered():
    """A newly registered task appears in list_available_tasks() output."""
    task_id = "_validation_list_test"
    register_task(_make_metadata(task_id))
    listed = list_available_tasks()
    assert task_id in listed


def test_list_available_filters_by_role():
    """list_available_tasks(role=...) filters by the documented role contract."""
    task_id_localizer = "_validation_localizer_role"
    register_task(_make_metadata(task_id_localizer, roles={"localizer"}))
    task_id_feedback = "_validation_feedback_role"
    register_task(_make_metadata(task_id_feedback, roles={"feedback"}))

    feedback_tasks = list_available_tasks(role="feedback")
    localizer_tasks = list_available_tasks(role="localizer")
    assert task_id_feedback in feedback_tasks
    assert task_id_feedback not in localizer_tasks
    assert task_id_localizer in localizer_tasks
    assert task_id_localizer not in feedback_tasks


# =============================================================================
# TaskParameter validation
# =============================================================================


def test_parameter_validate_accepts_in_range():
    """A value inside [min_value, max_value] passes validation silently."""
    p = TaskParameter(key="thr", label="Threshold", type="float", default=0.5, min_value=0.0, max_value=1.0)
    p.validate(0.0)
    p.validate(0.5)
    p.validate(1.0)


def test_parameter_validate_rejects_below_min():
    """A value below min_value raises ValueError."""
    p = TaskParameter(key="thr", label="Threshold", type="float", default=0.5, min_value=0.0, max_value=1.0)
    with pytest.raises(ValueError, match=">= 0.0"):
        p.validate(-0.1)


def test_parameter_validate_rejects_above_max():
    """A value above max_value raises ValueError."""
    p = TaskParameter(key="thr", label="Threshold", type="float", default=0.5, min_value=0.0, max_value=1.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        p.validate(1.5)


def test_parameter_validate_rejects_non_numeric_for_int():
    """A non-numeric string raises ValueError for int type."""
    p = TaskParameter(key="n", label="N", type="int", default=10)
    with pytest.raises(ValueError, match="number"):
        p.validate("not a number")


def test_parameter_validate_choices_enforced():
    """A str-typed param with choices rejects values outside the choices list."""
    p = TaskParameter(key="mode", label="Mode", type="str", default="a", choices=["a", "b", "c"])
    p.validate("a")
    p.validate("c")
    with pytest.raises(ValueError, match="one of"):
        p.validate("d")


# =============================================================================
# TaskMetadata.validate_config aggregation
# =============================================================================


def test_validate_config_accepts_all_in_range():
    """A config with every param in range passes silently."""
    md = TaskMetadata(
        task_id="_val_config_ok",
        display_name="ok",
        description="",
        parameters=[
            TaskParameter(key="a", label="A", type="float", default=0.5, min_value=0.0, max_value=1.0),
            TaskParameter(key="b", label="B", type="int", default=5, min_value=1, max_value=10),
        ],
    )
    md.validate_config({"a": 0.5, "b": 5})  # passes silently


def test_validate_config_aggregates_multiple_errors():
    """validate_config collects every parameter error before raising.

    With two out-of-range params and one cross-parameter rule that also fires,
    the raised TaskConfigError must list all three problems -- not just the first.
    """

    def _cross(cfg):
        if cfg.get("a", 0) >= cfg.get("b", 0):
            raise ValueError("a must be < b")

    md = TaskMetadata(
        task_id="_val_config_aggregate",
        display_name="agg",
        description="",
        parameters=[
            TaskParameter(key="a", label="A", type="float", default=0.5, min_value=0.0, max_value=1.0),
            TaskParameter(key="b", label="B", type="int", default=5, min_value=1, max_value=10),
        ],
        cross_parameter_validator=_cross,
    )

    with pytest.raises(TaskConfigError) as excinfo:
        md.validate_config({"a": 5.0, "b": 0, "extra_ignored": True})

    # Three errors expected: a > max, b < min, cross-param rule.
    message = str(excinfo.value)
    assert "A must be <= 1.0" in message
    assert "B must be >= 1" in message
    assert "a must be < b" in message


def test_validate_config_ignores_unknown_keys():
    """Keys in config that are not declared parameters are ignored (no error)."""
    md = TaskMetadata(
        task_id="_val_config_extra",
        display_name="extra",
        description="",
        parameters=[TaskParameter(key="a", label="A", type="float", default=0.5, min_value=0.0, max_value=1.0)],
    )
    # Extra "unknown" key does not raise.
    md.validate_config({"a": 0.7, "unknown_param": "ignored"})


# =============================================================================
# n_networks defaulting
# =============================================================================


def test_metadata_default_n_networks_is_one():
    """A metadata without explicit n_networks defaults to 1."""
    md = _make_metadata("_val_default_n_networks")
    assert md.n_networks == 1


def test_metadata_default_roles_is_feedback_only():
    """A metadata without explicit roles defaults to {'feedback'}."""
    md = _make_metadata("_val_default_roles")
    assert md.roles == {"feedback"}


def test_metadata_default_design_type_is_block():
    """A metadata without explicit design_type defaults to 'block'."""
    md = _make_metadata("_val_default_design")
    assert md.design_type == "block"
