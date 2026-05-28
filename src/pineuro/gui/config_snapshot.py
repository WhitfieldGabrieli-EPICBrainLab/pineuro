"""Study configuration snapshot for participant folders.

Saves a copy of the study configuration into each participant-session
directory for reproducibility. On re-run, compares the saved snapshot
against the current config and warns the user if protocol-relevant
fields have changed.

Snapshot file: ``study_config.yaml`` at the participant-session level.

This module is intentionally lightweight (stdlib + yaml only) so it
can be imported at the top level of GUI modules without violating
the lazy-import rule for heavy pineuro modules.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "study_config.yaml"

# Top-level config keys that define the experimental protocol.
# Source/simulator/scanner blocks are intentionally excluded — they're environment-
# specific, not protocol-defining. Per-localizer / per-feedback-run sub-keys like
# `extraction:` and `qc:`/`glm:` are part of `localizers` / `feedback_runs` and
# are compared recursively by the deep-diff.
PROTOCOL_KEYS = frozenset({"study", "sham", "display", "reference", "localizers", "feedback_runs"})


@dataclass
class ConfigDifference:
    """A single difference between a saved snapshot and the current config."""

    field_path: str
    saved_value: Any
    current_value: Any

    def __str__(self) -> str:
        return f"{self.field_path}: {self.saved_value!r} -> {self.current_value!r}"


def save_config_snapshot(
    study_config: Any,
    participant_dir: Path,
    participant_id: str,
    session_id: str | None = None,
) -> Path:
    """Save a study config snapshot to the participant directory.

    Args:
        study_config: A ``StudyConfig`` instance (or any object with a
            ``to_dict()`` method).
        participant_dir: The participant-session directory.
        participant_id: Participant identifier.
        session_id: Session identifier, or ``None`` for single-session.

    Returns:
        Path to the written snapshot file.
    """
    from pineuro import __version__

    config_dict = study_config.to_dict()
    config_dict["_snapshot"] = {
        "saved_at": datetime.now().isoformat(),
        "participant_id": participant_id,
        "session_id": session_id,
        "pineuro_version": __version__,
    }

    snapshot_path = participant_dir / SNAPSHOT_FILENAME
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved study config snapshot to %s", snapshot_path)
    return snapshot_path


def load_config_snapshot(participant_dir: Path) -> dict | None:
    """Load an existing config snapshot from a participant directory.

    Returns:
        The full config dict (including ``_snapshot`` and ``_metadata``),
        or ``None`` if no snapshot file exists.
    """
    snapshot_path = participant_dir / SNAPSHOT_FILENAME
    if not snapshot_path.exists():
        return None

    with open(snapshot_path) as f:
        data: dict = yaml.safe_load(f)

    return data


def archive_config_snapshot(participant_dir: Path) -> Path | None:
    """Rename an existing snapshot with a timestamp suffix for archival.

    The archived file is named ``study_config.{ISO_timestamp}.yaml``.

    Returns:
        Path to the archived file, or ``None`` if no snapshot existed.
    """
    snapshot_path = participant_dir / SNAPSHOT_FILENAME
    if not snapshot_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive_name = f"study_config.{timestamp}.yaml"
    archive_path = participant_dir / archive_name
    snapshot_path.rename(archive_path)

    logger.info("Archived config snapshot to %s", archive_path)
    return archive_path


def compare_protocol_configs(
    saved: dict,
    current: dict,
) -> list[ConfigDifference]:
    """Compare protocol-relevant fields between saved and current config.

    Only fields in :data:`PROTOCOL_KEYS` are compared. Within the
    ``scan`` section, source/simulator keys are excluded.

    Args:
        saved: The saved snapshot dict (as returned by :func:`load_config_snapshot`).
        current: The current config dict (from ``StudyConfig.to_dict()``).

    Returns:
        List of differences. Empty means the configs match.
    """
    differences: list[ConfigDifference] = []

    for key in sorted(PROTOCOL_KEYS):
        saved_section = saved.get(key, {})
        current_section = current.get(key, {})

        # Treat None as empty dict for robustness.
        if saved_section is None:
            saved_section = {}
        if current_section is None:
            current_section = {}

        differences.extend(_deep_diff(saved_section, current_section, prefix=key))

    return differences


def format_differences(differences: list[ConfigDifference]) -> str:
    """Format a list of differences as human-readable text."""
    if not differences:
        return "No differences."

    lines = [f"  {d}" for d in differences]
    return "\n".join(lines)


def _deep_diff(
    saved: Any,
    current: Any,
    prefix: str = "",
) -> list[ConfigDifference]:
    """Recursively compare two nested structures.

    Returns a list of :class:`ConfigDifference` for every leaf value
    that differs between *saved* and *current*.
    """
    differences: list[ConfigDifference] = []

    if isinstance(saved, dict) and isinstance(current, dict):
        all_keys = sorted(set(saved) | set(current))
        for k in all_keys:
            child_prefix = f"{prefix}.{k}" if prefix else k
            if k not in saved:
                differences.append(ConfigDifference(child_prefix, "<not set>", current[k]))
            elif k not in current:
                differences.append(ConfigDifference(child_prefix, saved[k], "<not set>"))
            else:
                differences.extend(_deep_diff(saved[k], current[k], prefix=child_prefix))
    elif (
        isinstance(saved, list)
        and isinstance(current, list)
        and saved
        and current
        and all(isinstance(x, dict) for x in saved)
        and all(isinstance(x, dict) for x in current)
    ):
        # Lists of dicts (e.g. localizers, feedback_runs): compare element-wise
        # with element-indexed paths so changes to individual fields surface.
        for i in range(max(len(saved), len(current))):
            child_prefix = f"{prefix}.{i}"
            if i >= len(saved):
                differences.append(ConfigDifference(child_prefix, "<not set>", current[i]))
            elif i >= len(current):
                differences.append(ConfigDifference(child_prefix, saved[i], "<not set>"))
            else:
                differences.extend(_deep_diff(saved[i], current[i], prefix=child_prefix))
    elif saved != current:
        differences.append(ConfigDifference(prefix, saved, current))

    return differences
