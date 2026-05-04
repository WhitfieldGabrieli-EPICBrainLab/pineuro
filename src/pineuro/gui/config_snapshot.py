"""Study configuration snapshot for participant folders.

Saves a copy of the study configuration into each participant-visit
directory for reproducibility. On re-run, compares the saved snapshot
against the current config and warns the user if protocol-relevant
fields have changed.

Snapshot file: ``study_config.yaml`` at the participant-visit level.

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
PROTOCOL_KEYS = frozenset({"scan", "mask", "feedback", "task", "sham", "display", "localizer_task", "study"})

# Scan sub-keys that are source/simulator related, not protocol-defining.
SCAN_EXCLUDED_KEYS = frozenset(
    {
        "localizer_source_dicoms",
        "feedback_source_dicoms",
        "transfer_source_dicoms",
        "reference_initial_dicoms",
        "reference_realtime_dicoms",
        "simulator_source_mapping",
        "simulator_enabled",
        "collection_speed",
        "feedback_simulator_speed",
        "simulator_trigger_mode",
        "scanner_dicom_dir",
        "scanner_ip",
    }
)


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
    visit_id: str | None = None,
) -> Path:
    """Save a study config snapshot to the participant directory.

    Args:
        study_config: A ``StudyConfig`` instance (or any object with a
            ``to_dict()`` method).
        participant_dir: The participant-visit directory.
        participant_id: Participant identifier.
        visit_id: Visit identifier, or ``None`` for single-visit.

    Returns:
        Path to the written snapshot file.
    """
    from pineuro import __version__

    config_dict = study_config.to_dict()
    config_dict["_snapshot"] = {
        "saved_at": datetime.now().isoformat(),
        "participant_id": participant_id,
        "visit_id": visit_id,
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

        if key == "scan":
            saved_section = {k: v for k, v in saved_section.items() if k not in SCAN_EXCLUDED_KEYS}
            current_section = {k: v for k, v in current_section.items() if k not in SCAN_EXCLUDED_KEYS}

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
    elif saved != current:
        differences.append(ConfigDifference(prefix, saved, current))

    return differences
