"""Data models for GUI state management and configuration.

This module defines the data structures used by the pineuro GUI to
manage study configurations. These models bridge the gap between
the GUI's visual representation and the underlying YAML configuration
files stored on disk.

Key Components
--------------
StudyMetadata : Study-level metadata
    Name, description, and timestamps for a study configuration.
    This information is displayed in the study manager cards.

StudyConfig : Complete study configuration
    Combines metadata with the actual pipeline configuration
    (session_config dict). This is the primary data structure
    passed between GUI screens.

Serialization
-------------
Both models support bidirectional conversion with dictionaries
for YAML serialization:

- to_dict() : Convert model to dictionary for saving
- from_dict() : Create model from loaded dictionary

The StudyConfig merges metadata into the YAML structure using
a special "_metadata" key to keep everything in one file.

File Structure
--------------
Studies are stored as YAML files with this structure:

    _metadata:
      name: "My Study"
      description: "Description..."
      created: "2024-01-15T10:30:00"
      modified: "2024-01-15T14:45:00"
    paths:
      base_dir: "/data/study"
      ...
    preprocessing:
      ...
    feedback:
      ...

Dependencies
------------
dataclasses, datetime, pathlib

See Also
--------
ConfigManager : Handles loading/saving of these models
pineuro.session : SessionConfig class for pipeline configuration
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class StudyMetadata:
    """Metadata for a study configuration.

    Attributes:
        name: Study name (must be unique, used as filename)
        description: Human-readable description
        created: Creation timestamp
        modified: Last modification timestamp
    """

    name: str
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())

    def update_modified(self) -> None:
        """Update the modified timestamp to now."""
        self.modified = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created": self.created,
            "modified": self.modified,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudyMetadata":
        """Create from dictionary loaded from YAML."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            created=data.get("created", datetime.now().isoformat()),
            modified=data.get("modified", datetime.now().isoformat()),
        )


@dataclass
class StudyConfig:
    """Complete study configuration including metadata and SessionConfig.

    This wraps a SessionConfig with study-level metadata for GUI management.

    Attributes:
        metadata: Study metadata (name, description, timestamps)
        session_config: The actual pipeline configuration (from pineuro.session)
        file_path: Path to the YAML file (None if not yet saved)
    """

    metadata: StudyMetadata
    session_config: dict  # Will contain SessionConfig as dict
    file_path: Path | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "_metadata": self.metadata.to_dict(),
            **self.session_config,
        }

    @classmethod
    def from_dict(cls, data: dict, file_path: Path | None = None) -> "StudyConfig":
        """Create from dictionary loaded from YAML."""
        metadata_dict = data.get("_metadata", {})
        metadata = StudyMetadata.from_dict(metadata_dict)
        session_config = {k: v for k, v in data.items() if k != "_metadata"}

        return cls(
            metadata=metadata,
            session_config=session_config,
            file_path=file_path,
        )

    @property
    def session_count(self) -> int:
        """Number of sessions defined in this study's protocol.

        A study's ``session_config`` carries an ordered ``sessions`` list; each
        entry holds that session's own ``localizers``/``feedback_runs``. Returns
        0 for a config that has no ``sessions`` key yet (e.g. a brand-new editor
        state before the first session is added).
        """
        sessions = self.session_config.get("sessions") or []
        return len(sessions)

    def build_session_config(
        self,
        session_index: int,
        participant_id: str,
        session_id: str,
    ) -> dict:
        """Flatten this study's protocol for one session into a runtime config dict.

        The study-wide/environment blocks (``paths``, ``simulator``, ``scanner``,
        ``reference``, ``sham``, ``display``, ``study`` ...) are copied verbatim;
        the chosen session's ``localizers``/``feedback_runs`` are lifted to the top
        level (the shape :meth:`pineuro.session.SessionConfig._from_dict` consumes),
        and a ``session`` metadata block is attached. ``paths.base_dir`` is rewritten
        to the participant-session directory (the GUI path convention: subdirs like
        ``mask_extraction/`` and ``feedback/`` hang directly off ``paths.base_dir``),
        stripping any stale participant component so re-builds are idempotent.

        The study config is not mutated; a new dict is returned.

        Args:
            session_index: 0-based index into ``sessions`` (session_number == index + 1).
            participant_id: Participant identifier (e.g. "sub-01").
            session_id: BIDS session id (e.g. "ses-02").

        Returns:
            A new session_config dict for this participant-session.

        Raises:
            ValueError: if the study defines no ``sessions``.
            IndexError: if ``session_index`` is out of range.
        """
        sessions = self.session_config.get("sessions")
        if not sessions:
            raise ValueError(
                "study config has no 'sessions' list; run the per-session migration script "
                "(scripts/migrate_studies_to_sessions.py)"
            )
        if not 0 <= session_index < len(sessions):
            raise IndexError(f"session_index {session_index} out of range for {len(sessions)} session(s)")

        session_proto = sessions[session_index]

        # Study-wide blocks: everything except the per-session protocol list.
        result: dict = {k: copy.deepcopy(v) for k, v in self.session_config.items() if k != "sessions"}

        # Lift the chosen session's protocol to the top level.
        result["localizers"] = copy.deepcopy(session_proto.get("localizers", []))
        result["feedback_runs"] = copy.deepcopy(session_proto.get("feedback_runs", []))

        # Attach the session metadata block (matches SessionMetadata fields).
        result["session"] = {
            "session_id": session_id,
            "session_number": session_index + 1,
            "notes": session_proto.get("notes", ""),
        }

        # Rewrite base_dir to the participant-session directory.
        paths = result.setdefault("paths", {})
        study_base = Path(str(paths.get("base_dir", "")))
        parts = study_base.parts
        if participant_id in parts:
            study_base = Path(*parts[: parts.index(participant_id)])
        paths["base_dir"] = str(study_base / participant_id / session_id)

        return result
