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

    # Legacy top-level YAML keys that no longer exist in the new schema.
    # Their presence triggers an immediate ValueError so partial migrations fail
    # loudly at GUI load time rather than silently at runtime.
    _LEGACY_TOP_LEVEL_KEYS = frozenset({"mask", "feedback", "task", "localizer_task", "scan"})

    @classmethod
    def from_dict(cls, data: dict, file_path: Path | None = None) -> "StudyConfig":
        """Create from dictionary loaded from YAML."""
        metadata_dict = data.get("_metadata", {})
        metadata = StudyMetadata.from_dict(metadata_dict)
        session_config = {k: v for k, v in data.items() if k != "_metadata"}

        # Reject old-shape YAMLs early — fail loudly at load, not at runtime.
        found_legacy = cls._LEGACY_TOP_LEVEL_KEYS & set(session_config.keys())
        if found_legacy:
            raise ValueError(
                f"Study config uses outdated schema (found legacy top-level keys: "
                f"{sorted(found_legacy)}). This version of pineuro removed the legacy "
                f"single-block schema; regenerate the YAML via the GUI study editor or "
                f"rebuild from scratch."
            )

        return cls(
            metadata=metadata,
            session_config=session_config,
            file_path=file_path,
        )
