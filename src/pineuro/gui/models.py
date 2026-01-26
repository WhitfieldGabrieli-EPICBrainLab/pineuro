"""
Data models for GUI state management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


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
    file_path: Optional[Path] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "_metadata": self.metadata.to_dict(),
            **self.session_config,
        }

    @classmethod
    def from_dict(cls, data: dict, file_path: Optional[Path] = None) -> "StudyConfig":
        """Create from dictionary loaded from YAML."""
        metadata_dict = data.pop("_metadata", {})
        metadata = StudyMetadata.from_dict(metadata_dict)

        return cls(
            metadata=metadata,
            session_config=data,
            file_path=file_path,
        )
