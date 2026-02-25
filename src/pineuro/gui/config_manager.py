"""Configuration manager for study YAML files.

This module manages the lifecycle of study configurations:
- Creating and loading study configurations from YAML files
- Saving configurations with automatic timestamp updates
- Duplicating existing studies
- Deleting studies
- Listing available studies sorted by modification date

All study configurations are stored in ~/.pineuro/studies/ by default.
Each study is a YAML file containing metadata (_metadata section) and
session configuration (SessionConfig compatible dictionary).

Example YAML structure:
    _metadata:
        name: "My Study"
        description: "Study description"
        created: "2026-01-16T10:00:00"
        modified: "2026-01-16T14:30:00"
    paths:
        base_dir: "/data/pineuro"
        ...
    scan:
        tr: 1.2
        ...
"""

import logging
from pathlib import Path

import yaml

from . import STUDIES_DIR
from .models import StudyConfig, StudyMetadata

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages study configuration files in the studies library."""

    def __init__(self, studies_dir: Path | None = None):
        """Initialize the config manager.

        Args:
            studies_dir: Directory for study configs (default: ~/.pineuro/studies/)
        """
        self.studies_dir = studies_dir or STUDIES_DIR
        self._ensure_studies_dir()

    def _ensure_studies_dir(self) -> None:
        """Create the studies directory if it doesn't exist."""
        self.studies_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Studies directory: {self.studies_dir}")

    def _get_study_path(self, study_name: str) -> Path:
        """Get the file path for a study by name.

        Args:
            study_name: Name of the study

        Returns:
            Path to the YAML file
        """
        # Sanitize filename
        safe_name = study_name.replace(" ", "_").replace("/", "_")
        return self.studies_dir / f"{safe_name}.yaml"

    def list_studies(self) -> list[StudyMetadata]:
        """List all available studies.

        Returns:
            List of StudyMetadata objects sorted by modification date (newest first)
        """
        studies = []

        for yaml_file in self.studies_dir.glob("*.yaml"):
            try:
                study_config = self.load_study_from_file(yaml_file)
                studies.append(study_config.metadata)
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
                continue

        # Sort by modified date, newest first
        studies.sort(key=lambda s: s.modified, reverse=True)
        return studies

    def load_study(self, study_name: str) -> StudyConfig | None:
        """Load a study configuration by name.

        Args:
            study_name: Name of the study to load

        Returns:
            StudyConfig if found, None otherwise
        """
        study_path = self._get_study_path(study_name)

        if not study_path.exists():
            logger.warning(f"Study not found: {study_name}")
            return None

        return self.load_study_from_file(study_path)

    def load_study_from_file(self, file_path: Path) -> StudyConfig:
        """Load a study configuration from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            StudyConfig loaded from file

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        logger.info(f"Loading study from {file_path}")

        with open(file_path) as f:
            data = yaml.safe_load(f)

        return StudyConfig.from_dict(data, file_path=file_path)

    def save_study(self, study_config: StudyConfig, overwrite: bool = True) -> Path:
        """Save a study configuration to YAML.

        Args:
            study_config: StudyConfig to save
            overwrite: If False, raise error if file exists

        Returns:
            Path to the saved file

        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        # Update modified timestamp
        study_config.metadata.update_modified()

        # Determine save path
        if study_config.file_path is None:
            study_config.file_path = self._get_study_path(study_config.metadata.name)

        # Check for overwrite
        if study_config.file_path.exists() and not overwrite:
            raise FileExistsError(f"Study file already exists: {study_config.file_path}")

        # Save to YAML
        logger.info(f"Saving study to {study_config.file_path}")

        with open(study_config.file_path, "w") as f:
            yaml.safe_dump(
                study_config.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        return study_config.file_path

    def delete_study(self, study_name: str) -> bool:
        """Delete a study configuration.

        Args:
            study_name: Name of the study to delete

        Returns:
            True if deleted, False if not found
        """
        study_path = self._get_study_path(study_name)

        if not study_path.exists():
            logger.warning(f"Study not found for deletion: {study_name}")
            return False

        logger.info(f"Deleting study: {study_path}")
        study_path.unlink()
        return True

    def _next_unique_name(self, base: str, existing_names: set[str]) -> str:
        """Return *base* if it is not in *existing_names*, otherwise append ``_2``, ``_3``, …

        Parameters
        ----------
        base : str
            Desired name.
        existing_names : set[str]
            Names already in use.

        Returns
        -------
        str
            A name that is not in *existing_names*.
        """
        if base not in existing_names:
            return base
        i = 2
        while f"{base}_{i}" in existing_names:
            i += 1
        return f"{base}_{i}"

    def duplicate_study(self, source_name: str, new_name: str | None = None) -> StudyConfig | None:
        """Duplicate an existing study.

        Args:
            source_name: Name of the study to duplicate
            new_name: Name for the copy. When *None*, a unique name is generated
                by appending ``" (Copy)"`` (and ``_2``, ``_3``, … if needed).

        Returns:
            New StudyConfig if successful, None if source not found
        """
        # Load source study
        source_study = self.load_study(source_name)
        if source_study is None:
            return None

        # Create new name — avoid collisions when duplicating multiple times
        if new_name is None:
            existing = {m.name for m in self.list_studies()}
            new_name = self._next_unique_name(f"{source_name} (Copy)", existing)

        # Create new metadata
        new_metadata = StudyMetadata(
            name=new_name,
            description=source_study.metadata.description,
        )

        # Create new config
        new_study = StudyConfig(
            metadata=new_metadata,
            session_config=source_study.session_config.copy(),
            file_path=None,  # Will be assigned on save
        )

        # Save the duplicate
        self.save_study(new_study)

        logger.info(f"Duplicated '{source_name}' to '{new_name}'")
        return new_study

    def export_study(self, study_name: str, dest_path: Path) -> bool:
        """Export a study configuration YAML to a user-chosen file path.

        Parameters
        ----------
        study_name : str
            Name of the study to export.
        dest_path : Path
            Destination file path for the exported YAML.

        Returns
        -------
        bool
            ``True`` if the file was written, ``False`` if the study was not found.
        """
        import shutil

        source = self._get_study_path(study_name)
        if not source.exists():
            logger.warning(f"Study not found for export: {study_name}")
            return False
        shutil.copy2(source, dest_path)
        logger.info(f"Exported '{study_name}' to {dest_path}")
        return True

    def import_study(self, source_path: Path) -> StudyConfig | None:
        """Import a study configuration YAML from a user-chosen file path.

        If the study name already exists, a unique name is generated by
        appending ``_2``, ``_3``, … to avoid overwriting existing studies.

        Parameters
        ----------
        source_path : Path
            Path to the YAML file to import.

        Returns
        -------
        StudyConfig or None
            The imported study, or ``None`` if the file could not be loaded.
        """
        study = self.load_study_from_file(source_path)
        if study is None:
            logger.warning(f"Failed to load study from: {source_path}")
            return None

        existing = {m.name for m in self.list_studies()}
        study.metadata.name = self._next_unique_name(study.metadata.name, existing)
        study.metadata.update_modified()
        study.file_path = None  # Will be assigned by save_study
        self.save_study(study)
        logger.info(f"Imported study as '{study.metadata.name}' from {source_path}")
        return study
