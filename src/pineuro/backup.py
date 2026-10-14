"""Session backup: mirror a participant-session folder to a second location.

Copies new and changed files from ``<data root>/<participant>/<session>`` to
``<destination>/<participant>/<session>``. A file is copied when it is missing
at the destination or its size or modification time differs, through a
temporary name, so an interrupted copy never leaves a truncated file under the
real name. Nothing at the destination is ever deleted: files later discarded or
cleaned up locally stay in the backup. ``tmp/`` (simulator and per-run scratch)
is never copied, and the raw DICOM copies in ``collection/`` only when the study
opts in.

The destination folder itself must already exist, so a backup to a drive that
is not connected fails instead of silently filling the local disk under the
drive's empty mount point. Standard library only, so the GUI can import it at
top level; the participant workflow runs it between acquisitions.
"""

import json
import logging
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The study YAML's top-level ``backup:`` block, with its defaults.
BACKUP_DEFAULTS: dict[str, Any] = {"enabled": False, "destination": "", "include_dicoms": True}

# Never backed up: simulator incoming copies and other per-run scratch.
_ALWAYS_EXCLUDED = frozenset({"tmp"})
# Raw DICOM copies; backed up only with ``include_dicoms``.
_DICOM_DIR = "collection"
_PARTIAL_SUFFIX = ".pineuro-partial"
MANIFEST_NAME = ".pineuro_backup.json"
# Modification times are compared with this tolerance because FAT/exFAT drives
# (common on external backup disks) store them at 2 s resolution.
_MTIME_TOLERANCE = 2.0


@dataclass
class BackupResult:
    """Outcome of one :func:`mirror_session` pass."""

    destination: Path
    copied: int = 0
    skipped: int = 0  # already up to date at the destination
    bytes_copied: int = 0
    errors: list[str] = field(default_factory=list)
    stopped: bool = False  # interrupted via stop_event before finishing

    @property
    def complete(self) -> bool:
        return not self.errors and not self.stopped


def backup_destination(session_dir: str | Path, destination: str | Path) -> Path:
    """Where *session_dir* (``<root>/<participant>/<session>``) is mirrored to."""
    session_dir = Path(session_dir)
    return Path(destination) / session_dir.parent.name / session_dir.name


def validate_backup_config(backup: dict[str, Any], data_root: str | Path) -> str | None:
    """Return an error message for an unusable ``backup:`` study block, or None."""
    if not backup.get("enabled"):
        return None
    destination = str(backup.get("destination") or "").strip()
    if not destination:
        return "Backup is enabled but no backup folder is set."
    if data_root and _same_path(destination, data_root):
        return "The backup folder cannot be the data directory itself."
    return None


def read_manifest(session_dir: str | Path, destination: str | Path) -> dict[str, Any] | None:
    """The record of the last backup pass of *session_dir*, or None if there is none."""
    path = backup_destination(session_dir, destination) / MANIFEST_NAME
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def mirror_session(
    session_dir: str | Path,
    destination: str | Path,
    *,
    include_dicoms: bool = True,
    stop_event: Any = None,
) -> BackupResult:
    """Copy new and changed files from *session_dir* to its backup location.

    Parameters
    ----------
    session_dir
        The participant-session folder, ``<data root>/<participant>/<session>``.
    destination
        Backup folder; must already exist. The session is mirrored to
        ``<destination>/<participant>/<session>``.
    include_dicoms
        Also copy the raw DICOM copies in ``collection/``.
    stop_event
        Anything with ``is_set()``; checked between files, so a pass can be
        interrupted (e.g. when an acquisition starts) and resumed later.
    """
    src_root = Path(session_dir)
    dst_root = backup_destination(src_root, destination)
    result = BackupResult(destination=dst_root)

    if not Path(destination).is_dir():
        result.errors.append(f"Backup folder {destination} is not available (is the drive connected?)")
        return result
    if _same_path(dst_root, src_root) or _is_within(dst_root, src_root):
        result.errors.append(f"Backup location {dst_root} is inside the session folder; nothing was copied")
        return result

    excluded = set(_ALWAYS_EXCLUDED)
    if not include_dicoms:
        excluded.add(_DICOM_DIR)

    for src in _iter_files(src_root, excluded):
        if stop_event is not None and stop_event.is_set():
            result.stopped = True
            break
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        try:
            src_stat = src.stat()
            if _up_to_date(src_stat, dst):
                result.skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            partial = dst.with_name(dst.name + _PARTIAL_SUFFIX)
            shutil.copy2(src, partial)
            os.replace(partial, dst)
        except OSError as exc:
            result.errors.append(f"{rel}: {exc}")
            continue
        result.copied += 1
        result.bytes_copied += src_stat.st_size

    _write_manifest(src_root, result)
    return result


def _iter_files(root: Path, excluded_top_level: set[str]) -> Iterator[Path]:
    """Regular files under *root* (symlinks skipped), minus excluded top-level folders."""
    stack: list[tuple[Path, bool]] = [(root, True)]
    while stack:
        current, is_top = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if not (is_top and entry.name in excluded_top_level):
                                stack.append((Path(entry.path), False))
                        elif entry.is_file(follow_symlinks=False) and not entry.name.endswith(_PARTIAL_SUFFIX):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except OSError as exc:
            logger.warning(f"Backup: cannot read {current}: {exc}")


def _up_to_date(src_stat: os.stat_result, dst: Path) -> bool:
    try:
        dst_stat = dst.stat()
    except OSError:
        return False
    return dst_stat.st_size == src_stat.st_size and abs(dst_stat.st_mtime - src_stat.st_mtime) <= _MTIME_TOLERANCE


def _write_manifest(src_root: Path, result: BackupResult) -> None:
    record = {
        "source": str(src_root),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "complete": result.complete,
        "copied": result.copied,
        "skipped": result.skipped,
        "bytes_copied": result.bytes_copied,
        "errors": len(result.errors),
    }
    path = result.destination / MANIFEST_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + _PARTIAL_SUFFIX)
        partial.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(partial, path)
    except OSError as exc:
        result.errors.append(f"{MANIFEST_NAME}: {exc}")


def _same_path(a: str | Path, b: str | Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return path.resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
