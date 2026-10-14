"""Disk-space checks for the data a study writes.

Backs the GUI's free-space bar and its pre-acquisition low-space check: free
space on the volume holding a study's data root, and the size of the largest
previous session as the projected need of the next one. Standard library only,
so GUI modules can import it at top level (see the GUI lazy-import rule).
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiskSpace:
    """Capacity of the volume holding a path."""

    path: Path  # the existing directory the figures were read from
    free: int  # bytes available to this user
    total: int  # bytes
    volume: int  # st_dev of ``path``, to tell volumes apart


def existing_parent(path: str | Path) -> Path | None:
    """Return *path* or its nearest existing ancestor, or None if none exists."""
    p = Path(path)
    for candidate in (p, *p.parents):
        if candidate.exists():
            return candidate
    return None


def disk_space(path: str | Path) -> DiskSpace | None:
    """Free and total space on the volume holding *path*.

    A directory that does not exist yet (e.g. a new study's data root) is
    measured at its nearest existing ancestor. Returns None if nothing on the
    path exists or the volume cannot be queried.
    """
    target = existing_parent(path)
    if target is None:
        return None
    try:
        usage = shutil.disk_usage(target)
        volume = os.stat(target).st_dev
    except OSError as exc:
        logger.debug(f"Cannot query disk space for {target}: {exc}")
        return None
    return DiskSpace(path=target, free=usage.free, total=usage.total, volume=volume)


def format_bytes(n: int) -> str:
    """Format a byte count in decimal (SI) units, as macOS reports disk space, e.g. ``'212.4 GB'``."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


def directory_size(path: str | Path) -> int:
    """Total size in bytes of the files under *path* (symlinks not followed, unreadable entries skipped)."""
    total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def largest_session_size(study_root: str | Path, exclude: str | Path | None = None) -> int | None:
    """Size of the largest session directory (``<study_root>/<participant>/ses-*``) in a study.

    Used as the projected disk need of the next session, so the low-space
    warning rests on this study's own data rather than a fixed threshold.
    *exclude* (the session being run) is skipped. Returns None when the study
    has no other session yet.
    """
    root = Path(study_root)
    skip = Path(exclude).resolve() if exclude is not None else None
    try:
        participants = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return None
    sizes: list[int] = []
    for participant in participants:
        try:
            sessions = [s for s in participant.iterdir() if s.is_dir() and s.name.startswith("ses-")]
        except OSError:
            continue
        for session in sessions:
            if skip is not None and session.resolve() == skip:
                continue
            sizes.append(directory_size(session))
    return max(sizes) if sizes else None


def low_space_reason(space: DiskSpace, projected: int | None) -> str | None:
    """Why *space* looks too small for another session, or None if it does not.

    Only warns against the data-derived need from :func:`largest_session_size`;
    with no previous session there is no basis for a threshold, so nothing is
    flagged.
    """
    if projected is None or space.free >= projected:
        return None
    return f"less than the largest previous session of this study ({format_bytes(projected)})"
