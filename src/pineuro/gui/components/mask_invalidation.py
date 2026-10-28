"""Locate and invalidate a participant's mask outputs when an operator changes them.

When a brain mask is recomputed, or a task mask is re-thresholded or replaced at
review, everything derived from the old mask must go: the extraction outputs
(for a new brain mask), and the coregistered copies under
``mask_extraction/04_reference`` and ``feedback/01_reference``. Deleting them
(rather than only flagging steps for overwrite) keeps the participant workflow's
file-based step detection correct across an app restart: those steps simply
return to PENDING.

Free of heavy imports (stdlib, plus ``pineuro.gui.utils`` inside a function), so it
is safe to import at the GUI's top level.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The coregistration step directories (initial reference, then the real-time one)
_COREGISTRATION_DIRS = (("mask_extraction", "04_reference"), ("feedback", "01_reference"))


def is_multi_localizer(config: dict[str, Any]) -> bool:
    """Whether the session has more than one localizer (per-localizer extraction folders)."""
    return len(config.get("localizers") or []) > 1


def localizer_task_name(loc: dict[str, Any]) -> str:
    """The task name a localizer's preprocessing and extraction files are named by."""
    loc_type = loc.get("type", "rest")
    return "rest" if loc_type == "rest" else str(loc_type)


def extraction_root(base_dir: Path, config: dict[str, Any], loc_idx: int) -> Path:
    """The ``root_dir`` a localizer's mask extraction writes to.

    ``mask_extraction/`` for a single localizer, ``mask_extraction/loc_{idx}_{task}/``
    when the session has several.
    """
    mask_extraction_dir = Path(base_dir) / "mask_extraction"
    if not is_multi_localizer(config):
        return mask_extraction_dir
    loc = (config.get("localizers") or [])[loc_idx]
    return mask_extraction_dir / f"loc_{loc_idx}_{localizer_task_name(loc)}"


def copy_to_shared_masks(base_dir: Path, files: list[Path]) -> list[Path]:
    """Copy a localizer's mask files to the shared ``mask_extraction/03_masks``."""
    shared_dir = Path(base_dir) / "mask_extraction" / "03_masks"
    shared_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in files:
        dest = shared_dir / f.name
        shutil.copy2(f, dest)
        copied.append(dest)
    return copied


def store_participant_mask(root_dir: Path, bids_prefix: str, mask_name: str, source: Path) -> Path:
    """Copy a participant-space replacement mask into the extraction root's ``mask_edits/``.

    The recorded override then points at the copy, which stays valid if the
    original is edited or moved (and survives :func:`invalidate_localizer_extraction`).
    """
    dest = Path(root_dir) / "mask_edits" / f"{bids_prefix}_{mask_name}_edited.nii.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return dest


def _remove(path: Path, removed: list[Path]) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        removed.append(path)
    elif path.exists():
        path.unlink()
        removed.append(path)


def invalidate_coregistration(base_dir: Path, *, include_transforms: bool) -> list[Path]:
    """Delete the coregistered masks so both coregistration steps run again.

    Removes ``masks/`` and ``qc/`` under ``mask_extraction/04_reference`` and
    ``feedback/01_reference``, plus ``xfm/`` when ``include_transforms`` (the
    localizer's boldref changed). The converted reference scans and their means
    depend only on the DICOMs and are kept.

    Returns
    -------
    list[Path]
        The paths removed.
    """
    removed: list[Path] = []
    subdirs = ("masks", "qc", "xfm") if include_transforms else ("masks", "qc")
    for parent, name in _COREGISTRATION_DIRS:
        reference_dir = Path(base_dir) / parent / name
        for sub in subdirs:
            _remove(reference_dir / sub, removed)
    if removed:
        logger.info(f"Invalidated coregistration outputs: {[str(p) for p in removed]}")
    return removed


def invalidate_localizer_extraction(
    base_dir: Path, config: dict[str, Any], loc_idx: int, bids_prefix: str
) -> list[Path]:
    """Delete a localizer's mask-extraction outputs after its brain mask changed.

    Removes the extraction root's ``01_*`` and ``02_*`` folders and ``03_masks``
    (and ``05_tissue`` with a single localizer, whose tissue masks are built from
    the same boldref), and the approval marker. ``reproduce_masks.py``,
    ``mask_overrides.json`` and ``mask_edits/`` are kept, so the operator's
    per-mask choices are re-applied when extraction runs again. With several localizers, this
    localizer's masks are also removed from the shared ``03_masks``.

    Returns
    -------
    list[Path]
        The paths removed.
    """
    root = extraction_root(base_dir, config, loc_idx)
    removed: list[Path] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child.name[:3] in ("01_", "02_") or child.name == "03_masks"):
                _remove(child, removed)
        if not is_multi_localizer(config):
            _remove(root / "05_tissue", removed)
        _remove(root / ".masks_approved", removed)

    if is_multi_localizer(config):
        from pineuro.gui.utils import localizer_mask_names

        loc = (config.get("localizers") or [])[loc_idx]
        shared_dir = Path(base_dir) / "mask_extraction" / "03_masks"
        for name in localizer_mask_names(loc):
            _remove(shared_dir / f"{bids_prefix}_{name}_thr.nii.gz", removed)

    if removed:
        logger.info(f"Invalidated mask extraction outputs: {[str(p) for p in removed]}")
    return removed
