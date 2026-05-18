"""Per-participant manual-mask file selections.

A manual-mask localizer stores only participant-agnostic data in the study config
(mask space + network names). The actual NIfTI files differ per participant, so they
are chosen in the participant workflow and persisted here, in ``manual_masks.json``
inside the participant's base directory.

The file is a JSON object keyed by localizer ``id`` so multi-localizer studies keep
independent records::

    {
      "manual": {
        "mask_space": "anat",
        "masks": {"DMN": "C:/data/sub-01/dmn.nii.gz", "FPN": "C:/data/sub-01/fpn.nii.gz"},
        "reference": "C:/data/sub-01/ref.nii.gz",
        "anat": "C:/data/sub-01/t1.nii.gz"
      }
    }

This module is intentionally dependency-light (``json`` + ``pathlib`` only) so both
``pipeline_runner`` and ``participant_workflow`` can import it without circular imports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANUAL_MASKS_FILENAME = "manual_masks.json"


def manual_masks_path(base_dir: Path | str) -> Path:
    """Return the path to a participant's manual-mask record file."""
    return Path(base_dir) / MANUAL_MASKS_FILENAME


def _load_all(base_dir: Path | str) -> dict[str, Any]:
    """Load the whole manual_masks.json mapping; empty dict if absent/invalid."""
    path = manual_masks_path(base_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def read_manual_masks(base_dir: Path | str, loc_id: str) -> dict[str, Any] | None:
    """Return the manual-mask record for one localizer, or None if not configured."""
    record = _load_all(base_dir).get(loc_id)
    return record if isinstance(record, dict) else None


def write_manual_masks(base_dir: Path | str, loc_id: str, record: dict[str, Any]) -> None:
    """Merge one localizer's manual-mask record into manual_masks.json.

    Other localizers' records are preserved (read-merge-write).
    """
    data = _load_all(base_dir)
    data[loc_id] = record
    path = manual_masks_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def manual_masks_complete(base_dir: Path | str, loc: dict[str, Any]) -> bool:
    """Whether a localizer's per-participant manual masks are fully configured.

    Requires a record with every network name mapped to an existing file, an
    existing reference EPI, and (for anatomical-space masks) an existing T1.
    """
    record = read_manual_masks(base_dir, loc.get("id", "manual"))
    if not record:
        return False

    masks = record.get("masks") or {}
    for name in loc.get("manual_mask_names") or []:
        path = masks.get(name)
        if not path or not Path(path).is_file():
            return False

    reference = record.get("reference")
    if not reference or not Path(reference).is_file():
        return False

    mask_space = record.get("mask_space") or loc.get("manual_mask_space", "native")
    if mask_space == "anat":
        anat = record.get("anat")
        if not anat or not Path(anat).is_file():
            return False

    return True
