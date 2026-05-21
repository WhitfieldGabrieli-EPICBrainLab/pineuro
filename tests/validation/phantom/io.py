"""On-disk layouts — turn phantom arrays into the files the pineuro pipeline reads.

Note on simulator sources: a directory handed to ``VolumeSimulator`` must
contain *only* volume files, because ``pineuro.utils.get_volume_files`` returns
every file in the directory. Write per-volume NIfTIs without sidecars for
simulator sources (pass ``tr`` to the simulator explicitly); write sidecars only
for directories fed straight to the coordinator / estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from .motion import motion_image_comment


def write_nifti(
    data: NDArray,
    affine: NDArray[np.float64],
    path: Path | str,
    *,
    tr: float | None = None,
) -> Path:
    """Write a 3D or 4D array as a NIfTI file. For 4D data, sets pixdim[4] = tr."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=np.float64))
    if tr is not None and np.ndim(data) == 4:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    return path


def sidecar_path(volume_path: Path) -> Path:
    """JSON sidecar path for a NIfTI volume (``vol.nii[.gz]`` -> ``vol.json``)."""
    name = volume_path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        stem = name[: -len(".nii")]
    else:
        stem = volume_path.stem
    return volume_path.with_name(stem + ".json")


def write_json_sidecar(
    volume_path: Path,
    tr: float,
    *,
    motion: NDArray[np.float64] | None = None,
    is_reference: bool = False,
) -> Path:
    """Write a dcm2niix-style JSON sidecar (``RepetitionTime`` + optional motion)."""
    meta: dict = {"RepetitionTime": tr}
    if is_reference:
        meta["ImageComments"] = "Reference volume for motion correction."
    elif motion is not None:
        meta["ImageComments"] = motion_image_comment(motion)
    path = sidecar_path(volume_path)
    path.write_text(json.dumps(meta, indent=2))
    return path


def write_per_volume_niftis(
    data4d: NDArray[np.float64],
    affine: NDArray[np.float64],
    out_dir: Path | str,
    *,
    prefix: str = "vol_",
    ext: str = ".nii.gz",
    tr: float | None = None,
    motion_params: NDArray[np.float64] | None = None,
    write_sidecars: bool = False,
) -> list[Path]:
    """Split a 4D array into one NIfTI per timepoint.

    Filenames are zero-padded (``vol_0000``, ``vol_0001``, ...) so lexical sort
    equals temporal order — the ordering ``get_volume_files`` and the simulator
    rely on.

    Sidecars are written only when ``write_sidecars=True`` or ``motion_params``
    is given; see the module docstring for why simulator sources must omit them.
    Volume 0 is marked the motion-correction reference.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_volumes = data4d.shape[3]
    width = max(4, len(str(n_volumes - 1)))
    want_sidecars = write_sidecars or motion_params is not None

    paths: list[Path] = []
    for t in range(n_volumes):
        vol_path = out_dir / f"{prefix}{t:0{width}d}{ext}"
        write_nifti(data4d[..., t], affine, vol_path)
        paths.append(vol_path)
        if want_sidecars:
            if tr is None:
                raise ValueError("writing sidecars requires tr")
            motion_row = None if motion_params is None else motion_params[t]
            is_ref = motion_params is not None and t == 0
            write_json_sidecar(vol_path, tr, motion=motion_row, is_reference=is_ref)
    return paths


def write_events_tsv(
    path: Path | str,
    onsets: list[float],
    durations: list[float],
    trial_types: list[str],
) -> Path:
    """Write a minimal BIDS ``events.tsv`` (onset, duration, trial_type columns)."""
    path = Path(path)
    lines = ["onset\tduration\ttrial_type"]
    for onset, duration, trial_type in zip(onsets, durations, trial_types, strict=True):
        lines.append(f"{onset}\t{duration}\t{trial_type}")
    path.write_text("\n".join(lines) + "\n")
    return path
