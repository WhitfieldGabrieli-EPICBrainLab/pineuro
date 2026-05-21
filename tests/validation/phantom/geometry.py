"""Spatial geometry for the phantom library — tissue compartments and ROIs.

ANTs registration and Otsu brain extraction both need continuous intensity
gradients and internal structure to behave well, so the structural phantom is
an anisotropic ellipsoid with concentric CSF / grey-matter / white-matter
compartments and smoothed edges (see the validation plan's Realism section).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

# Tissue label codes used in GroundTruth.tissue_labels.
LABEL_BACKGROUND = 0
LABEL_GM = 1
LABEL_WM = 2
LABEL_CSF = 3


def make_tissue_compartments(
    vol_shape: tuple[int, int, int],
    *,
    radii: tuple[float, float, float] | None = None,
    gm: float = 1000.0,
    wm: float = 700.0,
    csf: float = 1300.0,
    smooth_sigma: float = 1.5,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Build an anisotropic ellipsoidal brain with concentric tissue shells.

    Returns ``(intensity, labels)``: ``intensity`` is a smoothed float volume
    suitable as a structural phantom; ``labels`` is the *crisp* (un-smoothed)
    integer label volume (0 background, 1 GM, 2 WM, 3 CSF) — use ``labels > 0``
    as the ground-truth brain mask.

    The radii are deliberately unequal so the phantom has no rotational
    symmetry, which is required for meaningful motion / coregistration tests.
    """
    nx, ny, nz = vol_shape
    if radii is None:
        radii = (nx * 0.38, ny * 0.32, nz * 0.42)

    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    xx, yy, zz = np.mgrid[0:nx, 0:ny, 0:nz].astype(np.float64)
    radial = np.sqrt(((xx - cx) / radii[0]) ** 2 + ((yy - cy) / radii[1]) ** 2 + ((zz - cz) / radii[2]) ** 2)

    labels = np.full(vol_shape, LABEL_BACKGROUND, dtype=np.int16)
    labels[radial < 1.00] = LABEL_CSF
    labels[radial < 0.85] = LABEL_GM
    labels[radial < 0.55] = LABEL_WM

    intensity = np.zeros(vol_shape, dtype=np.float64)
    intensity[labels == LABEL_CSF] = csf
    intensity[labels == LABEL_GM] = gm
    intensity[labels == LABEL_WM] = wm
    if smooth_sigma > 0:
        intensity = gaussian_filter(intensity, sigma=smooth_sigma)

    return intensity, labels


def bright_box(
    vol_shape: tuple[int, int, int],
    *,
    fraction: float = 0.6,
    intensity: float = 1000.0,
    smooth_sigma: float = 1.5,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """A simple centred bright box — a structure-free alternative brain.

    Used where internal tissue contrast is unnecessary (e.g. plain feedback
    recovery tests). Returns ``(intensity, labels)`` like
    :func:`make_tissue_compartments`.
    """
    labels = np.zeros(vol_shape, dtype=np.int16)
    box = box_slices(vol_shape, fraction)
    labels[box] = LABEL_GM
    image = np.zeros(vol_shape, dtype=np.float64)
    image[box] = intensity
    if smooth_sigma > 0:
        image = gaussian_filter(image, sigma=smooth_sigma)
    return image, labels


def box_slices(vol_shape: tuple[int, int, int], fraction: float = 0.6) -> tuple[slice, slice, slice]:
    """Centred (x, y, z) slices spanning ``fraction`` of each axis."""
    starts = [int(round(v * (1.0 - fraction) / 2.0)) for v in vol_shape]
    sizes = [int(round(v * fraction)) for v in vol_shape]
    return tuple(slice(s, s + n) for s, n in zip(starts, sizes, strict=True))  # type: ignore[return-value]


def place_roi(
    vol_shape: tuple[int, int, int],
    *,
    n_voxels: int = 40,
    center: tuple[int, int, int] | None = None,
) -> NDArray[np.int_]:
    """Return ``(n_voxels, 3)`` voxel coordinates of a compact ROI.

    The ROI is the ``n_voxels`` voxels nearest ``center`` (a roughly spherical
    contiguous blob), so it is well localised for mask-recovery tests.
    """
    nx, ny, nz = vol_shape
    if center is None:
        center = (nx // 2, ny // 2, nz // 2)
    cx, cy, cz = center
    xx, yy, zz = np.mgrid[0:nx, 0:ny, 0:nz]
    dist = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    nearest = np.argsort(dist, axis=None, kind="stable")[:n_voxels]
    coords = np.column_stack(np.unravel_index(nearest, vol_shape))
    return coords.astype(np.int_)


def gaussian_blob(
    vol_shape: tuple[int, int, int],
    center: tuple[float, float, float],
    sigma: float,
) -> NDArray[np.float64]:
    """A unit-peak isotropic Gaussian spatial map — a smooth ICA source map."""
    nx, ny, nz = vol_shape
    xx, yy, zz = np.mgrid[0:nx, 0:ny, 0:nz].astype(np.float64)
    cx, cy, cz = center
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma**2))
