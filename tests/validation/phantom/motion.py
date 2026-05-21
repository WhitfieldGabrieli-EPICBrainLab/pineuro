"""Known rigid head motion for the phantom library.

Generates a known per-volume rigid-motion sequence, applies it to a 4D phantom
by resampling, and formats the dcm2niix-style JSON ``ImageComments`` string
that ``FeedbackEstimator._extract_motion_params_from_json`` parses.

Motion convention — matches pineuro's feedback path exactly:
``[trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]``, translations in mm,
rotations in degrees. Volume 0 is always the zero reference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import affine_transform

MOTION_CONVENTION = "trans_mm_xyz__rot_deg_xyz"
N_MOTION_PARAMS = 6


def random_motion_sequence(
    n_volumes: int,
    *,
    max_trans_mm: float = 2.0,
    max_rot_deg: float = 1.5,
    seed: int = 0,
) -> NDArray[np.float64]:
    """A smooth, bounded per-volume rigid-motion sequence, shape ``(n_volumes, 6)``.

    Each parameter follows a low-frequency sinusoid (random amplitude, period
    and phase), giving realistic temporally-correlated drift that registration
    can plausibly recover. Volume 0 is shifted to exactly zero (the reference).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_volumes, dtype=np.float64)
    motion = np.zeros((n_volumes, N_MOTION_PARAMS), dtype=np.float64)
    for col in range(N_MOTION_PARAMS):
        max_amp = max_trans_mm if col < 3 else max_rot_deg
        amplitude = max_amp * rng.uniform(0.4, 1.0)
        period = rng.uniform(n_volumes / 4.0, n_volumes / 1.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        motion[:, col] = amplitude * np.sin(2.0 * np.pi * t / period + phase)
    motion -= motion[0]  # volume 0 is the zero reference
    return motion


def _euler_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> NDArray[np.float64]:
    """Rotation matrix R = Rz @ Ry @ Rx from intrinsic Euler angles (degrees)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def apply_known_motion(
    data4d: NDArray[np.float64],
    motion_params: NDArray[np.float64],
    voxel_size_mm: NDArray[np.float64] | tuple[float, float, float],
    *,
    order: int = 1,
    cval: float = 0.0,
) -> NDArray[np.float64]:
    """Resample each volume of ``data4d`` by its known rigid transform.

    Volume ``t`` is displaced by ``motion_params[t]`` (rotation about the volume
    centre, then translation). Translations are given in mm and converted to
    voxels via ``voxel_size_mm``; rotations are in degrees.

    Returns a new 4D array of the same shape.
    """
    nx, ny, nz, nt = data4d.shape
    voxel_size = np.asarray(voxel_size_mm, dtype=np.float64)
    center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0])
    out = np.empty_like(data4d, dtype=np.float64)

    for t in range(nt):
        tx, ty, tz, rx, ry, rz = motion_params[t]
        rotation = _euler_to_matrix(rx, ry, rz)
        trans_vox = np.array([tx, ty, tz]) / voxel_size
        # affine_transform fills output[o] from input[matrix @ o + offset];
        # we want output(R(p-c)+c+d) = input(p), hence matrix = R^-1 = R^T.
        matrix = rotation.T
        offset = center - matrix @ (center + trans_vox)
        out[..., t] = affine_transform(data4d[..., t].astype(np.float64), matrix, offset=offset, order=order, cval=cval)
    return out


def motion_image_comment(motion_row: NDArray[np.float64]) -> str:
    """Format a 6-vector as a dcm2niix-style ``ImageComments`` motion string.

    Uses fixed-point formatting (never scientific notation) so the values match
    pineuro's ``_MOTION_PATTERN`` regex (``Motion:\\s*([-\\d.]+),...``).
    """
    return "Motion: " + ",".join(f"{float(v):.6f}" for v in motion_row)
