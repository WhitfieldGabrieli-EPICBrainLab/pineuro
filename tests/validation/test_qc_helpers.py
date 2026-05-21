"""Validation -- qc.py helper math (summarize_mask, _pick_slices).

qc.py is mostly plot/report generation, but two helpers compute concrete numbers
that downstream reports depend on:

- ``summarize_mask(mask_path, name)`` returns a ``MaskSummary`` carrying:
    - voxels: int count of non-zero voxels (a sum)
    - volume_ml: voxels × voxel_x · voxel_y · voxel_z / 1000   (mL)
    - centroid_ijk: mean of (i, j, k) coordinates over the in-mask voxels

- ``_pick_slices(n_slices, count)`` picks ``count`` evenly-spaced indices in
  ``[0, n_slices - 1]``, or returns ``list(range(n_slices))`` if there are
  fewer slices than ``count``.

Both functions are pure-deterministic given their inputs, so every expected
value below is derivable by hand from the mask geometry / index arithmetic.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pineuro.qc import _pick_slices, summarize_mask

pytestmark = pytest.mark.validation


def _write_mask(path, data, voxel_dims=(2.0, 2.0, 2.0)):
    """Write a binary mask NIfTI with a specified diagonal affine."""
    affine = np.diag([voxel_dims[0], voxel_dims[1], voxel_dims[2], 1.0])
    img = nib.Nifti1Image(data.astype(np.float64), affine)
    nib.save(img, str(path))
    return path


# =============================================================================
# summarize_mask
# =============================================================================


def test_summarize_mask_voxel_count(tmp_path, record):
    """A mask's voxel count equals the number of marked voxels.

    A 4×4×4 inner cube inside a 10×10×10 volume has exactly 64 non-zero voxels.
    """
    mask = np.zeros((10, 10, 10), dtype=np.float64)
    mask[3:7, 3:7, 3:7] = 1.0
    expected_voxels = 4 * 4 * 4  # = 64
    path = _write_mask(tmp_path / "cube.nii.gz", mask)

    summary = summarize_mask(path, name="cube")
    err = abs(summary.voxels - expected_voxels)
    assert (
        record(
            "qc_helpers",
            "summarize_mask_voxel_count_error",
            err,
            threshold=0.0,
            op="==",
            description=(
                "A region's reported voxel count, compared against a cube with a "
                "known 64 voxels. The two must match exactly."
            ),
        )
        == 0.0
    )


def test_summarize_mask_volume_ml_at_2mm_isotropic(tmp_path, record):
    """A mask's volume in millilitres at 2 mm voxels."""
    mask = np.zeros((10, 10, 10), dtype=np.float64)
    mask[3:7, 3:7, 3:7] = 1.0
    path = _write_mask(tmp_path / "cube.nii.gz", mask, voxel_dims=(2.0, 2.0, 2.0))

    summary = summarize_mask(path, name="cube")
    expected_volume_ml = 64.0 * (2.0 * 2.0 * 2.0) / 1000.0  # = 0.512 mL
    err = abs(summary.volume_ml - expected_volume_ml)
    assert (
        record(
            "qc_helpers",
            "summarize_mask_volume_ml_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "A region's volume in millilitres, compared against a hand "
                "calculation: 64 voxels at 2 mm on a side come to 0.512 mL."
            ),
        )
        < 1e-12
    )


def test_summarize_mask_volume_ml_at_3mm_isotropic(tmp_path, record):
    """A mask's volume scales with voxel size."""
    mask = np.zeros((10, 10, 10), dtype=np.float64)
    mask[3:7, 3:7, 3:7] = 1.0
    path = _write_mask(tmp_path / "cube.nii.gz", mask, voxel_dims=(3.0, 3.0, 3.0))

    summary = summarize_mask(path, name="cube")
    expected_volume_ml = 64.0 * (3.0 * 3.0 * 3.0) / 1000.0  # = 1.728 mL
    err = abs(summary.volume_ml - expected_volume_ml)
    assert (
        record(
            "qc_helpers",
            "summarize_mask_volume_ml_3mm_error",
            err,
            threshold=1e-12,
            op="<",
            description=(
                "The same region measured with larger 3 mm voxels. Its reported "
                "volume should grow with voxel size, to 1.728 mL."
            ),
        )
        < 1e-12
    )


def test_summarize_mask_centroid_of_symmetric_cube(tmp_path, record):
    """The centroid of a symmetric mask is exactly the centre of its bounding box.

    A cube spanning indices [3, 6] inclusive on each axis has centre at 4.5.
    summarize_mask rounds the centroid to 2 decimal places, so the recorded
    centroid is exactly (4.5, 4.5, 4.5).
    """
    mask = np.zeros((10, 10, 10), dtype=np.float64)
    mask[3:7, 3:7, 3:7] = 1.0  # inclusive [3, 6] on each axis -> mean = 4.5
    path = _write_mask(tmp_path / "cube.nii.gz", mask)

    summary = summarize_mask(path, name="cube")
    expected = (4.5, 4.5, 4.5)
    max_err = max(abs(c - e) for c, e in zip(summary.centroid_ijk, expected, strict=True))
    assert (
        record(
            "qc_helpers",
            "summarize_mask_centroid_max_error",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "The reported centre of a symmetric cube, compared against its true "
                "geometric centre. The largest coordinate difference between them."
            ),
        )
        == 0.0
    )


def test_summarize_mask_centroid_of_asymmetric_mask(tmp_path, record):
    """A non-symmetric mask's centroid equals its hand-computed mean coordinate.

    Two single-voxel points at (0,0,0) and (4,4,4) -> centroid (2,2,2) exactly.
    """
    mask = np.zeros((6, 6, 6), dtype=np.float64)
    mask[0, 0, 0] = 1.0
    mask[4, 4, 4] = 1.0
    path = _write_mask(tmp_path / "two.nii.gz", mask)

    summary = summarize_mask(path, name="two")
    expected = (2.0, 2.0, 2.0)
    max_err = max(abs(c - e) for c, e in zip(summary.centroid_ijk, expected, strict=True))
    assert (
        record(
            "qc_helpers",
            "summarize_mask_centroid_asymmetric_error",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "The reported centre of an off-centre region, compared against its hand-computed average position."
            ),
        )
        == 0.0
    )


def test_summarize_mask_empty_mask_returns_nan_centroid(tmp_path):
    """Empty mask -> 0 voxels, 0 mL, NaN centroid (contract)."""
    mask = np.zeros((10, 10, 10), dtype=np.float64)
    path = _write_mask(tmp_path / "empty.nii.gz", mask)

    summary = summarize_mask(path, name="empty")
    assert summary.voxels == 0
    assert summary.volume_ml == 0.0
    assert all(np.isnan(c) for c in summary.centroid_ijk)


def test_summarize_mask_passes_through_metadata(tmp_path):
    """name, chosen_component, correlation flow through unchanged (contract)."""
    mask = np.zeros((5, 5, 5), dtype=np.float64)
    mask[2, 2, 2] = 1.0
    path = _write_mask(tmp_path / "single.nii.gz", mask)

    summary = summarize_mask(path, name="DefaultA", chosen_component=3, correlation=0.42)
    assert summary.name == "DefaultA"
    assert summary.chosen_component == 3
    assert summary.correlation == 0.42


# =============================================================================
# _pick_slices
# =============================================================================


def test_pick_slices_returns_all_when_fewer_than_count():
    """If n_slices <= count, return list(range(n_slices)) verbatim."""
    assert _pick_slices(3, count=6) == [0, 1, 2]
    assert _pick_slices(6, count=6) == [0, 1, 2, 3, 4, 5]
    assert _pick_slices(0, count=6) == []


def test_pick_slices_evenly_spaced(record):
    """Slice selection picks evenly-spaced slices for QC figures.

    The hand-computed expected list uses np.linspace(0, 99, 5).round() ==
    [0, 24.75, 49.5, 74.25, 99] -> round-half-to-even gives [0, 25, 50, 74, 99].
    """
    produced = _pick_slices(100, count=5)
    expected = [0, 25, 50, 74, 99]
    assert produced == expected
    max_diff = float(max(abs(a - b) for a, b in zip(produced, expected, strict=True)))
    assert (
        record(
            "qc_helpers",
            "pick_slices_max_diff",
            max_diff,
            threshold=0.0,
            op="==",
            description=(
                "QC figures show a handful of evenly-spaced slices. The slices "
                "chosen are compared against a hand-picked even spacing."
            ),
        )
        == 0.0
    )


def test_pick_slices_bounds_inclusive():
    """First and last indices are always 0 and n_slices - 1 (boundary contract)."""
    for n in (10, 50, 100, 200):
        for count in (3, 5, 8):
            picked = _pick_slices(n, count=count)
            assert picked[0] == 0
            assert picked[-1] == n - 1
            assert len(picked) == count


def test_pick_slices_indices_strictly_increasing():
    """Picked slice indices are monotonically non-decreasing."""
    picked = _pick_slices(100, count=6)
    # picked[1:] is one shorter than picked, so strict=False is appropriate here.
    assert all(a <= b for a, b in zip(picked, picked[1:], strict=False))
