"""Validation — brain extraction and mask erosion.

brain_extraction runs real ANTsPy Otsu masking (slow); erode_mask is a
deterministic scipy binary erosion (fast, exact).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from pineuro.preprocessing import brain_extraction
from pineuro.utils import erode_mask

from .phantom import bright_box, write_nifti

pytestmark = pytest.mark.validation

_AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])


@pytest.mark.slow
def test_brain_extraction_recovers_brain(tmp_path, record):
    """Brain extraction selects exactly the brain, no skull or background.

    The phantom is exactly two intensity levels — background 0.0 and a solid
    convex brain box at 1000.0, unsmoothed, well clear of the volume edges. Two
    consequences make the output deterministic:

    - Otsu thresholding is *unambiguous*: any threshold strictly between 0 and
      1000 partitions the volume identically into {box} vs {background}.
    - ANTs `get_mask(cleanup=2)`'s morphological cascade (erode-2 →
      largest-component → dilate-1 → fill-holes) on a solid convex box is a
      deterministic ~1-voxel inward shrink — it cannot push the mask *outside*
      the box, and removes at most a thin boundary shell.

    So a correct brain_extraction yields a mask satisfying the exact sandwich
    ``erosion(brain, 3) ⊆ mask ⊆ brain``. Both violation counts below are
    therefore exactly 0 — not a Dice floor. A regression that picks a wrong
    threshold (grabbing background) trips ``background_leak``; one that drops or
    shrinks the mask trips ``core_missing``.
    """
    image, labels = bright_box((48, 48, 36), intensity=1000.0, smooth_sigma=0.0)
    true_brain = labels > 0
    in_path = tmp_path / "mean.nii.gz"
    write_nifti(image, _AFFINE, in_path)

    _, mask_file = brain_extraction(in_path, tmp_path / "be")
    mask_img = nib.load(str(mask_file))
    mask_raw = np.asarray(mask_img.dataobj)
    mask = mask_raw > 0

    # Deterministic structural contract: a strictly binary mask carrying the
    # input geometry.
    assert set(np.unique(mask_raw)).issubset({0.0, 1.0})
    assert mask_raw.shape == image.shape
    np.testing.assert_allclose(mask_img.affine, _AFFINE, atol=1e-4)

    # Exact: no background voxel is selected (mask ⊆ brain).
    background_leak = int(np.sum(mask & ~true_brain))
    assert (
        record(
            "preprocessing",
            "brain_extraction_background_leak",
            background_leak,
            threshold=0.0,
            op="==",
            description=(
                "Voxels outside the brain that brain extraction wrongly kept. On a "
                "phantom with a sharp brain edge, there must be none."
            ),
        )
        == 0.0
    )

    # Exact: the brain core (everything but a 3-voxel boundary shell) is fully
    # captured (erosion(brain, 3) ⊆ mask).
    core = binary_erosion(true_brain, iterations=3)
    core_missing = int(np.sum(core & ~mask))
    assert (
        record(
            "preprocessing",
            "brain_extraction_core_missing",
            core_missing,
            threshold=0.0,
            op="==",
            description=(
                "Voxels well inside the brain that brain extraction dropped. The "
                "cleanup step may trim a thin edge, but the brain's interior must be "
                "kept whole."
            ),
        )
        == 0.0
    )


def test_erode_removes_known_layers(tmp_path, record):
    """Mask erosion removes the expected number of voxel layers."""
    mask = np.zeros((40, 40, 40), dtype=np.float64)
    mask[8:32, 8:32, 8:32] = 1.0  # a solid 24-voxel cube, clear of the borders
    in_path = tmp_path / "cube.nii.gz"
    write_nifti(mask, _AFFINE, in_path)

    out_path = tmp_path / "eroded.nii.gz"
    erode_mask(in_path, out_path, n_erode=2)
    eroded = np.asarray(nib.load(str(out_path)).dataobj) > 0

    expected_count = 20**3  # 24 - 2*2 == 20 voxels per side
    count_error = abs(int(eroded.sum()) - expected_count)  # an integer voxel-count diff
    assert (
        record(
            "preprocessing",
            "erode_voxel_count_error",
            count_error,
            threshold=0.0,
            op="==",
            description=(
                "Eroding a mask peels a fixed number of voxel layers off its "
                "surface. Peeling two layers off a 24-voxel cube must leave a "
                "20-voxel cube; this checks the count."
            ),
        )
        == 0.0
    )
    assert np.all(mask[eroded] > 0)  # the eroded mask lies strictly inside the original


def test_erode_preserves_affine(tmp_path):
    """erode_mask leaves the NIfTI affine untouched."""
    mask = np.zeros((30, 30, 30), dtype=np.float64)
    mask[10:20, 10:20, 10:20] = 1.0
    affine = np.diag([2.5, 2.5, 3.0, 1.0])
    in_path = tmp_path / "m.nii.gz"
    write_nifti(mask, affine, in_path)

    out_path = tmp_path / "e.nii.gz"
    erode_mask(in_path, out_path, n_erode=1)
    # erode_mask copies the affine; the diag values (2.5, 3.0, 1.0) are float32-exact.
    np.testing.assert_array_equal(nib.load(str(out_path)).affine, affine)
