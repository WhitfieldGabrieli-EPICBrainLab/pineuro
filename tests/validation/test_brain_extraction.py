"""Validation — brain extraction and mask erosion.

brain_extraction runs real ANTsPy get_mask (an intensity threshold at a
fraction of the image mean, then morphological cleanup; slow); erode_mask is a
deterministic scipy binary erosion (fast, exact).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from pineuro.preprocessing import brain_extraction, preview_brain_mask
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

    - The threshold (0.5 × the image mean) is *unambiguous*: any threshold
      strictly between 0 and 1000 partitions the volume identically into
      {box} vs {background}.
    - ANTs `get_mask(cleanup=2)`'s morphological cascade (erode-2 →
      largest-component → dilate-2 → fill-holes) on a solid convex box is a
      deterministic opening — it cannot push the mask *outside* the box, and
      removes at most a thin boundary shell.

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


_SHAPE = (48, 48, 36)


def _phantom_mask(tmp_path, image, **params):
    in_path = tmp_path / "mean.nii.gz"
    write_nifti(image, _AFFINE, in_path)
    _, mask_file = brain_extraction(in_path, tmp_path / "be", **params)
    return np.asarray(nib.load(str(mask_file)).dataobj) > 0


@pytest.mark.slow
def test_higher_threshold_fraction_excludes_dim_attached_tissue(tmp_path, record):
    """A dim slab joined to the brain along a whole face is kept at the default
    fraction and excluded by a fraction whose cut-off lies between the two intensities.

    Brain box at 1000 with a 6-voxel slab at 300 sharing one face. The union is a
    single convex box, so cleanup (an opening) cannot detach the slab; only the
    intensity cut-off can. At 0.5 the cut-off (0.5 × the volume mean, about 75)
    is below 300, so the slab's interior, ``erosion(union, 2) ∩ slab``, must be
    in the mask. At ``f = (300 + 1000) / (2 × mean)`` the cut-off is 650: the
    thresholded set is the box alone, and an opening plus hole filling of a box
    stays inside it, so the slab leak is exactly 0.
    """
    image = np.zeros(_SHAPE)
    image[12:36, 12:36, 8:28] = 1000.0
    image[36:42, 12:36, 8:28] = 300.0
    slab = image == 300.0
    mean = image.mean()
    assert 0.5 * mean < 300.0  # precondition: the default cut-off keeps the slab

    kept = _phantom_mask(tmp_path / "default", image)
    union_core = binary_erosion(image > 0, iterations=2)
    assert not np.any(union_core & slab & ~kept)

    excluded = _phantom_mask(tmp_path / "higher", image, threshold_fraction=(300.0 + 1000.0) / (2 * mean))
    slab_leak = int(np.sum(excluded & slab))
    assert (
        record(
            "preprocessing",
            "brain_mask_threshold_slab_leak",
            slab_leak,
            threshold=0.0,
            op="==",
            description=(
                "Voxels of a dim attached slab that survive a brain-mask cut-off set between "
                "the slab and brain intensities. None may survive."
            ),
        )
        == 0.0
    )


@pytest.mark.slow
def test_higher_cleanup_detaches_tissue_on_a_narrow_neck(tmp_path, record):
    """A blob as bright as the brain, joined by a 3×3-voxel neck, is kept at
    cleanup 1 and removed at cleanup 3.

    Brain box 24×24×20, blob 8×8×8, neck 4 voxels long with a 3×3 cross-section,
    all at 1000. Erosion by 1 leaves the neck's centre line, so the blob stays in
    the largest component and its 1-voxel-eroded core must be kept. Erosion by 3
    removes the whole neck (its cross-section is 3 wide), the eroded box is far
    larger than the eroded blob (8 voxels), and dilating the box's eroded core by
    3 reaches at most x = 34 while the blob starts at x = 36. So exactly 0 blob
    voxels remain.
    """
    image = np.zeros(_SHAPE)
    image[8:32, 8:32, 6:26] = 1000.0  # brain
    image[32:36, 17:20, 14:17] = 1000.0  # neck
    image[36:44, 14:22, 12:20] = 1000.0  # blob
    blob = np.zeros(_SHAPE, dtype=bool)
    blob[36:44, 14:22, 12:20] = True

    kept = _phantom_mask(tmp_path / "c1", image, cleanup=1)
    assert not np.any(binary_erosion(blob, iterations=1) & ~kept)

    removed = _phantom_mask(tmp_path / "c3", image, cleanup=3)
    blob_leak = int(np.sum(removed & blob))
    assert (
        record(
            "preprocessing",
            "brain_mask_cleanup_blob_leak",
            blob_leak,
            threshold=0.0,
            op="==",
            description=(
                "Voxels of a bright blob joined to the brain by a 3-voxel-wide neck that survive "
                "cleanup of radius 3. The neck cannot survive that erosion, so none may."
            ),
        )
        == 0.0
    )


@pytest.mark.slow
def test_preview_matches_committed_mask(tmp_path):
    """The review preview is exactly the eroded mask preprocessing commits."""
    image, _ = bright_box(_SHAPE, intensity=1000.0, smooth_sigma=1.5)
    in_path = tmp_path / "mean.nii.gz"
    write_nifti(image, _AFFINE, in_path)

    _, mask_file = brain_extraction(in_path, tmp_path / "be", threshold_fraction=0.8, cleanup=3)
    committed = tmp_path / "eroded.nii.gz"
    erode_mask(mask_file, committed, n_erode=2)

    preview = preview_brain_mask(in_path, threshold_fraction=0.8, cleanup=3)
    np.testing.assert_array_equal(preview > 0, np.asarray(nib.load(str(committed)).dataobj) > 0)
