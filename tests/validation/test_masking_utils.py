"""Validation — masking and registration utilities.

compute_mean and apply_mask are exact numpy operations (fast); coregistration
runs real ANTsPy registration (slow).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import shift as ndimage_shift

from pineuro.utils import apply_mask, apply_xfm, compute_mean, compute_xfm

from .phantom import PhantomSpec, build_phantom, make_tissue_compartments, write_nifti

pytestmark = pytest.mark.validation


def test_compute_mean_exact(tmp_path, record):
    """The temporal mean image matches a direct average."""
    data, gt = build_phantom(PhantomSpec(vol_shape=(20, 20, 14), n_volumes=40))
    in_path = tmp_path / "func.nii.gz"
    write_nifti(data, gt.affine, in_path, tr=gt.tr)
    out_path = tmp_path / "mean.nii.gz"
    compute_mean(in_path, out_path)

    produced = np.asarray(nib.load(str(out_path)).dataobj)
    expected = data.mean(axis=-1)
    max_err = float(np.max(np.abs(produced - expected)))
    assert (
        record(
            "preprocessing",
            "compute_mean_max_error",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "pineuro averages a run of volumes into one mean image. This is the "
                "largest voxel gap from a straightforward average of the same volumes."
            ),
        )
        < 1e-9
    )


def test_apply_mask_zeros_outside(tmp_path):
    """apply_mask leaves in-mask voxels unchanged and zeros everything else."""
    data, gt = build_phantom(PhantomSpec(vol_shape=(20, 20, 14), n_volumes=10))
    mask = gt.roi("roi").mask(gt.vol_shape)
    in_path, mask_path, out_path = (tmp_path / n for n in ("func.nii.gz", "mask.nii.gz", "masked.nii.gz"))
    write_nifti(data, gt.affine, in_path, tr=gt.tr)
    write_nifti(mask, gt.affine, mask_path)
    apply_mask(in_path, mask_path, out_path)

    masked = np.asarray(nib.load(str(out_path)).dataobj)
    inside = mask > 0
    assert np.all(masked[~inside, :] == 0.0)
    np.testing.assert_array_equal(masked[inside, :], data[inside, :])  # in-mask voxels: data * 1.0


def test_apply_mask_broadcasts_3d_over_4d(tmp_path):
    """A 3D mask is applied identically to every volume of a 4D timeseries."""
    data, gt = build_phantom(PhantomSpec(vol_shape=(18, 18, 12), n_volumes=8))
    mask = gt.brain_mask()
    in_path, mask_path, out_path = (tmp_path / n for n in ("func.nii.gz", "mask.nii.gz", "masked.nii.gz"))
    write_nifti(data, gt.affine, in_path, tr=gt.tr)
    write_nifti(mask, gt.affine, mask_path)
    apply_mask(in_path, mask_path, out_path)

    masked = np.asarray(nib.load(str(out_path)).dataobj)
    for t in range(data.shape[3]):
        np.testing.assert_array_equal(masked[..., t], data[..., t] * (mask > 0))


@pytest.mark.slow
def test_apply_xfm_identity_is_exact(tmp_path, record):
    """A do-nothing spatial transform leaves the image unchanged.

    Resampling onto the same grid with an identity transform samples every
    output voxel at the exact location of the corresponding input voxel, so a
    correct apply_xfm is a perfect no-op -- zero interpolation error. This is
    the deterministic, pineuro-owned transform-application path, distinct from
    the iterative optimiser inside compute_xfm.
    """
    import ants

    intensity, _ = make_tissue_compartments((40, 40, 28))
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    img_path = tmp_path / "img.nii.gz"
    write_nifti(intensity, affine, img_path)

    identity = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    xfm_path = str(tmp_path / "identity.mat")
    ants.write_transform(identity, xfm_path)

    out_path = tmp_path / "warped.nii.gz"
    apply_xfm(img_path, img_path, [xfm_path], out_path, interpolator="nearestNeighbor")
    warped = np.asarray(nib.load(str(out_path)).dataobj)

    max_err = float(np.max(np.abs(warped - intensity.astype(warped.dtype))))
    assert (
        record(
            "preprocessing",
            "apply_xfm_identity_error",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "Applying a transform that moves nothing should return the image "
                "untouched. This is the largest change to any voxel; it must be zero."
            ),
        )
        == 0.0
    )


@pytest.mark.slow
def test_apply_xfm_integer_translation_is_exact(tmp_path, record):
    """A whole-voxel shift relocates the image without blurring it.

    A translation of exactly one voxel (3 mm at 3 mm isotropic) maps every
    output grid point onto an exact input grid point, so nearest-neighbour
    resampling relocates voxel values with zero interpolation error -- the
    output equals the input shifted by an exact integer offset.
    """
    import ants

    intensity, _ = make_tissue_compartments((40, 40, 28))
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    img_path = tmp_path / "img.nii.gz"
    write_nifti(intensity, affine, img_path)

    # 12-param affine: identity 3x3 matrix + a 3 mm (= 1 voxel) x-translation.
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 3.0, 0.0, 0.0], dtype=np.float64))
    xfm_path = str(tmp_path / "translate.mat")
    ants.write_transform(tx, xfm_path)

    out_path = tmp_path / "warped.nii.gz"
    apply_xfm(img_path, img_path, [xfm_path], out_path, interpolator="nearestNeighbor")
    warped = np.asarray(nib.load(str(out_path)).dataobj)

    # A +3 mm-x transform relocates content by exactly +1 voxel along x
    # (output[v] == input[v - 1], zero-filled at the leading face). order=0 is
    # nearest-neighbour, so this is a pure relocation -- max error is exactly 0,
    # confirming apply_xfm introduced no interpolation blur.
    expected = ndimage_shift(intensity, shift=(1.0, 0.0, 0.0), order=0, mode="constant", cval=0.0)
    max_err = float(np.max(np.abs(warped - expected.astype(warped.dtype))))
    assert (
        record(
            "preprocessing",
            "apply_xfm_translation_error",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "Shifting an image by exactly one voxel should move every value "
                "over cleanly, with no blurring. This compares the result against a "
                "plain array shift."
            ),
        )
        == 0.0
    )


@pytest.mark.slow
def test_compute_xfm_output_structure(tmp_path):
    """compute_xfm returns a warped image on the fixed grid plus transform files.

    Exercises the pineuro wrapper plumbing: regardless of where the ANTs
    optimiser converges, compute_xfm must return a warped image carrying the
    fixed image's geometry and at least one transform file that exists on disk.
    (The optimiser's recovery accuracy is ANTs' concern, not validated here.)
    """
    intensity, _ = make_tissue_compartments((40, 40, 28))
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    fixed_path = tmp_path / "fixed.nii.gz"
    moved_path = tmp_path / "moved.nii.gz"
    write_nifti(intensity, affine, fixed_path)
    write_nifti(ndimage_shift(intensity, shift=(2.0, -1.5, 1.0), order=1), affine, moved_path)

    warped_path, transform_files = compute_xfm(
        moved_path, fixed_path, tmp_path / "xfm" / "moved2fixed", type_of_transform="Rigid"
    )
    warped_img = nib.load(str(warped_path))
    assert warped_img.shape == intensity.shape
    np.testing.assert_allclose(warped_img.affine, affine, atol=1e-4)
    assert len(transform_files) >= 1
    assert all(Path(f).exists() for f in transform_files)
