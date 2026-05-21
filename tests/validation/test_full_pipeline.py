"""Validation — crown jewel: the stages compose end to end.

A synthetic phantom with a known ROI signal is corrupted with known motion,
then pushed through the real pipeline — ANTs motion correction, temporal mean,
ANTs brain extraction — and finally the incremental-GLM feedback estimator.

Two things are asserted, both **exact**:

1. **Composition** — every stage's output carries the expected grid geometry
   (shape + affine), so the next stage receives well-formed input.
2. **Feedback correctness** — the estimator's per-volume output matches an
   independent SVD/lstsq reference computed on the *same* motion-corrected 4D
   array the estimator consumed, to 1e-9.

ANTs motion-correction recovery accuracy is deliberately NOT asserted: it is the
optimizer's concern, not pineuro's. By comparing the feedback output to a
reference computed on the *actual* motion-corrected data (rather than the
originally-injected signal), the feedback GLM is validated exactly, decoupled
from how precisely ANTs converged. Brain-extraction correctness is covered
exactly by `test_brain_extraction` on a clean 2-level phantom.

This starts from NIfTI (not synthetic DICOM): per the plan's realism analysis,
dcm2niix is too strict for robust synthetic-DICOM generation, and every
pipeline stage downstream of DICOM conversion operates on NIfTI.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pineuro.feedback import FeedbackEstimator
from pineuro.preprocessing import brain_extraction, motion_correction
from pineuro.utils import compute_mean

from .phantom import (
    PhantomSpec,
    ROISpec,
    apply_known_motion,
    build_phantom,
    random_motion_sequence,
    write_nifti,
)
from .references import _reference_inverse_sd_timeseries

pytestmark = [pytest.mark.validation, pytest.mark.slow]

_VOXEL_MM = (3.0, 3.0, 3.0)


def test_pipeline_composes_and_recovers_signal(tmp_path, record):
    """The full pipeline runs end to end and the feedback result is exact.

    A phantom with a known ROI signal is corrupted with known motion, then run
    through motion correction → temporal mean → brain extraction → feedback.

    (1) Composition is exact: every stage's output carries the expected shape
        and affine, and brain extraction returns a strictly binary mask.
    (2) Feedback correctness is exact: the estimator output matches an
        independent ``np.linalg.lstsq`` (LAPACK SVD) reference computed on the
        SAME motion-corrected 4D array the estimator consumed — to 1e-9. This
        tests "given whatever data motion correction produced, does the feedback
        GLM compute correctly?" without depending on ANTs' recovery accuracy.
    """
    # noise_sd > 0 is required: the inverse-SD ROI combine weights voxels by
    # 1/sqrt(rss), so a noise-free phantom (near-zero per-voxel residual after
    # the intercept-only fit) makes those weights numerically ill-conditioned
    # and amplifies the tiny QR-vs-SVD difference. Moderate noise keeps the
    # residual variance non-degenerate; pineuro and the reference are still
    # compared on the *same* data, so the test stays exact (same rationale as
    # test_inverse_sd_matches_batch_reference).
    spec = PhantomSpec(
        vol_shape=(48, 48, 32),
        n_volumes=30,
        n_baseline=8,
        tr=1.0,
        seed=21,
        rois=[ROISpec("target", n_voxels=120, beta=1.0, noise_sd=6.0)],
    )
    static, gt = build_phantom(spec)

    # --- Corrupt every volume with known head motion ---
    motion = random_motion_sequence(gt.n_volumes, max_trans_mm=2.0, max_rot_deg=1.0, seed=2)
    corrupted = apply_known_motion(static, motion, voxel_size_mm=_VOXEL_MM)
    raw_path = tmp_path / "raw.nii.gz"
    write_nifti(corrupted, gt.affine, raw_path, tr=gt.tr)

    # --- Preprocessing: motion correction → temporal mean → brain extraction ---
    mcf_path = tmp_path / "mcf.nii.gz"
    motion_correction(raw_path, mcf_path)
    mean_path = tmp_path / "mean.nii.gz"
    compute_mean(mcf_path, mean_path)
    _, brain_mask_path = brain_extraction(mean_path, tmp_path / "be")

    # --- (1) Composition: every stage preserves the grid geometry exactly ---
    mcf_img = nib.load(str(mcf_path))
    mean_img = nib.load(str(mean_path))
    mask_img = nib.load(str(brain_mask_path))
    assert mcf_img.shape == (*gt.vol_shape, gt.n_volumes)
    assert mean_img.shape == gt.vol_shape
    assert mask_img.shape == gt.vol_shape
    for img in (mcf_img, mean_img, mask_img):
        np.testing.assert_allclose(img.affine, gt.affine, atol=1e-4)
    assert set(np.unique(np.asarray(mask_img.dataobj))).issubset({0.0, 1.0})

    # --- (2) Feedback correctness: estimator output == SVD reference, exact ---
    roi_mask = gt.roi("target").mask(gt.vol_shape)
    roi_mask_path = tmp_path / "roi_mask.nii.gz"
    write_nifti(roi_mask, gt.affine, roi_mask_path)
    corrected = np.asarray(nib.load(str(mcf_path)).dataobj)

    estimator = FeedbackEstimator(
        [roi_mask_path],
        n_baseline=gt.n_baseline,
        n_volumes=gt.n_volumes,
        tr=gt.tr,
        mask_names=["target"],
        motion_correction="none",
        include_drift=False,
        include_motion_regressors=False,
        roi_combine_method="inverse_sd",
    )
    for t in range(gt.n_volumes):
        estimator.process_volume("volume.nii", volume_data=corrected[..., t], volume_affine=gt.affine)
    pineuro_out = np.array(estimator.timeseries["target"], dtype=np.float64)

    # Reference computed on the EXACT data the estimator received (corrected),
    # so the comparison isolates the feedback GLM from motion-correction accuracy.
    reference = _reference_inverse_sd_timeseries(corrected, roi_mask, gt.n_baseline)

    nb = gt.n_baseline
    max_err = float(np.max(np.abs(pineuro_out[nb:] - reference[nb:])))
    assert (
        record(
            "pipeline",
            "end_to_end_glm_vs_reference_max_err",
            max_err,
            threshold=1e-9,
            op="<",
            description=(
                "A phantom is run through the whole pipeline — motion correction, "
                "averaging, brain extraction, then feedback. The feedback result is "
                "compared against a batch reference on the same data."
            ),
        )
        < 1e-9
    )
