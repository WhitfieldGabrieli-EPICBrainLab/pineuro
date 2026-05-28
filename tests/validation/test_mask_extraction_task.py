"""Validation — task-based mask extraction (real nilearn FirstLevelModel, slow).

A synthetic localizer phantom carries a known HRF-convolved block signal in one
ROI; mask_extraction_task fits a whole-brain GLM and its top-N threshold should
localize that ROI.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pineuro.mask_extraction import mask_extraction_task

from .metrics import dice
from .phantom import PhantomSpec, ROISpec, boxcar_blocks, build_phantom, write_nifti

pytestmark = [pytest.mark.validation, pytest.mark.slow]


def test_task_mask_recovers_activation_roi(tmp_path, record):
    """Task-based mask extraction recovers a known activation region.

    With spatial smoothing disabled there is no signal bleed across the ROI
    boundary, so the GLM ranks every ROI voxel above every non-ROI voxel and the
    top_n mask equals the true ROI set -- Dice is exactly 1.0. Also checks that
    top_n selects exactly the requested voxel count.
    """
    tr, n_volumes = 1.0, 120
    blocks = [(12, 24), (36, 48), (60, 72), (84, 96), (108, 120)]
    block_signal = boxcar_blocks(n_volumes, blocks, amplitude=1.0, tr=tr, convolve=True)

    roi_voxels = 80
    spec = PhantomSpec(
        vol_shape=(40, 40, 28),
        n_volumes=n_volumes,
        n_baseline=0,
        tr=tr,
        rois=[ROISpec("target", n_voxels=roi_voxels, signal=block_signal, beta=60.0, noise_sd=5.0)],
        seed=7,
    )
    data, gt = build_phantom(spec)

    func_path = tmp_path / "func.nii.gz"
    mask_path = tmp_path / "brain_mask.nii.gz"
    write_nifti(data, gt.affine, func_path, tr=tr)
    write_nifti(gt.brain_mask(), gt.affine, mask_path)

    events = pd.DataFrame(
        {
            "onset": [start * tr for start, _ in blocks],
            "duration": [(stop - start) * tr for start, stop in blocks],
            "trial_type": ["task"] * len(blocks),
        }
    )

    mask_files = mask_extraction_task(
        participant_id="sub-val",
        session_id="ses-01",
        root_dir=tmp_path / "mask_extraction",
        input_file=func_path,
        mask_file=mask_path,
        events=events,
        contrasts={"target": "task"},
        tr=tr,
        threshold=roi_voxels,
        threshold_method="top_n",
        smoothing_fwhm=None,  # no spatial bleed -> top_n mask matches the ROI exactly
        qc_enabled=False,
    )
    assert len(mask_files) == 1
    mask = np.asarray(nib.load(str(mask_files[0])).dataobj) > 0

    assert int(mask.sum()) == roi_voxels  # top_n selects exactly `threshold` voxels
    overlap = dice(mask, gt.roi("target").mask(gt.vol_shape))
    assert (
        record(
            "mask_extraction",
            "task_mask_dice",
            overlap,
            threshold=1.0,
            op="==",
            description=(
                "A phantom activates a known region during a task. The mask pineuro "
                "extracts should match that region exactly; a Dice overlap of 1.0 "
                "means a perfect match."
            ),
        )
        == 1.0
    )
