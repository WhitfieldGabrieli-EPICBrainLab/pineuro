"""Self-tests for the phantom synthetic-data library.

These verify the phantom generator itself (not pineuro): that what it claims to
inject is actually injected, so the rest of the validation suite can trust its
ground truth. Marked ``validation`` so they live with the rest of the suite.
"""

from __future__ import annotations

import json
import re

import nibabel as nib
import numpy as np
import pytest

from .metrics import relative_error
from .phantom import (
    PhantomSpec,
    ROISpec,
    apply_known_motion,
    build_phantom,
    double_gamma_hrf,
    gaussian_blob,
    load_ground_truth,
    motion_image_comment,
    random_motion_sequence,
    save_ground_truth,
    write_per_volume_niftis,
)

pytestmark = pytest.mark.validation

# pineuro's own motion-parsing regex (utils.py: _MOTION_PATTERN).
_MOTION_RE = re.compile(r"Motion:\s*([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)")


def test_build_phantom_shapes():
    data, gt = build_phantom(PhantomSpec(vol_shape=(32, 32, 24), n_volumes=60))
    assert data.shape == (32, 32, 24, 60)
    assert gt.n_volumes == 60
    assert gt.signal.shape == (60,)
    assert set(gt.rois) == {"roi"}
    assert gt.tissue_labels is not None
    # Everything outside the brain is exactly zero.
    background = gt.tissue_labels == 0
    assert np.all(data[background, :] == 0.0)


def test_roi_carries_shared_signal(record):
    """A phantom region carries the signal it was given."""
    # Noise-free ROI: every clean voxel is exactly baseline + signal, so averaging
    # the ROI voxels recovers the injected signal to machine precision.
    data, gt = build_phantom(PhantomSpec(seed=7, background_noise_sd=0.0, rois=[ROISpec("roi", noise_sd=0.0)]))
    roi = gt.roi("roi")
    ii, jj, kk = roi.coords[:, 0], roi.coords[:, 1], roi.coords[:, 2]
    roi_mean = data[ii, jj, kk, :].mean(axis=0)
    r = float(np.corrcoef(roi_mean, gt.signal)[0, 1])
    err = abs(1.0 - r)
    assert (
        record(
            "phantom",
            "roi_signal_recovery_error",
            err,
            threshold=1e-9,
            op="<",
            description=(
                "The test phantoms are built to carry a chosen signal in a region. "
                "Averaging that region should return the signal; this is how far it "
                "falls short."
            ),
        )
        < 1e-9
    )


def test_signal_sd_matches_target(record):
    """A phantom signal has the requested strength."""
    _, gt = build_phantom(PhantomSpec(signal_kind="events", signal_amplitude=25.0))
    err = relative_error(float(np.std(gt.signal)), 25.0)
    assert (
        record(
            "phantom",
            "signal_sd_rel_error",
            err,
            threshold=1e-9,
            op="<",
            description=(
                "A phantom signal is requested at a set strength. This is the "
                "relative gap between the strength it was built with and the "
                "strength asked for."
            ),
        )
        < 1e-9
    )


def test_ground_truth_roundtrip(tmp_path):
    _, gt = build_phantom(PhantomSpec(vol_shape=(24, 24, 16), n_volumes=40))
    save_ground_truth(gt, tmp_path)
    loaded = load_ground_truth(tmp_path)
    assert loaded.tr == gt.tr
    assert loaded.n_volumes == gt.n_volumes
    assert loaded.vol_shape == gt.vol_shape
    np.testing.assert_array_equal(loaded.signal, gt.signal)
    np.testing.assert_array_equal(loaded.roi("roi").coords, gt.roi("roi").coords)
    np.testing.assert_array_equal(loaded.roi("roi").signal, gt.roi("roi").signal)  # NPZ round-trip is lossless
    # The readable summary is also written.
    assert (tmp_path / "ground_truth.json").exists()


def test_double_gamma_hrf_shape(record):
    """The phantom's HRF has the correct shape."""
    hrf = double_gamma_hrf(tr=1.0)
    sum_error = abs(float(hrf.sum()) - 1.0)
    assert (
        record(
            "phantom",
            "hrf_sum_error",
            sum_error,
            threshold=1e-9,
            op="<",
            description=(
                "The HRF curve used to build the phantoms is normalised, so its "
                "values sum to one. This is how far that sum strays from one."
            ),
        )
        < 1e-9
    )
    # Peak gamma shape 6 has mode (6-1)*scale = 5, so at tr=1 the HRF peaks exactly
    # at the 5 s sample; the undershoot lobe is negative.
    assert int(np.argmax(hrf)) == 5
    assert hrf.min() < 0.0


def test_apply_known_motion_translates_content():
    blob = gaussian_blob((24, 24, 24), center=(10.0, 12.0, 12.0), sigma=2.5)
    data4d = blob[..., None]
    motion = np.zeros((1, 6))
    motion[0, 0] = 6.0  # +6 mm along x
    moved = apply_known_motion(data4d, motion, voxel_size_mm=(3.0, 3.0, 3.0))[..., 0]
    # 6 mm / 3 mm-per-voxel == +2 voxels along x. An exact integer-voxel shift with
    # order-1 interpolation is bit-exact: moved[x] == blob[x - 2], zero-filled below.
    np.testing.assert_array_equal(moved[2:], blob[:-2])
    assert np.all(moved[:2] == 0.0)


def test_random_motion_sequence_reference_is_zero():
    motion = random_motion_sequence(80, max_trans_mm=2.0, max_rot_deg=1.5, seed=3)
    assert motion.shape == (80, 6)
    np.testing.assert_array_equal(motion[0], np.zeros(6))
    assert np.abs(motion[:, :3]).max() <= 4.0  # bounded translations (2x amplitude)
    assert np.abs(motion[:, 3:]).max() <= 3.0  # bounded rotations


def test_motion_image_comment_matches_pineuro_regex():
    values = [0.1, -0.2, 0.3, -0.01, 0.02, -0.03]
    comment = motion_image_comment(np.array(values))
    match = _MOTION_RE.search(comment)
    assert match is not None
    parsed = [float(g) for g in match.groups()]
    # motion_image_comment formats with ":.6f"; these <=2-decimal values round-trip exactly.
    np.testing.assert_array_equal(parsed, values)


def test_write_per_volume_niftis_simulator_layout(tmp_path):
    data, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=12))
    paths = write_per_volume_niftis(data, gt.affine, tmp_path / "vols")
    assert len(paths) == 12
    assert paths == sorted(paths)  # lexical sort == temporal order
    assert not list((tmp_path / "vols").glob("*.json"))  # no sidecars for a simulator source
    first = np.asarray(nib.load(str(paths[0])).dataobj)
    assert first.shape == (16, 16, 12)
    np.testing.assert_array_equal(first, data[..., 0])  # float64 NIfTI round-trip is lossless


def test_write_per_volume_motion_sidecars(tmp_path):
    data, gt = build_phantom(PhantomSpec(vol_shape=(16, 16, 12), n_volumes=8))
    motion = random_motion_sequence(8, seed=1)
    write_per_volume_niftis(data, gt.affine, tmp_path / "vols", tr=gt.tr, motion_params=motion)
    sidecars = sorted((tmp_path / "vols").glob("*.json"))
    assert len(sidecars) == 8
    ref_meta = json.loads(sidecars[0].read_text())
    assert "Reference volume" in ref_meta["ImageComments"]
    vol1_meta = json.loads(sidecars[1].read_text())
    assert vol1_meta["ImageComments"].startswith("Motion:")
    assert vol1_meta["RepetitionTime"] == gt.tr
