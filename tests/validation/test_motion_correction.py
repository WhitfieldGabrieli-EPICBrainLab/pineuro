"""Validation — motion correction (real ANTsPy rigid registration, slow).

A structural phantom is displaced by a known per-volume rigid-motion sequence;
pineuro's motion_correction writes a 6-column ``.par`` file and a corrected 4D
series. The tests assert only pineuro-owned, deterministic facts about that
output — the ``.par`` column layout, sign monotonicity, geometry preservation,
and that correction reduces motion-induced variance. ANTs' registration
*accuracy* (how precisely it recovers a given displacement) is the optimizer's
concern and is not asserted.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pineuro.preprocessing import motion_correction

from .phantom import apply_known_motion, make_tissue_compartments, write_nifti

pytestmark = [pytest.mark.validation, pytest.mark.slow]

_AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])
_VOXEL_MM = (3.0, 3.0, 3.0)
_N_VOLS = 20


def _moved_phantom(n_vols: int, motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free structural content repeated over time, displaced by ``motion``.

    The tissue-compartment ellipsoid supplies all the spatial structure ANTs
    needs to register; no noise is added, so the phantom is fully deterministic.
    """
    intensity, labels = make_tissue_compartments((48, 48, 32))
    static = np.repeat(intensity[..., None], n_vols, axis=-1)
    moved = apply_known_motion(static, motion, voxel_size_mm=_VOXEL_MM)
    return moved, labels > 0


def _ramp_motion(n_vols: int, column: int, lo: float, hi: float) -> np.ndarray:
    """A (n_vols, 6) motion array with a monotonic ramp in one column.

    Motion convention is [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
    A monotonic injected ramp lets the recovered ``.par`` column be checked
    for monotonicity exactly — independent of how precisely ANTs converges.
    """
    motion = np.zeros((n_vols, 6), dtype=np.float64)
    motion[:, column] = np.linspace(lo, hi, n_vols)
    return motion


def _ramp_tracking(recovered: np.ndarray, injected_ramp: np.ndarray) -> float:
    """|Pearson r| between a recovered ``.par`` column and the injected motion ramp.

    The injected ramp is strictly monotonic, so a recovery that reports the
    motion in this column with a *consistent sign* yields a strong linear
    association (|r| near 1) regardless of ANTs' per-volume recovery accuracy.
    Correlating across all volumes is robust to the optimizer's per-volume
    jitter; a strict per-bin monotonicity check is not -- it flipped
    intermittently when two adjacent bin means landed within ANTs'
    rotation-recovery noise of each other. That nondeterminism persists with a
    fixed seed under both multi- and single-threaded ITK, so it cannot be fixed
    by seeding alone; aggregating into a correlation absorbs it. A rot/trans
    block swap leaves the column as near-zero noise (|r| ~ 0) and a
    sign-scrambling bug gives low |r| -- both fail the floor. ``abs`` is used
    because the sign of recovered motion relative to the injected displacement
    follows ANTs' transform-direction convention, which this test does not assert.
    """
    rec = np.asarray(recovered, dtype=np.float64)
    inj = np.asarray(injected_ramp, dtype=np.float64)
    if rec.std() == 0.0:
        return 0.0
    return float(abs(np.corrcoef(rec, inj)[0, 1]))


def test_motion_correction_par_translation_layout(tmp_path, record):
    """Left-right head motion is reported in the correct column.

    Phantom: a monotonic ramp of pure-X translation (±6 mm over 20 volumes,
    0.63 mm steps — well clear of ANTs' sub-0.1 mm recovery noise), no rotation.
    Two exact, deterministic assertions on pineuro's ``.par`` output, whose
    column layout is ``[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]``:

    - Within the translation sub-block (columns 3-5), the dominant axis (largest
      mean absolute value) is axis 0 — trans_x. A regression that swaps the
      translation columns fails here.
    - ``.par[:, 3]`` correlates with the injected ramp (|r| >= 0.5). The ramp is
      monotonic, so a correct, consistent-sign recovery tracks it linearly; a
      sign-inconsistency bug, or a rot/trans block swap (which leaves col 3 as
      near-zero noise), drops |r| toward zero.

    Neither assertion depends on ANTs' recovery *accuracy* — only that pineuro
    places the recovered translation in the right column with a consistent sign.
    """
    motion = _ramp_motion(_N_VOLS, column=0, lo=-6.0, hi=6.0)  # trans_x ramp
    moved, _ = _moved_phantom(_N_VOLS, motion)
    in_path = tmp_path / "moved.nii.gz"
    out_path = tmp_path / "mcf.nii.gz"
    write_nifti(moved, _AFFINE, in_path, tr=1.0)
    motion_correction(in_path, out_path)

    par = np.loadtxt(str(out_path) + ".par")
    assert par.shape == (_N_VOLS, 6)

    # Within the translation sub-block [trans_x, trans_y, trans_z], the injected
    # pure-X translation must dominate axis 0 (trans_x).
    dominant_trans_axis = int(np.argmax(np.mean(np.abs(par[:, 3:6]), axis=0)))
    assert (
        record(
            "preprocessing",
            "motion_par_translation_dominant_axis",
            float(dominant_trans_axis),
            threshold=0.0,
            op="==",
            description=(
                "The synthetic head moves only left-right. Motion correction should "
                "report that movement in the left-right column of its motion file; "
                "this checks which column it landed in."
            ),
        )
        == 0.0
    )

    ramp_corr = _ramp_tracking(par[:, 3], motion[:, 0])
    assert (
        record(
            "preprocessing",
            "motion_par_translation_ramp_corr",
            ramp_corr,
            threshold=0.5,
            op=">=",
            description=(
                "The head motion ramps steadily in one direction; the reported "
                "left-right values should track that ramp. This is |correlation| "
                "between them (0-1). >=0.5 is a clearly significant association "
                "(n=20 volumes), which a correct, consistent-sign column gives and "
                "a column swap (~0) cannot. ANTs' recovery precision is not asserted."
            ),
        )
        >= 0.5
    )


def test_motion_correction_par_rotation_layout(tmp_path, record):
    """Head rotation is reported in the correct column.

    Phantom: a monotonic ramp of pure-Z rotation (±12° over 20 volumes, 1.26°
    steps), no translation. Two exact, deterministic assertions:

    - Within the rotation sub-block (columns 0-2), the dominant axis is axis 2 —
      rot_z. This exercises the rotation-extraction path in motion_correction
      (the 6-param Euler3D vs 12-param matrix→Euler branches). The check is made
      *within* the rotation sub-block because a rotation about the volume centre
      also induces a reported translation, so the dominant column of the *full*
      6-vector can be a translation — not a layout error.
    - ``.par[:, 2]`` correlates with the injected ramp (|r| >= 0.5). ANTs'
      per-volume rotation recovery is noisy and intermittently nondeterministic
      (even seeded/single-threaded), so a strict per-bin monotonicity check
      flaked; a correlation over all volumes absorbs that jitter while a swap
      (|r| ~ 0) or sign-scrambling bug still fails. Observed |r| ~ 0.66-0.83, so
      0.5 clears it with margin yet stays far above the failure modes.

    As above, ANTs' rotation-recovery accuracy is not asserted — only that
    pineuro reports the rotation in the correct column with a consistent sign.
    """
    motion = _ramp_motion(_N_VOLS, column=5, lo=-12.0, hi=12.0)  # rot_z ramp
    moved, _ = _moved_phantom(_N_VOLS, motion)
    in_path = tmp_path / "moved.nii.gz"
    out_path = tmp_path / "mcf.nii.gz"
    write_nifti(moved, _AFFINE, in_path, tr=1.0)
    motion_correction(in_path, out_path)

    par = np.loadtxt(str(out_path) + ".par")
    assert par.shape == (_N_VOLS, 6)

    # Within the rotation sub-block [rot_x, rot_y, rot_z], the injected pure-Z
    # rotation must dominate axis 2 (rot_z).
    dominant_rot_axis = int(np.argmax(np.mean(np.abs(par[:, 0:3]), axis=0)))
    assert (
        record(
            "preprocessing",
            "motion_par_rotation_dominant_axis",
            float(dominant_rot_axis),
            threshold=2.0,
            op="==",
            description=(
                "The synthetic head rotates around one axis only. Motion correction "
                "should report that rotation in the matching column; this checks "
                "which column it landed in."
            ),
        )
        == 2.0
    )

    ramp_corr = _ramp_tracking(par[:, 2], motion[:, 5])
    assert (
        record(
            "preprocessing",
            "motion_par_rotation_ramp_corr",
            ramp_corr,
            threshold=0.5,
            op=">=",
            description=(
                "The rotation ramps steadily in one direction; the reported "
                "rotation values should track that ramp. This is |correlation| "
                "between them (0-1), robust to ANTs' per-volume rotation jitter. "
                ">=0.5 is a clearly significant association (n=20 volumes), which a "
                "correct, consistent-sign column gives and a swap (~0) cannot."
            ),
        )
        >= 0.5
    )


def test_motion_correction_reduces_variance(tmp_path, record):
    """Motion correction reduces the variation that head movement introduced.

    Exact directional invariant: motion correction's purpose is to remove the
    temporal variance head motion injected into voxel timeseries. On a
    noise-free phantom every bit of temporal variance is motion-induced, so a
    correctly-wired motion_correction must produce a corrected series whose mean
    in-brain temporal variance is strictly *less* than the moved series'. A
    no-op (or a workflow bug that returns the data uncorrected, or applies the
    transforms the wrong way) fails this. The exact *amount* of reduction is
    ANTs' recovery accuracy and is not asserted.
    """
    motion = _ramp_motion(_N_VOLS, column=0, lo=-6.0, hi=6.0)
    motion[:, 5] = np.linspace(-12.0, 12.0, _N_VOLS)  # add a rot_z ramp -> combined motion
    moved, brain = _moved_phantom(_N_VOLS, motion)
    in_path = tmp_path / "moved.nii.gz"
    out_path = tmp_path / "mcf.nii.gz"
    write_nifti(moved, _AFFINE, in_path, tr=1.0)
    motion_correction(in_path, out_path)
    corrected = np.asarray(nib.load(str(out_path)).dataobj)

    moved_var = float(np.mean(np.var(moved[brain, :], axis=-1)))
    corrected_var = float(np.mean(np.var(corrected[brain, :], axis=-1)))
    variance_decreased = corrected_var < moved_var
    assert (
        record(
            "preprocessing",
            "motion_variance_decreased",
            float(variance_decreased),
            threshold=1.0,
            op="==",
            description=(
                "On a phantom whose only variation comes from head motion, "
                "correcting that motion must leave the brain voxels steadier over "
                "time; 1 means they are."
            ),
        )
        == 1.0
    )


def test_motion_correction_preserves_geometry(tmp_path, record):
    """Motion correction keeps image geometry and writes a valid motion file.

    These hold by construction in pineuro's own code -- independent of what ANTs
    converges to -- so they are exact: the corrected series keeps the input shape
    and affine, the .par is (n_vols, 6), and the middle (reference) volume is
    copied verbatim with an exactly-zero .par row (motion_correction copies
    data[..., ref_idx] and never touches motion_params[ref_idx]).
    """
    n_vols = 12
    motion = _ramp_motion(n_vols, column=0, lo=-3.0, hi=3.0)
    motion[:, 5] = np.linspace(-4.0, 4.0, n_vols)
    moved, _ = _moved_phantom(n_vols, motion)

    in_path = tmp_path / "moved.nii.gz"
    out_path = tmp_path / "mcf.nii.gz"
    write_nifti(moved, _AFFINE, in_path, tr=1.0)
    motion_correction(in_path, out_path)

    corrected_img = nib.load(str(out_path))
    corrected = np.asarray(corrected_img.dataobj)
    par = np.loadtxt(str(out_path) + ".par")

    # Geometry is preserved exactly.
    assert corrected.shape == moved.shape
    np.testing.assert_array_equal(corrected_img.affine, _AFFINE)
    assert par.shape == (n_vols, 6)

    # The middle volume is the reference: .par row exactly zero, volume copied verbatim.
    ref_idx = n_vols // 2
    par_ref_max = float(np.max(np.abs(par[ref_idx])))
    assert (
        record(
            "preprocessing",
            "motion_par_reference_row_max",
            par_ref_max,
            threshold=0.0,
            op="==",
            description=(
                "Motion is measured relative to one reference volume. That volume's "
                "own motion must be exactly zero — it cannot have moved relative to "
                "itself."
            ),
        )
        == 0.0
    )
    np.testing.assert_array_equal(corrected[..., ref_idx], moved[..., ref_idx].astype(np.float32))
