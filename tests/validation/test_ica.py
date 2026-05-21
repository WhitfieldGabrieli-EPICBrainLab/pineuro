"""Validation — ICA decomposition and spatial correlation.

spatial_correlation is an exact vectorized Pearson computation (fast); run_ica
runs real nilearn CanICA (slow). The ICA phantom is a clean linear mixture of
disjoint smooth spatial sources with independent timecourses.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pineuro.mask_extraction import run_ica, spatial_correlation

from .phantom import hrf_convolved_events, write_nifti

pytestmark = pytest.mark.validation

_AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])


def test_spatial_correlation_recovers_overlap(tmp_path, record):
    """Each network map matches its own brain region best."""
    shape = (27, 24, 18)
    parcellation = np.zeros(shape, dtype=np.int16)
    parcellation[3:9, 3:21, 3:15] = 1
    parcellation[10:16, 3:21, 3:15] = 2
    parcellation[17:23, 3:21, 3:15] = 3

    # One component per network: each an exact indicator of its parcellation region.
    components = np.stack([(parcellation == k + 1).astype(np.float64) for k in range(3)], axis=-1)
    mask = (parcellation > 0).astype(np.float64)

    comp_path = tmp_path / "components.nii.gz"
    parc_path = tmp_path / "parcellation.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_nifti(components, _AFFINE, comp_path, tr=1.0)
    write_nifti(parcellation.astype(np.float64), _AFFINE, parc_path)
    write_nifti(mask, _AFFINE, mask_path)

    corr = spatial_correlation(comp_path, parc_path, mask_path, ["netA", "netB", "netC"], tmp_path / "corr.csv")
    matrix = corr.to_numpy()
    assert matrix.shape == (3, 3)

    # Each component is an exact copy of its region, so the diagonal correlation is
    # mathematically 1.0 — recorded as an error against 1.0, not a loose lower bound.
    diagonal_error = 1.0 - float(np.min(np.diag(matrix)))
    assert (
        record(
            "mask_extraction",
            "spatial_corr_diagonal_error",
            diagonal_error,
            threshold=1e-9,
            op="<",
            description=(
                "Each network map is checked against the brain region it should "
                "match. They are built identical, so the match should be perfect; "
                "this is how far the weakest match falls short."
            ),
        )
        < 1e-9
    )
    # Each component's best match is its own network.
    assert all(int(np.argmax(matrix[k])) == k for k in range(3))


def _build_ica_mixture(
    vol_shape: tuple[int, int, int], n_volumes: int, tr: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A 4D phantom: a linear mixture of three disjoint box sources.

    Sharply disjoint binary spatial sources driven by independent HRF-event
    timecourses — a clean, fully-deterministic input for run_ica. The phantom's
    decomposition *quality* is CanICA's concern; this module only feeds run_ica
    a well-formed 4D input and checks the structure of what it returns.
    """
    nx, ny, nz = vol_shape

    sources: list[np.ndarray] = []
    for x0, x1 in ((5, 11), (13, 19), (21, 27)):
        region = np.zeros(vol_shape, dtype=np.float64)
        region[x0:x1, 6 : ny - 6, 5 : nz - 5] = 1.0
        sources.append(region)
    timecourses = [hrf_convolved_events(n_volumes, tr, n_events=22, seed=seed + 13 * k + 1) for k in range(3)]

    mask = np.zeros(vol_shape, dtype=np.float64)
    mask[4 : nx - 4, 4 : ny - 4, 3 : nz - 3] = 1.0

    data = np.zeros((nx, ny, nz, n_volumes), dtype=np.float64)
    data[mask > 0] = 1000.0
    for region, timecourse in zip(sources, timecourses, strict=True):
        data += 60.0 * region[..., None] * timecourse[None, None, None, :]
    data *= (mask > 0)[..., None]
    return data, mask


@pytest.mark.slow
def test_run_ica_output_structure(tmp_path, record):
    """ICA produces a correctly-structured set of network maps.

    Asserts only pineuro-owned, deterministic facts about run_ica's output —
    not CanICA's decomposition *accuracy* (an iterative-optimizer property, out
    of scope per the suite's design). On a synthetic mixture, run_ica must:

    - return a components file holding exactly ``n_components`` maps carrying the
      input's shape and affine;
    - produce only finite component values (no NaN / Inf);
    - leave components exactly 0 outside the mask (CanICA's masker fills only
      in-mask voxels);
    - write a metadata file recording the requested n_components.

    pineuro's real ICA-domain math — matching components to atlas networks via
    ``spatial_correlation`` — is validated exactly (1e-9) in
    ``test_spatial_correlation_recovers_overlap`` above.
    """
    vol_shape, n_volumes, tr = (32, 32, 24), 120, 1.0
    n_components = 3
    data, mask = _build_ica_mixture(vol_shape, n_volumes, tr, seed=0)

    func_path = tmp_path / "func.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    write_nifti(data, _AFFINE, func_path, tr=tr)
    write_nifti(mask, _AFFINE, mask_path)

    components_file, metadata_file = run_ica(func_path, mask_path, tmp_path / "ica", "synth", n_components=n_components)

    # Exactly n_components maps, carrying the input geometry.
    comp_img = nib.load(str(components_file))
    assert comp_img.shape == (*vol_shape, n_components)
    np.testing.assert_allclose(comp_img.affine, _AFFINE, atol=1e-4)
    components = np.asarray(comp_img.dataobj)

    # Exact: the component count equals what was requested.
    n_maps = int(components.shape[3])
    assert (
        record(
            "mask_extraction",
            "ica_component_count",
            float(n_maps),
            threshold=float(n_components),
            op="==",
            description=("The number of network maps ICA produced. It must equal the number requested."),
        )
        == n_components
    )

    # Exact: every component value is finite.
    n_nonfinite = int(np.sum(~np.isfinite(components)))
    assert (
        record(
            "mask_extraction",
            "ica_nonfinite_voxels",
            float(n_nonfinite),
            threshold=0.0,
            op="==",
            description=(
                "Voxels in the network maps holding an invalid (not-a-number or infinite) value. There must be none."
            ),
        )
        == 0.0
    )

    # Exact: components are zero outside the mask (the masker fills only in-mask voxels).
    outside = mask <= 0
    n_outside_nonzero = int(np.sum(components[outside] != 0.0))
    assert (
        record(
            "mask_extraction",
            "ica_out_of_mask_nonzero",
            float(n_outside_nonzero),
            threshold=0.0,
            op="==",
            description=(
                "Voxels outside the brain that were given a nonzero value. A "
                "network is defined only inside the brain, so there must be none."
            ),
        )
        == 0.0
    )

    # Exact: the metadata file is written and records the requested n_components.
    assert metadata_file.exists()
    meta = np.load(metadata_file)
    assert int(meta["n_components"]) == n_components
