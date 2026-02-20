"""Shared pytest fixtures for pineuro test suite.

Provides factory fixtures for creating synthetic neuroimaging data
(NIfTI volumes, masks, motion parameters, etc.) used across all test modules.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


# =============================================================================
# NIfTI Fixtures
# =============================================================================


@pytest.fixture
def synthetic_nifti_3d(tmp_path):
    """Factory fixture: creates a 3D NIfTI file with random data.

    Usage::

        def test_something(synthetic_nifti_3d):
            path = synthetic_nifti_3d(shape=(8, 8, 8))
    """

    def _make(shape=(8, 8, 8), affine=None, data=None, name="vol3d.nii.gz"):
        if affine is None:
            affine = np.eye(4)
        if data is None:
            rng = np.random.default_rng(42)
            data = rng.standard_normal(shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        path = tmp_path / name
        nib.save(img, path)
        return path

    return _make


@pytest.fixture
def synthetic_nifti_4d(tmp_path):
    """Factory fixture: creates a 4D NIfTI file with random timeseries data.

    Usage::

        def test_something(synthetic_nifti_4d):
            path = synthetic_nifti_4d(shape=(8, 8, 8), n_volumes=20)
    """

    def _make(shape=(8, 8, 8), n_volumes=20, affine=None, tr=2.0, data=None, name="vol4d.nii.gz"):
        if affine is None:
            affine = np.eye(4)
        if data is None:
            rng = np.random.default_rng(42)
            data = rng.standard_normal((*shape, n_volumes)).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header["pixdim"][4] = tr
        path = tmp_path / name
        nib.save(img, path)
        return path

    return _make


@pytest.fixture
def sample_mask(tmp_path):
    """Factory fixture: creates a binary NIfTI mask.

    Usage::

        def test_something(sample_mask):
            path = sample_mask(shape=(8, 8, 8), fraction=0.3)
    """

    def _make(shape=(8, 8, 8), fraction=0.3, affine=None, name="mask.nii.gz"):
        if affine is None:
            affine = np.eye(4)
        rng = np.random.default_rng(42)
        data = (rng.random(shape) < fraction).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        path = tmp_path / name
        nib.save(img, path)
        return path

    return _make


@pytest.fixture
def sample_motion_par(tmp_path):
    """Factory fixture: creates a .par motion parameter file (6 columns).

    Format: 6 columns per row (rx, ry, rz, tx, ty, tz), space-separated.

    Usage::

        def test_something(sample_motion_par):
            path = sample_motion_par(n_volumes=20)
    """

    def _make(n_volumes=20, name="motion.par"):
        rng = np.random.default_rng(42)
        # Small rotations (radians) and translations (mm)
        rotations = rng.standard_normal((n_volumes, 3)) * 0.001
        translations = rng.standard_normal((n_volumes, 3)) * 0.1
        data = np.hstack([rotations, translations])
        path = tmp_path / name
        np.savetxt(path, data, fmt="%.6f")
        return path

    return _make


@pytest.fixture
def sample_json_sidecar(tmp_path):
    """Factory fixture: creates a JSON sidecar file for NIfTI volumes.

    Usage::

        def test_something(sample_json_sidecar):
            path = sample_json_sidecar(motion=[0.001, 0.0, -0.001, 0.1, 0.0, -0.1])
    """

    def _make(motion=None, tr=2.0, name="sidecar.json"):
        sidecar = {"RepetitionTime": tr}
        if motion is not None:
            sidecar["ImageComments"] = f"Motion: {','.join(str(m) for m in motion)}"
        path = tmp_path / name
        path.write_text(json.dumps(sidecar))
        return path

    return _make


@pytest.fixture
def sample_feedbacklog(tmp_path):
    """Factory fixture: creates a feedbacklog TSV file.

    Usage::

        def test_something(sample_feedbacklog):
            path = sample_feedbacklog(n_volumes=50, mask_names=["DMN", "Motor"])
    """

    def _make(n_volumes=50, mask_names=None, name="feedbacklog.tsv"):
        if mask_names is None:
            mask_names = ["DMN"]
        rng = np.random.default_rng(42)
        lines = []
        # Header
        cols = ["volume", "elapsed_time", "fd"]
        cols.extend([f"activation_{m}" for m in mask_names])
        cols.extend(["dvars", "tsnr"])
        lines.append("\t".join(cols))
        # Data rows
        for i in range(n_volumes):
            row = [str(i + 1), f"{i * 2.0:.3f}", f"{rng.random() * 0.5:.4f}"]
            row.extend([f"{rng.standard_normal():.4f}" for _ in mask_names])
            row.extend([f"{rng.random() * 50:.2f}", f"{40 + rng.random() * 20:.2f}"])
            lines.append("\t".join(row))
        path = tmp_path / name
        path.write_text("\n".join(lines))
        return path

    return _make


@pytest.fixture
def sample_design_matrix(tmp_path):
    """Factory fixture: creates a simple TSV design matrix.

    Usage::

        def test_something(sample_design_matrix):
            path = sample_design_matrix(n_volumes=50, n_regressors=3)
    """

    def _make(n_volumes=50, n_regressors=3, name="design.tsv"):
        rng = np.random.default_rng(42)
        cols = [f"reg_{i}" for i in range(n_regressors)]
        data = rng.standard_normal((n_volumes, n_regressors))
        lines = ["\t".join(cols)]
        for row in data:
            lines.append("\t".join(f"{v:.6f}" for v in row))
        path = tmp_path / name
        path.write_text("\n".join(lines))
        return path

    return _make
