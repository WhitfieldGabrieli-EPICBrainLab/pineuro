"""Validation -- VolumeSimulator file round-trip contract.

The simulator's data contract: source files appear in target_dir, in the same
temporal order they appear in source_dir, with byte-identical content. For
NIfTI sources, the matching JSON sidecar is copied alongside each volume.

This test does not exercise the *timing* path (the simulator's TR-paced sleep)
because that is environment-dependent. It exercises the *data* path that
real-time processing downstream depends on -- if the simulator copies the
wrong file or misorders volumes, every downstream component sees corrupt input.

Tests use speed=100 so the simulated TR delays compress to milliseconds.

Note on source-dir contents: ``utils.get_volume_files`` returns ALL files in
the source dir (it cannot rely on extensions because DICOM files often have
none). Tests that count volumes therefore use NIfTI-only sources without
co-located JSON sidecars; a dedicated test exercises the sidecar copy contract
in isolation with a single volume + sidecar.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from pineuro.simulator import VolumeSimulator

pytestmark = pytest.mark.validation


def _write_nifti(path, data):
    """Write a 3D NIfTI volume (no sidecar)."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(data.astype(np.float64), affine)
    nib.save(img, str(path))
    return path


def _make_synthetic_nifti_source(tmp_path, n_volumes):
    """Create a source dir of n NIfTI files, each carrying its index as a marker.

    No sidecars: ``utils.get_volume_files`` lists every file in the dir, so
    co-located JSON sidecars would inflate the iteration count.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paths = []
    for i in range(n_volumes):
        # Marker at voxel [0,0,0] uniquely identifies the source position.
        data = np.full((4, 4, 4), float(i + 1), dtype=np.float64)
        vol_path = source_dir / f"vol_{i:03d}.nii"
        _write_nifti(vol_path, data)
        paths.append(vol_path)
    return source_dir, paths


def test_simulator_construct_requires_source_to_exist(tmp_path):
    """A missing source_dir raises FileNotFoundError (contract)."""
    with pytest.raises(FileNotFoundError):
        VolumeSimulator(tmp_path / "nope", tmp_path / "tgt", tr=1.0, speed=100.0)


def test_simulator_construct_requires_volume_files(tmp_path):
    """An empty source_dir raises ValueError (contract)."""
    source_dir = tmp_path / "source_empty"
    source_dir.mkdir()
    with pytest.raises(ValueError, match="No volume files"):
        VolumeSimulator(source_dir, tmp_path / "tgt", tr=1.0, speed=100.0)


def test_simulator_copies_all_volumes_in_source_order(tmp_path):
    """The simulator yields each volume once, in source-file lexical order."""
    source_dir, source_paths = _make_synthetic_nifti_source(tmp_path, n_volumes=5)
    target_dir = tmp_path / "target"
    sim = VolumeSimulator(source_dir, target_dir, tr=0.01, speed=100.0)

    yields = list(sim.copy_volumes_iter())
    # Yields are 1-indexed per the docstring "Yields ... volume number (1-indexed)".
    assert yields == list(range(1, len(source_paths) + 1))


def test_simulator_target_files_byte_match_source(tmp_path):
    """Each copied volume in target_dir is byte-identical to the corresponding source."""
    source_dir, source_paths = _make_synthetic_nifti_source(tmp_path, n_volumes=4)
    target_dir = tmp_path / "target"
    sim = VolumeSimulator(source_dir, target_dir, tr=0.01, speed=100.0)
    list(sim.copy_volumes_iter())  # drain

    for src in source_paths:
        dst = target_dir / src.name
        assert dst.exists()
        assert dst.read_bytes() == src.read_bytes()


def test_simulator_preserves_voxel_order_via_marker(tmp_path):
    """Reading back each copied NIfTI recovers the marker for that source position.

    The marker at voxel [0,0,0] of the i-th source file must equal the marker
    of the i-th copied file. A bug that swaps or misorders volumes would
    surface here as a mismatched marker.
    """
    source_dir, source_paths = _make_synthetic_nifti_source(tmp_path, n_volumes=6)
    target_dir = tmp_path / "target"
    sim = VolumeSimulator(source_dir, target_dir, tr=0.01, speed=100.0)
    list(sim.copy_volumes_iter())

    for i, src in enumerate(source_paths):
        dst = target_dir / src.name
        produced = np.asarray(nib.load(str(dst)).dataobj)[0, 0, 0]
        assert produced == float(i + 1)


def test_simulator_n_volumes_limits_iteration(tmp_path):
    """Passing n_volumes copies only the first n source files."""
    source_dir, source_paths = _make_synthetic_nifti_source(tmp_path, n_volumes=5)
    target_dir = tmp_path / "target"
    sim = VolumeSimulator(source_dir, target_dir, tr=0.01, speed=100.0)
    yields = list(sim.copy_volumes_iter(n_volumes=3))

    assert yields == [1, 2, 3]
    # The first 3 volumes are in target_dir; the rest are not.
    assert (target_dir / source_paths[0].name).exists()
    assert (target_dir / source_paths[1].name).exists()
    assert (target_dir / source_paths[2].name).exists()
    assert not (target_dir / source_paths[3].name).exists()
    assert not (target_dir / source_paths[4].name).exists()


def test_simulator_clears_target_before_run(tmp_path):
    """clear_target=True (default) removes pre-existing files in target_dir."""
    source_dir, _ = _make_synthetic_nifti_source(tmp_path, n_volumes=2)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    stale_file = target_dir / "stale_old_file.nii"
    stale_file.write_text("stale content")

    sim = VolumeSimulator(source_dir, target_dir, tr=0.01, speed=100.0, clear_target=True)
    list(sim.copy_volumes_iter())
    assert not stale_file.exists()


def test_simulator_copies_json_sidecar_with_volume(tmp_path):
    """When a .nii file has a matching .json sidecar, both are copied.

    Isolated to a single volume + sidecar to avoid the get_volume_files
    multi-file iteration quirk (it lists ALL files in source_dir, including
    sidecars). Here the source contains exactly one .nii and one .json, so the
    iteration count is 2 and we check that both files materialise in target.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    vol = source_dir / "vol_001.nii"
    _write_nifti(vol, np.full((3, 3, 3), 7.0, dtype=np.float64))
    sidecar = vol.with_suffix(".json")
    sidecar.write_text(json.dumps({"RepetitionTime": 1.5, "marker": "alpha"}))

    target_dir = tmp_path / "target"
    sim = VolumeSimulator(source_dir, target_dir, tr=0.01, speed=100.0)
    list(sim.copy_volumes_iter())

    dst_vol = target_dir / "vol_001.nii"
    dst_json = target_dir / "vol_001.json"
    assert dst_vol.exists()
    assert dst_json.exists()
    # Sidecar content survives verbatim.
    assert json.loads(dst_json.read_text()) == {"RepetitionTime": 1.5, "marker": "alpha"}
    # Volume bytes survive verbatim.
    assert dst_vol.read_bytes() == vol.read_bytes()
