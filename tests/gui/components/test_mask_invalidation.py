"""Tests for pineuro.gui.components.mask_invalidation."""

from pathlib import Path

from pineuro.gui.components.mask_invalidation import (
    copy_to_shared_masks,
    extraction_root,
    invalidate_coregistration,
    invalidate_localizer_extraction,
    store_participant_mask,
)

_PREFIX = "sub-01_ses-01"


def _config(n_localizers: int) -> dict:
    task = {"type": "rit", "id": "rit", "contrasts": {"rumination_network": "rumination - distraction"}}
    rest = {"type": "rest", "id": "rest", "networks": ["DefaultA"]}
    return {"localizers": [task, rest][:n_localizers]}


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


class TestExtractionRoot:
    def test_single_localizer_uses_mask_extraction(self, tmp_path):
        assert extraction_root(tmp_path, _config(1), 0) == tmp_path / "mask_extraction"

    def test_multi_localizer_uses_per_localizer_folder(self, tmp_path):
        assert extraction_root(tmp_path, _config(2), 0) == tmp_path / "mask_extraction" / "loc_0_rit"
        assert extraction_root(tmp_path, _config(2), 1) == tmp_path / "mask_extraction" / "loc_1_rest"


class TestInvalidateCoregistration:
    def _tree(self, base: Path) -> dict[str, Path]:
        files = {}
        for parent in ("mask_extraction/04_reference", "feedback/01_reference"):
            for sub in ("masks", "qc", "xfm"):
                files[f"{parent}/{sub}"] = _touch(base / parent / sub / "f.nii.gz")
            files[f"{parent}/mean"] = _touch(base / parent / f"{_PREFIX}_task-reference_run-01_mean.nii.gz")
        return files

    def test_removes_masks_and_qc_keeps_transforms_and_scans(self, tmp_path):
        files = self._tree(tmp_path)
        invalidate_coregistration(tmp_path, include_transforms=False)

        for key, path in files.items():
            removed = key.endswith(("/masks", "/qc"))
            assert path.exists() is not removed, key

    def test_include_transforms_also_removes_xfm(self, tmp_path):
        files = self._tree(tmp_path)
        invalidate_coregistration(tmp_path, include_transforms=True)

        for key, path in files.items():
            assert path.exists() is key.endswith("/mean"), key

    def test_nothing_to_remove(self, tmp_path):
        assert invalidate_coregistration(tmp_path, include_transforms=True) == []


class TestInvalidateLocalizerExtraction:
    def test_single_localizer_keeps_script_and_overrides(self, tmp_path):
        root = tmp_path / "mask_extraction"
        stale = [
            _touch(root / "01_glm" / "map.nii.gz"),
            _touch(root / "02_mnireg" / "xfm.mat"),
            _touch(root / "03_masks" / f"{_PREFIX}_rumination_network_thr.nii.gz"),
            _touch(root / "05_tissue" / "wm.nii.gz"),
            _touch(root / ".masks_approved"),
        ]
        kept = [_touch(root / "reproduce_masks.py"), _touch(root / "mask_overrides.json")]

        invalidate_localizer_extraction(tmp_path, _config(1), 0, _PREFIX)

        assert not any(p.exists() for p in stale)
        assert all(p.exists() for p in kept)

    def test_multi_localizer_clears_only_its_shared_masks(self, tmp_path):
        loc_dir = tmp_path / "mask_extraction" / "loc_0_rit"
        _touch(loc_dir / "03_masks" / f"{_PREFIX}_rumination_network_thr.nii.gz")
        shared = tmp_path / "mask_extraction" / "03_masks"
        own = _touch(shared / f"{_PREFIX}_rumination_network_thr.nii.gz")
        other = _touch(shared / f"{_PREFIX}_DefaultA_thr.nii.gz")
        tissue = _touch(tmp_path / "mask_extraction" / "05_tissue" / "wm.nii.gz")

        invalidate_localizer_extraction(tmp_path, _config(2), 0, _PREFIX)

        assert not (loc_dir / "03_masks").exists()
        assert not own.exists()
        assert other.exists()
        assert tissue.exists()  # shared tissue masks belong to the session, not this localizer


def test_store_participant_mask_copies_into_mask_edits(tmp_path):
    source = _touch(tmp_path / "elsewhere" / "net_edited_by_hand.nii.gz")
    root = tmp_path / "mask_extraction"

    stored = store_participant_mask(root, _PREFIX, "net", source)
    assert stored == root / "mask_edits" / f"{_PREFIX}_net_edited.nii.gz"
    assert stored.read_text() == "x"

    # Choosing the stored copy again is a no-op, not a copy onto itself
    assert store_participant_mask(root, _PREFIX, "net", stored) == stored


def test_mask_edits_survive_extraction_invalidation(tmp_path):
    root = tmp_path / "mask_extraction"
    edit = _touch(root / "mask_edits" / f"{_PREFIX}_rumination_network_edited.nii.gz")
    _touch(root / "03_masks" / f"{_PREFIX}_rumination_network_thr.nii.gz")

    invalidate_localizer_extraction(tmp_path, _config(1), 0, _PREFIX)
    assert edit.exists()


def test_copy_to_shared_masks(tmp_path):
    src = _touch(tmp_path / "mask_extraction" / "loc_0_rit" / "03_masks" / "m_thr.nii.gz")
    copied = copy_to_shared_masks(tmp_path, [src])
    assert copied == [tmp_path / "mask_extraction" / "03_masks" / "m_thr.nii.gz"]
    assert copied[0].read_text() == "x"
