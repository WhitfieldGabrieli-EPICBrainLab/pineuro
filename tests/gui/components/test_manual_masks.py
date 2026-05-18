"""Tests for pineuro.gui.components.manual_masks — per-participant manual-mask records."""

from pineuro.gui.components.manual_masks import (
    manual_masks_complete,
    read_manual_masks,
    write_manual_masks,
)


def _record(masks: dict[str, str], reference: str, anat: str | None = None, space: str = "native") -> dict:
    return {"mask_space": space, "masks": masks, "reference": reference, "anat": anat}


class TestReadWrite:
    def test_read_missing_returns_none(self, tmp_path):
        assert read_manual_masks(tmp_path, "manual") is None

    def test_write_then_read_round_trip(self, tmp_path):
        record = _record({"DMN": "/data/dmn.nii.gz"}, "/data/ref.nii.gz")
        write_manual_masks(tmp_path, "manual", record)
        assert read_manual_masks(tmp_path, "manual") == record

    def test_write_merges_multiple_localizers(self, tmp_path):
        # Writing a second localizer must not clobber the first.
        write_manual_masks(tmp_path, "loc_a", _record({"DMN": "/a/dmn.nii.gz"}, "/a/ref.nii.gz"))
        write_manual_masks(tmp_path, "loc_b", _record({"FPN": "/b/fpn.nii.gz"}, "/b/ref.nii.gz"))
        assert read_manual_masks(tmp_path, "loc_a")["masks"] == {"DMN": "/a/dmn.nii.gz"}
        assert read_manual_masks(tmp_path, "loc_b")["masks"] == {"FPN": "/b/fpn.nii.gz"}

    def test_read_unknown_localizer_returns_none(self, tmp_path):
        write_manual_masks(tmp_path, "manual", _record({"DMN": "/data/dmn.nii.gz"}, "/data/ref.nii.gz"))
        assert read_manual_masks(tmp_path, "other") is None


class TestManualMasksComplete:
    def test_false_when_unconfigured(self, tmp_path):
        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "native"}
        assert manual_masks_complete(tmp_path, loc) is False

    def test_true_when_all_files_present(self, tmp_path):
        dmn = tmp_path / "dmn.nii.gz"
        ref = tmp_path / "ref.nii.gz"
        for p in (dmn, ref):
            p.touch()
        write_manual_masks(tmp_path, "manual", _record({"DMN": str(dmn)}, str(ref)))
        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "native"}
        assert manual_masks_complete(tmp_path, loc) is True

    def test_false_when_a_mask_file_is_missing(self, tmp_path):
        ref = tmp_path / "ref.nii.gz"
        ref.touch()
        write_manual_masks(tmp_path, "manual", _record({"DMN": str(tmp_path / "absent.nii.gz")}, str(ref)))
        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "native"}
        assert manual_masks_complete(tmp_path, loc) is False

    def test_false_when_a_named_network_has_no_entry(self, tmp_path):
        dmn = tmp_path / "dmn.nii.gz"
        ref = tmp_path / "ref.nii.gz"
        for p in (dmn, ref):
            p.touch()
        write_manual_masks(tmp_path, "manual", _record({"DMN": str(dmn)}, str(ref)))
        # FPN is named in the study config but not configured for this participant
        loc = {"id": "manual", "manual_mask_names": ["DMN", "FPN"], "manual_mask_space": "native"}
        assert manual_masks_complete(tmp_path, loc) is False

    def test_false_when_reference_missing(self, tmp_path):
        dmn = tmp_path / "dmn.nii.gz"
        dmn.touch()
        write_manual_masks(tmp_path, "manual", _record({"DMN": str(dmn)}, str(tmp_path / "absent.nii.gz")))
        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "native"}
        assert manual_masks_complete(tmp_path, loc) is False

    def test_anat_space_requires_t1(self, tmp_path):
        dmn = tmp_path / "dmn.nii.gz"
        ref = tmp_path / "ref.nii.gz"
        t1 = tmp_path / "t1.nii.gz"
        for p in (dmn, ref, t1):
            p.touch()
        loc = {"id": "manual", "manual_mask_names": ["DMN"], "manual_mask_space": "anat"}

        # Anat space without a T1 → incomplete
        write_manual_masks(tmp_path, "manual", _record({"DMN": str(dmn)}, str(ref), space="anat"))
        assert manual_masks_complete(tmp_path, loc) is False

        # With the T1 → complete
        write_manual_masks(tmp_path, "manual", _record({"DMN": str(dmn)}, str(ref), anat=str(t1), space="anat"))
        assert manual_masks_complete(tmp_path, loc) is True
