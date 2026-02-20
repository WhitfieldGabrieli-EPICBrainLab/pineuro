"""Integration tests for mask extraction using Rockland_NFB data.

These tests require preprocessed data from the preprocessing integration tests.
Run with: pytest -m integration
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestIcaMaskExtractionReal:
    def test_ica_on_preprocessed_data(self, rockland_rest_dicoms, tmp_path):
        """Run ICA mask extraction on real preprocessed resting-state data."""
        from pineuro.mask_extraction import mask_extraction_rest
        from pineuro.preprocessing import preprocessing

        root_dir = tmp_path / "study"
        root_dir.mkdir()

        # First preprocess
        preproc_file, mask_file, ref_file = preprocessing(
            participant_id="sub-A00059346",
            task="rest",
            root_dir=root_dir,
            input_dicoms=rockland_rest_dicoms,
            qc_enabled=False,
        )

        # Then extract masks
        mask_files = mask_extraction_rest(
            participant_id="sub-A00059346",
            root_dir=root_dir,
            input_file=preproc_file,
            mask_file=mask_file,
            ref_file=ref_file,
            yeo_targets=["DefaultA"],  # Just one network for speed
            n_components=10,  # Fewer components for speed
            qc_enabled=False,
        )

        assert len(mask_files) >= 1
        assert "DefaultA" in mask_files

        import nibabel as nib

        for name, path in mask_files.items():
            assert path.exists()
            img = nib.load(path)
            assert len(img.shape) == 3
