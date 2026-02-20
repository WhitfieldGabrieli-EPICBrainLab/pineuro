"""Integration tests for preprocessing pipeline using Rockland_NFB data.

These tests require the Rockland_NFB dataset in sourcedata/.
Run with: pytest -m integration
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestDicomToNiftiReal:
    def test_convert_rest_dicoms(self, rockland_rest_dicoms, tmp_path):
        """Convert real resting-state DICOMs to NIfTI."""
        from pineuro.utils import dicom_to_nifti

        output_dir = tmp_path / "nifti"
        output_dir.mkdir()

        result = dicom_to_nifti(rockland_rest_dicoms, output_dir, "rest_bold")

        assert result.exists()
        assert result.suffix == ".gz" or result.name.endswith(".nii")

        import nibabel as nib

        img = nib.load(result)
        # Rockland rest data should be 4D
        assert len(img.shape) == 4
        assert img.shape[3] > 1  # Multiple volumes


class TestPreprocessingPipelineReal:
    def test_full_pipeline_single_run(self, rockland_rest_dicoms, tmp_path):
        """Run full preprocessing pipeline on 1 subject, 1 rest run."""
        from pineuro.preprocessing import preprocessing

        root_dir = tmp_path / "study"
        root_dir.mkdir()

        preproc_file, mask_file, ref_file = preprocessing(
            participant_id="sub-A00059346",
            task="rest",
            root_dir=root_dir,
            input_dicoms=rockland_rest_dicoms,
            qc_enabled=False,  # Skip QC report for speed
        )

        assert preproc_file.exists()
        assert mask_file.exists()
        assert ref_file.exists()

        import nibabel as nib

        # Preprocessed file should be 4D
        preproc_img = nib.load(preproc_file)
        assert len(preproc_img.shape) == 4

        # Mask should be 3D binary
        mask_img = nib.load(mask_file)
        assert len(mask_img.shape) == 3
        mask_data = mask_img.get_fdata()
        unique_values = set(mask_data.flatten())
        assert unique_values <= {0.0, 1.0}
