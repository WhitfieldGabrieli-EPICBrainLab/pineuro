"""Integration tests for feedback estimation using Rockland_NFB data.

These tests require preprocessed data and masks.
Run with: pytest -m integration
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestFeedbackEstimatorReal:
    def test_process_real_volumes(self, rockland_rest_dicoms, tmp_path):
        """Process real NIfTI volumes through FeedbackEstimator."""
        import nibabel as nib

        from pineuro.feedback import FeedbackEstimator
        from pineuro.preprocessing import preprocessing

        root_dir = tmp_path / "study"
        root_dir.mkdir()

        # Preprocess to get real data
        preproc_file, mask_file, ref_file = preprocessing(
            participant_id="sub-A00059346",
            task="rest",
            root_dir=root_dir,
            input_dicoms=rockland_rest_dicoms,
            qc_enabled=False,
        )

        # Use the brain mask as a simple ROI
        estimator = FeedbackEstimator(
            mask_paths=[mask_file],
            mask_names=["WholeBrain"],
            n_baseline=5,
            n_volumes=20,
            tr=1.4,  # Rockland rest TR
            include_motion_regressors=False,
        )

        # Load volumes and process them one by one
        img = nib.load(preproc_file)
        data = img.get_fdata()
        n_volumes = min(data.shape[3], 20)

        activations = []
        for i in range(n_volumes):
            vol_data = data[..., i].astype(np.float32)
            result = estimator.process_volume(vol_data)
            if result is not None:
                activations.append(result)

        # Should have activations after the baseline period
        assert len(activations) > 0
        # All activations should be finite
        for act in activations:
            assert "WholeBrain" in act
            assert np.isfinite(act["WholeBrain"])
