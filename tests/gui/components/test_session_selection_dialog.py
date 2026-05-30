"""Tests for session-start validation in the session selection dialog."""

from pineuro.gui.components.session_selection_dialog import validate_session_start
from pineuro.gui.models import StudyConfig, StudyMetadata


def _study(sessions: list) -> StudyConfig:
    return StudyConfig(
        metadata=StudyMetadata(name="S"),
        session_config={"paths": {"base_dir": "/data"}, "sessions": sessions},
    )


def _approved_masks(base_dir, participant, session_id, names):
    """Create thresholded mask files for a localizer in an earlier session."""
    masks_dir = base_dir / participant / session_id / "mask_extraction" / "03_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (masks_dir / f"{participant}_{session_id}_{n}_thr.nii.gz").write_bytes(b"")


def test_none_study_config_is_ok():
    assert validate_session_start(None, "sub-01", 0, "/data") == []


def test_fresh_session_is_ok(tmp_path):
    study = _study([{"localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA"]}], "feedback_runs": []}])
    assert validate_session_start(study, "sub-01", 0, tmp_path) == []


def test_out_of_range_session(tmp_path):
    study = _study([{"localizers": [], "feedback_runs": []}])
    errors = validate_session_start(study, "sub-01", 3, tmp_path)
    assert errors and "not defined" in errors[0]


def test_forward_reference_rejected(tmp_path):
    # Session 1 (index 0) cannot reuse ses-02.
    study = _study(
        [
            {
                "localizers": [
                    {"id": "rest", "type": "rest", "networks": ["DefaultA"], "reuse_from_session": "ses-02"}
                ],
                "feedback_runs": [],
            }
        ]
    )
    errors = validate_session_start(study, "sub-01", 0, tmp_path)
    assert errors and "not an earlier session" in errors[0]


def test_missing_source_masks_rejected(tmp_path):
    study = _study(
        [
            {"localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA"]}], "feedback_runs": []},
            {
                "localizers": [
                    {"id": "rest", "type": "rest", "networks": ["DefaultA"], "reuse_from_session": "ses-01"}
                ],
                "feedback_runs": [],
            },
        ]
    )
    errors = validate_session_start(study, "sub-01", 1, tmp_path)
    assert errors and "were not found" in errors[0]


def test_reuse_with_present_masks_ok(tmp_path):
    _approved_masks(tmp_path, "sub-01", "ses-01", ["DefaultA"])
    study = _study(
        [
            {"localizers": [{"id": "rest", "type": "rest", "networks": ["DefaultA"]}], "feedback_runs": []},
            {
                "localizers": [
                    {"id": "rest", "type": "rest", "networks": ["DefaultA"], "reuse_from_session": "ses-01"}
                ],
                "feedback_runs": [],
            },
        ]
    )
    assert validate_session_start(study, "sub-01", 1, tmp_path) == []
