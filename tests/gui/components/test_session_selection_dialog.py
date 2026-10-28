"""Tests for session-start validation in the session selection dialog."""

from pathlib import Path

from pineuro.gui.components.session_selection_dialog import pending_slot_assignment, validate_session_start
from pineuro.gui.models import StudyConfig, StudyMetadata


def _study(sessions: list, sham: dict | None = None) -> StudyConfig:
    session_config: dict = {"paths": {"base_dir": "/data"}, "sessions": sessions}
    if sham is not None:
        session_config["sham"] = sham
    return StudyConfig(metadata=StudyMetadata(name="S"), session_config=session_config)


def _slot_study(tmp_path, free_slots: int = 1) -> tuple[StudyConfig, Path]:
    """A two-session sham study whose randomization file is a slot list (sub-01 in slot 001)."""
    rows = ["slot\tsubject_id\tcondition", "001\tsub-01\treal"]
    rows += [f"{i + 2:03d}\t\tsham" for i in range(free_slots)]
    rand = tmp_path / "slots.tsv"
    rand.write_text("\n".join(rows) + "\n")
    session = {"localizers": [], "feedback_runs": []}
    return _study([session, session], sham={"enabled": True, "randomization_file": str(rand)}), rand


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


def test_unlisted_participant_with_free_slot_starts_first_session(tmp_path):
    study, rand = _slot_study(tmp_path)
    assert validate_session_start(study, "sub-02", 0, tmp_path) == []
    assert pending_slot_assignment(study, "sub-02") == (rand, "002")


def test_unlisted_participant_blocked_at_later_session(tmp_path):
    study, _ = _slot_study(tmp_path)
    errors = validate_session_start(study, "sub-02", 1, tmp_path)
    assert errors and "first session" in errors[0]


def test_full_slot_list_blocks_new_participant(tmp_path):
    study, _ = _slot_study(tmp_path, free_slots=0)
    errors = validate_session_start(study, "sub-02", 0, tmp_path)
    assert errors and "every slot" in errors[0]
    assert pending_slot_assignment(study, "sub-02") is None


def test_listed_participant_has_no_pending_slot(tmp_path):
    study, _ = _slot_study(tmp_path)
    assert validate_session_start(study, "sub-01", 1, tmp_path) == []
    assert pending_slot_assignment(study, "sub-01") is None


def test_plain_list_has_no_pending_slot(tmp_path):
    rand = tmp_path / "rand.tsv"
    rand.write_text("subject_id\tcondition\nsub-01\treal\n")
    study = _study([{"localizers": [], "feedback_runs": []}], sham={"enabled": True, "randomization_file": str(rand)})
    assert validate_session_start(study, "sub-02", 0, tmp_path)
    assert pending_slot_assignment(study, "sub-02") is None
