"""Validation -- a sham study blocks an unlisted participant before a run starts.

The original bug #2: entering a participant ID not in the randomization file
(e.g. ``rtchr001`` instead of ``sub-rtchr01``) produced no error -- the run
proceeded as "real" because ``get_condition_for_participant`` silently defaults
to "real" for unknown IDs. For a study with sham enabled study-wide this is a
blinding/integrity failure: an unlisted participant cannot be assigned a
condition, so the session must be refused rather than defaulted.

``validate_session_start`` is the operator-facing gate (its returned errors
disable the Start/Create buttons). These tests pin: with sham enabled it rejects
an unlisted participant and a missing/unconfigured randomization file, accepts a
listed one, and stays silent when sham is disabled. A reconciliation test keeps
``get_condition_for_participant``'s permissive default documented and explains
why callers in a sham study must check membership first. For a slot list, the
gate instead lets a new participant's first session assign them the next unused
slot, and each participant must then receive exactly that slot's condition.

Expected values follow directly from the rtCHR config shape and the documented
gate contract -- nothing is recomputed from pineuro's implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from pineuro.gui.components.session_selection_dialog import pending_slot_assignment, validate_session_start
from pineuro.sham import assign_next_slot, get_condition_for_participant, load_randomization

pytestmark = pytest.mark.validation


def _write_randomization(path):
    """Write a randomization TSV listing sub-rtchr01..03 (sub-rtchr01 is sham)."""
    pd.DataFrame(
        {
            "subject_id": ["sub-rtchr01", "sub-rtchr02", "sub-rtchr03"],
            "condition": ["sham", "real", "sham"],
            "matched_participant": ["", "", ""],
            "matched_session": ["", "", ""],
            "notes": ["", "", ""],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def _study_config(randomization_file, *, sham_enabled: bool = True):
    """A minimal StudyConfig stand-in: only ``session_config`` is read by the gate."""
    sham: dict = {"enabled": sham_enabled}
    if randomization_file is not None:
        sham["randomization_file"] = str(randomization_file)
    return SimpleNamespace(
        session_config={
            "sham": sham,
            "sessions": [
                {"localizers": [{"type": "rest", "id": "rest", "networks": ["DefaultA", "ContA"]}]},
            ],
        }
    )


# =============================================================================
# validate_session_start -- sham membership gate (bug #2)
# =============================================================================


def test_unlisted_participant_is_rejected(tmp_path):
    """A typo'd ID not in the randomization file produces an error naming both."""
    rand = _write_randomization(tmp_path / "rand.tsv")
    cfg = _study_config(rand)
    errors = validate_session_start(cfg, "rtchr001", 0, tmp_path)
    assert errors  # non-empty -> Start button disabled
    joined = " ".join(errors)
    assert "rtchr001" in joined
    assert rand.name in joined


def test_listed_participant_is_accepted(tmp_path):
    """A correctly-listed participant raises no membership error."""
    rand = _write_randomization(tmp_path / "rand.tsv")
    cfg = _study_config(rand)
    errors = validate_session_start(cfg, "sub-rtchr01", 0, tmp_path)
    assert errors == []


def test_missing_randomization_file_is_rejected(tmp_path):
    """Sham enabled but the configured randomization file is absent -> error."""
    cfg = _study_config(tmp_path / "does_not_exist.tsv")
    errors = validate_session_start(cfg, "sub-rtchr01", 0, tmp_path)
    assert errors
    assert any("not found" in e.lower() for e in errors)


def test_unconfigured_randomization_file_is_rejected(tmp_path):
    """Sham enabled but no randomization file configured at all -> error."""
    cfg = _study_config(None)
    errors = validate_session_start(cfg, "sub-rtchr01", 0, tmp_path)
    assert errors
    assert any("no randomization file" in e.lower() for e in errors)


def test_membership_not_checked_when_sham_disabled(tmp_path):
    """With sham disabled, the membership gate does not apply (any ID is fine)."""
    cfg = _study_config(None, sham_enabled=False)
    errors = validate_session_start(cfg, "rtchr001", 0, tmp_path)
    assert errors == []


# =============================================================================
# Membership contract the gate (and the runtime guard) key on
# =============================================================================


def test_randomization_membership_distinguishes_typo(tmp_path):
    """Loaded randomization keys verbatim: the typo is absent, the real ID present."""
    rand = _write_randomization(tmp_path / "rand.tsv")
    randomization = load_randomization(rand)
    assert "rtchr001" not in randomization
    assert "sub-rtchr01" in randomization


def test_lookup_default_is_why_callers_must_check_membership():
    """``get_condition_for_participant`` defaults unknown IDs to 'real' by design.

    This permissive default is intentional for the helper, which is exactly why a
    sham study must reject an unlisted participant *before* this lookup (the gate
    above and the runtime guard in pipeline_runner). Documents the contract so the
    default is not "fixed" in a way that surprises legitimate non-sham callers.
    """
    info = get_condition_for_participant({}, "rtchr001")
    assert info["condition"] == "real"


# =============================================================================
# Slot lists -- allocation by scan order, not by study ID
# =============================================================================


def _write_slot_list(path):
    """Write a new pre-allocated list (no subject_id column yet): 001 real, 002 sham replaying 001, 003 real."""
    pd.DataFrame(
        {
            "slot": ["001", "002", "003"],
            "condition": ["real", "sham", "real"],
            "matched_slot": ["", "001", ""],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def test_slot_assignment_gives_each_id_its_slots_condition(tmp_path):
    """Gapped study IDs, assigned in scan order, receive the slots' pre-allocated conditions."""
    rand = _write_slot_list(tmp_path / "slots.tsv")
    cfg = _study_config(rand)
    scanned = ["sub-charms255", "sub-charms270", "sub-charms271"]
    for pid in scanned:
        assert validate_session_start(cfg, pid, 0, tmp_path) == []
        assert pending_slot_assignment(cfg, pid) is not None
        assign_next_slot(rand, pid)

    randomization = load_randomization(rand)
    conditions = [get_condition_for_participant(randomization, pid)["condition"] for pid in scanned]
    assert conditions == ["real", "sham", "real"]
    # The sham participant replays whoever took slot 001.
    assert randomization["sub-charms270"]["matched_participant"] == "sub-charms255"


def test_assigned_participant_is_not_reassigned(tmp_path):
    """Once assigned, a participant is listed: no pending slot, and a second assignment is refused."""
    rand = _write_slot_list(tmp_path / "slots.tsv")
    cfg = _study_config(rand)
    assert assign_next_slot(rand, "sub-charms255") == "001"
    assert pending_slot_assignment(cfg, "sub-charms255") is None
    with pytest.raises(ValueError, match="already listed"):
        assign_next_slot(rand, "sub-charms255")
    assert load_randomization(rand)["sub-charms255"]["condition"] == "real"
