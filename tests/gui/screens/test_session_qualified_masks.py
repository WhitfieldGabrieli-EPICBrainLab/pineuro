"""Tests for session-qualified mask selection in the study editor.

Covers the feedback tab's source-qualified dropdown encoding and the study
editor's cross-session mask validation, both of which back per-feedback reuse of
an earlier session's localizer masks.
"""

from pineuro.gui.screens._feedback_tab import FeedbackTabBuilder, _mask_key, _split_mask_key
from pineuro.gui.screens.study_editor import StudyEditorScreen


class TestMaskKeyEncoding:
    def test_round_trips_qualified(self):
        key = _mask_key("ses-01", "Trigger")
        assert _split_mask_key(key) == ("ses-01", "Trigger")

    def test_round_trips_current_session(self):
        key = _mask_key("", "DefaultA")
        assert _split_mask_key(key) == ("", "DefaultA")

    def test_bare_name_parses_as_current(self):
        # A legacy bare name (no separator) is treated as the current session.
        assert _split_mask_key("DefaultA") == ("", "DefaultA")


class TestMaskValueFor:
    def _builder(self, mask_sources):
        # Bypass the heavy __init__; _mask_value_for only needs _mask_sources.
        b = FeedbackTabBuilder.__new__(FeedbackTabBuilder)
        b._mask_sources = mask_sources
        return b

    def test_resolves_source_from_mask_sources(self):
        b = self._builder({"Trigger": "ses-01"})
        assert b._mask_value_for("Trigger") == _mask_key("ses-01", "Trigger")

    def test_unqualified_when_absent(self):
        b = self._builder({})
        assert b._mask_value_for("DefaultA") == _mask_key("", "DefaultA")

    def test_none_for_falsy(self):
        assert self._builder({})._mask_value_for(None) is None
        assert self._builder({})._mask_value_for("") is None


class TestValidateSessions:
    def _validate(self, sessions):
        editor = StudyEditorScreen.__new__(StudyEditorScreen)
        return editor._validate_sessions(sessions)

    def _localizer_session(self):
        return {
            "localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b", "Movement": "c>d"}}],
            "feedback_runs": [{"task": "ritexperiencesampling", "network_assignment": {"target_network": "Trigger"}}],
        }

    def test_feedback_only_reuse_is_valid(self):
        sessions = [
            self._localizer_session(),
            {
                "localizers": [],
                "feedback_runs": [
                    {
                        "task": "rittriggeredball",
                        "network_assignment": {"trigger_network": "Trigger", "movement_network": "Movement"},
                        "mask_sources": {"Trigger": "ses-01", "Movement": "ses-01"},
                    }
                ],
            },
        ]
        assert self._validate(sessions) is None

    def test_unqualified_mask_must_be_produced_this_session(self):
        sessions = [
            {
                "localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}],
                "feedback_runs": [{"network_assignment": {"target_network": "Nope"}}],
            }
        ]
        err = self._validate(sessions)
        assert err is not None and "Nope" in err

    def test_source_must_produce_the_mask(self):
        sessions = [
            {"localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}], "feedback_runs": []},
            {
                "localizers": [],
                "feedback_runs": [{"network_assignment": {"t": "Movement"}, "mask_sources": {"Movement": "ses-01"}}],
            },
        ]
        err = self._validate(sessions)
        assert err is not None and "not produced by ses-01" in err

    def test_source_must_be_earlier(self):
        sessions = [
            {
                "localizers": [],
                "feedback_runs": [{"network_assignment": {"t": "Trigger"}, "mask_sources": {"Trigger": "ses-02"}}],
            },
            {"localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}], "feedback_runs": []},
        ]
        err = self._validate(sessions)
        assert err is not None and "not an earlier session" in err

    def test_collision_two_sources_rejected(self):
        sessions = [
            {"localizers": [{"id": "rit", "type": "rit", "contrasts": {"Trigger": "a>b"}}], "feedback_runs": []},
            {
                "localizers": [{"id": "rit2", "type": "rit", "contrasts": {"Trigger": "a>b"}}],
                "feedback_runs": [
                    # One run wants the fresh Trigger, another wants ses-01's Trigger.
                    {"network_assignment": {"t": "Trigger"}},
                    {"network_assignment": {"t": "Trigger"}, "mask_sources": {"Trigger": "ses-01"}},
                ],
            },
        ]
        err = self._validate(sessions)
        assert err is not None and "sourced from both" in err
