"""Tests for pineuro.gui.utils validation functions."""

import pytest

from pineuro.gui.utils import (
    ValidationError,
    validate_participant_id,
    validate_positive_int,
    validate_positive_number,
    validate_session_config,
)

# =============================================================================
# validate_participant_id
# =============================================================================


class TestValidateParticipantId:
    def test_empty_string_invalid(self):
        ok, msg = validate_participant_id("")
        assert not ok
        assert msg is not None

    def test_valid_simple_alphanumeric(self):
        ok, msg = validate_participant_id("sub01")
        assert ok
        assert msg is None

    def test_valid_bids_with_hyphen(self):
        ok, msg = validate_participant_id("sub-01")
        assert ok

    def test_valid_digits_only(self):
        ok, msg = validate_participant_id("001")
        assert ok

    def test_valid_single_char(self):
        ok, msg = validate_participant_id("A")
        assert ok

    def test_valid_uppercase(self):
        ok, msg = validate_participant_id("P001")
        assert ok

    def test_leading_hyphen_invalid(self):
        ok, msg = validate_participant_id("-sub01")
        assert not ok

    def test_space_invalid(self):
        ok, msg = validate_participant_id("my id")
        assert not ok

    def test_exclamation_mark_invalid(self):
        ok, msg = validate_participant_id("sub01!")
        assert not ok

    def test_underscore_invalid(self):
        ok, msg = validate_participant_id("sub_01")
        assert not ok

    def test_dot_invalid(self):
        ok, msg = validate_participant_id("sub.01")
        assert not ok

    def test_at_symbol_invalid(self):
        ok, msg = validate_participant_id("sub@01")
        assert not ok

    def test_error_message_is_string(self):
        ok, msg = validate_participant_id("")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_valid_returns_none_error(self):
        ok, msg = validate_participant_id("sub-01")
        assert msg is None


# =============================================================================
# validate_positive_number
# =============================================================================


class TestValidatePositiveNumber:
    def test_valid_float_string(self):
        assert validate_positive_number("1.5", "TR") == pytest.approx(1.5)

    def test_valid_integer_string(self):
        assert validate_positive_number("2", "TR") == pytest.approx(2.0)

    def test_valid_float_value(self):
        assert validate_positive_number(1.2, "TR") == pytest.approx(1.2)

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_number("0", "TR")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_number("-1.5", "TR")

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError, match="number"):
            validate_positive_number("abc", "TR")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_positive_number(None, "TR")

    def test_error_includes_name(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_number("abc", "MyField")
        assert "MyField" in str(exc_info.value)


# =============================================================================
# validate_positive_int
# =============================================================================


class TestValidatePositiveInt:
    def test_valid_int_string(self):
        assert validate_positive_int("5", "n_volumes") == 5

    def test_valid_int_value(self):
        assert validate_positive_int(10, "n_volumes") == 10

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_int("0", "n_volumes")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive_int("-3", "n_volumes")

    def test_non_integer_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            validate_positive_int("abc", "n_volumes")

    def test_float_string_truncated(self):
        # int("1.5") raises ValueError → ValidationError
        with pytest.raises(ValidationError):
            validate_positive_int("1.5", "n_volumes")


# =============================================================================
# validate_session_config
# =============================================================================


class TestValidateSessionConfig:
    def _valid_config(self):
        return {
            "localizers": [
                {
                    "id": "rest",
                    "type": "rest",
                    "tr": 1.5,
                    "n_volumes": 200,
                    "n_runs": 2,
                    "networks": ["Default", "Cont"],
                    "extraction": {
                        "n_yeo": 7,
                        "smooth_fwhm": 5.0,
                        "highpass": 0.01,
                    },
                }
            ],
            "feedback_runs": [
                {
                    "task": "ball",
                    "mask_source": "rest",
                    "n_baseline": 10,
                    "n_volumes": 200,
                }
            ],
        }

    def test_valid_config_passes(self):
        ok, err = validate_session_config(self._valid_config())
        assert ok is True
        assert err is None

    def test_invalid_tr_fails(self):
        config = self._valid_config()
        config["localizers"][0]["tr"] = 0
        ok, err = validate_session_config(config)
        assert ok is False
        assert err is not None

    def test_negative_tr_fails(self):
        config = self._valid_config()
        config["localizers"][0]["tr"] = -1.0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_n_volumes_fails(self):
        config = self._valid_config()
        config["localizers"][0]["n_volumes"] = -1
        ok, err = validate_session_config(config)
        assert ok is False

    def test_zero_n_runs_fails(self):
        config = self._valid_config()
        config["localizers"][0]["n_runs"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_single_run_fails(self):
        config = self._valid_config()
        config["localizers"][0]["n_runs"] = 1
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_n_yeo_fails(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["n_yeo"] = 5  # only 7 or 17
        ok, err = validate_session_config(config)
        assert ok is False

    def test_valid_n_yeo_17(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["n_yeo"] = 17
        ok, err = validate_session_config(config)
        assert ok is True

    def test_empty_networks_fails(self):
        config = self._valid_config()
        config["localizers"][0]["networks"] = []
        ok, err = validate_session_config(config)
        assert ok is False

    def test_single_network_fails(self):
        config = self._valid_config()
        config["localizers"][0]["networks"] = ["Default"]
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_smooth_fwhm_fails(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["smooth_fwhm"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_highpass_fails(self):
        config = self._valid_config()
        config["localizers"][0]["extraction"]["highpass"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_invalid_feedback_n_baseline_fails(self):
        config = self._valid_config()
        config["feedback_runs"][0]["n_baseline"] = 0
        ok, err = validate_session_config(config)
        assert ok is False

    def test_missing_feedback_runs_fails(self):
        config = self._valid_config()
        config["feedback_runs"] = []
        ok, err = validate_session_config(config)
        assert ok is False

    def test_error_message_is_string(self):
        config = self._valid_config()
        config["localizers"][0]["tr"] = 0
        ok, err = validate_session_config(config)
        assert isinstance(err, str)
        assert len(err) > 0
