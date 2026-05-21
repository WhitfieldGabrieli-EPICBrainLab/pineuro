"""Validation — sham activation replay.

ReplayActivationProvider must return a source participant's logged activations
volume-for-volume, looping when the current session outlasts the source.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pineuro.sham import ReplayActivationProvider

pytestmark = pytest.mark.validation


def _write_feedbacklog(path, values):
    """Write a minimal feedbacklog.tsv with a {mask}_delivered column."""
    frame = pd.DataFrame(
        {
            "volume": list(range(1, len(values) + 1)),
            "roi_delivered": values,
        }
    )
    frame.to_csv(path, sep="\t", index=False)
    return path


def test_replay_returns_source_values(tmp_path, record):
    """Sham mode replays a recorded participant's feedback unchanged."""
    values = [0.10, 0.52, -0.31, 0.84, 0.27]
    log = _write_feedbacklog(tmp_path / "feedbacklog.tsv", values)
    provider = ReplayActivationProvider(log, ["roi"])

    max_err = 0.0
    for volume in range(1, len(values) + 1):
        delivered = provider.get_activation(volume, elapsed_time=0.0)["roi"]
        max_err = max(max_err, abs(delivered - values[volume - 1]))
    assert (
        record(
            "sham",
            "replay_value_error",
            max_err,
            threshold=0.0,
            op="==",
            description=(
                "Sham feedback replays the values logged for an earlier "
                "participant. This is the largest gap between what is replayed and "
                "what was logged; it must be zero."
            ),
        )
        == 0.0
    )


def test_replay_loops_when_exhausted(tmp_path):
    """When the session outlasts the source, replay wraps modulo source length."""
    values = [1.0, 2.0, 3.0]
    log = _write_feedbacklog(tmp_path / "feedbacklog.tsv", values)
    provider = ReplayActivationProvider(log, ["roi"])

    # volume v maps to source index (v - 1) % 3; replay returns the stored value verbatim.
    assert provider.get_activation(4, 0.0)["roi"] == values[0]
    assert provider.get_activation(5, 0.0)["roi"] == values[1]
    assert provider.get_activation(9, 0.0)["roi"] == values[2]


def test_replay_supports_plain_mask_columns(tmp_path):
    """A feedbacklog with bare mask-name columns (no _delivered) also replays."""
    frame = pd.DataFrame({"volume": [1, 2], "roi": [0.4, 0.9]})
    log = tmp_path / "feedbacklog.tsv"
    frame.to_csv(log, sep="\t", index=False)
    provider = ReplayActivationProvider(log, ["roi"])
    assert provider.get_activation(1, 0.0)["roi"] == 0.4  # verbatim TSV round-trip
    assert provider.get_activation(2, 0.0)["roi"] == 0.9
