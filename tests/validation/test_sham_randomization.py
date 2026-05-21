"""Validation -- sham randomization generation, loading, and lookup.

sham.py randomization functions are pure-Python lookup/assignment logic with
specific deterministic contracts:

- ``generate_randomization`` with a fixed seed produces a deterministic
  assignment; the structural invariants (correct count of each condition,
  no duplicates, replay matching constraints) are exact.
- ``load_randomization`` round-trips a TSV verbatim; every field a writer wrote
  comes back as the literal value (modulo NaN handling for empty cells).
- ``get_condition_for_participant`` is a pure dict lookup with a
  documented default for unknown IDs.

All expected values below are derivable from the function contracts and the
input data -- no formula is recomputed from pineuro's implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pineuro.sham import (
    generate_randomization,
    get_condition_for_participant,
    load_randomization,
)

pytestmark = pytest.mark.validation


# =============================================================================
# generate_randomization
# =============================================================================


def test_generate_assigns_exact_sham_count():
    """The number of sham assignments equals n_sham; the rest are real."""
    df = generate_randomization(
        participant_ids=[f"sub-{i:02d}" for i in range(1, 11)],
        n_sham=4,
        seed=0,
    )
    assert (df["condition"] == "sham").sum() == 4
    assert (df["condition"] == "real").sum() == 6  # 10 - 4


def test_generate_assigns_all_listed_ids():
    """Every input participant_id appears in the output exactly once."""
    ids = [f"sub-{i:02d}" for i in range(1, 8)]
    df = generate_randomization(participant_ids=ids, n_sham=3, seed=1)
    assert set(df["subject_id"]) == set(ids)
    assert len(df) == len(ids)


def test_generate_is_deterministic_under_seed():
    """Two calls with the same seed produce identical condition assignments."""
    ids = [f"sub-{i:02d}" for i in range(1, 11)]
    df_a = generate_randomization(participant_ids=ids, n_sham=4, seed=42)
    df_b = generate_randomization(participant_ids=ids, n_sham=4, seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_generate_rejects_zero_or_full_sham():
    """n_sham must be strictly between 0 and len(participants)."""
    ids = [f"sub-{i:02d}" for i in range(1, 5)]
    with pytest.raises(ValueError):
        generate_randomization(participant_ids=ids, n_sham=0, seed=0)
    with pytest.raises(ValueError):
        generate_randomization(participant_ids=ids, n_sham=len(ids), seed=0)


def test_generate_rejects_duplicates():
    """Duplicate participant_ids are rejected."""
    with pytest.raises(ValueError, match="duplicate"):
        generate_randomization(participant_ids=["sub-01", "sub-02", "sub-01"], n_sham=1, seed=0)


def test_generate_replay_first_subject_is_real():
    """method='replay' guarantees the first participant is 'real' (replay needs prior data)."""
    ids = [f"sub-{i:02d}" for i in range(1, 11)]
    # With many seeds we should see at least one where sub-01 would have been sham
    # if not for the swap; here we just assert the postcondition every time.
    for seed in range(5):
        df = generate_randomization(participant_ids=ids, n_sham=5, method="replay", seed=seed)
        assert df.iloc[0]["condition"] == "real"


def test_generate_replay_every_sham_has_matched_participant():
    """method='replay' assigns a matched real participant to every sham participant."""
    ids = [f"sub-{i:02d}" for i in range(1, 11)]
    df = generate_randomization(participant_ids=ids, n_sham=4, method="replay", matched_visit="ses-01", seed=7)
    sham_rows = df[df["condition"] == "sham"]
    real_ids = set(df[df["condition"] == "real"]["subject_id"])
    for _, row in sham_rows.iterrows():
        assert row["matched_participant"] in real_ids
        assert row["matched_visit"] == "ses-01"


def test_generate_simple_method_leaves_matched_columns_empty():
    """method='simple' produces no matched_participant / matched_visit values."""
    ids = [f"sub-{i:02d}" for i in range(1, 6)]
    df = generate_randomization(participant_ids=ids, n_sham=2, method="simple", seed=0)
    assert df["matched_participant"].isna().all()
    assert df["matched_visit"].isna().all()


def test_generate_writes_tsv_when_path_given(tmp_path):
    """When output_path is given, the TSV is written and is round-trippable via load."""
    ids = [f"sub-{i:02d}" for i in range(1, 6)]
    path = tmp_path / "randomization.tsv"
    df = generate_randomization(
        participant_ids=ids,
        n_sham=2,
        method="replay",
        matched_visit="ses-01",
        output_path=path,
        seed=99,
    )
    assert path.exists()

    # Round-trip back via load_randomization: every assignment survives verbatim.
    loaded = load_randomization(path)
    for _, row in df.iterrows():
        sid = row["subject_id"]
        assert loaded[sid]["condition"] == row["condition"]
        # matched_participant: NaN in df -> None in loaded; non-NaN -> exact match.
        if pd.isna(row["matched_participant"]):
            assert loaded[sid]["matched_participant"] is None
        else:
            assert loaded[sid]["matched_participant"] == row["matched_participant"]


# =============================================================================
# load_randomization
# =============================================================================


def _write_tsv(path, rows):
    """Write rows (list of dicts) to a TSV at the given path."""
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_load_round_trips_verbatim(tmp_path):
    """Each cell loaded matches the literal value written, for known inputs."""
    rows = [
        {
            "subject_id": "sub-01",
            "condition": "real",
            "matched_participant": "",
            "matched_visit": "",
        },
        {
            "subject_id": "sub-02",
            "condition": "sham",
            "matched_participant": "sub-01",
            "matched_visit": "ses-02",
        },
    ]
    path = _write_tsv(tmp_path / "r.tsv", rows)
    loaded = load_randomization(path)
    assert set(loaded.keys()) == {"sub-01", "sub-02"}
    assert loaded["sub-01"]["condition"] == "real"
    assert loaded["sub-01"]["matched_participant"] is None  # empty cell -> None
    assert loaded["sub-02"]["condition"] == "sham"
    assert loaded["sub-02"]["matched_participant"] == "sub-01"
    assert loaded["sub-02"]["matched_visit"] == "ses-02"


def test_load_rejects_invalid_condition(tmp_path):
    """A condition outside {'real', 'sham'} raises ValueError."""
    path = _write_tsv(
        tmp_path / "bad.tsv",
        [{"subject_id": "sub-01", "condition": "placebo", "matched_participant": "", "matched_visit": ""}],
    )
    with pytest.raises(ValueError, match="Invalid condition"):
        load_randomization(path)


def test_load_rejects_missing_columns(tmp_path):
    """Missing required columns raise ValueError."""
    path = _write_tsv(tmp_path / "no_subject.tsv", [{"condition": "real"}])
    with pytest.raises(ValueError, match="subject_id"):
        load_randomization(path)


def test_load_missing_file_raises(tmp_path):
    """Loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_randomization(tmp_path / "no_such_file.tsv")


# =============================================================================
# get_condition_for_participant
# =============================================================================


def test_lookup_known_participant_returns_their_assignment():
    """A known participant's stored condition is returned verbatim."""
    randomization = {
        "sub-01": {"condition": "real", "matched_participant": None, "matched_visit": None},
        "sub-02": {"condition": "sham", "matched_participant": "sub-01", "matched_visit": "ses-01"},
    }
    info = get_condition_for_participant(randomization, "sub-02", visit_id="ses-02")
    assert info["condition"] == "sham"
    assert info["matched_participant"] == "sub-01"
    # matched_visit was already set in the dict, so visit_id does not override.
    assert info["matched_visit"] == "ses-01"


def test_lookup_unknown_participant_returns_default():
    """An unknown participant falls back to default_condition with no matched info."""
    info = get_condition_for_participant({}, "sub-99", default_condition="real")
    assert info["condition"] == "real"
    assert info["matched_participant"] is None
    assert info["matched_visit"] is None


def test_lookup_inherits_visit_id_when_unset():
    """If matched_visit is unset in the randomization, the lookup uses visit_id."""
    randomization = {
        "sub-02": {"condition": "sham", "matched_participant": "sub-01", "matched_visit": None},
    }
    info = get_condition_for_participant(randomization, "sub-02", visit_id="ses-03")
    assert info["matched_visit"] == "ses-03"
