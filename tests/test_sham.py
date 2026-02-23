"""Tests for pineuro.sham — randomization, condition lookup, replay provider."""

import pandas as pd
import pytest

from pineuro.sham import (
    ReplayActivationProvider,
    generate_randomization,
    get_condition_for_participant,
    load_randomization,
)

# =============================================================================
# generate_randomization
# =============================================================================


class TestGenerateRandomization:
    def test_simple_method(self):
        ids = [f"sub-{i:02d}" for i in range(1, 11)]
        df = generate_randomization(ids, n_sham=3, method="simple", seed=42)

        assert len(df) == 10
        assert set(df.columns) >= {"subject_id", "condition"}
        assert (df["condition"] == "sham").sum() == 3
        assert (df["condition"] == "real").sum() == 7

    def test_replay_method_first_is_real(self):
        ids = [f"sub-{i:02d}" for i in range(1, 11)]
        df = generate_randomization(ids, n_sham=5, method="replay", seed=42)

        # First participant must always be real (for replay to work)
        assert df.iloc[0]["condition"] == "real"

    def test_replay_has_matched_participant(self):
        ids = [f"sub-{i:02d}" for i in range(1, 7)]
        df = generate_randomization(ids, n_sham=2, method="replay", matched_visit="ses-01", seed=42)

        sham_rows = df[df["condition"] == "sham"]
        for _, row in sham_rows.iterrows():
            assert pd.notna(row["matched_participant"])
            assert row["matched_visit"] == "ses-01"

    def test_balanced_split(self):
        ids = [f"sub-{i:02d}" for i in range(1, 21)]
        df = generate_randomization(ids, n_sham=10, method="simple", seed=42)
        assert (df["condition"] == "sham").sum() == 10
        assert (df["condition"] == "real").sum() == 10

    def test_reproducible_with_seed(self):
        ids = [f"sub-{i:02d}" for i in range(1, 11)]
        df1 = generate_randomization(ids, n_sham=3, seed=123)
        df2 = generate_randomization(ids, n_sham=3, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_result(self):
        ids = [f"sub-{i:02d}" for i in range(1, 11)]
        df1 = generate_randomization(ids, n_sham=3, seed=1)
        df2 = generate_randomization(ids, n_sham=3, seed=2)
        # Conditions should differ (extremely unlikely to be same with different seeds)
        assert not df1["condition"].equals(df2["condition"])

    def test_writes_to_file(self, tmp_path):
        ids = ["sub-01", "sub-02", "sub-03"]
        output = tmp_path / "randomization.tsv"
        generate_randomization(ids, n_sham=1, output_path=output, seed=42)
        assert output.exists()
        loaded = pd.read_csv(output, sep="\t")
        assert len(loaded) == 3

    def test_invalid_n_sham_zero(self):
        with pytest.raises(ValueError):
            generate_randomization(["sub-01", "sub-02"], n_sham=0)

    def test_invalid_n_sham_all(self):
        with pytest.raises(ValueError):
            generate_randomization(["sub-01", "sub-02"], n_sham=2)

    def test_empty_participants(self):
        with pytest.raises(ValueError):
            generate_randomization([], n_sham=1)

    def test_duplicate_participants(self):
        with pytest.raises(ValueError):
            generate_randomization(["sub-01", "sub-01"], n_sham=1)


# =============================================================================
# load_randomization
# =============================================================================


class TestLoadRandomization:
    def test_roundtrip(self, tmp_path):
        ids = [f"sub-{i:02d}" for i in range(1, 6)]
        output = tmp_path / "rand.tsv"
        generate_randomization(ids, n_sham=2, method="simple", output_path=output, seed=42)

        result = load_randomization(output)
        assert len(result) == 5
        assert all(r["condition"] in ("real", "sham") for r in result.values())

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_randomization(tmp_path / "nonexistent.tsv")

    def test_missing_subject_id_column(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("condition\nreal\n")
        with pytest.raises(ValueError, match="subject_id"):
            load_randomization(f)

    def test_invalid_condition(self, tmp_path):
        f = tmp_path / "bad.tsv"
        f.write_text("subject_id\tcondition\nsub-01\tinvalid\n")
        with pytest.raises(ValueError, match="Invalid condition"):
            load_randomization(f)


# =============================================================================
# get_condition_for_participant
# =============================================================================


class TestGetConditionForParticipant:
    def test_real_participant(self):
        randomization = {
            "sub-01": {"condition": "real", "matched_participant": None, "matched_visit": None},
        }
        result = get_condition_for_participant(randomization, "sub-01")
        assert result["condition"] == "real"

    def test_sham_participant(self):
        randomization = {
            "sub-02": {"condition": "sham", "matched_participant": "sub-01", "matched_visit": "ses-01"},
        }
        result = get_condition_for_participant(randomization, "sub-02")
        assert result["condition"] == "sham"
        assert result["matched_participant"] == "sub-01"

    def test_missing_defaults_to_real(self):
        randomization = {}
        result = get_condition_for_participant(randomization, "sub-99")
        assert result["condition"] == "real"


# =============================================================================
# ReplayActivationProvider
# =============================================================================


class TestReplayActivationProvider:
    def _make_feedbacklog(self, tmp_path, n_volumes=10, mask_names=None):
        if mask_names is None:
            mask_names = ["DMN"]
        lines = ["volume\telapsed_time"]
        for name in mask_names:
            lines[0] += f"\t{name}"
        for i in range(n_volumes):
            row = f"{i + 1}\t{i * 2.0:.3f}"
            for j, _ in enumerate(mask_names):
                row += f"\t{(i + 1) * 0.1 + j:.4f}"
            lines.append(row)
        path = tmp_path / "feedbacklog.tsv"
        path.write_text("\n".join(lines))
        return path

    def test_basic_replay(self, tmp_path):
        path = self._make_feedbacklog(tmp_path, n_volumes=5, mask_names=["DMN"])
        provider = ReplayActivationProvider(path, mask_names=["DMN"])

        result = provider.get_activation(volume=1, elapsed_time=0.0)
        assert "DMN" in result
        assert isinstance(result["DMN"], float)

    def test_sequential_values(self, tmp_path):
        path = self._make_feedbacklog(tmp_path, n_volumes=5, mask_names=["DMN"])
        provider = ReplayActivationProvider(path, mask_names=["DMN"])

        v1 = provider.get_activation(1, 0.0)["DMN"]
        v2 = provider.get_activation(2, 2.0)["DMN"]
        assert v1 != v2  # Different volumes should give different values

    def test_loops_when_exhausted(self, tmp_path):
        path = self._make_feedbacklog(tmp_path, n_volumes=3, mask_names=["DMN"])
        provider = ReplayActivationProvider(path, mask_names=["DMN"])

        v1 = provider.get_activation(1, 0.0)["DMN"]
        v4 = provider.get_activation(4, 6.0)["DMN"]  # Should loop back to volume 1
        assert v1 == v4

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReplayActivationProvider(tmp_path / "nope.tsv", mask_names=["DMN"])

    def test_missing_mask_column_raises(self, tmp_path):
        path = self._make_feedbacklog(tmp_path, mask_names=["DMN"])
        with pytest.raises(ValueError, match="Mask columns not found"):
            ReplayActivationProvider(path, mask_names=["Motor"])
