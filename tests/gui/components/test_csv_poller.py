"""Tests for pineuro.gui.components.csv_poller — read_live_metrics."""

from pathlib import Path

from pineuro.gui.components.csv_poller import read_live_metrics
from pineuro.tasks.task_registry import LiveMetric


def _write_log(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines))


class TestReadLiveMetrics:
    def test_glob_pattern_matches_multiple_columns(self, tmp_path):
        log = tmp_path / "psychopylog.tsv"
        _write_log(
            log,
            ["frame", "time", "volume", "ContA_hits", "DefaultA_hits"],
            [
                ["1", "0.0", "1", "0", "0"],
                ["2", "1.0", "2", "3", "1"],
            ],
        )
        metrics = [
            LiveMetric(
                key="hits",
                label="Hits",
                column_pattern="*_hits",
                name_strip_suffix="_hits",
            )
        ]
        out = read_live_metrics(log, metrics)
        assert out == {"hits": {"ContA": "3", "DefaultA": "1"}}

    def test_literal_column_with_value_format(self, tmp_path):
        log = tmp_path / "psychopylog.tsv"
        _write_log(
            log,
            ["frame", "time", "volume", "score"],
            [
                ["1", "0.0", "1", "0.0"],
                ["2", "1.0", "2", "12.34"],
            ],
        )
        metrics = [
            LiveMetric(
                key="score",
                label="Score",
                column_pattern="score",
                value_format="{value:.1f}",
            )
        ]
        out = read_live_metrics(log, metrics)
        assert out == {"score": {"score": "12.3"}}

    def test_missing_file_returns_empty(self, tmp_path):
        log = tmp_path / "missing.tsv"
        metrics = [LiveMetric(key="hits", label="Hits", column_pattern="*_hits")]
        assert read_live_metrics(log, metrics) == {}

    def test_empty_metrics_returns_empty(self, tmp_path):
        log = tmp_path / "psychopylog.tsv"
        _write_log(log, ["frame", "score"], [["1", "2.0"]])
        assert read_live_metrics(log, []) == {}

    def test_header_only_returns_empty(self, tmp_path):
        log = tmp_path / "psychopylog.tsv"
        log.write_text("frame\tscore\n")
        metrics = [LiveMetric(key="score", label="Score", column_pattern="score")]
        assert read_live_metrics(log, metrics) == {}

    def test_no_matching_columns_omits_metric(self, tmp_path):
        log = tmp_path / "psychopylog.tsv"
        _write_log(log, ["frame", "score"], [["1", "0.0"], ["2", "5.5"]])
        metrics = [
            LiveMetric(key="hits", label="Hits", column_pattern="*_hits"),
            LiveMetric(key="score", label="Score", column_pattern="score"),
        ]
        out = read_live_metrics(log, metrics)
        assert "hits" not in out
        assert out["score"] == {"score": "5.5"}

    def test_multiple_metrics_independent(self, tmp_path):
        log = tmp_path / "psychopylog.tsv"
        _write_log(
            log,
            ["frame", "triggered_prompts", "timeout_prompts"],
            [["1", "0", "0"], ["2", "2", "1"]],
        )
        metrics = [
            LiveMetric(key="triggered", label="Triggered", column_pattern="triggered_prompts"),
            LiveMetric(key="timeout", label="Timeout", column_pattern="timeout_prompts"),
        ]
        out = read_live_metrics(log, metrics)
        assert out == {
            "triggered": {"triggered_prompts": "2"},
            "timeout": {"timeout_prompts": "1"},
        }
