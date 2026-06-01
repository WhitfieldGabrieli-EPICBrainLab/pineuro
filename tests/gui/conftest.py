"""Shared fixtures for pineuro GUI tests.

`mock_flet_page` lives in the top-level tests/conftest.py; this adds small
file-writing helpers reused across the GUI test modules (CSV/TSV log fixtures).
"""

import pytest


@pytest.fixture
def write_tsv():
    """Factory: write a TSV file from a header row + data rows; returns the Path.

    Usage::

        def test_x(tmp_path, write_tsv):
            log = write_tsv(tmp_path / "log.tsv", ["volume", "ContA"], [["1", "0.5"]])
    """

    def _make(path, header, rows):
        lines = ["\t".join(header)] + ["\t".join(str(c) for c in row) for row in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    return _make
