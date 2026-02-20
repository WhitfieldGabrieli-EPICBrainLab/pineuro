"""Fixtures for integration tests requiring Rockland_NFB test data.

Integration tests are skipped automatically when the data is not present.
Run with: pytest -m integration
"""

from pathlib import Path

import pytest

# Resolve sourcedata relative to the pineuro package root (parent of tests/)
_SOURCEDATA = Path(__file__).resolve().parent.parent.parent / "sourcedata" / "Rockland_NFB"

# First available subject for single-subject tests
_FIRST_SUBJECT = "A00059346"


@pytest.fixture
def rockland_root():
    """Path to Rockland_NFB sourcedata directory. Skips if not present."""
    if not _SOURCEDATA.is_dir():
        pytest.skip("Rockland_NFB test data not available")
    return _SOURCEDATA


@pytest.fixture
def rockland_subject(rockland_root):
    """Path to first Rockland subject directory (A00059346)."""
    subject_dir = rockland_root / _FIRST_SUBJECT
    if not subject_dir.is_dir():
        pytest.skip(f"Subject {_FIRST_SUBJECT} not found in Rockland data")
    return subject_dir


@pytest.fixture
def rockland_rest_dicoms(rockland_subject):
    """Path to resting-state DICOMs for first subject."""
    rest_dir = rockland_subject / "rest_1400"
    if not rest_dir.is_dir():
        pytest.skip("rest_1400 DICOMs not found")
    return rest_dir


@pytest.fixture
def rockland_task_dicoms(rockland_subject):
    """Path to DMN tracking test DICOMs for first subject."""
    task_dir = rockland_subject / "dmn_tracking_test"
    if not task_dir.is_dir():
        pytest.skip("dmn_tracking_test DICOMs not found")
    return task_dir
