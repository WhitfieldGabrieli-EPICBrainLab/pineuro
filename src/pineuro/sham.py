"""Sham neurofeedback delivery for control conditions.

This module provides functionality for delivering sham (control) feedback in
neurofeedback studies. It supports two delivery methods:

1. Alternative Masks: Use different network masks (e.g., motor instead of DMN)
   for sham participants while computing and logging real activations.

2. Replay: Replay activation values from another participant's feedbacklog.tsv
   while computing and logging real activations.

Critical Blinding Requirements
-----------------------------
- All masks (target + alternative) are extracted for ALL participants
- Real activations are always computed (even for sham participants)
- Both real and delivered activations are logged in feedbacklog.tsv
- Task and GUI display show delivered values only (maintains blinding)
- Condition is only revealed post-hoc in the feedbacklog

Components
----------
- ShamActivationProvider: Base class for sham activation sources
- ReplayActivationProvider: Replays activations from another participant's feedbacklog
- load_randomization(): Load subject -> condition assignments from TSV
- resolve_sham_masks(): Get alternative mask paths for sham condition
- find_feedbacklog_for_replay(): Locate matched participant's feedbacklog.tsv
- get_condition_for_participant(): Look up condition assignment

Randomization File Format
------------------------
TSV file with columns:
- subject_id: Participant identifier (e.g., "sub-01")
- condition: "real" or "sham"
- matched_participant: For replay method - which participant's data to replay
- matched_visit: For replay method - which visit's data (optional, defaults to same)
- notes: Optional notes/comments

Example:
    subject_id    condition    matched_participant    matched_visit    notes
    sub-01        real                                                 First participant
    sub-02        sham         sub-01                 ses-01           Replay sub-01 data
    sub-03        real
    sub-04        sham                                                 Alt masks (no match)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShamActivationProvider(ABC):
    """Base class for providing sham activation values.

    Sham providers generate activation values to be delivered to participants
    in the sham/control condition, while real activations are still computed
    and logged for analysis.
    """

    @abstractmethod
    def get_activation(self, volume: int, elapsed_time: float) -> dict[str, float]:
        """Return sham activation values for the given volume/time.

        Parameters
        ----------
        volume : int
            Current volume number (1-indexed)
        elapsed_time : float
            Elapsed time since session start in seconds

        Returns
        -------
        dict[str, float]
            Activation values keyed by mask name
        """
        pass


class ReplayActivationProvider(ShamActivationProvider):
    """Replays activation values from another participant's feedbacklog.tsv.

    This provider loads a source participant's feedbacklog and returns their
    activation values for each volume. If the current session has more volumes
    than the source, it loops back to the beginning.

    Parameters
    ----------
    feedbacklog_path : Path
        Path to source participant's feedbacklog.tsv
    mask_names : list[str]
        Expected mask names (must match columns in source feedbacklog)

    Raises
    ------
    FileNotFoundError
        If feedbacklog_path does not exist
    ValueError
        If required mask columns are missing from source feedbacklog
    """

    def __init__(self, feedbacklog_path: Path, mask_names: list[str]) -> None:
        self.feedbacklog_path = Path(feedbacklog_path)
        self.mask_names = mask_names

        if not self.feedbacklog_path.exists():
            raise FileNotFoundError(
                f"Source feedbacklog not found: {self.feedbacklog_path}"
            )

        self._data = self._load_feedbacklog()
        self._n_source_volumes = len(self._data)

        logger.info(
            f"ReplayActivationProvider initialized with {self._n_source_volumes} "
            f"volumes from {self.feedbacklog_path}"
        )

    def _load_feedbacklog(self) -> pd.DataFrame:
        """Load and validate source feedbacklog."""
        df = pd.read_csv(self.feedbacklog_path, sep='\t')

        # Check for required columns
        # Handle both old format (mask names directly) and new format (mask_delivered)
        missing_masks = []
        self._column_mapping = {}

        for name in self.mask_names:
            # Try new format first (mask_delivered), then old format (mask name only)
            if f"{name}_delivered" in df.columns:
                self._column_mapping[name] = f"{name}_delivered"
            elif name in df.columns:
                self._column_mapping[name] = name
            else:
                missing_masks.append(name)

        if missing_masks:
            raise ValueError(
                f"Mask columns not found in feedbacklog: {missing_masks}. "
                f"Available columns: {df.columns.tolist()}"
            )

        return df

    def get_activation(self, volume: int, elapsed_time: float) -> dict[str, float]:
        """Get activation values for volume, looping if source is exhausted.

        Parameters
        ----------
        volume : int
            Current volume number (1-indexed)
        elapsed_time : float
            Elapsed time since session start (not used, for interface compatibility)

        Returns
        -------
        dict[str, float]
            Activation values for each mask
        """
        # Convert to 0-indexed and loop if needed
        source_idx = (volume - 1) % self._n_source_volumes

        if volume > self._n_source_volumes:
            logger.debug(
                f"Volume {volume} exceeds source length {self._n_source_volumes}, "
                f"looping to index {source_idx}"
            )

        row = self._data.iloc[source_idx]
        return {
            name: float(row[self._column_mapping[name]])
            for name in self.mask_names
        }


def load_randomization(file_path: Path) -> dict[str, dict]:
    """Load subject -> condition assignments from TSV/CSV file.

    Parameters
    ----------
    file_path : Path
        Path to randomization file (TSV or CSV)

    Returns
    -------
    dict[str, dict]
        Mapping from subject_id to condition info:
        {
            "sub-01": {
                "condition": "real",
                "matched_participant": None,
                "matched_visit": None,
            },
            "sub-02": {
                "condition": "sham",
                "matched_participant": "sub-01",
                "matched_visit": "ses-01",
            },
            ...
        }

    Raises
    ------
    FileNotFoundError
        If file_path does not exist
    ValueError
        If required columns are missing or condition values are invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Randomization file not found: {file_path}")

    # Determine separator from extension
    sep = '\t' if file_path.suffix.lower() in ('.tsv', '.txt') else ','
    df = pd.read_csv(file_path, sep=sep)

    # Validate required columns
    if 'subject_id' not in df.columns:
        raise ValueError("Randomization file must have 'subject_id' column")
    if 'condition' not in df.columns:
        raise ValueError("Randomization file must have 'condition' column")

    result = {}
    for _, row in df.iterrows():
        subject_id = str(row['subject_id'])
        condition = str(row['condition']).lower()

        if condition not in ('real', 'sham'):
            raise ValueError(
                f"Invalid condition '{condition}' for {subject_id}. "
                f"Must be 'real' or 'sham'"
            )

        matched_participant = row.get('matched_participant')
        matched_visit = row.get('matched_visit')

        # Convert NaN to None
        if pd.isna(matched_participant):
            matched_participant = None
        else:
            matched_participant = str(matched_participant)

        if pd.isna(matched_visit):
            matched_visit = None
        else:
            matched_visit = str(matched_visit)

        result[subject_id] = {
            'condition': condition,
            'matched_participant': matched_participant,
            'matched_visit': matched_visit,
        }

    logger.info(f"Loaded randomization for {len(result)} participants from {file_path}")
    return result


def get_condition_for_participant(
    randomization: dict[str, dict],
    participant_id: str,
    visit_id: Optional[str] = None,
    default_condition: str = "real",
) -> dict:
    """Look up condition assignment for a participant.

    Parameters
    ----------
    randomization : dict
        Randomization mapping from load_randomization()
    participant_id : str
        Participant identifier (e.g., "sub-01")
    visit_id : str, optional
        Current visit identifier (used as default for matched_visit)
    default_condition : str
        Condition to use if participant not in randomization (default: "real")

    Returns
    -------
    dict
        Condition info with keys: condition, matched_participant, matched_visit
    """
    if participant_id in randomization:
        info = randomization[participant_id].copy()
        # Default matched_visit to current visit if not specified
        if info.get('matched_visit') is None and visit_id is not None:
            info['matched_visit'] = visit_id
        return info

    logger.warning(
        f"Participant {participant_id} not in randomization, "
        f"defaulting to '{default_condition}'"
    )
    return {
        'condition': default_condition,
        'matched_participant': None,
        'matched_visit': None,
    }


def resolve_sham_masks(
    alternative_mask_targets: list[str],
    mask_extraction_dir: Path,
    participant_id: str,
    twovol_subdir: str = "04_2vol",
) -> list[Path]:
    """Get alternative mask paths for sham condition.

    Returns mask file paths for the alternative_mask_targets networks,
    which will be used instead of the real target networks for sham participants.

    Parameters
    ----------
    alternative_mask_targets : list[str]
        Network names to use for sham (e.g., ["SomMotA", "SomMotB"])
    mask_extraction_dir : Path
        Directory containing mask extraction outputs
    participant_id : str
        Participant identifier
    twovol_subdir : str
        Subdirectory containing 2vol-coregistered masks (default: "04_2vol")

    Returns
    -------
    list[Path]
        Paths to alternative mask files (in same order as alternative_mask_targets)

    Raises
    ------
    FileNotFoundError
        If any alternative mask file is not found
    """
    mask_extraction_dir = Path(mask_extraction_dir)
    masks_dir = mask_extraction_dir / twovol_subdir / "masks"

    if not masks_dir.exists():
        # Try looking in 03_masks directly (for non-2vol-registered masks)
        masks_dir = mask_extraction_dir / "03_masks"

    mask_files = []
    for target in alternative_mask_targets:
        # Look for thresholded mask file with various naming patterns
        patterns = [
            f"{participant_id}_{target}_thr_to_*.nii.gz",  # 2vol coregistered
            f"{participant_id}_{target}_thr.nii.gz",       # Non-coregistered
            f"*_{target}_thr*.nii.gz",                     # Any matching pattern
        ]

        found = False
        for pattern in patterns:
            matches = list(masks_dir.glob(pattern))
            if matches:
                mask_files.append(matches[0])
                found = True
                logger.debug(f"Found alternative mask for {target}: {matches[0]}")
                break

        if not found:
            raise FileNotFoundError(
                f"Alternative mask for '{target}' not found in {masks_dir}. "
                f"Tried patterns: {patterns}"
            )

    return mask_files


def find_feedbacklog_for_replay(
    source_dir: Path,
    matched_participant: str,
    matched_visit: Optional[str],
    task: str,
    run: int,
) -> Path:
    """Locate matched participant's feedbacklog.tsv for replay.

    Parameters
    ----------
    source_dir : Path
        Base directory containing participant data
    matched_participant : str
        Participant ID to replay from (e.g., "sub-01")
    matched_visit : str, optional
        Visit ID to replay from (e.g., "ses-01"). If None, searches without visit.
    task : str
        Task name (e.g., "feedback", "transfer")
    run : int
        Run number (e.g., 1 for run-01)

    Returns
    -------
    Path
        Path to matched feedbacklog.tsv

    Raises
    ------
    FileNotFoundError
        If feedbacklog is not found
    """
    source_dir = Path(source_dir)

    # Build search patterns based on whether visit is specified
    if matched_visit:
        # Multi-visit: source_dir/participant/visit/feedback/02_feedback_output/
        base_path = source_dir / matched_participant / matched_visit / "feedback" / "02_feedback_output"
    else:
        # Single-visit: source_dir/participant/feedback/02_feedback_output/
        base_path = source_dir / matched_participant / "feedback" / "02_feedback_output"

    # Try various naming patterns
    run_str = f"run-{run:02d}"
    patterns = [
        f"{matched_participant}_task-{task}_{run_str}_feedbacklog.tsv",
        f"{matched_participant}_task-{task}_{run_str}_acq-*_feedbacklog.tsv",
        f"*_task-{task}_{run_str}*_feedbacklog.tsv",
    ]

    for pattern in patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            # Return most recent if multiple matches
            feedbacklog = sorted(matches)[-1]
            logger.info(f"Found feedbacklog for replay: {feedbacklog}")
            return feedbacklog

    # Try broader search if specific patterns fail
    all_logs = list(base_path.glob("*feedbacklog.tsv"))
    if all_logs:
        # Filter by task AND run number
        task_run_logs = [p for p in all_logs if f"task-{task}" in p.name and run_str in p.name]
        if task_run_logs:
            feedbacklog = sorted(task_run_logs)[-1]
            logger.info(f"Found feedbacklog for replay (broad search): {feedbacklog}")
            return feedbacklog

    raise FileNotFoundError(
        f"Feedbacklog not found for replay. "
        f"Searched in: {base_path} "
        f"Matched participant: {matched_participant}, "
        f"Visit: {matched_visit}, Task: {task}, Run: {run}"
    )
