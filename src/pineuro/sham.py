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
- matched_session: For replay method - which session's data (optional, defaults to same)
- notes: Optional notes/comments

Example:
    subject_id    condition    matched_participant    matched_session    notes
    sub-01        real                                                 First participant
    sub-02        sham         sub-01                 ses-01           Replay sub-01 data
    sub-03        real
    sub-04        sham                                                 Alt masks (no match)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

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
            raise FileNotFoundError(f"Source feedbacklog not found: {self.feedbacklog_path}")

        self._data = self._load_feedbacklog()
        self._n_source_volumes = len(self._data)
        self._loop_warned = False

        logger.info(
            f"ReplayActivationProvider initialized with {self._n_source_volumes} volumes from {self.feedbacklog_path}"
        )

    def _load_feedbacklog(self) -> pd.DataFrame:
        """Load and validate source feedbacklog."""
        df = pd.read_csv(self.feedbacklog_path, sep="\t")

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
                f"Mask columns not found in feedbacklog: {missing_masks}. Available columns: {df.columns.tolist()}"
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

        if volume > self._n_source_volumes and not self._loop_warned:
            logger.warning(
                f"Volume {volume} exceeds source session length ({self._n_source_volumes} volumes); "
                "looping from beginning. This warning is emitted once."
            )
            self._loop_warned = True

        row = self._data.iloc[source_idx]
        return {name: float(row[self._column_mapping[name]]) for name in self.mask_names}


def _nan_to_str(val) -> str | None:
    """Convert a pandas value to str, treating NaN as None."""
    return None if pd.isna(val) else str(val)


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
                "matched_session": None,
            },
            "sub-02": {
                "condition": "sham",
                "matched_participant": "sub-01",
                "matched_session": "ses-01",
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
    sep = "\t" if file_path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(file_path, sep=sep)

    # Validate required columns
    if "subject_id" not in df.columns:
        raise ValueError("Randomization file must have 'subject_id' column")
    if "condition" not in df.columns:
        raise ValueError("Randomization file must have 'condition' column")

    result = {}
    for _, row in df.iterrows():
        subject_id = str(row["subject_id"])
        condition = str(row["condition"]).lower()

        if condition not in ("real", "sham"):
            raise ValueError(f"Invalid condition '{condition}' for {subject_id}. Must be 'real' or 'sham'")

        matched_participant = _nan_to_str(row.get("matched_participant"))
        matched_session = _nan_to_str(row.get("matched_session"))

        result[subject_id] = {
            "condition": condition,
            "matched_participant": matched_participant,
            "matched_session": matched_session,
        }

    logger.info(f"Loaded randomization for {len(result)} participants from {file_path}")
    return result


def generate_randomization(
    participant_ids: list[str],
    n_sham: int,
    method: str = "simple",
    matched_session: str | None = None,
    output_path: Path | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a randomization file assigning participants to real/sham conditions.

    Creates a TSV-formatted DataFrame (and optionally writes it to disk) that is
    directly compatible with ``load_randomization()``.

    Parameters
    ----------
    participant_ids : list[str]
        Participant identifiers (e.g., ["sub-01", "sub-02", ..., "sub-20"])
    n_sham : int
        Number of participants to assign to the sham condition.
        Must be > 0 and < len(participant_ids).
    method : str
        Assignment method:
        - "simple": Random real/sham assignment with no replay matching.
        - "replay": Random assignment plus automatic sham→real participant
          matching for feedbacklog replay (round-robin if more sham than real).
          The first participant (participant_ids[0]) is always assigned "real"
          since replay requires prior data to exist.
    matched_session : str, optional
        Session ID to populate the matched_session column for replay participants.
        Ignored when method="simple".
    output_path : Path, optional
        If provided, write the randomization table to this path as TSV.
    seed : int, optional
        Random seed for reproducible assignments.

    Returns
    -------
    pd.DataFrame
        Randomization table with columns: subject_id, condition,
        matched_participant, matched_session, notes.

    Raises
    ------
    ValueError
        If n_sham is out of range or participant_ids contains duplicates.
    """
    # --- Validate inputs ---
    if len(participant_ids) == 0:
        raise ValueError("participant_ids must not be empty")
    if n_sham <= 0 or n_sham >= len(participant_ids):
        raise ValueError(
            f"n_sham must be between 1 and {len(participant_ids) - 1} "
            f"(got {n_sham} for {len(participant_ids)} participants)"
        )
    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError("participant_ids contains duplicates")

    rng = np.random.default_rng(seed)

    # --- Shuffle and assign conditions ---
    indices = np.arange(len(participant_ids))
    rng.shuffle(indices)

    conditions = ["real"] * len(participant_ids)
    for i in range(n_sham):
        conditions[indices[i]] = "sham"

    # First participant must be real for replay (no prior data to replay from)
    if method == "replay" and conditions[0] == "sham":
        first_real = next(i for i in range(len(conditions)) if conditions[i] == "real")
        conditions[0], conditions[first_real] = conditions[first_real], conditions[0]

    # --- Build matched_participant column for replay method ---
    matched_participants: list[str | None] = [None] * len(participant_ids)
    matched_sessions: list[str | None] = [None] * len(participant_ids)

    if method == "replay":
        real_ids = [participant_ids[i] for i in range(len(participant_ids)) if conditions[i] == "real"]
        sham_indices = [i for i in range(len(participant_ids)) if conditions[i] == "sham"]

        # Round-robin match sham → real
        for j, sham_idx in enumerate(sham_indices):
            matched_participants[sham_idx] = real_ids[j % len(real_ids)]
            if matched_session:
                matched_sessions[sham_idx] = matched_session

    # --- Build DataFrame ---
    df = pd.DataFrame(
        {
            "subject_id": participant_ids,
            "condition": conditions,
            "matched_participant": matched_participants,
            "matched_session": matched_sessions,
            "notes": [""] * len(participant_ids),
        }
    )

    # --- Write TSV if path provided ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Generated randomization file: {output_path} ({len(participant_ids)} participants, {n_sham} sham)")

    return df


def get_condition_for_participant(
    randomization: dict[str, dict],
    participant_id: str,
    session_id: str | None = None,
    default_condition: str = "real",
) -> dict:
    """Look up condition assignment for a participant.

    Parameters
    ----------
    randomization : dict
        Randomization mapping from load_randomization()
    participant_id : str
        Participant identifier (e.g., "sub-01")
    session_id : str, optional
        Current session identifier (used as default for matched_session)
    default_condition : str
        Condition to use if participant not in randomization (default: "real")

    Returns
    -------
    dict
        Condition info with keys: condition, matched_participant, matched_session
    """
    if participant_id in randomization:
        info = randomization[participant_id].copy()
        # Default matched_session to current session if not specified
        if info.get("matched_session") is None and session_id is not None:
            info["matched_session"] = session_id
        return info

    logger.warning(f"Participant {participant_id} not in randomization, defaulting to '{default_condition}'")
    return {
        "condition": default_condition,
        "matched_participant": None,
        "matched_session": None,
    }


def resolve_sham_masks(
    alternative_mask_targets: list[str],
    mask_extraction_dir: Path,
    participant_id: str,
    session_id: str,
    reference_subdir: str = "04_reference",
    sham_mask_mode: str = "two_networks",
    hemisphere_network: str | None = None,
    target_hemisphere: str = "right",
) -> list[Path]:
    """Get alternative mask paths for sham condition.

    Returns mask file paths for the alternative networks/hemispheres,
    which will be used instead of the real target networks for sham participants.

    Supports two modes:
    1. 'two_networks': Returns masks for two distinct Yeo networks (legacy)
    2. 'hemisphere_split': Returns left/right hemisphere masks from a single network

    Parameters
    ----------
    alternative_mask_targets : list[str]
        For 'two_networks' mode: network names to use for sham (e.g., ["SomMotA", "SomMotB"])
        Ignored for 'hemisphere_split' mode.
    mask_extraction_dir : Path
        Directory containing mask extraction outputs
    participant_id : str
        Participant identifier
    reference_subdir : str
        Subdirectory containing reference-coregistered masks (default: "04_reference")
    sham_mask_mode : str
        Mode for mask selection: "two_networks" or "hemisphere_split"
    hemisphere_network : str, optional
        For 'hemisphere_split' mode: the network that was split (e.g., "SomMotA")
    target_hemisphere : str
        For 'hemisphere_split' mode: which hemisphere is target ("right" or "left")

    Returns
    -------
    list[Path]
        Paths to alternative mask files. For hemisphere_split, returns
        [target_hemi_mask, non_target_hemi_mask].

    Raises
    ------
    FileNotFoundError
        If any alternative mask file is not found
    ValueError
        If hemisphere_split mode is used without hemisphere_network
    """
    mask_extraction_dir = Path(mask_extraction_dir)
    masks_dir = mask_extraction_dir / reference_subdir / "masks"

    if not masks_dir.exists():
        # Try looking in 03_masks directly (for non-reference-registered masks)
        masks_dir = mask_extraction_dir / "03_masks"

    # Build target list based on mode
    if sham_mask_mode == "hemisphere_split":
        if hemisphere_network is None:
            raise ValueError("hemisphere_network is required for hemisphere_split mode")
        # Determine non-target hemisphere
        non_target_hemisphere = "left" if target_hemisphere == "right" else "right"
        # Build targets: [target_hemi, non_target_hemi]
        targets = [
            f"{hemisphere_network}_{target_hemisphere}",
            f"{hemisphere_network}_{non_target_hemisphere}",
        ]
        logger.info(
            f"Hemisphere split mode: {hemisphere_network} -> "
            f"target={target_hemisphere}, non-target={non_target_hemisphere}"
        )
    else:
        # two_networks mode (legacy)
        targets = alternative_mask_targets

    bids_prefix = f"{participant_id}_{session_id}"
    mask_files = []
    for target in targets:
        # Look for thresholded mask file with various naming patterns
        patterns = [
            f"{bids_prefix}_{target}_thr_to_*.nii.gz",  # reference coregistered
            f"{bids_prefix}_{target}_thr.nii.gz",  # Non-coregistered
            f"*_{target}_thr*.nii.gz",  # Any matching pattern
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
                f"Alternative mask for '{target}' not found in {masks_dir}. Tried patterns: {patterns}"
            )

    return mask_files


def find_feedbacklog_for_replay(
    source_dir: Path,
    matched_participant: str,
    matched_session: str,
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
    matched_session : str
        Session ID to replay from (e.g., "ses-01").
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
    base_path = source_dir / matched_participant / matched_session / "feedback" / "02_feedback_output"

    # Try various naming patterns
    run_str = f"run-{run:02d}"
    bids_prefix = f"{matched_participant}_{matched_session}"
    patterns = [
        f"{bids_prefix}_task-{task}_{run_str}_feedbacklog.tsv",
        f"{bids_prefix}_task-{task}_{run_str}_acq-*_feedbacklog.tsv",
        f"{bids_prefix}_task-{task}_acq-*_{run_str}_feedbacklog.tsv",
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
        f"Session: {matched_session}, Task: {task}, Run: {run}"
    )
