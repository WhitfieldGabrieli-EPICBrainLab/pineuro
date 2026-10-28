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
- next_unassigned_slot() / assign_next_slot(): Fill a pre-allocated slot list
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

Slot lists
----------
A file with a ``slot`` column is a pre-allocated list whose participants are
assigned as they are scanned. A row with a blank subject_id is an unused slot
(a new list has no subject_id column at all); assign_next_slot() writes a
participant into the first one (in file order) and records ``assigned_at``. For replay, ``matched_slot`` names the slot whose
participant is replayed and is resolved to that participant's subject_id.

Example:
    slot    subject_id       condition    matched_slot    assigned_at
    001     sub-charms255    real                         2026-10-30T09:12:04
    002     sub-charms270    sham         001             2026-11-02T14:40:51
    003                      real
"""

import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
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

        missing_masks = [name for name in self.mask_names if name not in df.columns]
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
        return {name: float(row[name]) for name in self.mask_names}


def _nan_to_str(val) -> str | None:
    """Convert a pandas value to str, treating NaN and blank cells as None."""
    return None if pd.isna(val) or not str(val).strip() else str(val)


def _randomization_sep(file_path: Path) -> str:
    """Field separator for a randomization file, from its extension."""
    return "\t" if file_path.suffix.lower() in (".tsv", ".txt") else ","


def _read_randomization_verbatim(file_path: Path) -> pd.DataFrame:
    """Read a randomization file with every cell as a string and blanks as ``""``."""
    if not file_path.exists():
        raise FileNotFoundError(f"Randomization file not found: {file_path}")
    df = pd.read_csv(file_path, sep=_randomization_sep(file_path), dtype=str, keep_default_na=False)
    if "subject_id" not in df.columns:
        if "slot" not in df.columns:
            raise ValueError("Randomization file must have 'subject_id' column")
        # A new slot list has no subject_id column until its first participant is assigned
        df.insert(df.columns.get_loc("slot") + 1, "subject_id", "")
    return df


def next_unassigned_slot(file_path: Path) -> str | None:
    """Return the slot the next new participant would be assigned to.

    Slots fill in file order, so this is the first row whose subject_id is blank.

    Parameters
    ----------
    file_path : Path
        Path to randomization file (TSV or CSV)

    Returns
    -------
    str or None
        The slot, or None if every slot is assigned.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist
    ValueError
        If the file has no ``slot`` column (participants are listed directly)
    """
    file_path = Path(file_path)
    df = _read_randomization_verbatim(file_path)
    if "slot" not in df.columns:
        raise ValueError(f"Randomization file {file_path.name} has no 'slot' column")
    free = df.index[df["subject_id"].str.strip() == ""]
    return str(df.at[free[0], "slot"]) if len(free) else None


def assign_next_slot(file_path: Path, participant_id: str) -> str:
    """Assign a participant to the next unused slot of a pre-allocated randomization list.

    Writes the participant's ID and an ``assigned_at`` timestamp into the first
    row (in file order) whose subject_id is blank, so the file records the slot ->
    participant mapping. Every other cell is kept verbatim, and the file is
    replaced atomically. The condition is neither logged nor returned, so the
    assignment does not unblind the operator.

    Parameters
    ----------
    file_path : Path
        Path to randomization file (TSV or CSV) with a ``slot`` column
    participant_id : str
        Participant identifier to assign (e.g., "sub-charms270")

    Returns
    -------
    str
        The assigned slot

    Raises
    ------
    FileNotFoundError
        If file_path does not exist
    ValueError
        If the file has no ``slot`` column, the participant is already listed,
        or every slot is assigned
    """
    file_path = Path(file_path)
    df = _read_randomization_verbatim(file_path)
    if "slot" not in df.columns:
        raise ValueError(f"Randomization file {file_path.name} has no 'slot' column")

    subject_ids = df["subject_id"].str.strip()
    if (subject_ids == participant_id).any():
        raise ValueError(f"Participant {participant_id} is already listed in {file_path.name}")
    free = df.index[subject_ids == ""]
    if not len(free):
        raise ValueError(f"All {len(df)} slots in {file_path.name} are assigned")

    row = free[0]
    slot = str(df.at[row, "slot"])
    df.at[row, "subject_id"] = participant_id
    if "assigned_at" not in df.columns:
        df["assigned_at"] = ""
    df.at[row, "assigned_at"] = datetime.now().isoformat(timespec="seconds")

    partial = file_path.with_name(file_path.name + ".partial")
    try:
        df.to_csv(partial, sep=_randomization_sep(file_path), index=False)
        os.replace(partial, file_path)
    finally:
        partial.unlink(missing_ok=True)

    logger.info(f"Assigned {participant_id} to randomization slot {slot} in {file_path}")
    return slot


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

        In a slot list (a ``slot`` column), unused slots (blank subject_id) are
        left out, and a ``matched_slot`` is resolved to the subject_id assigned to
        that slot (None while the slot is unused). An explicit
        matched_participant takes precedence over matched_slot.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist
    ValueError
        If required columns are missing, condition values are invalid, or a
        slot is blank or duplicated
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Randomization file not found: {file_path}")

    # Read every cell as a string so IDs stay verbatim ("001" would otherwise load as 1)
    df = pd.read_csv(file_path, sep=_randomization_sep(file_path), dtype=str)

    # Validate required columns
    if "subject_id" not in df.columns:
        if "slot" not in df.columns:
            raise ValueError("Randomization file must have 'subject_id' column")
        df["subject_id"] = None  # a new slot list: no participant assigned yet
    if "condition" not in df.columns:
        raise ValueError("Randomization file must have 'condition' column")

    slot_subjects: dict[str, str | None] = {}
    if "slot" in df.columns:
        slots = [_nan_to_str(s) or "" for s in df["slot"]]
        if "" in slots:
            raise ValueError("Randomization file has a row with a blank 'slot'")
        duplicates = sorted({s for s in slots if slots.count(s) > 1})
        if duplicates:
            raise ValueError(f"Randomization file has duplicate slots: {duplicates}")
        slot_subjects = {s: _nan_to_str(sid) for s, sid in zip(slots, df["subject_id"], strict=True)}

    result = {}
    for _, row in df.iterrows():
        subject_id = _nan_to_str(row["subject_id"])
        condition = str(row["condition"]).lower()

        if condition not in ("real", "sham"):
            label = subject_id or f"slot {row.get('slot')}"
            raise ValueError(f"Invalid condition '{condition}' for {label}. Must be 'real' or 'sham'")

        if subject_id is None:
            continue  # unused slot

        matched_participant = _nan_to_str(row.get("matched_participant"))
        matched_slot = _nan_to_str(row.get("matched_slot"))
        if matched_participant is None and matched_slot is not None:
            matched_participant = slot_subjects.get(matched_slot)
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
    slots: bool = False,
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
        Must be > 0 and < len(participant_ids), and at most
        len(participant_ids) // 2 for method="replay".
    method : str
        Assignment method:
        - "simple": Random real/sham assignment with no replay matching.
        - "replay": Random assignment plus one-to-one sham→real participant
          matching for feedbacklog replay. Participants are assumed to be
          scanned in list order, so the assignment is drawn uniformly from the
          orders in which every sham participant has a real participant
          earlier in the list, and each sham participant is matched to the
          earliest earlier real participant not yet matched. The first
          participant (participant_ids[0]) is therefore always "real".
    matched_session : str, optional
        Session ID to populate the matched_session column for replay participants.
        Ignored when method="simple".
    output_path : Path, optional
        If provided, write the randomization table to this path as TSV.
    seed : int, optional
        Random seed for reproducible assignments.
    slots : bool
        If True, ``participant_ids`` are the slots of a pre-allocated list whose
        participants are assigned as they are scanned (``assign_next_slot()``).
        The table then has a ``slot`` column and no subject_id column, and
        replay matches go in ``matched_slot`` instead of ``matched_participant``.

    Returns
    -------
    pd.DataFrame
        Randomization table with columns: subject_id, condition,
        matched_participant, matched_session, notes. With ``slots``: slot,
        condition, plus matched_slot for replay and matched_session when it
        is given (subject_id is added at the first assignment).

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
    if method == "replay" and n_sham > len(participant_ids) // 2:
        raise ValueError(
            f"method='replay' pairs each sham participant with a different real participant, "
            f"so n_sham can be at most {len(participant_ids) // 2} (got {n_sham})"
        )

    rng = np.random.default_rng(seed)

    # --- Shuffle and assign conditions ---
    # For replay, participants are scanned in list order and each sham participant
    # replays a different, earlier real participant, so every prefix of the list needs
    # at least as many real as sham participants. Redrawing until that holds samples
    # uniformly from the valid orders (and makes participant_ids[0] real).
    while True:
        indices = np.arange(len(participant_ids))
        rng.shuffle(indices)

        conditions = ["real"] * len(participant_ids)
        for i in range(n_sham):
            conditions[indices[i]] = "sham"

        if method != "replay" or np.cumsum([1 if c == "real" else -1 for c in conditions]).min() >= 0:
            break

    # --- Build matched_participant column for replay method ---
    matched_participants: list[str | None] = [None] * len(participant_ids)
    matched_sessions: list[str | None] = [None] * len(participant_ids)

    if method == "replay":
        # Each sham participant replays the earliest earlier real participant not yet paired.
        unpaired: deque[int] = deque()
        for i, condition in enumerate(conditions):
            if condition == "real":
                unpaired.append(i)
                continue
            matched_participants[i] = participant_ids[unpaired.popleft()]
            if matched_session:
                matched_sessions[i] = matched_session

    # --- Build DataFrame ---
    if slots:
        # Only columns that carry values: assign_next_slot() adds subject_id at the first assignment.
        columns: dict[str, list] = {"slot": participant_ids, "condition": conditions}
        if method == "replay":
            columns["matched_slot"] = matched_participants
            if matched_session:
                columns["matched_session"] = matched_sessions
        df = pd.DataFrame(columns)
    else:
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
        f"{bids_prefix}_task-{task}_{run_str}_attempt-*_feedbacklog.tsv",
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
