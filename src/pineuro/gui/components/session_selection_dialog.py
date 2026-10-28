"""Session selection dialog for multi-session studies.

This dialog allows users to:
- View existing sessions for a participant and continue one
- Create the next session declared by the study protocol

Each session's protocol (its localizers, feedback tasks, and which masks are
reused from earlier sessions) is defined in the study config's ``sessions`` list,
so the dialog no longer asks the operator to choose a mask source — it validates
that any session a reused localizer points at already exists with approved masks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft

from pineuro.gui import styles as st

if TYPE_CHECKING:
    from pineuro.gui.models import StudyConfig
    from pineuro.session import SessionMetadata

logger = logging.getLogger(__name__)


def _session_number_from_id(session_id: str) -> int:
    """Parse a BIDS session id ``'ses-NN'`` into its 1-based session number."""
    prefix = "ses-"
    suffix = session_id[len(prefix) :]
    if session_id.startswith(prefix) and suffix.isdigit():
        return int(suffix)
    return 1


def _unlisted_participant_errors(rand_path: Path, participant_id: str, session_index: int) -> list[str]:
    """Errors for a sham-study participant who is not in the randomization file.

    A slot list (a ``slot`` column) assigns a new participant to its next unused
    slot when their first session starts, so that case is not an error.
    """
    from pineuro.sham import next_unassigned_slot

    try:
        free_slot = next_unassigned_slot(rand_path)
    except ValueError:  # no slot column: participants are listed directly
        return [
            f"Participant '{participant_id}' is not listed in the randomization file "
            f"({rand_path.name}). Sham is enabled study-wide, so an unlisted participant "
            f"cannot be assigned a condition — check for a typo or add them to the file."
        ]
    if free_slot is None:
        return [
            f"Participant '{participant_id}' is not listed in the randomization file "
            f"({rand_path.name}), and every slot in it is assigned. Check for a typo in the "
            f"participant ID or add slots to the file."
        ]
    if session_index > 0:
        return [
            f"Participant '{participant_id}' is not listed in the randomization file "
            f"({rand_path.name}). A slot is assigned only at a participant's first session, "
            f"so check for a typo in the participant ID."
        ]
    return []


def pending_slot_assignment(study_config: StudyConfig | None, participant_id: str) -> tuple[Path, str] | None:
    """The randomization slot a participant's first session would assign them to.

    Returns ``(randomization_file, slot)`` when sham is enabled, the randomization
    file is a slot list with an unused slot, and the participant is not listed in
    it yet; otherwise ``None``.
    """
    if study_config is None:
        return None
    sham_cfg = study_config.session_config.get("sham", {}) or {}
    rand_file = sham_cfg.get("randomization_file")
    if not sham_cfg.get("enabled", False) or not rand_file:
        return None

    from pineuro.sham import load_randomization, next_unassigned_slot

    rand_path = Path(rand_file)
    try:
        if participant_id in load_randomization(rand_path):
            return None
        slot = next_unassigned_slot(rand_path)
    except (OSError, ValueError):  # unreadable, or not a slot list (validate_session_start reports it)
        return None
    return None if slot is None else (rand_path, slot)


def validate_session_start(
    study_config: StudyConfig | None,
    participant_id: str,
    session_index: int,
    base_dir: Path,
) -> list[str]:
    """Validate that session ``session_index`` (0-based) can be started.

    Checks, for the chosen session's protocol:
    - the session index is within the study's declared ``sessions`` list, and
    - every localizer that reuses masks (``reuse_from_session``) points at an
      *earlier* session that exists on disk with that localizer's thresholded
      masks present.

    Returns a list of human-readable error strings; empty means the session may
    start. Returns ``[]`` when ``study_config`` is ``None`` (nothing to validate).
    """
    if study_config is None:
        return []

    from ..utils import localizer_mask_names

    errors: list[str] = []

    # Sham gate: when sham is enabled study-wide, the participant must appear in the
    # randomization file. Otherwise no condition can be assigned and the run would
    # silently default to "real" (the original bug). This mirrors the runtime guard in
    # pipeline_runner so the operator gets the error before the session starts. A slot
    # list instead assigns an unlisted participant a slot as their first session starts.
    sham_cfg = study_config.session_config.get("sham", {}) or {}
    if sham_cfg.get("enabled", False):
        rand_file = sham_cfg.get("randomization_file")
        if not rand_file:
            errors.append("Sham is enabled for this study but no randomization file is configured.")
        else:
            rand_path = Path(rand_file)
            if not rand_path.exists():
                errors.append(f"Sham is enabled but the randomization file was not found: {rand_path}.")
            else:
                from pineuro.sham import load_randomization

                try:
                    randomization = load_randomization(rand_path)
                except Exception as exc:
                    errors.append(f"Sham is enabled but the randomization file could not be read: {exc}")
                else:
                    if participant_id not in randomization:
                        errors.extend(_unlisted_participant_errors(rand_path, participant_id, session_index))

    sessions = study_config.session_config.get("sessions") or []
    if not 0 <= session_index < len(sessions):
        errors.append(
            f"Session {session_index + 1} is not defined in this study (it declares {len(sessions)} session(s))."
        )
        return errors

    proto = sessions[session_index]
    for loc in proto.get("localizers", []):
        src = loc.get("reuse_from_session")
        if not src:
            continue
        loc_label = loc.get("id") or loc.get("type") or "localizer"
        src_num = _session_number_from_id(src) if isinstance(src, str) and src.startswith("ses-") else None
        if src_num is None:
            errors.append(f"Localizer '{loc_label}' reuses an invalid session id '{src}'.")
            continue
        if src_num >= session_index + 1:
            errors.append(f"Localizer '{loc_label}' reuses {src}, which is not an earlier session.")
            continue
        masks_dir = base_dir / participant_id / src / "mask_extraction" / "03_masks"
        names = localizer_mask_names(loc)
        missing = [n for n in names if not list(masks_dir.glob(f"*_{n}_thr.nii.gz"))]
        if not masks_dir.exists() or missing:
            errors.append(
                f"Localizer '{loc_label}' reuses {src}, but its masks were not found there "
                f"(missing: {', '.join(missing) or 'all'}). Run and approve {src} first."
            )
    return errors


class SessionSelectionDialog:
    """Dialog for selecting or creating sessions for a participant."""

    def __init__(
        self,
        page: ft.Page,
        participant_id: str,
        base_dir: Path,
        on_select: Callable[[str, SessionMetadata | None], None],
        study_config: StudyConfig | None = None,
    ):
        """Initialize the session selection dialog.

        Args:
            page: Flet Page object
            participant_id: Participant ID
            base_dir: Study base directory
            on_select: Callback when session is selected/created. Called with
                ``(session_id, session_metadata)`` where ``session_metadata`` is
                ``None`` when continuing an existing session.
            study_config: The study being run. Provides the declared session count
                and per-session reuse requirements (used to bound new-session
                creation and to validate reused-mask sources).
        """
        self.page = page
        self.participant_id = participant_id
        self.base_dir = Path(base_dir)
        self.on_select = on_select
        self.study_config = study_config
        from pineuro.session import detect_existing_sessions

        self.existing_sessions = detect_existing_sessions(self.base_dir, participant_id)
        self.dialog: ft.AlertDialog | None = None

    @property
    def session_count(self) -> int | None:
        """Number of sessions the study declares, or ``None`` if unknown."""
        if self.study_config is None:
            return None
        return self.study_config.session_count

    def show(self):
        """Show the session selection dialog."""
        if not self.existing_sessions:
            self._show_new_participant_dialog()
        else:
            self._show_multi_session_dialog()

    def _make_session_metadata(self, session_id: str, session_number: int, notes: str) -> SessionMetadata:
        from pineuro.session import SessionMetadata

        return SessionMetadata(session_id=session_id, session_number=session_number, notes=notes)

    def _show_new_participant_dialog(self):
        """Show dialog for a new participant without existing data (starts ses-01)."""
        default_session_id = "ses-01"
        errors = validate_session_start(self.study_config, self.participant_id, 0, self.base_dir)
        # Never shows the slot's condition, so the assignment keeps the operator blind.
        pending_slot = None if errors else pending_slot_assignment(self.study_config, self.participant_id)

        body: list[ft.Control] = [
            ft.Text(f"Participant: {self.participant_id}", size=st.SUBSECTION, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("No existing data found for this participant.", size=st.CAPTION, color=st.MUTED),
            ft.Text(f"A new session ({default_session_id}) will be created.", size=st.CAPTION),
        ]
        if pending_slot is not None:
            rand_path, slot = pending_slot
            body.append(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.INFO_OUTLINE, color=st.INFO, size=st.ICON_SM),
                        ft.Text(
                            f"{self.participant_id} is not yet in the randomization list. Starting assigns "
                            f"them to the next unused slot ({slot}), recorded in {rand_path.name}.",
                            size=st.CAPTION,
                            expand=True,
                        ),
                    ],
                    spacing=st.GAP_SM,
                )
            )
        body.extend(self._error_controls(errors))
        assign_error = ft.Text("", size=st.CAPTION, color=st.ERROR, visible=False)
        body.append(assign_error)

        def start_new(e):
            if pending_slot is not None:
                from pineuro.sham import assign_next_slot

                try:
                    assign_next_slot(pending_slot[0], self.participant_id)
                except Exception as exc:  # e.g. the file is open in another program
                    logger.error(f"Randomization slot assignment failed: {exc}")
                    assign_error.value = f"Could not assign a randomization slot: {exc}"
                    assign_error.visible = True
                    assign_error.update()
                    return
            self.page.pop_dialog()
            self.on_select(default_session_id, self._make_session_metadata(default_session_id, 1, ""))

        def cancel(e):
            self.page.pop_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("Start New Session"),
            content=ft.Container(content=ft.Column(body, tight=True, spacing=st.GAP_SM), width=st.DIALOG_SM),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.FilledButton(
                    "Start" if pending_slot is None else "Assign Slot and Start",
                    icon=ft.Icons.PLAY_ARROW,
                    on_click=start_new,
                    disabled=bool(errors),
                ),
            ],
        )
        self.page.show_dialog(self.dialog)

    def _show_multi_session_dialog(self):
        """Show dialog when one or more sessions already exist for this participant."""
        session_tiles: list[ft.Control] = []

        for session in self.existing_sessions:
            session_id = session["session_id"]
            status_text = session["status"]

            session_tiles.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.FOLDER_OPEN),
                    title=ft.Text(session_id),
                    subtitle=ft.Text(status_text, size=st.CAPTION),
                    on_click=lambda e, vid=session_id: self._select_existing_session(vid),
                )
            )

        from pineuro.session import next_session_id

        new_session_id = next_session_id(self.base_dir, self.participant_id)
        new_session_number = _session_number_from_id(new_session_id)

        # Bound new-session creation by the study's declared session count.
        count = self.session_count
        can_create_new = count is None or new_session_number <= count

        body: list[ft.Control] = [
            ft.Text(f"Participant: {self.participant_id}", size=st.SUBSECTION, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("Existing sessions:", size=st.CAPTION, weight=ft.FontWeight.W_500),
            ft.Container(
                content=ft.Column(session_tiles, spacing=0),
                border=ft.Border.all(1, st.PANEL_BORDER),
                border_radius=8,
                padding=0,
            ),
            ft.Container(height=st.GAP_MD),
        ]
        if can_create_new:
            body.append(ft.Text("Or create the next session:", size=st.CAPTION, weight=ft.FontWeight.W_500))
        else:
            body.append(
                ft.Text(
                    f"All {count} session(s) declared by this study have been created.",
                    size=st.CAPTION,
                    color=st.MUTED,
                )
            )

        def create_new_session(e):
            self.page.pop_dialog()
            self._show_new_session_dialog(new_session_id, new_session_number)

        def cancel(e):
            self.page.pop_dialog()

        actions: list[ft.Control] = [ft.TextButton("Cancel", on_click=cancel)]
        if can_create_new:
            actions.append(
                ft.FilledButton(
                    f"New Session ({new_session_id})",
                    icon=ft.Icons.ADD,
                    on_click=create_new_session,
                )
            )

        self.dialog = ft.AlertDialog(
            title=ft.Text("Select Session"),
            content=ft.Container(
                content=ft.Column(body, tight=True, spacing=st.GAP_SM), width=st.DIALOG_MD, height=350
            ),
            actions=actions,
        )
        self.page.show_dialog(self.dialog)

    def _select_existing_session(self, session_id: str):
        """Handle selection of an existing session (continue — no new metadata)."""
        self.page.pop_dialog()
        self.on_select(session_id, None)

    def _show_new_session_dialog(self, new_session_id: str, session_number: int):
        """Confirm creation of the next session (protocol-driven; validates reuse)."""
        errors = validate_session_start(self.study_config, self.participant_id, session_number - 1, self.base_dir)

        notes_field = ft.TextField(label="Notes (optional)", multiline=True, max_lines=3, width=350)

        body: list[ft.Control] = [
            ft.Text(f"Creating session: {new_session_id}", size=st.SUBSECTION, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text(
                "This session's tasks and mask sources are defined by the study protocol.",
                size=st.CAPTION,
                color=st.MUTED,
            ),
        ]
        body.extend(self._error_controls(errors))
        body.extend([ft.Container(height=st.GAP_MD), notes_field])

        def create_session(e):
            self.page.pop_dialog()
            session_metadata = self._make_session_metadata(
                new_session_id, session_number, (notes_field.value or "").strip()
            )
            logger.info(f"Creating new session: {new_session_id} (session_number={session_number})")
            self.on_select(new_session_id, session_metadata)

        def back(e):
            self.page.pop_dialog()
            self._show_multi_session_dialog()

        self.dialog = ft.AlertDialog(
            title=ft.Text("New Session Options"),
            content=ft.Container(content=ft.Column(body, tight=True, spacing=st.GAP_SM), width=st.DIALOG_SM),
            actions=[
                ft.TextButton("Back", on_click=back),
                ft.FilledButton("Create", icon=ft.Icons.CHECK, on_click=create_session, disabled=bool(errors)),
            ],
        )
        self.page.show_dialog(self.dialog)

    @staticmethod
    def _error_controls(errors: list[str]) -> list[ft.Control]:
        """Render validation errors as red warning rows (empty list when valid)."""
        controls: list[Any] = []
        for msg in errors:
            controls.append(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.WARNING_AMBER, color=st.ERROR, size=st.ICON_SM),
                        ft.Text(msg, size=st.CAPTION, color=st.ERROR, expand=True),
                    ],
                    spacing=st.GAP_SM,
                )
            )
        return controls
