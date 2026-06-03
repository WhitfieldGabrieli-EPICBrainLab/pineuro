"""Study Editor screen for creating and editing study configurations.

This module provides a tabbed interface for configuring all aspects of a pineuro study.
Localizer and feedback tabs are created dynamically based on user-specified counts.
"""

import contextlib
import copy
import logging
from collections.abc import Callable

import flet as ft

from pineuro.tasks.task_registry import get_task_metadata, list_available_tasks

from .. import styles as st
from ..config_manager import ConfigManager
from ..models import StudyConfig, StudyMetadata
from ..utils import (
    frc_referenced_masks,
    localizer_is_reused,
    localizer_mask_names,
    resolve_use_transfer_runs,
    reuse_realtime_error,
)
from ._feedback_tab import FeedbackTabBuilder
from ._localizer_tab import LocalizerTabBuilder

logger = logging.getLogger(__name__)


def _ses_num(session_id: str) -> int | None:
    """Parse a BIDS session id ``'ses-NN'`` into its 1-based number, or None."""
    prefix = "ses-"
    suffix = session_id[len(prefix) :] if isinstance(session_id, str) else ""
    if isinstance(session_id, str) and session_id.startswith(prefix) and suffix.isdigit():
        return int(suffix)
    return None


class StudyEditorScreen:
    """Study editor screen with dynamic tabbed parameter sections."""

    def __init__(
        self,
        page: ft.Page,
        config_manager: ConfigManager,
        study_config: StudyConfig | None = None,
        on_cancel=None,
        on_save=None,
        on_start_session=None,
    ):
        self.page = page
        self.config_manager = config_manager
        self.is_new = study_config is None
        self.on_cancel_callback = on_cancel
        self.on_save_callback = on_save
        self.on_start_session_callback = on_start_session

        # Work on a deep copy so in-progress edits do not mutate the caller's
        # StudyConfig; changes are persisted only via the save callback.
        self.study_config: StudyConfig = (
            copy.deepcopy(study_config) if study_config is not None else self._create_default_config()
        )
        self.validation_errors: dict[str, str] = {}

        # Per-session protocol: the study config carries an ordered ``sessions``
        # list and the editor edits one session at a time. Normalize legacy/flat
        # shapes into ``sessions`` and start on the first session.
        self._active_session_index = 0
        self._ensure_sessions()

        # Master-detail navigation state. The left sidebar selects a *scope*:
        # "study" (study-wide config) or "session" (the active session's protocol).
        # Every study — new or existing — opens on study-wide so the user reviews
        # name/paths/scanner before drilling into a session. ``_last_inner_index``
        # remembers which session sub-tab (Protocol / Localizer N / Feedback N) was
        # selected so switching sessions lands on the same sub-tab (compare-across-sessions).
        self._active_scope = "study"
        self._last_inner_index = 0

        # Initialize localizer and feedback configs from the active session
        self._init_localizer_configs()
        self._init_feedback_configs()

        # Create Study tab and Paths tab fields (non-dynamic)
        self._create_study_fields()
        self._create_paths_fields()

        # Baseline snapshot for unsaved-change detection on Start Session.
        self._saved_snapshot: dict = self._capture_snapshot()

    # ------------------------------------------------------------------
    # Unsaved-change detection
    # ------------------------------------------------------------------

    def _capture_snapshot(self) -> dict:
        """Deep-copy of study config for dirty-check; excludes metadata.modified."""
        snap = copy.deepcopy(self.study_config.to_dict())
        meta = snap.get("_metadata")
        if isinstance(meta, dict):
            meta.pop("modified", None)
        return snap

    def _has_unsaved_changes(self) -> bool:
        return self._capture_snapshot() != self._saved_snapshot

    # ------------------------------------------------------------------
    # Config initialization
    # ------------------------------------------------------------------

    def _ensure_sessions(self):
        """Normalize ``session_config`` to carry a non-empty ``sessions`` list.

        A flat config (top-level ``localizers``/``feedback_runs``) is wrapped into
        a single session; those top-level keys are then removed so the study config
        has exactly one home for per-session protocol.
        """
        sc = self.study_config.session_config
        sessions = sc.get("sessions")
        if not sessions:
            sc["sessions"] = [
                {
                    "notes": "",
                    "localizers": sc.pop("localizers", []),
                    "feedback_runs": sc.pop("feedback_runs", []),
                }
            ]
        else:
            sc.pop("localizers", None)
            sc.pop("feedback_runs", None)
        self._active_session_index = max(0, min(self._active_session_index, len(sc["sessions"]) - 1))

    def _active_session(self) -> dict:
        """Return the session dict currently being edited (clamps the index)."""
        sessions = self.study_config.session_config["sessions"]
        self._active_session_index = max(0, min(self._active_session_index, len(sessions) - 1))
        session: dict = sessions[self._active_session_index]
        return session

    def _init_localizer_configs(self):
        """Initialize localizer config list from the active session."""
        self._localizer_configs = self._active_session().setdefault("localizers", [])

    def _init_feedback_configs(self):
        """Initialize feedback run config list from the active session."""
        self._feedback_configs = self._active_session().setdefault("feedback_runs", [])

    # ------------------------------------------------------------------
    # Default config
    # ------------------------------------------------------------------

    def _create_default_config(self) -> StudyConfig:
        metadata = StudyMetadata(name="", description="")
        session_config = {
            "paths": {
                "participant_id": "",
                "base_dir": "",
                "example_participant_id": "",
            },
            "simulator": {
                "enabled": True,
                "collection_speed": 1.0,
                "feedback_speed": 1.0,
                "trigger_mode": "manual",
                "source_mapping": "",
            },
            "scanner": {"dicom_dir": "/BOLD_DICOM", "ip": ""},
            "reference": {
                "initial_enabled": True,
                "initial_n_volumes": 2,
                "realtime_enabled": True,
                "realtime_n_volumes": 2,
            },
            "sham": {
                "enabled": False,
                "blind_mode": False,
                "randomization_file": "",
            },
            "display": {
                "control_monitor": "left",
                "presenter_monitor": "right",
            },
            "localizers": [],
            "feedback_runs": [],
        }
        return StudyConfig(metadata=metadata, session_config=session_config)

    # ------------------------------------------------------------------
    # Study tab fields
    # ------------------------------------------------------------------

    def _create_study_fields(self):
        """Create form fields for the Study tab."""
        config = self.study_config.session_config

        self.name_field = ft.TextField(
            label="Study Name",
            value=self.study_config.metadata.name,
            hint_text="Unique name shown in the study manager",
            on_change=self._on_name_change,
        )
        self.description_field = ft.TextField(
            label="Description",
            value=self.study_config.metadata.description,
            hint_text="Brief summary shown on the study card",
            multiline=True,
            min_lines=2,
            max_lines=4,
        )
        self.example_participant_id_field = ft.TextField(
            label="Example Participant ID",
            value=config["paths"].get("example_participant_id", "sub-001"),
            hint_text="Shown as hint when starting a session (e.g., sub-001)",
        )

        # Simulator mode
        simulator_cfg = config.get("simulator", {})
        self.simulator_enabled_checkbox = ft.Checkbox(
            label="Simulator Mode",
            value=simulator_cfg.get("enabled", True),
            on_change=self._on_simulator_mode_changed,
        )
        self.collection_speed_field = ft.TextField(
            label="Localizer speed (× realtime)",
            value=str(simulator_cfg.get("collection_speed", 1.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Speed for localizer/setup runs",
        )
        self.feedback_simulator_speed_field = ft.TextField(
            label="Feedback speed (× realtime)",
            value=str(simulator_cfg.get("feedback_speed", 1.0)),
            keyboard_type=ft.KeyboardType.NUMBER,
            hint_text="Speed for feedback runs",
        )
        self.simulator_trigger_mode_dropdown = ft.Dropdown(
            label="Simulator trigger mode",
            value=simulator_cfg.get("trigger_mode", "manual"),
            options=[
                ft.dropdown.Option("manual", "Manual (press key)"),
                ft.dropdown.Option("os_key", "Synthetic OS keypress"),
            ],
            hint_text="Manual mimics a real scan; OS keypress is hands-off",
        )

        # Scanner settings
        scanner_cfg = config.get("scanner", {})
        self.scanner_dicom_dir_field = ft.TextField(
            label="Scanner DICOM Directory",
            value=scanner_cfg.get("dicom_dir", "/BOLD_DICOM"),
            hint_text="Directory where scanner exports DICOMs",
        )
        self.scanner_ip_field = ft.TextField(
            label="Scanner IP Address",
            value=scanner_cfg.get("ip", ""),
            hint_text="e.g., 192.168.2.1",
        )
        self.scanner_setup_guide_btn = ft.OutlinedButton(
            "Setup Guide",
            icon=ft.Icons.HELP_OUTLINE,
            on_click=lambda e: self._show_scanner_setup_guide(),
        )
        self.scanner_trigger_keys_field = ft.TextField(
            label="Scanner Trigger Keys",
            value=", ".join(scanner_cfg.get("trigger_keys") or ["t", "+", "5", "="]),
            hint_text="Comma-separated keys delivered by the scanner per volume (e.g. t, +, 5, =)",
        )

        # Reference scan settings (session-level)
        reference_cfg = config.get("reference", {})
        self.reference_initial_enabled_checkbox = ft.Checkbox(
            label="Collect initial reference scan (before localizers)",
            value=reference_cfg.get("initial_enabled", True),
            on_change=self._on_reference_scan_changed,
        )
        self.reference_initial_n_volumes_field = ft.TextField(
            width=st.COMPACT_W,
            label="Volumes",
            value=str(reference_cfg.get("initial_n_volumes", 2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=reference_cfg.get("initial_enabled", True),
        )
        self.reference_realtime_enabled_checkbox = ft.Checkbox(
            label="Collect real-time reference scan (before feedback)",
            value=reference_cfg.get("realtime_enabled", True),
            on_change=self._on_reference_scan_changed,
        )
        self.reference_realtime_n_volumes_field = ft.TextField(
            width=st.COMPACT_W,
            label="Volumes",
            value=str(reference_cfg.get("realtime_n_volumes", 2)),
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=reference_cfg.get("realtime_enabled", True),
        )

        # Display settings
        display_config = config.get("display", {})
        self.control_monitor_dropdown = ft.Dropdown(
            label="Control Monitor (GUI)",
            options=[
                ft.dropdown.Option(key="left", text="Left"),
                ft.dropdown.Option(key="right", text="Right"),
            ],
            value=display_config.get("control_monitor", "left"),
        )
        self.presenter_monitor_dropdown = ft.Dropdown(
            label="Presenter Monitor (Task)",
            options=[
                ft.dropdown.Option(key="left", text="Left"),
                ft.dropdown.Option(key="right", text="Right"),
            ],
            value=display_config.get("presenter_monitor", "right"),
        )

        # Sham / control condition (study-wide: whether sham runs, the participant
        # randomization, and operator blinding — the per-task mask construction lives in
        # each feedback run's tab).
        sham_cfg = config.get("sham", {})
        sham_enabled = sham_cfg.get("enabled", False)
        self.sham_enabled_checkbox = ft.Checkbox(
            label="Enable Sham/Control Conditions",
            value=sham_enabled,
            on_change=self._on_study_sham_enabled_changed,
        )
        # Blind Mode and the randomization file only apply when sham is enabled; their
        # visibility is gated on the checkbox above (initialized here, toggled live).
        self.sham_blind_mode_checkbox = ft.Checkbox(
            label="Blind Mode (hide condition from operator)",
            value=sham_cfg.get("blind_mode", False),
            visible=sham_enabled,
        )
        self.sham_randomization_file_field = ft.TextField(
            label="Randomization File",
            value=sham_cfg.get("randomization_file", ""),
            hint_text="TSV: subject_id, condition (real/sham), matched_participant, matched_session",
        )
        self.sham_randomization_row = st.browse_row(
            self.sham_randomization_file_field,
            self._create_file_browse_button(
                self.sham_randomization_file_field,
                filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv"), ("All files", "*.*")],
            ),
            ft.IconButton(
                icon=ft.Icons.SHUFFLE,
                tooltip="Generate randomization file",
                on_click=self._on_generate_randomization,
            ),
        )
        self.sham_randomization_row.visible = sham_enabled

        # Type dropdown options (lazily computed — avoid triggering heavy task module imports)
        self._localizer_type_options_cache: list[ft.dropdown.Option] | None = None
        self._feedback_type_options_cache: list[ft.dropdown.Option] | None = None

    @property
    def _localizer_type_options(self) -> list[ft.dropdown.Option]:
        if self._localizer_type_options_cache is None:
            opts = [
                ft.dropdown.Option(key="rest", text="Rest (ICA Network Extraction)"),
                ft.dropdown.Option(key="manual", text="Manual Mask Import"),
            ]
            available_loc_tasks = list_available_tasks(role="localizer")
            opts.extend(
                ft.dropdown.Option(key=task_id, text=f"{meta.display_name} (GLM Contrast)")
                for task_id, meta in available_loc_tasks.items()
                if meta.default_contrasts
            )
            self._localizer_type_options_cache = opts
        return self._localizer_type_options_cache

    @property
    def _feedback_type_options(self) -> list[ft.dropdown.Option]:
        if self._feedback_type_options_cache is None:
            feedback_tasks = list_available_tasks(role="feedback")
            self._feedback_type_options_cache = [
                ft.dropdown.Option(key=task_id, text=meta.display_name) for task_id, meta in feedback_tasks.items()
            ]
        return self._feedback_type_options_cache

    def _create_paths_fields(self):
        """Create form fields for the Paths tab."""
        config = self.study_config.session_config

        self.base_dir_field = ft.TextField(
            label="Base data directory",
            value=config["paths"]["base_dir"],
            hint_text="Root directory for participant data (e.g., /data/pineuro)",
        )

        # Simulator source mode toggle
        sim_cfg = config.get("simulator", {})
        has_mapping = bool(sim_cfg.get("source_mapping", "").strip())
        self.simulator_source_mode_radio = ft.RadioGroup(
            content=ft.Row(
                [
                    ft.Radio(value="individual", label="Same paths for all participants"),
                    ft.Radio(value="mapping", label="Per-subject mapping file"),
                ]
            ),
            value="mapping" if has_mapping else "individual",
            on_change=self._on_simulator_source_mode_changed,
        )

        self.simulator_source_mapping_field = ft.TextField(
            label="Mapping File",
            value=sim_cfg.get("source_mapping", ""),
            hint_text="CSV/TSV file mapping subjects to DICOM paths",
        )

    # ------------------------------------------------------------------
    # Tab builders management
    # ------------------------------------------------------------------

    def _create_tab_builders(self):
        """Create LocalizerTabBuilder and FeedbackTabBuilder instances."""
        # Create localizer builders
        self._localizer_builders: list[LocalizerTabBuilder] = []
        for i, cfg in enumerate(self._localizer_configs):
            builder = LocalizerTabBuilder(
                index=i,
                config=cfg,
                page=self.page,
                on_change=self._on_localizer_changed,
                create_browse_button=self._create_browse_button,
                create_file_browse_button=self._create_file_browse_button,
            )
            self._localizer_builders.append(builder)

        # Session-qualified masks available to this session's feedback runs
        all_masks = self._aggregate_mask_options()

        # Create feedback builders
        self._feedback_builders: list[FeedbackTabBuilder] = []
        for i, cfg in enumerate(self._feedback_configs):
            builder = FeedbackTabBuilder(
                index=i,
                config=cfg,
                page=self.page,
                on_change=self._on_feedback_changed,
                available_mask_options=all_masks,
                create_browse_button=self._create_browse_button,
                create_file_browse_button=self._create_file_browse_button,
                sham_enabled=lambda: bool(self.sham_enabled_checkbox.value),
            )
            self._feedback_builders.append(builder)

    def _aggregate_mask_options(self) -> list[tuple[str, str, str]]:
        """Session-qualified masks a feedback run in the active session can use.

        Current-session localizer masks (source ``""``) from the live builders, plus
        every earlier session's localizer masks (source ``"ses-NN"``). Returned as
        ``(source, name, label)`` triples, de-duplicated with current/fresh first so
        the same ``(source, name)`` appears once. Earlier reused localizers are
        skipped so a source always names the session that produces the mask fresh.
        """
        options: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _add(source: str, name: str) -> None:
            key = (source, name)
            if not name or key in seen:
                return
            seen.add(key)
            label = f"{name}  (this session)" if not source else f"{name}  ({source})"
            options.append((source, name, label))

        for builder in self._localizer_builders:
            for name in builder.get_mask_names():
                _add("", name)
        sessions = self.study_config.session_config.get("sessions", [])
        for idx in range(self._active_session_index):
            src = f"ses-{idx + 1:02d}"
            for loc in sessions[idx].get("localizers", []):
                if localizer_is_reused(loc):
                    continue
                for name in localizer_mask_names(loc):
                    _add(src, name)
        return options

    def _on_localizer_changed(self):
        """Called when any localizer tab changes networks/type."""
        all_masks = self._aggregate_mask_options()
        for fb in self._feedback_builders:
            fb.update_available_masks(all_masks)
        self._sync_paths_tab_visibility()

    def _on_feedback_changed(self):
        """Called when any feedback tab changes."""
        self._sync_paths_tab_visibility()

    def _sync_paths_tab_visibility(self):
        """Sync reference simulator-source field visibility with the reference checkboxes.

        The reference source fields live on the study-wide pane; updates are guarded so
        this is a no-op when that pane is not currently mounted (e.g. fired from a
        localizer/feedback edit while a session is shown).
        """
        # Initial reference visibility
        ref_row = getattr(self, "_sim_loc_ref_row", None)
        if ref_row is not None:
            enabled = self.reference_initial_enabled_checkbox.value and bool(self._localizer_configs)
            ref_row.visible = enabled
            ref_label = getattr(self, "_sim_loc_ref_label", None)
            if ref_label is not None:
                ref_label.visible = enabled
            with contextlib.suppress(Exception):
                ref_row.update()
                if ref_label is not None:
                    ref_label.update()

        # Realtime reference visibility
        rt_ref_row = getattr(self, "_sim_fb_ref_row", None)
        if rt_ref_row is not None:
            enabled = self.reference_realtime_enabled_checkbox.value and bool(self._feedback_configs)
            rt_ref_row.visible = enabled
            rt_ref_label = getattr(self, "_sim_fb_ref_label", None)
            if rt_ref_label is not None:
                rt_ref_label.visible = enabled
            with contextlib.suppress(Exception):
                rt_ref_row.update()
                if rt_ref_label is not None:
                    rt_ref_label.update()

    # ------------------------------------------------------------------
    # Count control handlers
    # ------------------------------------------------------------------

    def _on_add_localizer(self, e):
        """Add a new localizer tab."""
        self._collect_all_configs()
        self._localizer_configs.append(
            {
                "type": "rest",
                "tr": 1.2,
                "n_volumes": 250,
                "n_runs": 2,
                "motion_monitoring": "siemens_moco",
                "networks": ["DefaultA", "ContA"],
                "extraction": {
                    "network_atlas": "yeo_17",
                    "n_components": 30,
                    "smooth_fwhm": 5.0,
                    "highpass": 0.01,
                    "threshold": 2000,
                    "threshold_method": "top_n",
                },
                "source_dicoms": [],
            }
        )
        self._rebuild_tabs()

    def _on_remove_localizer_at(self, index: int):
        """Remove localizer at given index."""
        self._collect_all_configs()
        self._localizer_configs.pop(index)
        self._rebuild_tabs()

    def _on_localizer_type_changed(self, index: int, e):
        """Handle localizer type change from Study tab overview."""
        self._collect_all_configs()
        new_type = self._loc_type_dropdowns[index].value or "rest"
        old_type = self._localizer_configs[index].get("type", "rest")

        if new_type == old_type:
            return

        # Reset type-specific defaults when switching between rest and task
        cfg = self._localizer_configs[index]
        cfg["type"] = new_type
        if new_type == "rest" and old_type != "rest":
            # Switching to rest: set ICA defaults
            cfg.setdefault("networks", ["DefaultA", "ContA"])
            extraction = cfg.setdefault("extraction", {})
            extraction.setdefault("network_atlas", "yeo_17")
            extraction.setdefault("n_components", 30)
            extraction.setdefault("smooth_fwhm", 5.0)
            extraction.setdefault("highpass", 0.01)
        elif new_type == "manual":
            # Switching to manual import: seed the participant-agnostic fields.
            # The actual mask files are chosen per participant in the workflow.
            cfg.setdefault("manual_mask_space", "native")
            cfg.setdefault("manual_mask_names", [])
        elif new_type != "rest":
            # Switching to task: set contrast and run-count defaults from metadata
            try:
                meta = get_task_metadata(new_type)
                if meta.default_contrasts:
                    cfg["contrasts"] = dict(meta.default_contrasts)
                cfg["n_runs"] = meta.default_n_runs
            except (KeyError, ValueError):
                pass
            cfg["include_reverse_contrasts"] = cfg.get("include_reverse_contrasts", False)

        self._rebuild_tabs()

    def _on_add_feedback(self, e):
        """Add a new feedback task tab."""
        self._collect_all_configs()
        template = {"task": "ball", "tr": 1.2, "n_volumes": 150, "n_baseline": 25, "max_runs": 1}
        self._feedback_configs.append(template)
        self._rebuild_tabs()

    def _on_remove_feedback_at(self, index: int):
        """Remove feedback task at given index."""
        self._collect_all_configs()
        self._feedback_configs.pop(index)
        self._rebuild_tabs()

    def _on_feedback_type_changed(self, index: int, e):
        """Handle feedback task type change from Study tab overview."""
        self._collect_all_configs()
        new_type = self._fb_type_dropdowns[index].value
        old_type = self._feedback_configs[index].get("task", "ball")

        if new_type == old_type:
            return

        cfg = self._feedback_configs[index]
        cfg["task"] = new_type
        # Clear old task-specific params since they're for the old task type
        cfg.pop("task_params", None)
        self._rebuild_tabs()

    # ------------------------------------------------------------------
    # Session control handlers
    # ------------------------------------------------------------------

    def _default_session(self) -> dict:
        """A new session's default protocol (blank — no localizers or feedback runs)."""
        return {
            "notes": "",
            "localizers": [],
            "feedback_runs": [],
            "reference_dicoms": {"initial": "", "realtime": ""},
        }

    def _switch_to_session(self, index: int):
        """Re-point the editor at session *index* and show its protocol."""
        self._active_session_index = index
        self._active_scope = "session"
        self._init_localizer_configs()
        self._init_feedback_configs()
        self._refresh_layout()

    def _on_select_session(self, index: int):
        # Selecting the already-active session is a no-op only when already on it;
        # from study-wide scope it must still switch into the session view.
        if self._active_scope == "session" and index == self._active_session_index:
            return
        self._collect_all_configs()
        self._switch_to_session(index)

    def _on_add_session(self, e):
        self._collect_all_configs()
        sessions = self.study_config.session_config["sessions"]
        sessions.append(self._default_session())
        self._switch_to_session(len(sessions) - 1)

    def _on_duplicate_session(self, e):
        self._collect_all_configs()
        sessions = self.study_config.session_config["sessions"]
        sessions.insert(self._active_session_index + 1, copy.deepcopy(self._active_session()))
        self._switch_to_session(self._active_session_index + 1)

    def _on_remove_session(self, e):
        sessions = self.study_config.session_config["sessions"]
        if len(sessions) <= 1:
            return
        removed_num = self._active_session_index + 1
        sessions.pop(self._active_session_index)
        # Clear now-invalid cross-session mask sources (pointing at or after the
        # removed session, whose numbering has shifted). The user re-selects them.
        for sess in sessions:
            for frc in sess.get("feedback_runs", []):
                srcs = frc.get("mask_sources", {})
                for name in list(srcs.keys()):
                    src_num = _ses_num(srcs[name])
                    if src_num is None or src_num >= removed_num:
                        del srcs[name]
        self._switch_to_session(max(0, self._active_session_index - 1))

    def _collect_all_configs(self):
        """Collect current field values from the session tab builders and sim fields.

        The collected localizers/feedback_runs/notes are written back into the active
        session so switching sessions (or saving) preserves edits.
        """
        # Remember which session sub-tab is open so switching sessions lands on the
        # same sub-tab (compare-across-sessions).
        if getattr(self, "_inner_tabs", None) is not None:
            with contextlib.suppress(Exception):
                self._last_inner_index = self._inner_tabs.selected_index or 0
        if hasattr(self, "_localizer_builders"):
            self._localizer_configs = [b.collect_config() for b in self._localizer_builders]
        if hasattr(self, "_feedback_builders"):
            self._feedback_configs = [b.collect_config() for b in self._feedback_builders]
        # Collect per-session simulator source paths into configs
        self._collect_simulator_paths()
        # Persist the collected protocol into the active session.
        sess = self._active_session()
        sess["localizers"] = self._localizer_configs
        sess["feedback_runs"] = self._feedback_configs
        if getattr(self, "_notes_field", None) is not None:
            sess["notes"] = self._notes_field.value or ""

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ft.Control:
        """Build the study editor UI (master-detail: sidebar selects the scope)."""
        # Create tab builders for the active session
        self._create_tab_builders()

        # Study-wide pane is built once and reused; the detail pane swaps between it
        # and the active session's protocol as the sidebar scope selection changes.
        self._study_wide_pane = self._build_study_wide_pane()
        self._detail_pane = ft.Container(expand=True)
        self._sidebar = ft.Container(
            width=250,
            bgcolor=st.SIDEBAR_BG,
            padding=ft.Padding.symmetric(vertical=st.GAP_MD, horizontal=st.GAP_SM),
            content=self._build_sidebar_content(),
        )
        self._render_detail()

        # Action buttons — primary: Save & Start (Filled); secondary: Save As
        # (Outlined); tertiary: Cancel (Text).
        action_buttons = ft.Row(
            [
                ft.TextButton("Cancel", on_click=self._on_cancel),
                ft.Row(
                    [
                        ft.OutlinedButton("Save As", on_click=self._on_save_as),
                        ft.FilledButton("Save", icon=ft.Icons.SAVE, on_click=self._on_save),
                        ft.FilledButton(
                            "Start Session",
                            icon=ft.Icons.PLAY_ARROW,
                            on_click=self._on_start_session,
                            disabled=self.is_new,
                        ),
                    ],
                    spacing=st.GAP_SM,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        return ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row(
                        [
                            st.title(
                                f"Study Editor - {self.study_config.metadata.name}",
                                expand=True,
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    padding=st.PAD,
                    bgcolor=st.HEADER_BG,
                ),
                # Master-detail body: scope sidebar + detail pane
                ft.Container(
                    content=ft.Row(
                        [
                            self._sidebar,
                            ft.VerticalDivider(width=1),
                            ft.Container(content=self._detail_pane, padding=st.PAD, expand=True),
                        ],
                        spacing=0,
                        expand=True,
                    ),
                    expand=True,
                ),
                # Actions
                ft.Container(
                    content=action_buttons,
                    padding=st.PAD,
                    bgcolor=st.SUBTLE_BG,
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _get_localizer_tab_label(self, index: int) -> str:
        """Get tab label for a localizer, including short type name."""
        cfg = self._localizer_configs[index]
        loc_type = cfg.get("type", "rest")
        if loc_type == "rest":
            short = "Rest"
        elif loc_type == "manual":
            short = "Manual"
        else:
            try:
                meta = get_task_metadata(loc_type)
                short = meta.display_name
            except (KeyError, ValueError):
                short = loc_type
        n = len(self._localizer_configs)
        prefix = f"Localizer {index + 1}" if n > 1 else "Localizer"
        return f"{prefix} ({short})"

    def _get_feedback_tab_label(self, index: int) -> str:
        """Get tab label for a feedback task, including short type name."""
        cfg = self._feedback_configs[index]
        task_type = cfg.get("task", "ball")
        try:
            meta = get_task_metadata(task_type)
            short = meta.display_name
        except (KeyError, ValueError):
            short = task_type
        n = len(self._feedback_configs)
        prefix = f"Feedback {index + 1}" if n > 1 else "Feedback"
        return f"{prefix} ({short})"

    # ------------------------------------------------------------------
    # Master-detail navigation (sidebar scope + detail pane)
    # ------------------------------------------------------------------

    def _loc_type_short(self, loc_type: str) -> str:
        """Short display name for a localizer type (Rest / Manual / task name)."""
        if loc_type == "rest":
            return "Rest"
        if loc_type == "manual":
            return "Manual"
        try:
            return get_task_metadata(loc_type).display_name
        except (KeyError, ValueError):
            return loc_type

    def _session_summary(self, index: int) -> str:
        """One-line protocol fingerprint for session *index*.

        Lists localizer types and feedback tasks, with a trailing ``<- ses-NN`` marker
        when a feedback run pulls masks from an earlier session. This gives each
        session a distinct fingerprint even when two sessions collect identical scans.
        """
        sessions = self.study_config.session_config.get("sessions", [])
        if not 0 <= index < len(sessions):
            return ""
        sess = sessions[index]
        parts = [self._loc_type_short(loc.get("type", "rest")) for loc in sess.get("localizers", [])]
        reuse_sources: set[str] = set()
        for frc in sess.get("feedback_runs", []):
            task = frc.get("task", "ball")
            try:
                parts.append(get_task_metadata(task).display_name)
            except (KeyError, ValueError):
                parts.append(task)
            for src in (frc.get("mask_sources", {}) or {}).values():
                if src:
                    reuse_sources.add(src)
        summary = "; ".join(parts) if parts else "(empty)"
        if reuse_sources:
            summary += "  <- " + ", ".join(sorted(reuse_sources))
        return summary

    def _build_sidebar_content(self) -> ft.Control:
        """Build the left scope list: Study-wide + one row per session."""
        sessions = self.study_config.session_config["sessions"]

        def _row(title: str, subtitle: str, selected: bool, on_click) -> ft.Control:
            lines: list[ft.Control] = [
                ft.Text(title, size=st.SUBSECTION, weight=ft.FontWeight.BOLD if selected else ft.FontWeight.W_500)
            ]
            if subtitle:
                lines.append(ft.Text(subtitle, size=st.CAPTION, color=st.MUTED, italic=True))
            return ft.Container(
                content=ft.Column(lines, spacing=1),
                on_click=on_click,
                bgcolor=st.SELECTED_BG if selected else None,
                border=ft.Border.only(left=ft.BorderSide(3, st.ACCENT)) if selected else None,
                border_radius=6,
                padding=ft.Padding.symmetric(vertical=st.GAP_SM, horizontal=st.GAP_MD),
                ink=True,
            )

        items: list[ft.Control] = [
            ft.Text("STUDY", size=st.CAPTION, weight=ft.FontWeight.BOLD, color=st.MUTED),
            _row(
                "Study-wide settings",
                "Paths, scanner, reference, display",
                self._active_scope == "study",
                lambda e: self._select_study_wide(),
            ),
            ft.Container(height=st.GAP_SM),
            ft.Row(
                [
                    ft.Text("SESSIONS", size=st.CAPTION, weight=ft.FontWeight.BOLD, color=st.MUTED),
                    ft.IconButton(
                        icon=ft.Icons.ADD, icon_size=st.ICON_MD, tooltip="Add session", on_click=self._on_add_session
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
        ]
        for i in range(len(sessions)):
            active = self._active_scope == "session" and i == self._active_session_index
            items.append(
                _row(
                    f"ses-{i + 1:02d}", self._session_summary(i), active, lambda e, idx=i: self._on_select_session(idx)
                )
            )
        return ft.Column(items, spacing=st.GAP_XS, scroll=ft.ScrollMode.AUTO, expand=True)

    def _build_session_header(self, index: int) -> ft.Control:
        """Scope banner + session lifecycle actions shown above the inner tabs."""
        n = len(self.study_config.session_config["sessions"])
        ses_id = f"ses-{index + 1:02d}"
        banner = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.INFO_OUTLINE, size=st.ICON_SM, color=st.ACCENT),
                    ft.Text(
                        f"Editing {ses_id} (Session {index + 1} of {n}) — the localizer & feedback "
                        f"tabs below apply to {ses_id} only.",
                        size=st.CAPTION,
                        color=st.ON_HEADER,
                    ),
                ],
                spacing=st.GAP_SM,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=st.HEADER_BG,
            border=ft.Border.only(left=ft.BorderSide(4, st.ACCENT)),
            border_radius=6,
            padding=st.GAP_MD,
            expand=True,
        )
        return ft.Row(
            [
                banner,
                ft.IconButton(
                    icon=ft.Icons.COPY, tooltip="Duplicate this session", on_click=self._on_duplicate_session
                ),
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE,
                    tooltip="Remove this session",
                    on_click=self._on_remove_session,
                    disabled=n <= 1,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _build_inner_tabs(self, index: int) -> ft.Control:
        """Inner tabs for the active session: Protocol + Localizer N + Feedback N."""
        n_loc = len(self._localizer_configs)
        n_fb = len(self._feedback_configs)

        tabs: list[ft.Control] = [ft.Tab(label="Protocol")]
        for i in range(n_loc):
            tabs.append(ft.Tab(label=self._get_localizer_tab_label(i)))
        for i in range(n_fb):
            tabs.append(ft.Tab(label=self._get_feedback_tab_label(i)))

        content: list[ft.Control] = [self._build_protocol_tab()]
        for loc_builder in self._localizer_builders:
            content.append(loc_builder.build())
        for fb_builder in self._feedback_builders:
            content.append(fb_builder.build())

        n = len(tabs)
        self._inner_tab_bar = ft.TabBar(tabs=tabs)
        self._inner_tab_view = ft.TabBarView(controls=content, expand=True)
        self._inner_tabs = ft.Tabs(
            content=ft.Column([self._inner_tab_bar, self._inner_tab_view], spacing=0, expand=True),
            length=n,
            selected_index=max(0, min(self._last_inner_index, n - 1)),
            expand=True,
        )
        return self._inner_tabs

    def _render_detail(self):
        """Point the detail pane at the current scope's content."""
        if self._active_scope == "study":
            self._detail_pane.content = self._study_wide_pane
        else:
            # Recreate (and build) the active session's tab builders so their lazy
            # network/contrast widgets exist for this session. Only done here — the
            # study-wide branch must NOT recreate builders, or in-progress network
            # selections (lazy checkboxes) would be dropped on the next collect.
            self._create_tab_builders()
            idx = self._active_session_index
            self._detail_pane.content = ft.Column(
                [self._build_session_header(idx), self._build_inner_tabs(idx)],
                spacing=st.GAP_SM,
                expand=True,
            )

    def _refresh_layout(self):
        """Rebuild the sidebar (summaries + highlight) and the detail pane."""
        if getattr(self, "_sidebar", None) is not None:
            self._sidebar.content = self._build_sidebar_content()
        self._render_detail()
        with contextlib.suppress(Exception):
            self.page.update()

    def _select_study_wide(self):
        """Switch the detail pane to study-wide config."""
        if self._active_scope == "study":
            return
        self._collect_all_configs()
        self._active_scope = "study"
        self._refresh_layout()

    def _rebuild_tabs(self):
        """Back-compat entry point: count/type-change handlers call this to refresh."""
        self._refresh_layout()

    # ------------------------------------------------------------------
    # Study-wide pane
    # ------------------------------------------------------------------

    def _build_study_wide_pane(self) -> ft.Container:
        """Study-wide configuration that applies to every session.

        Holds study metadata, output directory, reference scans, scanner/simulator
        settings, the simulator source mode + mapping/reference sources, and display
        settings. Per-session protocol (localizers/feedback) lives on the session
        Protocol tab, not here.
        """
        self.scanner_dicom_dir_browse_btn = self._create_browse_button(self.scanner_dicom_dir_field)

        is_simulator = self.simulator_enabled_checkbox.value
        self.collection_speed_field.visible = is_simulator
        self.feedback_simulator_speed_field.visible = is_simulator
        self.simulator_trigger_mode_dropdown.visible = is_simulator
        self.scanner_dicom_dir_field.visible = not is_simulator
        self.scanner_dicom_dir_browse_btn.visible = not is_simulator
        self.scanner_ip_field.visible = not is_simulator
        self.scanner_setup_guide_btn.visible = not is_simulator

        self.simulator_instruction_text = st.caption(
            "Simulator replays volumes from source directories at configurable speed for offline testing. "
            "Speed settings only control the rate at which DICOMs are delivered to the pipeline — "
            "the PsychoPy task itself always runs at real time."
            if is_simulator
            else "Scanner DICOM directory will be monitored for real-time acquisition.",
        )

        # Simulator source paths: mode radio + (mapping file | reference-scan sources).
        # Per-localizer/feedback sources are session-scoped and live on the Protocol tab.
        is_mapping = self.simulator_source_mode_radio.value == "mapping"
        self.simulator_source_mapping_row = st.browse_row(
            self.simulator_source_mapping_field,
            self._create_file_browse_button(
                self.simulator_source_mapping_field,
                filetypes=[("CSV/TSV files", "*.csv *.tsv"), ("All files", "*.*")],
            ),
        )
        self.simulator_mapping_container = ft.Column(
            [
                st.caption(
                    "Columns: subject, rest1, rest2, ..., reference_realtime, feedback_run-01, ...",
                    italic=False,
                ),
                self.simulator_source_mapping_row,
            ],
            visible=is_mapping,
        )
        self.simulator_paths_container = ft.Column(
            [
                ft.Divider(),
                st.section("Simulator Source Paths"),
                st.caption(
                    "DICOM source directories for offline testing. Per-session sources "
                    "(reference scans and per-task runs) live on each session's Protocol tab.",
                ),
                self.simulator_source_mode_radio,
                self.simulator_mapping_container,
            ],
            visible=is_simulator,
        )

        return ft.Container(
            content=ft.Column(
                [
                    st.section("Study Information"),
                    st.caption("Study metadata shown in the GUI and used for organizing session data."),
                    st.field_row(self.name_field),
                    st.field_row(self.description_field),
                    st.field_row(self.example_participant_id_field),
                    ft.Divider(),
                    st.section("Output Directory"),
                    st.caption("All processed data, masks, and feedback output will be saved under this directory."),
                    st.browse_row(self.base_dir_field, self._create_browse_button(self.base_dir_field)),
                    ft.Divider(),
                    st.section("Reference Scans"),
                    st.caption(
                        "Short scans used for spatial registration. Can be re-collected at any time during the session."
                    ),
                    ft.Row(
                        [
                            ft.Container(self.reference_initial_enabled_checkbox, expand=True),
                            self.reference_initial_n_volumes_field,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        [
                            ft.Container(self.reference_realtime_enabled_checkbox, expand=True),
                            self.reference_realtime_n_volumes_field,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Divider(),
                    st.section("Scanner & Simulator"),
                    self.simulator_enabled_checkbox,
                    st.field_pair(self.collection_speed_field, self.feedback_simulator_speed_field),
                    st.field_row(self.simulator_trigger_mode_dropdown),
                    st.browse_row(self.scanner_dicom_dir_field, self.scanner_dicom_dir_browse_btn),
                    st.browse_row(self.scanner_ip_field, self.scanner_setup_guide_btn),
                    st.field_row(self.scanner_trigger_keys_field),
                    self.simulator_instruction_text,
                    self.simulator_paths_container,
                    ft.Divider(),
                    st.section("Display Settings"),
                    st.field_pair(self.control_monitor_dropdown, self.presenter_monitor_dropdown),
                    st.field_row(self._build_monitor_preview()),
                    st.caption("Configure which monitor shows the control GUI vs the participant task display."),
                    ft.Divider(),
                    st.section("Sham / Control Condition"),
                    st.caption(
                        "Study-wide control settings. Each feedback task's sham mask construction "
                        "(delivery method, networks) is configured in that task's tab."
                    ),
                    self.sham_enabled_checkbox,
                    self.sham_blind_mode_checkbox,
                    self.sham_randomization_row,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=st.GAP_MD,
            ),
            padding=st.PAD,
        )

    # ------------------------------------------------------------------
    # Session Protocol tab
    # ------------------------------------------------------------------

    def _build_protocol_tab(self) -> ft.Container:
        """The active session's protocol: experiment design, notes, simulator sources.

        A session computes its localizers fresh. To use an earlier session's masks,
        a feedback run selects them directly (the feedback tab's mask dropdowns list
        every earlier session's masks); cross-session reuse is materialized when the
        study is flattened, so no per-localizer reuse control is needed here.
        """
        # --- Experiment Design: Localizer Tasks ---
        loc_rows: list[ft.Control] = []
        self._loc_type_dropdowns: list[ft.Dropdown] = []
        for i, cfg in enumerate(self._localizer_configs):
            dd = ft.Dropdown(
                label=f"Localizer {i + 1}",
                options=[ft.dropdown.Option(key=o.key, text=o.text) for o in self._localizer_type_options],
                value=cfg.get("type", "rest"),
                on_select=lambda e, idx=i: self._on_localizer_type_changed(idx, e),
            )
            self._loc_type_dropdowns.append(dd)
            remove_btn = ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip="Remove this localizer",
                on_click=lambda e, idx=i: self._on_remove_localizer_at(idx),
            )
            loc_rows.append(st.browse_row(dd, remove_btn))

        add_loc_btn = ft.TextButton("Add Localizer Task", icon=ft.Icons.ADD, on_click=self._on_add_localizer)

        # --- Experiment Design: Feedback Tasks ---
        fb_rows: list[ft.Control] = []
        self._fb_type_dropdowns: list[ft.Dropdown] = []
        for i, cfg in enumerate(self._feedback_configs):
            dd = ft.Dropdown(
                label=f"Feedback {i + 1}",
                options=[ft.dropdown.Option(key=o.key, text=o.text) for o in self._feedback_type_options],
                value=cfg.get("task", "ball"),
                on_select=lambda e, idx=i: self._on_feedback_type_changed(idx, e),
            )
            self._fb_type_dropdowns.append(dd)
            remove_btn = ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip="Remove this feedback task",
                on_click=lambda e, idx=i: self._on_remove_feedback_at(idx),
            )
            fb_rows.append(st.browse_row(dd, remove_btn))

        add_fb_btn = ft.TextButton("Add Feedback Task", icon=ft.Icons.ADD, on_click=self._on_add_feedback)

        self._notes_field = ft.TextField(
            label="Session notes",
            value=self._active_session().get("notes", ""),
            multiline=True,
            min_lines=1,
            max_lines=3,
            hint_text="Optional notes for this session",
        )

        # Per-session simulator source DICOMs (only shown in individual simulator mode).
        is_simulator = self.simulator_enabled_checkbox.value
        is_individual = self.simulator_source_mode_radio.value != "mapping"
        self._session_sim_container = ft.Column(
            [
                ft.Divider(),
                st.subsection("Simulator Source DICOMs (this session)"),
                *self._build_session_simulator_paths(),
            ],
            visible=is_simulator and is_individual,
            spacing=st.GAP_MD,
        )

        return ft.Container(
            content=ft.Column(
                [
                    st.section("Experiment Design"),
                    st.caption(
                        "Define this session's localizer and feedback tasks. Each has its own configuration tab above."
                    ),
                    st.subsection("Localizer Tasks"),
                    *loc_rows,
                    add_loc_btn,
                    st.subsection("Feedback Tasks"),
                    *fb_rows,
                    add_fb_btn,
                    ft.Divider(),
                    st.field_row(self._notes_field),
                    self._session_sim_container,
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=st.GAP_MD,
            ),
            padding=st.PAD,
        )

    # ------------------------------------------------------------------
    # Simulator source paths (split: per-session sources vs study-wide references)
    # ------------------------------------------------------------------

    def _build_session_simulator_paths(self) -> list[ft.Control]:
        """Per-session simulator source DICOM fields for the active session.

        Includes the session's reference-scan sources (initial/realtime) followed by
        the per-localizer-run and per-feedback-run sources.
        """
        controls: list[ft.Control] = []

        # Store fields for later collection (per active session).
        self._sim_loc_source_fields: list[list[ft.TextField]] = []
        self._sim_fb_source_fields: list[ft.TextField] = []
        self._sim_fb_transfer_fields: list[ft.TextField] = []

        # --- Reference-scan sources (per session; enabled flags are study-wide) ---
        controls.extend(self._build_reference_simulator_paths())

        # --- Localizer sources ---
        for loc_idx, cfg in enumerate(self._localizer_configs):
            loc_type = cfg.get("type", "rest")
            # Manual localizers import pre-built masks — no localizer scan, so
            # no source DICOMs. Keep index alignment with an empty field list.
            if loc_type == "manual":
                self._sim_loc_source_fields.append([])
                continue
            label = f"Localizer {loc_idx + 1} ({self._loc_type_short(loc_type)})"
            controls.append(st.subsection(label))

            source_dicoms = cfg.get("source_dicoms", [])
            if isinstance(source_dicoms, str):
                source_dicoms = [source_dicoms]

            try:
                n_runs = int(cfg.get("n_runs", 2))
            except (ValueError, TypeError):
                n_runs = 2

            loc_fields: list[ft.TextField] = []
            for run_idx in range(n_runs):
                val = source_dicoms[run_idx] if run_idx < len(source_dicoms) else ""
                field = ft.TextField(
                    label=f"Run {run_idx + 1} source DICOMs",
                    value=val,
                    hint_text="Path to DICOM directory",
                )
                loc_fields.append(field)
                controls.append(st.browse_row(field, self._create_browse_button(field)))

            self._sim_loc_source_fields.append(loc_fields)
            controls.append(ft.Divider())

        # --- Feedback sources ---
        for fb_idx, cfg in enumerate(self._feedback_configs):
            task_type = cfg.get("task", "ball")
            try:
                meta = get_task_metadata(task_type)
                label = f"Feedback {fb_idx + 1} ({meta.display_name})"
            except (KeyError, ValueError):
                label = f"Feedback {fb_idx + 1} ({task_type})"

            controls.append(st.subsection(label))

            fb_field = ft.TextField(
                label="Feedback run source DICOMs",
                value=cfg.get("source_dicoms", ""),
                hint_text="Path to DICOM directory",
            )
            self._sim_fb_source_fields.append(fb_field)
            controls.append(st.browse_row(fb_field, self._create_browse_button(fb_field)))

            # Transfer run source (only if transfers enabled)
            use_transfers = resolve_use_transfer_runs(cfg)
            tf_field = ft.TextField(
                label="Transfer run source DICOMs",
                value=cfg.get("transfer_source_dicoms", ""),
                hint_text="Path to DICOM directory",
                visible=use_transfers,
            )
            self._sim_fb_transfer_fields.append(tf_field)
            tf_row = st.browse_row(tf_field, self._create_browse_button(tf_field))
            tf_row.visible = use_transfers
            controls.append(tf_row)

            controls.append(ft.Divider())

        return controls

    def _build_reference_simulator_paths(self) -> list[ft.Control]:
        """Reference-scan simulator source DICOM fields for the active session.

        The reference scan transcends individual tasks but its replayed DICOMs differ
        per session, so the paths live on each session's protocol. The enabled flags and
        volume counts remain study-wide (in the top-level ``reference`` block).
        """
        controls: list[ft.Control] = []
        reference_dicoms = self._active_session().get("reference_dicoms", {})

        # The initial reference registers fresh localizer masks, and the realtime
        # reference precedes feedback — so each is only collected (and thus only
        # needs a simulator source) when this session actually has that acquisition.
        # Initial reference
        init_enabled = self.reference_initial_enabled_checkbox.value and bool(self._localizer_configs)
        self._sim_loc_ref_label = st.subsection("Initial Reference Scan")
        self._sim_loc_ref_label.visible = init_enabled
        controls.append(self._sim_loc_ref_label)
        self._sim_loc_ref_field = ft.TextField(
            label="Initial Reference Scan source DICOMs",
            value=reference_dicoms.get("initial", ""),
            hint_text="Path to DICOM directory",
        )
        self._sim_loc_ref_row = st.browse_row(
            self._sim_loc_ref_field, self._create_browse_button(self._sim_loc_ref_field)
        )
        self._sim_loc_ref_row.visible = init_enabled
        controls.append(self._sim_loc_ref_row)

        # Real-time reference
        rt_enabled = self.reference_realtime_enabled_checkbox.value and bool(self._feedback_configs)
        self._sim_fb_ref_label = st.subsection("Real-Time Reference Scan")
        self._sim_fb_ref_label.visible = rt_enabled
        controls.append(self._sim_fb_ref_label)
        self._sim_fb_ref_field = ft.TextField(
            label="Real-Time Reference Scan source DICOMs",
            value=reference_dicoms.get("realtime", ""),
            hint_text="Path to DICOM directory",
        )
        self._sim_fb_ref_row = st.browse_row(self._sim_fb_ref_field, self._create_browse_button(self._sim_fb_ref_field))
        self._sim_fb_ref_row.visible = rt_enabled
        controls.append(self._sim_fb_ref_row)

        return controls

    def _collect_simulator_paths(self):
        """Collect this session's simulator source paths from the Protocol tab fields.

        Reference scan DICOM paths are written to the active session's ``reference_dicoms``
        block; per-run source DICOMs are written to localizer/feedback config dicts.
        Always persists so switching sessions preserves the edits.
        """
        # Localizer sources
        for loc_idx, loc_fields in enumerate(getattr(self, "_sim_loc_source_fields", [])):
            if loc_idx < len(self._localizer_configs):
                self._localizer_configs[loc_idx]["source_dicoms"] = [f.value.strip() for f in loc_fields]

        # Feedback sources
        for fb_idx, fb_field in enumerate(getattr(self, "_sim_fb_source_fields", [])):
            if fb_idx < len(self._feedback_configs):
                self._feedback_configs[fb_idx]["source_dicoms"] = fb_field.value.strip()

        # Feedback transfer sources
        for fb_idx, tf_field in enumerate(getattr(self, "_sim_fb_transfer_fields", [])):
            if fb_idx < len(self._feedback_configs):
                self._feedback_configs[fb_idx]["transfer_source_dicoms"] = tf_field.value.strip()

        # Reference scan DICOM paths (per session)
        _loc_ref = getattr(self, "_sim_loc_ref_field", None)
        _fb_ref = getattr(self, "_sim_fb_ref_field", None)
        if _loc_ref is not None or _fb_ref is not None:
            ref_dicoms = self._active_session().setdefault("reference_dicoms", {})
            if _loc_ref is not None:
                ref_dicoms["initial"] = _loc_ref.value.strip()
            if _fb_ref is not None:
                ref_dicoms["realtime"] = _fb_ref.value.strip()

    # ------------------------------------------------------------------
    # Event handlers (Study tab)
    # ------------------------------------------------------------------

    def _on_reference_scan_changed(self, e):
        """Handle reference scan enable/disable toggle."""
        self.reference_initial_n_volumes_field.visible = self.reference_initial_enabled_checkbox.value
        self.reference_realtime_n_volumes_field.visible = self.reference_realtime_enabled_checkbox.value
        self._sync_paths_tab_visibility()
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_name_change(self, e):
        if not self.name_field.value.strip():
            self.name_field.error_text = "Study name is required"
            self.validation_errors["name"] = "Study name is required"
        else:
            self.name_field.error_text = None
            self.validation_errors.pop("name", None)
        self.name_field.update()

    def _on_simulator_mode_changed(self, e):
        is_simulator = self.simulator_enabled_checkbox.value
        self.collection_speed_field.visible = is_simulator
        self.feedback_simulator_speed_field.visible = is_simulator
        self.simulator_trigger_mode_dropdown.visible = is_simulator
        self.scanner_dicom_dir_field.visible = not is_simulator
        if hasattr(self, "scanner_dicom_dir_browse_btn"):
            self.scanner_dicom_dir_browse_btn.visible = not is_simulator
        self.scanner_ip_field.visible = not is_simulator
        self.scanner_setup_guide_btn.visible = not is_simulator

        if hasattr(self, "simulator_instruction_text"):
            self.simulator_instruction_text.value = (
                "Simulator replays volumes from source directories at configurable speed for offline testing. "
                "Speed settings only control the rate at which DICOMs are delivered to the pipeline — "
                "the PsychoPy task itself always runs at real time."
                if is_simulator
                else "Scanner DICOM directory will be monitored for real-time acquisition."
            )
        if hasattr(self, "simulator_paths_container"):
            self.simulator_paths_container.visible = is_simulator
        if getattr(self, "_session_sim_container", None) is not None:
            self._session_sim_container.visible = is_simulator and self.simulator_source_mode_radio.value != "mapping"

        self.page.update()

    def _on_simulator_source_mode_changed(self, e):
        mode = self.simulator_source_mode_radio.value
        if hasattr(self, "simulator_mapping_container"):
            self.simulator_mapping_container.visible = mode == "mapping"
        if getattr(self, "_session_sim_container", None) is not None:
            self._session_sim_container.visible = mode != "mapping" and self.simulator_enabled_checkbox.value
        self.page.update()

    # ------------------------------------------------------------------
    # Browse button factories
    # ------------------------------------------------------------------

    def _create_browse_button(self, field: ft.TextField) -> ft.IconButton:
        """Create a browse button for directory selection."""

        def on_click(e):
            from pineuro.gui.utils import pick_directory

            path = pick_directory(title=f"Select {field.label}", initial_path=field.value or None)
            if path:
                field.value = path
                field.update()

        return ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Browse for folder",
            on_click=on_click,
        )

    def _create_file_browse_button(
        self,
        field: ft.TextField,
        filetypes: list[tuple[str, str]] | None = None,
        on_pick: Callable[[str], None] | None = None,
    ) -> ft.IconButton:
        """Create a browse button for file selection.

        Parameters
        ----------
        field : ft.TextField
            Field whose value is set to the picked path.
        filetypes : list of (label, pattern), optional
            tkinter file-type filters (ignored on macOS).
        on_pick : callable, optional
            Called with the picked path after ``field.value`` is set, e.g. to
            auto-populate a sibling field.
        """

        def on_click(e):
            from pineuro.gui.utils import pick_file

            path = pick_file(
                title=f"Select {field.label}",
                filetypes=filetypes,
                initial_path=field.value or None,
            )
            if path:
                field.value = path
                field.update()
                if on_pick is not None:
                    on_pick(path)

        return ft.IconButton(
            icon=ft.Icons.FILE_OPEN,
            tooltip="Browse for file",
            on_click=on_click,
        )

    # ------------------------------------------------------------------
    # Sham randomization
    # ------------------------------------------------------------------

    def _on_study_sham_enabled_changed(self, e):
        """Toggle Blind Mode + randomization controls with the study-wide sham switch."""
        self._update_study_sham_visibility()

    def _update_study_sham_visibility(self):
        """Show Blind Mode and the randomization file only when sham is enabled."""
        enabled = bool(self.sham_enabled_checkbox.value)
        self.sham_blind_mode_checkbox.visible = enabled
        self.sham_randomization_row.visible = enabled
        with contextlib.suppress(Exception):
            self.page.update()

    def _on_generate_randomization(self, e):
        """Open a dialog to generate a study-wide randomization file."""
        prefix_field = ft.TextField(
            label="Participant ID prefix",
            value="sub-",
            width=200,
            hint_text="e.g. sub- → sub-01, sub-rtchr0 → sub-rtchr001",
        )
        n_participants_field = ft.TextField(
            label="Number of participants",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        n_sham_field = ft.TextField(
            label="Number of sham",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )
        method_dropdown = ft.Dropdown(
            label="Method",
            width=200,
            options=[
                ft.dropdown.Option(key="simple", text="Simple"),
                ft.dropdown.Option(key="replay", text="Replay"),
            ],
            value="simple",
        )
        seed_field = ft.TextField(
            label="Seed (optional)",
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Generate Randomization File"),
            content=ft.Container(
                content=ft.Column(
                    [prefix_field, n_participants_field, n_sham_field, method_dropdown, seed_field],
                    tight=True,
                    spacing=st.GAP_SM,
                ),
                width=300,
            ),
        )

        def generate(e):
            prefix = prefix_field.value.strip()
            if not prefix:
                prefix_field.error_text = "Required"
                prefix_field.update()
                return

            try:
                n_total = int(n_participants_field.value)
                if n_total < 2:
                    raise ValueError
            except (TypeError, ValueError):
                n_participants_field.error_text = "Must be an integer >= 2"
                n_participants_field.update()
                return
            n_participants_field.error_text = None

            try:
                n_sham = int(n_sham_field.value)
                if n_sham < 1 or n_sham >= n_total:
                    raise ValueError
            except (TypeError, ValueError):
                n_sham_field.error_text = f"Must be between 1 and {n_total - 1}"
                n_sham_field.update()
                return
            n_sham_field.error_text = None

            seed_val = None
            if seed_field.value and seed_field.value.strip():
                try:
                    seed_val = int(seed_field.value)
                except ValueError:
                    seed_field.error_text = "Must be an integer"
                    seed_field.update()
                    return

            from pineuro.gui.utils import pick_save_file

            save_path = pick_save_file(
                title="Save randomization file",
                default_extension=".tsv",
                filetypes=[("TSV files", "*.tsv"), ("CSV files", "*.csv")],
            )
            if not save_path:
                return

            from pineuro.sham import generate_randomization

            participant_ids = [f"{prefix}{i:02d}" for i in range(1, n_total + 1)]
            generate_randomization(
                participant_ids=participant_ids,
                n_sham=n_sham,
                method=method_dropdown.value,
                output_path=save_path,
                seed=seed_val,
            )

            self.sham_randomization_file_field.value = save_path
            self.sham_randomization_file_field.update()
            self.page.pop_dialog()
            _snack = ft.SnackBar(ft.Text(f"Randomization file saved: {save_path}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.page.pop_dialog()),
            ft.ElevatedButton("Generate", on_click=generate),
        ]
        self.page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # Monitor preview
    # ------------------------------------------------------------------

    def _build_monitor_preview(self) -> ft.Container:
        try:
            from pineuro.display import get_monitors

            monitors = get_monitors()
            if not monitors:
                return ft.Container(
                    content=ft.Text("No monitors detected", size=st.CAPTION, color=st.WARNING),
                    padding=st.GAP_MD,
                )

            monitor_texts = []
            for m in monitors:
                position = "Left" if m.index == 0 else "Right"
                monitor_texts.append(
                    ft.Text(f"{position}: {m.name} ({m.width}x{m.height})", size=st.CAPTION, color=st.MUTED)
                )

            return ft.Container(
                content=ft.Column(
                    [ft.Text("Detected monitors:", size=st.CAPTION, weight=ft.FontWeight.BOLD), *monitor_texts],
                    spacing=st.GAP_XS,
                ),
                bgcolor=st.SUBTLE_BG,
                padding=st.GAP_MD,
                border_radius=5,
            )
        except Exception as e:
            logger.warning(f"Failed to detect monitors: {e}")
            return ft.Container(
                content=ft.Text(f"Monitor detection unavailable: {e}", size=st.CAPTION, color=st.WARNING),
                padding=st.GAP_MD,
            )

    def _show_scanner_setup_guide(self):
        """Show platform-specific SMB share setup instructions dialog."""
        from pineuro.scanner_setup import get_current_platform_key, get_setup_instructions

        watch_dir = self.scanner_dicom_dir_field.value.strip() or "/BOLD_DICOM"
        scanner_ip = self.scanner_ip_field.value.strip()

        instructions = get_setup_instructions(watch_dir, scanner_ip)
        platform_key = get_current_platform_key()
        initial_index = {"windows": 0, "macos": 1, "linux": 2}.get(platform_key, 0)

        def _make_content(text: str) -> ft.Container:
            return ft.Container(
                content=ft.Column(
                    [ft.Text(text, font_family="monospace", selectable=True, size=st.CAPTION)],
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=st.GAP_MD,
            )

        tabs = ft.Tabs(
            length=3,
            selected_index=initial_index,
            content=ft.Column(
                [
                    ft.TabBar(
                        tabs=[ft.Tab(label="Windows"), ft.Tab(label="macOS"), ft.Tab(label="Linux")],
                    ),
                    ft.TabBarView(
                        controls=[
                            _make_content(instructions["windows"]),
                            _make_content(instructions["macos"]),
                            _make_content(instructions["linux"]),
                        ],
                        expand=True,
                    ),
                ],
                expand=True,
            ),
            expand=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Scanner Setup Guide"),
            content=ft.Container(
                content=ft.Column(
                    [
                        st.subsection("SMB Share Setup"),
                        ft.Container(content=tabs, height=300),
                        ft.Divider(),
                        st.subsection("Scanner Console Configuration"),
                        ft.Text(instructions["scanner_notes"], size=st.CAPTION),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=st.DIALOG_LG,
                height=550,
            ),
            actions=[ft.TextButton("Close", on_click=lambda e: _close())],
        )

        def _close():
            dialog.open = False
            self.page.update()

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    # ------------------------------------------------------------------
    # Save / Cancel / Start
    # ------------------------------------------------------------------

    def _on_cancel(self, e):
        logger.info("Study editor cancelled")
        if self.on_cancel_callback:
            self.on_cancel_callback()

    def _validate_sessions(self, sessions: list, realtime_enabled: bool = True) -> str | None:
        """Return an error string if the per-session protocol is invalid, else None.

        Requires at least one session, each with a localizer or feedback task; every
        mask a feedback run uses must be produced by a fresh localizer in the chosen
        source session (this session for an unqualified mask, or the earlier session
        recorded in its ``mask_sources``); a mask name must not be sourced from two
        different sessions within one session; and cross-session reuse requires the
        realtime reference (``realtime_enabled``) to coregister source-session masks.
        """
        if not sessions:
            return "At least one session is required"

        def _fresh_names(sess: dict) -> set[str]:
            names: set[str] = set()
            for loc in sess.get("localizers", []):
                if not localizer_is_reused(loc):
                    names.update(localizer_mask_names(loc))
            return names

        for idx, sess in enumerate(sessions):
            locs = sess.get("localizers", [])
            fbs = sess.get("feedback_runs", [])
            if not locs and not fbs:
                return f"Session {idx + 1} has no localizer or feedback task"

            this_fresh = _fresh_names(sess)
            chosen_source: dict[str, str] = {}  # name -> source label (collision check)
            for frc in fbs:
                srcs = frc.get("mask_sources", {}) or {}
                for name in sorted(frc_referenced_masks(frc)):
                    source = srcs.get(name, "")
                    prior = chosen_source.get(name)
                    if prior is not None and prior != source:
                        a = prior or "this session"
                        b = source or "this session"
                        return f"Session {idx + 1}: mask '{name}' is sourced from both {a} and {b}"
                    chosen_source[name] = source

                    if not source:
                        if name not in this_fresh:
                            return (
                                f"Session {idx + 1}: feedback mask '{name}' is not produced by a localizer "
                                f"in this session (choose an earlier session that produces it)"
                            )
                        continue
                    src_num = _ses_num(source)
                    if src_num is None or not (1 <= src_num < idx + 1):
                        return (
                            f"Session {idx + 1}: mask '{name}' sources from '{source}', which is not an earlier session"
                        )
                    if name not in _fresh_names(sessions[src_num - 1]):
                        return f"Session {idx + 1}: mask '{name}' is not produced by {source}"

            reuse_error = reuse_realtime_error(sessions, idx, realtime_enabled)
            if reuse_error:
                return reuse_error
        return None

    def _save_study(self) -> bool:
        """Validate and persist the current study to disk.

        Shows a snackbar on success or failure. Refreshes the baseline
        snapshot on success so the editor is no longer "dirty".
        Does NOT invoke ``on_save_callback`` (no navigation).

        Returns:
            True if the study was written; False otherwise.
        """
        if self.validation_errors:
            _snack = ft.SnackBar(ft.Text("Please fix validation errors before saving"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return False

        self._update_study_from_form()

        # Validate sessions: at least one session, each with a localizer or feedback
        # task, and no localizer reusing a non-earlier session.
        config = self.study_config.session_config
        session_error = self._validate_sessions(
            config.get("sessions", []), config.get("reference", {}).get("realtime_enabled", True)
        )
        if session_error:
            _snack = ft.SnackBar(ft.Text(session_error), bgcolor=st.ERROR_BG)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return False

        try:
            path = self.config_manager.save_study(self.study_config, overwrite=True)
        except Exception as ex:
            logger.error(f"Failed to save study: {ex}")
            _snack = ft.SnackBar(ft.Text(f"Error saving study: {ex}"))
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return False

        _snack = ft.SnackBar(ft.Text(f"Study saved: {self.study_config.metadata.name}"))
        _snack.open = True
        self.page.overlay.append(_snack)
        self.page.update()
        logger.info(f"Study saved: {path}")
        self._saved_snapshot = self._capture_snapshot()
        return True

    def _on_save(self, e):
        if self._save_study() and self.on_save_callback:
            self.on_save_callback()

    def _on_save_as(self, e):
        new_name_field = ft.TextField(
            label="New study name",
            value=f"{self.study_config.metadata.name} (Copy)",
            autofocus=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Save Study As"),
            content=ft.Container(content=new_name_field, width=400),
        )

        def save_with_new_name(e):
            new_name = new_name_field.value.strip()
            if not new_name:
                new_name_field.error_text = "Study name is required"
                new_name_field.update()
                return

            existing = self.config_manager.list_studies()
            if any(s.name == new_name for s in existing):
                new_name_field.error_text = "A study with this name already exists"
                new_name_field.update()
                return

            self._update_study_from_form()
            self.study_config.metadata.name = new_name
            self.study_config.file_path = None

            try:
                path = self.config_manager.save_study(self.study_config, overwrite=False)
                self.page.pop_dialog()
                _snack = ft.SnackBar(ft.Text(f"Study saved as: {new_name}"))
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                logger.info(f"Study saved as: {path}")
                self._saved_snapshot = self._capture_snapshot()
                if self.on_save_callback:
                    self.on_save_callback()
            except Exception as ex:
                logger.error(f"Failed to save study: {ex}")
                new_name_field.error_text = str(ex)
                new_name_field.update()

        dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.page.pop_dialog()),
            ft.ElevatedButton("Save", on_click=save_with_new_name),
        ]
        self.page.show_dialog(dialog)

    def _on_start_session(self, e):
        self._update_study_from_form()
        if self._has_unsaved_changes():
            self._prompt_unsaved_changes_before_start()
            return
        self._show_participant_id_dialog()

    def _prompt_unsaved_changes_before_start(self):
        def close():
            self.page.pop_dialog()

        def on_save_and_start(_):
            close()
            if self._save_study():
                self._show_participant_id_dialog()

        def on_start_without_saving(_):
            close()
            self._show_participant_id_dialog()

        dialog = ft.AlertDialog(
            title=ft.Text("Save changes before starting?"),
            content=ft.Text(
                "You have unsaved changes to this study. Save them before starting the participant session?"
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: close()),
                ft.TextButton("Start without Saving", on_click=on_start_without_saving),
                ft.FilledButton("Save and Start", on_click=on_save_and_start),
            ],
        )
        self.page.show_dialog(dialog)

    def _show_participant_id_dialog(self):
        from pathlib import Path

        from ..components.session_selection_dialog import SessionSelectionDialog

        example_id = self.example_participant_id_field.value.strip() or "sub-001"

        participant_id_field = ft.TextField(
            label="Participant ID",
            hint_text=f"e.g., {example_id}",
            autofocus=True,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Start Participant Session"),
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            f"Study: {self.study_config.metadata.name}", size=st.SUBSECTION, weight=ft.FontWeight.BOLD
                        ),
                        ft.Divider(),
                        participant_id_field,
                        ft.Text(
                            "Enter the participant ID to begin the neurofeedback session.",
                            size=st.CAPTION,
                            color=st.MUTED,
                            italic=True,
                        ),
                    ],
                    tight=True,
                    spacing=st.GAP_SM,
                ),
                width=400,
            ),
        )

        def start_session(e):
            from ..utils import validate_participant_id

            participant_id = participant_id_field.value.strip()
            ok, msg = validate_participant_id(participant_id)
            if not ok:
                participant_id_field.error_text = msg
                participant_id_field.update()
                return

            self.page.pop_dialog()
            logger.info(f"Starting session for participant: {participant_id}")

            base_dir = Path(self.study_config.session_config["paths"]["base_dir"])

            def on_session_selected(session_id, session_metadata):
                if self.on_start_session_callback:
                    self.on_start_session_callback(self.study_config, participant_id, session_id, session_metadata)

            session_dialog = SessionSelectionDialog(
                page=self.page,
                participant_id=participant_id,
                base_dir=base_dir,
                on_select=on_session_selected,
                study_config=self.study_config,
            )
            session_dialog.show()

        dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.page.pop_dialog()),
            ft.FilledButton("Start", icon=ft.Icons.PLAY_ARROW, on_click=start_session),
        ]
        self.page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # Save logic
    # ------------------------------------------------------------------

    def _update_study_from_form(self):
        """Update study config from form field values (new-shape YAML)."""
        self.study_config.metadata.name = self.name_field.value.strip()
        self.study_config.metadata.description = self.description_field.value.strip()

        config = self.study_config.session_config

        # Paths
        config["paths"]["example_participant_id"] = self.example_participant_id_field.value.strip() or "sub-001"
        config["paths"]["base_dir"] = self.base_dir_field.value.strip() or "/data/pineuro"

        config.pop("study", None)

        # Reference scans (enabled flags + volume counts are study-wide; the DICOM source
        # paths are per-session and live in each session's ``reference_dicoms``).
        if "reference" not in config:
            config["reference"] = {}
        # Drop any stale study-wide source paths from older configs.
        config["reference"].pop("initial_dicoms", None)
        config["reference"].pop("realtime_dicoms", None)
        config["reference"]["initial_enabled"] = self.reference_initial_enabled_checkbox.value
        try:
            config["reference"]["initial_n_volumes"] = int(self.reference_initial_n_volumes_field.value or 2)
        except (ValueError, TypeError):
            config["reference"]["initial_n_volumes"] = 2
        config["reference"]["realtime_enabled"] = self.reference_realtime_enabled_checkbox.value
        try:
            config["reference"]["realtime_n_volumes"] = int(self.reference_realtime_n_volumes_field.value or 2)
        except (ValueError, TypeError):
            config["reference"]["realtime_n_volumes"] = 2

        # Simulator
        if "simulator" not in config:
            config["simulator"] = {}
        config["simulator"]["enabled"] = self.simulator_enabled_checkbox.value
        try:
            config["simulator"]["collection_speed"] = float(self.collection_speed_field.value or 1.0)
        except (ValueError, TypeError):
            config["simulator"]["collection_speed"] = 1.0
        try:
            config["simulator"]["feedback_speed"] = float(self.feedback_simulator_speed_field.value or 1.0)
        except (ValueError, TypeError):
            config["simulator"]["feedback_speed"] = 1.0
        config["simulator"]["trigger_mode"] = self.simulator_trigger_mode_dropdown.value or "manual"

        # Scanner
        if "scanner" not in config:
            config["scanner"] = {}
        config["scanner"]["dicom_dir"] = self.scanner_dicom_dir_field.value.strip() or "/BOLD_DICOM"
        config["scanner"]["ip"] = self.scanner_ip_field.value.strip()
        trigger_keys = [k.strip() for k in (self.scanner_trigger_keys_field.value or "").split(",") if k.strip()]
        config["scanner"]["trigger_keys"] = trigger_keys or ["t", "+", "5", "="]

        # Simulator source mapping
        if self.simulator_source_mode_radio.value == "mapping":
            config["simulator"]["source_mapping"] = self.simulator_source_mapping_field.value.strip()
        else:
            config["simulator"]["source_mapping"] = ""

        # Display
        if "display" not in config:
            config["display"] = {}
        config["display"]["control_monitor"] = self.control_monitor_dropdown.value or "left"
        config["display"]["presenter_monitor"] = self.presenter_monitor_dropdown.value or "right"

        # Sham (study-wide gating/randomization; per-task construction is collected per feedback run)
        config["sham"] = {
            "enabled": self.sham_enabled_checkbox.value,
            "blind_mode": self.sham_blind_mode_checkbox.value,
            "randomization_file": self.sham_randomization_file_field.value.strip(),
        }

        # --- Collect protocol from tab builders into the active session ---
        self._localizer_configs = [b.collect_config() for b in self._localizer_builders]
        self._feedback_configs = [b.collect_config() for b in self._feedback_builders]

        # Collect this session's simulator paths (per-run sources into the protocol
        # configs, reference paths into the active session's reference_dicoms).
        self._collect_simulator_paths()

        sess = self._active_session()
        sess["localizers"] = self._localizer_configs
        sess["feedback_runs"] = self._feedback_configs
        if getattr(self, "_notes_field", None) is not None:
            sess["notes"] = self._notes_field.value or ""

        # Per-session protocol lives only under ``sessions`` — never at the top level.
        config.pop("localizers", None)
        config.pop("feedback_runs", None)
