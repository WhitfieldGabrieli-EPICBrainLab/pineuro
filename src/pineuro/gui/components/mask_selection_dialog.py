"""Mask selection dialog for reviewing and customizing extracted network masks.

Provides a two-stage dialog flow:
1. Accept/Reject each mask based on the ICA-derived result
2. Select alternative (different ICA component or Yeo atlas fallback) for rejected masks
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from pineuro.gui import styles as st

if TYPE_CHECKING:
    from pineuro.mask_extraction import MaskSelection

from .qc_viewer import open_qc_report

logger = logging.getLogger(__name__)


@dataclass
class MaskReviewInfo:
    """Information about a network mask for the review dialog."""

    network_name: str
    current_component: int
    current_correlation: float
    alternative_components: list[tuple[int, float]]  # [(idx, corr), ...] ranked
    overlay_path: Path | None  # Path to current mask overlay image
    alternative_overlay_paths: dict[int, Path]  # component_idx -> overlay path
    yeo_overlay_path: Path | None  # Path to Yeo atlas overlay
    accepted: bool = True  # Default to accepted
    combined_overlay_path: Path | None = None  # Path to combined mask preview (when two components selected)


class MaskSelectionDialog:
    """Modal dialog for reviewing and customizing extracted network masks.

    Implements a two-stage flow:
    1. Stage 1: Accept/Reject - Show each mask with overlay, user accepts or rejects
    2. Stage 2: Select Alternative - For rejected masks, show alternatives to choose from

    Usage:
        dialog = MaskSelectionDialog(
            page=page,
            masks_info=masks_info,
            on_complete=lambda selections: print(selections),
        )
        dialog.show()
    """

    # Correlation quality thresholds
    CORR_GOOD = 0.15
    CORR_MODERATE = 0.1

    def __init__(
        self,
        page: ft.Page,
        masks_info: list[MaskReviewInfo],
        participant_id: str,
        session_id: str,
        masks_dir: Path,
        components_file: Path,
        yeo_subject_space: Path,
        network_atlas: str = "yeo_17",
        threshold: int = 2000,
        on_complete: Callable[[dict[str, MaskSelection]], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        # Additional context for script generation
        mask_extraction_dir: Path | None = None,
        input_file: Path | None = None,
        brain_mask_file: Path | None = None,
        ref_file: Path | None = None,
        networks: list[str] | None = None,
        n_components: int = 30,
        smooth_fwhm: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        # Hemisphere split (sham verification)
        hemisphere_network: str | None = None,
        hemisphere_overlays: dict[str, Path] | None = None,
        target_hemisphere: str = "right",
        # Tissue masks (WM/CSF nuisance regression, view-only)
        tissue_mask_overlays: dict[str, Path] | None = None,
        # Full HTML QC report (optional, for "View Full Report" button)
        html_report_path: Path | None = None,
    ):
        """Initialize the mask selection dialog.

        Args:
            page: Flet Page object
            masks_info: List of MaskReviewInfo for each network
            participant_id: Participant ID for regenerating masks
            masks_dir: Directory containing masks (03_masks/)
            components_file: Path to ICA components file
            yeo_subject_space: Path to Yeo atlas in subject space
            network_atlas: Network atlas key (e.g. "yeo_17", "schaefer_400_7net")
            threshold: Voxel threshold for masks
            on_complete: Callback with final selections when user applies changes
            on_cancel: Callback when dialog is cancelled
            on_log: Callback for logging messages to the log console
            mask_extraction_dir: Root directory for mask extraction (for script generation)
            input_file: Path to preprocessed fMRI data (for script generation)
            brain_mask_file: Path to brain mask (for script generation)
            ref_file: Path to BOLD reference (for script generation)
            networks: Target network names (for script generation)
            n_components: Number of ICA components (for script generation)
            smooth_fwhm: Gaussian smoothing FWHM in mm (for script generation)
            highpass: High-pass filter frequency in Hz (for script generation)
            lowpass: Low-pass filter frequency in Hz (for script generation)
            hemisphere_network: Network name that was split into hemispheres (for sham)
            hemisphere_overlays: Dict mapping 'left'/'right' to overlay paths
            target_hemisphere: Which hemisphere is the target ('right' or 'left')
        """
        self.page = page
        self.masks_info = {m.network_name: m for m in masks_info}
        self.participant_id = participant_id
        self.session_id = session_id
        self._bids_prefix = f"{participant_id}_{session_id}"
        self.masks_dir = masks_dir
        self.components_file = components_file
        self.yeo_subject_space = yeo_subject_space
        from pineuro.atlas_registry import ATLAS_SOURCES

        self.network_atlas = network_atlas
        self.n_networks = ATLAS_SOURCES.get(network_atlas, {}).get("n_networks", 17)
        self.threshold = threshold
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self.on_log = on_log

        # Script generation context
        self.mask_extraction_dir = mask_extraction_dir or masks_dir.parent
        self.input_file = input_file
        self.brain_mask_file = brain_mask_file
        self.ref_file = ref_file
        self.networks = networks or list(self.masks_info.keys())
        self.n_components = n_components
        self.smooth_fwhm = smooth_fwhm
        self.highpass = highpass
        self.lowpass = lowpass

        # Hemisphere split info (for sham verification)
        self.hemisphere_network = hemisphere_network
        self.hemisphere_overlays = hemisphere_overlays or {}
        self.target_hemisphere = target_hemisphere

        # Tissue mask overlays (WM/CSF, view-only)
        self.tissue_mask_overlays = tissue_mask_overlays or {}
        self.html_report_path = html_report_path

        self._dialog: ft.AlertDialog | None = None
        self._stage = 1  # 1 = Accept/Reject, 2 = Select Alternative
        self._current_reject_network: str | None = None
        self._selections: dict[str, MaskSelection] = {}

        # Combine mode state (for merging two ICA components)
        self._combine_mode = False
        self._combine_checkboxes: dict[int, ft.Checkbox] = {}  # comp_idx -> checkbox
        self._combine_preview_container: ft.Container | None = None

        # Load existing selections
        from pineuro.mask_extraction import MaskSelection, load_mask_selections

        existing = load_mask_selections(masks_dir)
        for name, info in self.masks_info.items():
            if name in existing:
                self._selections[name] = existing[name]
            else:
                # Create default selection from current info
                self._selections[name] = MaskSelection(
                    network_name=name,
                    source_type="ica",
                    component_index=info.current_component,
                    correlation=info.current_correlation,
                    selection_reason="auto",
                )

    def _create_image(self, path: Path, height: int = st.PREVIEW_H_SM, width: int | None = None) -> ft.Image:
        """Create an Image widget from a local file path."""
        return ft.Image(
            src=str(path),
            fit=ft.BoxFit.CONTAIN,
            height=height,
            width=width,
        )

    def show(self):
        """Display the dialog."""
        self._stage = 1
        self._show_stage1()

    def _show_stage1(self):
        """Show Stage 1: Accept/Reject dialog."""
        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.BIOTECH, color=st.ACCENT),
                    ft.Text("Review Network Masks", weight=ft.FontWeight.BOLD),
                ],
                spacing=st.GAP_SM,
            ),
            content=self._build_stage1_content(),
            actions=self._build_stage1_actions(),
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(self._dialog)
        self._dialog.open = True
        self.page.update()

    def _build_stage1_content(self) -> ft.Control:
        """Build content for Stage 1: Accept/Reject each mask."""
        mask_cards: list[ft.Control] = []

        for name, info in self.masks_info.items():
            selection = self._selections.get(name)
            is_accepted = info.accepted

            # Determine correlation quality color (use selection's correlation if available)
            corr = selection.correlation if selection else info.current_correlation
            if corr >= self.CORR_GOOD:
                quality_color = st.SUCCESS
                quality_icon = ft.Icons.CHECK_CIRCLE
            elif corr >= self.CORR_MODERATE:
                quality_color = st.WARNING
                quality_icon = ft.Icons.WARNING_AMBER
            else:
                quality_color = st.ERROR
                quality_icon = ft.Icons.ERROR

            # Source label and overlay path based on current selection
            overlay_path = info.overlay_path  # Default: original auto-selected
            if selection and selection.source_type == "yeo_atlas":
                source_label = "Yeo Atlas"
                component_label = ""
                # Use Yeo overlay if available
                if info.yeo_overlay_path and info.yeo_overlay_path.exists():
                    overlay_path = info.yeo_overlay_path
            elif selection and selection.source_type == "ica_combined":
                # Combined ICA components
                source_label = "ICA Combined"
                comp_idx_1 = selection.component_index
                comp_idx_2 = selection.secondary_component_index
                component_label = f"IC {comp_idx_1} + IC {comp_idx_2}"
                # Use combined overlay if available
                if info.combined_overlay_path and info.combined_overlay_path.exists():
                    overlay_path = info.combined_overlay_path
            else:
                source_label = "ICA"
                comp_idx = selection.component_index if selection else info.current_component
                component_label = f"IC {comp_idx}"
                # Always look up overlay from alternative_overlay_paths for current component
                # This handles both initial display and after Reload
                if comp_idx in info.alternative_overlay_paths:
                    alt_path = info.alternative_overlay_paths[comp_idx]
                    if alt_path.exists():
                        overlay_path = alt_path

            # Build overlay image if available
            overlay_content: ft.Control
            if overlay_path and overlay_path.exists():
                overlay_content = self._create_image(overlay_path, height=st.PREVIEW_H_SM)
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=st.NEUTRAL),
                    height=st.PREVIEW_H_SM,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=st.NEUTRAL_BG,
                )

            # Build card for this mask
            card = ft.Container(
                content=ft.Column(
                    [
                        # Header row
                        ft.Row(
                            [
                                ft.Text(name, size=st.SECTION, weight=ft.FontWeight.W_500),
                                ft.Container(expand=True),
                                ft.Icon(quality_icon, color=quality_color, size=st.ICON_MD),
                                ft.Text(
                                    f"{source_label} {component_label} (r = {corr:.2f})",
                                    color=quality_color,
                                    size=st.BODY,
                                ),
                            ],
                        ),
                        # Overlay image
                        ft.Container(
                            content=overlay_content,
                            border=ft.Border.all(1, st.PANEL_BORDER),
                            border_radius=5,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        ),
                        # Action buttons
                        ft.Row(
                            [
                                ft.Container(expand=True),
                                ft.OutlinedButton(
                                    "Reject",
                                    icon=ft.Icons.CLOSE,
                                    on_click=lambda e, n=name: self._on_reject_clicked(n),
                                ),
                                ft.FilledButton(
                                    "Accept",
                                    icon=ft.Icons.CHECK,
                                    on_click=lambda e, n=name: self._on_accept_clicked(n),
                                    bgcolor=st.SUCCESS if is_accepted else None,
                                ),
                            ],
                            spacing=st.GAP_SM,
                        ),
                    ],
                    spacing=st.GAP_SM,
                ),
                padding=st.GAP,
                bgcolor=st.SUCCESS_BG if is_accepted else st.SUBTLE_BG,
                border_radius=8,
                border=ft.Border.all(2, st.SUCCESS_BORDER if is_accepted else st.PANEL_BORDER),
            )
            mask_cards.append(card)

        # Add hemisphere-split masks as info-only cards (for sham verification)
        if self.hemisphere_network and self.hemisphere_overlays:
            hemisphere_cards = self._build_hemisphere_info_cards()
            if hemisphere_cards:
                mask_cards.append(ft.Divider(height=2, color=st.INFO_BORDER))
                mask_cards.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.INFO_OUTLINE, color=st.INFO, size=st.ICON_SM),
                                ft.Text(
                                    "Sham Hemisphere Masks (verification only)",
                                    size=st.CAPTION,
                                    color=st.INFO,
                                    weight=ft.FontWeight.W_500,
                                ),
                            ],
                            spacing=st.GAP_SM,
                        ),
                        padding=ft.Padding.only(left=st.GAP_XS, bottom=st.GAP_XS),
                    )
                )
                mask_cards.extend(hemisphere_cards)

        # Add tissue masks as info-only cards (WM/CSF nuisance regression)
        if self.tissue_mask_overlays:
            tissue_cards = self._build_tissue_info_cards()
            if tissue_cards:
                mask_cards.append(ft.Divider(height=2, color=st.MASK_TISSUE_BORDER))
                mask_cards.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.INFO_OUTLINE, color=st.MASK_TISSUE, size=st.ICON_SM),
                                ft.Text(
                                    "Tissue Masks (nuisance regression, view only)",
                                    size=st.CAPTION,
                                    color=st.MASK_TISSUE_FG,
                                    weight=ft.FontWeight.W_500,
                                ),
                            ],
                            spacing=st.GAP_SM,
                        ),
                        padding=ft.Padding.only(left=st.GAP_XS, bottom=st.GAP_XS),
                    )
                )
                mask_cards.extend(tissue_cards)

        return ft.Container(
            content=ft.Column(
                mask_cards,
                spacing=st.GAP,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=st.DIALOG_MD,
            height=500,  # Increased to accommodate hemisphere cards
            padding=st.GAP_MD,
        )

    def _build_hemisphere_info_cards(self) -> list[ft.Control]:
        """Build info-only cards for hemisphere-split masks (sham verification).

        These cards show the left and right hemisphere masks for visual verification
        but do not have accept/reject buttons since they are derived from the
        parent network mask.
        """
        cards: list[ft.Control] = []

        for hemisphere in ["left", "right"]:
            overlay_path = self.hemisphere_overlays.get(hemisphere)
            if not overlay_path:
                continue

            # Build overlay image
            overlay_content: ft.Control
            if overlay_path and overlay_path.exists():
                overlay_content = self._create_image(overlay_path, height=st.PREVIEW_H_XS)
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=st.NEUTRAL),
                    height=st.PREVIEW_H_XS,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=st.NEUTRAL_BG,
                )

            # Determine which is target vs non-target based on config
            if hemisphere == self.target_hemisphere:
                role_label = "TARGET"
                role_color = st.SUCCESS
            else:
                role_label = "NON-TARGET"
                role_color = st.WARNING

            card = ft.Container(
                content=ft.Column(
                    [
                        # Header row
                        ft.Row(
                            [
                                st.subsection(f"{self.hemisphere_network} {hemisphere.upper()}"),
                                ft.Container(expand=True),
                                ft.Container(
                                    content=ft.Text(role_label, size=st.CAPTION, color=st.ON_NEUTRAL),
                                    bgcolor=role_color,
                                    padding=ft.Padding.symmetric(horizontal=st.GAP_SM, vertical=st.GAP_XS),
                                    border_radius=10,
                                ),
                            ],
                        ),
                        # Overlay image
                        ft.Container(
                            content=overlay_content,
                            border=ft.Border.all(1, st.PANEL_BORDER),
                            border_radius=5,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        ),
                    ],
                    spacing=st.GAP_SM,
                ),
                padding=st.GAP_MD,
                bgcolor=st.INFO_BG,
                border_radius=6,
                border=ft.Border.all(1, st.INFO_BORDER),
            )
            cards.append(card)

        return cards

    def _build_tissue_info_cards(self) -> list[ft.Control]:
        """Build info-only cards for tissue masks (WM/CSF nuisance regression).

        These cards show the white matter and CSF masks for visual verification
        but do not have accept/reject buttons since they are derived from
        MNI template probability maps.
        """
        labels = {"wm": "White Matter", "csf": "CSF"}
        cards: list[ft.Control] = []

        for tissue_type in ["wm", "csf"]:
            overlay_path = self.tissue_mask_overlays.get(tissue_type)
            if not overlay_path:
                continue

            overlay_content: ft.Control
            if overlay_path and overlay_path.exists():
                overlay_content = self._create_image(overlay_path, height=st.PREVIEW_H_XS)
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=st.NEUTRAL),
                    height=st.PREVIEW_H_XS,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=st.NEUTRAL_BG,
                )

            card = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                st.subsection(labels.get(tissue_type, tissue_type.upper())),
                                ft.Container(expand=True),
                                ft.Text(
                                    "MNI template prior",
                                    size=st.CAPTION,
                                    color=st.MASK_TISSUE_LABEL,
                                ),
                            ],
                        ),
                        ft.Container(
                            content=overlay_content,
                            border=ft.Border.all(1, st.PANEL_BORDER),
                            border_radius=5,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        ),
                    ],
                    spacing=st.GAP_SM,
                ),
                padding=st.GAP_MD,
                bgcolor=st.MASK_TISSUE_BG,
                border_radius=6,
                border=ft.Border.all(1, st.MASK_TISSUE_BORDER),
            )
            cards.append(card)

        return cards

    def _build_stage1_actions(self) -> list[ft.Control]:
        """Build action buttons for Stage 1."""
        actions: list[ft.Control] = []

        # View full report button (if HTML path available)
        if self.html_report_path and self.html_report_path.exists():
            actions.append(
                ft.TextButton(
                    "View Full Report",
                    icon=ft.Icons.OPEN_IN_NEW,
                    on_click=self._on_view_full_report,
                )
            )

        actions.extend(
            [
                ft.OutlinedButton(
                    "Generate Script",
                    icon=ft.Icons.CODE,
                    on_click=self._on_generate_script_clicked,
                ),
                ft.OutlinedButton(
                    "Reload",
                    icon=ft.Icons.REFRESH,
                    on_click=self._on_reload_clicked,
                    tooltip="Reload selections and images from disk (after running external script)",
                ),
                ft.OutlinedButton(
                    "Cancel",
                    on_click=self._on_cancel_clicked,
                ),
                ft.FilledButton(
                    "Accept All",
                    icon=ft.Icons.DONE_ALL,
                    on_click=self._on_accept_all_clicked,
                ),
            ]
        )
        return actions

    def _on_view_full_report(self, e):
        """Handle View Full Report button click.

        Regenerates any masks whose selections differ from what's on disk,
        then regenerates the report with overwrite=True so overlays match
        the current selections.
        """
        from pineuro import qc
        from pineuro.mask_extraction import (
            load_mask_selections,
            regenerate_mask_from_selection,
            save_mask_selections,
        )

        if self.on_log:
            self.on_log("[INFO] Regenerating QC report...")

        # Regenerate any masks modified in the dialog but not yet applied to disk
        modified = False
        try:
            existing = load_mask_selections(self.masks_dir)
            for name, selection in self._selections.items():
                old = existing.get(name)
                if old and (
                    selection.source_type != old.source_type
                    or selection.component_index != old.component_index
                    or selection.secondary_component_index != old.secondary_component_index
                ):
                    modified = True
                    logger.info(f"Regenerating {name} mask for report preview")
                    regenerate_mask_from_selection(
                        participant_id=self.participant_id,
                        session_id=self.session_id,
                        masks_dir=self.masks_dir,
                        selection=selection,
                        components_file=self.components_file,
                        yeo_subject_space=self.yeo_subject_space,
                        n_networks=self.n_networks,
                        threshold=self.threshold,
                    )
            # Save selections so the report reads correct component/correlation info
            if modified:
                save_mask_selections(self.masks_dir, self._selections)
        except Exception as ex:
            logger.warning(f"Failed to regenerate modified masks for report: {ex}")

        try:
            report_path = qc.generate_mask_extraction_report(
                mask_extraction_dir=self.mask_extraction_dir,
                participant_id=self.participant_id,
                session_id=self.session_id,
                network_names=self.networks,
                n_networks=self.n_networks,
                overwrite=True,
            )
            self.html_report_path = report_path
        except Exception as ex:
            logger.warning(f"Failed to regenerate QC report: {ex}")

        if self.html_report_path:
            logger.info(f"Opening full report: {self.html_report_path}")
            open_qc_report(self.html_report_path)

    def _generate_alternative_overlays_for_network(self, network_name: str):
        """Generate alternative ICA and Yeo overlays for a network on-demand.

        This is called when the user rejects a mask and enters stage 2,
        avoiding upfront generation of all overlays.
        """
        from pineuro import qc

        info = self.masks_info[network_name]
        qc_dir = self.mask_extraction_dir / "qc"

        # Check if we need to generate alternative overlays
        # We generate for all components that don't already have overlays
        missing_overlays = []
        for comp_idx, _corr in info.alternative_components:
            if comp_idx not in info.alternative_overlay_paths or not info.alternative_overlay_paths[comp_idx].exists():
                missing_overlays.append(comp_idx)

        if missing_overlays:
            if self.on_log:
                self.on_log(f"[INFO] Generating alternative overlays for {network_name}...")

            # Generate overlays for all alternatives (including current best for completeness)
            new_overlays = qc.generate_alternative_overlays(
                mask_extraction_dir=self.mask_extraction_dir,
                participant_id=self.participant_id,
                session_id=self.session_id,
                network_names=[network_name],
                n_alternatives=self.n_components,  # Generate for all components
                output_dir=qc_dir,
                n_networks=self.n_networks,
                ref_file=self.ref_file,
            )

            # Update info with newly generated overlay paths
            if network_name in new_overlays:
                info.alternative_overlay_paths.update(new_overlays[network_name])

            if self.on_log:
                self.on_log(f"[INFO] Generated {len(new_overlays.get(network_name, {}))} alternative overlays")

        # Check if we need to generate Yeo overlay
        if info.yeo_overlay_path is None or not info.yeo_overlay_path.exists():
            if self.on_log:
                self.on_log(f"[INFO] Generating Yeo atlas overlay for {network_name}...")

            yeo_overlay = qc.generate_yeo_network_overlay(
                mask_extraction_dir=self.mask_extraction_dir,
                participant_id=self.participant_id,
                session_id=self.session_id,
                network_name=network_name,
                output_dir=qc_dir,
                n_networks=self.n_networks,
                ref_file=self.ref_file,
            )

            if yeo_overlay:
                info.yeo_overlay_path = yeo_overlay

    def _show_stage2(self, network_name: str, reset_combine_mode: bool = True):
        """Show Stage 2: Select alternative for a rejected mask.

        Parameters
        ----------
        network_name : str
            Name of the network to select alternative for
        reset_combine_mode : bool
            If True, reset combine mode to off. Set to False when refreshing
            the dialog from the combine toggle to preserve the toggle state.
        """
        self._stage = 2
        self._current_reject_network = network_name

        # Reset combine mode for each new mask (don't carry over from previous)
        # But preserve it when refreshing from the toggle
        if reset_combine_mode:
            self._combine_mode = False

        # Close stage 1 dialog
        if self._dialog:
            self._dialog.open = False
            self.page.update()

        # Generate alternative overlays on-demand (only if not already generated)
        self._generate_alternative_overlays_for_network(network_name)

        # Create stage 2 dialog
        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.SWAP_HORIZ, color=st.ACCENT),
                    ft.Text(f"Select Alternative for {network_name}", weight=ft.FontWeight.BOLD),
                ],
                spacing=st.GAP_SM,
            ),
            content=self._build_stage2_content(network_name),
            actions=self._build_stage2_actions(),
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(self._dialog)
        self._dialog.open = True
        self.page.update()

    def _build_stage2_content(self, network_name: str) -> ft.Control:
        """Build content for Stage 2: Select alternative."""
        info = self.masks_info[network_name]
        alternatives = info.alternative_components

        # Reset combine mode state
        self._combine_checkboxes = {}

        # Combine mode toggle
        combine_toggle = ft.Switch(
            label="Combine two components",
            value=self._combine_mode,
            on_change=lambda e: self._on_combine_mode_toggled(e, network_name),
        )

        # Radio group for single-select mode
        self._alternative_selection = ft.RadioGroup(
            content=ft.Column([], spacing=st.GAP_SM),
        )

        alternative_items: list[ft.Control] = []

        # Get the currently selected component to mark it
        current_selection = self._selections.get(network_name)
        _currently_selected_idx = current_selection.component_index if current_selection else None

        # Add ICA alternatives (show all, including currently selected)
        for idx, (comp_idx, corr) in enumerate(alternatives):
            # Build overlay if available - use larger images for better visibility
            overlay: ft.Control
            if comp_idx in info.alternative_overlay_paths:
                overlay_path = info.alternative_overlay_paths[comp_idx]
                if overlay_path.exists():
                    overlay = self._create_image(overlay_path, height=st.OVERLAY_H, width=st.MASK_OVERLAY_W)
                else:
                    overlay = ft.Container(
                        width=st.MASK_OVERLAY_W,
                        height=st.OVERLAY_H,
                        bgcolor=st.NEUTRAL_BG,
                        alignment=ft.Alignment(0, 0),
                        content=ft.Text("No preview", size=st.CAPTION, color=st.NEUTRAL),
                    )
            else:
                overlay = ft.Container(
                    width=st.MASK_OVERLAY_W,
                    height=st.OVERLAY_H,
                    bgcolor=st.NEUTRAL_BG,
                    alignment=ft.Alignment(0, 0),
                    content=ft.Text("No preview", size=st.CAPTION, color=st.NEUTRAL),
                )

            # Correlation quality indicator
            if corr >= self.CORR_GOOD:
                corr_color = st.SUCCESS
            elif corr >= self.CORR_MODERATE:
                corr_color = st.WARNING
            else:
                corr_color = st.ERROR

            # Create both radio and checkbox for this component
            radio = ft.Radio(value=f"ica_{comp_idx}")
            checkbox = ft.Checkbox(
                value=False,
                on_change=lambda e, idx=comp_idx: self._on_combine_checkbox_changed(e, network_name, idx),
            )
            self._combine_checkboxes[comp_idx] = checkbox

            # Container that switches between radio/checkbox based on combine mode
            selector = radio if not self._combine_mode else checkbox

            item = ft.Container(
                content=ft.Row(
                    [
                        selector,
                        ft.Column(
                            [
                                ft.Text(f"IC {comp_idx}", weight=ft.FontWeight.W_500),
                                ft.Text(f"r = {corr:.3f}", size=st.CAPTION, color=corr_color),
                            ],
                            spacing=st.GAP_XS,
                            width=80,
                        ),
                        overlay,
                    ],
                    spacing=st.GAP_SM,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=st.GAP_MD,
                border=ft.Border.all(1, st.PANEL_BORDER),
                border_radius=5,
            )
            alternative_items.append(item)

        # Add Yeo atlas option inside radio group (only in single-select mode)
        yeo_item = self._build_yeo_atlas_option(info) if not self._combine_mode else None

        # Update radio group content (include yeo_item so its Radio is inside the RadioGroup)
        radio_items = alternative_items
        if yeo_item:
            radio_items = alternative_items + [yeo_item]
        self._alternative_selection.content = ft.Column(radio_items, spacing=st.GAP_XS)

        # Combined preview container (shown only in combine mode)
        self._combine_preview_container = ft.Container(
            content=ft.Column(
                [
                    ft.Divider(),
                    ft.Text("Combined Mask Preview:", size=st.CAPTION, weight=ft.FontWeight.W_500),
                    ft.Container(
                        content=ft.Text(
                            "Select exactly 2 components above, then click 'Preview Combined'",
                            color=st.NEUTRAL,
                            size=st.CAPTION,
                        ),
                        height=st.OVERLAY_H,
                        alignment=ft.Alignment(0, 0),
                        bgcolor=st.SUBTLE_BG,
                        border=ft.Border.all(1, st.PANEL_BORDER),
                        border_radius=5,
                    ),
                ],
                spacing=st.GAP_SM,
            ),
            visible=self._combine_mode,
            padding=ft.Padding.only(top=st.GAP_MD),
        )

        # Build the content column
        content_items: list[ft.Control] = [
            ft.Container(
                content=ft.Row(
                    [
                        combine_toggle,
                        ft.Container(expand=True),
                        st.caption("Combine two ICA components when a network is split"),
                    ],
                ),
                padding=ft.Padding.only(bottom=st.GAP_MD),
                border=ft.Border.only(bottom=ft.BorderSide(1, st.PANEL_BORDER)),
            ),
            # Combined preview at the top (visible only in combine mode)
            self._combine_preview_container,
            ft.Text(
                "Select two components to combine:" if self._combine_mode else "Select an alternative mask:",
                size=st.BODY,
                color=st.MUTED,
            ),
            self._alternative_selection,
        ]

        return ft.Container(
            content=ft.Column(
                content_items,
                spacing=st.GAP_SM,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=st.DIALOG_LG,
            height=650,
            padding=st.GAP_MD,
        )

    def _build_yeo_atlas_option(self, info: MaskReviewInfo) -> ft.Control:
        """Build the Yeo atlas fallback option for Stage 2."""
        yeo_overlay: ft.Control
        if info.yeo_overlay_path and info.yeo_overlay_path.exists():
            yeo_overlay = self._create_image(info.yeo_overlay_path, height=st.OVERLAY_H, width=st.MASK_OVERLAY_W)
        else:
            yeo_overlay = ft.Container(
                width=st.MASK_OVERLAY_W,
                height=st.OVERLAY_H,
                bgcolor=st.NEUTRAL_BG,
                alignment=ft.Alignment(0, 0),
                content=ft.Text("No preview", size=st.CAPTION, color=st.NEUTRAL),
            )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Divider(),
                    ft.Text("OR use atlas mask:", size=st.CAPTION, color=st.MUTED),
                    ft.Row(
                        [
                            ft.Radio(value="yeo_atlas"),
                            ft.Column(
                                [
                                    ft.Text("Yeo Atlas", weight=ft.FontWeight.W_500),
                                    ft.Text(
                                        "Less individualized",
                                        size=st.CAPTION,
                                        color=st.WARNING,
                                    ),
                                ],
                                spacing=st.GAP_XS,
                                width=80,
                            ),
                            yeo_overlay,
                        ],
                        spacing=st.GAP_SM,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=st.GAP_XS,
            ),
            padding=st.GAP_MD,
        )

    def _on_combine_mode_toggled(self, e, network_name: str):
        """Handle combine mode toggle."""
        self._combine_mode = e.control.value
        # Clear any checkbox selections when toggling
        for checkbox in self._combine_checkboxes.values():
            checkbox.value = False
        # Refresh the stage 2 dialog (preserve combine mode state)
        if self._dialog:
            self._dialog.open = False
            self.page.update()
        self._show_stage2(network_name, reset_combine_mode=False)

    def _on_combine_checkbox_changed(self, e, network_name: str, comp_idx: int):
        """Handle checkbox change in combine mode."""
        # Count how many are selected
        selected = [idx for idx, cb in self._combine_checkboxes.items() if cb.value]

        # Enforce max 2 selections
        if len(selected) > 2:
            # Uncheck the one that was just checked
            self._combine_checkboxes[comp_idx].value = False
            _snack = ft.SnackBar(ft.Text("Select exactly 2 components to combine."), bgcolor=st.WARNING)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        self.page.update()

    def _get_selected_combine_components(self) -> list[int]:
        """Get list of selected component indices in combine mode."""
        return [idx for idx, cb in self._combine_checkboxes.items() if cb.value]

    def _build_stage2_actions(self) -> list[ft.Control]:
        """Build action buttons for Stage 2."""
        actions: list[ft.Control] = [
            ft.TextButton("Back", on_click=self._on_back_clicked),
        ]

        if self._combine_mode:
            actions.append(
                ft.OutlinedButton(
                    "Preview Combined",
                    icon=ft.Icons.PREVIEW,
                    on_click=self._on_preview_combined_clicked,
                )
            )

        actions.append(
            ft.FilledButton(
                "Select",
                icon=ft.Icons.CHECK,
                on_click=self._on_select_alternative_clicked,
            )
        )

        return actions

    def _on_preview_combined_clicked(self, e):
        """Handle Preview Combined button - generate and show preview of combined mask."""
        if not self._current_reject_network:
            logger.warning("No current reject network set")
            return

        selected = self._get_selected_combine_components()
        if len(selected) != 2:
            _snack = ft.SnackBar(ft.Text("Select exactly 2 components to preview."), bgcolor=st.WARNING)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        # Check required files exist
        if not self.ref_file or not Path(self.ref_file).exists():
            error_msg = f"BOLD reference file not found: {self.ref_file}"
            logger.error(error_msg)
            if self.on_log:
                self.on_log(f"[ERROR] {error_msg}")
            _snack = ft.SnackBar(ft.Text("Cannot preview: BOLD reference file not found"), bgcolor=st.ERROR)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        network_name = self._current_reject_network
        comp_idx_1, comp_idx_2 = selected[0], selected[1]
        info = self.masks_info[network_name]

        # Get individual component correlations for comparison
        corr_1, corr_2 = 0.0, 0.0
        for idx, c in info.alternative_components:
            if idx == comp_idx_1:
                corr_1 = c
            elif idx == comp_idx_2:
                corr_2 = c

        if self.on_log:
            self.on_log(f"[INFO] Generating preview of combined IC {comp_idx_1} + IC {comp_idx_2}...")

        try:
            from pineuro import qc
            from pineuro.mask_extraction import combine_ica_components, compute_mask_network_correlation

            # Generate temporary combined mask
            temp_dir = self.mask_extraction_dir / "qc" / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_mask = (
                temp_dir / f"{self._bids_prefix}_{network_name}_combined_IC{comp_idx_1}_IC{comp_idx_2}_thr.nii.gz"
            )

            logger.info(f"Generating combined mask: {temp_mask}")
            combine_ica_components(
                components_file=self.components_file,
                component_idx_1=comp_idx_1,
                component_idx_2=comp_idx_2,
                output_file=temp_mask,
                threshold=self.threshold,
                overwrite=True,
            )

            # Compute spatial correlation of combined mask with Yeo network
            combined_corr = compute_mask_network_correlation(
                mask_file=temp_mask,
                yeo_subject_space=self.yeo_subject_space,
                network_name=network_name,
                n_networks=self.n_networks,
            )

            # Generate overlay
            logger.info(f"Generating overlay with ref_file: {self.ref_file}")
            overlay_path = qc.generate_mask_overlay(
                mask_path=temp_mask,
                ref_path=self.ref_file,
                output_dir=temp_dir,
                network_name=f"{network_name}_combined",
                atlas_path=self.yeo_subject_space,
                n_networks=self.n_networks,
                overwrite=True,
            )

            logger.info(f"Overlay generated: {overlay_path}")

            # Determine correlation quality color
            if combined_corr >= self.CORR_GOOD:
                corr_color = st.SUCCESS
            elif combined_corr >= self.CORR_MODERATE:
                corr_color = st.WARNING
            else:
                corr_color = st.ERROR

            # Update the preview container
            if overlay_path and overlay_path.exists():
                info.combined_overlay_path = overlay_path

                # Update the preview container content with correlation info
                if self._combine_preview_container:
                    new_content = ft.Column(
                        [
                            ft.Divider(),
                            ft.Row(
                                [
                                    ft.Text("Combined:", size=st.CAPTION, weight=ft.FontWeight.W_500),
                                    ft.Text(
                                        f"IC {comp_idx_1} + IC {comp_idx_2}",
                                        size=st.CAPTION,
                                        color=st.INFO,
                                    ),
                                    ft.Text(
                                        f"r={combined_corr:.3f}",
                                        size=st.CAPTION,
                                        weight=ft.FontWeight.BOLD,
                                        color=corr_color,
                                    ),
                                    ft.Text(
                                        f"(IC{comp_idx_1}={corr_1:.2f}, IC{comp_idx_2}={corr_2:.2f})",
                                        size=st.CAPTION,
                                        color=st.MUTED,
                                    ),
                                ],
                                spacing=st.GAP_SM,
                            ),
                            ft.Container(
                                content=self._create_image(overlay_path, height=st.OVERLAY_H, width=st.MASK_OVERLAY_W),
                                border=ft.Border.all(1, st.INFO_BORDER),
                                border_radius=5,
                            ),
                        ],
                        spacing=st.GAP_XS,
                    )
                    self._combine_preview_container.content = new_content
                    self._combine_preview_container.update()

                if self.on_log:
                    self.on_log(
                        f"[SUCCESS] Combined mask preview: r={combined_corr:.3f} "
                        f"(vs IC{comp_idx_1}={corr_1:.3f}, IC{comp_idx_2}={corr_2:.3f})"
                    )
            else:
                error_msg = f"Overlay not generated or not found: {overlay_path}"
                logger.error(error_msg)
                if self.on_log:
                    self.on_log(f"[ERROR] {error_msg}")

        except Exception as ex:
            logger.error(f"Failed to generate combined preview: {ex}")
            _snack = ft.SnackBar(ft.Text(f"Failed to generate preview: {ex}"), bgcolor=st.ERROR)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

    def _on_accept_clicked(self, network_name: str):
        """Handle accept button click for a mask."""
        from pineuro.mask_extraction import MaskSelection

        info = self.masks_info[network_name]
        info.accepted = True

        # Keep or restore the current selection
        if network_name not in self._selections:
            self._selections[network_name] = MaskSelection(
                network_name=network_name,
                source_type="ica",
                component_index=info.current_component,
                correlation=info.current_correlation,
                selection_reason="auto",
            )

        self._refresh_stage1()

    def _on_reject_clicked(self, network_name: str):
        """Handle reject button click - show alternative selection."""
        info = self.masks_info[network_name]
        info.accepted = False
        self._show_stage2(network_name)

    def _on_accept_all_clicked(self, e):
        """Handle Accept All button - accept all masks and close."""
        for name, info in self.masks_info.items():
            info.accepted = True

        self._apply_and_close()

    def _on_back_clicked(self, e):
        """Handle Back button in Stage 2 - return to Stage 1."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()

        self._stage = 1
        self._current_reject_network = None
        self._show_stage1()

    def _on_select_alternative_clicked(self, e):
        """Handle Select button in Stage 2 - apply alternative selection."""
        from pineuro.mask_extraction import MaskSelection

        network_name = self._current_reject_network
        if not network_name:
            return

        # Handle combine mode
        if self._combine_mode:
            selected = self._get_selected_combine_components()
            if len(selected) != 2:
                _snack = ft.SnackBar(ft.Text("Select exactly 2 components to combine."), bgcolor=st.WARNING)
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                return

            comp_idx_1, comp_idx_2 = selected[0], selected[1]
            info = self.masks_info[network_name]

            # Find correlations for individual components (for reference)
            corr_1, corr_2 = 0.0, 0.0
            for idx, c in info.alternative_components:
                if idx == comp_idx_1:
                    corr_1 = c
                elif idx == comp_idx_2:
                    corr_2 = c

            # Generate the combined mask and compute its actual correlation with the Yeo network
            combined_corr = corr_1  # Fallback to component 1's correlation
            try:
                from pineuro.mask_extraction import combine_ica_components, compute_mask_network_correlation

                # Generate combined mask (or use existing temp one from preview)
                temp_dir = self.mask_extraction_dir / "qc" / "temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_mask = (
                    temp_dir / f"{self._bids_prefix}_{network_name}_combined_IC{comp_idx_1}_IC{comp_idx_2}_thr.nii.gz"
                )

                if not temp_mask.exists():
                    combine_ica_components(
                        components_file=self.components_file,
                        component_idx_1=comp_idx_1,
                        component_idx_2=comp_idx_2,
                        output_file=temp_mask,
                        threshold=self.threshold,
                        overwrite=True,
                    )

                # Compute actual spatial correlation of combined mask with Yeo network
                combined_corr = compute_mask_network_correlation(
                    mask_file=temp_mask,
                    yeo_subject_space=self.yeo_subject_space,
                    network_name=network_name,
                    n_networks=self.n_networks,
                )

                if self.on_log:
                    self.on_log(
                        f"[INFO] Combined mask correlation: r={combined_corr:.3f} "
                        f"(vs IC{comp_idx_1}={corr_1:.3f}, IC{comp_idx_2}={corr_2:.3f})"
                    )

            except Exception as ex:
                logger.warning(f"Failed to compute combined mask correlation: {ex}")
                if self.on_log:
                    self.on_log(f"[WARNING] Could not compute combined correlation: {ex}")

            self._selections[network_name] = MaskSelection(
                network_name=network_name,
                source_type="ica_combined",
                component_index=comp_idx_1,
                correlation=combined_corr,  # Use actual combined mask correlation
                selection_reason="manual_combined",
                secondary_component_index=comp_idx_2,
                secondary_correlation=corr_2,
            )

            # Update info for display consistency
            info.current_correlation = combined_corr

            if self.on_log:
                self.on_log(
                    f"[INFO] Selected combined IC {comp_idx_1} + IC {comp_idx_2} for {network_name} (r={combined_corr:.3f})"
                )

        else:
            # Single-select mode (original logic)
            if not self._alternative_selection.value:
                _snack = ft.SnackBar(ft.Text("Please select an alternative."), bgcolor=st.WARNING)
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                return

            selection_value = self._alternative_selection.value

            if selection_value == "yeo_atlas":
                # User chose Yeo atlas fallback
                self._selections[network_name] = MaskSelection(
                    network_name=network_name,
                    source_type="yeo_atlas",
                    component_index=None,
                    correlation=0.0,
                    selection_reason="yeo_fallback",
                )
            elif selection_value.startswith("ica_"):
                # User chose a different ICA component
                comp_idx = int(selection_value.split("_")[1])
                info = self.masks_info[network_name]

                # Find correlation for this component
                corr = 0.0
                for idx, c in info.alternative_components:
                    if idx == comp_idx:
                        corr = c
                        break

                self._selections[network_name] = MaskSelection(
                    network_name=network_name,
                    source_type="ica",
                    component_index=comp_idx,
                    correlation=corr,
                    selection_reason="manual",
                )

        # Mark as accepted with new selection
        self.masks_info[network_name].accepted = True

        # If this is the hemisphere network, regenerate hemisphere masks immediately
        # so the user can verify them before final approval
        if self.hemisphere_network and network_name == self.hemisphere_network:
            # First close the current Stage 2 dialog
            if self._dialog:
                self._dialog.open = False
                self.page.update()

            # Now regenerate the masks
            self._regenerate_hemisphere_masks_preview()

            # Set state for Stage 1
            self._stage = 1
            self._current_reject_network = None

            # Show Stage 1 with updated overlays
            self._show_stage1()
        else:
            # Regular path - just go back
            self._on_back_clicked(None)

    def _regenerate_hemisphere_masks_preview(self):
        """Regenerate hemisphere masks for preview when hemisphere network selection changes.

        This is called immediately after the user selects a different ICA component
        for the hemisphere network, allowing them to verify the hemisphere split
        before final approval.
        """
        from nibabel.loadsave import load, save
        from nibabel.nifti1 import Nifti1Image

        from pineuro import qc
        from pineuro.mask_extraction import split_mask_by_hemisphere

        if self.hemisphere_network is None:
            return
        selection = self._selections.get(self.hemisphere_network)
        if not selection:
            return

        if self.on_log:
            self.on_log("[INFO] Regenerating hemisphere masks for preview...")

        try:
            # First, regenerate the unthresholded mask from the selected component
            network_file = self.masks_dir / f"{self._bids_prefix}_{self.hemisphere_network}.nii.gz"

            if selection.source_type == "ica" and selection.component_index is not None:
                # Extract the selected ICA component
                components_img = load(self.components_file)
                components_data = components_img.get_fdata()  # type: ignore[attr-defined]
                ic_data = components_data[..., selection.component_index]
                ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)  # type: ignore[attr-defined]
                save(ic_img, network_file)

            elif selection.source_type == "yeo_atlas":
                # Use Yeo atlas - extract the network region
                from pineuro.utils import get_yeo_labels

                labels = get_yeo_labels(n_networks=self.n_networks)
                network_index = labels.index(self.hemisphere_network) + 1
                atlas_img = load(self.yeo_subject_space)
                atlas_data = atlas_img.get_fdata()  # type: ignore[attr-defined]
                network_mask = (atlas_data == network_index).astype(float)
                save(Nifti1Image(network_mask, atlas_img.affine, atlas_img.header), network_file)  # type: ignore[attr-defined]

            # Now split into hemispheres
            left_file = self.masks_dir / f"{self._bids_prefix}_{self.hemisphere_network}_left_thr.nii.gz"
            right_file = self.masks_dir / f"{self._bids_prefix}_{self.hemisphere_network}_right_thr.nii.gz"

            split_mask_by_hemisphere(network_file, left_file, right_file, threshold=self.threshold, overwrite=True)

            # Regenerate hemisphere overlays
            qc_dir = self.mask_extraction_dir.parent / "qc"
            new_overlays = qc.generate_hemisphere_overlays(
                mask_extraction_dir=self.mask_extraction_dir,
                participant_id=self.participant_id,
                session_id=self.session_id,
                hemisphere_network=self.hemisphere_network,
                output_dir=qc_dir,
                n_networks=self.n_networks,
                overwrite=True,
            )

            # Copy overlays to component-specific paths to avoid Flet image caching
            # (Flet caches images by path, so we need unique paths for each component)
            import shutil

            comp_suffix = f"_IC{selection.component_index}" if selection.component_index is not None else "_yeo"
            updated_overlays = {}
            for hemi, src_path in new_overlays.items():
                if src_path and src_path.exists():
                    # Create unique filename with component index
                    new_name = src_path.stem + comp_suffix + src_path.suffix
                    dest_path = src_path.parent / new_name
                    shutil.copy2(src_path, dest_path)
                    updated_overlays[hemi] = dest_path
                else:
                    updated_overlays[hemi] = src_path

            # Update stored overlays with the unique paths
            self.hemisphere_overlays.update(updated_overlays)

            if self.on_log:
                self.on_log("[SUCCESS] Hemisphere masks regenerated for preview")

        except Exception as ex:
            logger.error(f"Failed to regenerate hemisphere masks for preview: {ex}")
            if self.on_log:
                self.on_log(f"[WARNING] Failed to regenerate hemisphere masks: {ex}")

    def _on_cancel_clicked(self, e):
        """Handle Cancel button."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
        if self.on_cancel:
            self.on_cancel()

    def _on_reload_clicked(self, e):
        """Handle Reload button - reload selections, correlations, and images from disk."""
        try:
            import pandas as pd  # type: ignore[import-untyped]

            from pineuro.mask_extraction import get_alternative_components, load_mask_selections

            # Check if approval marker was removed (e.g., by running reproduce_masks.py)
            approval_marker = self.mask_extraction_dir / ".masks_approved"
            approval_removed = not approval_marker.exists()

            # Re-read mask_selections.json
            existing = load_mask_selections(self.masks_dir)

            # Re-read correlation matrix if it exists (updated by script)
            corr_file = self.masks_dir / "network_IC_spatial_correlation.csv"
            corr_df = None
            if corr_file.exists():
                corr_df = pd.read_csv(corr_file, index_col=0)

            # Re-discover overlay paths
            qc_dir = self.mask_extraction_dir / "qc"
            overlays_dir = qc_dir / "overlays" / "alternatives"

            # Update internal selections and masks_info with what's on disk
            for name, info in self.masks_info.items():
                if name in existing:
                    sel = existing[name]
                    self._selections[name] = sel

                    # Update mask info from the reloaded selection
                    info.current_correlation = sel.correlation
                    if sel.source_type == "ica" and sel.component_index is not None:
                        info.current_component = sel.component_index

                        # Update overlay path for the current selection
                        overlay_path = overlays_dir / f"{self._bids_prefix}_{name}_IC{sel.component_index}_overlay.png"
                        if overlay_path.exists():
                            info.overlay_path = overlay_path

                # Reset accepted state - masks need re-approval after regeneration
                # (auto selections are accepted by default, manual need review)
                info.accepted = True

                # Re-discover alternative overlay paths
                new_alt_paths = {}
                for ic_idx in range(self.n_components):
                    alt_path = overlays_dir / f"{self._bids_prefix}_{name}_IC{ic_idx}_overlay.png"
                    if alt_path.exists():
                        new_alt_paths[ic_idx] = alt_path
                info.alternative_overlay_paths = new_alt_paths

                # Re-discover Yeo overlay path
                yeo_path = overlays_dir / f"{self._bids_prefix}_{name}_yeo_overlay.png"
                if yeo_path.exists():
                    info.yeo_overlay_path = yeo_path

                # Update alternative_components from correlation matrix
                if corr_df is not None and name in corr_df.columns:
                    alternatives = get_alternative_components(corr_df, name, top_n=self.n_components)
                    info.alternative_components = alternatives

            # Log the reload
            if self.on_log:
                self.on_log("[INFO] Reloaded mask selections, correlations, and images from disk")
                if approval_removed:
                    self.on_log("[INFO] Masks need re-approval (approval marker was removed)")

            # Refresh the dialog
            self._refresh_stage1()

            # Show confirmation with appropriate message
            if approval_removed:
                snack_message = "Reloaded - masks need re-approval"
                snack_color = st.WARNING
            else:
                snack_message = "Reloaded selections and images from disk"
                snack_color = st.INFO

            _snack = ft.SnackBar(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.REFRESH, color=st.ON_NEUTRAL),
                        ft.Text(snack_message),
                    ],
                    spacing=st.GAP_SM,
                ),
                bgcolor=snack_color,
                duration=3000,
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        except Exception as ex:
            logger.error(f"Failed to reload from disk: {ex}")
            _snack = ft.SnackBar(ft.Text(f"Failed to reload: {ex}"), bgcolor=st.ERROR)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

    def _on_generate_script_clicked(self, e):
        """Handle Generate Script button."""
        from pineuro.script_generator import generate_mask_extraction_script, save_script

        # Check if we have enough context for script generation
        if not self.input_file or not self.brain_mask_file or not self.ref_file:
            _snack = ft.SnackBar(ft.Text("Cannot generate script: missing input file paths."), bgcolor=st.WARNING)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()
            return

        try:
            # Rebuild the resolved rest/ICA params and replay them through the
            # shared dispatcher (same path the live pipeline uses).
            params = {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "root_dir": self.mask_extraction_dir,
                "input_file": self.input_file,
                "mask_file": self.brain_mask_file,
                "ref_file": self.ref_file,
                "networks": self.networks,
                "network_atlas": self.network_atlas,
                "n_components": self.n_components,
                "threshold": self.threshold,
                "smooth_fwhm": self.smooth_fwhm,
                "highpass": self.highpass,
                "lowpass": self.lowpass,
                "overwrite": True,
                "qc_enabled": True,
            }
            script_content = generate_mask_extraction_script("rest", params, selections=self._selections)

            # Save to file
            script_path = self.mask_extraction_dir / "reproduce_masks.py"
            save_script(script_content, script_path)

            logger.info(f"Generated script: {script_path}")

            # Log to console
            if self.on_log:
                self.on_log(f"[SUCCESS] Script generated: {script_path}")

            # Show success message
            _snack = ft.SnackBar(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=st.ON_NEUTRAL),
                        ft.Text(f"Script saved to: {script_path.name}"),
                    ],
                    spacing=st.GAP_SM,
                ),
                bgcolor=st.SUCCESS,
                duration=4000,
            )
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

        except Exception as ex:
            logger.error(f"Failed to generate script: {ex}")
            _snack = ft.SnackBar(ft.Text(f"Failed to generate script: {ex}"), bgcolor=st.ERROR)
            _snack.open = True
            self.page.overlay.append(_snack)
            self.page.update()

    def _refresh_stage1(self):
        """Refresh Stage 1 dialog to reflect changes."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
        self._show_stage1()

    def _apply_and_close(self):
        """Save selections, regenerate modified masks, and close."""
        from pineuro.mask_extraction import (
            get_yeo_labels,
            load_mask_selections,
            regenerate_mask_from_selection,
            save_mask_selections,
        )

        # Check which masks need regeneration
        existing = load_mask_selections(self.masks_dir)
        modified_networks = []

        for name, selection in self._selections.items():
            if name in existing:
                old = existing[name]
                if (
                    selection.source_type != old.source_type
                    or selection.component_index != old.component_index
                    # Also check secondary_component_index for combined selections
                    or selection.secondary_component_index != old.secondary_component_index
                ):
                    modified_networks.append(name)
            else:
                # New selection
                if selection.selection_reason != "auto":
                    modified_networks.append(name)

        # Regenerate modified masks
        for name in modified_networks:
            selection = self._selections[name]
            logger.info(f"Regenerating mask for {name}: {selection}")
            try:
                regenerate_mask_from_selection(
                    participant_id=self.participant_id,
                    masks_dir=self.masks_dir,
                    selection=selection,
                    components_file=self.components_file,
                    yeo_subject_space=self.yeo_subject_space,
                    n_networks=self.n_networks,
                    threshold=self.threshold,
                )
            except Exception as ex:
                logger.error(f"Failed to regenerate mask for {name}: {ex}")
                _snack = ft.SnackBar(ft.Text(f"Failed to regenerate {name}: {ex}"), bgcolor=st.ERROR)
                _snack.open = True
                self.page.overlay.append(_snack)
                self.page.update()
                return

        # Regenerate subject space overlay PNGs for modified masks
        if modified_networks and self.ref_file and self.ref_file.exists():
            try:
                from pineuro.qc import plot_mask_overlay_tripanel, select_slices_from_atlas

                qc_dir = self.mask_extraction_dir.parent / "qc"
                subject_overlay_dir = qc_dir / "overlays" / "subject"
                subject_overlay_dir.mkdir(parents=True, exist_ok=True)

                # Load Yeo atlas for slice selection
                yeo2sub_files = (
                    sorted((self.mask_extraction_dir / "02_mnireg").glob("*yeo2sub.nii.gz"))
                    if (self.mask_extraction_dir / "02_mnireg").exists()
                    else []
                )
                yeo2sub = yeo2sub_files[0] if yeo2sub_files else None
                labels = get_yeo_labels(self.n_networks)

                for name in modified_networks:
                    thr_file = self.masks_dir / f"{self._bids_prefix}_{name}_thr.nii.gz"
                    overlay_png = subject_overlay_dir / f"{self._bids_prefix}_{name}_overlay.png"
                    if thr_file.exists():
                        logger.info(f"Regenerating subject space overlay for {name}")
                        atlas_slices = select_slices_from_atlas(
                            atlas_path=yeo2sub if yeo2sub else self.ref_file,
                            labels=labels,
                            target_label=name,
                            top_n_per_plane=1,
                            mask_path=thr_file,
                        )
                        from nibabel import load as nib_load

                        mask_img = nib_load(thr_file)
                        voxels = int(mask_img.get_fdata().sum())
                        plot_mask_overlay_tripanel(
                            mask_path=thr_file,
                            ref_path=self.ref_file,
                            out_path=overlay_png,
                            slices=atlas_slices,
                            title=f"{name} ({voxels} voxels)",
                        )
            except Exception as ex:
                logger.warning(f"Failed to regenerate subject space overlays: {ex}")

        # Regenerate hemisphere-split masks if hemisphere_network was modified
        if self.hemisphere_network and self.hemisphere_network in modified_networks:
            logger.info(f"Regenerating hemisphere-split masks for {self.hemisphere_network}")
            try:
                from pineuro import qc
                from pineuro.mask_extraction import split_mask_by_hemisphere

                # The unthresholded mask should have been regenerated above
                network_file = self.masks_dir / f"{self._bids_prefix}_{self.hemisphere_network}.nii.gz"
                left_file = self.masks_dir / f"{self._bids_prefix}_{self.hemisphere_network}_left_thr.nii.gz"
                right_file = self.masks_dir / f"{self._bids_prefix}_{self.hemisphere_network}_right_thr.nii.gz"

                if network_file.exists():
                    split_mask_by_hemisphere(
                        network_file, left_file, right_file, threshold=self.threshold, overwrite=True
                    )
                    logger.info(f"Regenerated hemisphere masks: {left_file.name}, {right_file.name}")

                    # Regenerate hemisphere overlays
                    if self.on_log:
                        self.on_log(f"[INFO] Regenerating hemisphere overlays for {self.hemisphere_network}...")

                    qc_dir = self.mask_extraction_dir.parent / "qc"
                    new_overlays = qc.generate_hemisphere_overlays(
                        mask_extraction_dir=self.mask_extraction_dir,
                        participant_id=self.participant_id,
                        session_id=self.session_id,
                        hemisphere_network=self.hemisphere_network,
                        output_dir=qc_dir,
                        n_networks=self.n_networks,
                        overwrite=True,
                    )
                    # Update stored overlays for display
                    self.hemisphere_overlays.update(new_overlays)
                else:
                    logger.warning(f"Cannot regenerate hemisphere masks: {network_file} not found")

            except Exception as ex:
                logger.error(f"Failed to regenerate hemisphere masks: {ex}")
                if self.on_log:
                    self.on_log(f"[WARNING] Failed to regenerate hemisphere masks: {ex}")

        # Save selections
        save_mask_selections(self.masks_dir, self._selections)

        self._close_dialog()

        if self.on_complete:
            self.on_complete(self._selections)

    def _close_dialog(self):
        """Close the dialog."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
