"""Mask selection dialog for reviewing and customizing extracted network masks.

Provides a two-stage dialog flow:
1. Accept/Reject each mask based on the ICA-derived result
2. Select alternative (different ICA component or Yeo atlas fallback) for rejected masks
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import flet as ft

from pineuro.mask_extraction import (
    MaskSelection,
    get_alternative_components,
    load_mask_selections,
    save_mask_selections,
    regenerate_mask_from_selection,
)

logger = logging.getLogger(__name__)


@dataclass
class MaskReviewInfo:
    """Information about a network mask for the review dialog."""

    network_name: str
    current_component: int
    current_correlation: float
    alternative_components: List[Tuple[int, float]]  # [(idx, corr), ...] ranked
    overlay_path: Optional[Path]  # Path to current mask overlay image
    alternative_overlay_paths: Dict[int, Path]  # component_idx -> overlay path
    yeo_overlay_path: Optional[Path]  # Path to Yeo atlas overlay
    accepted: bool = True  # Default to accepted
    combined_overlay_path: Optional[Path] = None  # Path to combined mask preview (when two components selected)


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
        masks_info: List[MaskReviewInfo],
        participant_id: str,
        masks_dir: Path,
        components_file: Path,
        yeo_subject_space: Path,
        n_yeo: int = 17,
        threshold: int = 2000,
        on_complete: Optional[Callable[[Dict[str, MaskSelection]], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        # Additional context for script generation
        mask_extraction_dir: Optional[Path] = None,
        input_file: Optional[Path] = None,
        brain_mask_file: Optional[Path] = None,
        ref_file: Optional[Path] = None,
        yeo_targets: Optional[List[str]] = None,
        n_components: int = 30,
        smooth_fwhm: Optional[float] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        # Hemisphere split (sham verification)
        hemisphere_network: Optional[str] = None,
        hemisphere_overlays: Optional[Dict[str, Path]] = None,
        target_hemisphere: str = "right",
    ):
        """Initialize the mask selection dialog.

        Args:
            page: Flet Page object
            masks_info: List of MaskReviewInfo for each network
            participant_id: Participant ID for regenerating masks
            masks_dir: Directory containing masks (03_masks/)
            components_file: Path to ICA components file
            yeo_subject_space: Path to Yeo atlas in subject space
            n_yeo: Yeo parcellation (7 or 17)
            threshold: Voxel threshold for masks
            on_complete: Callback with final selections when user applies changes
            on_cancel: Callback when dialog is cancelled
            on_log: Callback for logging messages to the log console
            mask_extraction_dir: Root directory for mask extraction (for script generation)
            input_file: Path to preprocessed fMRI data (for script generation)
            brain_mask_file: Path to brain mask (for script generation)
            ref_file: Path to BOLD reference (for script generation)
            yeo_targets: Target network names (for script generation)
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
        self.masks_dir = masks_dir
        self.components_file = components_file
        self.yeo_subject_space = yeo_subject_space
        self.n_yeo = n_yeo
        self.threshold = threshold
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self.on_log = on_log

        # Script generation context
        self.mask_extraction_dir = mask_extraction_dir or masks_dir.parent
        self.input_file = input_file
        self.brain_mask_file = brain_mask_file
        self.ref_file = ref_file
        self.yeo_targets = yeo_targets or list(self.masks_info.keys())
        self.n_components = n_components
        self.smooth_fwhm = smooth_fwhm
        self.highpass = highpass
        self.lowpass = lowpass

        # Hemisphere split info (for sham verification)
        self.hemisphere_network = hemisphere_network
        self.hemisphere_overlays = hemisphere_overlays or {}
        self.target_hemisphere = target_hemisphere

        self._dialog: Optional[ft.AlertDialog] = None
        self._stage = 1  # 1 = Accept/Reject, 2 = Select Alternative
        self._current_reject_network: Optional[str] = None
        self._selections: Dict[str, MaskSelection] = {}

        # Combine mode state (for merging two ICA components)
        self._combine_mode = False
        self._combine_checkboxes: Dict[int, ft.Checkbox] = {}  # comp_idx -> checkbox
        self._combine_preview_container: Optional[ft.Container] = None

        # Load existing selections
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

    def _create_image(self, path: Path, height: int = 150, width: Optional[int] = None) -> ft.Image:
        """Create an Image widget from a local file path."""
        return ft.Image(
            src=str(path),
            fit="contain",
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
                    ft.Icon(ft.Icons.BIOTECH, color=ft.Colors.BLUE),
                    ft.Text("Review Network Masks", weight=ft.FontWeight.BOLD),
                ],
                spacing=10,
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
        mask_cards = []

        for name, info in self.masks_info.items():
            selection = self._selections.get(name)
            is_accepted = info.accepted

            # Determine correlation quality color (use selection's correlation if available)
            corr = selection.correlation if selection else info.current_correlation
            if corr >= self.CORR_GOOD:
                quality_color = ft.Colors.GREEN
                quality_icon = ft.Icons.CHECK_CIRCLE
            elif corr >= self.CORR_MODERATE:
                quality_color = ft.Colors.ORANGE
                quality_icon = ft.Icons.WARNING_AMBER
            else:
                quality_color = ft.Colors.RED
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
                overlay_content = self._create_image(overlay_path, height=150)
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=ft.Colors.GREY_500),
                    height=150,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=ft.Colors.GREY_200,
                )

            # Build card for this mask
            card = ft.Container(
                content=ft.Column(
                    [
                        # Header row
                        ft.Row(
                            [
                                ft.Text(name, size=16, weight=ft.FontWeight.W_500),
                                ft.Container(expand=True),
                                ft.Icon(quality_icon, color=quality_color, size=20),
                                ft.Text(
                                    f"{source_label} {component_label} (r = {corr:.2f})",
                                    color=quality_color,
                                    size=13,
                                ),
                            ],
                        ),
                        # Overlay image
                        ft.Container(
                            content=overlay_content,
                            border=ft.border.all(1, ft.Colors.GREY_300),
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
                                    bgcolor=ft.Colors.GREEN if is_accepted else None,
                                ),
                            ],
                            spacing=10,
                        ),
                    ],
                    spacing=10,
                ),
                padding=15,
                bgcolor=ft.Colors.GREEN_50 if is_accepted else ft.Colors.GREY_100,
                border_radius=8,
                border=ft.border.all(
                    2, ft.Colors.GREEN_400 if is_accepted else ft.Colors.GREY_300
                ),
            )
            mask_cards.append(card)

        # Add hemisphere-split masks as info-only cards (for sham verification)
        if self.hemisphere_network and self.hemisphere_overlays:
            hemisphere_cards = self._build_hemisphere_info_cards()
            if hemisphere_cards:
                mask_cards.append(ft.Divider(height=2, color=ft.Colors.BLUE_200))
                mask_cards.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.BLUE, size=16),
                                ft.Text(
                                    "Sham Hemisphere Masks (verification only)",
                                    size=12,
                                    color=ft.Colors.BLUE_700,
                                    weight=ft.FontWeight.W_500,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=ft.padding.only(left=5, bottom=5),
                    )
                )
                mask_cards.extend(hemisphere_cards)

        return ft.Container(
            content=ft.Column(
                mask_cards,
                spacing=15,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=500,
            height=500,  # Increased to accommodate hemisphere cards
            padding=10,
        )

    def _build_hemisphere_info_cards(self) -> List[ft.Control]:
        """Build info-only cards for hemisphere-split masks (sham verification).

        These cards show the left and right hemisphere masks for visual verification
        but do not have accept/reject buttons since they are derived from the
        parent network mask.
        """
        cards = []

        for hemisphere in ["left", "right"]:
            overlay_path = self.hemisphere_overlays.get(hemisphere)
            if not overlay_path:
                continue

            # Build overlay image
            overlay_content: ft.Control
            if overlay_path and overlay_path.exists():
                overlay_content = self._create_image(overlay_path, height=120)
            else:
                overlay_content = ft.Container(
                    content=ft.Text("No preview available", color=ft.Colors.GREY_500),
                    height=120,
                    alignment=ft.Alignment(0, 0),
                    bgcolor=ft.Colors.GREY_200,
                )

            # Determine which is target vs non-target based on config
            if hemisphere == self.target_hemisphere:
                role_label = "TARGET"
                role_color = ft.Colors.GREEN
            else:
                role_label = "NON-TARGET"
                role_color = ft.Colors.ORANGE

            card = ft.Container(
                content=ft.Column(
                    [
                        # Header row
                        ft.Row(
                            [
                                ft.Text(
                                    f"{self.hemisphere_network} {hemisphere.upper()}",
                                    size=14,
                                    weight=ft.FontWeight.W_500,
                                ),
                                ft.Container(expand=True),
                                ft.Container(
                                    content=ft.Text(role_label, size=11, color=ft.Colors.WHITE),
                                    bgcolor=role_color,
                                    padding=ft.padding.symmetric(horizontal=8, vertical=2),
                                    border_radius=10,
                                ),
                            ],
                        ),
                        # Overlay image
                        ft.Container(
                            content=overlay_content,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=5,
                            clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        ),
                    ],
                    spacing=8,
                ),
                padding=12,
                bgcolor=ft.Colors.BLUE_50,
                border_radius=6,
                border=ft.border.all(1, ft.Colors.BLUE_200),
            )
            cards.append(card)

        return cards

    def _build_stage1_actions(self) -> List[ft.Control]:
        """Build action buttons for Stage 1."""
        return [
            ft.OutlinedButton(
                "Open in FSLeyes",
                icon=ft.Icons.OPEN_IN_NEW,
                on_click=self._on_open_fsleyes_clicked,
                tooltip="Open thresholded masks and BOLD reference in FSLeyes",
            ),
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
            if comp_idx not in info.alternative_overlay_paths:
                missing_overlays.append(comp_idx)
            elif not info.alternative_overlay_paths[comp_idx].exists():
                missing_overlays.append(comp_idx)

        if missing_overlays:
            if self.on_log:
                self.on_log(f"[INFO] Generating alternative overlays for {network_name}...")

            # Generate overlays for all alternatives (including current best for completeness)
            new_overlays = qc.generate_alternative_overlays(
                mask_extraction_dir=self.mask_extraction_dir,
                participant_id=self.participant_id,
                network_names=[network_name],
                n_alternatives=self.n_components,  # Generate for all components
                output_dir=qc_dir,
                n_yeo=self.n_yeo,
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
                network_name=network_name,
                output_dir=qc_dir,
                n_yeo=self.n_yeo,
            )

            if yeo_overlay:
                info.yeo_overlay_path = yeo_overlay

    def _show_stage2(self, network_name: str):
        """Show Stage 2: Select alternative for a rejected mask."""
        self._stage = 2
        self._current_reject_network = network_name

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
                    ft.Icon(ft.Icons.SWAP_HORIZ, color=ft.Colors.BLUE),
                    ft.Text(f"Select Alternative for {network_name}", weight=ft.FontWeight.BOLD),
                ],
                spacing=10,
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
            content=ft.Column([], spacing=10),
        )

        alternative_items = []

        # Get the currently selected component to mark it
        current_selection = self._selections.get(network_name)
        currently_selected_idx = current_selection.component_index if current_selection else None

        # Add ICA alternatives (show all, including currently selected)
        for idx, (comp_idx, corr) in enumerate(alternatives):
            # Build overlay if available - use larger images for better visibility
            overlay: ft.Control
            if comp_idx in info.alternative_overlay_paths:
                overlay_path = info.alternative_overlay_paths[comp_idx]
                if overlay_path.exists():
                    overlay = self._create_image(overlay_path, height=180, width=500)
                else:
                    overlay = ft.Container(
                        width=500,
                        height=180,
                        bgcolor=ft.Colors.GREY_200,
                        alignment=ft.Alignment(0, 0),
                        content=ft.Text("No preview", size=12, color=ft.Colors.GREY_500),
                    )
            else:
                overlay = ft.Container(
                    width=500,
                    height=180,
                    bgcolor=ft.Colors.GREY_200,
                    alignment=ft.Alignment(0, 0),
                    content=ft.Text("No preview", size=12, color=ft.Colors.GREY_500),
                )

            # Correlation quality indicator
            if corr >= self.CORR_GOOD:
                corr_color = ft.Colors.GREEN
            elif corr >= self.CORR_MODERATE:
                corr_color = ft.Colors.ORANGE
            else:
                corr_color = ft.Colors.RED

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
                                ft.Text(f"r = {corr:.3f}", size=12, color=corr_color),
                            ],
                            spacing=2,
                            width=80,
                        ),
                        overlay,
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=10,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=5,
            )
            alternative_items.append(item)

        # Add Yeo atlas option (only in single-select mode)
        yeo_item = self._build_yeo_atlas_option(info) if not self._combine_mode else None

        # Update radio group content
        self._alternative_selection.content = ft.Column(alternative_items, spacing=5)

        # Combined preview container (shown only in combine mode)
        self._combine_preview_container = ft.Container(
            content=ft.Column(
                [
                    ft.Divider(),
                    ft.Text("Combined Mask Preview:", size=12, weight=ft.FontWeight.W_500),
                    ft.Container(
                        content=ft.Text(
                            "Select exactly 2 components above, then click 'Preview Combined'",
                            color=ft.Colors.GREY_500,
                            size=12,
                        ),
                        height=180,
                        alignment=ft.Alignment(0, 0),
                        bgcolor=ft.Colors.GREY_100,
                        border=ft.border.all(1, ft.Colors.GREY_300),
                        border_radius=5,
                    ),
                ],
                spacing=8,
            ),
            visible=self._combine_mode,
            padding=ft.padding.only(top=10),
        )

        # Build the content column
        content_items = [
            ft.Container(
                content=ft.Row(
                    [
                        combine_toggle,
                        ft.Container(expand=True),
                        ft.Text(
                            "Combine two ICA components when a network is split",
                            size=11,
                            color=ft.Colors.GREY_600,
                            italic=True,
                        ),
                    ],
                ),
                padding=ft.padding.only(bottom=10),
                border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.GREY_300)),
            ),
            # Combined preview at the top (visible only in combine mode)
            self._combine_preview_container,
            ft.Text(
                "Select two components to combine:" if self._combine_mode else "Select an alternative mask:",
                size=13,
                color=ft.Colors.GREY_700,
            ),
            self._alternative_selection,
        ]

        if yeo_item and not self._combine_mode:
            content_items.append(yeo_item)

        return ft.Container(
            content=ft.Column(
                content_items,
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
            ),
            width=700,
            height=650,
            padding=10,
        )

    def _build_yeo_atlas_option(self, info: MaskReviewInfo) -> ft.Control:
        """Build the Yeo atlas fallback option for Stage 2."""
        yeo_overlay: ft.Control
        if info.yeo_overlay_path and info.yeo_overlay_path.exists():
            yeo_overlay = self._create_image(info.yeo_overlay_path, height=180, width=500)
        else:
            yeo_overlay = ft.Container(
                width=500,
                height=180,
                bgcolor=ft.Colors.GREY_200,
                alignment=ft.Alignment(0, 0),
                content=ft.Text("No preview", size=12, color=ft.Colors.GREY_500),
            )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Divider(),
                    ft.Text("OR use atlas mask:", size=12, color=ft.Colors.GREY_600),
                    ft.Row(
                        [
                            ft.Radio(value="yeo_atlas"),
                            ft.Column(
                                [
                                    ft.Text("Yeo Atlas", weight=ft.FontWeight.W_500),
                                    ft.Text(
                                        "Less individualized",
                                        size=11,
                                        color=ft.Colors.ORANGE,
                                    ),
                                ],
                                spacing=2,
                                width=80,
                            ),
                            yeo_overlay,
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=5,
            ),
            padding=10,
        )

    def _on_combine_mode_toggled(self, e, network_name: str):
        """Handle combine mode toggle."""
        self._combine_mode = e.control.value
        # Clear any checkbox selections when toggling
        for checkbox in self._combine_checkboxes.values():
            checkbox.value = False
        # Refresh the stage 2 dialog
        if self._dialog:
            self._dialog.open = False
            self.page.update()
        self._show_stage2(network_name)

    def _on_combine_checkbox_changed(self, e, network_name: str, comp_idx: int):
        """Handle checkbox change in combine mode."""
        # Count how many are selected
        selected = [idx for idx, cb in self._combine_checkboxes.items() if cb.value]

        # Enforce max 2 selections
        if len(selected) > 2:
            # Uncheck the one that was just checked
            self._combine_checkboxes[comp_idx].value = False
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Select exactly 2 components to combine."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        self.page.update()

    def _get_selected_combine_components(self) -> List[int]:
        """Get list of selected component indices in combine mode."""
        return [idx for idx, cb in self._combine_checkboxes.items() if cb.value]

    def _build_stage2_actions(self) -> List[ft.Control]:
        """Build action buttons for Stage 2."""
        actions = [
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
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Select exactly 2 components to preview."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        # Check required files exist
        if not self.ref_file or not Path(self.ref_file).exists():
            error_msg = f"BOLD reference file not found: {self.ref_file}"
            logger.error(error_msg)
            if self.on_log:
                self.on_log(f"[ERROR] {error_msg}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Cannot preview: BOLD reference file not found"),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        network_name = self._current_reject_network
        comp_idx_1, comp_idx_2 = selected[0], selected[1]

        if self.on_log:
            self.on_log(f"[INFO] Generating preview of combined IC {comp_idx_1} + IC {comp_idx_2}...")

        try:
            from pineuro.mask_extraction import combine_ica_components
            from pineuro import qc

            # Generate temporary combined mask
            temp_dir = self.mask_extraction_dir / "qc" / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_mask = temp_dir / f"{self.participant_id}_{network_name}_combined_IC{comp_idx_1}_IC{comp_idx_2}_thr.nii.gz"

            logger.info(f"Generating combined mask: {temp_mask}")
            combine_ica_components(
                components_file=self.components_file,
                component_idx_1=comp_idx_1,
                component_idx_2=comp_idx_2,
                output_file=temp_mask,
                threshold=self.threshold,
                overwrite=True,
            )

            # Generate overlay
            logger.info(f"Generating overlay with ref_file: {self.ref_file}")
            overlay_path = qc.generate_mask_overlay(
                mask_path=temp_mask,
                ref_path=self.ref_file,
                output_dir=temp_dir,
                network_name=f"{network_name}_combined",
                atlas_path=self.yeo_subject_space,
                n_yeo=self.n_yeo,
                overwrite=True,
            )

            logger.info(f"Overlay generated: {overlay_path}")

            # Update the preview container
            if overlay_path and overlay_path.exists():
                info = self.masks_info[network_name]
                info.combined_overlay_path = overlay_path

                # Update the preview container content
                if self._combine_preview_container:
                    new_content = ft.Column(
                        [
                            ft.Divider(),
                            ft.Row(
                                [
                                    ft.Text("Combined Mask Preview:", size=12, weight=ft.FontWeight.W_500),
                                    ft.Container(expand=True),
                                    ft.Text(
                                        f"IC {comp_idx_1} + IC {comp_idx_2}",
                                        size=12,
                                        color=ft.Colors.BLUE,
                                    ),
                                ],
                            ),
                            ft.Container(
                                content=self._create_image(overlay_path, height=180, width=500),
                                border=ft.border.all(1, ft.Colors.BLUE_300),
                                border_radius=5,
                            ),
                        ],
                        spacing=8,
                    )
                    self._combine_preview_container.content = new_content
                    self._combine_preview_container.update()

                if self.on_log:
                    self.on_log(f"[SUCCESS] Combined mask preview generated")
            else:
                error_msg = f"Overlay not generated or not found: {overlay_path}"
                logger.error(error_msg)
                if self.on_log:
                    self.on_log(f"[ERROR] {error_msg}")

        except Exception as ex:
            logger.error(f"Failed to generate combined preview: {ex}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to generate preview: {ex}"),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()

    def _on_accept_clicked(self, network_name: str):
        """Handle accept button click for a mask."""
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
        network_name = self._current_reject_network
        if not network_name:
            return

        # Handle combine mode
        if self._combine_mode:
            selected = self._get_selected_combine_components()
            if len(selected) != 2:
                self.page.snack_bar = ft.SnackBar(
                    ft.Text("Select exactly 2 components to combine."),
                    bgcolor=ft.Colors.ORANGE,
                )
                self.page.snack_bar.open = True
                self.page.update()
                return

            comp_idx_1, comp_idx_2 = selected[0], selected[1]
            info = self.masks_info[network_name]

            # Find correlations for both components
            corr_1, corr_2 = 0.0, 0.0
            for idx, c in info.alternative_components:
                if idx == comp_idx_1:
                    corr_1 = c
                elif idx == comp_idx_2:
                    corr_2 = c

            self._selections[network_name] = MaskSelection(
                network_name=network_name,
                source_type="ica_combined",
                component_index=comp_idx_1,
                correlation=corr_1,
                selection_reason="manual_combined",
                secondary_component_index=comp_idx_2,
                secondary_correlation=corr_2,
            )

            if self.on_log:
                self.on_log(f"[INFO] Selected combined IC {comp_idx_1} + IC {comp_idx_2} for {network_name}")

        else:
            # Single-select mode (original logic)
            if not self._alternative_selection.value:
                self.page.snack_bar = ft.SnackBar(
                    ft.Text("Please select an alternative."),
                    bgcolor=ft.Colors.ORANGE,
                )
                self.page.snack_bar.open = True
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
        from pineuro.mask_extraction import split_mask_by_hemisphere
        from pineuro import qc
        from nibabel.loadsave import load, save
        from nibabel.nifti1 import Nifti1Image

        selection = self._selections.get(self.hemisphere_network)
        if not selection:
            return

        if self.on_log:
            self.on_log(f"[INFO] Regenerating hemisphere masks for preview...")

        try:
            # First, regenerate the unthresholded mask from the selected component
            network_file = self.masks_dir / f"{self.participant_id}_{self.hemisphere_network}.nii.gz"

            if selection.source_type == "ica" and selection.component_index is not None:
                # Extract the selected ICA component
                components_img = load(self.components_file)
                components_data = components_img.get_fdata()
                ic_data = components_data[..., selection.component_index]
                ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)
                save(ic_img, network_file)

            elif selection.source_type == "yeo_atlas":
                # Use Yeo atlas - extract the network region
                from pineuro.utils import get_yeo_labels
                labels = get_yeo_labels(n_networks=self.n_yeo)
                network_index = labels.index(self.hemisphere_network) + 1
                atlas_img = load(self.yeo_subject_space)
                atlas_data = atlas_img.get_fdata()
                network_mask = (atlas_data == network_index).astype(float)
                save(Nifti1Image(network_mask, atlas_img.affine, atlas_img.header), network_file)

            # Now split into hemispheres
            left_file = self.masks_dir / f"{self.participant_id}_{self.hemisphere_network}_left_thr.nii.gz"
            right_file = self.masks_dir / f"{self.participant_id}_{self.hemisphere_network}_right_thr.nii.gz"

            split_mask_by_hemisphere(
                network_file, left_file, right_file,
                threshold=self.threshold, overwrite=True
            )

            # Regenerate hemisphere overlays
            qc_dir = self.mask_extraction_dir.parent / "qc"
            new_overlays = qc.generate_hemisphere_overlays(
                mask_extraction_dir=self.mask_extraction_dir,
                participant_id=self.participant_id,
                hemisphere_network=self.hemisphere_network,
                output_dir=qc_dir,
                n_yeo=self.n_yeo,
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
                self.on_log(f"[SUCCESS] Hemisphere masks regenerated for preview")

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

    def _on_open_fsleyes_clicked(self, e):
        """Open thresholded masks and BOLD reference in FSLeyes."""
        import shutil
        import subprocess

        # Check if fsleyes is available
        if not shutil.which("fsleyes"):
            self.page.snack_bar = ft.SnackBar(
                ft.Text("FSLeyes not found. Please ensure FSL is installed and in PATH."),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        # Check BOLD reference exists
        if not self.ref_file or not Path(self.ref_file).exists():
            self.page.snack_bar = ft.SnackBar(
                ft.Text("BOLD reference file not found."),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        # Collect all thresholded mask files
        mask_files = sorted(self.masks_dir.glob("*_thr.nii.gz"))
        if not mask_files:
            self.page.snack_bar = ft.SnackBar(
                ft.Text("No thresholded masks found."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        # Build FSLeyes command: fsleyes <ref> <mask1> <mask2> ...
        cmd = ["fsleyes", str(self.ref_file)]
        for mask_file in mask_files:
            cmd.append(str(mask_file))

        # Launch FSLeyes (non-blocking)
        try:
            subprocess.Popen(cmd, start_new_session=True)
            if self.on_log:
                self.on_log(f"[INFO] Opened FSLeyes with {len(mask_files)} masks")
        except Exception as ex:
            logger.error(f"Failed to launch FSLeyes: {ex}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to launch FSLeyes: {ex}"),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()

    def _on_reload_clicked(self, e):
        """Handle Reload button - reload selections, correlations, and images from disk."""
        try:
            import pandas as pd

            # Check if approval marker was removed (e.g., by running reproduce_masks.py)
            approval_marker = self.mask_extraction_dir / ".masks_approved"
            approval_removed = not approval_marker.exists()

            # Re-read mask_selections.json
            existing = load_mask_selections(self.masks_dir)

            # Re-read correlation matrix if it exists (updated by script)
            corr_file = self.masks_dir / "Yeo_IC_spatial_correlation.csv"
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
                        overlay_path = overlays_dir / f"{self.participant_id}_{name}_IC{sel.component_index}_overlay.png"
                        if overlay_path.exists():
                            info.overlay_path = overlay_path

                # Reset accepted state - masks need re-approval after regeneration
                # (auto selections are accepted by default, manual need review)
                info.accepted = True

                # Re-discover alternative overlay paths
                new_alt_paths = {}
                for ic_idx in range(self.n_components):
                    alt_path = overlays_dir / f"{self.participant_id}_{name}_IC{ic_idx}_overlay.png"
                    if alt_path.exists():
                        new_alt_paths[ic_idx] = alt_path
                info.alternative_overlay_paths = new_alt_paths

                # Re-discover Yeo overlay path
                yeo_path = overlays_dir / f"{self.participant_id}_{name}_yeo_overlay.png"
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
                snack_color = ft.Colors.ORANGE
            else:
                snack_message = "Reloaded selections and images from disk"
                snack_color = ft.Colors.BLUE

            self.page.snack_bar = ft.SnackBar(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.REFRESH, color=ft.Colors.WHITE),
                        ft.Text(snack_message),
                    ],
                    spacing=10,
                ),
                bgcolor=snack_color,
                duration=3000,
            )
            self.page.snack_bar.open = True
            self.page.update()

        except Exception as ex:
            logger.error(f"Failed to reload from disk: {ex}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to reload: {ex}"),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()

    def _on_generate_script_clicked(self, e):
        """Handle Generate Script button."""
        from pineuro.script_generator import generate_mask_extraction_script, save_script

        # Check if we have enough context for script generation
        if not self.input_file or not self.brain_mask_file or not self.ref_file:
            self.page.snack_bar = ft.SnackBar(
                ft.Text("Cannot generate script: missing input file paths."),
                bgcolor=ft.Colors.ORANGE,
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        try:
            # Generate script content
            script_content = generate_mask_extraction_script(
                participant_id=self.participant_id,
                mask_extraction_dir=self.mask_extraction_dir,
                input_file=self.input_file,
                mask_file=self.brain_mask_file,
                ref_file=self.ref_file,
                n_yeo=self.n_yeo,
                yeo_targets=self.yeo_targets,
                n_components=self.n_components,
                threshold=self.threshold,
                smooth_fwhm=self.smooth_fwhm,
                highpass=self.highpass,
                lowpass=self.lowpass,
                selections=self._selections,
            )

            # Save to file
            script_path = self.mask_extraction_dir / "reproduce_masks.py"
            save_script(script_content, script_path)

            logger.info(f"Generated script: {script_path}")

            # Log to console
            if self.on_log:
                self.on_log(f"[SUCCESS] Script generated: {script_path}")

            # Show success message
            self.page.snack_bar = ft.SnackBar(
                ft.Row(
                    [
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.WHITE),
                        ft.Text(f"Script saved to: {script_path.name}"),
                    ],
                    spacing=10,
                ),
                bgcolor=ft.Colors.GREEN,
                duration=4000,
            )
            self.page.snack_bar.open = True
            self.page.update()

        except Exception as ex:
            logger.error(f"Failed to generate script: {ex}")
            self.page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to generate script: {ex}"),
                bgcolor=ft.Colors.RED,
            )
            self.page.snack_bar.open = True
            self.page.update()

    def _refresh_stage1(self):
        """Refresh Stage 1 dialog to reflect changes."""
        if self._dialog:
            self._dialog.open = False
            self.page.update()
        self._show_stage1()

    def _apply_and_close(self):
        """Save selections, regenerate modified masks, and close."""
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
                    n_yeo=self.n_yeo,
                    threshold=self.threshold,
                )
            except Exception as ex:
                logger.error(f"Failed to regenerate mask for {name}: {ex}")
                self.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Failed to regenerate {name}: {ex}"),
                    bgcolor=ft.Colors.RED,
                )
                self.page.snack_bar.open = True
                self.page.update()
                return

        # Regenerate hemisphere-split masks if hemisphere_network was modified
        if self.hemisphere_network and self.hemisphere_network in modified_networks:
            logger.info(f"Regenerating hemisphere-split masks for {self.hemisphere_network}")
            try:
                from pineuro.mask_extraction import split_mask_by_hemisphere
                from pineuro import qc

                # The unthresholded mask should have been regenerated above
                network_file = self.masks_dir / f"{self.participant_id}_{self.hemisphere_network}.nii.gz"
                left_file = self.masks_dir / f"{self.participant_id}_{self.hemisphere_network}_left_thr.nii.gz"
                right_file = self.masks_dir / f"{self.participant_id}_{self.hemisphere_network}_right_thr.nii.gz"

                if network_file.exists():
                    split_mask_by_hemisphere(
                        network_file, left_file, right_file,
                        threshold=self.threshold, overwrite=True
                    )
                    logger.info(f"Regenerated hemisphere masks: {left_file.name}, {right_file.name}")

                    # Regenerate hemisphere overlays
                    if self.on_log:
                        self.on_log(f"[INFO] Regenerating hemisphere overlays for {self.hemisphere_network}...")

                    qc_dir = self.mask_extraction_dir.parent / "qc"
                    new_overlays = qc.generate_hemisphere_overlays(
                        mask_extraction_dir=self.mask_extraction_dir,
                        participant_id=self.participant_id,
                        hemisphere_network=self.hemisphere_network,
                        output_dir=qc_dir,
                        n_yeo=self.n_yeo,
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
