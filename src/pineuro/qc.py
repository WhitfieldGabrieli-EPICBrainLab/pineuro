"""
Quality control visualization and reporting for the pineuro pipeline.

This module provides comprehensive QC visualization and HTML report generation
for all stages of the neurofeedback pipeline: preprocessing, mask extraction,
and feedback sessions. Each stage has a dedicated high-level report generator
that auto-discovers outputs and creates professional HTML reports with
embedded plots.

Key Components
--------------
generate_preprocessing_report : Create QC report for preprocessing stage
    Visualizes motion parameters, framewise displacement, brain mask
    coverage, and multi-run coregistration quality.

generate_mask_extraction_report : Create QC report for mask extraction stage
    Visualizes ICA components, network-Yeo correlations, mask overlays
    in multiple spaces (subject, MNI, 2vol), and mask metrics.

generate_feedback_report : Create QC report for feedback sessions
    Visualizes activation timeseries, framewise displacement, design
    matrix, and run completion status.

Core Plotting Functions
-----------------------
plot_mask_overlay_tripanel : Tri-planar (sagittal/coronal/axial) mask overlays
plot_motion_timeseries : 6-DOF motion parameter timeseries (2×3 grid)
plot_motion_framewise_displacement : Framewise displacement with threshold
plot_brain_mask_coverage : Brain extraction quality verification
plot_registration_overlay : Edge overlay for registration verification
plot_corr_heatmap : ICA × Yeo network spatial correlation heatmap
plot_ica_component_grid : Grid display of ICA components
plot_activation_timeseries : Per-mask activation over time
plot_design_matrix : GLM design matrix heatmap
plot_fd_timeseries : Framewise displacement during feedback

Helper Functions
----------------
select_slices_from_atlas : Automatic slice selection based on network coverage
summarize_mask : Compute mask metrics (voxels, volume, centroid)
generate_alternative_overlays : Overlays for alternative ICA components
generate_yeo_network_overlay : Fallback overlay using Yeo atlas directly
generate_hemisphere_overlays : Left/right hemisphere split overlays

Data Classes
------------
MaskSummary : Container for per-mask QC metrics

Workflow
--------
1. **Automatic QC** (recommended): Enable `qc_enabled=True` in pipeline
   functions (preprocessing, mask_extraction_rest, run_feedback_session)
   to generate QC reports automatically during execution.

2. **Manual QC**: Call report generators directly for post-hoc analysis:
   - generate_preprocessing_report() for motion and registration QC
   - generate_mask_extraction_report() for ICA and mask QC
   - generate_feedback_report() for activation and session QC

Output Structure
----------------
All reports follow a consistent output pattern:

    {stage_dir}/qc/
    ├── {participant}_*_qc.html    # Main HTML report
    ├── *.png                       # Individual plot images
    └── overlays/                   # Mask overlay images
        ├── subject/                # Subject space overlays
        ├── mni/                    # MNI space overlays (if generated)
        └── 2vol/                   # Feedback space overlays

Design Philosophy
-----------------
- **Lightweight**: Only matplotlib, numpy, pandas, nibabel dependencies
- **Static HTML**: No JavaScript, pure CSS - works offline
- **Professional styling**: Consistent colorblind-safe palette, clean layout
- **Auto-discovery**: Finds outputs automatically from directory structure
- **Incremental**: Skips existing plots unless overwrite=True

Dependencies
------------
matplotlib : Plotting and visualization
numpy : Array operations for slice extraction
pandas : Data loading (TSV files)
nibabel : NIfTI file loading and spatial information

Notes
-----
- All plots use Agg backend (non-interactive, thread-safe)
- Colorblind-safe palette based on seaborn defaults
- Images are 120 DPI PNGs for web viewing
- Radiological convention: patient's left on viewer's right

Examples
--------
Generate QC report for preprocessing:

>>> from pineuro import qc
>>> report = qc.generate_preprocessing_report(
...     preprocessing_dir=Path('/data/sub-01/preprocessing'),
...     participant_id='sub-01',
...     task='rest'
... )

Generate QC report for mask extraction with browser preview:

>>> report = qc.generate_mask_extraction_report(
...     mask_extraction_dir=Path('/data/sub-01/mask_extraction'),
...     participant_id='sub-01',
...     network_names=['DefaultA', 'ContA'],
...     open_browser=True
... )

Generate QC report for feedback session:

>>> report = qc.generate_feedback_report(
...     feedback_dir=Path('/data/sub-01/feedback'),
...     subject='sub-01',
...     task='nback',
...     run=1,
...     expected_volumes=200
... )
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
import webbrowser

# Third-party imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (thread-safe, no GUI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nibabel import load

# ============================================================================
# SECTION 1: Core Plotting Primitives
# ============================================================================

def _configure_plot_style():
    """Apply consistent, publication-quality style to all plots.

    This function sets matplotlib rcParams to achieve:
    - Clean, modern sans-serif fonts (Arial, DejaVu Sans)
    - Colorblind-safe palette from seaborn
    - Professional styling (no top/right spines, subtle grid)
    - Consistent DPI (120) for web viewing
    - White backgrounds for clean HTML embedding

    The style is applied once on module import and affects all
    subsequent plots created by this module.

    Notes
    -----
    Uses Agg backend which is thread-safe and doesn't require
    a display, making it suitable for server-side generation.
    """
    plt.rcParams.update({
        # Figure
        'figure.dpi': 120,
        'figure.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 120,
        'savefig.facecolor': 'white',

        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Lines and colors
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,

        # Axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,

        # Colors (seaborn-like palette, colorblind-safe)
        'axes.prop_cycle': plt.cycler('color', [
            '#0173B2', '#DE8F05', '#029E73', '#CC78BC',
            '#CA9161', '#949494', '#ECE133', '#56B4E9'
        ])
    })


# Apply style configuration on module import
_configure_plot_style()


def _open_html_in_browser(html_path: Path, screen: int | None = None) -> None:
    """
    Open an HTML file in the default web browser on a specific screen.

    Parameters
    ----------
    html_path : Path
        Path to the HTML file to open
    screen : int, optional
        Screen number to display browser on (0 = primary, 1 = secondary, etc.).
        Defaults to pineuro.config.QC_BROWSER_SCREEN.
    """
    import logging
    import platform
    import subprocess

    from . import config

    logger = logging.getLogger(__name__)

    if not html_path.exists():
        logger.warning(f"Cannot open HTML: file does not exist: {html_path}")
        return

    # Use config default if not specified
    if screen is None:
        screen = config.QC_BROWSER_SCREEN

    try:
        # Convert to file:// URL for cross-platform compatibility
        file_url = html_path.resolve().as_uri()

        # Open browser first (non-blocking)
        webbrowser.open(file_url)
        logger.info(f"Opened QC report in browser: {html_path}")

        # On Linux X11, attempt to position browser window on specified screen
        # Skip on Wayland where xrandr/wmctrl won't work correctly
        if (platform.system() == 'Linux' and screen != 0
                and not config.is_wayland()):
            try:
                # Get screen geometry using xrandr to handle different monitor setups
                result = subprocess.run(
                    ['xrandr', '--current'],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )

                if result.returncode == 0:
                    # Parse xrandr output to find screen positions
                    # Format: "Screen N: connected primary WxH+X+Y"
                    import re
                    screens = []
                    for line in result.stdout.split('\n'):
                        match = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                        if match and ' connected' in line:
                            width, height, x_offset, y_offset = map(int, match.groups())
                            screens.append({'x': x_offset, 'y': y_offset, 'width': width, 'height': height})

                    # If we found enough screens and target screen exists
                    if len(screens) > screen:
                        target_x = screens[screen]['x']
                        target_y = screens[screen]['y']
                        screen_width = screens[screen]['width']
                        screen_height = screens[screen]['height']

                        # Size window to fit screen with some margins
                        # Use 95% of screen width/height to avoid overlap and leave room for borders
                        window_width = int(screen_width * 0.95)
                        window_height = int(screen_height * 0.95)

                        # Center window on target screen
                        window_x = target_x + (screen_width - window_width) // 2
                        window_y = target_y + (screen_height - window_height) // 2

                        # Use wmctrl to move and resize browser window (non-blocking background process)
                        # Match browser window by title containing the filename
                        window_title = html_path.name
                        subprocess.Popen(
                            ['bash', '-c',
                             f'sleep 0.8 && wmctrl -r "{window_title}" -e "0,{window_x},{window_y},{window_width},{window_height}" 2>/dev/null || true'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        logger.debug(f"Scheduled window positioning: screen {screen}, position ({window_x}, {window_y}), size {window_width}x{window_height}")
                    else:
                        logger.debug(f"Could not detect screen {screen} geometry")

            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                # xrandr or wmctrl not available - browser still opened successfully
                logger.debug(f"Window positioning unavailable: {type(e).__name__}")
        elif screen != 0:
            # Log why window positioning was skipped
            if config.is_wayland():
                logger.debug("Window positioning skipped (Wayland detected)")
            else:
                logger.debug(f"Window positioning not supported on {platform.system()}")

    except Exception as e:
        logger.warning(f"Failed to open HTML in browser: {e}")



@dataclass
class MaskSummary:
    """Container for per-mask QC metrics.

    Stores computed metrics for a single network mask, used for
    summary tables in QC reports. All spatial measurements are
    derived from the thresholded mask NIfTI file.

    Attributes
    ----------
    name : str
        Display name of the network mask (e.g., 'DefaultA', 'ContA').
    voxels : int
        Total number of non-zero voxels in the thresholded mask.
    volume_ml : float
        Mask volume in milliliters, computed from voxel count and
        voxel dimensions from the NIfTI header.
    centroid_ijk : tuple of float
        Center of mass in voxel indices (i, j, k). Useful for
        verifying mask location matches expected network anatomy.
    chosen_component : int
        Index of the ICA component selected for this network.
        Set to -1 if component selection is not applicable.
    correlation : float
        Spatial correlation (Pearson r) between the selected ICA
        component and the corresponding Yeo atlas network region.
        Higher values indicate better network identification.

    Examples
    --------
    >>> summary = MaskSummary(
    ...     name='DefaultA',
    ...     voxels=2543,
    ...     volume_ml=10.17,
    ...     centroid_ijk=(45.2, 60.1, 35.8),
    ...     chosen_component=3,
    ...     correlation=0.42
    ... )
    """
    name: str
    voxels: int
    volume_ml: float
    centroid_ijk: Tuple[float, float, float]
    chosen_component: int
    correlation: float = 0.0


def _pick_slices(n_slices: int, count: int = 6) -> list[int]:
    """Evenly sample slice indices across a dimension for visualization.

    Used when automatic slice selection is not available (e.g., no atlas
    coverage information). Samples slices uniformly to show a representative
    cross-section of the volume.

    Parameters
    ----------
    n_slices : int
        Total number of slices along the dimension.
    count : int, default=6
        Number of slices to select. If n_slices <= count, returns all indices.

    Returns
    -------
    list of int
        Evenly-spaced slice indices from 0 to n_slices-1.

    Examples
    --------
    >>> _pick_slices(100, count=5)
    [0, 25, 50, 75, 100]
    >>> _pick_slices(3, count=6)  # Returns all if fewer slices than count
    [0, 1, 2]
    """
    if n_slices <= count:
        return list(range(n_slices))
    lin = np.linspace(0, n_slices - 1, count)
    return [int(round(x)) for x in lin]


def select_slices_from_atlas(
        atlas_path: Path,
        labels: Sequence[str],
        target_label: str,
        top_n_per_plane: int = 1,
        mask_path: Path | None = None
) -> Dict[str, list[int]]:
    """Choose optimal slice indices for visualizing a network mask.

    Selects slices that maximize network visibility by finding indices
    with the highest voxel coverage. This ensures overlay plots show
    the most informative views of each network rather than arbitrary
    middle slices.

    The algorithm:
    1. Create a binary mask of the target network (from atlas or mask_path)
    2. For each plane (axial/coronal/sagittal), count voxels per slice
    3. Select the top N slices with highest counts per plane
    4. Always include the centroid slice for anatomical reference

    Parameters
    ----------
    atlas_path : Path
        Path to the Yeo atlas already resampled to participant space.
        Used for slice selection when mask_path is not provided.
    labels : sequence of str
        Atlas network labels in order. Index position + 1 gives the
        corresponding voxel value in the atlas NIfTI (e.g., labels[0]
        corresponds to atlas value 1).
    target_label : str
        Name of the network to select slices for (must be in labels).
    top_n_per_plane : int, default=1
        Number of highest-coverage slices to return per plane.
        The centroid slice is always added if not already included.
    mask_path : Path, optional
        If provided, use this mask for slice selection instead of atlas.
        Preferred when displaying participant-specific ICA-derived masks,
        as they may differ spatially from atlas regions.

    Returns
    -------
    dict
        Slice indices keyed by plane name:
        {"axial": [z1, z2, ...], "coronal": [y1, ...], "sagittal": [x1, ...]}

    Notes
    -----
    Fallback behavior if network not found or empty:
    - Returns middle slice for each plane
    - Ensures plots are still generated even for sparse masks

    Examples
    --------
    >>> slices = select_slices_from_atlas(
    ...     atlas_path=Path('/data/yeo_to_subject.nii.gz'),
    ...     labels=['VisA', 'VisB', 'DefaultA', ...],
    ...     target_label='DefaultA',
    ...     top_n_per_plane=2
    ... )
    >>> print(slices)
    {'axial': [45, 52], 'coronal': [60, 65], 'sagittal': [30, 35]}
    """
    # --- Determine which mask to use for slice selection ---
    # If a participant-specific mask is provided, prefer it over the atlas
    # because ICA-derived masks may have different spatial extent than
    # the group-level Yeo parcellation
    if mask_path is not None:
        slice_img = load(mask_path)
        slice_data = slice_img.get_fdata()
        mask = slice_data > 0  # Binary mask from any non-zero voxels
    else:
        # Use atlas to create mask for the target network
        atlas_img = load(atlas_path)
        atlas_data = atlas_img.get_fdata().astype(int)

        # Fallback: if network label unknown, use center slices
        if target_label not in labels:
            return {
                "axial": _pick_slices(atlas_data.shape[2], 1),
                "coronal": [atlas_data.shape[1] // 2],
                "sagittal": [atlas_data.shape[0] // 2],
            }

        # Atlas values are 1-indexed: labels[0] -> value 1, labels[1] -> value 2, etc.
        target_idx = labels.index(target_label) + 1
        mask = atlas_data == target_idx

    # Fallback for empty masks: use center slices to ensure valid output
    if not mask.any():
        shape = mask.shape
        return {
            "axial": _pick_slices(shape[2], 1),
            "coronal": [shape[1] // 2],
            "sagittal": [shape[0] // 2],
        }

    # --- Compute mask centroid for anatomical reference ---
    # The centroid slice is always included to show the "center" of the network,
    # even if it's not the slice with maximum coverage
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)

    def _top_slices(along: int, top_n: int) -> list[int]:
        """Find slices with highest voxel coverage along a given axis.

        For example, to find the best axial slices (along=2, the Z axis):
        - Sum voxels across X and Y for each Z position
        - Select top N slices by count
        - Always include the centroid slice
        """
        # Sum mask along all axes except 'along' to get per-slice coverage
        # For axial (along=2): sum over axes (0,1) -> counts per Z slice
        axes = tuple(i for i in range(3) if i != along)
        counts = mask.sum(axis=axes)

        # Sort slices by voxel count (descending)
        order = counts.argsort()[::-1]

        # Take top N slices that have non-zero coverage
        chosen: list[int] = []
        for idx in order[:top_n]:
            if counts[idx] > 0 and 0 <= int(idx) < counts.shape[0]:
                chosen.append(int(idx))

        # Always include centroid slice for anatomical reference
        c_idx = int(round(centroid[along]))
        if 0 <= c_idx < counts.shape[0] and c_idx not in chosen:
            chosen.append(c_idx)

        # Final fallback: middle slice if nothing else selected
        return chosen or [counts.shape[0] // 2]

    # Axis mapping: sagittal=X(0), coronal=Y(1), axial=Z(2)
    return {
        "axial": _top_slices(2, top_n_per_plane),      # Best Z slices
        "coronal": _top_slices(1, top_n_per_plane),    # Best Y slices
        "sagittal": _top_slices(0, top_n_per_plane),   # Best X slices
    }


def plot_mask_overlay_tripanel(
        mask_path: Path,
        ref_path: Path,
        out_path: Path,
        slices: Dict[str, list[int]],
        title: str | None = None,
        cmap: str = "autumn"
) -> None:
    """Create a tri-planar (sagittal, coronal, axial) mask overlay visualization.

    Generates a publication-quality figure showing the network mask overlaid
    on the BOLD reference in three orthogonal planes. Uses radiological
    convention (patient's left on viewer's right) for clinical compatibility.

    The overlay uses a dual-contour technique for visibility:
    1. White outer contour for contrast against dark brain regions
    2. Red inner contour for clear boundary definition
    3. Semi-transparent color fill to show mask extent

    Parameters
    ----------
    mask_path : Path
        Path to the mask NIfTI file (thresholded, binary or weighted).
    ref_path : Path
        Path to the BOLD reference NIfTI file. Must be in the same
        space and have identical dimensions as the mask.
    out_path : Path
        Output path for the PNG file. Parent directories are created
        if they don't exist.
    slices : dict
        Slice indices keyed by plane name:
        {"axial": [z_idx], "coronal": [y_idx], "sagittal": [x_idx]}
        Uses first index from each list if multiple provided.
    title : str, optional
        Figure title displayed above the panels.
    cmap : str, default="autumn"
        Matplotlib colormap for the mask fill color. "autumn" provides
        red-orange tones that contrast well with grayscale anatomy.

    Raises
    ------
    ValueError
        If reference and mask volumes have different shapes.

    Notes
    -----
    **Radiological Convention:**
    - Axial/Coronal: Patient's left appears on viewer's right
    - Sagittal: Anterior (frontal) appears on viewer's left

    The function automatically detects volume orientation from the NIfTI
    affine matrix and applies appropriate flips to achieve radiological
    convention regardless of the original acquisition orientation.

    **Display intensity:**
    Reference intensity is windowed to the 2nd-98th percentile range
    to handle outliers and optimize tissue contrast.

    Examples
    --------
    >>> slices = {'axial': [45], 'coronal': [60], 'sagittal': [30]}
    >>> plot_mask_overlay_tripanel(
    ...     mask_path=Path('/data/masks/DefaultA_thr.nii.gz'),
    ...     ref_path=Path('/data/sub-01_boldref.nii.gz'),
    ...     out_path=Path('/data/qc/DefaultA_overlay.png'),
    ...     slices=slices,
    ...     title='Default Mode Network A'
    ... )
    """
    # --- Load volumes ---
    mask_img = load(mask_path)
    ref_img = load(ref_path)

    mask = mask_img.get_fdata()
    ref = ref_img.get_fdata()

    if ref.shape != mask.shape:
        raise ValueError(f"Shape mismatch for overlay: ref {ref.shape} vs mask {mask.shape}")

    # =========================================================================
    # ORIENTATION DETECTION FOR RADIOLOGICAL CONVENTION
    # =========================================================================
    # NIfTI affine matrix encodes the mapping from voxel indices (i,j,k) to
    # world coordinates (x,y,z in RAS+ convention: Right, Anterior, Superior).
    #
    # The diagonal elements indicate axis direction:
    # - affine[0,0] > 0: X increases left-to-right (R orientation)
    # - affine[0,0] < 0: X increases right-to-left (L orientation)
    # - affine[1,1] > 0: Y increases posterior-to-anterior (A orientation)
    # - affine[1,1] < 0: Y increases anterior-to-posterior (P orientation)
    #
    # Radiological convention requires:
    # - Axial/Coronal views: patient's LEFT on viewer's RIGHT
    # - Sagittal view: ANTERIOR (frontal lobe) on viewer's LEFT
    # =========================================================================
    import nibabel as nib
    ref_orient = nib.aff2axcodes(ref_img.affine)  # e.g., ('R', 'A', 'S')

    # Detect X-axis direction (affects axial and coronal flip)
    x_negative = ref_img.affine[0, 0] < 0  # True = L orientation (need flip)

    # Detect Y-axis direction (affects sagittal flip)
    y_negative = ref_img.affine[1, 1] < 0  # True = A orientation (no flip needed)

    # --- Set up figure with three panels ---
    planes = [("sagittal", 0), ("coronal", 1), ("axial", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

    # Robust intensity windowing: clip to 2nd-98th percentile to handle
    # outliers (e.g., bright vessels or susceptibility artifacts)
    ref_min, ref_max = np.percentile(ref, [2, 98])

    def _get_slice(data: np.ndarray, plane: str, idx: int) -> np.ndarray:
        if plane == "axial":
            return data[:, :, idx]
        if plane == "coronal":
            return data[:, idx, :]
        if plane == "sagittal":
            return data[idx, :, :]
        raise ValueError(f"Unknown plane: {plane}")

    # --- Render each plane ---
    for ax, (plane, axis_idx) in zip(axes, planes):
        # Get slice index, falling back to middle slice if not specified
        plane_slices = slices.get(plane, [])
        if not plane_slices:
            plane_slices = [_pick_slices(ref.shape[axis_idx], 1)[0]]
        idx = plane_slices[0]  # Use first slice if multiple provided

        # Extract 2D slices from 3D volumes
        base = _get_slice(ref, plane, idx)
        overlay = _get_slice(mask, plane, idx)

        # =================================================================
        # APPLY RADIOLOGICAL CONVENTION FLIPS
        # =================================================================
        # After slicing, we have a 2D array that needs to be properly
        # oriented for display. The .T transpose swaps axes for correct
        # aspect ratio, and fliplr() handles left-right orientation.
        # =================================================================
        if plane == "sagittal":
            # Sagittal view: Y-Z plane (anterior-posterior × inferior-superior)
            # Standard convention: frontal lobe (anterior) on LEFT side
            if y_negative:
                # A orientation: Y decreases left-to-right, anterior already left
                base = base.T
                overlay = overlay.T
            else:
                # P orientation: Y increases left-to-right, flip for anterior-left
                base = np.fliplr(base.T)
                overlay = np.fliplr(overlay.T)
        elif x_negative and plane in ["coronal", "axial"]:
            # L orientation in X: patient's right is on left, need flip
            # to achieve radiological standard (patient left on viewer right)
            base = np.fliplr(base.T)
            overlay = np.fliplr(overlay.T)
        else:
            # R orientation: already radiological (patient left on right)
            base = base.T
            overlay = overlay.T

        # --- Draw the overlay with dual-contour technique ---
        # 1. Grayscale anatomical background
        ax.imshow(base, cmap="gray", origin="lower", vmin=ref_min, vmax=ref_max)

        # 2. White outer contour provides contrast against dark brain regions
        ax.contour(overlay, levels=[0.5], colors=["white"], linewidths=2.0, alpha=0.8)

        # 3. Red inner contour defines the mask boundary clearly
        ax.contour(overlay, levels=[0.5], colors=["#d62728"], linewidths=1.2, alpha=0.9)

        # 4. Semi-transparent color fill shows mask extent without obscuring anatomy
        ax.imshow(
            np.ma.masked_where(overlay <= 0, overlay),  # Only show non-zero regions
            cmap=cmap,
            alpha=0.35,  # 35% opacity for visibility through the fill
            origin="lower",
        )

        ax.set_axis_off()
        ax.set_title(f"{plane.capitalize()} (slice {idx})", fontweight='normal')

    if title:
        fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================================
# SECTION 2: Preprocessing QC
# ============================================================================

def plot_motion_timeseries(
        motion_par_path: Path,
        out_path: Path,
        title: str | None = None
) -> None:
    """
    Plot motion parameters (6 DOF) over time.

    Parameters
    ----------
    motion_par_path : Path
        Path to mcflirt .par file with motion parameters
    out_path : Path
        Path to save the PNG
    title : str, optional
        Figure title
    """
    # Load motion parameters (mcflirt .par format: 6 columns for rotation and translation)
    params = np.loadtxt(motion_par_path)

    if params.ndim == 1:
        params = params.reshape(1, -1)

    n_volumes = params.shape[0]

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    axes = axes.flatten()

    labels = ['Rotation X (rad)', 'Rotation Y (rad)', 'Rotation Z (rad)',
              'Translation X (mm)', 'Translation Y (mm)', 'Translation Z (mm)']

    for idx, (ax, label) in enumerate(zip(axes, labels)):
        data = params[:, idx]

        ax.plot(range(n_volumes), data, linewidth=1.5, color='#0173B2')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Highlight outliers (> 3 SD)
        mean_val = np.mean(data)
        std_val = np.std(data)
        outliers = np.abs(data - mean_val) > 3 * std_val
        if outliers.any():
            ax.scatter(np.where(outliers)[0], data[outliers],
                      color='#d62728', s=30, zorder=10, label=f'Outliers (>{3}σ)')
            ax.legend(loc='best', fontsize=8)

        ax.set_xlabel('Volume', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(label.split(' ')[0] + ' ' + label.split(' ')[1], fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_motion_framewise_displacement(
        motion_par_path: Path,
        out_path: Path,
        threshold: float = 0.5,
        title: str | None = None
) -> None:
    """
    Plot framewise displacement (FD) over time.

    Parameters
    ----------
    motion_par_path : Path
        Path to mcflirt .par file
    out_path : Path
        Path to save the PNG
    threshold : float, default=0.5
        FD threshold in mm for highlighting excessive motion
    title : str, optional
        Figure title
    """
    # Load motion parameters
    params = np.loadtxt(motion_par_path)

    if params.ndim == 1:
        params = params.reshape(1, -1)

    # Calculate framewise displacement
    # FD = sum of absolute derivatives of 6 motion parameters
    # Rotations are converted to mm by assuming 50mm radius
    diff = np.diff(params, axis=0)
    diff[:, :3] *= 50  # Convert rotations (radians) to mm
    fd = np.sum(np.abs(diff), axis=1)

    n_volumes = len(fd)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    ax.plot(range(1, n_volumes + 1), fd, linewidth=1.5, color='#0173B2', label='FD')
    ax.axhline(threshold, color='#d62728', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold} mm)')

    # Shade regions above threshold
    above_thresh = fd > threshold
    if above_thresh.any():
        for i in range(n_volumes):
            if above_thresh[i]:
                ax.axvspan(i + 0.5, i + 1.5, color='#d62728', alpha=0.2)

    # Summary statistics
    mean_fd = np.mean(fd)
    pct_above = (above_thresh.sum() / n_volumes) * 100
    ax.text(0.02, 0.98, f'Mean FD: {mean_fd:.3f} mm\n{pct_above:.1f}% volumes > {threshold} mm',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Volume', fontweight='bold')
    ax.set_ylabel('Framewise Displacement (mm)', fontweight='bold')
    ax.legend(loc='upper right')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_brain_mask_coverage(
        mask_path: Path,
        ref_path: Path,
        out_path: Path,
        title: str | None = None
) -> None:
    """
    Plot brain mask overlay on BOLD reference to verify extraction quality.

    This is a wrapper around plot_mask_overlay_tripanel for preprocessing QC.

    Parameters
    ----------
    mask_path : Path
        Path to brain mask NIfTI
    ref_path : Path
        Path to BOLD reference NIfTI
    out_path : Path
        Path to save the PNG
    title : str, optional
        Figure title
    """
    # Use automatic slice selection (middle slices)
    mask_img = load(mask_path)
    mask_data = mask_img.get_fdata()

    slices = {
        "axial": [mask_data.shape[2] // 2],
        "coronal": [mask_data.shape[1] // 2],
        "sagittal": [mask_data.shape[0] // 2]
    }

    plot_mask_overlay_tripanel(
        mask_path=mask_path,
        ref_path=ref_path,
        out_path=out_path,
        slices=slices,
        title=title or "Brain Mask Coverage",
        cmap="autumn"
    )


def plot_registration_overlay(
        ref_path: Path,
        moving_path: Path,
        out_path: Path,
        title: str | None = None
) -> None:
    """
    Plot edge overlay to verify registration quality.

    Parameters
    ----------
    ref_path : Path
        Path to reference (target) image
    moving_path : Path
        Path to moving (registered) image
    out_path : Path
        Path to save the PNG
    title : str, optional
        Figure title
    """
    from scipy import ndimage

    ref_img = load(ref_path)
    moving_img = load(moving_path)

    ref = ref_img.get_fdata()
    moving = moving_img.get_fdata()

    if ref.shape != moving.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape} vs moving {moving.shape}")

    # Check orientation for radiological convention
    import nibabel as nib
    x_negative = ref_img.affine[0, 0] < 0  # True for L orientation
    y_negative = ref_img.affine[1, 1] < 0  # True for A orientation

    # Detect edges using Sobel filter
    ref_edges = ndimage.sobel(ref)
    moving_edges = ndimage.sobel(moving)

    # Select middle slices
    slices_idx = {
        "axial": ref.shape[2] // 2,
        "coronal": ref.shape[1] // 2,
        "sagittal": ref.shape[0] // 2
    }

    def _get_slice(data: np.ndarray, plane: str) -> np.ndarray:
        idx = slices_idx[plane]
        if plane == "axial":
            return data[:, :, idx]
        elif plane == "coronal":
            return data[:, idx, :]
        else:  # sagittal
            return data[idx, :, :]

    # Create tripanel
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    ref_min, ref_max = np.percentile(ref, [2, 98])

    for ax, plane in zip(axes, ["sagittal", "coronal", "axial"]):
        ref_slice = _get_slice(ref, plane)
        moving_edge_slice = _get_slice(moving_edges, plane)

        # Apply radiological convention based on plane and orientation
        if plane == "sagittal":
            # Sagittal: Y-axis determines anterior-posterior
            if y_negative:
                # A orientation: anterior already on left
                ref_slice = ref_slice.T
                moving_edge_slice = moving_edge_slice.T
            else:
                # P orientation: flip to get anterior on left
                ref_slice = np.fliplr(ref_slice.T)
                moving_edge_slice = np.fliplr(moving_edge_slice.T)
        elif x_negative and plane in ["coronal", "axial"]:
            # L orientation: flip to show patient's left on right
            ref_slice = np.fliplr(ref_slice.T)
            moving_edge_slice = np.fliplr(moving_edge_slice.T)
        else:
            # R orientation: already radiological
            ref_slice = ref_slice.T
            moving_edge_slice = moving_edge_slice.T

        # Show reference in grayscale
        ax.imshow(ref_slice, cmap="gray", origin="lower", vmin=ref_min, vmax=ref_max)

        # Overlay edges from moving image in red
        edge_thresh = np.percentile(moving_edge_slice[moving_edge_slice > 0], 75)
        edge_mask = moving_edge_slice > edge_thresh
        ax.contour(edge_mask, levels=[0.5], colors=['#d62728'], linewidths=1.2, alpha=0.8)

        ax.set_axis_off()
        ax.set_title(f"{plane.capitalize()} (slice {slices_idx[plane]})", fontweight='normal')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def generate_preprocessing_report(
        preprocessing_dir: Path,
        participant_id: str,
        task: str,
        output_dir: Path | None = None,
        overwrite: bool = False
) -> Path:
    """
    Generate comprehensive QC report for preprocessing outputs.

    Automatically discovers motion parameters, brain masks, BOLD references,
    and coregistered runs to create a complete quality control report.

    Parameters
    ----------
    preprocessing_dir : Path
        Root preprocessing directory (contains 02_motion_correction/, 03_brain_extraction/, etc.)
    participant_id : str
        Participant identifier
    task : str
        Task name (e.g., 'rest')
    output_dir : Path, optional
        QC output directory. Defaults to preprocessing_dir / 'qc'
    overwrite : bool, default=False
        Regenerate all plots even if they exist

    Returns
    -------
    Path
        Path to generated HTML report

    Example
    -------
    >>> from pineuro import qc
    >>> report = qc.generate_preprocessing_report(
    ...     preprocessing_dir=Path('/data/sub-01/preprocessing'),
    ...     participant_id='sub-01',
    ...     task='rest'
    ... )
    >>> print(f"QC report: {report}")
    """
    import logging
    from glob import glob

    logger = logging.getLogger(__name__)

    preprocessing_dir = Path(preprocessing_dir)
    output_dir = Path(output_dir) if output_dir is not None else preprocessing_dir / 'qc'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating preprocessing QC report for {participant_id}, task={task}")

    # Auto-discover outputs
    mcf_dir = preprocessing_dir / "02_motion_correction"
    brain_dir = preprocessing_dir / "03_brain_extraction"
    coreg_dir = preprocessing_dir / "04_coregistered"

    # Find motion parameter files (.par files from mcflirt -plots)
    par_files = sorted(glob(str(mcf_dir / "*.par")))

    # Find brain mask and reference
    mask_pattern = f"{participant_id}_task-{task}_preproc_mask.nii.gz"
    ref_pattern = f"{participant_id}_task-{task}_boldref.nii.gz"

    mask_file = preprocessing_dir / mask_pattern
    ref_file = preprocessing_dir / ref_pattern

    # Check for multi-run coregistered data
    coreg_files = []
    if coreg_dir.exists():
        coreg_files = sorted(glob(str(coreg_dir / f"{participant_id}_task-{task}_run-*.nii.gz")))

    # Build QC sections
    sections = []

    # Summary box
    summary_data = {
        'Participant': participant_id,
        'Task': task,
        'Number of runs': str(len(par_files)) if par_files else 'N/A',
        'Multi-run coregistration': 'Yes' if coreg_files else 'No'
    }
    sections.append({'type': 'summary', 'content': summary_data})

    # Generate motion plots
    if par_files:
        sections.append({'type': 'heading', 'content': 'Motion Parameters'})

        motion_plots = []
        fd_plots = []

        for par_file in par_files:
            par_path = Path(par_file)
            run_name = par_path.stem.replace('.par', '')

            # Motion timeseries
            motion_out = output_dir / f"motion_timeseries_{run_name}.png"
            if overwrite or not motion_out.exists():
                logger.info(f"Generating motion timeseries for {run_name}")
                plot_motion_timeseries(
                    motion_par_path=par_path,
                    out_path=motion_out,
                    title=f"Motion Parameters - {run_name}"
                )
            motion_plots.append({
                'title': f'{run_name}',
                'path': motion_out.relative_to(output_dir).as_posix(),
                'alt': f'Motion parameters for {run_name}'
            })

            # Framewise displacement
            fd_out = output_dir / f"framewise_displacement_{run_name}.png"
            if overwrite or not fd_out.exists():
                logger.info(f"Generating framewise displacement for {run_name}")
                plot_motion_framewise_displacement(
                    motion_par_path=par_path,
                    out_path=fd_out,
                    title=f"Framewise Displacement - {run_name}"
                )
            fd_plots.append({
                'title': f'{run_name}',
                'path': fd_out.relative_to(output_dir).as_posix(),
                'alt': f'Framewise displacement for {run_name}'
            })

        sections.append({'type': 'grid', 'content': motion_plots})

        sections.append({'type': 'heading', 'content': 'Framewise Displacement'})
        sections.append({'type': 'grid', 'content': fd_plots})
    else:
        logger.warning(f"No motion parameter files found in {mcf_dir}")

    # Brain mask coverage
    if mask_file.exists() and ref_file.exists():
        sections.append({'type': 'heading', 'content': 'Brain Mask Coverage'})

        mask_out = output_dir / "brain_mask_coverage.png"
        if overwrite or not mask_out.exists():
            logger.info("Generating brain mask coverage plot")
            plot_brain_mask_coverage(
                mask_path=mask_file,
                ref_path=ref_file,
                out_path=mask_out,
                title="Brain Mask on BOLD Reference"
            )

        sections.append({
            'type': 'single_image',
            'path': mask_out.relative_to(output_dir).as_posix(),
            'alt': 'Brain mask coverage'
        })
    else:
        logger.warning(f"Brain mask or reference not found: {mask_file}, {ref_file}")

    # Multi-run coregistration overlays
    if len(coreg_files) > 1:
        sections.append({'type': 'heading', 'content': 'Multi-Run Coregistration'})

        reg_plots = []
        # Compare run 2+ to run 1
        ref_run = Path(coreg_files[0])
        for coreg_file in coreg_files[1:]:
            moving_path = Path(coreg_file)
            run_name = moving_path.stem

            reg_out = output_dir / f"registration_overlay_{run_name}.png"
            if overwrite or not reg_out.exists():
                logger.info(f"Generating registration overlay for {run_name}")
                plot_registration_overlay(
                    ref_path=ref_run,
                    moving_path=moving_path,
                    out_path=reg_out,
                    title=f"Registration: {run_name} → run-01"
                )

            reg_plots.append({
                'title': f'{run_name} → run-01',
                'path': reg_out.relative_to(output_dir).as_posix(),
                'alt': f'Registration overlay for {run_name}'
            })

        sections.append({'type': 'grid', 'content': reg_plots})

    # Generate HTML report
    html_report = output_dir / f"{participant_id}_task-{task}_preprocessing_qc.html"
    html = _generate_html_template(
        title=f"Preprocessing QC - {participant_id}",
        participant_id=participant_id,
        stage=f"Preprocessing (task: {task})",
        sections=sections
    )
    html_report.write_text(html)

    logger.info(f"Preprocessing QC report saved to: {html_report}")
    return html_report


# ============================================================================
# SECTION 3: Mask Extraction QC
# ============================================================================

def plot_corr_heatmap(
        corr_df: pd.DataFrame,
        chosen_components: Mapping[str, int],
        out_path: Path
) -> None:
    """
    Render ICA component × network correlation heatmap.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation matrix (components x networks)
    chosen_components : mapping
        Mapping from network label to selected ICA component index
    out_path : Path
        Path to save the heatmap PNG
    """
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    im = ax.imshow(corr_df.values, aspect="auto", cmap="viridis", vmin=0, vmax=0.5)
    ax.set_xlabel("Yeo Networks", fontweight='bold')
    ax.set_ylabel("ICA Component Index", fontweight='bold')
    ax.set_xticks(range(corr_df.shape[1]))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr_df.shape[0]))
    ax.set_yticklabels(range(corr_df.shape[0]))
    ax.set_title("ICA Component × Yeo Network Spatial Correlations", pad=10)

    # Mark chosen components with prominent markers
    for net, comp_idx in chosen_components.items():
        if net in corr_df.columns and 0 <= comp_idx < corr_df.shape[0]:
            j = list(corr_df.columns).index(net)
            # Add white outer ring for visibility
            ax.scatter(j, comp_idx, marker="o", s=80, color="white", zorder=10)
            ax.scatter(j, comp_idx, marker="o", color="#d62728", s=50, edgecolors="white", linewidths=1.5, zorder=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Spatial Correlation (r)")
    cbar.ax.set_ylabel("Spatial Correlation (r)", rotation=270, labelpad=20, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_ica_component_grid(
        components_path: Path,
        ref_path: Path,
        out_path: Path,
        n_components: int = 10,
        title: str | None = None
) -> None:
    """
    Plot ICA components in a grid layout (5x2 for 10 components).

    Parameters
    ----------
    components_path : Path
        Path to ICA components NIfTI file (4D volume)
    ref_path : Path
        Path to reference image for background
    out_path : Path
        Path to save the PNG
    n_components : int, default=10
        Number of components to display
    title : str, optional
        Figure title
    """
    components_img = load(components_path)
    components_data = components_img.get_fdata()

    ref_img = load(ref_path)
    ref_data = ref_img.get_fdata()

    # Check orientation
    import nibabel as nib
    x_negative = ref_img.affine[0, 0] < 0
    y_negative = ref_img.affine[1, 1] < 0

    # Determine grid layout
    n_cols = 2
    n_rows = (n_components + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5), constrained_layout=True)
    axes = axes.flatten() if n_components > 1 else [axes]

    # Get middle axial slice
    z_mid = ref_data.shape[2] // 2
    ref_slice = ref_data[:, :, z_mid]

    # Apply orientation for axial view
    if x_negative:
        ref_slice = np.fliplr(ref_slice.T)
    else:
        ref_slice = ref_slice.T

    ref_min, ref_max = np.percentile(ref_slice, [2, 98])

    for idx in range(n_components):
        ax = axes[idx]

        # Extract component
        comp_data = components_data[..., idx]
        comp_slice = comp_data[:, :, z_mid]

        # Apply same orientation
        if x_negative:
            comp_slice = np.fliplr(comp_slice.T)
        else:
            comp_slice = comp_slice.T

        # Show reference as background
        ax.imshow(ref_slice, cmap="gray", origin="lower", vmin=ref_min, vmax=ref_max, alpha=0.5)

        # Overlay component (positive and negative)
        comp_max = np.abs(comp_slice).max()
        if comp_max > 0:
            ax.imshow(
                np.ma.masked_where(np.abs(comp_slice) < comp_max * 0.1, comp_slice),
                cmap="RdBu_r",
                origin="lower",
                vmin=-comp_max,
                vmax=comp_max,
                alpha=0.8
            )

        ax.set_axis_off()
        ax.set_title(f"IC {idx}", fontsize=10, fontweight='normal')

    # Hide unused subplots
    for idx in range(n_components, len(axes)):
        axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarize_mask(
        mask_path: Path,
        name: str,
        chosen_component: int = -1,
        correlation: float = 0.0
) -> MaskSummary:
    """Compute basic metrics for a network mask.

    Loads a thresholded mask NIfTI and computes summary statistics
    for inclusion in QC reports. Metrics help verify mask quality:
    voxel count should be close to threshold, centroid should be
    anatomically plausible for the network.

    Parameters
    ----------
    mask_path : Path
        Path to the thresholded mask NIfTI file (binary or weighted).
    name : str
        Display name for the mask (used in reports).
    chosen_component : int, default=-1
        Index of the ICA component used to create this mask.
        Set to -1 if not applicable (e.g., for atlas-derived masks).
    correlation : float, default=0.0
        Spatial correlation with the corresponding Yeo network region.

    Returns
    -------
    MaskSummary
        Dataclass containing computed metrics: voxels, volume_ml,
        centroid_ijk, chosen_component, and correlation.

    Notes
    -----
    Volume is computed as: voxels × (voxel_x × voxel_y × voxel_z) / 1000
    where voxel dimensions come from the NIfTI header's zooms.

    Examples
    --------
    >>> summary = summarize_mask(
    ...     mask_path=Path('/data/sub-01/masks/DefaultA_thr.nii.gz'),
    ...     name='DefaultA',
    ...     chosen_component=3,
    ...     correlation=0.42
    ... )
    >>> print(f"{summary.name}: {summary.voxels} voxels, {summary.volume_ml:.1f} mL")
    DefaultA: 2000 voxels, 8.0 mL
    """
    img = load(mask_path)
    data = img.get_fdata()
    mask = data > 0

    voxel_dims = img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])

    voxels = int(mask.sum())
    volume_ml = (voxels * voxel_volume_mm3) / 1000.0

    coords = np.argwhere(mask)
    if coords.size:
        centroid = tuple(coords.mean(axis=0).tolist())
    else:
        centroid = (float("nan"), float("nan"), float("nan"))

    return MaskSummary(
        name=name,
        voxels=voxels,
        volume_ml=volume_ml,
        centroid_ijk=tuple(round(c, 2) for c in centroid),
        chosen_component=chosen_component,
        correlation=correlation
    )


def generate_mask_extraction_report(
        mask_extraction_dir: Path,
        participant_id: str,
        network_names: Sequence[str],
        output_dir: Path | None = None,
        n_yeo: int | None = None,
        overwrite: bool = False,
        open_browser: bool = False
) -> Path:
    """
    Generate comprehensive QC report for mask extraction outputs.

    Automatically discovers ICA components, masks, correlations, and overlays
    to create a complete quality control report.

    Parameters
    ----------
    mask_extraction_dir : Path
        Root mask extraction directory (contains 01_ica/, 02_mnireg/, 03_masks/)
    participant_id : str
        Participant identifier
    network_names : sequence of str
        Names of extracted networks (e.g., ['Default', 'ContA', 'ContB'])
    output_dir : Path, optional
        QC output directory. Defaults to mask_extraction_dir.parent / 'qc'
    n_yeo : int, optional
        Yeo parcellation (7 or 17 networks). If not provided, infers from
        network names by checking if any match 17-network labels.
    overwrite : bool, default=False
        Regenerate all plots even if they exist
    open_browser : bool, default=False
        Automatically open the HTML report in the default web browser

    Returns
    -------
    Path
        Path to generated HTML report

    Example
    -------
    >>> from pineuro import qc
    >>> report = qc.generate_mask_extraction_report(
    ...     mask_extraction_dir=Path('/data/sub-01/mask_extraction'),
    ...     participant_id='sub-01',
    ...     network_names=['Default', 'ContA', 'ContB'],
    ...     open_browser=True
    ... )
    >>> print(f"QC report: {report}")
    """
    import logging
    from glob import glob

    logger = logging.getLogger(__name__)

    mask_extraction_dir = Path(mask_extraction_dir)
    output_dir = Path(output_dir) if output_dir is not None else mask_extraction_dir.parent / 'qc'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating mask extraction QC report for {participant_id}")

    # Auto-discover outputs
    ica_dir = mask_extraction_dir / "01_ica"
    masks_dir = mask_extraction_dir / "03_masks"

    # Find ICA components and reference
    components_file = ica_dir / f"{participant_id}_components.nii.gz"
    # Search for boldref in preprocessing directory (sibling to mask_extraction)
    preprocessing_dir = mask_extraction_dir.parent / "preprocessing"
    if preprocessing_dir.exists():
        ref_files = sorted(glob(str(preprocessing_dir / "*boldref.nii.gz")))
    else:
        # Fallback: search in parent directory
        ref_files = sorted(glob(str(mask_extraction_dir.parent / "**/*boldref.nii.gz"), recursive=True))
    ref_file = Path(ref_files[0]) if ref_files else None

    # Find correlation matrix
    corr_file = masks_dir / "Yeo_IC_spatial_correlation.csv"

    # Find network masks
    network_masks = {}
    for network in network_names:
        mask_file = masks_dir / f"{participant_id}_{network}_thr.nii.gz"
        if mask_file.exists():
            network_masks[network] = mask_file

    # Find overlays
    overlays_dir = output_dir / "overlays"
    subject_overlay_dir = overlays_dir / "subject"

    # Build QC sections
    sections = []

    # Summary box
    summary_data = {
        'Participant': participant_id,
        'ICA components': str(components_file.exists()),
        'Networks extracted': f"{len(network_masks)} / {len(network_names)}",
        'Correlation matrix': str(corr_file.exists())
    }
    sections.append({'type': 'summary', 'content': summary_data})

    # Load correlation data if available
    corr_df = None
    chosen_components = {}
    if corr_file.exists():
        corr_df = pd.read_csv(corr_file)
        # The CSV has no index column - just network columns with correlation values
        # Rows represent ICA components (0, 1, 2, ...)
        chosen_components = {net: int(corr_df[net].idxmax()) for net in network_names if net in corr_df.columns}

    # ICA component grid
    if components_file.exists() and ref_file and ref_file.exists():
        sections.append({'type': 'heading', 'content': 'ICA Components'})

        ica_grid_out = output_dir / "ica_component_grid.png"
        if overwrite or not ica_grid_out.exists():
            logger.info("Generating ICA component grid")
            plot_ica_component_grid(
                components_path=components_file,
                ref_path=ref_file,
                out_path=ica_grid_out,
                n_components=min(10, corr_df.shape[0] if corr_df is not None else 10),
                title="ICA Components (axial view)"
            )

        sections.append({
            'type': 'single_image',
            'path': ica_grid_out.relative_to(output_dir).as_posix(),
            'alt': 'ICA component grid'
        })

    # Correlation heatmap
    if corr_df is not None:
        sections.append({'type': 'heading', 'content': 'ICA × Yeo Network Correlations'})

        heatmap_out = output_dir / "correlation_heatmap.png"
        if overwrite or not heatmap_out.exists():
            logger.info("Generating correlation heatmap")
            plot_corr_heatmap(corr_df, chosen_components, heatmap_out)

        sections.append({
            'type': 'single_image',
            'path': heatmap_out.relative_to(output_dir).as_posix(),
            'alt': 'Correlation heatmap'
        })

    # Determine number of networks and get Yeo labels (needed for both subject and MNI overlays)
    from pineuro.utils import get_yeo_labels
    if n_yeo is not None:
        n_networks = n_yeo
    else:
        # Infer from network names - check if any names match 17-network labels
        labels_17 = get_yeo_labels(17)
        if any(name in labels_17 for name in network_names):
            n_networks = 17
        else:
            n_networks = 7
    labels = get_yeo_labels(n_networks)

    # Generate mask overlays if they don't exist
    # Subject space overlays
    if ref_file and ref_file.exists() and network_masks:
        subject_overlay_dir.mkdir(parents=True, exist_ok=True)

        # Load Yeo atlas in subject space for slice selection
        yeo2sub_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "*yeo2sub.nii.gz")))
        yeo2sub = Path(yeo2sub_files[0]) if yeo2sub_files else None

        for network, mask_path in network_masks.items():
            overlay_png = subject_overlay_dir / f"{participant_id}_{network}_overlay.png"
            if overwrite or not overlay_png.exists():
                logger.info(f"Generating {network} overlay (BOLD reference)")

                # Get mask summary for voxel count
                summary = summarize_mask(mask_path, name=network)

                # Select slices using mask
                atlas_slices = select_slices_from_atlas(
                    atlas_path=yeo2sub if yeo2sub else ref_file,
                    labels=labels,
                    target_label=network,
                    top_n_per_plane=1,
                    mask_path=mask_path
                )

                plot_mask_overlay_tripanel(
                    mask_path=mask_path,
                    ref_path=ref_file,
                    out_path=overlay_png,
                    slices=atlas_slices,
                    title=f"{network} ({summary.voxels} voxels)"
                )

    # MNI space overlays removed for performance optimization
    # (was taking ~15-20s per network with no significant QC benefit)

    # 2vol space overlays (feedback coregistration)
    # Check both locations: initial 2vol in mask_extraction/04_2vol, realtime in feedback/01_2vol
    initial_twovol_dir = mask_extraction_dir / "04_2vol"
    feedback_dir = mask_extraction_dir.parent / "feedback"
    realtime_twovol_dir = feedback_dir / "01_2vol" if feedback_dir.exists() else None

    # Use initial 2vol dir if it exists, otherwise fall back to realtime location
    twovol_dir = initial_twovol_dir if initial_twovol_dir.exists() else realtime_twovol_dir

    if twovol_dir and twovol_dir.exists() and network_masks:
        twovol_overlay_dir = overlays_dir / "2vol"
        twovol_overlay_dir.mkdir(parents=True, exist_ok=True)

        # Find 2vol mean files (one per feedback run)
        twovol_mean_files = sorted(glob(str(twovol_dir / f"{participant_id}_task-2vol_run-*_mean.nii.gz")))

        if twovol_mean_files:
            # Find Yeo atlas warped to 2vol space
            twovol_masks_dir = twovol_dir / "masks"

            for mean_file in twovol_mean_files:
                mean_path = Path(mean_file)
                run_str = mean_path.stem.split("_run-")[1].replace("_mean", "")

                # Find corresponding Yeo atlas
                yeo_2vol = twovol_masks_dir / f"Yeo_to_task-2vol_run-{run_str}_mean.nii.gz"

                if not yeo_2vol.exists():
                    logger.warning(f"Yeo atlas not found for 2vol run {run_str}: {yeo_2vol}")
                    continue

                for network, mask_path in network_masks.items():
                    # Find warped mask for this run
                    mask_2vol = twovol_masks_dir / f"{participant_id}_{network}_thr_to_task-2vol_run-{run_str}_mean.nii.gz"

                    if not mask_2vol.exists():
                        logger.warning(f"2vol mask not found for {network}, run {run_str}: {mask_2vol}")
                        continue

                    overlay_png = twovol_overlay_dir / f"{participant_id}_{network}_run-{run_str}_overlay.png"
                    if overwrite or not overlay_png.exists():
                        logger.info(f"Generating {network} overlay (2vol run {run_str})")

                        # Get mask summary for voxel count
                        summary = summarize_mask(mask_2vol, name=network)

                        # Select slices using warped mask
                        atlas_slices_2vol = select_slices_from_atlas(
                            atlas_path=yeo_2vol,
                            labels=labels,
                            target_label=network,
                            top_n_per_plane=1,
                            mask_path=mask_2vol
                        )

                        plot_mask_overlay_tripanel(
                            mask_path=mask_2vol,
                            ref_path=mean_path,
                            out_path=overlay_png,
                            slices=atlas_slices_2vol,
                            title=f"{network} (2vol run {run_str}, {summary.voxels} voxels)"
                        )

    # Display overlays in report (subject and 2vol - MNI removed for performance)
    overlay_spaces = [
        ("BOLD reference", subject_overlay_dir),
        ("2vol feedback", overlays_dir / "2vol")
    ]

    for space_name, overlay_dir in overlay_spaces:
        if overlay_dir.exists():
            overlays = sorted(glob(str(overlay_dir / f"{participant_id}_*_overlay.png")))
            if overlays:
                sections.append({'type': 'heading', 'content': f'Mask Overlays ({space_name})'})
                grid_items = []
                for overlay_path in overlays:
                    overlay_file = Path(overlay_path)
                    # For 2vol, network name includes run info
                    if space_name == "2vol feedback":
                        # Extract network_run-XX from filename
                        stem = overlay_file.stem.replace(f"{participant_id}_", "").replace("_overlay", "")
                        grid_items.append({
                            'title': stem,
                            'path': overlay_file.relative_to(output_dir).as_posix(),
                            'alt': f'{stem} overlay ({space_name})'
                        })
                    else:
                        network_name = overlay_file.stem.replace(f"{participant_id}_", "").replace("_overlay", "")
                        grid_items.append({
                            'title': network_name,
                            'path': overlay_file.relative_to(output_dir).as_posix(),
                            'alt': f'{network_name} overlay ({space_name})'
                        })
                sections.append({'type': 'grid', 'content': grid_items})

    # Mask summary table
    if network_masks:
        sections.append({'type': 'heading', 'content': 'Mask Summary'})
        table_rows = []
        for network, mask_path in network_masks.items():
            chosen_comp = chosen_components.get(network, -1)
            correlation = float(corr_df.loc[chosen_comp, network]) if corr_df is not None and network in corr_df.columns else 0.0
            summary = summarize_mask(mask_path, name=network, chosen_component=chosen_comp, correlation=correlation)
            table_rows.append([
                summary.name,
                str(summary.voxels),
                f"{summary.volume_ml:.2f}",
                str(summary.centroid_ijk),
                str(summary.chosen_component),
                f"{summary.correlation:.3f}"
            ])

        sections.append({
            'type': 'table',
            'headers': ['Name', 'Voxels', 'Volume (mL)', 'Centroid (i,j,k)', 'Chosen component', 'Correlation (r)'],
            'rows': table_rows
        })

    # Generate HTML report
    html_report = output_dir / f"{participant_id}_mask_extraction_qc.html"
    html = _generate_html_template(
        title=f"Mask Extraction QC - {participant_id}",
        participant_id=participant_id,
        stage="Mask Extraction",
        sections=sections
    )
    html_report.write_text(html)

    logger.info(f"Mask extraction QC report saved to: {html_report}")

    if open_browser:
        _open_html_in_browser(html_report)

    return html_report


def generate_task_mask_extraction_report(
        mask_extraction_dir: Path,
        participant_id: str,
        mask_names: Sequence[str],
        contrasts: dict[str, str],
        stat_type: str = "t",
        output_dir: Path | None = None,
        overwrite: bool = False,
        open_browser: bool = False
) -> Path:
    """
    Generate QC report for task-based (GLM) mask extraction outputs.

    Creates an HTML report with design matrix visualization, contrast stat maps,
    thresholded mask overlays, and mask summary statistics.

    Parameters
    ----------
    mask_extraction_dir : Path
        Root mask extraction directory (contains 01_glm/, 02_masks/)
    participant_id : str
        Participant identifier
    mask_names : sequence of str
        Names of extracted masks (e.g., ['CEN', 'DMN'])
    contrasts : dict[str, str]
        Mapping of mask name to contrast expression
    stat_type : str
        Statistical map type ("t" or "z")
    output_dir : Path, optional
        QC output directory. Defaults to mask_extraction_dir.parent / 'qc'
    overwrite : bool, default=False
        Regenerate all plots even if they exist
    open_browser : bool, default=False
        Automatically open the HTML report in the default web browser

    Returns
    -------
    Path
        Path to generated HTML report
    """
    import logging
    from glob import glob

    logger = logging.getLogger(__name__)

    mask_extraction_dir = Path(mask_extraction_dir)
    output_dir = Path(output_dir) if output_dir is not None else mask_extraction_dir.parent / 'qc'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating task-based mask extraction QC report for {participant_id}")

    glm_dir = mask_extraction_dir / "01_glm"
    # Cluster method uses 03_masks/; top_n uses 02_masks/
    masks_dir = mask_extraction_dir / "03_masks"
    if not masks_dir.exists():
        masks_dir = mask_extraction_dir / "02_masks"

    # Find BOLD reference for overlays
    preprocessing_dir = mask_extraction_dir.parent / "preprocessing"
    if preprocessing_dir.exists():
        ref_files = sorted(glob(str(preprocessing_dir / "*boldref.nii.gz")))
    else:
        ref_files = sorted(glob(str(mask_extraction_dir.parent / "**/*boldref.nii.gz"), recursive=True))
    ref_file = Path(ref_files[0]) if ref_files else None

    # Find GLM metadata
    glm_info_file = glm_dir / f"{participant_id}_glm_info.json"
    glm_info = None
    if glm_info_file.exists():
        import json
        with open(glm_info_file, 'r') as f:
            glm_info = json.load(f)

    # Find network masks
    network_masks = {}
    for mask_name in mask_names:
        mask_file = masks_dir / f"{participant_id}_{mask_name}_thr.nii.gz"
        if mask_file.exists():
            network_masks[mask_name] = mask_file

    # Build QC sections
    sections = []

    # Summary box
    summary_data = {
        'Participant': participant_id,
        'Extraction method': 'GLM contrast (task localizer)',
        'Stat type': stat_type,
        'Masks extracted': f"{len(network_masks)} / {len(mask_names)}",
    }
    if glm_info:
        summary_data['Conditions'] = ', '.join(glm_info.get('conditions', []))
        summary_data['N volumes'] = str(glm_info.get('n_volumes', '?'))
        summary_data['HRF model'] = glm_info.get('hrf_model', '?')
    sections.append({'type': 'summary', 'content': summary_data})

    # Contrast definitions
    sections.append({'type': 'heading', 'content': 'Contrast Definitions'})
    contrast_rows = [[name, expr] for name, expr in contrasts.items()]
    sections.append({
        'type': 'table',
        'headers': ['Mask Name', 'Contrast Expression'],
        'rows': contrast_rows
    })

    # Design matrix visualization
    design_file = glm_dir / f"{participant_id}_design_matrix.tsv"
    if design_file.exists():
        sections.append({'type': 'heading', 'content': 'Design Matrix'})

        design_png = output_dir / "design_matrix.png"
        if overwrite or not design_png.exists():
            logger.info("Generating design matrix plot")
            _plot_task_design_matrix(design_file, design_png)

        sections.append({
            'type': 'single_image',
            'path': design_png.relative_to(output_dir).as_posix(),
            'alt': 'GLM design matrix'
        })

    # Contrast stat map overlays
    if ref_file and ref_file.exists():
        subject_overlay_dir = output_dir / "overlays" / "subject"
        subject_overlay_dir.mkdir(parents=True, exist_ok=True)

        for mask_name in mask_names:
            stat_map_path = glm_dir / f"{participant_id}_{mask_name}_{stat_type}map.nii.gz"
            mask_path = network_masks.get(mask_name)

            if mask_path and mask_path.exists():
                overlay_png = subject_overlay_dir / f"{participant_id}_{mask_name}_overlay.png"
                if overwrite or not overlay_png.exists():
                    logger.info(f"Generating {mask_name} mask overlay")

                    summary = summarize_mask(mask_path, name=mask_name)

                    # Use mask itself for slice selection (no atlas needed)
                    atlas_slices = select_slices_from_atlas(
                        atlas_path=ref_file,
                        labels=[],
                        target_label="",
                        top_n_per_plane=1,
                        mask_path=mask_path
                    )

                    plot_mask_overlay_tripanel(
                        mask_path=mask_path,
                        ref_path=ref_file,
                        out_path=overlay_png,
                        slices=atlas_slices,
                        title=f"{mask_name}: {contrasts.get(mask_name, '')} ({summary.voxels} voxels)"
                    )

    # Display overlays
    subject_overlay_dir = output_dir / "overlays" / "subject"
    if subject_overlay_dir.exists():
        overlays = sorted(glob(str(subject_overlay_dir / f"{participant_id}_*_overlay.png")))
        if overlays:
            sections.append({'type': 'heading', 'content': 'Mask Overlays (BOLD reference)'})
            grid_items = []
            for overlay_path in overlays:
                overlay_file = Path(overlay_path)
                mask_name = overlay_file.stem.replace(f"{participant_id}_", "").replace("_overlay", "")
                grid_items.append({
                    'title': mask_name,
                    'path': overlay_file.relative_to(output_dir).as_posix(),
                    'alt': f'{mask_name} overlay'
                })
            sections.append({'type': 'grid', 'content': grid_items})

    # Mask summary table
    if network_masks:
        sections.append({'type': 'heading', 'content': 'Mask Summary'})
        table_rows = []
        for mask_name, mask_path in network_masks.items():
            summary = summarize_mask(mask_path, name=mask_name)
            table_rows.append([
                summary.name,
                contrasts.get(mask_name, ''),
                str(summary.voxels),
                f"{summary.volume_ml:.2f}",
                str(summary.centroid_ijk),
            ])

        sections.append({
            'type': 'table',
            'headers': ['Name', 'Contrast', 'Voxels', 'Volume (mL)', 'Centroid (i,j,k)'],
            'rows': table_rows
        })

    # Generate HTML report
    html_report = output_dir / f"{participant_id}_mask_extraction_qc.html"
    html = _generate_html_template(
        title=f"Task Mask Extraction QC - {participant_id}",
        participant_id=participant_id,
        stage="Mask Extraction (GLM)",
        sections=sections
    )
    html_report.write_text(html)

    logger.info(f"Task mask extraction QC report saved to: {html_report}")

    if open_browser:
        _open_html_in_browser(html_report)

    return html_report


def _plot_task_design_matrix(design_file: Path, out_path: Path) -> None:
    """Plot GLM design matrix heatmap for QC report.

    Parameters
    ----------
    design_file : Path
        Path to design matrix TSV file
    out_path : Path
        Output path for the PNG plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    design_df = pd.read_csv(design_file, sep='\t', index_col=0)

    fig, ax = plt.subplots(figsize=(max(6, len(design_df.columns) * 0.8), 8))
    im = ax.imshow(design_df.values, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Regressors')
    ax.set_ylabel('Volume')
    ax.set_xticks(range(len(design_df.columns)))
    ax.set_xticklabels(design_df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_title('GLM Design Matrix')
    fig.colorbar(im, ax=ax, shrink=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_alternative_overlays(
        mask_extraction_dir: Path,
        participant_id: str,
        network_names: Sequence[str],
        n_alternatives: int = 3,
        output_dir: Path | None = None,
        n_yeo: int | None = None,
        overwrite: bool = False
) -> Dict[str, Dict[int, Path]]:
    """
    Generate overlay images for alternative ICA components.

    Creates overlay images for the top N alternative ICA components for each
    network, enabling visual comparison in the mask selection dialog.

    Parameters
    ----------
    mask_extraction_dir : Path
        Root mask extraction directory (contains 01_ica/, 02_mnireg/, 03_masks/)
    participant_id : str
        Participant identifier
    network_names : sequence of str
        Names of extracted networks (e.g., ['DefaultA', 'ContA'])
    n_alternatives : int, default=3
        Number of alternative components to generate overlays for
    output_dir : Path, optional
        QC output directory. Defaults to mask_extraction_dir.parent / 'qc'
    n_yeo : int, optional
        Yeo parcellation (7 or 17 networks). If not provided, infers from
        network names by checking if any match 17-network labels.
    overwrite : bool, default=False
        Regenerate overlays even if they exist

    Returns
    -------
    dict
        Dictionary mapping network_name -> {component_idx: overlay_path}
    """
    import logging
    from glob import glob
    from nibabel.nifti1 import Nifti1Image
    from nibabel.loadsave import save

    from pineuro.utils import get_yeo_labels

    logger = logging.getLogger(__name__)

    mask_extraction_dir = Path(mask_extraction_dir)
    output_dir = Path(output_dir) if output_dir is not None else mask_extraction_dir.parent / 'qc'
    alternatives_dir = output_dir / "overlays" / "alternatives"
    alternatives_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating alternative overlays for {participant_id}")

    # Find required files
    ica_dir = mask_extraction_dir / "01_ica"
    masks_dir = mask_extraction_dir / "03_masks"

    # Find components file
    components_files = sorted(glob(str(ica_dir / "*_components.nii.gz")))
    if not components_files:
        logger.warning("No ICA components file found")
        return {}
    components_file = Path(components_files[0])

    # Find BOLD reference
    preprocessing_dir = mask_extraction_dir.parent / "preprocessing"
    if preprocessing_dir.exists():
        ref_files = sorted(glob(str(preprocessing_dir / "*boldref.nii.gz")))
    else:
        ref_files = sorted(glob(str(mask_extraction_dir.parent / "**/*boldref.nii.gz"), recursive=True))

    if not ref_files:
        logger.warning("No BOLD reference found")
        return {}
    ref_file = Path(ref_files[0])

    # Load correlation matrix
    corr_file = masks_dir / "Yeo_IC_spatial_correlation.csv"
    if not corr_file.exists():
        logger.warning("No correlation matrix found")
        return {}
    corr_df = pd.read_csv(corr_file)

    # Find Yeo atlas in subject space for slice selection
    yeo2sub_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "Yeo_to_*.nii.gz")))
    yeo2sub = Path(yeo2sub_files[0]) if yeo2sub_files else None

    # Determine Yeo network count
    if n_yeo is not None:
        n_networks = n_yeo
    else:
        # Infer from network names - check if any names match 17-network labels
        labels_17 = get_yeo_labels(17)
        if any(name in labels_17 for name in network_names):
            n_networks = 17
        else:
            n_networks = 7
    labels = get_yeo_labels(n_networks)

    # Load ICA components
    components_img = load(components_file)
    components_data = components_img.get_fdata()
    n_components = components_data.shape[3]

    result: Dict[str, Dict[int, Path]] = {}

    for network in network_names:
        if network not in corr_df.columns:
            logger.warning(f"Network {network} not in correlation matrix")
            continue

        result[network] = {}

        # Get top N+1 components for this network (including current best)
        correlations = corr_df[network]
        top_indices = correlations.argsort()[::-1][: n_alternatives + 1]

        for comp_idx in top_indices:
            comp_idx = int(comp_idx)
            corr_val = float(correlations.iloc[comp_idx])

            overlay_png = alternatives_dir / f"{participant_id}_{network}_IC{comp_idx}_overlay.png"

            if not overwrite and overlay_png.exists():
                result[network][comp_idx] = overlay_png
                continue

            logger.info(f"Generating {network} IC{comp_idx} overlay (r={corr_val:.3f})")

            # Extract component and threshold
            ic_data = components_data[..., comp_idx]
            threshold = 2000
            flat = ic_data.ravel()
            mask = np.zeros(flat.shape, dtype=bool)
            mask[np.argpartition(flat, -threshold)[-threshold:]] = True
            mask = mask.reshape(ic_data.shape)
            thresholded = ((ic_data * mask) > 0).astype("uint8")

            # Save temporary mask
            temp_mask = alternatives_dir / f"_temp_{network}_IC{comp_idx}.nii.gz"
            save(Nifti1Image(thresholded, components_img.affine, components_img.header), temp_mask)

            try:
                # Select slices
                atlas_slices = select_slices_from_atlas(
                    atlas_path=yeo2sub if yeo2sub else ref_file,
                    labels=labels,
                    target_label=network,
                    top_n_per_plane=1,
                    mask_path=temp_mask
                )

                # Generate overlay
                plot_mask_overlay_tripanel(
                    mask_path=temp_mask,
                    ref_path=ref_file,
                    out_path=overlay_png,
                    slices=atlas_slices,
                    title=f"{network} IC {comp_idx} (r={corr_val:.2f})"
                )

                result[network][comp_idx] = overlay_png
            finally:
                # Clean up temp file
                if temp_mask.exists():
                    temp_mask.unlink()

    return result


def generate_yeo_network_overlay(
        mask_extraction_dir: Path,
        participant_id: str,
        network_name: str,
        output_dir: Path | None = None,
        n_yeo: int = 17,
        overwrite: bool = False
) -> Path | None:
    """
    Generate overlay image for a Yeo atlas network (fallback option).

    Parameters
    ----------
    mask_extraction_dir : Path
        Root mask extraction directory
    participant_id : str
        Participant identifier
    network_name : str
        Name of the network (e.g., 'DefaultA')
    output_dir : Path, optional
        QC output directory
    n_yeo : int, default=17
        Yeo parcellation (7 or 17)
    overwrite : bool, default=False
        Regenerate overlay even if it exists

    Returns
    -------
    Path or None
        Path to generated overlay, or None if generation failed
    """
    import logging
    from glob import glob
    from nibabel.nifti1 import Nifti1Image
    from nibabel.loadsave import save

    from pineuro.utils import get_yeo_labels

    logger = logging.getLogger(__name__)

    mask_extraction_dir = Path(mask_extraction_dir)
    output_dir = Path(output_dir) if output_dir is not None else mask_extraction_dir.parent / 'qc'
    alternatives_dir = output_dir / "overlays" / "alternatives"
    alternatives_dir.mkdir(parents=True, exist_ok=True)

    overlay_png = alternatives_dir / f"{participant_id}_{network_name}_yeo_overlay.png"

    if not overwrite and overlay_png.exists():
        return overlay_png

    # Find required files
    yeo2sub_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "Yeo_to_*.nii.gz")))
    if not yeo2sub_files:
        logger.warning("No Yeo atlas in subject space found")
        return None
    yeo2sub = Path(yeo2sub_files[0])

    # Find BOLD reference
    preprocessing_dir = mask_extraction_dir.parent / "preprocessing"
    if preprocessing_dir.exists():
        ref_files = sorted(glob(str(preprocessing_dir / "*boldref.nii.gz")))
    else:
        ref_files = sorted(glob(str(mask_extraction_dir.parent / "**/*boldref.nii.gz"), recursive=True))

    if not ref_files:
        logger.warning("No BOLD reference found")
        return None
    ref_file = Path(ref_files[0])

    logger.info(f"Generating Yeo atlas overlay for {network_name}")

    # Get network label index
    labels = get_yeo_labels(n_networks=n_yeo)
    if network_name not in labels:
        logger.warning(f"Network {network_name} not in Yeo-{n_yeo} labels")
        return None
    network_index = labels.index(network_name) + 1

    # Load atlas and extract network region
    atlas_img = load(yeo2sub)
    atlas_data = atlas_img.get_fdata()
    network_mask = (atlas_data == network_index).astype("uint8")

    # Save temporary mask
    temp_mask = alternatives_dir / f"_temp_{network_name}_yeo.nii.gz"
    save(Nifti1Image(network_mask, atlas_img.affine, atlas_img.header), temp_mask)

    try:
        # Select slices
        atlas_slices = select_slices_from_atlas(
            atlas_path=yeo2sub,
            labels=labels,
            target_label=network_name,
            top_n_per_plane=1,
            mask_path=temp_mask
        )

        n_voxels = int(network_mask.sum())

        # Generate overlay
        plot_mask_overlay_tripanel(
            mask_path=temp_mask,
            ref_path=ref_file,
            out_path=overlay_png,
            slices=atlas_slices,
            title=f"{network_name} Yeo Atlas ({n_voxels} voxels)"
        )

        return overlay_png
    finally:
        # Clean up temp file
        if temp_mask.exists():
            temp_mask.unlink()


def generate_atlas_region_overlay(
        atlas_subject_path: Path,
        ref_path: Path,
        output_dir: Path,
        participant_id: str,
        atlas_name: str,
        overwrite: bool = False,
) -> Path | None:
    """Generate overlay of an atlas region (e.g. Harvard-Oxford) in subject space.

    Used by the task mask review dialog to show the anatomical constraint
    region alongside the extracted cluster mask — analogous to the Yeo atlas
    overlay in the ICA review dialog.

    Parameters
    ----------
    atlas_subject_path : Path
        Atlas binary mask already warped to subject space (e.g.
        ``03_masks/sub-XXX_atlas_harvard_oxford_stg.nii.gz``).
    ref_path : Path
        BOLD reference in subject space.
    output_dir : Path
        QC output directory (overlay saved into ``overlays/subject/``).
    participant_id : str
        Participant identifier.
    atlas_name : str
        Atlas name used in the filename (e.g. ``"harvard_oxford_stg"``).
    overwrite : bool
        Regenerate even if the overlay already exists.

    Returns
    -------
    Path or None
        Path to the generated PNG, or None on failure.
    """
    import logging

    logger = logging.getLogger(__name__)

    overlays_dir = Path(output_dir) / "overlays" / "subject"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    overlay_png = overlays_dir / f"{participant_id}_atlas_{atlas_name}_overlay.png"

    if not overwrite and overlay_png.exists():
        return overlay_png

    atlas_subject_path = Path(atlas_subject_path)
    ref_path = Path(ref_path)

    if not atlas_subject_path.exists():
        logger.warning(f"Atlas in subject space not found: {atlas_subject_path}")
        return None
    if not ref_path.exists():
        logger.warning(f"BOLD reference not found: {ref_path}")
        return None

    logger.info(f"Generating atlas region overlay for {atlas_name}")

    try:
        atlas_img = load(atlas_subject_path)
        n_voxels = int((atlas_img.get_fdata() > 0).sum())

        atlas_slices = select_slices_from_atlas(
            atlas_path=ref_path,
            labels=[],
            target_label="",
            top_n_per_plane=1,
            mask_path=atlas_subject_path,
        )

        plot_mask_overlay_tripanel(
            mask_path=atlas_subject_path,
            ref_path=ref_path,
            out_path=overlay_png,
            slices=atlas_slices,
            title=f"Atlas: {atlas_name} ({n_voxels} voxels)",
        )
        return overlay_png
    except Exception as ex:
        logger.warning(f"Failed to generate atlas overlay: {ex}")
        return None


def generate_mask_overlay(
        mask_path: Path,
        ref_path: Path,
        output_dir: Path,
        network_name: str,
        atlas_path: Path | None = None,
        n_yeo: int = 17,
        overwrite: bool = False
) -> Path | None:
    """
    Generate a simple overlay image for any mask file.

    This is a lightweight utility for generating mask overlay previews,
    particularly useful for combined ICA component masks.

    Parameters
    ----------
    mask_path : Path
        Path to the mask file (thresholded binary NIfTI)
    ref_path : Path
        Path to the BOLD reference for background
    output_dir : Path
        Directory to save the overlay image
    network_name : str
        Name for the overlay file (used in filename and title)
    atlas_path : Path, optional
        Path to Yeo atlas in subject space for slice selection.
        If not provided, slices are selected from the mask itself.
    n_yeo : int, default=17
        Yeo parcellation (7 or 17) for labels
    overwrite : bool, default=False
        Regenerate overlay even if it exists

    Returns
    -------
    Path or None
        Path to generated overlay, or None if generation failed
    """
    import logging

    from pineuro.utils import get_yeo_labels

    logger = logging.getLogger(__name__)

    mask_path = Path(mask_path)
    ref_path = Path(ref_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    overlay_png = output_dir / f"{network_name}_overlay.png"

    if not overwrite and overlay_png.exists():
        return overlay_png

    if not mask_path.exists():
        logger.warning(f"Mask file not found: {mask_path}")
        return None

    if not ref_path.exists():
        logger.warning(f"Reference file not found: {ref_path}")
        return None

    logger.info(f"Generating mask overlay for {network_name}")

    try:
        # Load mask to get voxel count
        mask_img = load(mask_path)
        mask_data = mask_img.get_fdata()
        n_voxels = int((mask_data > 0).sum())

        # Select slices based on mask or atlas
        labels = get_yeo_labels(n_networks=n_yeo)

        # Extract base network name (remove _combined suffix if present)
        base_network = network_name.replace("_combined", "").replace("_thr", "")
        target_label = base_network if base_network in labels else labels[0]

        if atlas_path and atlas_path.exists():
            slices = select_slices_from_atlas(
                atlas_path=atlas_path,
                labels=labels,
                target_label=target_label,
                top_n_per_plane=1,
                mask_path=mask_path
            )
        else:
            # Select slices directly from mask
            slices = select_slices_from_atlas(
                atlas_path=mask_path,  # Use mask as "atlas" - will be ignored
                labels=labels,
                target_label=target_label,
                top_n_per_plane=1,
                mask_path=mask_path  # This will be used for selection
            )

        # Generate overlay
        plot_mask_overlay_tripanel(
            mask_path=mask_path,
            ref_path=ref_path,
            out_path=overlay_png,
            slices=slices,
            title=f"{network_name} ({n_voxels} voxels)"
        )

        return overlay_png

    except Exception as e:
        logger.error(f"Failed to generate overlay for {network_name}: {e}")
        return None


def generate_hemisphere_overlays(
        mask_extraction_dir: Path,
        participant_id: str,
        hemisphere_network: str,
        output_dir: Path | None = None,
        n_yeo: int = 17,
        overwrite: bool = False
) -> Dict[str, Path]:
    """
    Generate overlay images for hemisphere-split masks (for sham verification).

    Creates overlay images for left and right hemisphere masks from a split network,
    enabling visual verification in the mask selection dialog.

    Parameters
    ----------
    mask_extraction_dir : Path
        Root mask extraction directory (contains 01_ica/, 02_mnireg/, 03_masks/)
    participant_id : str
        Participant identifier
    hemisphere_network : str
        Name of the network that was split (e.g., 'SomMotA')
    output_dir : Path, optional
        QC output directory. Defaults to mask_extraction_dir.parent / 'qc'
    n_yeo : int, default=17
        Yeo parcellation (7 or 17 networks)
    overwrite : bool, default=False
        Regenerate overlays even if they exist

    Returns
    -------
    dict
        Dictionary mapping hemisphere ('left', 'right') to overlay path.
        Example: {'left': Path('...left_overlay.png'), 'right': Path('...right_overlay.png')}
    """
    import logging
    from glob import glob

    from pineuro.utils import get_yeo_labels

    logger = logging.getLogger(__name__)

    mask_extraction_dir = Path(mask_extraction_dir)
    output_dir = Path(output_dir) if output_dir is not None else mask_extraction_dir.parent / 'qc'
    overlays_dir = output_dir / "overlays" / "hemisphere_split"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = mask_extraction_dir / "03_masks"

    # Find required files
    left_mask = masks_dir / f"{participant_id}_{hemisphere_network}_left_thr.nii.gz"
    right_mask = masks_dir / f"{participant_id}_{hemisphere_network}_right_thr.nii.gz"

    if not left_mask.exists() or not right_mask.exists():
        logger.warning(
            f"Hemisphere-split masks not found for {hemisphere_network}. "
            f"Expected: {left_mask.name}, {right_mask.name}"
        )
        return {}

    # Find BOLD reference
    preprocessing_dir = mask_extraction_dir.parent / "preprocessing"
    if preprocessing_dir.exists():
        ref_files = sorted(glob(str(preprocessing_dir / "*boldref.nii.gz")))
    else:
        ref_files = sorted(glob(str(mask_extraction_dir.parent / "**/*boldref.nii.gz"), recursive=True))

    if not ref_files:
        logger.warning("No BOLD reference found for hemisphere overlays")
        return {}
    ref_file = Path(ref_files[0])

    # Find Yeo atlas in subject space for slice selection
    yeo2sub_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "Yeo_to_*.nii.gz")))
    yeo2sub = Path(yeo2sub_files[0]) if yeo2sub_files else None

    labels = get_yeo_labels(n_yeo)

    result: Dict[str, Path] = {}

    for hemisphere, mask_path in [("left", left_mask), ("right", right_mask)]:
        overlay_png = overlays_dir / f"{participant_id}_{hemisphere_network}_{hemisphere}_overlay.png"

        if not overwrite and overlay_png.exists():
            result[hemisphere] = overlay_png
            continue

        logger.info(f"Generating {hemisphere_network} {hemisphere} hemisphere overlay")

        # Get voxel count for title
        mask_img = load(mask_path)
        mask_data = mask_img.get_fdata()
        n_voxels = int((mask_data > 0).sum())

        try:
            # Select slices based on the mask itself
            atlas_slices = select_slices_from_atlas(
                atlas_path=yeo2sub if yeo2sub else ref_file,
                labels=labels,
                target_label=hemisphere_network,
                top_n_per_plane=1,
                mask_path=mask_path
            )

            # Generate overlay
            plot_mask_overlay_tripanel(
                mask_path=mask_path,
                ref_path=ref_file,
                out_path=overlay_png,
                slices=atlas_slices,
                title=f"{hemisphere_network} {hemisphere.upper()} ({n_voxels} voxels)"
            )

            result[hemisphere] = overlay_png
        except Exception as ex:
            logger.warning(f"Failed to generate {hemisphere} hemisphere overlay: {ex}")

    return result


# ============================================================================
# SECTION 4: Feedback QC
# ============================================================================

def plot_activation_timeseries(
        feedback_log_path: Path,
        out_path: Path,
        n_baseline: int | None = None,
        title: str | None = None
) -> None:
    """
    Plot per-mask activation timeseries from feedback log.

    Parameters
    ----------
    feedback_log_path : Path
        Path to feedback log TSV file
    out_path : Path
        Path to save the PNG
    n_baseline : int, optional
        Number of baseline volumes (will be shaded differently)
    title : str, optional
        Figure title
    """
    # Load feedback log
    df = pd.read_csv(feedback_log_path, sep='\t')

    # Get mask columns (all columns except metadata)
    meta_cols = ['volume', 'time', 'is_baseline', 'is_outlier', 'fd', 'condition']
    mask_cols = [col for col in df.columns if col not in meta_cols]

    # Ensure numeric columns are properly typed (handles edge cases where pandas infers object dtype)
    numeric_cols = ['volume', 'time', 'fd'] + mask_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    n_volumes = len(df)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Plot each mask's activation
    for mask_name in mask_cols:
        ax.plot(df['volume'], df[mask_name], linewidth=1.5, label=mask_name, alpha=0.8)

    # Shade baseline period
    if n_baseline is not None and n_baseline > 0:
        ax.axvspan(-0.5, n_baseline - 0.5, color='gray', alpha=0.1, label='Baseline')
        ax.axvline(n_baseline - 0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    # Add zero reference line
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)

    # Labels and legend
    ax.set_xlabel('Volume Index', fontweight='bold')
    ax.set_ylabel('Activation (normalized)', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(-0.5, n_volumes - 0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    else:
        ax.set_title('Mask Activation Timeseries', fontsize=12, fontweight='bold', pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_design_matrix(
        design_path: Path,
        out_path: Path,
        title: str | None = None
) -> None:
    """
    Plot design matrix heatmap showing regressors × volumes.

    Each regressor is normalized to [0, 1] range for visualization to prevent
    regressors with large magnitudes (e.g., drift) from dominating the color scale.

    Parameters
    ----------
    design_path : Path
        Path to design matrix TSV file
    out_path : Path
        Path to save the PNG
    title : str, optional
        Figure title
    """
    # Load design matrix
    df = pd.read_csv(design_path, sep='\t')

    # Ensure all columns are numeric (handles edge cases where pandas infers object dtype)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Remove volume_idx if present
    if 'volume_idx' in df.columns:
        df = df.drop(columns=['volume_idx'])

    # Normalize each regressor to [0, 1] to prevent drift from dominating color scale
    # This allows all regressors to be visible regardless of their magnitude
    df_normalized = df.copy()
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            # Constant column (e.g., all zeros)
            df_normalized[col] = 0.0

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Plot heatmap with normalized data
    im = ax.imshow(df_normalized.T.values, aspect='auto', cmap='viridis',
                   vmin=0, vmax=1, interpolation='nearest')

    # Set ticks and labels
    ax.set_xlabel('Volume Index', fontweight='bold')
    ax.set_ylabel('Regressor', fontweight='bold')
    ax.set_yticks(range(len(df_normalized.columns)))
    ax.set_yticklabels(df_normalized.columns, fontsize=9)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Regressor Value [0-1]', rotation=270, labelpad=20, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    else:
        ax.set_title('Design Matrix (normalized per regressor)', fontsize=12, fontweight='bold', pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_fd_timeseries(
        feedback_log_path: Path,
        out_path: Path,
        n_baseline: int | None = None,
        threshold: float = 0.5,
        title: str | None = None
) -> None:
    """
    Plot framewise displacement timeseries from feedback log.

    Parameters
    ----------
    feedback_log_path : Path
        Path to feedback log TSV file
    out_path : Path
        Path to save the PNG
    n_baseline : int, optional
        Number of baseline volumes (will be shaded differently)
    threshold : float, default=0.5
        FD threshold in mm for highlighting excessive motion
    title : str, optional
        Figure title
    """
    # Load feedback log
    df = pd.read_csv(feedback_log_path, sep='\t')

    # Check if FD column exists
    if 'fd' not in df.columns:
        logger.warning("No 'fd' column in feedback log, skipping FD plot")
        return

    # Ensure numeric
    df['fd'] = pd.to_numeric(df['fd'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    fd = df['fd'].values
    volumes = df['volume'].values
    n_volumes = len(fd)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    ax.plot(volumes, fd, linewidth=1.5, color='#0173B2', label='FD')
    ax.axhline(threshold, color='#d62728', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold} mm)')

    # Shade baseline period
    if n_baseline is not None and n_baseline > 0:
        ax.axvspan(-0.5, n_baseline - 0.5, color='gray', alpha=0.1, label='Baseline')
        ax.axvline(n_baseline - 0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    # Shade regions above threshold
    above_thresh = fd > threshold
    if above_thresh.any():
        for i in range(n_volumes):
            if above_thresh[i]:
                ax.axvspan(volumes[i] - 0.5, volumes[i] + 0.5, color='#d62728', alpha=0.2)

    # Summary statistics
    mean_fd = np.nanmean(fd)
    pct_above = (np.sum(above_thresh) / n_volumes) * 100
    ax.text(0.02, 0.98, f'Mean FD: {mean_fd:.3f} mm\n{pct_above:.1f}% volumes > {threshold} mm',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Volume Index', fontweight='bold')
    ax.set_ylabel('Framewise Displacement (mm)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, n_volumes - 0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    else:
        ax.set_title('Framewise Displacement', fontsize=12, fontweight='bold', pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def generate_feedback_report(
        feedback_dir: Path,
        subject: str,
        task: str,
        run: int,
        output_dir: Path | None = None,
        overwrite: bool = False,
        open_browser: bool = False,
        expected_volumes: int | None = None,
        acq: str | None = None,
        fd_threshold: float = 0.5,
) -> Path:
    """
    Generate comprehensive QC report for feedback session outputs.

    Automatically discovers feedback logs, design matrices, and generates
    quality control visualizations.

    Parameters
    ----------
    feedback_dir : Path
        Root feedback directory (contains feedback logs and design files)
    subject : str
        Subject identifier
    task : str
        Task name (e.g., 'nback')
    run : int
        Run number
    output_dir : Path, optional
        QC output directory. Defaults to feedback_dir / 'qc'
    overwrite : bool, default=False
        Regenerate all plots even if they exist
    open_browser : bool, default=False
        Automatically open the HTML report in the default web browser
    expected_volumes : int, optional
        Expected number of volumes for a complete run. If provided, used to
        determine COMPLETE vs PARTIAL status. If None, status shows as PARTIAL
        with a note that target is unknown.
    acq : str, optional
        Acquisition letter (e.g., 'a', 'b'). If provided, uses BIDS acq entity
        in file naming pattern. If None, uses legacy naming without acq.
    fd_threshold : float, default=0.5
        Framewise displacement threshold in mm for highlighting excessive motion
        in the FD timeseries plot.

    Returns
    -------
    Path
        Path to generated HTML report

    Example
    -------
    >>> from pineuro import qc
    >>> report = qc.generate_feedback_report(
    ...     feedback_dir=Path('/data/sub-01/feedback'),
    ...     subject='sub-01',
    ...     task='nback',
    ...     run=1,
    ...     open_browser=True
    ... )
    >>> print(f"QC report: {report}")
    """
    import logging
    from glob import glob

    logger = logging.getLogger(__name__)

    feedback_dir = Path(feedback_dir)
    output_dir = Path(output_dir) if output_dir is not None else feedback_dir / 'qc'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating feedback QC report for {subject}, task={task}, run={run}, acq={acq}")

    # Auto-discover outputs - use acq-based naming if acq provided
    if acq is not None:
        # Format acq as 2-digit number if int
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        feedback_log_pattern = f"{subject}_task-{task}_acq-{acq_str}_run-{run:02d}_feedbacklog.tsv"
        design_pattern = f"{subject}_task-{task}_acq-{acq_str}_run-{run:02d}_design.tsv"
    else:
        feedback_log_pattern = f"{subject}_task-{task}_run-{run:02d}_feedbacklog.tsv"
        design_pattern = f"{subject}_task-{task}_run-{run:02d}_design.tsv"

    feedback_log = feedback_dir / feedback_log_pattern
    design_file = feedback_dir / design_pattern

    # Build QC sections
    sections = []

    # Summary box
    summary_data = {
        'Subject': subject,
        'Task': task,
        'Run': f'{run:02d}',
        'Acquisition': f'{acq:02d}' if isinstance(acq, int) else (acq if acq else 'N/A'),
        'Feedback log': str(feedback_log.exists()),
        'Design matrix': str(design_file.exists())
    }

    # Extract additional info from feedback log if available
    n_volumes_recorded = 0
    if feedback_log.exists():
        df = pd.read_csv(feedback_log, sep='\t')
        meta_cols = ['volume', 'time', 'is_baseline', 'is_outlier', 'fd', 'condition']
        mask_cols = [col for col in df.columns if col not in meta_cols]
        n_volumes_recorded = len(df)
        summary_data['Volumes recorded'] = str(n_volumes_recorded)
        summary_data['Masks'] = str(len(mask_cols))

        # Determine status based on expected vs recorded volumes
        if expected_volumes is not None:
            summary_data['Expected volumes'] = str(expected_volumes)
            if n_volumes_recorded >= expected_volumes:
                summary_data['Status'] = 'COMPLETE'
            else:
                pct = (n_volumes_recorded / expected_volumes) * 100
                summary_data['Status'] = f'PARTIAL ({n_volumes_recorded}/{expected_volumes} = {pct:.1f}%)'
        else:
            # No expected_volumes provided - check for sequential volumes
            if 'volume' in df.columns and len(df) > 0:
                last_vol = df['volume'].iloc[-1]
                if last_vol == n_volumes_recorded:
                    summary_data['Status'] = 'PARTIAL (target volume count unknown)'
                else:
                    summary_data['Status'] = 'PARTIAL (missing volumes detected)'

        # Compute FD statistics
        if 'fd' in df.columns:
            fd_values = pd.to_numeric(df['fd'], errors='coerce').values
            fd_mean = np.nanmean(fd_values)
            fd_max = np.nanmax(fd_values)
            fd_above = np.sum(fd_values > fd_threshold)
            fd_pct = (fd_above / len(fd_values)) * 100 if len(fd_values) > 0 else 0
            summary_data['Mean FD'] = f'{fd_mean:.3f} mm'
            summary_data['Max FD'] = f'{fd_max:.3f} mm'
            summary_data[f'Volumes > {fd_threshold}mm'] = f'{fd_above} ({fd_pct:.1f}%)'
            # Motion quality rating
            if fd_pct < 5:
                summary_data['Motion quality'] = 'Good'
            elif fd_pct < 15:
                summary_data['Motion quality'] = 'Moderate'
            else:
                summary_data['Motion quality'] = 'High motion'
    else:
        summary_data['Status'] = 'NO DATA'

    # Try to find and parse PsychoPy log for hit counts
    if acq is not None:
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        psychopy_pattern = f"{subject}_task-{task}_acq-{acq_str}_run-{run:02d}_psychopylog.tsv"
    else:
        psychopy_pattern = f"{subject}_task-{task}_run-{run:02d}_psychopylog.tsv"
    psychopy_log = feedback_dir / psychopy_pattern

    if psychopy_log.exists():
        try:
            psych_df = pd.read_csv(psychopy_log, sep='\t')
            hit_cols = [col for col in psych_df.columns if col.endswith('_hits')]
            total_hits = 0
            hit_strs = []
            for col in hit_cols:
                network_name = col.replace('_hits', '')
                hits = int(psych_df[col].max())
                total_hits += hits
                hit_strs.append(f'{network_name}: {hits}')
            if hit_strs:
                summary_data['Network hits'] = ', '.join(hit_strs)
                summary_data['Total hits'] = str(total_hits)
        except Exception as e:
            logger.warning(f"Failed to parse PsychoPy log for hits: {e}")

    sections.append({'type': 'summary', 'content': summary_data})

    # Determine n_baseline from feedback log or design matrix
    n_baseline = None
    if feedback_log.exists() and 'is_baseline' in df.columns:
        # Count volumes where is_baseline == True
        n_baseline = int((df['is_baseline'] == True).sum())
    elif design_file.exists():
        design_df = pd.read_csv(design_file, sep='\t')
        # Check if there's a 'baseline' or 'intercept' column
        if 'baseline' in design_df.columns:
            # Count volumes where baseline == 1
            n_baseline = int((design_df['baseline'] == 1).sum())
        elif feedback_log.exists():
            # Estimate from first ~20% of volumes
            n_baseline = max(1, int(0.2 * len(df)))

    # Activation timeseries
    if feedback_log.exists():
        sections.append({'type': 'heading', 'content': 'Activation Timeseries'})

        timeseries_out = output_dir / "activation_timeseries.png"
        if overwrite or not timeseries_out.exists():
            logger.info("Generating activation timeseries")
            plot_activation_timeseries(
                feedback_log_path=feedback_log,
                out_path=timeseries_out,
                n_baseline=n_baseline,
                title=f"Activation Timeseries - {subject} (task: {task}, run: {run})"
            )

        sections.append({
            'type': 'single_image',
            'path': timeseries_out.relative_to(output_dir).as_posix(),
            'alt': 'Activation timeseries'
        })

    # FD timeseries
    if feedback_log.exists():
        sections.append({'type': 'heading', 'content': 'Framewise Displacement'})

        fd_out = output_dir / "fd_timeseries.png"
        if overwrite or not fd_out.exists():
            logger.info("Generating FD timeseries")
            plot_fd_timeseries(
                feedback_log_path=feedback_log,
                out_path=fd_out,
                n_baseline=n_baseline,
                threshold=fd_threshold,
                title=f"Framewise Displacement - {subject} (task: {task}, run: {run})"
            )

        if fd_out.exists():
            sections.append({
                'type': 'single_image',
                'path': fd_out.relative_to(output_dir).as_posix(),
                'alt': 'Framewise displacement timeseries'
            })


    # Design matrix
    if design_file.exists():
        sections.append({'type': 'heading', 'content': 'Design Matrix'})

        design_out = output_dir / "design_matrix.png"
        if overwrite or not design_out.exists():
            logger.info("Generating design matrix visualization")
            plot_design_matrix(
                design_path=design_file,
                out_path=design_out,
                title=f"Design Matrix - {subject} (task: {task}, run: {run})"
            )

        sections.append({
            'type': 'single_image',
            'path': design_out.relative_to(output_dir).as_posix(),
            'alt': 'Design matrix'
        })

    # Generate HTML report with acq entity if provided
    if acq is not None:
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        html_report = output_dir / f"{subject}_task-{task}_acq-{acq_str}_run-{run:02d}_feedback_qc.html"
        stage_str = f"Feedback (task: {task}, run: {run}, acq: {acq_str})"
    else:
        html_report = output_dir / f"{subject}_task-{task}_run-{run:02d}_feedback_qc.html"
        stage_str = f"Feedback (task: {task}, run: {run})"
    html = _generate_html_template(
        title=f"Feedback QC - {subject}",
        participant_id=subject,
        stage=stage_str,
        sections=sections
    )
    html_report.write_text(html)

    logger.info(f"Feedback QC report saved to: {html_report}")

    if open_browser:
        _open_html_in_browser(html_report)

    return html_report


# ============================================================================
# SECTION 5: HTML Generation
# ============================================================================

def _generate_html_template(
        title: str,
        participant_id: str,
        stage: str,
        sections: list[dict]
) -> str:
    """Generate a professional HTML QC report with consistent styling.

    Creates a complete, self-contained HTML document with embedded CSS
    for professional-quality QC reports. The resulting HTML is static
    (no JavaScript) and works offline for easy sharing.

    Parameters
    ----------
    title : str
        Page title displayed in browser tab and as main heading.
    participant_id : str
        Participant identifier displayed in the header.
    stage : str
        Pipeline stage name (e.g., "Preprocessing", "Mask Extraction",
        "Feedback (task: nback, run: 1)") for context.
    sections : list of dict
        Content sections to render. Each dict must have a 'type' key
        and appropriate content:

        - ``{'type': 'heading', 'content': 'Section Title'}``
          Renders as an <h2> heading.

        - ``{'type': 'summary', 'content': {'Key': 'Value', ...}}``
          Renders as a styled info box with key-value pairs.

        - ``{'type': 'grid', 'content': [{'title': ..., 'path': ..., 'alt': ...}, ...]}``
          Renders images in a responsive grid layout.

        - ``{'type': 'single_image', 'path': 'relative/path.png', 'alt': '...'}``
          Renders a single full-width image.

        - ``{'type': 'table', 'headers': [...], 'rows': [[...], ...]}``
          Renders a styled data table.

    Returns
    -------
    str
        Complete HTML document as a string, ready to write to a file.

    Notes
    -----
    **Styling Features:**
    - Modern sans-serif typography (system fonts for fast loading)
    - Responsive grid layout that adapts to screen width
    - Colorblind-safe blue accent color (#0173B2)
    - Clean, professional appearance suitable for publications
    - Print-friendly styling

    **Image Paths:**
    All image paths in sections should be relative to the HTML file's
    location. Use Path.relative_to() to compute paths correctly.

    Examples
    --------
    >>> sections = [
    ...     {'type': 'summary', 'content': {'Subject': 'sub-01', 'Task': 'rest'}},
    ...     {'type': 'heading', 'content': 'Motion Parameters'},
    ...     {'type': 'single_image', 'path': 'motion.png', 'alt': 'Motion plot'}
    ... ]
    >>> html = _generate_html_template(
    ...     title='Preprocessing QC - sub-01',
    ...     participant_id='sub-01',
    ...     stage='Preprocessing',
    ...     sections=sections
    ... )
    >>> Path('report.html').write_text(html)
    """
    from datetime import datetime

    # Build HTML sections
    html_sections = []
    for section in sections:
        section_type = section.get('type')

        if section_type == 'heading':
            html_sections.append(f"<h2>{section['content']}</h2>")

        elif section_type == 'summary':
            html_sections.append('<div class="summary-box">')
            for key, value in section['content'].items():
                html_sections.append(f"<p><strong>{key}:</strong> {value}</p>")
            html_sections.append('</div>')

        elif section_type == 'grid':
            html_sections.append('<div class="grid">')
            for item in section['content']:
                html_sections.append('<div class="plot-container">')
                if 'title' in item:
                    html_sections.append(f"<h3>{item['title']}</h3>")
                html_sections.append(f'<img src="{item["path"]}" alt="{item.get("alt", "")}">')
                html_sections.append('</div>')
            html_sections.append('</div>')

        elif section_type == 'single_image':
            html_sections.append(f'<img src="{section["path"]}" alt="{section.get("alt", "")}">')

        elif section_type == 'table':
            html_sections.append('<table>')
            html_sections.append('<tr>')
            for header in section['headers']:
                html_sections.append(f'<th>{header}</th>')
            html_sections.append('</tr>')
            for row in section['rows']:
                html_sections.append('<tr>')
                for cell in row:
                    html_sections.append(f'<td>{cell}</td>')
                html_sections.append('</tr>')
            html_sections.append('</table>')

    # Complete HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 2rem auto;
            max-width: 1200px;
            padding: 0 2rem;
            color: #333;
            line-height: 1.6;
            background: #fafafa;
        }}
        h1 {{
            color: #0173B2;
            border-bottom: 3px solid #0173B2;
            padding-bottom: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        h2 {{
            color: #555;
            margin-top: 2rem;
            border-bottom: 1px solid #ddd;
            padding-bottom: 0.3rem;
        }}
        h3 {{
            color: #666;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }}
        .plot-container {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #0173B2;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .summary-box {{
            background: #e3f2fd;
            padding: 1rem;
            border-left: 4px solid #0173B2;
            margin: 1rem 0;
            border-radius: 4px;
        }}
        .summary-box p {{
            margin: 0.5rem 0;
        }}
        .footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><strong>Participant:</strong> {participant_id} | <strong>Stage:</strong> {stage}</p>

    {chr(10).join(html_sections)}

    <div class="footer">
        <p>Generated by pineuro on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
    return html


def write_html_report(
        out_path: Path,
        participant_id: str,
        overlays_by_space: Iterable[Tuple[str, Iterable[Tuple[str, Path]]]],
        heatmap_png: Path | None,
        summaries: Iterable[MaskSummary]
) -> None:
    """Create an HTML QC report for mask extraction (legacy API).

    This is the original mask extraction report generator, kept for
    backwards compatibility. For new code, prefer using
    generate_mask_extraction_report() which auto-discovers outputs.

    The report includes:
    1. ICA × Yeo network correlation heatmap (if provided)
    2. Mask overlays organized by coordinate space
    3. Summary table with mask metrics

    Parameters
    ----------
    out_path : Path
        Output path for the HTML report file.
    participant_id : str
        Participant identifier displayed in report header.
    overlays_by_space : iterable of (space_name, overlays)
        Mask overlay images organized by coordinate space.
        Each item is a tuple of (space_name, overlay_list) where
        overlay_list contains (mask_name, png_path) tuples.
        Example: [("BOLD reference", [("DefaultA", Path("..."))])]
    heatmap_png : Path or None
        Path to the ICA × Yeo correlation heatmap PNG.
        If None, the correlation section is omitted.
    summaries : iterable of MaskSummary
        Mask metrics for the summary table. One MaskSummary per
        extracted network mask.

    See Also
    --------
    generate_mask_extraction_report : Preferred high-level API
    MaskSummary : Dataclass for mask metrics

    Examples
    --------
    >>> summaries = [
    ...     MaskSummary('DefaultA', 2000, 8.0, (45, 60, 35), 3, 0.42),
    ...     MaskSummary('ContA', 1800, 7.2, (50, 40, 45), 7, 0.38),
    ... ]
    >>> overlays = [
    ...     ("BOLD reference", [
    ...         ("DefaultA", Path("overlays/DefaultA.png")),
    ...         ("ContA", Path("overlays/ContA.png")),
    ...     ])
    ... ]
    >>> write_html_report(
    ...     out_path=Path("qc_report.html"),
    ...     participant_id="sub-01",
    ...     overlays_by_space=overlays,
    ...     heatmap_png=Path("heatmap.png"),
    ...     summaries=summaries
    ... )
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _rel(p: Path) -> str:
        try:
            return p.relative_to(out_path.parent).as_posix()
        except ValueError:
            return p.as_posix()

    # Build sections using new template
    sections = []

    # ICA × Yeo correlation heatmap
    if heatmap_png and heatmap_png.exists():
        sections.append({'type': 'heading', 'content': 'ICA × Yeo Correlation'})
        sections.append({'type': 'single_image', 'path': _rel(heatmap_png), 'alt': 'Correlation heatmap'})

    # Mask overlays by space
    for space_name, overlays in overlays_by_space:
        sections.append({'type': 'heading', 'content': f'Mask Overlays ({space_name})'})
        grid_items = []
        for name, img_path in overlays:
            grid_items.append({
                'title': name,
                'path': _rel(img_path),
                'alt': f'Mask overlay: {name} ({space_name})'
            })
        sections.append({'type': 'grid', 'content': grid_items})

    # Mask summary table
    sections.append({'type': 'heading', 'content': 'Mask Summary'})
    table_rows = []
    for s in summaries:
        table_rows.append([
            s.name,
            str(s.voxels),
            f"{s.volume_ml:.2f}",
            str(s.centroid_ijk),
            str(s.chosen_component),
            f"{s.correlation:.3f}"
        ])
    sections.append({
        'type': 'table',
        'headers': ['Name', 'Voxels', 'Volume (mL)', 'Centroid (i,j,k)', 'Chosen component', 'Correlation (r)'],
        'rows': table_rows
    })

    # Generate HTML using template
    html = _generate_html_template(
        title=f"pineuro QC - {participant_id}",
        participant_id=participant_id,
        stage="Mask Extraction",
        sections=sections
    )

    out_path.write_text(html)
