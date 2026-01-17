"""
Quality control utilities for mask extraction outputs.

Summary
-------
Generate lightweight QC artifacts for the mask extraction pipeline:
- Mask overlays on participant BOLD reference (and other spaces when provided)
- ICA component × Yeo network correlation heatmap
- Mask summaries (voxels, volume, centroid, chosen component)
- Simple per-participant HTML report linking all artifacts

Functions
---------
- plot_mask_overlay: Save slice overlays of a mask on reference
- plot_corr_heatmap: Visualize component × network correlations
- summarize_mask: Compute basic mask metrics
- write_html_report: Assemble an HTML QC page for a participant

Dependencies
------------
matplotlib, numpy, pandas, nibabel

Notes
-----
These helpers are intentionally lightweight and rely only on in-tree
dependencies. Images are written as PNGs; HTML is pure static to allow quick
local inspection before running feedback.
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
    """Apply consistent, publication-quality style to all plots."""
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
        Defaults to 0 (primary screen).
    """
    import logging
    import os
    import platform
    import subprocess

    logger = logging.getLogger(__name__)

    if not html_path.exists():
        logger.warning(f"Cannot open HTML: file does not exist: {html_path}")
        return

    # Use primary screen if not specified
    if screen is None:
        screen = 0

    try:
        # Convert to file:// URL for cross-platform compatibility
        file_url = html_path.resolve().as_uri()

        # Open browser first (non-blocking)
        webbrowser.open(file_url)
        logger.info(f"Opened QC report in browser: {html_path}")

        # On Linux, attempt to position browser window on specified screen
        if platform.system() == 'Linux' and screen != 0:
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
            logger.debug(f"Window positioning not supported on {platform.system()}")

    except Exception as e:
        logger.warning(f"Failed to open HTML in browser: {e}")



@dataclass
class MaskSummary:
    """Container for per-mask QC metrics."""
    name: str
    voxels: int
    volume_ml: float
    centroid_ijk: Tuple[float, float, float]
    chosen_component: int
    correlation: float = 0.0  # Spatial correlation with Yeo network


def _pick_slices(n_slices: int, count: int = 6) -> list[int]:
    """Evenly sample slice indices across the axial dimension."""
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
    """
    Choose slice indices based on the Yeo atlas presence for a target network.

    Parameters
    ----------
    atlas_path : Path
        Path to the Yeo atlas already resampled to participant space
    labels : sequence of str
        Atlas labels in order (index+1 matches NIfTI values)
    target_label : str
        Network label to select slices for
    top_n_per_plane : int, default=1
        Number of highest-coverage slices per plane to include
    mask_path : Path, optional
        If provided, use this mask for slice selection instead of atlas
        (useful when displaying participant-specific masks)

    Returns
    -------
    dict
        Slice indices keyed by plane: {"axial": [...], "coronal": [...], "sagittal": [...]}
    """
    # If mask provided, use it for slice selection; otherwise use atlas
    if mask_path is not None:
        slice_img = load(mask_path)
        slice_data = slice_img.get_fdata()
        mask = slice_data > 0  # Binary mask
    else:
        atlas_img = load(atlas_path)
        atlas_data = atlas_img.get_fdata().astype(int)

        if target_label not in labels:
            return {
                "axial": _pick_slices(atlas_data.shape[2], 1),
                "coronal": [atlas_data.shape[1] // 2],
                "sagittal": [atlas_data.shape[0] // 2],
            }

        target_idx = labels.index(target_label) + 1
        mask = atlas_data == target_idx

    if not mask.any():
        shape = mask.shape
        return {
            "axial": _pick_slices(shape[2], 1),
            "coronal": [shape[1] // 2],
            "sagittal": [shape[0] // 2],
        }

    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)

    def _top_slices(along: int, top_n: int) -> list[int]:
        axes = tuple(i for i in range(3) if i != along)
        counts = mask.sum(axis=axes)
        order = counts.argsort()[::-1]
        chosen: list[int] = []
        for idx in order[:top_n]:
            if counts[idx] > 0 and 0 <= int(idx) < counts.shape[0]:
                chosen.append(int(idx))
        # Always include centroid slice
        c_idx = int(round(centroid[along]))
        if 0 <= c_idx < counts.shape[0] and c_idx not in chosen:
            chosen.append(c_idx)
        return chosen or [counts.shape[0] // 2]

    return {
        "axial": _top_slices(2, top_n_per_plane),
        "coronal": _top_slices(1, top_n_per_plane),
        "sagittal": _top_slices(0, top_n_per_plane),
    }


def plot_mask_overlay_tripanel(
        mask_path: Path,
        ref_path: Path,
        out_path: Path,
        slices: Dict[str, list[int]],
        title: str | None = None,
        cmap: str = "autumn"
) -> None:
    """
    Save tri-planar overlays (sagittal, coronal, axial) using provided slice indices.

    Parameters
    ----------
    mask_path : Path
        Path to the mask NIfTI file
    ref_path : Path
        Path to the BOLD reference NIfTI file (same space as mask)
    out_path : Path
        Path to save the overlay PNG
    slices : dict
        Slice indices keyed by plane {"axial": [...], "coronal": [...], "sagittal": [...]}
    title : str, optional
        Figure title
    cmap : str, default="autumn"
        Matplotlib colormap for mask fill
    """
    mask_img = load(mask_path)
    ref_img = load(ref_path)

    mask = mask_img.get_fdata()
    ref = ref_img.get_fdata()

    if ref.shape != mask.shape:
        raise ValueError(f"Shape mismatch for overlay: ref {ref.shape} vs mask {mask.shape}")

    # Check orientation for proper display
    import nibabel as nib
    ref_orient = nib.aff2axcodes(ref_img.affine)

    # For coronal/axial: X-axis determines left-right
    x_negative = ref_img.affine[0, 0] < 0  # True for LAS/RAS-, False for RAS/LAS+

    # For sagittal: Y-axis determines anterior-posterior
    # Positive Y = anterior to posterior (P orientation code)
    # Negative Y = posterior to anterior (A orientation code)
    y_negative = ref_img.affine[1, 1] < 0  # True if A in second position

    planes = [("sagittal", 0), ("coronal", 1), ("axial", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

    ref_min, ref_max = np.percentile(ref, [2, 98])

    def _get_slice(data: np.ndarray, plane: str, idx: int) -> np.ndarray:
        if plane == "axial":
            return data[:, :, idx]
        if plane == "coronal":
            return data[:, idx, :]
        if plane == "sagittal":
            return data[idx, :, :]
        raise ValueError(f"Unknown plane: {plane}")

    for ax, (plane, axis_idx) in zip(axes, planes):
        plane_slices = slices.get(plane, [])
        if not plane_slices:
            plane_slices = [_pick_slices(ref.shape[axis_idx], 1)[0]]
        idx = plane_slices[0]

        base = _get_slice(ref, plane, idx)
        overlay = _get_slice(mask, plane, idx)

        # Apply radiological convention based on plane and orientation
        # For sagittal: anterior (frontal lobe) should be on LEFT
        # For coronal/axial: patient's left should be on RIGHT

        if plane == "sagittal":
            # Sagittal slice is Y-Z plane (anterior-posterior × inferior-superior)
            # Y-axis determines anterior-posterior orientation
            # Standard: anterior on left (when Y increases = P to A direction, need flip)
            if y_negative:
                # A orientation (posterior to anterior): anterior already on left
                base = base.T
                overlay = overlay.T
            else:
                # P orientation (anterior to posterior): need flip to get anterior on left
                base = np.fliplr(base.T)
                overlay = np.fliplr(overlay.T)
        elif x_negative and plane in ["coronal", "axial"]:
            # L orientation: patient's right on left, need flip
            base = np.fliplr(base.T)
            overlay = np.fliplr(overlay.T)
        else:
            # R orientation: patient's left already on right
            base = base.T
            overlay = overlay.T

        ax.imshow(base, cmap="gray", origin="lower", vmin=ref_min, vmax=ref_max)
        # Add white boundary for better visibility
        ax.contour(overlay, levels=[0.5], colors=["white"], linewidths=2.0, alpha=0.8)
        # Add red contour for definition
        ax.contour(overlay, levels=[0.5], colors=["#d62728"], linewidths=1.2, alpha=0.9)
        ax.imshow(
            np.ma.masked_where(overlay <= 0, overlay),
            cmap=cmap,
            alpha=0.35,
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
    >>> from pyrtnf import qc
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
    """
    Compute basic mask metrics.

    Parameters
    ----------
    mask_path : Path
        Path to the mask NIfTI file
    name : str
        Display name for the mask
    chosen_component : int, default=-1
        ICA component index selected for this mask
    correlation : float, default=0.0
        Spatial correlation with Yeo network
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
    >>> from pyrtnf import qc
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
    mni_overlay_dir = overlays_dir / "mni"

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
    from pyrtnf.utils import get_yeo_labels
    n_networks = 7 if len(network_names) <= 7 else 17
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

    # MNI space overlays
    if network_masks:
        import templateflow.api as tflow
        from typing import cast

        mni_overlay_dir.mkdir(parents=True, exist_ok=True)

        # Get MNI template for background
        mni_template = cast(Path, tflow.get(
            "MNI152NLin2009cAsym",
            resolution=2,
            desc="fMRIPrep",
            suffix="boldref"
        ))

        # Get Yeo atlas in MNI space
        yeo_mni_result = tflow.get(
            "MNI152NLin2009cAsym",
            resolution=2,
            atlas="Yeo2011",
            desc=f"{n_networks}Networks" if n_networks == 17 else "7Networks",
            suffix="dseg"
        )
        # tflow.get() may return a list, a single Path, or empty list if not found
        if not yeo_mni_result or (isinstance(yeo_mni_result, list) and len(yeo_mni_result) == 0):
            logger.warning(f"Yeo {n_networks}-network atlas not found in templateflow, skipping MNI overlays")
            yeo_mni = None
        else:
            yeo_mni = Path(yeo_mni_result[0] if isinstance(yeo_mni_result, list) else yeo_mni_result)

        # Find MNI transform files (try multiple naming patterns)
        sub2mni_xfm_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "*sub2mni_*.mat")))
        if not sub2mni_xfm_files:
            # Try alternate pattern: *_to_MNI_*.mat
            sub2mni_xfm_files = sorted(glob(str(mask_extraction_dir / "02_mnireg" / "*_to_MNI_*.mat")))

        if not sub2mni_xfm_files:
            logger.warning("No sub2mni transform files found, skipping MNI overlays")
        else:
            from pyrtnf.utils import apply_xfm

            for network, mask_path in network_masks.items():
                mask_mni = mni_overlay_dir / f"{participant_id}_{network}_mni.nii.gz"
                overlay_png = mni_overlay_dir / f"{participant_id}_{network}_overlay.png"

                if overwrite or not overlay_png.exists():
                    logger.info(f"Generating {network} overlay (MNI)")

                    # Warp mask to MNI if needed
                    if overwrite or not mask_mni.exists():
                        # Use MNI template as fixed image
                        apply_xfm(
                            moving=mask_path,
                            fixed=mni_template,
                            transform_files=[Path(sub2mni_xfm_files[0])],
                            output_file=mask_mni,
                            interpolator='nearestNeighbor',
                            overwrite=overwrite
                        )

                    # Select slices using warped mask (use Yeo atlas if available, otherwise just the mask)
                    if yeo_mni is not None:
                        atlas_slices_mni = select_slices_from_atlas(
                            atlas_path=yeo_mni,
                            labels=labels,
                            target_label=network,
                            top_n_per_plane=1,
                            mask_path=mask_mni if mask_mni.exists() else None
                        )
                    else:
                        # No Yeo atlas available - use mask for slice selection
                        atlas_slices_mni = select_slices_from_atlas(
                            atlas_path=mni_template,  # Dummy atlas (not used when mask_path provided)
                            labels=labels,
                            target_label=network,
                            top_n_per_plane=1,
                            mask_path=mask_mni if mask_mni.exists() else mask_path
                        )

                    plot_mask_overlay_tripanel(
                        mask_path=mask_mni if mask_mni.exists() else mask_path,
                        ref_path=mni_template,
                        out_path=overlay_png,
                        slices=atlas_slices_mni,
                        title=f"{network} (MNI)"
                    )

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

    # Display overlays in report (subject, MNI, and 2vol)
    overlay_spaces = [
        ("BOLD reference", subject_overlay_dir),
        ("MNI", mni_overlay_dir),
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

    # Get mask columns (all columns except volume, time, is_baseline, is_outlier)
    meta_cols = ['volume', 'time', 'is_baseline', 'is_outlier']
    mask_cols = [col for col in df.columns if col not in meta_cols]

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


def plot_baseline_distributions(
        feedback_log_path: Path,
        out_path: Path,
        n_baseline: int,
        title: str | None = None
) -> None:
    """
    Plot baseline distribution histograms to verify normalization.

    Parameters
    ----------
    feedback_log_path : Path
        Path to feedback log TSV file
    out_path : Path
        Path to save the PNG
    n_baseline : int
        Number of baseline volumes
    title : str, optional
        Figure title
    """
    # Load feedback log
    df = pd.read_csv(feedback_log_path, sep='\t')

    # Get mask columns
    meta_cols = ['volume', 'time', 'is_baseline', 'is_outlier']
    mask_cols = [col for col in df.columns if col not in meta_cols]

    # Extract baseline data
    baseline_df = df.iloc[:n_baseline]

    # Determine subplot layout
    n_masks = len(mask_cols)
    n_cols = min(3, n_masks)
    n_rows = (n_masks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    if n_masks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_masks > 1 else [axes]

    for idx, mask_name in enumerate(mask_cols):
        ax = axes[idx]
        baseline_vals = baseline_df[mask_name].values

        # Histogram
        ax.hist(baseline_vals, bins=15, color='#0173B2', alpha=0.7, edgecolor='black')

        # Add mean and std lines
        mean_val = np.mean(baseline_vals)
        std_val = np.std(baseline_vals)
        ax.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(mean_val + std_val, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 SD: {std_val:.3f}')

        ax.set_xlabel('Activation', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(mask_name, fontsize=10)
        ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(n_masks, len(axes)):
        axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Baseline Distributions', fontsize=14, fontweight='bold')

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


def generate_feedback_report(
        feedback_dir: Path,
        subject: str,
        task: str,
        run: int,
        output_dir: Path | None = None,
        overwrite: bool = False,
        open_browser: bool = False,
        expected_volumes: int | None = None,
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

    Returns
    -------
    Path
        Path to generated HTML report

    Example
    -------
    >>> from pyrtnf import qc
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

    logger.info(f"Generating feedback QC report for {subject}, task={task}, run={run}")

    # Auto-discover outputs
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
        'Feedback log': str(feedback_log.exists()),
        'Design matrix': str(design_file.exists())
    }

    # Extract additional info from feedback log if available
    n_volumes_recorded = 0
    if feedback_log.exists():
        df = pd.read_csv(feedback_log, sep='\t')
        meta_cols = ['volume', 'time', 'is_baseline', 'is_outlier']
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
    else:
        summary_data['Status'] = 'NO DATA'

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

    # Baseline distributions
    if feedback_log.exists() and n_baseline and n_baseline > 0:
        sections.append({'type': 'heading', 'content': 'Baseline Distributions'})

        baseline_out = output_dir / "baseline_distributions.png"
        if overwrite or not baseline_out.exists():
            logger.info("Generating baseline distributions")
            plot_baseline_distributions(
                feedback_log_path=feedback_log,
                out_path=baseline_out,
                n_baseline=n_baseline,
                title=f"Baseline Distributions (n={n_baseline})"
            )

        sections.append({
            'type': 'single_image',
            'path': baseline_out.relative_to(output_dir).as_posix(),
            'alt': 'Baseline distributions'
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

    # Generate HTML report
    html_report = output_dir / f"{subject}_task-{task}_run-{run:02d}_feedback_qc.html"
    html = _generate_html_template(
        title=f"Feedback QC - {subject}",
        participant_id=subject,
        stage=f"Feedback (task: {task}, run: {run})",
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
    """
    Generate professional HTML report with consistent styling.

    Parameters
    ----------
    title : str
        Page title
    participant_id : str
        Participant identifier
    stage : str
        Pipeline stage (e.g., "Preprocessing", "Mask Extraction", "Feedback")
    sections : list of dict
        List of content sections. Each dict should have:
        - 'type': 'heading', 'summary', 'grid', 'table', or 'single_image'
        - 'content': appropriate content for the type

    Returns
    -------
    str
        Complete HTML document
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
        <p>Generated by pyrtnf on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
    """
    Create a professional HTML QC report for mask extraction.

    Parameters
    ----------
    out_path : Path
        Output HTML path
    participant_id : str
        Participant identifier for display
    overlays_by_space : iterable of (space_name, overlays)
        Mask overlay image paths grouped by space; each overlays iterable is (name, path)
    heatmap_png : Path or None
        Path to heatmap PNG
    summaries : iterable of MaskSummary
        Mask summary rows to include
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
        title=f"pyrtnf QC - {participant_id}",
        participant_id=participant_id,
        stage="Mask Extraction",
        sections=sections
    )

    out_path.write_text(html)
