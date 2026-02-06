#!/usr/bin/env python3
"""Compare fMRIPrep offline preprocessing with pineuro real-time activation.

This script compares activation timeseries between:
1. fMRIPrep-preprocessed feedback runs + nilearn GLM (gold-standard offline)
2. pineuro real-time feedback estimates

Both pipelines use the same pineuro masks to isolate the preprocessing/GLM
differences from mask extraction differences.

Requires:
- setup_validation.py to have been run first (pineuro preprocessing + mask extraction)
- fMRIPrep outputs at --fmriprep-dir

Usage:
    # Run comparison for all subjects
    python compare_fmriprep_pineuro.py compare \
        --csv tests/remind_test_data.csv \
        --output-dir /data/output \
        --fmriprep-dir /projects/remind/derivatives/fmriprep/fmriprep-23.2.1

    # Check status
    python compare_fmriprep_pineuro.py status --output-dir /data/output
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.image import load_img
from scipy import stats

# Add pineuro to path if running from repo
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from setup_validation import (
    get_feedback_dir,
    get_mask_extraction_dir,
    get_preprocessing_dir,
    get_subject_dir,
    load_subjects_csv,
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RunConfig:
    """Configuration for a single feedback run."""

    task: str
    run: int
    fmriprep_bold: Path
    fmriprep_confounds: Path
    pineuro_feedbacklog: Path
    pineuro_design: Optional[Path] = None


# =============================================================================
# fMRIPrep Path Helpers
# =============================================================================


def get_fmriprep_nf_func_dir(fmriprep_dir: Path, subject: str) -> Path:
    """Get fMRIPrep func directory for neurofeedback session."""
    return fmriprep_dir / subject / "ses-nf" / "func"


# =============================================================================
# Core Functions
# =============================================================================


def find_feedback_runs(
    subject: str,
    output_dir: Path,
    fmriprep_dir: Path,
) -> list[RunConfig]:
    """Find all feedback runs with both fMRIPrep and pineuro outputs."""
    runs = []

    # Find fMRIPrep feedback runs (in T1w space to match pineuro masks)
    fmriprep_func = get_fmriprep_nf_func_dir(fmriprep_dir, subject)
    if not fmriprep_func.exists():
        print(f"  Warning: fMRIPrep func dir not found: {fmriprep_func}")
        return runs

    # Look for preprocessed BOLD files
    bold_pattern = "*_task-*_run-*_space-T1w_desc-preproc_bold.nii.gz"
    bold_files = sorted(fmriprep_func.glob(bold_pattern))

    for bold_file in bold_files:
        # Parse task and run from filename
        parts = bold_file.name.split("_")
        task = None
        run = None
        for part in parts:
            if part.startswith("task-"):
                task = part.replace("task-", "")
            elif part.startswith("run-"):
                run = int(part.replace("run-", ""))

        if not task or run is None:
            continue

        # Skip rest runs - only want feedback/task runs
        if task == "rest":
            continue

        # Find corresponding confounds file
        confounds_pattern = f"*_task-{task}_run-{run:02d}_desc-confounds_timeseries.tsv"
        confounds_files = list(fmriprep_func.glob(confounds_pattern))
        if not confounds_files:
            print(f"  Warning: No confounds file for {task} run {run}")
            continue
        confounds_file = confounds_files[0]

        # Find pineuro feedback log
        subject_dir = get_subject_dir(output_dir, subject)
        feedback_output_dir = get_feedback_dir(subject_dir) / "02_feedback_output"
        pineuro_pattern = f"{subject}_task-{task}_run-{run:02d}_feedbacklog.tsv"

        feedbacklog = None
        for search_dir in [feedback_output_dir, get_feedback_dir(subject_dir)]:
            if search_dir.exists():
                candidates = list(search_dir.rglob(pineuro_pattern))
                if candidates:
                    feedbacklog = candidates[0]
                    break

        if not feedbacklog:
            print(f"  Warning: No pineuro feedbacklog for {task} run {run}")
            continue

        # Find pineuro design file (optional)
        design_pattern = f"{subject}_task-{task}_run-{run:02d}_design.tsv"
        design_file = None
        for search_dir in [feedback_output_dir, get_feedback_dir(subject_dir)]:
            if search_dir.exists():
                candidates = list(search_dir.rglob(design_pattern))
                if candidates:
                    design_file = candidates[0]
                    break

        runs.append(
            RunConfig(
                task=task,
                run=run,
                fmriprep_bold=bold_file,
                fmriprep_confounds=confounds_file,
                pineuro_feedbacklog=feedbacklog,
                pineuro_design=design_file,
            )
        )

    return runs


def load_pineuro_masks(
    subject: str,
    output_dir: Path,
    targets: list[str],
) -> dict[str, Path]:
    """Load pineuro network masks."""
    masks = {}
    subject_dir = get_subject_dir(output_dir, subject)
    mask_dir = get_mask_extraction_dir(subject_dir) / "03_masks"

    for target in targets:
        # Try thresholded mask first
        thr_path = mask_dir / f"{subject}_{target}_thr.nii.gz"
        if thr_path.exists():
            masks[target] = thr_path
        else:
            # Try unthresholded
            unthr_path = mask_dir / f"{subject}_{target}.nii.gz"
            if unthr_path.exists():
                masks[target] = unthr_path
            else:
                print(f"  Warning: Mask not found for {target}")

    return masks


def transform_masks_to_fmriprep_space(
    masks: dict[str, Path],
    fmriprep_bold: Path,
    pineuro_boldref: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Transform pineuro masks from pineuro BOLD space to fMRIPrep T1w space.

    Since fMRIPrep data is in T1w space and pineuro masks are in pineuro BOLD space,
    we need to align them. We do a rigid registration between the boldrefs.
    """
    import ants

    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = {}

    # Load references
    fmriprep_bold_img = ants.image_read(str(fmriprep_bold))
    # Get first volume as reference
    if len(fmriprep_bold_img.shape) == 4:
        fmriprep_ref_data = fmriprep_bold_img.numpy()[:, :, :, 0]
        fmriprep_ref = ants.from_numpy(
            fmriprep_ref_data,
            origin=fmriprep_bold_img.origin[:3],
            spacing=fmriprep_bold_img.spacing[:3],
            direction=fmriprep_bold_img.direction[:3, :3],
        )
    else:
        fmriprep_ref = fmriprep_bold_img

    pineuro_ref = ants.image_read(str(pineuro_boldref))

    # Compute registration: pineuro → fMRIPrep
    reg_file = output_dir / "pineuro_to_fmriprep.mat"
    if not reg_file.exists() or overwrite:
        print("  Computing pineuro→fMRIPrep alignment...")
        reg = ants.registration(
            fixed=fmriprep_ref,
            moving=pineuro_ref,
            type_of_transform='Rigid',
            verbose=False,
        )
        # Save transform
        import shutil
        shutil.copy(reg['fwdtransforms'][0], str(reg_file))

    # Transform each mask
    for network, mask_path in masks.items():
        out_path = output_dir / f"{network}_fmriprep_space.nii.gz"

        if out_path.exists() and not overwrite:
            transformed[network] = out_path
            continue

        mask_img = ants.image_read(str(mask_path))

        # Apply transform
        mask_transformed = ants.apply_transforms(
            fixed=fmriprep_ref,
            moving=mask_img,
            transformlist=[str(reg_file)],
            interpolator='nearestNeighbor',
        )

        ants.image_write(mask_transformed, str(out_path))
        transformed[network] = out_path
        print(f"  Transformed {network} mask: {mask_transformed.numpy().sum():.0f} voxels")

    return transformed


def extract_fmriprep_activation(
    bold_file: Path,
    mask_file: Path,
    confounds_file: Path,
) -> np.ndarray:
    """Extract mean activation from fMRIPrep BOLD using simple averaging.

    Mean signal in mask after regressing out motion confounds.
    """
    from nilearn.maskers import NiftiMasker
    from nilearn.signal import clean

    # Load mask
    mask_img = load_img(mask_file)

    # Load BOLD
    bold_img = load_img(bold_file)

    # Load confounds
    confounds_df = pd.read_csv(confounds_file, sep='\t')

    # Select relevant confounds (motion parameters)
    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    available_cols = [c for c in motion_cols if c in confounds_df.columns]

    if not available_cols:
        print(f"  Warning: No motion confounds found, using raw signal")
        confounds = None
    else:
        confounds = confounds_df[available_cols].values
        # Handle NaN in first row
        confounds = np.nan_to_num(confounds, nan=0.0)

    # Create masker
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=False,
        detrend=False,
    )

    # Extract timeseries
    masked_data = masker.fit_transform(bold_img)  # (n_volumes, n_voxels)

    # Regress out confounds if available
    if confounds is not None:
        masked_data = clean(masked_data, confounds=confounds, detrend=True, standardize=False)

    # Mean across voxels
    mean_activation = masked_data.mean(axis=1)

    return mean_activation


def load_pineuro_activation(feedbacklog_path: Path, network: str) -> np.ndarray:
    """Load pineuro activation timeseries from feedback log."""
    df = pd.read_csv(feedbacklog_path, sep='\t')

    # Find the activation column for this network
    candidates = [
        f'activation_{network}',
        f'activation_{network.lower()}',
        'activation',
        f'{network}_activation',
    ]

    activation_col = None
    for col in candidates:
        if col in df.columns:
            activation_col = col
            break

    if activation_col is None:
        # Use first column that contains 'activation'
        activation_cols = [c for c in df.columns if 'activation' in c.lower()]
        if activation_cols:
            activation_col = activation_cols[0]
            print(f"  Using activation column: {activation_col}")
        else:
            raise ValueError(f"No activation column found in {feedbacklog_path}")

    return df[activation_col].values


def compare_timeseries(
    fmriprep_activation: np.ndarray,
    pineuro_activation: np.ndarray,
    n_baseline: int = 25,
) -> dict:
    """Compare two activation timeseries."""
    # Align lengths
    min_len = min(len(fmriprep_activation), len(pineuro_activation))
    fmriprep = fmriprep_activation[:min_len]
    pineuro = pineuro_activation[:min_len]

    # Full timeseries comparison
    r_full, p_full = stats.pearsonr(fmriprep, pineuro)

    # Task period only (after baseline)
    if min_len > n_baseline:
        fmriprep_task = fmriprep[n_baseline:]
        pineuro_task = pineuro[n_baseline:]
        r_task, p_task = stats.pearsonr(fmriprep_task, pineuro_task)
    else:
        r_task, p_task = np.nan, np.nan

    # Linear regression for slope/intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(pineuro, fmriprep)

    # RMSE
    rmse = np.sqrt(np.mean((fmriprep - pineuro) ** 2))

    return {
        'r_full': r_full,
        'p_full': p_full,
        'r_task': r_task,
        'p_task': p_task,
        'slope': slope,
        'intercept': intercept,
        'rmse': rmse,
        'n_volumes': min_len,
    }


def create_comparison_plot(
    fmriprep_activation: np.ndarray,
    pineuro_activation: np.ndarray,
    metrics: dict,
    network: str,
    task: str,
    run: int,
    output_path: Path,
    n_baseline: int = 25,
    tr: float = 1.2,
) -> None:
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Align lengths
    min_len = min(len(fmriprep_activation), len(pineuro_activation))
    fmriprep = fmriprep_activation[:min_len]
    pineuro = pineuro_activation[:min_len]
    time = np.arange(min_len) * tr

    # Z-score for visualization
    fmriprep_z = (fmriprep - fmriprep.mean()) / fmriprep.std()
    pineuro_z = (pineuro - pineuro.mean()) / pineuro.std()

    # Plot 1: Timeseries overlay
    ax1 = axes[0, 0]
    ax1.plot(time, fmriprep_z, 'b-', label='fMRIPrep', alpha=0.7)
    ax1.plot(time, pineuro_z, 'r-', label='pineuro', alpha=0.7)
    ax1.axvline(n_baseline * tr, color='gray', linestyle='--', label='Baseline end')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Activation (z-score)')
    ax1.set_title(f'{network} - {task} run {run}\nr={metrics["r_full"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(pineuro_z, fmriprep_z, alpha=0.5, s=20)
    x_line = np.linspace(pineuro_z.min(), pineuro_z.max(), 100)
    slope_z, intercept_z, _, _, _ = stats.linregress(pineuro_z, fmriprep_z)
    y_line = slope_z * x_line + intercept_z
    ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'slope={slope_z:.2f}')
    ax2.plot([-3, 3], [-3, 3], 'k--', alpha=0.5, label='identity')
    ax2.set_xlabel('pineuro (z-score)')
    ax2.set_ylabel('fMRIPrep (z-score)')
    ax2.set_title(f'Scatter (r={metrics["r_full"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = fmriprep_z - pineuro_z
    ax3.plot(time, residuals, 'g-', alpha=0.7)
    ax3.axhline(0, color='k', linestyle='--')
    ax3.axvline(n_baseline * tr, color='gray', linestyle='--')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Residual (fMRIPrep - pineuro)')
    ax3.set_title(f'Residuals (RMSE={metrics["rmse"]:.3f})')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Metrics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
    Comparison Metrics
    ==================

    Correlation (full):    r = {metrics['r_full']:.4f}  (p = {metrics['p_full']:.2e})
    Correlation (task):    r = {metrics['r_task']:.4f}  (p = {metrics['p_task']:.2e})

    Regression:
      Slope:     {metrics['slope']:.4f}
      Intercept: {metrics['intercept']:.4f}

    Error:
      RMSE:      {metrics['rmse']:.4f}

    N volumes:   {metrics['n_volumes']}
    """
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Comparison Function
# =============================================================================


def run_comparison_for_subject(
    subject: str,
    output_dir: Path,
    fmriprep_dir: Path,
    targets: list[str],
    tr: float = 1.2,
    n_baseline: int = 25,
    overwrite: bool = False,
) -> dict:
    """Run comparison for a single subject."""
    print(f"\n{'#' * 60}")
    print(f"# COMPARE: {subject}")
    print(f"{'#' * 60}")

    subject_dir = get_subject_dir(output_dir, subject)
    comparison_dir = subject_dir / "fmriprep_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Find feedback runs
    runs = find_feedback_runs(subject, output_dir, fmriprep_dir)
    if not runs:
        print("  No matching feedback runs found")
        return {}

    print(f"  Found {len(runs)} feedback runs")

    # Load pineuro masks
    masks = load_pineuro_masks(subject, output_dir, targets)
    if not masks:
        print("  No pineuro masks found")
        return {}

    print(f"  Loaded masks: {list(masks.keys())}")

    # Get pineuro boldref
    preproc_dir = get_preprocessing_dir(subject_dir)
    pineuro_boldref = preproc_dir / f"{subject}_task-rest_boldref.nii.gz"
    if not pineuro_boldref.exists():
        print(f"  Error: Boldref not found: {pineuro_boldref}")
        return {}

    all_results = {}

    for run_config in runs:
        run_key = f"{run_config.task}_run-{run_config.run:02d}"
        print(f"\n  Processing {run_key}...")

        run_results = {}
        run_dir = comparison_dir / run_key
        run_dir.mkdir(exist_ok=True)

        # Transform masks to fMRIPrep space for this run
        masks_fmriprep = transform_masks_to_fmriprep_space(
            masks,
            run_config.fmriprep_bold,
            pineuro_boldref,
            run_dir,
            overwrite=overwrite,
        )

        for network, mask_path in masks_fmriprep.items():
            print(f"    {network}...")

            # Extract fMRIPrep activation
            try:
                fmriprep_activation = extract_fmriprep_activation(
                    run_config.fmriprep_bold,
                    mask_path,
                    run_config.fmriprep_confounds,
                )
            except Exception as e:
                print(f"      Error extracting fMRIPrep activation: {e}")
                continue

            # Load pineuro activation
            try:
                pineuro_activation = load_pineuro_activation(
                    run_config.pineuro_feedbacklog,
                    network,
                )
            except Exception as e:
                print(f"      Error loading pineuro activation: {e}")
                continue

            # Compare
            metrics = compare_timeseries(
                fmriprep_activation,
                pineuro_activation,
                n_baseline=n_baseline,
            )

            run_results[network] = metrics

            print(f"      r={metrics['r_full']:.4f}, slope={metrics['slope']:.4f}, RMSE={metrics['rmse']:.4f}")

            # Create plot
            plot_path = run_dir / f"{network}_comparison.png"
            create_comparison_plot(
                fmriprep_activation,
                pineuro_activation,
                metrics,
                network,
                run_config.task,
                run_config.run,
                plot_path,
                n_baseline=n_baseline,
                tr=tr,
            )

        all_results[run_key] = run_results

    # Save subject summary
    save_subject_summary(subject, all_results, comparison_dir)

    return all_results


def save_subject_summary(subject: str, results: dict, output_dir: Path) -> None:
    """Save subject-level summary."""
    summary_path = output_dir / "comparison_summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"fMRIPrep vs pineuro Comparison Summary\n")
        f.write(f"Subject: {subject}\n")
        f.write("=" * 60 + "\n\n")

        for run_key, run_results in results.items():
            f.write(f"\n{run_key}:\n")
            f.write("-" * 40 + "\n")

            for network, metrics in run_results.items():
                f.write(f"\n  {network}:\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {key}: {value:.4f}\n")
                    else:
                        f.write(f"    {key}: {value}\n")

    print(f"\n  Summary saved: {summary_path}")


# =============================================================================
# Group Results TSV
# =============================================================================


def save_group_results(all_results: list[dict], output_dir: Path) -> None:
    """Save structured TSV with one row per subject/run/network."""
    rows = []
    for r in all_results:
        for run_key, run_results in r["results"].items():
            # Parse task and run from key like "nback_run-01"
            parts = run_key.rsplit("_run-", 1)
            task = parts[0] if len(parts) == 2 else run_key
            run_num = int(parts[1]) if len(parts) == 2 else 0

            for network, metrics in run_results.items():
                rows.append({
                    "subject": r["subject"],
                    "task": task,
                    "run": run_num,
                    "network": network,
                    "r_full": metrics.get("r_full"),
                    "r_task": metrics.get("r_task"),
                    "slope": metrics.get("slope"),
                    "intercept": metrics.get("intercept"),
                    "rmse": metrics.get("rmse"),
                    "n_volumes": metrics.get("n_volumes"),
                })

    if rows:
        df = pd.DataFrame(rows)
        tsv_path = output_dir / "group_results_fmriprep.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"\nGroup results saved: {tsv_path}")


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison for all subjects."""
    configs = load_subjects_csv(args.csv)

    print("=" * 60)
    print("FMRIPREP vs PINEURO COMPARISON")
    print("=" * 60)
    print(f"CSV:        {args.csv}")
    print(f"Output:     {args.output_dir}")
    print(f"fMRIPrep:   {args.fmriprep_dir}")
    print(f"Subjects:   {len(configs)}")
    print(f"TR:         {args.tr}")
    print(f"N baseline: {args.n_baseline}")
    print()

    all_results = []

    for config in configs:
        try:
            results = run_comparison_for_subject(
                subject=config.subject,
                output_dir=args.output_dir,
                fmriprep_dir=args.fmriprep_dir,
                targets=args.targets.split(","),
                tr=args.tr,
                n_baseline=args.n_baseline,
                overwrite=args.overwrite,
            )
            all_results.append({"subject": config.subject, "results": results})
        except Exception as e:
            print(f"\nERROR processing {config.subject}: {e}")
            import traceback
            traceback.print_exc()
            continue

    save_group_results(all_results, args.output_dir)
    print_summary(all_results)


def cmd_status(args: argparse.Namespace) -> None:
    """Check status of all subjects."""
    if not args.output_dir.exists():
        print(f"Output directory not found: {args.output_dir}")
        return

    print("=" * 60)
    print("FMRIPREP COMPARISON STATUS")
    print("=" * 60)

    if args.csv:
        configs = load_subjects_csv(args.csv)
        subjects = [c.subject for c in configs]
    else:
        subjects = sorted([d.name for d in args.output_dir.iterdir() if d.is_dir()])

    for subject in subjects:
        subject_dir = args.output_dir / subject
        comparison_dir = subject_dir / "fmriprep_comparison"

        if comparison_dir.exists():
            summary = comparison_dir / "comparison_summary.txt"
            if summary.exists():
                print(f"  {subject}: complete")
            else:
                print(f"  {subject}: in progress")
        else:
            print(f"  {subject}: not started")


def print_summary(all_results: list[dict]) -> None:
    """Print final summary of comparison results."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if not all_results:
        print("\nNo comparisons were completed.")
        return

    # Collect metrics by network
    r_by_network: dict[str, list[float]] = {}
    slope_by_network: dict[str, list[float]] = {}
    rmse_by_network: dict[str, list[float]] = {}

    for r in all_results:
        print(f"\n{r['subject']}:")
        for run_key, run_results in r["results"].items():
            for network, metrics in run_results.items():
                print(f"  {run_key} {network}: r={metrics['r_full']:.4f}, "
                      f"slope={metrics['slope']:.4f}, RMSE={metrics['rmse']:.4f}")

                if network not in r_by_network:
                    r_by_network[network] = []
                    slope_by_network[network] = []
                    rmse_by_network[network] = []

                r_by_network[network].append(metrics['r_full'])
                slope_by_network[network].append(metrics['slope'])
                rmse_by_network[network].append(metrics['rmse'])

    # Print statistics
    print("\n" + "-" * 60)
    print("STATISTICS BY NETWORK")
    print("-" * 60)

    for network in r_by_network.keys():
        r_arr = np.array(r_by_network[network])
        slope_arr = np.array(slope_by_network[network])
        rmse_arr = np.array(rmse_by_network[network])

        print(f"\n{network} (n={len(r_arr)} runs):")
        print(f"  Correlation: mean={r_arr.mean():.4f}, std={r_arr.std():.4f}, "
              f"min={r_arr.min():.4f}, max={r_arr.max():.4f}")
        print(f"  Slope:       mean={slope_arr.mean():.4f}, std={slope_arr.std():.4f}")
        print(f"  RMSE:        mean={rmse_arr.mean():.4f}, std={rmse_arr.std():.4f}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fMRIPrep offline preprocessing with pineuro real-time activation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run fMRIPrep vs pineuro comparison",
    )
    compare_parser.add_argument("--csv", type=Path, required=True, help="Path to subjects CSV")
    compare_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    compare_parser.add_argument("--fmriprep-dir", type=Path, required=True, help="fMRIPrep derivatives directory")
    compare_parser.add_argument("--targets", type=str, default="DefaultA,ContA", help="Target networks")
    compare_parser.add_argument("--tr", type=float, default=1.2, help="TR in seconds")
    compare_parser.add_argument("--n-baseline", type=int, default=25, help="Number of baseline volumes")
    compare_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check comparison status")
    status_parser.add_argument("--csv", type=Path, help="Path to subjects CSV (optional)")
    status_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    args = parser.parse_args()

    # Resolve paths
    if hasattr(args, "csv") and args.csv:
        args.csv = args.csv.resolve()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.resolve()
    if hasattr(args, "fmriprep_dir") and args.fmriprep_dir:
        args.fmriprep_dir = args.fmriprep_dir.resolve()

    # Run command
    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
