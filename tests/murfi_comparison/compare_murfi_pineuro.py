#!/usr/bin/env python3
"""Compare MURFI and pineuro feedback outputs - GUI-compatible workflow.

This script automates the pineuro GUI workflow for MURFI validation:
1. Setup stage: Copy DICOMs, preprocess, extract masks
2. (Manual) Validate masks in GUI
3. Compare stage: Run feedback sessions, run MURFI, compare outputs

Usage:
    # Stage 1: Setup (preprocessing + mask extraction)
    python compare_murfi_pineuro.py setup \\
        --csv tests/remind_test_data.csv \\
        --output-dir /data/murfi_validation

    # Check status
    python compare_murfi_pineuro.py status --output-dir /data/murfi_validation

    # Stage 2: Compare (after GUI mask approval)
    python compare_murfi_pineuro.py compare \\
        --csv tests/remind_test_data.csv \\
        --output-dir /data/murfi_validation \\
        --murfi-sif /path/to/murfi.sif
"""

import argparse
import csv
import gzip
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add pineuro to path if running from repo
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SubjectConfig:
    """Configuration for a single subject from the CSV."""

    subject: str
    rest1: Path
    rest2: Path
    twovol: Path
    feedback_runs: list[Path] = field(default_factory=list)

    @property
    def rest_runs(self) -> list[Path]:
        return [self.rest1, self.rest2]


# =============================================================================
# CSV Loading
# =============================================================================


def load_subjects_csv(csv_path: Path) -> list[SubjectConfig]:
    """Load subject configurations from CSV file.

    CSV format:
        subject,rest1,rest2,2vol,feedback_run-01,feedback_run-02,...
    """
    configs = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject = row["subject"].strip()
            if not subject:
                continue

            # Get rest and 2vol paths
            rest1 = Path(row["rest1"].strip())
            rest2 = Path(row["rest2"].strip())
            twovol = Path(row["2vol"].strip())

            # Get all feedback runs (feedback_run-01, feedback_run-02, etc.)
            feedback_runs = []
            for key, value in row.items():
                if key.startswith("feedback_run-") and value and value.strip():
                    feedback_runs.append(Path(value.strip()))

            configs.append(
                SubjectConfig(
                    subject=subject,
                    rest1=rest1,
                    rest2=rest2,
                    twovol=twovol,
                    feedback_runs=feedback_runs,
                )
            )

    return configs


# =============================================================================
# Directory Structure Helpers
# =============================================================================


def get_subject_dir(output_dir: Path, subject: str) -> Path:
    """Get subject directory path."""
    return output_dir / subject


def get_collection_dir(subject_dir: Path) -> Path:
    """Get collection directory path."""
    return subject_dir / "collection"


def get_preprocessing_dir(subject_dir: Path) -> Path:
    """Get preprocessing directory path."""
    return subject_dir / "preprocessing"


def get_mask_extraction_dir(subject_dir: Path) -> Path:
    """Get mask extraction directory path."""
    return subject_dir / "mask_extraction"


def get_feedback_dir(subject_dir: Path) -> Path:
    """Get feedback directory path."""
    return subject_dir / "feedback"


def get_murfi_dir(subject_dir: Path) -> Path:
    """Get MURFI output directory path."""
    return subject_dir / "murfi"


def get_comparison_dir(subject_dir: Path) -> Path:
    """Get comparison results directory path."""
    return subject_dir / "comparison"


# =============================================================================
# Status Checking
# =============================================================================


def check_collection_complete(subject_dir: Path) -> bool:
    """Check if DICOM collection is complete."""
    collection_dir = get_collection_dir(subject_dir)
    return (
        (collection_dir / "rest_run-01").exists()
        and (collection_dir / "rest_run-02").exists()
        and (collection_dir / "2vol_run-01").exists()
    )


def check_preprocessing_complete(subject_dir: Path, subject: str) -> bool:
    """Check if preprocessing is complete."""
    preproc_dir = get_preprocessing_dir(subject_dir)
    preproc_file = preproc_dir / f"{subject}_task-rest_preproc.nii.gz"
    return preproc_file.exists()


def check_mask_extraction_complete(subject_dir: Path) -> bool:
    """Check if mask extraction is complete."""
    mask_dir = get_mask_extraction_dir(subject_dir) / "04_2vol"
    # Look for coregistered mask files
    masks = list(mask_dir.glob("*_thr_to_task-2vol*.nii.gz"))
    return len(masks) >= 2


def check_masks_approved(subject_dir: Path) -> bool:
    """Check if masks have been approved via GUI."""
    return (get_mask_extraction_dir(subject_dir) / ".masks_approved").exists()


def check_feedback_run_complete(subject_dir: Path, subject: str, run: int) -> bool:
    """Check if a feedback run is complete."""
    feedback_dir = get_feedback_dir(subject_dir) / "02_feedback_output"
    feedbacklog = feedback_dir / f"{subject}_task-feedback_run-{run:02d}_feedbacklog.tsv"
    return feedbacklog.exists()


def check_murfi_run_complete(subject_dir: Path, run: int, n_volumes: int) -> bool:
    """Check if MURFI run is complete."""
    log_path = get_murfi_dir(subject_dir) / f"run-{run:02d}" / "log" / "log.rtl"
    if not log_path.exists():
        return False

    # Check if log has enough volumes
    pattern = r"RtRoiCombine: (dmn|cen) at tr (\d+) result is:"
    max_tr = 0
    try:
        with open(log_path) as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    tr = int(match.group(2))
                    max_tr = max(max_tr, tr)
        return max_tr >= n_volumes
    except Exception:
        return False


def check_comparison_complete(subject_dir: Path, run: int) -> bool:
    """Check if comparison is complete for a run."""
    report = get_comparison_dir(subject_dir) / f"run-{run:02d}" / "comparison_report.txt"
    return report.exists()


def get_subject_status(subject_dir: Path, subject: str, n_feedback_runs: int, n_volumes: int) -> dict:
    """Get comprehensive status for a subject."""
    return {
        "collection_complete": check_collection_complete(subject_dir),
        "preprocessing_complete": check_preprocessing_complete(subject_dir, subject),
        "mask_extraction_complete": check_mask_extraction_complete(subject_dir),
        "masks_approved": check_masks_approved(subject_dir),
        "feedback_runs": [
            check_feedback_run_complete(subject_dir, subject, run)
            for run in range(1, n_feedback_runs + 1)
        ],
        "murfi_runs": [
            check_murfi_run_complete(subject_dir, run, n_volumes)
            for run in range(1, n_feedback_runs + 1)
        ],
        "comparisons": [
            check_comparison_complete(subject_dir, run)
            for run in range(1, n_feedback_runs + 1)
        ],
    }


# =============================================================================
# Setup Stage Functions
# =============================================================================


def copy_dicoms_to_collection(config: SubjectConfig, subject_dir: Path) -> None:
    """Copy source DICOMs to collection directory."""
    collection_dir = get_collection_dir(subject_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Copy rest runs
    for i, rest_path in enumerate(config.rest_runs, 1):
        dest = collection_dir / f"rest_run-{i:02d}"
        if not dest.exists():
            print(f"  Copying {rest_path.name} -> rest_run-{i:02d}")
            shutil.copytree(rest_path, dest)
        else:
            print(f"  rest_run-{i:02d} already exists, skipping")

    # Copy 2vol
    dest = collection_dir / "2vol_run-01"
    if not dest.exists():
        print(f"  Copying {config.twovol.name} -> 2vol_run-01")
        shutil.copytree(config.twovol, dest)
    else:
        print(f"  2vol_run-01 already exists, skipping")

    # Copy feedback runs
    for i, fb_path in enumerate(config.feedback_runs, 1):
        dest = collection_dir / f"feedback_run-{i:02d}"
        if not dest.exists():
            print(f"  Copying {fb_path.name} -> feedback_run-{i:02d}")
            shutil.copytree(fb_path, dest)
        else:
            print(f"  feedback_run-{i:02d} already exists, skipping")


def run_preprocessing(
    subject: str,
    subject_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Run preprocessing on resting-state data."""
    from pineuro import preprocessing

    collection_dir = get_collection_dir(subject_dir)
    preproc_dir = get_preprocessing_dir(subject_dir)

    input_dirs = [
        collection_dir / "rest_run-01",
        collection_dir / "rest_run-02",
    ]

    print(f"\n{'=' * 60}")
    print("PREPROCESSING")
    print("=" * 60)
    print(f"  Input: {input_dirs}")
    print(f"  Output: {preproc_dir}")

    preproc_file, mask_file, ref_file = preprocessing(
        participant_id=subject,
        task="rest",
        root_dir=preproc_dir,
        input_dicoms=input_dirs,
        overwrite=overwrite,
        qc_enabled=True,
    )

    print(f"\nPreprocessing complete:")
    print(f"  Preprocessed: {preproc_file}")
    print(f"  Mask: {mask_file}")
    print(f"  Reference: {ref_file}")

    return preproc_file, mask_file, ref_file


def run_mask_extraction(
    subject: str,
    subject_dir: Path,
    preproc_file: Path,
    mask_file: Path,
    ref_file: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Run ICA-based mask extraction."""
    from pineuro import mask_extraction_rest

    mask_dir = get_mask_extraction_dir(subject_dir)

    print(f"\n{'=' * 60}")
    print("MASK EXTRACTION")
    print("=" * 60)
    print(f"  Input: {preproc_file}")
    print(f"  Output: {mask_dir}")

    mask_files = mask_extraction_rest(
        participant_id=subject,
        root_dir=mask_dir,
        input_file=preproc_file,
        mask_file=mask_file,
        ref_file=ref_file,
        n_yeo=17,
        yeo_targets=["DefaultA", "ContA"],
        n_components=20,
        threshold=2000,
        smooth_fwhm=5.0,
        overwrite=overwrite,
        qc_enabled=True,
    )

    print(f"\nMask extraction complete:")
    for mf in mask_files:
        print(f"  {mf}")

    return mask_files


def run_2vol_coregistration(
    subject: str,
    subject_dir: Path,
    mask_files: list[Path],
    ref_file: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Coregister masks to 2vol scanner position."""
    from pineuro import coregister_2vol

    collection_dir = get_collection_dir(subject_dir)
    mask_dir = get_mask_extraction_dir(subject_dir)
    twovol_dir = collection_dir / "2vol_run-01"

    print(f"\n{'=' * 60}")
    print("2VOL COREGISTRATION")
    print("=" * 60)
    print(f"  2vol: {twovol_dir}")
    print(f"  Output: {mask_dir}")

    coreg_masks = coregister_2vol(
        participant_id=subject,
        root_dir=str(mask_dir),
        input_dicom=twovol_dir,
        network_files=mask_files,
        ref_file=ref_file,
        overwrite=overwrite,
        output_subdir="04_2vol",
    )

    print(f"\nCoregistration complete:")
    for mf in coreg_masks:
        print(f"  {mf}")

    # Build dict mapping network name to path
    masks = {}
    for mask_path in coreg_masks:
        name = mask_path.name
        if "_ContA_" in name:
            masks["ContA"] = mask_path
        elif "_DefaultA_" in name:
            masks["DefaultA"] = mask_path

    return masks


def run_setup_for_subject(
    config: SubjectConfig,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run complete setup for a subject."""
    subject_dir = get_subject_dir(output_dir, config.subject)

    print(f"\n{'#' * 60}")
    print(f"# SETUP: {config.subject}")
    print(f"{'#' * 60}")

    # Step 1: Copy DICOMs to collection
    print("\nStep 1: Copying DICOMs to collection...")
    copy_dicoms_to_collection(config, subject_dir)

    # Step 2: Preprocessing
    preproc_file, mask_file, ref_file = run_preprocessing(
        config.subject, subject_dir, overwrite
    )

    # Step 3: Mask extraction
    mask_files = run_mask_extraction(
        config.subject, subject_dir, preproc_file, mask_file, ref_file, overwrite
    )

    # Step 4: 2vol coregistration
    masks = run_2vol_coregistration(
        config.subject, subject_dir, mask_files, ref_file, overwrite
    )

    return masks


# =============================================================================
# MURFI Functions (preserved from old implementation)
# =============================================================================


def copy_and_convert_mask(src: Path, dst: Path) -> None:
    """Copy a mask to uncompressed int16 NIfTI for MURFI."""
    import nibabel as nib

    dst.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(src)
    data = img.get_fdata()
    data_int16 = data.astype(np.int16)

    new_img = nib.Nifti1Image(data_int16, img.affine, img.header)
    new_img.header.set_data_dtype(np.int16)

    nib.save(new_img, dst)
    print(f"  Converted mask: {src.name} -> {dst.name} (int16)")


def generate_murfi_xml(
    xml_path: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
) -> None:
    """Generate MURFI XML configuration file."""
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<study name="rtDMN">
    <option name="softwareDir"> /opt/murfi </option>
</study>

<scanner>
  <option name="imageSource">     DICOM  </option>
  <option name="disabled">     false </option>
  <option name="tr">             {tr} </option>
  <option name="measurements">   {n_volumes} </option>
  <option name="saveImages">     true </option>
  <option name="receiveImages">  true </option>
  <option name="onlyReadMoCo">  true </option>
  <option name="preHeader">     true </option>
  <option name="inputDicomDir">   tmp/murfi_input </option>
</scanner>

<infoserver>
  <option name="disabled"> false </option>
  <option name="port"> 15001 </option>
</infoserver>

<processor>
  <module name="mosaic">
    <option name="disabled"> false </option>
    <output>display</output>
    <output>log</output>
  </module>

  <module name="mask-gen">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="roiID"> brain </option>
    <option name="threshold"> 0.5 </option>
    <option name="useBet"> false </option>
    <option name="betParms">-f 0.5 -g 0 -n -m</option>
    <option name="dynamic"> false </option>
    <option name="save"> true </option>
  </module>

  <module name="mask-load">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="roiID"> dmn </option>
    <option name="filename"> dmn </option>
    <option name="align"> false </option>
    <option name="save"> true </option>
  </module>

  <module name="mask-load">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="roiID"> cen </option>
    <option name="filename"> cen </option>
    <option name="align"> false </option>
    <option name="save"> true </option>
  </module>

  <module name="incremental-glm">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <output>display</output>
    <option name="maskRoiID">brain</option>

    <design>
      <option name="disabled"> false </option>
      <option name="modelMotionDerivatives"> true </option>
      <option name="maxTrendOrder"> 1 </option>

      <option name="condition" conditionName="Regulation">
        {' '.join(['1'] * n_volumes)}
      </option>

      <option name="conditionShift">{n_baseline}</option>
    </design>
  </module>

  <module name="current-activation">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="print"> true </option>
    <option name="modelFitModuleID"> incremental-glm </option>
    <option name="modelFitRoiID"> brain </option>
    <option name="maskRoiID"> brain </option>
    <option name="numDataPointsForErrEst"> {n_baseline} </option>
  </module>

  <module name="roi-combine">
    <output> infoserver </output>
    <output>display</output>
    <option name="disabled"> false </option>
    <option name="print"> true </option>
    <option name="makeavail">true</option>
    <option name="method"> weighted-ave </option>
    <option name="maskRoiID"> dmn </option>
    <option name="inputModuleID"> current-activation </option>
    <option name="inputDataName"> activation-img </option>
    <option name="inputRoiID"> brain </option>
    <option name="weightModuleID"> incremental-glm </option>
    <option name="weightDataName"> residual_mse-img </option>
  </module>

  <module name="roi-combine">
    <output> infoserver </output>
    <output>display</output>
    <option name="disabled"> false </option>
    <option name="print"> true </option>
    <option name="makeavail">true</option>
    <option name="method"> weighted-ave </option>
    <option name="maskRoiID"> cen </option>
    <option name="inputModuleID"> current-activation </option>
    <option name="inputDataName"> activation-img </option>
    <option name="inputRoiID"> brain </option>
    <option name="weightModuleID"> incremental-glm </option>
    <option name="weightDataName"> residual_mse-img </option>
  </module>

</processor>
'''

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xml_path, "w") as f:
        f.write(xml_content)

    print(f"  Generated MURFI XML: {xml_path}")


def setup_murfi_environment(
    subject: str,
    masks: dict[str, Path],
    run_dir: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
) -> Path:
    """Create MURFI directory structure and copy masks."""
    murfi_dir = run_dir / subject

    (murfi_dir / "mask").mkdir(parents=True, exist_ok=True)
    (murfi_dir / "xml").mkdir(exist_ok=True)
    (murfi_dir / "log").mkdir(exist_ok=True)

    print("\nSetting up MURFI environment...")
    copy_and_convert_mask(masks["ContA"], murfi_dir / "mask" / "cen.nii")
    copy_and_convert_mask(masks["DefaultA"], murfi_dir / "mask" / "dmn.nii")
    generate_murfi_xml(murfi_dir / "xml" / "rtdmn.xml", tr, n_volumes, n_baseline)

    return murfi_dir


def run_murfi(
    subject: str,
    dicom_dir: Path,
    murfi_sif: Path,
    run_dir: Path,
    tr: float,
    n_volumes: int,
) -> Path:
    """Run MURFI in background, serve DICOMs, wait for completion."""
    subject_dir = run_dir / subject
    xml_path = subject_dir / "xml" / "rtdmn.xml"
    log_path = subject_dir / "log" / "log.rtl"

    murfi_input_dir = Path("tmp/murfi_input")
    murfi_input_dir.mkdir(parents=True, exist_ok=True)

    for f in murfi_input_dir.glob("*"):
        f.unlink()

    # Copy DICOMs to a path without spaces (apptainer limitation)
    dicom_copy_dir = run_dir / "dicom_source"
    if dicom_copy_dir.exists():
        shutil.rmtree(dicom_copy_dir)
    print(f"\nCopying DICOMs for MURFI...")
    print(f"  From: {dicom_dir}")
    print(f"  To: {dicom_copy_dir}")
    shutil.copytree(dicom_dir, dicom_copy_dir)

    env = os.environ.copy()
    env["MURFI_SUBJECTS_DIR"] = str(run_dir.absolute())
    env["MURFI_SUBJECT_NAME"] = subject

    print(f"\nStarting MURFI...")
    print(f"  MURFI_SUBJECTS_DIR={env['MURFI_SUBJECTS_DIR']}")
    print(f"  MURFI_SUBJECT_NAME={env['MURFI_SUBJECT_NAME']}")

    murfi_cmd = [
        "apptainer",
        "exec",
        str(murfi_sif),
        "murfi",
        "-f",
        str(xml_path.absolute()),
    ]

    murfi_proc = subprocess.Popen(
        murfi_cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    print("  Waiting for MURFI to initialize...")
    time.sleep(5)

    if murfi_proc.poll() is not None:
        raise RuntimeError(f"MURFI failed to start (exit code: {murfi_proc.returncode})")

    serve_cmd = [
        "apptainer",
        "exec",
        str(murfi_sif),
        "servedicoms",
        str(dicom_copy_dir.absolute()),
        "tmp/murfi_input",
        str(int(tr * 1000)),
    ]

    print(f"\nServing DICOMs...")
    serve_proc = subprocess.Popen(
        serve_cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    expected_time = n_volumes * tr + 30
    print(f"  Waiting up to {expected_time:.0f}s for servedicoms...")

    try:
        serve_proc.wait(timeout=expected_time)
    except subprocess.TimeoutExpired:
        print("  servedicoms timed out - killing")
        serve_proc.kill()
        try:
            os.killpg(os.getpgid(serve_proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    print("  Waiting for MURFI to finish...")
    time.sleep(3)

    # Kill MURFI
    try:
        os.kill(murfi_proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.killpg(os.getpgid(murfi_proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    subprocess.run(["pkill", "-9", "-f", "murfi -f"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "apptainer.*murfi.sif"], capture_output=True)
    time.sleep(0.5)

    print(f"MURFI complete: {log_path}")
    return log_path


def parse_murfi_log(log_path: Path, n_volumes: int) -> pd.DataFrame:
    """Parse MURFI log.rtl for activation values."""
    pattern = r"RtRoiCombine: (dmn|cen) at tr (\d+) result is: ([0-9e.+-]+)"

    all_matches = []
    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                roi, tr, value = match.groups()
                all_matches.append((roi, int(tr), float(value)))

    if not all_matches:
        raise ValueError(f"No RtRoiCombine entries found in {log_path}")

    # Detect run boundaries
    runs: list[list[tuple[str, int, float]]] = []
    current_run: list[tuple[str, int, float]] = []
    prev_tr = 0

    for roi, tr, value in all_matches:
        if tr < prev_tr - 10:
            if current_run:
                runs.append(current_run)
            current_run = []
        current_run.append((roi, tr, value))
        prev_tr = tr

    if current_run:
        runs.append(current_run)

    print(f"  Found {len(runs)} run(s) in MURFI log")

    last_run = runs[-1]

    cen_values: dict[int, float] = {}
    dmn_values: dict[int, float] = {}

    for roi, tr, value in last_run:
        if roi == "cen":
            cen_values[tr] = value
        else:
            dmn_values[tr] = value

    data = {"volume": list(range(1, n_volumes + 1))}
    data["cen"] = [cen_values.get(v, 0.0) for v in data["volume"]]
    data["dmn"] = [dmn_values.get(v, 0.0) for v in data["volume"]]

    return pd.DataFrame(data)


# =============================================================================
# Comparison Functions
# =============================================================================


def compare_outputs(
    pineuro_log: Path,
    murfi_log: Path,
    output_dir: Path,
    n_volumes: int,
) -> dict:
    """Load both outputs and compute comparison metrics."""
    pineuro_df = pd.read_csv(pineuro_log, sep="\t")
    murfi_df = parse_murfi_log(murfi_log, n_volumes)

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print("=" * 60)

    results = {}
    for pn_col, murfi_col in [("ContA", "cen"), ("DefaultA", "dmn")]:
        pn = pineuro_df[pn_col].values
        mu = murfi_df[murfi_col].values

        mask = (pn != 0) | (mu != 0)
        pn_valid = pn[mask]
        mu_valid = mu[mask]

        if len(pn_valid) == 0:
            print(f"\n{pn_col}: No valid values!")
            continue

        corr = np.corrcoef(pn_valid, mu_valid)[0, 1]
        slope, intercept = np.polyfit(mu_valid, pn_valid, 1)
        rmse = np.sqrt(np.mean((pn_valid - mu_valid) ** 2))

        results[pn_col] = {
            "correlation": corr,
            "slope": slope,
            "intercept": intercept,
            "rmse": rmse,
            "n_values": len(pn_valid),
        }

        print(f"\n{pn_col} vs {murfi_col}:")
        print(f"  Correlation: {corr:.6f}")
        print(f"  Slope:       {slope:.6f}")
        print(f"  Intercept:   {intercept:.6f}")
        print(f"  RMSE:        {rmse:.6f}")
        print(f"  N values:    {len(pn_valid)}")

    # Compute difference correlation
    pn_diff = pineuro_df["ContA"].values - pineuro_df["DefaultA"].values
    mu_diff = murfi_df["cen"].values - murfi_df["dmn"].values

    diff_mask = (pn_diff != 0) | (mu_diff != 0)
    pn_diff_valid = pn_diff[diff_mask]
    mu_diff_valid = mu_diff[diff_mask]

    diff_corr = np.corrcoef(pn_diff_valid, mu_diff_valid)[0, 1] if len(pn_diff_valid) > 0 else 0.0
    results["Difference"] = {"correlation": diff_corr, "n_values": len(pn_diff_valid)}
    print(f"\n(ContA - DefaultA) vs (cen - dmn):")
    print(f"  Correlation: {diff_corr:.6f}")

    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    volumes = pineuro_df["volume"].values

    ax1 = axes[0]
    ax1.plot(volumes, pineuro_df["DefaultA"].values, label="pineuro (DefaultA)", alpha=0.8)
    ax1.plot(volumes, murfi_df["dmn"].values, label="MURFI (dmn)", alpha=0.8, linestyle="--")
    ax1.set_ylabel("Activation")
    ax1.set_title(f"DefaultA vs dmn (r = {results.get('DefaultA', {}).get('correlation', 0):.4f})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(volumes, pineuro_df["ContA"].values, label="pineuro (ContA)", alpha=0.8)
    ax2.plot(volumes, murfi_df["cen"].values, label="MURFI (cen)", alpha=0.8, linestyle="--")
    ax2.set_ylabel("Activation")
    ax2.set_title(f"ContA vs cen (r = {results.get('ContA', {}).get('correlation', 0):.4f})")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(volumes, pn_diff, label="pineuro (ContA - DefaultA)", alpha=0.8)
    ax3.plot(volumes, mu_diff, label="MURFI (cen - dmn)", alpha=0.8, linestyle="--")
    ax3.set_xlabel("Volume")
    ax3.set_ylabel("Activation Difference")
    ax3.set_title(f"Difference (r = {results.get('Difference', {}).get('correlation', 0):.4f})")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # Save report
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("MURFI vs pineuro Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"N volumes: {n_volumes}\n\n")

        for network, metrics in results.items():
            f.write(f"{network}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    print(f"Report saved: {report_path}")

    return results


# =============================================================================
# Feedback Stage Functions
# =============================================================================


def find_masks(subject_dir: Path) -> dict[str, Path]:
    """Find coregistered masks for a subject."""
    mask_dir = get_mask_extraction_dir(subject_dir) / "04_2vol"
    masks = {}

    for mask_path in mask_dir.glob("*_thr_to_task-2vol*.nii.gz"):
        name = mask_path.name
        if "_ContA_" in name:
            masks["ContA"] = mask_path
        elif "_DefaultA_" in name:
            masks["DefaultA"] = mask_path

    if len(masks) != 2:
        raise FileNotFoundError(f"Could not find both masks in {mask_dir}")

    return masks


def run_pineuro_feedback(
    subject: str,
    run: int,
    dicom_dir: Path,
    masks: dict[str, Path],
    output_dir: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
) -> Path:
    """Run pineuro feedback session."""
    from pineuro import run_feedback_session, compute_boxcar_task_regressor

    feedback_dir = output_dir / "02_feedback_output"
    collection_dir = output_dir / "collection"

    feedback_dir.mkdir(parents=True, exist_ok=True)
    collection_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning pineuro feedback run-{run:02d}...")
    print(f"  DICOM source: {dicom_dir}")
    print(f"  Output: {feedback_dir}")

    task_regressor = compute_boxcar_task_regressor(
        n_baseline=n_baseline,
        n_volumes=n_volumes,
        tr=tr,
    )

    result = run_feedback_session(
        mask_paths=[masks["ContA"], masks["DefaultA"]],
        mask_names=["ContA", "DefaultA"],
        subject=subject,
        task="feedback",
        run=run,
        output_dir=feedback_dir,
        collection_dir=collection_dir,
        n_baseline=n_baseline,
        n_volumes=n_volumes,
        tr=tr,
        task_regressors=task_regressor.reshape(-1, 1),
        simulator_source_dir=dicom_dir,
        simulator_speed=100.0,
        moco_enabled=True,
        qc_enabled=False,
    )

    coordinator = result["coordinator"]
    coordinator.wait_until_ready(timeout=10)
    print("  Sending trigger...")
    coordinator.send_trigger()

    thread = result["thread"]
    thread.join()

    feedbacklog = feedback_dir / f"{subject}_task-feedback_run-{run:02d}_feedbacklog.tsv"
    print(f"pineuro complete: {feedbacklog}")

    return feedbacklog


def run_comparison_for_run(
    subject: str,
    run: int,
    subject_dir: Path,
    murfi_sif: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
    skip_existing: bool = True,
) -> Optional[dict]:
    """Run comparison for a single feedback run."""
    collection_dir = get_collection_dir(subject_dir)
    feedback_dir = get_feedback_dir(subject_dir)
    murfi_run_dir = get_murfi_dir(subject_dir) / f"run-{run:02d}"
    comparison_run_dir = get_comparison_dir(subject_dir) / f"run-{run:02d}"

    dicom_dir = collection_dir / f"feedback_run-{run:02d}"

    # Check if comparison already done
    if skip_existing and check_comparison_complete(subject_dir, run):
        print(f"\n  Run {run:02d}: Comparison already complete, skipping")
        return None

    # Find masks
    masks = find_masks(subject_dir)

    print(f"\n{'=' * 60}")
    print(f"RUN {run:02d}")
    print("=" * 60)

    # Run pineuro feedback
    pineuro_log = feedback_dir / "02_feedback_output" / f"{subject}_task-feedback_run-{run:02d}_feedbacklog.tsv"
    if not pineuro_log.exists():
        pineuro_log = run_pineuro_feedback(
            subject=subject,
            run=run,
            dicom_dir=dicom_dir,
            masks=masks,
            output_dir=feedback_dir,
            tr=tr,
            n_volumes=n_volumes,
            n_baseline=n_baseline,
        )
    else:
        print(f"\npineuro feedback already complete: {pineuro_log}")

    # Setup and run MURFI
    murfi_log = murfi_run_dir / subject / "log" / "log.rtl"
    if not check_murfi_run_complete(subject_dir, run, n_volumes):
        setup_murfi_environment(
            subject=subject,
            masks=masks,
            run_dir=murfi_run_dir,
            tr=tr,
            n_volumes=n_volumes,
            n_baseline=n_baseline,
        )
        murfi_log = run_murfi(
            subject=subject,
            dicom_dir=dicom_dir,
            murfi_sif=murfi_sif,
            run_dir=murfi_run_dir,
            tr=tr,
            n_volumes=n_volumes,
        )
    else:
        print(f"\nMURFI already complete: {murfi_log}")

    # Compare outputs
    results = compare_outputs(pineuro_log, murfi_log, comparison_run_dir, n_volumes)

    # Cleanup intermediate files
    print("\nCleaning up intermediate files...")
    cleanup_dirs = [
        feedback_dir / "collection",
        feedback_dir / "02_feedback_output" / f"nifti_task-feedback_run-{run:02d}",
        murfi_run_dir / "dicom_source",
    ]
    for cleanup_dir in cleanup_dirs:
        if cleanup_dir.exists():
            shutil.rmtree(cleanup_dir)
            print(f"  Deleted: {cleanup_dir}")

    return {"subject": subject, "run": run, "results": results}


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_setup(args: argparse.Namespace) -> None:
    """Run setup stage for all subjects."""
    configs = load_subjects_csv(args.csv)

    print("=" * 60)
    print("SETUP STAGE")
    print("=" * 60)
    print(f"CSV:        {args.csv}")
    print(f"Output:     {args.output_dir}")
    print(f"Subjects:   {len(configs)}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        try:
            run_setup_for_subject(config, args.output_dir, args.overwrite)
        except Exception as e:
            print(f"\nERROR processing {config.subject}: {e}")
            continue

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Open pineuro-gui")
    print("2. Navigate to the output directory")
    print("3. Review mask extraction QC reports for each participant")
    print("4. Approve masks (creates .masks_approved flag)")
    print("5. Run: python compare_murfi_pineuro.py compare ...")


def cmd_status(args: argparse.Namespace) -> None:
    """Check status of all subjects."""
    if not args.output_dir.exists():
        print(f"Output directory not found: {args.output_dir}")
        return

    print("=" * 60)
    print("STATUS")
    print("=" * 60)

    subjects = sorted([d.name for d in args.output_dir.iterdir() if d.is_dir()])

    if not subjects:
        print("No subjects found.")
        return

    for subject in subjects:
        subject_dir = args.output_dir / subject
        status = get_subject_status(subject_dir, subject, args.n_runs, args.n_volumes)

        setup_done = status["mask_extraction_complete"]
        approved = status["masks_approved"]
        fb_done = sum(status["feedback_runs"])
        comp_done = sum(status["comparisons"])
        n_runs = len(status["feedback_runs"])

        status_str = []
        if not setup_done:
            status_str.append("needs setup")
        elif not approved:
            status_str.append("needs approval")
        elif fb_done < n_runs or comp_done < n_runs:
            status_str.append(f"compare: {comp_done}/{n_runs}")
        else:
            status_str.append("complete")

        print(f"{subject}: {', '.join(status_str)}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison stage for all subjects."""
    configs = load_subjects_csv(args.csv)

    print("=" * 60)
    print("COMPARE STAGE")
    print("=" * 60)
    print(f"CSV:        {args.csv}")
    print(f"Output:     {args.output_dir}")
    print(f"MURFI SIF:  {args.murfi_sif}")
    print(f"Subjects:   {len(configs)}")
    print()

    all_results = []

    for config in configs:
        subject_dir = get_subject_dir(args.output_dir, config.subject)

        # Check if masks are approved
        if not args.skip_approval_check and not check_masks_approved(subject_dir):
            print(f"\n{config.subject}: Masks not approved, skipping")
            print("  Run GUI to approve masks, or use --skip-approval-check")
            continue

        print(f"\n{'#' * 60}")
        print(f"# COMPARE: {config.subject}")
        print(f"{'#' * 60}")

        # Run comparison for each feedback run
        for run in range(1, len(config.feedback_runs) + 1):
            try:
                result = run_comparison_for_run(
                    subject=config.subject,
                    run=run,
                    subject_dir=subject_dir,
                    murfi_sif=args.murfi_sif,
                    tr=args.tr,
                    n_volumes=args.n_volumes,
                    n_baseline=args.n_baseline,
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"\nERROR in {config.subject} run-{run:02d}: {e}")
                continue

    # Print summary
    print_summary(all_results)


def cmd_full(args: argparse.Namespace) -> None:
    """Run full pipeline (setup + compare)."""
    # Run setup
    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE")
    print("=" * 60)

    cmd_setup(args)

    # Auto-approve masks if requested
    if args.skip_approval_check:
        print("\n--skip-approval-check: Proceeding without mask approval")

    # Run compare
    cmd_compare(args)


def print_summary(all_results: list[dict]) -> None:
    """Print final summary of comparison results."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if not all_results:
        print("\nNo new comparisons were run.")
        return

    all_good = True
    for r in all_results:
        print(f"\n{r['subject']} run-{r['run']:02d}:")
        for network, metrics in r["results"].items():
            corr = metrics["correlation"]
            status = "PASS" if corr > 0.98 else "WARN" if corr > 0.9 else "FAIL"
            if corr < 0.98:
                all_good = False
            print(f"  {network}: r={corr:.6f} [{status}]")

    print()
    if all_good:
        print("All correlations > 0.98 - MURFI and pineuro match!")
    else:
        print("Some correlations < 0.98 - investigate differences")

    # Compute statistics
    print("\n" + "-" * 60)
    print("CORRELATION STATISTICS")
    print("-" * 60)

    corr_by_network: dict[str, list[float]] = {"ContA": [], "DefaultA": [], "Difference": []}
    for r in all_results:
        for network in corr_by_network.keys():
            if network in r["results"]:
                corr_by_network[network].append(r["results"][network]["correlation"])

    for network, corrs in corr_by_network.items():
        if not corrs:
            continue
        arr = np.array(corrs)
        print(f"\n{network} (n={len(arr)}):")
        print(f"  Mean:   {arr.mean():.6f}")
        print(f"  Median: {np.median(arr):.6f}")
        print(f"  Std:    {arr.std():.6f}")
        print(f"  Min:    {arr.min():.6f}")
        print(f"  Max:    {arr.max():.6f}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MURFI and pineuro feedback outputs (GUI-compatible workflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Run setup stage (copy DICOMs, preprocess, extract masks)",
    )
    setup_parser.add_argument("--csv", type=Path, required=True, help="Path to subjects CSV")
    setup_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    setup_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of all subjects")
    status_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    status_parser.add_argument("--n-runs", type=int, default=5, help="Number of feedback runs")
    status_parser.add_argument("--n-volumes", type=int, default=150, help="Volumes per run")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run comparison stage (feedback + MURFI + compare)",
    )
    compare_parser.add_argument("--csv", type=Path, required=True, help="Path to subjects CSV")
    compare_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    compare_parser.add_argument("--murfi-sif", type=Path, required=True, help="MURFI apptainer image")
    compare_parser.add_argument("--tr", type=float, default=1.2, help="TR in seconds")
    compare_parser.add_argument("--n-volumes", type=int, default=150, help="Volumes per run")
    compare_parser.add_argument("--n-baseline", type=int, default=25, help="Baseline volumes")
    compare_parser.add_argument(
        "--skip-approval-check",
        action="store_true",
        help="Skip mask approval check",
    )

    # Full command
    full_parser = subparsers.add_parser("full", help="Run full pipeline (setup + compare)")
    full_parser.add_argument("--csv", type=Path, required=True, help="Path to subjects CSV")
    full_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    full_parser.add_argument("--murfi-sif", type=Path, required=True, help="MURFI apptainer image")
    full_parser.add_argument("--tr", type=float, default=1.2, help="TR in seconds")
    full_parser.add_argument("--n-volumes", type=int, default=150, help="Volumes per run")
    full_parser.add_argument("--n-baseline", type=int, default=25, help="Baseline volumes")
    full_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    full_parser.add_argument(
        "--skip-approval-check",
        action="store_true",
        help="Skip mask approval check",
    )

    args = parser.parse_args()

    # Resolve paths
    if hasattr(args, "csv") and args.csv:
        args.csv = args.csv.resolve()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.resolve()
    if hasattr(args, "murfi_sif") and args.murfi_sif:
        args.murfi_sif = args.murfi_sif.resolve()

    # Run command
    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "full":
        cmd_full(args)


if __name__ == "__main__":
    main()
