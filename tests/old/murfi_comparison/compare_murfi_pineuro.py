#!/usr/bin/env python3
"""Compare MURFI and pineuro feedback outputs on the same input data.

This script runs both pineuro and MURFI on the same DICOM data and compares
their activation outputs to verify matching behavior.

Usage:
    python compare_murfi_pineuro.py \\
        --csv test_subjects.csv \\
        --output-dir tests/murfi_comparison/outputs

CSV format:
    subject,feedback_run,rest1,rest2,2vol,feedback
    sub-remind3007,1,/path/to/rest1,/path/to/rest2,/path/to/2vol,/path/to/feedback_dicoms
"""

import argparse
import gzip
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add pineuro to path if running from repo
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class RunConfig:
    """Configuration for a single feedback run comparison."""
    subject: str
    feedback_run: int
    rest_runs: list[Path]
    twovol_dir: Path
    feedback_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare MURFI and pineuro feedback outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_murfi_pineuro.py \\
      --csv test_subjects.csv \\
      --output-dir tests/murfi_comparison/outputs

CSV format (columns):
  subject, feedback_run, rest1, rest2, 2vol, feedback
""",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV file with test subjects",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write outputs",
    )

    # MURFI options
    parser.add_argument(
        "--murfi-sif",
        type=Path,
        required=True,
        help="Path to MURFI apptainer image",
    )

    # Scan parameters
    parser.add_argument(
        "--tr",
        type=float,
        default=1.2,
        help="TR in seconds (default: 1.2)",
    )
    parser.add_argument(
        "--n-volumes",
        type=int,
        default=150,
        help="Number of volumes (default: 150)",
    )
    parser.add_argument(
        "--n-baseline",
        type=int,
        default=25,
        help="Baseline volumes (default: 25)",
    )
    parser.add_argument(
        "--regenerate-reports",
        action="store_true",
        help="Regenerate comparison reports even if they exist",
    )

    return parser.parse_args()


def load_csv(csv_path: Path) -> list[RunConfig]:
    """Load test configurations from CSV file."""
    df = pd.read_csv(csv_path)

    configs = []
    for _, row in df.iterrows():
        config = RunConfig(
            subject=row["subject"],
            feedback_run=int(row["feedback_run"]),
            rest_runs=[Path(row["rest1"]), Path(row["rest2"])],
            twovol_dir=Path(row["2vol"]),
            feedback_dir=Path(row["feedback"]),
        )
        configs.append(config)

    return configs


def find_mask(mask_dir: Path, network: str) -> Path:
    """Find mask file for a network in the mask directory."""
    # Look for pattern like sub-remind3045_ContA_thr_to_task-2vol_run-XX_mean.nii.gz
    matches = list(mask_dir.glob(f"*_{network}_thr_*.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"No mask found for {network} in {mask_dir}")
    if len(matches) > 1:
        print(f"Warning: Multiple masks found for {network}, using {matches[0]}")
    return matches[0]


def run_preprocessing(
    subject: str, rest_runs: list[Path], output_dir: Path
) -> tuple[Path, Path, Path]:
    """Run preprocessing on resting-state runs.

    Returns:
        Tuple of (preproc_file, mask_file, ref_file)
    """
    from pineuro import preprocessing

    preproc_dir = output_dir / "preprocessing"

    print(f"\n{'=' * 60}")
    print("PREPROCESSING")
    print("=" * 60)
    print(f"  Rest runs: {[str(p) for p in rest_runs]}")
    print(f"  Output: {preproc_dir}")

    preproc_file, mask_file, ref_file = preprocessing(
        participant_id=subject,
        task="rest",
        root_dir=preproc_dir,
        input_dicoms=rest_runs,
        overwrite=False,
        qc_enabled=True,
    )

    print(f"\nPreprocessing complete:")
    print(f"  Preprocessed: {preproc_file}")
    print(f"  Mask: {mask_file}")
    print(f"  Reference: {ref_file}")

    return preproc_file, mask_file, ref_file


def run_mask_extraction(
    subject: str, output_dir: Path, preproc_file: Path, mask_file: Path, ref_file: Path
) -> list[Path]:
    """Run ICA-based mask extraction.

    Returns:
        List of extracted mask paths
    """
    from pineuro import mask_extraction_rest

    mask_dir = output_dir / "mask_extraction"

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
        overwrite=False,
        qc_enabled=True,
    )

    print(f"\nMask extraction complete:")
    for mf in mask_files:
        print(f"  {mf}")

    return mask_files


def run_2vol_coregistration(
    subject: str, twovol_dir: Path, output_dir: Path, mask_files: list[Path], ref_file: Path
) -> dict[str, Path]:
    """Coregister masks to feedback scanner position.

    Returns:
        Dict mapping network name to coregistered mask path
    """
    from pineuro import coregister_2vol

    feedback_dir = output_dir / "masks"

    print(f"\n{'=' * 60}")
    print("2VOL COREGISTRATION")
    print("=" * 60)
    print(f"  2vol DICOM: {twovol_dir}")
    print(f"  Output: {feedback_dir}")

    coreg_masks = coregister_2vol(
        participant_id=subject,
        root_dir=feedback_dir,
        input_dicom=twovol_dir,
        network_files=mask_files,
        ref_file=ref_file,
        overwrite=False,
        output_subdir="01_2vol",
    )

    print(f"\nCoregistration complete:")
    for mf in coreg_masks:
        print(f"  {mf}")

    # Build dict mapping network name to path
    masks = {}
    for mask_path in coreg_masks:
        # Extract network name from path like sub-01_ContA_thr_to_task-2vol_run-01_mean.nii.gz
        name = mask_path.name
        if "_ContA_" in name:
            masks["ContA"] = mask_path
        elif "_DefaultA_" in name:
            masks["DefaultA"] = mask_path

    return masks


def generate_masks(
    subject: str, rest_runs: list[Path], twovol_dir: Path, output_dir: Path
) -> dict[str, Path]:
    """Generate masks from resting-state runs.

    Pipeline:
        1. Preprocess resting-state data
        2. Extract network masks via ICA
        3. Coregister masks to feedback scanner position (2vol)

    Returns:
        Dict mapping network name ('ContA', 'DefaultA') to mask path
    """
    # Step 1: Preprocessing
    preproc_file, mask_file, ref_file = run_preprocessing(subject, rest_runs, output_dir)

    # Step 2: Mask extraction
    mask_files = run_mask_extraction(subject, output_dir, preproc_file, mask_file, ref_file)

    # Step 3: 2vol coregistration
    return run_2vol_coregistration(subject, twovol_dir, output_dir, mask_files, ref_file)


def copy_and_decompress_mask(src: Path, dst: Path) -> None:
    """Copy a .nii.gz mask to .nii (uncompressed) for MURFI."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".gz":
        with gzip.open(src, "rb") as f_in:
            with open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(src, dst)

    print(f"Copied mask: {src.name} -> {dst.name}")


def copy_and_convert_mask(src: Path, dst: Path) -> None:
    """Copy a mask to uncompressed int16 NIfTI for MURFI.

    MURFI expects masks as int16 (short) data type.
    """
    import nibabel as nib

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Load source mask
    img = nib.load(src)
    data = img.get_fdata()

    # Convert to int16
    data_int16 = data.astype(np.int16)

    # Create new image with int16 dtype
    new_img = nib.Nifti1Image(data_int16, img.affine, img.header)
    new_img.header.set_data_dtype(np.int16)

    # Save uncompressed
    nib.save(new_img, dst)

    print(f"Converted mask: {src.name} -> {dst.name} (int16)")


def generate_murfi_xml(
    xml_path: Path, tr: float, n_volumes: int, n_baseline: int
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

<!-- processing steps (every TR)-->
<processor>
  <!-- creates a mosaic for the GUI -->
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

  <!-- dmn mask -->
  <module name="mask-load">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="roiID"> dmn </option>
    <option name="filename"> dmn </option>
    <option name="align"> false </option>
    <option name="save"> true </option>
  </module>

  <!-- cen mask -->
  <module name="mask-load">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="roiID"> cen </option>
    <option name="filename"> cen </option>
    <option name="align"> false </option>
    <option name="save"> true </option>
  </module>

  <!-- incremental GLM -->
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

  <!-- current activation -->
  <module name="current-activation">
    <option name="disabled"> false </option>
    <option name="makeavail">true</option>
    <option name="print"> true </option>
    <option name="modelFitModuleID"> incremental-glm </option>
    <option name="modelFitRoiID"> brain </option>
    <option name="maskRoiID"> brain </option>
    <option name="numDataPointsForErrEst"> {n_baseline} </option>
  </module>

  <!-- ROI combine for DMN -->
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

  <!-- ROI combine for CEN -->
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

    print(f"Generated MURFI XML: {xml_path}")


def run_pineuro_feedback(
    subject: str,
    feedback_run: int,
    dicom_dir: Path,
    masks: dict[str, Path],
    output_dir: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
) -> Path:
    """Run pineuro with simulation, return feedbacklog path."""
    from pineuro import run_feedback_session, compute_boxcar_task_regressor

    print(f"Using masks:")
    print(f"  ContA: {masks['ContA']}")
    print(f"  DefaultA: {masks['DefaultA']}")

    pineuro_dir = output_dir / "pineuro"
    collection_dir = output_dir / "pineuro_collection"

    pineuro_dir.mkdir(parents=True, exist_ok=True)
    collection_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning pineuro feedback...")
    print(f"  DICOM source: {dicom_dir}")
    print(f"  Output dir: {pineuro_dir}")

    # MURFI uses all 1s for the task condition (continuous "on" after baseline)
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
        run=feedback_run,
        output_dir=pineuro_dir,
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
    print("  Coordinator ready, sending trigger...")
    coordinator.send_trigger()

    thread = result["thread"]
    thread.join()

    feedbacklog = pineuro_dir / f"{subject}_task-feedback_run-{feedback_run:02d}_feedbacklog.tsv"
    print(f"pineuro complete: {feedbacklog}")

    return feedbacklog


def setup_murfi_environment(
    subject: str,
    masks: dict[str, Path],
    output_dir: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
) -> Path:
    """Create MURFI directory structure and copy masks."""
    murfi_dir = output_dir / "murfi" / subject

    (murfi_dir / "mask").mkdir(parents=True, exist_ok=True)
    (murfi_dir / "xml").mkdir(exist_ok=True)
    (murfi_dir / "log").mkdir(exist_ok=True)

    copy_and_convert_mask(masks["ContA"], murfi_dir / "mask" / "cen.nii")
    copy_and_convert_mask(masks["DefaultA"], murfi_dir / "mask" / "dmn.nii")

    generate_murfi_xml(murfi_dir / "xml" / "rtdmn.xml", tr, n_volumes, n_baseline)

    return murfi_dir


def run_murfi(
    subject: str,
    dicom_dir: Path,
    murfi_sif: Path,
    output_dir: Path,
    tr: float,
    n_volumes: int,
) -> Path:
    """Run MURFI in background, serve DICOMs, wait for completion."""
    murfi_base_dir = output_dir / "murfi"
    subject_dir = murfi_base_dir / subject
    xml_path = subject_dir / "xml" / "rtdmn.xml"
    log_path = subject_dir / "log" / "log.rtl"

    murfi_input_dir = Path("tmp/murfi_input")
    murfi_input_dir.mkdir(parents=True, exist_ok=True)

    for f in murfi_input_dir.glob("*"):
        f.unlink()

    # Copy DICOMs to a path without spaces (apptainer/servedicoms can't handle spaces)
    dicom_copy_dir = output_dir / "dicom_source"
    if dicom_copy_dir.exists():
        shutil.rmtree(dicom_copy_dir)
    print(f"\nCopying DICOMs to path without spaces...")
    print(f"  From: {dicom_dir}")
    print(f"  To: {dicom_copy_dir}")
    shutil.copytree(dicom_dir, dicom_copy_dir)
    dicom_dir_for_murfi = dicom_copy_dir

    env = os.environ.copy()
    env["MURFI_SUBJECTS_DIR"] = str(murfi_base_dir.absolute())
    env["MURFI_SUBJECT_NAME"] = subject

    print(f"\nStarting MURFI...")
    print(f"  MURFI_SUBJECTS_DIR={env['MURFI_SUBJECTS_DIR']}")
    print(f"  MURFI_SUBJECT_NAME={env['MURFI_SUBJECT_NAME']}")
    print(f"  XML config: {xml_path}")

    murfi_cmd = [
        "apptainer",
        "exec",
        str(murfi_sif),
        "murfi",
        "-f",
        str(xml_path.absolute()),
    ]

    print(f"  Command: {' '.join(murfi_cmd)}")

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
        print(f"MURFI failed to start! (exit code: {murfi_proc.returncode})")
        raise RuntimeError("MURFI failed to start")

    serve_cmd = [
        "apptainer",
        "exec",
        str(murfi_sif),
        "servedicoms",
        str(dicom_dir_for_murfi.absolute()),
        "tmp/murfi_input",
        str(int(tr * 1000)),
    ]

    print(f"\nServing DICOMs...")
    print(f"  Command: {' '.join(serve_cmd)}")

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
        print(f"  servedicoms complete (exit code: {serve_proc.returncode})")
    except subprocess.TimeoutExpired:
        print(f"  servedicoms timed out - killing it")
        serve_proc.kill()
        try:
            os.killpg(os.getpgid(serve_proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # Brief wait for MURFI to process any remaining buffered volumes
    print("  Waiting for MURFI to finish processing...")
    time.sleep(3)

    # Force kill MURFI and any child processes
    print("  Killing MURFI...")

    # Kill the specific PID first
    try:
        os.kill(murfi_proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    # Kill the entire process group
    try:
        os.killpg(os.getpgid(murfi_proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        murfi_proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass

    # Aggressively kill any remaining MURFI processes
    # Be careful not to kill this Python script (path contains "murfi_comparison")
    subprocess.run(["pkill", "-9", "-f", "murfi -f"], capture_output=True)  # MURFI binary with -f flag
    subprocess.run(["pkill", "-9", "-f", "apptainer.*murfi.sif"], capture_output=True)
    time.sleep(0.5)

    print(f"MURFI complete: {log_path}")

    return log_path


def parse_murfi_log(log_path: Path, n_volumes: int) -> pd.DataFrame:
    """Parse MURFI log.rtl for activation values.

    MURFI log format example:
        RtRoiCombine: dmn at tr 9 result is: 0.609096
        RtRoiCombine: cen at tr 9 result is: -0.609096

    Notes:
    - Values start at tr 9 (after 8 baseline volumes with no output)
    - dmn and cen appear on separate lines
    - Log may contain multiple runs; we want the LAST complete run
    - TR numbers reset to 1 at start of each run
    """
    pattern = r"RtRoiCombine: (dmn|cen) at tr (\d+) result is: ([0-9e.+-]+)"

    # Collect all matches
    all_matches = []
    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                roi, tr, value = match.groups()
                all_matches.append((roi, int(tr), float(value)))

    if not all_matches:
        raise ValueError(f"No RtRoiCombine entries found in {log_path}")

    # Detect run boundaries: TR resets to small value after reaching high TR
    runs: list[list[tuple[str, int, float]]] = []
    current_run: list[tuple[str, int, float]] = []
    prev_tr = 0

    for roi, tr, value in all_matches:
        # Detect run boundary: TR jumped backwards significantly
        if tr < prev_tr - 10:  # New run started
            if current_run:
                runs.append(current_run)
            current_run = []
        current_run.append((roi, tr, value))
        prev_tr = tr

    if current_run:
        runs.append(current_run)

    print(f"Found {len(runs)} run(s) in MURFI log")

    # Use the last complete run
    last_run = runs[-1]

    # Extract cen and dmn values by TR
    cen_values: dict[int, float] = {}
    dmn_values: dict[int, float] = {}

    for roi, tr, value in last_run:
        if roi == "cen":
            cen_values[tr] = value
        else:
            dmn_values[tr] = value

    # Build DataFrame with all volumes (1 to n_volumes)
    # Volumes 1-8 have no output (baseline), fill with 0
    data = {"volume": list(range(1, n_volumes + 1))}
    data["cen"] = [cen_values.get(v, 0.0) for v in data["volume"]]
    data["dmn"] = [dmn_values.get(v, 0.0) for v in data["volume"]]

    df = pd.DataFrame(data)
    print(f"Parsed {len(cen_values)} cen values, {len(dmn_values)} dmn values")

    return df


def compare_outputs(
    pineuro_log: Path, murfi_log: Path, output_dir: Path, n_volumes: int
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

        # Skip volumes where both are zero (baseline)
        mask = (pn != 0) | (mu != 0)
        pn_valid = pn[mask]
        mu_valid = mu[mask]

        if len(pn_valid) == 0:
            print(f"\n{pn_col}: No valid values to compare!")
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

    # Compute difference correlation (ContA - DefaultA) vs (cen - dmn)
    pn_conta = pineuro_df["ContA"].values
    pn_default = pineuro_df["DefaultA"].values
    mu_cen = murfi_df["cen"].values
    mu_dmn = murfi_df["dmn"].values

    pn_diff = pn_conta - pn_default
    mu_diff = mu_cen - mu_dmn

    # Mask for valid values (where either difference is non-zero)
    diff_mask = (pn_diff != 0) | (mu_diff != 0)
    pn_diff_valid = pn_diff[diff_mask]
    mu_diff_valid = mu_diff[diff_mask]

    diff_corr = np.corrcoef(pn_diff_valid, mu_diff_valid)[0, 1] if len(pn_diff_valid) > 0 else 0.0
    results["Difference"] = {"correlation": diff_corr, "n_values": len(pn_diff_valid)}
    print(f"\n(ContA - DefaultA) vs (cen - dmn):")
    print(f"  Correlation: {diff_corr:.6f}")
    print(f"  N values:    {len(pn_diff_valid)}")

    # Create comparison plot with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    volumes = pineuro_df["volume"].values

    # Plot 1: DefaultA vs dmn
    ax1 = axes[0]
    ax1.plot(volumes, pineuro_df["DefaultA"].values, label="pineuro (DefaultA)", alpha=0.8)
    ax1.plot(volumes, murfi_df["dmn"].values, label="MURFI (dmn)", alpha=0.8, linestyle="--")
    ax1.set_ylabel("Activation")
    ax1.set_title(f"DefaultA vs dmn (r = {results.get('DefaultA', {}).get('correlation', 0):.4f})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: ContA vs cen
    ax2 = axes[1]
    ax2.plot(volumes, pineuro_df["ContA"].values, label="pineuro (ContA)", alpha=0.8)
    ax2.plot(volumes, murfi_df["cen"].values, label="MURFI (cen)", alpha=0.8, linestyle="--")
    ax2.set_ylabel("Activation")
    ax2.set_title(f"ContA vs cen (r = {results.get('ContA', {}).get('correlation', 0):.4f})")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Difference (ContA - DefaultA) vs (cen - dmn)
    ax3 = axes[2]
    ax3.plot(volumes, pn_diff, label="pineuro (ContA - DefaultA)", alpha=0.8)
    ax3.plot(volumes, mu_diff, label="MURFI (cen - dmn)", alpha=0.8, linestyle="--")
    ax3.set_xlabel("Volume")
    ax3.set_ylabel("Activation Difference")
    ax3.set_title(f"Difference: (ContA - DefaultA) vs (cen - dmn) (r = {results.get('Difference', {}).get('correlation', 0):.4f})")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # Save report
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("MURFI vs pineuro Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Subject: {pineuro_log.parent.name}\n")
        f.write(f"N volumes: {n_volumes}\n\n")

        for network, metrics in results.items():
            f.write(f"{network}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    print(f"\nReport saved: {report_path}")

    return results


def check_pineuro_complete(log_path: Path, n_volumes: int) -> bool:
    """Check if pineuro log has all expected volumes."""
    if not log_path.exists():
        return False
    try:
        df = pd.read_csv(log_path, sep="\t")
        return len(df) >= n_volumes
    except Exception:
        return False


def check_murfi_complete(log_path: Path, n_volumes: int) -> bool:
    """Check if MURFI log has all expected volumes."""
    if not log_path.exists():
        return False
    try:
        pattern = r"RtRoiCombine: (dmn|cen) at tr (\d+) result is:"
        max_tr = 0
        with open(log_path) as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    tr = int(match.group(2))
                    max_tr = max(max_tr, tr)
        # MURFI should have entries up to n_volumes
        return max_tr >= n_volumes
    except Exception:
        return False


def run_single_comparison(
    config: RunConfig,
    masks: dict[str, Path],
    args: argparse.Namespace,
) -> dict | None:
    """Run comparison for a single feedback run."""
    run_output_dir = args.output_dir / config.subject / f"run-{config.feedback_run:02d}"
    report_path = run_output_dir / "comparison_report.txt"

    # Expected output paths
    pineuro_log = (
        run_output_dir / "pineuro" / f"{config.subject}_task-feedback_run-{config.feedback_run:02d}_feedbacklog.tsv"
    )
    murfi_log = run_output_dir / "murfi" / config.subject / "log" / "log.rtl"

    # Skip entirely if report exists (unless --regenerate-reports)
    if report_path.exists() and not args.regenerate_reports:
        print(f"\n{'#' * 60}")
        print(f"# {config.subject} run-{config.feedback_run} - SKIPPED (report exists)")
        print(f"{'#' * 60}")
        return None

    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# {config.subject} run-{config.feedback_run}")
    print(f"{'#' * 60}")

    # Run pineuro (skip if complete)
    if check_pineuro_complete(pineuro_log, args.n_volumes):
        print(f"\npineuro complete ({args.n_volumes} volumes), skipping: {pineuro_log}")
    else:
        pineuro_log = run_pineuro_feedback(
            subject=config.subject,
            feedback_run=config.feedback_run,
            dicom_dir=config.feedback_dir,
            masks=masks,
            output_dir=run_output_dir,
            tr=args.tr,
            n_volumes=args.n_volumes,
            n_baseline=args.n_baseline,
        )

    # Setup and run MURFI (skip if complete)
    if check_murfi_complete(murfi_log, args.n_volumes):
        print(f"\nMURFI complete ({args.n_volumes} volumes), skipping: {murfi_log}")
    else:
        setup_murfi_environment(
            subject=config.subject,
            masks=masks,
            output_dir=run_output_dir,
            tr=args.tr,
            n_volumes=args.n_volumes,
            n_baseline=args.n_baseline,
        )
        murfi_log = run_murfi(
            subject=config.subject,
            dicom_dir=config.feedback_dir,
            murfi_sif=args.murfi_sif,
            output_dir=run_output_dir,
            tr=args.tr,
            n_volumes=args.n_volumes,
        )

    # Compare outputs
    results = compare_outputs(pineuro_log, murfi_log, run_output_dir, args.n_volumes)

    # Cleanup intermediate files after successful comparison
    print("\nCleaning up intermediate files...")
    cleanup_dirs = [
        run_output_dir / "pineuro_collection",  # Incoming DICOMs for pineuro
        run_output_dir / "pineuro" / f"nifti_task-feedback_run-{config.feedback_run:02d}",  # NIfTI files
        run_output_dir / "dicom_source",  # Copied DICOMs for MURFI
    ]
    for cleanup_dir in cleanup_dirs:
        if cleanup_dir.exists():
            shutil.rmtree(cleanup_dir)
            print(f"  Deleted: {cleanup_dir}")

    return {
        "subject": config.subject,
        "run": config.feedback_run,
        "results": results,
    }


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Resolve paths
    args.csv = args.csv.resolve()
    args.output_dir = args.output_dir.resolve()
    args.murfi_sif = args.murfi_sif.resolve()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not args.murfi_sif.exists():
        raise FileNotFoundError(f"MURFI SIF not found: {args.murfi_sif}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configurations from CSV
    configs = load_csv(args.csv)

    print("=" * 60)
    print("MURFI vs pineuro Comparison")
    print("=" * 60)
    print(f"CSV:         {args.csv}")
    print(f"Output dir:  {args.output_dir}")
    print(f"TR:          {args.tr}s")
    print(f"N volumes:   {args.n_volumes}")
    print(f"N baseline:  {args.n_baseline}")
    print(f"Total runs:  {len(configs)}")

    # Group configs by subject
    subjects = {}
    for config in configs:
        if config.subject not in subjects:
            subjects[config.subject] = []
        subjects[config.subject].append(config)

    print(f"Subjects:    {list(subjects.keys())}")
    print()

    # Track all results
    all_results = []

    # Process each subject
    for subject, subject_configs in subjects.items():
        print(f"\n{'=' * 60}")
        print(f"SUBJECT: {subject}")
        print(f"{'=' * 60}")

        # Get first config for mask generation (rest runs and 2vol are same for all runs)
        first_config = subject_configs[0]
        subject_output_dir = args.output_dir / subject

        # Generate masks once per subject
        masks = generate_masks(
            subject=subject,
            rest_runs=first_config.rest_runs,
            twovol_dir=first_config.twovol_dir,
            output_dir=subject_output_dir,
        )
        print(f"\nMasks ready:")
        print(f"  ContA: {masks['ContA']}")
        print(f"  DefaultA: {masks['DefaultA']}")

        # Run comparison for each feedback run
        for config in subject_configs:
            result = run_single_comparison(config, masks, args)
            if result is not None:
                all_results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if not all_results:
        print("\nNo new comparisons were run (all reports already exist).")
    else:
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

        # Compute descriptive statistics for correlations
        print("\n" + "-" * 60)
        print("CORRELATION STATISTICS")
        print("-" * 60)

        # Collect correlations by network
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


if __name__ == "__main__":
    main()
