#!/usr/bin/env python3
"""Compare MURFI and pineuro feedback outputs.

Runs pineuro and MURFI on the same DICOMs and compares activation timeseries.
Requires setup_validation.py to have been run first (preprocessing + mask extraction).

Usage:
    # First: run shared setup
    python setup_validation.py setup --csv remind_test_data.csv --output-dir /data/output

    # Then: run comparison (after GUI mask approval)
    python compare_murfi_pineuro.py compare \
        --csv analysis/remind_test_data.csv \
        --output-dir /data/output \
        --murfi-sif /path/to/murfi.sif

    # Check status
    python compare_murfi_pineuro.py status --output-dir /data/output
"""

import argparse
import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add pineuro to path if running from repo
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from setup_validation import (
    check_mask_extraction_complete,
    check_masks_approved,
    get_collection_dir,
    get_feedback_dir,
    get_mask_extraction_dir,
    get_preprocessing_dir,
    get_subject_dir,
    load_subjects_csv,
)

# =============================================================================
# Status Checking
# =============================================================================


def check_reference_coreg_complete(subject_dir: Path) -> bool:
    """Check if reference coregistration is complete."""
    mask_dir = get_feedback_dir(subject_dir) / "01_reference" / "masks"
    masks = list(mask_dir.glob("*_thr_to_task-reference*.nii.gz"))
    return len(masks) >= 2


def check_feedback_run_complete(subject_dir: Path, subject: str, run: int, n_volumes: int | None = None) -> bool:
    """Check if a pineuro feedback run is complete."""
    feedback_dir = get_feedback_dir(subject_dir) / "02_feedback_output"
    feedbacklog = feedback_dir / f"{subject}_task-feedback_run-{run:02d}_feedbacklog.tsv"

    if not feedbacklog.exists():
        return False

    if n_volumes is None:
        return True

    try:
        df = pd.read_csv(feedbacklog, sep="\t")
        return len(df) >= n_volumes
    except Exception:
        return False


def check_murfi_run_complete(subject_dir: Path, run: int, n_volumes: int, subject: str) -> bool:
    """Check if MURFI run is complete."""
    murfi_dir = subject_dir / "murfi"
    log_path = murfi_dir / f"run-{run:02d}" / subject / "log" / "log.rtl"

    if not log_path.exists():
        return False

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
    report = subject_dir / "murfi" / f"run-{run:02d}" / "comparison_report.txt"
    return report.exists()


# =============================================================================
# Reference Coregistration
# =============================================================================


def run_reference_coregistration(
    subject: str,
    subject_dir: Path,
    mask_files: list[Path],
    ref_file: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Coregister masks to reference_realtime scanner position."""
    from pineuro import coregister_reference

    collection_dir = get_collection_dir(subject_dir)
    feedback_dir = get_feedback_dir(subject_dir)
    reference_dir = collection_dir / "task-reference_run-01"

    print(f"\n{'=' * 60}")
    print("REFERENCE REALTIME COREGISTRATION")
    print("=" * 60)

    coreg_masks = coregister_reference(
        participant_id=subject,
        root_dir=str(feedback_dir),
        input_dicom=reference_dir,
        network_files=mask_files,
        ref_file=ref_file,
        overwrite=overwrite,
        output_subdir="01_reference",
    )

    print("\nCoregistration complete:")
    for mf in coreg_masks:
        print(f"  {mf}")

    masks = {}
    for mask_path in coreg_masks:
        name = mask_path.name
        if "_ContA_" in name:
            masks["ContA"] = mask_path
        elif "_DefaultA_" in name:
            masks["DefaultA"] = mask_path

    return masks


# =============================================================================
# MURFI Functions
# =============================================================================


def copy_and_convert_mask(src: Path, dst: Path) -> None:
    """Copy a mask to uncompressed int16 NIfTI for MURFI."""
    import nibabel as nib

    dst.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(src)
    data = img.get_fdata().astype(np.int16)
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    new_img.header.set_data_dtype(np.int16)
    nib.save(new_img, dst)
    print(f"  Converted mask: {src.name} -> {dst.name} (int16)")


def generate_murfi_xml(
    xml_path: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
    murfi_input_dir: str,
    algo_dump_path: str,
) -> None:
    """Generate MURFI XML configuration file."""
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
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
  <option name="inputDicomDir">   {murfi_input_dir} </option>
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
    <option name="dumpAlgoVars">false</option>

    <design>
      <option name="disabled"> false </option>
      <option name="modelMotionDerivatives"> true </option>
      <option name="maxTrendOrder"> 1 </option>

      <option name="condition" conditionName="Regulation">
        {" ".join(["1"] * n_volumes)}
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
"""

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

    murfi_input_dir = str((run_dir / "murfi_input").absolute())
    algo_dump_path = str((murfi_dir / "log" / "incrglmdump.txt").absolute())

    print("\nSetting up MURFI environment...")
    copy_and_convert_mask(masks["ContA"], murfi_dir / "mask" / "cen.nii")
    copy_and_convert_mask(masks["DefaultA"], murfi_dir / "mask" / "dmn.nii")
    generate_murfi_xml(
        murfi_dir / "xml" / "rtdmn.xml",
        tr,
        n_volumes,
        n_baseline,
        murfi_input_dir,
        algo_dump_path,
    )

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

    murfi_input_dir = run_dir / "murfi_input"
    murfi_input_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing files
    for f in murfi_input_dir.glob("*"):
        if f.is_file():
            f.unlink()

    # Copy DICOMs to a path without spaces (apptainer limitation)
    dicom_copy_dir = run_dir / "dicom_source"
    if dicom_copy_dir.exists():
        shutil.rmtree(dicom_copy_dir)
    print("\nCopying DICOMs for MURFI...")
    print(f"  From: {dicom_dir}")
    print(f"  To: {dicom_copy_dir}")
    shutil.copytree(dicom_dir, dicom_copy_dir)

    env = os.environ.copy()
    env["MURFI_SUBJECTS_DIR"] = str(run_dir.absolute())
    env["MURFI_SUBJECT_NAME"] = subject

    print("\nStarting MURFI...")
    print(f"  MURFI_SUBJECTS_DIR={env['MURFI_SUBJECTS_DIR']}")
    print(f"  MURFI_SUBJECT_NAME={env['MURFI_SUBJECT_NAME']}")
    print(f"  Input dir: {murfi_input_dir.absolute()}")

    murfi_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{run_dir.absolute()}:{run_dir.absolute()}",
        "--env",
        f"MURFI_SUBJECTS_DIR={run_dir.absolute()}",
        "--env",
        f"MURFI_SUBJECT_NAME={subject}",
        str(murfi_sif),
        "murfi",
        "-f",
        str(xml_path.absolute()),
    ]

    murfi_log = run_dir / "murfi_stderr.log"
    murfi_log_fh = open(murfi_log, "w")  # noqa: SIM115

    murfi_proc = subprocess.Popen(
        murfi_cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=murfi_log_fh,
        start_new_session=True,
    )

    print("  Waiting for MURFI to initialize...")
    time.sleep(5)

    if murfi_proc.poll() is not None:
        murfi_log_fh.close()
        with open(murfi_log) as f:
            stderr_content = f.read()
        raise RuntimeError(f"MURFI failed to start (exit code: {murfi_proc.returncode})\nStderr: {stderr_content}")

    serve_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{dicom_copy_dir.absolute()}:{dicom_copy_dir.absolute()}",
        "--bind",
        f"{run_dir.absolute()}:{run_dir.absolute()}",
        str(murfi_sif),
        "servedicoms",
        str(dicom_copy_dir.absolute()),
        str(murfi_input_dir.absolute()),
        str(int(tr * 1000)),
    ]

    print("\nServing DICOMs...")
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
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(serve_proc.pid), signal.SIGKILL)

    print("  Waiting for MURFI to finish...")
    time.sleep(3)

    # Kill MURFI
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(murfi_proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(murfi_proc.pid), signal.SIGKILL)

    subprocess.run(["pkill", "-9", "-f", "murfi -f"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "apptainer.*murfi.sif"], capture_output=True)
    time.sleep(0.5)

    murfi_log_fh.close()

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
# Feedback & Comparison
# =============================================================================


def find_masks(subject_dir: Path) -> dict[str, Path]:
    """Find coregistered masks in feedback/01_reference/masks/."""
    mask_dir = get_feedback_dir(subject_dir) / "01_reference" / "masks"
    masks = {}

    for mask_path in mask_dir.glob("*_thr_to_task-reference*.nii.gz"):
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
    """Run pineuro feedback session in simulator mode."""
    from pineuro.feedback import run_feedback_session
    from pineuro.utils import compute_boxcar_task_regressor

    feedback_dir = output_dir / "02_feedback_output"
    collection_dir = output_dir / "collection"

    feedback_dir.mkdir(parents=True, exist_ok=True)
    collection_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning pineuro feedback run-{run:02d}...")
    print(f"  DICOM source: {dicom_dir}")

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
        motion_correction="siemens_moco",
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


def compare_outputs(
    pineuro_log: Path,
    murfi_log: Path,
    output_dir: Path,
    n_volumes: int,
) -> dict:
    """Compare pineuro and MURFI activation timeseries."""
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

    # Difference correlation
    pn_diff = pineuro_df["ContA"].values - pineuro_df["DefaultA"].values
    mu_diff = murfi_df["cen"].values - murfi_df["dmn"].values

    diff_mask = (pn_diff != 0) | (mu_diff != 0)
    pn_diff_valid = pn_diff[diff_mask]
    mu_diff_valid = mu_diff[diff_mask]

    diff_corr = np.corrcoef(pn_diff_valid, mu_diff_valid)[0, 1] if len(pn_diff_valid) > 0 else 0.0
    results["Difference"] = {"correlation": diff_corr, "n_values": len(pn_diff_valid)}
    print("\n(ContA - DefaultA) vs (cen - dmn):")
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


def run_comparison_for_run(
    subject: str,
    run: int,
    subject_dir: Path,
    murfi_sif: Path,
    tr: float,
    n_volumes: int,
    n_baseline: int,
    skip_existing: bool = True,
) -> dict | None:
    """Run comparison for a single feedback run."""
    collection_dir = get_collection_dir(subject_dir)
    feedback_dir = get_feedback_dir(subject_dir)
    murfi_run_dir = subject_dir / "murfi" / f"run-{run:02d}"
    comparison_run_dir = murfi_run_dir  # comparison outputs alongside MURFI outputs

    dicom_dir = collection_dir / f"feedback_acq-01_run-{run:02d}"

    if skip_existing and check_comparison_complete(subject_dir, run):
        print(f"\n  Run {run:02d}: Comparison already complete, skipping")
        return None

    masks = find_masks(subject_dir)

    print(f"\n{'=' * 60}")
    print(f"RUN {run:02d}")
    print("=" * 60)

    # Run pineuro feedback
    pineuro_log = feedback_dir / "02_feedback_output" / f"{subject}_task-feedback_run-{run:02d}_feedbacklog.tsv"
    if not check_feedback_run_complete(subject_dir, subject, run, n_volumes):
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
    if not check_murfi_run_complete(subject_dir, run, n_volumes, subject):
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
        murfi_run_dir / "murfi_input",
    ]
    for cleanup_dir in cleanup_dirs:
        if cleanup_dir.exists():
            shutil.rmtree(cleanup_dir)
            print(f"  Deleted: {cleanup_dir}")

    return {"subject": subject, "run": run, "results": results}


# =============================================================================
# Group Results TSV
# =============================================================================


def save_group_results(all_results: list[dict], output_dir: Path) -> None:
    """Save structured TSV with one row per subject/run/network."""
    rows = []
    for r in all_results:
        for network, metrics in r["results"].items():
            rows.append(
                {
                    "subject": r["subject"],
                    "run": r["run"],
                    "network": network,
                    **{k: v for k, v in metrics.items()},
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        tsv_path = output_dir / "group_results_murfi.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"\nGroup results saved: {tsv_path}")


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison for all subjects."""
    configs = load_subjects_csv(args.csv)

    print("=" * 60)
    print("MURFI vs PINEURO COMPARISON")
    print("=" * 60)
    print(f"CSV:        {args.csv}")
    print(f"Output:     {args.output_dir}")
    print(f"MURFI SIF:  {args.murfi_sif}")
    print(f"Subjects:   {len(configs)}")
    print()

    all_results = []

    for config in configs:
        subject_dir = get_subject_dir(args.output_dir, config.subject)

        # Check prerequisites
        if not check_mask_extraction_complete(subject_dir):
            print(f"\n{config.subject}: Mask extraction not complete, run setup_validation.py first")
            continue

        if not args.skip_approval_check and not check_masks_approved(subject_dir):
            print(f"\n{config.subject}: Masks not approved, skipping")
            print("  Run GUI to approve masks, or use --skip-approval-check")
            continue

        print(f"\n{'#' * 60}")
        print(f"# COMPARE: {config.subject}")
        print(f"{'#' * 60}")

        # Run reference coregistration if needed
        if not check_reference_coreg_complete(subject_dir):
            print("\nRunning reference coregistration...")
            mask_dir = get_mask_extraction_dir(subject_dir) / "03_masks"
            mask_files = sorted(mask_dir.glob("*_thr.nii.gz"))
            preproc_dir = get_preprocessing_dir(subject_dir)
            ref_file = preproc_dir / f"{config.subject}_task-rest_boldref.nii.gz"
            run_reference_coregistration(config.subject, subject_dir, mask_files, ref_file, overwrite=False)

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

    # Save structured results
    save_group_results(all_results, args.output_dir)

    # Print summary
    print_summary(all_results)


def cmd_status(args: argparse.Namespace) -> None:
    """Check comparison status for all subjects."""
    if not args.output_dir.exists():
        print(f"Output directory not found: {args.output_dir}")
        return

    print("=" * 60)
    print("MURFI COMPARISON STATUS")
    print("=" * 60)

    if args.csv:
        configs = load_subjects_csv(args.csv)
        subjects = [(c.subject, len(c.feedback_runs)) for c in configs]
    else:
        subjects = [(d.name, args.n_runs) for d in sorted(args.output_dir.iterdir()) if d.is_dir()]

    for subject, n_runs in subjects:
        subject_dir = args.output_dir / subject

        approved = check_masks_approved(subject_dir)
        reference_done = check_reference_coreg_complete(subject_dir)
        comp_done = sum(check_comparison_complete(subject_dir, r) for r in range(1, n_runs + 1))

        if not check_mask_extraction_complete(subject_dir):
            status = "needs setup"
        elif not approved:
            status = "needs approval"
        elif not reference_done:
            status = "ready for compare"
        elif comp_done < n_runs:
            status = f"compare: {comp_done}/{n_runs}"
        else:
            status = "complete"

        print(f"  {subject}: {status}")


def print_summary(all_results: list[dict]) -> None:
    """Print final summary."""
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

    # Statistics
    print("\n" + "-" * 60)
    print("CORRELATION STATISTICS")
    print("-" * 60)

    corr_by_network: dict[str, list[float]] = {"ContA": [], "DefaultA": [], "Difference": []}
    for r in all_results:
        for network in corr_by_network:
            if network in r["results"]:
                corr_by_network[network].append(r["results"][network]["correlation"])

    for network, corrs in corr_by_network.items():
        if not corrs:
            continue
        arr = np.array(corrs)
        print(f"\n{network} (n={len(arr)}):")
        print(f"  Mean:   {arr.mean():.6f}")
        print(f"  Std:    {arr.std():.6f}")
        print(f"  Min:    {arr.min():.6f}")
        print(f"  Max:    {arr.max():.6f}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MURFI and pineuro feedback outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run pineuro + MURFI feedback and compare outputs",
    )
    compare_parser.add_argument("--csv", type=Path, required=True, help="Path to subjects CSV")
    compare_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    compare_parser.add_argument("--murfi-sif", type=Path, required=True, help="MURFI apptainer image")
    compare_parser.add_argument("--tr", type=float, default=1.2, help="TR in seconds")
    compare_parser.add_argument("--n-volumes", type=int, default=150, help="Volumes per run")
    compare_parser.add_argument("--n-baseline", type=int, default=25, help="Baseline volumes")
    compare_parser.add_argument("--skip-approval-check", action="store_true", help="Skip mask approval check")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check comparison status")
    status_parser.add_argument("--csv", type=Path, help="Path to subjects CSV (optional)")
    status_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    status_parser.add_argument("--n-runs", type=int, default=5, help="Number of feedback runs (if no CSV)")
    status_parser.add_argument("--n-volumes", type=int, default=150, help="Volumes per run")

    args = parser.parse_args()

    if hasattr(args, "csv") and args.csv:
        args.csv = args.csv.resolve()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.resolve()
    if hasattr(args, "murfi_sif") and args.murfi_sif:
        args.murfi_sif = args.murfi_sif.resolve()

    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
