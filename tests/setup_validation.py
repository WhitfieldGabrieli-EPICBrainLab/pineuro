#!/usr/bin/env python3
"""Shared setup for pineuro validation scripts.

Runs pineuro preprocessing and mask extraction on resting-state data,
preparing outputs that the comparison scripts (compare_murfi_pineuro.py,
compare_melodic_pineuro.py, compare_fmriprep_pineuro.py) all depend on.

Usage:
    # Run setup for all subjects
    python setup_validation.py setup \
        --csv tests/remind_test_data.csv \
        --output-dir /data/validation

    # Check status
    python setup_validation.py status --output-dir /data/validation
"""

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add pineuro to path if running from repo
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SubjectConfig:
    """Configuration for a single subject from the CSV.

    The CSV has columns:
        subject, rest1, rest2, 2vol_realtime, feedback_run-01, ..., feedback_run-05
    """

    subject: str
    rest1: Path
    rest2: Path
    twovol: Path  # 2vol_realtime (collected during NF session)
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
        subject,rest1,rest2,2vol_realtime,feedback_run-01,feedback_run-02,...
    """
    configs = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject = row["subject"].strip()
            if not subject:
                continue

            rest1 = Path(row["rest1"].strip())
            rest2 = Path(row["rest2"].strip())

            if "2vol_realtime" in row:
                twovol = Path(row["2vol_realtime"].strip())
            elif "2vol_initial" in row:
                twovol = Path(row["2vol_initial"].strip())
            else:
                raise KeyError("CSV must have '2vol_realtime' or '2vol_initial' column")

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
# Directory Helpers
# =============================================================================


def get_subject_dir(output_dir: Path, subject: str) -> Path:
    return output_dir / subject


def get_collection_dir(subject_dir: Path) -> Path:
    return subject_dir / "collection"


def get_preprocessing_dir(subject_dir: Path) -> Path:
    return subject_dir / "preprocessing"


def get_mask_extraction_dir(subject_dir: Path) -> Path:
    return subject_dir / "mask_extraction"


def get_feedback_dir(subject_dir: Path) -> Path:
    return subject_dir / "feedback"


# =============================================================================
# Status Checking
# =============================================================================


def check_collection_complete(subject_dir: Path) -> bool:
    """Check if DICOM collection is complete."""
    collection_dir = get_collection_dir(subject_dir)
    return (
        (collection_dir / "task-rest_run-01").exists()
        and (collection_dir / "task-rest_run-02").exists()
        and (collection_dir / "task-2vol_run-01").exists()
    )


def check_preprocessing_complete(subject_dir: Path, subject: str) -> bool:
    """Check if preprocessing is complete."""
    preproc_dir = get_preprocessing_dir(subject_dir)
    return (preproc_dir / f"{subject}_task-rest_preproc.nii.gz").exists()


def check_mask_extraction_complete(subject_dir: Path) -> bool:
    """Check if ICA mask extraction is complete (before 2vol coreg)."""
    mask_dir = get_mask_extraction_dir(subject_dir) / "03_masks"
    masks = list(mask_dir.glob("*_thr.nii.gz"))
    return len(masks) >= 2


def check_masks_approved(subject_dir: Path) -> bool:
    """Check if masks have been approved via GUI."""
    return (get_mask_extraction_dir(subject_dir) / ".masks_approved").exists()


# =============================================================================
# Setup Functions
# =============================================================================


def copy_dicoms_to_collection(config: SubjectConfig, subject_dir: Path) -> None:
    """Copy source DICOMs to collection directory.

    Uses GUI-compatible naming:
    - task-rest_run-01, task-rest_run-02
    - task-2vol_run-01
    - feedback_acq-01_run-01, feedback_acq-01_run-02, ...
    """
    collection_dir = get_collection_dir(subject_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Copy rest runs
    for i, rest_path in enumerate(config.rest_runs, 1):
        dest = collection_dir / f"task-rest_run-{i:02d}"
        if not dest.exists():
            print(f"  Copying {rest_path.name} -> task-rest_run-{i:02d}")
            shutil.copytree(rest_path, dest)
        else:
            print(f"  task-rest_run-{i:02d} already exists, skipping")

    # Copy 2vol
    dest = collection_dir / "task-2vol_run-01"
    if not dest.exists():
        print(f"  Copying {config.twovol.name} -> task-2vol_run-01")
        shutil.copytree(config.twovol, dest)
    else:
        print(f"  task-2vol_run-01 already exists, skipping")

    # Copy feedback runs
    for i, fb_path in enumerate(config.feedback_runs, 1):
        dest = collection_dir / f"feedback_acq-01_run-{i:02d}"
        if not dest.exists():
            print(f"  Copying {fb_path.name} -> feedback_acq-01_run-{i:02d}")
            shutil.copytree(fb_path, dest)
        else:
            print(f"  feedback_acq-01_run-{i:02d} already exists, skipping")


def run_preprocessing(
    subject: str,
    subject_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Run pineuro preprocessing on resting-state data."""
    from pineuro import preprocessing

    collection_dir = get_collection_dir(subject_dir)
    preproc_dir = get_preprocessing_dir(subject_dir)

    input_dirs = [
        collection_dir / "task-rest_run-01",
        collection_dir / "task-rest_run-02",
    ]

    print(f"\n{'=' * 60}")
    print("PREPROCESSING")
    print("=" * 60)

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

    mask_files = mask_extraction_rest(
        participant_id=subject,
        root_dir=mask_dir,
        input_file=preproc_file,
        mask_file=mask_file,
        ref_file=ref_file,
        n_yeo=17,
        yeo_targets=["DefaultA", "ContA"],
        n_components=30,
        threshold=2000,
        smooth_fwhm=5.0,
        highpass=0.01,
        overwrite=overwrite,
        qc_enabled=True,
    )

    print(f"\nMask extraction complete:")
    for mf in mask_files:
        print(f"  {mf}")

    return mask_files


def generate_study_config(
    study_name: str,
    output_dir: Path,
    csv_path: Path,
    tr: float = 1.2,
    n_volumes: int = 150,
    n_baseline: int = 25,
) -> Path:
    """Generate a GUI-compatible study config YAML."""
    from datetime import datetime

    import yaml

    config = {
        "_metadata": {
            "name": study_name,
            "description": "Validation study - auto-generated by setup_validation.py",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
        },
        "paths": {
            "participant_id": "",
            "base_dir": str(output_dir),
            "example_participant_id": "sub-remind3005",
        },
        "scan": {
            "localizer_tr": tr,
            "localizer_n_volumes": 200,
            "localizer_n_runs": 2,
            "localizer_task_type": "rest",
            "feedback_tr": tr,
            "feedback_n_volumes": n_volumes,
            "twovol_initial_enabled": False,
            "twovol_initial_n_volumes": 2,
            "twovol_realtime_enabled": True,
            "twovol_realtime_n_volumes": 2,
            "twovol_initial_dicoms": "",
            "twovol_realtime_dicoms": "",
            "localizer_source_dicoms": ["", ""],
            "feedback_source_dicoms": ["", "", "", "", ""],
            "transfer_source_dicoms": ["", ""],
            "simulator_source_mapping": str(csv_path),
            "simulator_enabled": True,
            "collection_speed": 10.0,
            "feedback_simulator_speed": 100.0,
            "scanner_dicom_dir": "/BOLD_DICOM",
            "murfi_design_path": "",
        },
        "mask": {
            "n_yeo": 17,
            "yeo_targets": ["DefaultA", "ContA"],
            "n_components": 20,
            "smooth_fwhm": 5.0,
            "highpass": 0.01,
            "lowpass": None,
            "mask_names": ["DefaultA", "ContA"],
        },
        "feedback": {
            "n_baseline": n_baseline,
            "n_volumes": n_volumes,
            "use_efficiency_weights": True,
            "fd_threshold": 0.5,
            "moco_enabled": True,
        },
        "localizer_task": {
            "ball": {
                "use_transfer_runs": False,
                "initial_scale_factor": 10.0,
                "fullscreen": True,
                "trigger_keys": ["t", "+", "5", "equal"],
            },
        },
        "task": {
            "task_type": "ball",
            "max_feedback_runs": 5,
            "ball": {
                "use_transfer_runs": False,
                "initial_scale_factor": 10.0,
                "fullscreen": True,
                "trigger_keys": ["t", "+", "5", "equal"],
            },
        },
        "sham": {
            "enabled": False,
        },
    }

    # Save to output directory
    output_config_path = output_dir / "validation_study.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Study config saved to: {output_config_path}")

    # Save to ~/.pineuro/studies/
    studies_dir = Path.home() / ".pineuro" / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    safe_name = study_name.replace(" ", "_").replace("/", "_")
    gui_config_path = studies_dir / f"{safe_name}.yaml"

    with open(gui_config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Study config saved to: {gui_config_path}")

    return gui_config_path


def run_setup_for_subject(
    config: SubjectConfig,
    output_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Run complete setup for a single subject.

    Steps:
    1. Copy DICOMs to collection directory
    2. Run pineuro preprocessing
    3. Run pineuro mask extraction

    Returns list of mask files.
    """
    subject_dir = get_subject_dir(output_dir, config.subject)

    print(f"\n{'#' * 60}")
    print(f"# SETUP: {config.subject}")
    print(f"{'#' * 60}")

    # Step 1: Copy DICOMs
    print("\nStep 1: Copying DICOMs to collection...")
    copy_dicoms_to_collection(config, subject_dir)

    # Step 2: Preprocessing
    print("\nStep 2: Preprocessing...")
    preproc_file, mask_file, ref_file = run_preprocessing(
        config.subject, subject_dir, overwrite
    )

    # Step 3: Mask extraction
    print("\nStep 3: Mask extraction...")
    mask_files = run_mask_extraction(
        config.subject, subject_dir, preproc_file, mask_file, ref_file, overwrite
    )

    return mask_files


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_setup(args: argparse.Namespace) -> None:
    """Run setup for all subjects."""
    configs = load_subjects_csv(args.csv)

    print("=" * 60)
    print("VALIDATION SETUP")
    print("=" * 60)
    print(f"CSV:        {args.csv}")
    print(f"Output:     {args.output_dir}")
    print(f"Study:      {args.study_name}")
    print(f"Subjects:   {len(configs)}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate study config for GUI
    print("Generating study config...")
    generate_study_config(
        study_name=args.study_name,
        output_dir=args.output_dir,
        csv_path=args.csv,
    )
    print()

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
    print(f"1. Open pineuro-gui")
    print(f"2. Select study: '{args.study_name}'")
    print("3. Review mask extraction QC reports for each participant")
    print("4. Approve masks (creates .masks_approved flag)")
    print("5. Run the comparison scripts (compare_murfi, compare_melodic, compare_fmriprep)")


def cmd_status(args: argparse.Namespace) -> None:
    """Check setup status for all subjects."""
    if not args.output_dir.exists():
        print(f"Output directory not found: {args.output_dir}")
        return

    print("=" * 60)
    print("SETUP STATUS")
    print("=" * 60)

    if args.csv:
        configs = load_subjects_csv(args.csv)
        subjects = [c.subject for c in configs]
    else:
        subjects = sorted([d.name for d in args.output_dir.iterdir() if d.is_dir()])

    if not subjects:
        print("No subjects found.")
        return

    for subject in subjects:
        subject_dir = args.output_dir / subject

        parts = []
        if check_collection_complete(subject_dir):
            parts.append("collected")
        if check_preprocessing_complete(subject_dir, subject):
            parts.append("preprocessed")
        if check_mask_extraction_complete(subject_dir):
            parts.append("masks")
        if check_masks_approved(subject_dir):
            parts.append("approved")

        status = ", ".join(parts) if parts else "not started"
        print(f"  {subject}: {status}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared setup for pineuro validation (preprocessing + mask extraction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Run pineuro preprocessing + mask extraction for all subjects",
    )
    setup_parser.add_argument("--csv", type=Path, required=True, help="Path to subjects CSV")
    setup_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    setup_parser.add_argument("--study-name", default="Pineuro Validation", help="Study name for GUI")
    setup_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check setup status")
    status_parser.add_argument("--csv", type=Path, help="Path to subjects CSV (optional)")
    status_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    args = parser.parse_args()

    if hasattr(args, "csv") and args.csv:
        args.csv = args.csv.resolve()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.resolve()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
