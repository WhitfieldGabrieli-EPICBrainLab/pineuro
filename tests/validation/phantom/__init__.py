"""Synthetic-data phantom library for the pineuro validation suite.

Generalizes ``analysis/weights_bug/murfi_synthetic/generate_synthetic.py`` into
a reusable digital-phantom generator: builds 3D/4D NIfTI volumes with tissue
compartments and embedded ROIs carrying known HRF-convolved signal plus
structured noise, alongside a :class:`GroundTruth` object that records exactly
what was injected so tests can assert recovery.

The library is deliberately self-contained (numpy / scipy / nibabel only) — it
does not import pineuro, so its ground truth is independent of the system
under test.
"""

from .builder import PhantomSpec, ROISpec, build_phantom
from .geometry import box_slices, bright_box, gaussian_blob, make_tissue_compartments, place_roi
from .ground_truth import GroundTruth, ROIGroundTruth, load_ground_truth, save_ground_truth
from .io import (
    sidecar_path,
    write_events_tsv,
    write_json_sidecar,
    write_nifti,
    write_per_volume_niftis,
)
from .motion import MOTION_CONVENTION, apply_known_motion, motion_image_comment, random_motion_sequence
from .signals import boxcar_blocks, double_gamma_hrf, hrf_convolved_events, sine_signal

__all__ = [
    # builder
    "PhantomSpec",
    "ROISpec",
    "build_phantom",
    # ground truth
    "GroundTruth",
    "ROIGroundTruth",
    "save_ground_truth",
    "load_ground_truth",
    # signals
    "double_gamma_hrf",
    "hrf_convolved_events",
    "boxcar_blocks",
    "sine_signal",
    # geometry
    "make_tissue_compartments",
    "bright_box",
    "box_slices",
    "place_roi",
    "gaussian_blob",
    # motion
    "random_motion_sequence",
    "apply_known_motion",
    "motion_image_comment",
    "MOTION_CONVENTION",
    # io
    "write_nifti",
    "write_per_volume_niftis",
    "write_json_sidecar",
    "write_events_tsv",
    "sidecar_path",
]
