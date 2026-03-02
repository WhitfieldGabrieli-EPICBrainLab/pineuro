"""Real-time QC monitoring during DICOM collection (localizer/rest).

Computes framewise displacement (FD), DVARS, and tSNR for each volume
as it is collected, and writes a collectionlog.tsv for GUI polling.
Also generates brain slice images for display.

The QC computations reuse the same algorithms as the feedback module's
FeedbackEstimator, but without any GLM/design-matrix/mask machinery.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Same constant as feedback.py — head radius for FD arc-length conversion
HEAD_RADIUS_MM = 50.0

# Regex for Siemens MoCo motion parameters in DICOM ImageComments
_MOTION_PATTERN = re.compile(
    r"Motion:\s*([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)"
)


def _compute_fd(
    motion_params: NDArray[np.float64],
    prev_motion: NDArray[np.float64] | None,
) -> float:
    """Compute framewise displacement from current and previous motion params.

    Parameters
    ----------
    motion_params : ndarray, shape (6,)
        [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
        Translations in mm, rotations in degrees.
    prev_motion : ndarray or None
        Previous volume's motion params. Returns 0.0 if None.
    """
    if prev_motion is None:
        return 0.0
    diff = motion_params - prev_motion
    diff_mm = diff.copy()
    diff_mm[3:6] *= HEAD_RADIUS_MM * np.pi / 180.0
    return float(np.sum(np.abs(diff_mm)))


def _compute_dvars(
    voxel_values: NDArray[np.float64],
    prev_voxel_values: NDArray[np.float64] | None,
) -> float:
    """Compute DVARS: spatial RMS of temporal derivative."""
    if prev_voxel_values is None:
        return 0.0
    diff = voxel_values - prev_voxel_values
    return float(np.sqrt(np.mean(diff**2)))


def _extract_motion_from_dicom_header(dcm) -> NDArray[np.float64] | None:
    """Extract Siemens MoCo motion params from a pydicom dataset.

    Returns
    -------
    ndarray, shape (6,) or None
        Motion parameters if found, else None.
    """
    comments = getattr(dcm, "ImageComments", None)
    if comments is None:
        return None
    if "Reference volume for motion correction" in comments:
        return np.zeros(6, dtype=np.float64)
    match = _MOTION_PATTERN.search(comments)
    if match:
        return np.array([float(m) for m in match.groups()], dtype=np.float64)
    return None


def _dicom_middle_slice_png(pixel_array: np.ndarray) -> str | None:
    """Convert a 3D volume's middle axial slice to a base64 PNG data URI."""
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        if pixel_array.ndim == 3:
            mid = pixel_array.shape[0] // 2
            slice_data = pixel_array[mid]
        elif pixel_array.ndim == 2:
            slice_data = pixel_array
        else:
            return None

        arr = slice_data.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)
        arr = (arr * 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        logger.debug(f"Slice PNG generation failed: {e}")
        return None


class CollectionQCMonitor:
    """Real-time QC monitoring during DICOM collection.

    Call :meth:`process_volume` for each collected volume to compute FD,
    DVARS, and tSNR.  Results are written to a TSV log file and optionally
    delivered via callback for immediate GUI update.

    Parameters
    ----------
    log_path : Path
        Path to the collectionlog.tsv output file.
    motion_correction : str
        Motion correction mode: ``"siemens_moco"`` reads motion params from
        DICOM ImageComments; ``"none"`` skips FD computation.
    on_qc_data : callable, optional
        ``(volume, fd, dvars, tsnr, slice_b64) -> None``
        Called after each volume with QC metrics and optional slice image.
    """

    def __init__(
        self,
        log_path: Path,
        motion_correction: str = "siemens_moco",
        on_qc_data: Callable[[int, float, float, float, str | None], None] | None = None,
    ):
        self.log_path = Path(log_path)
        self._motion_correction = motion_correction
        self._on_qc_data = on_qc_data

        # Running state
        self._prev_motion: NDArray[np.float64] | None = None
        self._prev_voxel_values: NDArray[np.float64] | None = None
        self._welford_count = 0
        self._welford_mean: NDArray[np.float64] | None = None
        self._welford_m2: NDArray[np.float64] | None = None

        # CSV writer
        self._csv_file: IO[str] | None = None
        self._csv_writer: Any = None
        self._start_time: float | None = None

    def _ensure_csv(self) -> None:
        """Open the TSV file and write the header if not yet open."""
        if self._csv_file is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.log_path, "w", newline="")  # noqa: SIM115
        self._csv_writer = csv.writer(self._csv_file, delimiter="\t")
        self._csv_writer.writerow(["volume", "time", "fd", "dvars", "tsnr"])
        self._start_time = time.monotonic()

    def close(self) -> None:
        """Close the log file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _update_tsnr(self, voxel_values: NDArray[np.float64]) -> float:
        """Update running tSNR via Welford's online algorithm."""
        self._welford_count += 1
        if self._welford_mean is None:
            self._welford_mean = voxel_values.copy()
            self._welford_m2 = np.zeros_like(voxel_values)
            return 0.0

        assert self._welford_m2 is not None
        delta = voxel_values - self._welford_mean
        self._welford_mean += delta / self._welford_count
        delta2 = voxel_values - self._welford_mean
        self._welford_m2 += delta * delta2

        if self._welford_count < 2:
            return 0.0

        variance = self._welford_m2 / (self._welford_count - 1)
        std = np.sqrt(variance)
        valid = std > 0
        if not np.any(valid):
            return 0.0
        voxel_tsnr = self._welford_mean[valid] / std[valid]
        return float(np.median(voxel_tsnr))

    def process_volume(self, volume_path: Path, volume_num: int) -> dict:
        """Process a collected volume and compute QC metrics.

        Parameters
        ----------
        volume_path : Path
            Path to the DICOM (or NIfTI) volume file.
        volume_num : int
            1-based volume number.

        Returns
        -------
        dict
            Keys: ``fd``, ``dvars``, ``tsnr``, ``slice_image`` (base64 or None).
        """
        import pydicom

        from .utils import is_nifti_file

        self._ensure_csv()

        fd = 0.0
        dvars_val = 0.0
        tsnr_val = 0.0
        slice_b64: str | None = None

        motion_params: NDArray[np.float64] | None = None
        try:
            if is_nifti_file(volume_path):
                # NIfTI path — read with nibabel
                import nibabel as nib

                img = nib.load(str(volume_path))
                data = np.asarray(img.dataobj, dtype=np.float64)  # type: ignore[attr-defined]
                voxels = data.ravel()

                # Motion: check JSON sidecar
                motion_params = None
                if self._motion_correction == "siemens_moco":
                    from .utils import get_json_sidecar_path

                    json_path = get_json_sidecar_path(volume_path)
                    if json_path.exists():
                        import json

                        with open(json_path) as f:
                            meta = json.load(f)
                        comments = meta.get("ImageComments", "")
                        if "Reference volume for motion correction" in comments:
                            motion_params = np.zeros(6, dtype=np.float64)
                        else:
                            match = _MOTION_PATTERN.search(comments)
                            if match:
                                motion_params = np.array(
                                    [float(m) for m in match.groups()], dtype=np.float64
                                )

                # Slice image from 3D data
                if data.ndim >= 3:
                    slice_b64 = _dicom_middle_slice_png(data)

            else:
                # DICOM path — read with pydicom
                dcm = pydicom.dcmread(volume_path)
                pixel_array = dcm.pixel_array
                voxels = pixel_array.astype(np.float64).ravel()

                # Motion from ImageComments
                motion_params = None
                if self._motion_correction == "siemens_moco":
                    motion_params = _extract_motion_from_dicom_header(dcm)

                # Slice image
                slice_b64 = _dicom_middle_slice_png(pixel_array)

            # Compute QC metrics
            if motion_params is not None:
                fd = _compute_fd(motion_params, self._prev_motion)
                self._prev_motion = motion_params

            dvars_val = _compute_dvars(voxels, self._prev_voxel_values)
            self._prev_voxel_values = voxels

            tsnr_val = self._update_tsnr(voxels)

        except Exception as e:
            logger.warning(f"QC computation failed for volume {volume_num}: {e}")

        # Write to log
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        if self._csv_writer is not None and self._csv_file is not None:
            self._csv_writer.writerow([volume_num, f"{elapsed:.3f}", f"{fd:.4f}", f"{dvars_val:.4f}", f"{tsnr_val:.2f}"])
            self._csv_file.flush()

        # Deliver via callback
        if self._on_qc_data is not None:
            try:
                self._on_qc_data(volume_num, fd, dvars_val, tsnr_val, slice_b64)
            except Exception as e:
                logger.debug(f"QC callback error: {e}")

        return {"fd": fd, "dvars": dvars_val, "tsnr": tsnr_val, "slice_image": slice_b64}
