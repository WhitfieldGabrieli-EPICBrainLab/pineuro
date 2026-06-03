"""Real-time QC monitoring during DICOM collection (localizer/rest).

Computes framewise displacement (FD), DVARS, and tSNR for each volume
as it is collected, and writes a collectionlog.tsv for GUI polling.
Also generates brain slice images for display.

FD, DVARS, and Siemens MoCo motion parsing are delegated to the shared helpers
in pineuro.utils, so this monitor and the feedback estimator cannot silently
diverge. No GLM/design-matrix machinery is involved here; tSNR/DVARS use a simple
intensity-gated in-brain voxel set (no extracted brain mask is available yet at
collection time) and censor volumes whose FD/DVARS exceed threshold.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from .utils import compute_dvars, compute_fd, parse_siemens_motion_comment, standardize_dvars

logger = logging.getLogger(__name__)


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
    return parse_siemens_motion_comment(comments)


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

        # arr is a 2-D uint8 array; PIL infers the "L" (8-bit grayscale) mode.
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        logger.warning(f"Slice PNG generation failed: {e}")
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
    motion_monitoring : str
        Motion-monitoring mode controlling how motion params are estimated for
        QC: ``"siemens_moco"`` reads motion params from DICOM ImageComments;
        ``"software"`` derives them via ANTsPy rigid registration; ``"none"``
        skips FD computation. No volumes are realigned — correction happens
        post-hoc in preprocessing.
    fd_threshold : float
        FD (mm) above which a volume is censored from the running tSNR estimate.
    dvars_threshold : float
        Standardized DVARS above which a volume is censored from the running tSNR estimate.
    on_qc_data : callable, optional
        ``(volume, fd, dvars, tsnr, slice_b64) -> None``
        Called after each volume with QC metrics and optional slice image.
    """

    #: Retained (non-censored) volumes required before a tSNR value is reported.
    #: Two is the mathematical floor for a variance estimate; until then hold the prior value.
    _MIN_TSNR_VOLUMES = 2

    def __init__(
        self,
        log_path: Path,
        motion_monitoring: str = "siemens_moco",
        fd_threshold: float = 0.5,
        dvars_threshold: float = 1.5,
        on_qc_data: Callable[[int, float, float, float, str | None], None] | None = None,
    ):
        self.log_path = Path(log_path)
        self._motion_monitoring = motion_monitoring
        self._fd_threshold = fd_threshold
        self._dvars_threshold = dvars_threshold
        self._on_qc_data = on_qc_data

        # Running state
        self._prev_motion: NDArray[np.float64] | None = None
        self._prev_voxel_values: NDArray[np.float64] | None = None
        self._welford_count = 0
        self._welford_mean: NDArray[np.float64] | None = None
        self._welford_m2: NDArray[np.float64] | None = None
        self._last_tsnr = 0.0
        # Fixed in-brain voxel mask (flat boolean), derived once from the first volume so
        # tSNR/DVARS measure tissue rather than a FOV-weighted blend with background air.
        self._brain_mask: NDArray[np.bool_] | None = None
        # Raw DVARS history for running standardization (scanner-intensity units are
        # arbitrary; the running median gives a scale-free metric where clean ≈ 1.0).
        self._dvars_history: list[float] = []

        # Software motion correction state (ANTsPy)
        self._ref_ants: Any = None  # Cached ANTs reference image
        self._ref_spacing: list[float] | None = None
        self._prev_transform: str | None = None  # Warm-start path
        self._geometry_cache: Any = None  # DICOM geometry cache for mosaic unpacking

        # CSV writer
        self._csv_file: IO[str] | None = None
        self._csv_writer: Any = None
        self._start_time: float | None = None
        # Warn at most once if the QC delivery callback raises (it runs per-volume).
        self._qc_callback_warned = False

    def _ensure_csv(self) -> None:
        """Open the TSV file and write the header if not yet open."""
        if self._csv_file is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.log_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._csv_writer = csv.writer(self._csv_file, delimiter="\t")
        self._csv_writer.writerow(["volume", "time", "fd", "dvars", "tsnr"])
        self._start_time = time.monotonic()

    def close(self) -> None:
        """Close the log file and release cached resources."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        # Release the cached ANTs reference image so its buffers can be freed.
        self._ref_ants = None

    def _software_motion_correct(self, vol_3d: NDArray[np.float64], spacing: list[float]) -> NDArray[np.float64]:
        """Run ANTsPy rigid registration and return motion parameters.

        First volume becomes the reference. Subsequent volumes are registered
        to it with warm-start from the previous transform.
        """
        from .feedback import _ants_register_volume

        if self._ref_ants is None:
            # First volume: cache as reference, motion is zero
            import ants

            self._ref_ants = ants.from_numpy(vol_3d.astype(np.float32), spacing=spacing)
            self._ref_spacing = spacing
            return np.zeros(6, dtype=np.float64)

        _corrected, motion_params, transform_path = _ants_register_volume(
            ref_ants=self._ref_ants,
            mov_vol=vol_3d,
            spacing=self._ref_spacing or spacing,
            initial_transform=self._prev_transform,
        )
        self._prev_transform = transform_path
        return motion_params

    def _brain_voxels(self, voxels: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return in-brain voxels using a fixed intensity gate derived from the first volume.

        Raveling the whole image mixes brain tissue (tSNR ~40-90) with background/air
        (mean/std ~1.9 at the Rician noise floor), making the median FOV/head-size
        dependent rather than a tissue measure. We threshold once on the first volume at
        10% of its 98th percentile (robust to a few bright outliers) and reuse that voxel
        set for every subsequent volume so Welford accumulates a consistent voxel set. A
        fixed gate also drops the black inter-tile gaps in 2-D mosaic DICOMs.
        """
        if self._brain_mask is None:
            thr = 0.1 * float(np.percentile(voxels, 98))
            mask = voxels > thr
            # Fall back to all voxels if the gate is degenerate (e.g. empty/uniform volume).
            self._brain_mask = mask if int(mask.sum()) >= 100 else np.ones_like(voxels, dtype=bool)
        return voxels[self._brain_mask]

    def _current_tsnr(self) -> float:
        """Report the running tSNR from current Welford state without accumulating."""
        if self._welford_m2 is None or self._welford_count < self._MIN_TSNR_VOLUMES:
            return self._last_tsnr
        variance = self._welford_m2 / (self._welford_count - 1)
        std = np.sqrt(variance)
        valid = std > 0
        if not np.any(valid):
            return self._last_tsnr
        assert self._welford_mean is not None
        voxel_tsnr = self._welford_mean[valid] / std[valid]
        return float(np.median(voxel_tsnr))

    def _update_tsnr(self, voxel_values: NDArray[np.float64]) -> float:
        """Accumulate one volume into the running tSNR (Welford's algorithm) and report it."""
        self._welford_count += 1
        if self._welford_mean is None:
            self._welford_mean = voxel_values.copy()
            self._welford_m2 = np.zeros_like(voxel_values)
        else:
            assert self._welford_m2 is not None
            delta = voxel_values - self._welford_mean
            self._welford_mean += delta / self._welford_count
            delta2 = voxel_values - self._welford_mean
            self._welford_m2 += delta * delta2

        self._last_tsnr = self._current_tsnr()
        return self._last_tsnr

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
        vol_3d: NDArray[np.float64] | None = None  # For software moco
        spacing: list[float] | None = None
        try:
            if is_nifti_file(volume_path):
                # NIfTI path — read with nibabel
                import nibabel as nib

                img = nib.load(str(volume_path))
                data = np.asarray(img.dataobj, dtype=np.float64)  # type: ignore[attr-defined]
                voxels = data.ravel()

                if self._motion_monitoring == "siemens_moco":
                    # Motion: check JSON sidecar
                    from .utils import get_json_sidecar_path

                    json_path = get_json_sidecar_path(volume_path)
                    if json_path.exists():
                        import json

                        with open(json_path) as f:
                            meta = json.load(f)
                        comments = meta.get("ImageComments", "")
                        parsed = parse_siemens_motion_comment(comments)
                        if parsed is not None:
                            motion_params = parsed
                elif self._motion_monitoring == "software" and data.ndim >= 3:
                    vol_3d = data
                    spacing = list(img.header.get_zooms()[:3])  # type: ignore[union-attr,attr-defined]

                # Slice image from 3D data
                if data.ndim >= 3:
                    slice_b64 = _dicom_middle_slice_png(data)

            else:
                # DICOM path
                if self._motion_monitoring == "software":
                    # Use _read_dicom_volume for proper mosaic unpacking → 3D
                    from .feedback import _read_dicom_volume

                    vol_data, _affine, vol_spacing, self._geometry_cache = _read_dicom_volume(
                        volume_path, geometry_cache=self._geometry_cache
                    )
                    voxels = vol_data.ravel()
                    vol_3d = vol_data
                    spacing = vol_spacing
                    # Slice image from unpacked 3D data
                    slice_b64 = _dicom_middle_slice_png(vol_data)
                else:
                    # Standard DICOM read (siemens_moco or none)
                    dcm = pydicom.dcmread(volume_path)
                    pixel_array = dcm.pixel_array
                    voxels = pixel_array.astype(np.float64).ravel()

                    if self._motion_monitoring == "siemens_moco":
                        motion_params = _extract_motion_from_dicom_header(dcm)

                    # Slice image
                    slice_b64 = _dicom_middle_slice_png(pixel_array)

            # Software motion correction (ANTsPy rigid registration)
            if self._motion_monitoring == "software" and vol_3d is not None and spacing is not None:
                motion_params = self._software_motion_correct(vol_3d, spacing)

            # Compute QC metrics on in-brain voxels only (not the whole raveled image,
            # which would mix tissue with background air and FOV-dependent voxel counts).
            brain_voxels = self._brain_voxels(voxels)

            if motion_params is not None:
                fd = compute_fd(motion_params, self._prev_motion)
                self._prev_motion = motion_params

            # DVARS standardized by the running median of prior raw DVARS so the threshold
            # is scale-free (raw DVARS is in arbitrary scanner-intensity units, ~tens).
            raw_dvars = compute_dvars(brain_voxels, self._prev_voxel_values)
            dvars_val = standardize_dvars(raw_dvars, self._dvars_history)
            if self._prev_voxel_values is not None:
                self._dvars_history.append(raw_dvars)
            self._prev_voxel_values = brain_voxels

            # Censor motion-contaminated volumes from the tSNR estimate: a transient spike
            # would otherwise permanently inflate the accumulated variance. FD is 0.0 here
            # when motion params are unavailable, so it never censors in that case.
            if fd > self._fd_threshold or dvars_val > self._dvars_threshold:
                tsnr_val = self._last_tsnr
            else:
                tsnr_val = self._update_tsnr(brain_voxels)

        except Exception as e:
            logger.warning(f"QC computation failed for volume {volume_num}: {e}")

        # Write to log
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        if self._csv_writer is not None and self._csv_file is not None:
            self._csv_writer.writerow(
                [volume_num, f"{elapsed:.3f}", f"{fd:.4f}", f"{dvars_val:.4f}", f"{tsnr_val:.2f}"]
            )
            self._csv_file.flush()

        # Deliver via callback
        if self._on_qc_data is not None:
            try:
                self._on_qc_data(volume_num, fd, dvars_val, tsnr_val, slice_b64)
            except Exception as e:
                if not self._qc_callback_warned:
                    logger.warning(f"QC callback error (further occurrences suppressed): {e}")
                    self._qc_callback_warned = True

        return {"fd": fd, "dvars": dvars_val, "tsnr": tsnr_val, "slice_image": slice_b64}
