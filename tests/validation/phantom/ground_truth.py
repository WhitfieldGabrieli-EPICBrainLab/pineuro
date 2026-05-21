"""Ground-truth records for the phantom library.

Every synthetic dataset built by :mod:`phantom.builder` is accompanied by a
:class:`GroundTruth` describing exactly what was injected — the shared signal
timeseries, ROI locations, betas, applied motion, tissue labels — so that
validation tests can assert pineuro recovered it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class ROIGroundTruth:
    """Ground truth for a single embedded region of interest."""

    name: str
    coords: NDArray[np.int_]  # (n_voxels, 3) — i, j, k voxel indices
    signal: NDArray[np.float64]  # (n_volumes,) — true signal carried by the ROI
    beta: float  # amplitude of the shared signal in this ROI
    noise_sd: NDArray[np.float64]  # (n_voxels,) — per-voxel residual noise SD
    center_world: NDArray[np.float64]  # (3,) — world-space centroid

    @property
    def n_voxels(self) -> int:
        return int(self.coords.shape[0])

    def flat_indices(self, vol_shape: tuple[int, int, int], order: str = "C") -> NDArray[np.int_]:
        """Linear voxel indices for ``coords`` under the given ravel order."""
        return np.ravel_multi_index((self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]), vol_shape, order=order)

    def mask(self, vol_shape: tuple[int, int, int]) -> NDArray[np.float64]:
        """Binary 3D mask volume for this ROI (1.0 inside, 0.0 outside)."""
        m = np.zeros(vol_shape, dtype=np.float64)
        m[self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]] = 1.0
        return m


@dataclass
class GroundTruth:
    """Everything that was injected into one synthetic phantom dataset."""

    tr: float
    n_volumes: int
    n_baseline: int
    vol_shape: tuple[int, int, int]
    affine: NDArray[np.float64]
    baseline_intensity: float
    signal: NDArray[np.float64]  # the shared true signal, (n_volumes,)
    signal_kind: str  # "events" | "boxcar" | "sine"
    rois: dict[str, ROIGroundTruth]
    seed: int
    design: NDArray[np.float64] | None = None  # (n_volumes, n_cond) modelled design
    betas: NDArray[np.float64] | None = None  # (*vol_shape, n_cond) per-voxel betas
    motion_params: NDArray[np.float64] | None = None  # (n_volumes, 6) applied motion
    motion_convention: str = "trans_mm_xyz__rot_deg_xyz"
    tissue_labels: NDArray[np.int_] | None = None  # 0 bg, 1 GM, 2 WM, 3 CSF

    def roi(self, name: str) -> ROIGroundTruth:
        return self.rois[name]

    def brain_mask(self) -> NDArray[np.float64]:
        """Binary 3D mask of all non-background voxels (requires tissue_labels)."""
        if self.tissue_labels is None:
            raise ValueError("phantom has no tissue_labels — build with structural=True")
        return (self.tissue_labels > 0).astype(np.float64)

    @property
    def baseline_signal(self) -> NDArray[np.float64]:
        """The shared signal restricted to the baseline window."""
        return self.signal[: self.n_baseline]

    @property
    def task_signal(self) -> NDArray[np.float64]:
        """The shared signal restricted to the post-baseline window."""
        return self.signal[self.n_baseline :]


def _summary(gt: GroundTruth) -> dict:
    """A small JSON-serializable summary of a GroundTruth (no bulk arrays)."""
    return {
        "tr": gt.tr,
        "n_volumes": gt.n_volumes,
        "n_baseline": gt.n_baseline,
        "vol_shape": list(gt.vol_shape),
        "baseline_intensity": gt.baseline_intensity,
        "signal_kind": gt.signal_kind,
        "signal_sd": float(np.std(gt.signal)),
        "signal_range": [float(np.min(gt.signal)), float(np.max(gt.signal))],
        "seed": gt.seed,
        "has_design": gt.design is not None,
        "has_betas": gt.betas is not None,
        "has_motion": gt.motion_params is not None,
        "motion_convention": gt.motion_convention,
        "rois": {
            name: {
                "n_voxels": roi.n_voxels,
                "beta": roi.beta,
                "center_world": [float(x) for x in roi.center_world],
                "noise_sd_min": float(np.min(roi.noise_sd)),
                "noise_sd_max": float(np.max(roi.noise_sd)),
            }
            for name, roi in gt.rois.items()
        },
    }


def save_ground_truth(gt: GroundTruth, directory: Path | str) -> Path:
    """Write ``ground_truth.npz`` (lossless) and ``ground_truth.json`` (readable)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "tr": np.asarray(gt.tr),
        "n_volumes": np.asarray(gt.n_volumes),
        "n_baseline": np.asarray(gt.n_baseline),
        "vol_shape": np.asarray(gt.vol_shape),
        "affine": np.asarray(gt.affine),
        "baseline_intensity": np.asarray(gt.baseline_intensity),
        "signal": np.asarray(gt.signal),
        "signal_kind": np.asarray(gt.signal_kind),
        "seed": np.asarray(gt.seed),
        "motion_convention": np.asarray(gt.motion_convention),
        "roi_names": np.asarray(list(gt.rois.keys()), dtype=object).astype(str),
    }
    for attr in ("design", "betas", "motion_params", "tissue_labels"):
        value = getattr(gt, attr)
        if value is not None:
            payload[attr] = np.asarray(value)
    for name, roi in gt.rois.items():
        payload[f"roi__{name}__coords"] = roi.coords
        payload[f"roi__{name}__signal"] = roi.signal
        payload[f"roi__{name}__noise_sd"] = roi.noise_sd
        payload[f"roi__{name}__center_world"] = roi.center_world
        payload[f"roi__{name}__beta"] = np.asarray(roi.beta)

    npz_path = directory / "ground_truth.npz"
    np.savez(npz_path, **payload)
    (directory / "ground_truth.json").write_text(json.dumps(_summary(gt), indent=2))
    return npz_path


def load_ground_truth(directory: Path | str) -> GroundTruth:
    """Rehydrate a :class:`GroundTruth` from ``ground_truth.npz``."""
    data = np.load(Path(directory) / "ground_truth.npz", allow_pickle=False)
    files = set(data.files)

    rois: dict[str, ROIGroundTruth] = {}
    for name in (str(n) for n in data["roi_names"]):
        rois[name] = ROIGroundTruth(
            name=name,
            coords=data[f"roi__{name}__coords"],
            signal=data[f"roi__{name}__signal"],
            beta=float(data[f"roi__{name}__beta"]),
            noise_sd=data[f"roi__{name}__noise_sd"],
            center_world=data[f"roi__{name}__center_world"],
        )

    return GroundTruth(
        tr=float(data["tr"]),
        n_volumes=int(data["n_volumes"]),
        n_baseline=int(data["n_baseline"]),
        vol_shape=tuple(int(x) for x in data["vol_shape"]),
        affine=data["affine"],
        baseline_intensity=float(data["baseline_intensity"]),
        signal=data["signal"],
        signal_kind=str(data["signal_kind"]),
        rois=rois,
        seed=int(data["seed"]),
        design=data["design"] if "design" in files else None,
        betas=data["betas"] if "betas" in files else None,
        motion_params=data["motion_params"] if "motion_params" in files else None,
        motion_convention=str(data["motion_convention"]),
        tissue_labels=data["tissue_labels"] if "tissue_labels" in files else None,
    )
