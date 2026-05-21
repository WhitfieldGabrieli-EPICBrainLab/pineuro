"""Assemble synthetic 4D BOLD phantom datasets with ground truth.

Generalizes ``analysis/weights_bug/murfi_synthetic/generate_synthetic.py``: a
structural "brain" holds one or more ROIs that carry a known shared signal plus
structured noise, while everything injected is recorded in a :class:`GroundTruth`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .geometry import bright_box, make_tissue_compartments, place_roi
from .ground_truth import GroundTruth, ROIGroundTruth
from .signals import boxcar_blocks, hrf_convolved_events, sine_signal

_DEFAULT_AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])


@dataclass
class ROISpec:
    """Specification for one ROI embedded in a phantom.

    Default ``noise_sd`` is 0.0 (noise-free): a phantom built with the defaults
    carries the injected signal exactly, with no random residual. Tests that
    need noise to exercise a specific path (e.g. inverse-SD weighting requires
    varying voxel variance) pass it explicitly and document the reason.
    """

    name: str
    n_voxels: int = 40
    center: tuple[int, int, int] | None = None
    beta: float = 1.0  # amplitude of the carried signal
    signal: NDArray[np.float64] | None = None  # overrides the shared signal
    noise_sd: float = 0.0  # residual noise SD for "clean" voxels (0.0 -> noise-free)
    noisy_fraction: float = 0.0  # fraction of voxels given noisy_sd instead
    noisy_sd: float = 80.0  # residual noise SD for "noisy" voxels (used only when noisy_fraction > 0)


@dataclass
class PhantomSpec:
    """All tunables for one synthetic phantom.

    Default ``background_noise_sd`` is 0.0 (noise-free): a phantom built with
    the defaults has exactly zero noise in non-ROI brain voxels. Tests that
    need noise pass it explicitly.
    """

    vol_shape: tuple[int, int, int] = (48, 48, 32)
    n_volumes: int = 100
    n_baseline: int = 20
    tr: float = 1.0
    affine: NDArray[np.float64] = field(default_factory=lambda: _DEFAULT_AFFINE.copy())
    baseline_intensity: float = 1000.0
    background_noise_sd: float = 0.0  # 0.0 -> noise-free background (default)
    signal_kind: str = "events"  # "events" | "boxcar" | "sine"
    signal_amplitude: float = 30.0  # target SD (events/sine) or block amplitude (boxcar)
    rois: list[ROISpec] = field(default_factory=lambda: [ROISpec("roi")])
    structural: bool = True  # tissue-compartment brain vs a plain bright box
    seed: int = 1234


def _build_signal(spec: PhantomSpec) -> NDArray[np.float64]:
    if spec.signal_kind == "events":
        return hrf_convolved_events(
            spec.n_volumes,
            spec.tr,
            target_sd=spec.signal_amplitude,
            seed=spec.seed + 1,
            n_baseline=spec.n_baseline,
        )
    if spec.signal_kind == "boxcar":
        return boxcar_blocks(
            spec.n_volumes,
            [(spec.n_baseline, spec.n_volumes)],
            amplitude=spec.signal_amplitude,
            tr=spec.tr,
            convolve=True,
        )
    if spec.signal_kind == "sine":
        return sine_signal(spec.n_volumes, period=spec.n_volumes / 4.0, amplitude=spec.signal_amplitude)
    raise ValueError(f"unknown signal_kind: {spec.signal_kind!r}")


def build_phantom(spec: PhantomSpec) -> tuple[NDArray[np.float64], GroundTruth]:
    """Build a synthetic 4D BOLD phantom and its :class:`GroundTruth`.

    Returns ``(data4d, ground_truth)`` where ``data4d`` has shape
    ``(*vol_shape, n_volumes)``. Brain voxels carry ``baseline + background
    noise``; ROI voxels carry ``baseline + carried signal + residual noise``;
    everything outside the brain is exactly zero.
    """
    rng = np.random.default_rng(spec.seed)
    nx, ny, nz = spec.vol_shape
    nt = spec.n_volumes

    signal = _build_signal(spec)

    if spec.structural:
        intensity, labels = make_tissue_compartments(spec.vol_shape)
    else:
        intensity, labels = bright_box(spec.vol_shape, intensity=spec.baseline_intensity)
    brain_mask = labels > 0

    data = np.zeros((nx, ny, nz, nt), dtype=np.float64)
    n_brain = int(brain_mask.sum())
    data[brain_mask, :] = intensity[brain_mask, None] + rng.normal(0.0, spec.background_noise_sd, (n_brain, nt))

    rois: dict[str, ROIGroundTruth] = {}
    occupied = np.zeros(spec.vol_shape, dtype=bool)
    for roi_spec in spec.rois:
        coords = place_roi(spec.vol_shape, n_voxels=roi_spec.n_voxels, center=roi_spec.center)
        ii, jj, kk = coords[:, 0], coords[:, 1], coords[:, 2]
        if occupied[ii, jj, kk].any():
            raise ValueError(f"ROI {roi_spec.name!r} overlaps an already-placed ROI")
        occupied[ii, jj, kk] = True

        n_vox = coords.shape[0]
        noise_sd = np.full(n_vox, roi_spec.noise_sd, dtype=np.float64)
        n_noisy = int(round(roi_spec.noisy_fraction * n_vox))
        if n_noisy > 0:
            noise_sd[-n_noisy:] = roi_spec.noisy_sd

        source_signal = roi_spec.signal if roi_spec.signal is not None else signal
        carried = np.asarray(source_signal, dtype=np.float64) * roi_spec.beta
        baseline = intensity[ii, jj, kk]
        noise = rng.normal(0.0, 1.0, (n_vox, nt)) * noise_sd[:, None]
        data[ii, jj, kk, :] = baseline[:, None] + carried[None, :] + noise

        center_voxel = np.append(coords.mean(axis=0), 1.0)
        center_world = (np.asarray(spec.affine) @ center_voxel)[:3]
        rois[roi_spec.name] = ROIGroundTruth(
            name=roi_spec.name,
            coords=coords,
            signal=carried,
            beta=roi_spec.beta,
            noise_sd=noise_sd,
            center_world=center_world,
        )

    ground_truth = GroundTruth(
        tr=spec.tr,
        n_volumes=nt,
        n_baseline=spec.n_baseline,
        vol_shape=spec.vol_shape,
        affine=np.asarray(spec.affine, dtype=np.float64),
        baseline_intensity=spec.baseline_intensity,
        signal=signal,
        signal_kind=spec.signal_kind,
        rois=rois,
        seed=spec.seed,
        tissue_labels=labels,
    )
    return data, ground_truth
