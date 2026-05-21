"""Synthetic signal generators for the phantom library.

These are deliberately independent of pineuro (the system under test): the HRF
here is an SPM-style double-gamma built directly from ``scipy.stats.gamma``, so
a phantom's injected signal is genuine ground truth rather than a copy of
pineuro's own ``compute_hrf``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gamma


def double_gamma_hrf(tr: float, duration: float = 32.0) -> NDArray[np.float64]:
    """SPM-style canonical double-gamma HRF sampled at ``tr``, normalized to sum 1.

    Peak gamma shape 6, undershoot gamma shape 16 scaled by 1/6 — the standard
    canonical haemodynamic response.
    """
    t = np.arange(0.0, duration, tr)
    hrf = gamma.pdf(t, 6.0) - gamma.pdf(t, 16.0) / 6.0
    total = hrf.sum()
    return hrf / total if total != 0 else hrf


def hrf_convolved_events(
    n_volumes: int,
    tr: float,
    *,
    n_events: int = 18,
    target_sd: float = 1.0,
    seed: int = 0,
    n_baseline: int = 0,
) -> NDArray[np.float64]:
    """Aperiodic BOLD-like signal: random neural impulses convolved with the HRF.

    Events fall only at or after ``n_baseline`` so the baseline window stays
    signal-free, as in a real neurofeedback run. The result is rescaled to have
    standard deviation ``target_sd``.
    """
    rng = np.random.default_rng(seed)
    neural = np.zeros(n_volumes, dtype=np.float64)
    low = min(n_baseline, n_volumes - 1)
    onsets = rng.integers(low, n_volumes, n_events)
    amplitudes = rng.uniform(0.5, 1.5, n_events)
    for onset, amplitude in zip(onsets, amplitudes, strict=True):
        neural[onset] += amplitude
    convolved = np.convolve(neural, double_gamma_hrf(tr))[:n_volumes]
    sd = convolved.std()
    return convolved / sd * target_sd if sd > 0 else convolved


def boxcar_blocks(
    n_volumes: int,
    blocks: list[tuple[int, int]],
    amplitude: float = 1.0,
    *,
    tr: float | None = None,
    convolve: bool = False,
) -> NDArray[np.float64]:
    """Block design: ``amplitude`` during each ``(onset, offset)`` volume block.

    With ``convolve=True`` the boxcar is convolved with the canonical HRF
    (requires ``tr``), giving a realistically lagged block response.
    """
    signal = np.zeros(n_volumes, dtype=np.float64)
    for onset, offset in blocks:
        signal[onset:offset] = amplitude
    if convolve:
        if tr is None:
            raise ValueError("convolve=True requires tr")
        signal = np.convolve(signal, double_gamma_hrf(tr))[:n_volumes]
    return signal


def sine_signal(n_volumes: int, period: float, amplitude: float = 1.0) -> NDArray[np.float64]:
    """A pure sinusoid — useful as a statistically independent ICA source."""
    t = np.arange(n_volumes, dtype=np.float64)
    return amplitude * np.sin(2.0 * np.pi * t / period)
