"""Excitation envelope extraction (first 50ms after onset).

# TODO Sprint 4: Refine excitation/harmonics separation (HPSS or model-based
# sinusoidal subtraction). Current method is approximate: the broadband attack
# overlaps with emerging harmonics, and naive harmonic subtraction may remove
# transient content we want to preserve. This is a debt for Sprint 1.
"""

import numpy as np
from piano_neuronal.config import EXCITATION_DURATION_MS


def extract_excitation(
    audio_mono: np.ndarray,
    sr: int,
    midi_note: int,
    duration_ms: float = EXCITATION_DURATION_MS,
) -> dict:
    """Extract excitation envelope from the first N ms of a note.

    Returns dict with:
        excitation_raw: np.ndarray — raw first N ms of the onset-aligned signal
        excitation_residual: np.ndarray — noise/transient component after harmonic subtraction
        duration_samples: int
    """
    n_samples = int(duration_ms / 1000.0 * sr)

    if len(audio_mono) < n_samples:
        n_samples = len(audio_mono)

    excitation_raw = audio_mono[:n_samples].copy()

    # Approximate harmonic subtraction: remove fundamental and first few harmonics
    # This is the Sprint 1 approximation — see TODO above for Sprint 4 refinement
    excitation_residual = _subtract_harmonics(excitation_raw, sr, midi_note)

    return {
        "excitation_raw": excitation_raw,
        "excitation_residual": excitation_residual,
        "duration_samples": n_samples,
    }


def _subtract_harmonics(
    signal: np.ndarray, sr: int, midi_note: int, n_harmonics: int = 10
) -> np.ndarray:
    """Subtract estimated harmonic components from signal.

    For each harmonic n, estimates amplitude and phase via least-squares
    fitting of a sinusoid at frequency f_n = n * f0 * sqrt(1 + B * n^2),
    then subtracts it. The residual is the excitation (noise + transient).
    """
    f0 = 440.0 * 2 ** ((midi_note - 69) / 12)
    t = np.arange(len(signal)) / sr

    residual = signal.copy()

    for n in range(1, n_harmonics + 1):
        fn = n * f0  # No B correction for excitation extraction (B is small)
        if fn > sr / 2:
            break

        # Least-squares fit: signal ≈ A*cos(2π*f*t) + B*sin(2π*f*t)
        cos_term = np.cos(2 * np.pi * fn * t)
        sin_term = np.sin(2 * np.pi * fn * t)

        # Solve for A, B
        A = np.column_stack([cos_term, sin_term])
        coeffs, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)

        # Subtract the fitted harmonic
        residual = residual - coeffs[0] * cos_term - coeffs[1] * sin_term

    return residual