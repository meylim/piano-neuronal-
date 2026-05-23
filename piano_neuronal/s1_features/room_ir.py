"""Room IR extraction via Wiener deconvolution (Close → Ambient).

This extracts the acoustic transfer function between the Close and Ambient
microphone positions — a ROOM IR, NOT the structural soundboard impulse response.
Usage: final convolution in Strate 3 (spatial signature).

The structural soundboard modal response is out of scope for Sprint 1.
"""

import numpy as np
from scipy.signal import fftconvolve

from piano_neuronal.config import IR_REGULARIZATION, IR_NOTES


def extract_room_ir(
    audio_close_mono: np.ndarray,
    audio_ambient_mono: np.ndarray,
    sr: int,
    onset_idx: int,
    regularization: float = IR_REGULARIZATION,
) -> dict:
    """Extract room IR via Wiener deconvolution.

    Treats the Close mic signal as the 'input' and the Ambient mic as the 'output'.
    The transfer function H(f) = P_xy(f) / (P_xx(f) + lambda) captures the
    acoustic path between the two mic positions (room + soundboard radiation pattern).

    Args:
        audio_close_mono: Close mic signal, mono, onset-aligned (1-D array)
        audio_ambient_mono: Ambient mic signal, mono, onset-aligned (1-D array)
        sr: sample rate
        onset_idx: sample index of onset (used to align Close and Ambient if needed)
        regularization: Wiener filter regularization parameter

    Returns:
        dict with 'ir' (impulse response array), 'ir_duration_s', 'ir_t60'
    """
    # Ensure same length
    min_len = min(len(audio_close_mono), len(audio_ambient_mono))
    x = audio_close_mono[:min_len]
    y = audio_ambient_mono[:min_len]

    if min_len < sr * 0.1:  # Need at least 100ms of audio
        return {"ir": np.array([]), "ir_duration_s": 0.0, "ir_t60": 0.0, "error": "Audio too short"}

    n = len(x)

    # FFT of both signals
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)

    # Cross-spectral density (numerator)
    P_xy = Y * np.conj(X)

    # Auto-spectral density (denominator)
    P_xx = X * np.conj(X)

    # Wiener filter: H(f) = P_xy(f) / (P_xx(f) + lambda)
    H = P_xy / (P_xx + regularization)

    # Impulse response via inverse FFT
    h = np.fft.irfft(H, n=n)

    # Trim IR to meaningful portion (first 2 seconds, or until it decays below -60 dB)
    peak_idx = np.argmax(np.abs(h))
    peak_val = np.abs(h[peak_idx])

    # Find where IR decays to -60 dB relative to peak
    threshold = peak_val * 1e-3  # -60 dB
    decay_indices = np.where(np.abs(h[peak_idx:]) < threshold)[0]

    if len(decay_indices) > 0:
        ir_end = peak_idx + decay_indices[0]
    else:
        ir_end = min(len(h), peak_idx + int(2.0 * sr))  # Cap at 2 seconds

    ir = h[:ir_end]

    # Estimate IR T60
    ir_t60 = _estimate_ir_t60(ir, sr)

    return {
        "ir": ir,
        "ir_duration_s": len(ir) / sr,
        "ir_t60": float(ir_t60),
    }


def _estimate_ir_t60(ir: np.ndarray, sr: int) -> float:
    """Estimate T60 from impulse response energy decay."""
    # Squared envelope
    energy = ir ** 2

    # Cumulative energy decay curve (Schroeder method)
    cumulative = np.cumsum(energy[::-1])[::-1]
    if cumulative[0] <= 0:
        return 0.0
    decay_curve = 10 * np.log10(cumulative / cumulative[0])

    # Find time where decay reaches -60 dB
    below_60 = np.where(decay_curve <= -60)[0]
    if len(below_60) > 0:
        t60_idx = below_60[0]
        return float(t60_idx / sr)

    # If never reaches -60 dB, extrapolate from -30 dB
    below_30 = np.where(decay_curve <= -30)[0]
    if len(below_30) > 0:
        t30_idx = below_30[0]
        return float(2 * t30_idx / sr)  # T60 ≈ 2 × T30

    return float(len(ir) / sr)