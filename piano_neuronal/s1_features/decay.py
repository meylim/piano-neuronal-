import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert

from piano_neuronal.config import DECAY_SKIP_MS, DECAY_MIN_DURATION_S


def bi_exponential_model(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float) -> np.ndarray:
    """Bi-exponential decay model for piano notes.
    y(t) = A1 * exp(-t / tau1) + A2 * exp(-t / tau2)

    tau1 = prompt soundboard decay (fast)
    tau2 = aftersound decay (slow, coupled string mode)
    """
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


def extract_t60(
    audio_mono: np.ndarray,
    sr: int,
    attack_skip_ms: float = DECAY_SKIP_MS,
    min_duration_s: float = DECAY_MIN_DURATION_S,
) -> dict:
    """Extract bi-exponential decay constants from a piano note.

    Returns dict with:
        tau_fast: float — prompt soundboard decay constant (seconds)
        tau_slow: float — aftersound (coupled) decay constant (seconds)
        t60_from_slow: float — T60 estimated from tau_slow (seconds)
        a_fast: float — amplitude of fast component
        a_slow: float — amplitude of slow component
        fit_r_squared: float — R² of the bi-exponential fit
    """
    # Use Hilbert envelope for smoother decay tracking
    analytic_signal = hilbert(audio_mono)
    envelope = np.abs(analytic_signal)

    # Smooth with a moving average to reduce ripple
    window_size = max(1, int(0.01 * sr))  # 10 ms window
    if len(envelope) > window_size:
        kernel = np.ones(window_size) / window_size
        envelope = np.convolve(envelope, kernel, mode="same")

    # Skip attack transient
    skip_samples = int(attack_skip_ms / 1000.0 * sr)
    if skip_samples >= len(envelope):
        return _null_decay_result("Audio shorter than attack skip")

    t = np.arange(len(envelope)) / sr
    t_decay = t[skip_samples:] - t[skip_samples]
    envelope_decay = envelope[skip_samples:]

    # Check minimum duration
    if len(t_decay) / sr < min_duration_s:
        return _null_decay_result(f"Decay too short: {len(t_decay)/sr:.2f}s < {min_duration_s}s")

    # Normalize for numerical stability
    max_val = np.max(envelope_decay)
    if max_val <= 0:
        return _null_decay_result("Zero envelope after skip")
    envelope_norm = envelope_decay / max_val

    # Find the portion above noise floor (above -60 dB of peak)
    noise_threshold = max(1e-4, np.max(envelope_norm) * 1e-3)
    above_noise = envelope_norm > noise_threshold
    if np.sum(above_noise) < 10:
        return _null_decay_result("Signal below noise floor")

    t_above = t_decay[above_noise]
    env_above = envelope_norm[above_noise]

    # Initial guesses: fast ~0.3s, slow ~3.0s
    p0 = [0.7, 0.3, 0.3, 3.0]
    bounds = (
        [0, 1e-3, 0, 1e-2],
        [np.inf, 10.0, np.inf, 30.0]
    )

    try:
        popt, _ = curve_fit(
            bi_exponential_model, t_above, env_above,
            p0=p0, bounds=bounds, maxfev=10000
        )
        a1, tau1, a2, tau2 = popt

        # Ensure tau_fast < tau_slow
        if tau1 > tau2:
            tau_fast, tau_slow = tau2, tau1
            a_fast, a_slow = a2, a1
        else:
            tau_fast, tau_slow = tau1, tau2
            a_fast, a_slow = a1, a2

        # R² quality
        y_pred = bi_exponential_model(t_above, *popt)
        ss_res = np.sum((env_above - y_pred) ** 2)
        ss_tot = np.sum((env_above - np.mean(env_above)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # T60 from the slow (aftersound) component
        # -60 dB = amplitude factor of 0.001
        # exp(-T60/tau_slow) = 0.001 → T60 = tau_slow * ln(1000)
        t60_from_slow = tau_slow * np.log(1000)

        return {
            "tau_fast": float(tau_fast),
            "tau_slow": float(tau_slow),
            "t60_from_slow": float(t60_from_slow),
            "a_fast": float(a_fast),
            "a_slow": float(a_slow),
            "fit_r_squared": float(r_squared),
        }

    except (RuntimeError, ValueError):
        return _null_decay_result("Curve fit failed")


def _null_decay_result(reason: str) -> dict:
    return {
        "tau_fast": 0.0,
        "tau_slow": 0.0,
        "t60_from_slow": 0.0,
        "a_fast": 0.0,
        "a_slow": 0.0,
        "fit_r_squared": 0.0,
        "fit_failure_reason": reason,
    }