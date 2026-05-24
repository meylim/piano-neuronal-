import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from piano_neuronal.config import get_n_fft_for_note, MIDI_NOTE_MIN, MIDI_NOTE_MAX


def interpolate_B_fallback(results: list[dict]) -> list[dict]:
    """Interpolate B for notes where extraction failed (B=0, r_squared=0).

    Groups by (velocity_layer, pedal, mic), sorts by midi_note,
    and fills B=0 entries via linear interpolation from neighbours.
    Notes at the edges inherit their nearest valid neighbour.
    """
    from itertools import groupby

    key_fn = lambda r: (r.get("velocity_layer", ""), r.get("pedal", ""), r.get("mic", ""))

    sorted_results = sorted(results, key=lambda r: (key_fn(r), r.get("midi_note", 0)))

    for key, group in groupby(sorted_results, key=key_fn):
        group_list = list(group)
        notes = np.array([r.get("midi_note", 0) for r in group_list])
        Bs = np.array([r.get("B", 0.0) for r in group_list])
        r2s = np.array([r.get("B_fit_r_squared", 0.0) for r in group_list])

        valid = (Bs > 0) & (r2s > 0.1)
        if valid.sum() < 2:
            continue

        invalid = ~valid
        if invalid.sum() == 0:
            continue

        Bs_interp = np.interp(notes, notes[valid], Bs[valid])
        for i in np.where(invalid)[0]:
            group_list[i]["B"] = float(Bs_interp[i])
            group_list[i]["B_fit_r_squared"] = -1.0  # Mark as interpolated

    return sorted_results


def inharmonicity_model(n: float, f0: float, B: float) -> float:
    """Physical model of string inharmonicity.
    f_n = n * f0 * sqrt(1 + B * n^2)
    """
    return n * f0 * np.sqrt(1 + B * n**2)


def extract_inharmonicity_B(
    audio_mono: np.ndarray,
    sr: int,
    midi_note: int,
) -> dict:
    """Extract inharmonicity coefficient B using STFT with adaptive n_fft.

    Returns dict with:
        B: float — inharmonicity coefficient
        f0_measured: float — measured fundamental frequency (Hz)
        n_harmonics: int — number of harmonics detected
        n_fft_used: int — the n_fft value used
        fit_r_squared: float — quality of the curve fit (1.0 = perfect)
    """
    n_fft = get_n_fft_for_note(midi_note)
    f0_target = 440.0 * 2 ** ((midi_note - 69) / 12)

    # Pad if needed
    if len(audio_mono) < n_fft:
        audio_mono = np.pad(audio_mono, (0, n_fft - len(audio_mono)))

    # Use the steady-state portion (skip attack, use middle portion)
    # This avoids broadband attack noise contaminating partial detection
    attack_samples = int(0.05 * sr)  # skip first 50ms
    if len(audio_mono) > attack_samples + n_fft:
        start = attack_samples
        audio_segment = audio_mono[start:start + n_fft]
    else:
        audio_segment = audio_mono

    # High-resolution STFT
    S = np.abs(np.fft.rfft(audio_segment * np.hanning(len(audio_segment))))
    freqs = np.fft.rfftfreq(len(audio_segment), 1.0 / sr)

    # Find peaks in spectrum — use lower threshold for high notes
    # High piano notes (MIDI>72) have few quiet partials; the default
    # 1% threshold misses them, yielding B=0 for 25% of the dataset.
    if midi_note > 72:
        min_height = np.max(S) * 0.001
    else:
        min_height = np.max(S) * 0.01
    min_distance = max(1, int(f0_target * 0.5 / (sr / len(audio_segment))))
    peaks, properties = find_peaks(S, height=min_height, distance=min_distance)
    peak_freqs = freqs[peaks]
    peak_heights = S[peaks]

    if len(peak_freqs) < 3:
        return {
            "B": 0.0, "f0_measured": f0_target,
            "n_harmonics": 0, "n_fft_used": n_fft,
            "fit_r_squared": 0.0,
        }

    # Identify harmonics: match peaks to expected harmonic positions
    measured_harmonics = []
    n_indices = []

    for peak_f in peak_freqs:
        n_approx = round(peak_f / f0_target)
        if n_approx < 1 or n_approx > 40:
            continue
        # Check if peak is near expected harmonic frequency
        expected = n_approx * f0_target
        if abs(peak_f - expected) < f0_target * 0.15:
            measured_harmonics.append(peak_f)
            n_indices.append(n_approx)

    if len(n_indices) < 3:
        return {
            "B": 0.0, "f0_measured": f0_target,
            "n_harmonics": len(n_indices), "n_fft_used": n_fft,
            "fit_r_squared": 0.0,
        }

    n_array = np.array(n_indices, dtype=float)
    f_array = np.array(measured_harmonics, dtype=float)

    # Curve fit for f0 and B
    bounds = ([f0_target * 0.95, 0], [f0_target * 1.05, 0.01])
    try:
        popt, pcov = curve_fit(
            inharmonicity_model, n_array, f_array,
            p0=[f0_target, 1e-4], bounds=bounds, maxfev=5000
        )
        f0_fit, B_fit = popt

        # Compute R-squared
        f_pred = inharmonicity_model(n_array, f0_fit, B_fit)
        ss_res = np.sum((f_array - f_pred) ** 2)
        ss_tot = np.sum((f_array - np.mean(f_array)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "B": float(B_fit),
            "f0_measured": float(f0_fit),
            "n_harmonics": len(n_indices),
            "n_fft_used": n_fft,
            "fit_r_squared": float(r_squared),
        }
    except (RuntimeError, ValueError):
        return {
            "B": 0.0, "f0_measured": f0_target,
            "n_harmonics": len(n_indices), "n_fft_used": n_fft,
            "fit_r_squared": 0.0,
        }