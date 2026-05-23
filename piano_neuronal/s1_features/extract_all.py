"""Orchestrator: run all feature extractors on a single note sample."""

import numpy as np
from piano_neuronal.s1_features.inharmonicity import extract_inharmonicity_B
from piano_neuronal.s1_features.decay import extract_t60
from piano_neuronal.s1_features.room_ir import extract_room_ir
from piano_neuronal.s1_features.excitation import extract_excitation
from piano_neuronal.s1_features.spectral import extract_spectral_features


def extract_all_features(
    audio_mono: np.ndarray,
    audio_close_mono: np.ndarray | None,
    audio_ambient_mono: np.ndarray | None,
    sr: int,
    midi_note: int,
    mic_type: str,
    pedal: str,
) -> dict:
    """Run all feature extractors on one note sample.

    Args:
        audio_mono: mono audio for feature extraction (mean of Close channels)
        audio_close_mono: Close mic mono (for IR extraction), or None
        audio_ambient_mono: Ambient mic mono (for IR extraction), or None
        sr: sample rate
        midi_note: MIDI note number
        mic_type: 'Close' or 'Ambient'
        pedal: 'On' or 'Off'

    Returns:
        Flat dict of all extracted features.
    """
    features = {}

    # Inharmonicity
    inharm_result = extract_inharmonicity_B(audio_mono, sr, midi_note)
    features["B"] = inharm_result["B"]
    features["f0_measured"] = inharm_result["f0_measured"]
    features["n_harmonics_detected"] = inharm_result["n_harmonics"]
    features["n_fft_used"] = inharm_result["n_fft_used"]
    features["B_fit_r_squared"] = inharm_result["fit_r_squared"]

    # Decay (bi-exponential)
    decay_result = extract_t60(audio_mono, sr)
    features["tau_fast"] = decay_result["tau_fast"]
    features["tau_slow"] = decay_result["tau_slow"]
    features["t60_from_slow"] = decay_result["t60_from_slow"]
    features["a_fast"] = decay_result["a_fast"]
    features["a_slow"] = decay_result["a_slow"]
    features["decay_r_squared"] = decay_result["fit_r_squared"]
    if "fit_failure_reason" in decay_result:
        features["decay_failure"] = decay_result["fit_failure_reason"]

    # Excitation
    exc_result = extract_excitation(audio_mono, sr, midi_note)
    features["excitation_duration_samples"] = exc_result["duration_samples"]
    # Store excitation arrays separately (not in flat dict — too large)

    # Spectral features
    spec_result = extract_spectral_features(audio_mono, sr)
    features["spectral_centroid_mean"] = spec_result["spectral_centroid_mean"]
    features["spectral_centroid_std"] = spec_result["spectral_centroid_std"]

    # Room IR (only for PedalOff, Close mic, representative notes)
    ir_result = None
    if mic_type == "Close" and pedal == "Off" and audio_close_mono is not None and audio_ambient_mono is not None:
        ir_result = extract_room_ir(audio_close_mono, audio_ambient_mono, sr, onset_idx=0)
        features["ir_duration_s"] = ir_result["ir_duration_s"]
        features["ir_t60"] = ir_result["ir_t60"]
        if "error" in ir_result:
            features["ir_error"] = ir_result["error"]

    return features, exc_result, ir_result