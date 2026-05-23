"""Spectral feature extraction: centroid and MFCC."""

import numpy as np
import librosa

from piano_neuronal.config import MFCC_N_MFCC, MFCC_N_FFT


def extract_spectral_features(
    audio_mono: np.ndarray,
    sr: int,
    n_mfcc: int = MFCC_N_MFCC,
    n_fft: int = MFCC_N_FFT,
) -> dict:
    """Extract spectral centroid and MFCC from the attack portion.

    Returns dict with:
        spectral_centroid_mean: float — mean centroid in Hz
        spectral_centroid_std: float — std of centroid in Hz
        mfcc_mean: np.ndarray — (n_mfcc,) averaged over attack frames
        mfcc_std: np.ndarray — (n_mfcc,) std over attack frames
    """
    # Use first 500ms for spectral features (captures attack and early sustain)
    attack_duration = min(len(audio_mono), int(0.5 * sr))
    attack_audio = audio_mono[:attack_duration]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=attack_audio, sr=sr, n_fft=n_fft, hop_length=n_fft // 4
    )[0]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=attack_audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_fft // 4
    )

    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "mfcc_mean": np.mean(mfcc, axis=1),  # (n_mfcc,)
        "mfcc_std": np.std(mfcc, axis=1),     # (n_mfcc,)
    }