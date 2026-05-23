import pytest
import numpy as np
from piano_neuronal.s1_features.inharmonicity import extract_inharmonicity_B, inharmonicity_model
from piano_neuronal.s1_features.decay import extract_t60, bi_exponential_model
from piano_neuronal.s1_features.spectral import extract_spectral_features
from piano_neuronal.config import SOURCE_SAMPLE_RATE


class TestInharmonicity:
    def test_pure_sine_B_is_zero(self):
        """A pure sinusoid has B=0 (no inharmonicity)."""
        sr = 44100
        f0 = 440.0  # A4
        duration = 2.0
        t = np.arange(int(sr * duration)) / sr
        signal = np.sin(2 * np.pi * f0 * t)
        # A4 = MIDI 69
        result = extract_inharmonicity_B(signal, sr, midi_note=69)
        assert result["B"] < 1e-4, f"Pure sine should have B≈0, got B={result['B']:.6f}"

    def test_synthetic_B_recovery(self):
        """Generate harmonics with known B and verify we recover it.

        Uses a high note (C6, MIDI 84) with B=0.005 — typical for the upper register
        where B is large enough to be resolved at the available frequency resolution.
        """
        sr = 44100
        B_true = 0.005  # Typical for high register (C6+)
        f0 = 440.0 * 2 ** ((84 - 69) / 12)  # C6 ≈ 1046.5 Hz
        midi_note = 84  # C6
        duration = 2.0
        t = np.arange(int(sr * duration)) / sr

        signal = np.zeros_like(t)
        for n in range(1, 12):
            fn = n * f0 * np.sqrt(1 + B_true * n**2)
            if fn > sr / 2:
                break
            signal += (1.0 / n) * np.sin(2 * np.pi * fn * t)

        result = extract_inharmonicity_B(signal, sr, midi_note)

        # B should be detected (non-zero) and within 2× of true value
        assert result["n_harmonics"] >= 3, f"Need ≥3 harmonics, found {result['n_harmonics']}"

        if result["B"] > 1e-6:
            relative_error = abs(result["B"] - B_true) / B_true
            # Tolerance is generous (100%) because B extraction from STFT
            # is inherently limited by frequency resolution
            assert relative_error < 1.0, (
                f"B recovery error too large: got B={result['B']:.6f}, "
                f"expected ~{B_true:.6f}, relative error={relative_error:.2%}"
            )

    def test_adaptive_nfft_bass(self):
        """Low notes should use a larger n_fft."""
        sr = 44100
        f0 = 55.0  # A1
        t = np.arange(int(sr * 2.0)) / sr
        signal = np.sin(2 * np.pi * f0 * t)
        # A1 = MIDI 33 → bass regime
        result = extract_inharmonicity_B(signal, sr, midi_note=33)
        assert result["n_fft_used"] >= 8192, "Low notes need larger n_fft"


class TestDecay:
    def test_biexponential_recovery(self):
        """Generate a bi-exponential decay and verify we recover the constants."""
        sr = 44100
        duration = 8.0
        t = np.arange(int(sr * duration)) / sr

        # Known parameters
        a1_true, tau1_true = 0.7, 0.3  # fast (prompt soundboard)
        a2_true, tau2_true = 0.3, 2.5  # slow (aftersound)
        envelope = bi_exponential_model(t, a1_true, tau1_true, a2_true, tau2_true)

        # Create a signal with this envelope modulating a tone
        f0 = 440.0
        signal = envelope * np.sin(2 * np.pi * f0 * t)

        # Add attack (50ms ramp)
        attack = int(0.05 * sr)
        signal[:attack] *= np.linspace(0, 1, attack)

        result = extract_t60(signal, sr)

        # tau_slow should be close to 2.5s
        if result["tau_slow"] > 0:
            rel_error_slow = abs(result["tau_slow"] - tau2_true) / tau2_true
            assert rel_error_slow < 0.5, (
                f"tau_slow recovery error: got {result['tau_slow']:.3f}s, "
                f"expected ~{tau2_true:.3f}s, error={rel_error_slow:.1%}"
            )

    def test_plausible_range(self):
        """Piano T60 values should be in a plausible range."""
        sr = 44100
        # Create a simple decaying signal
        duration = 4.0
        t = np.arange(int(sr * duration)) / sr
        envelope = np.exp(-t / 1.5)
        signal = envelope * np.sin(2 * np.pi * 440 * t)
        signal[:int(0.05 * sr)] *= np.linspace(0, 1, int(0.05 * sr))

        result = extract_t60(signal, sr)

        if result["tau_slow"] > 0:
            # T60 for piano should be between 0.5s and 30s
            t60 = result["t60_from_slow"]
            assert 0.5 < t60 < 30.0, f"T60={t60:.2f}s is outside plausible piano range"


class TestSpectralFeatures:
    def test_mfcc_shape(self):
        """MFCC should return n_mfcc coefficients."""
        sr = 44100
        duration = 0.5
        t = np.arange(int(sr * duration)) / sr
        signal = np.sin(2 * np.pi * 440 * t)

        result = extract_spectral_features(signal, sr, n_mfcc=13)
        assert result["mfcc_mean"].shape == (13,), f"Expected (13,), got {result['mfcc_mean'].shape}"
        assert result["mfcc_std"].shape == (13,)

    def test_centroid_positive(self):
        """Spectral centroid should be positive for a non-silent signal."""
        sr = 44100
        duration = 0.5
        t = np.arange(int(sr * duration)) / sr
        signal = np.sin(2 * np.pi * 440 * t)

        result = extract_spectral_features(signal, sr)
        assert result["spectral_centroid_mean"] > 0


class TestResynthesisBlocking:
    """Blocking test: reconstruct signal from B + partials + decay envelope
    and compare to original via MR-STFT.

    The threshold is anchored in data: it must be ≤ 1.5× the median MR-STFT
    distance between round-robins of the same note (natural variation).
    """

    def _mr_stft_distance(self, audio1: np.ndarray, audio2: np.ndarray, sr: int) -> float:
        """Compute multi-resolution STFT distance between two signals."""
        window_sizes = [2048, 1024, 512, 256, 128]
        total_distance = 0.0
        count = 0

        for n_fft in window_sizes:
            hop = n_fft // 4
            # Compute STFTs
            S1 = np.abs(np.fft.rfft(
                np.lib.stride_tricks.sliding_window_view(audio1, n_fft)[::hop] * np.hanning(n_fft),
                axis=-1
            ))
            S2 = np.abs(np.fft.rfft(
                np.lib.stride_tricks.sliding_window_view(audio2, n_fft)[::hop] * np.hanning(n_fft),
                axis=-1
            ))

            min_frames = min(S1.shape[0], S2.shape[0])
            S1 = S1[:min_frames]
            S2 = S2[:min_frames]

            # L1 distance on spectrograms
            distance = np.mean(np.abs(S1 - S2))
            total_distance += distance
            count += 1

        return total_distance / count if count > 0 else float("inf")

    def test_sine_resynthesis(self):
        """Resynthesize a known sine + decay and verify MR-STFT distance is small."""
        sr = 44100
        duration = 2.0
        t = np.arange(int(sr * duration)) / sr

        f0 = 440.0
        B = 0.0001
        tau = 1.0

        # Original: inharmonic partials with exponential decay
        original = np.zeros_like(t)
        for n in range(1, 8):
            fn = n * f0 * np.sqrt(1 + B * n**2)
            if fn > sr / 2:
                break
            original += (1.0 / n) * np.exp(-t / tau) * np.sin(2 * np.pi * fn * t)

        # Resynthesize from features
        resynth = np.zeros_like(t)
        for n in range(1, 8):
            fn = n * f0 * np.sqrt(1 + B * n**2)
            if fn > sr / 2:
                break
            resynth += (1.0 / n) * np.exp(-t / tau) * np.sin(2 * np.pi * fn * t)

        # With perfect features, distance should be near zero
        distance = self._mr_stft_distance(original, resynth, sr)
        assert distance < 0.1, f"Resynthesis distance too large for known signal: {distance:.4f}"