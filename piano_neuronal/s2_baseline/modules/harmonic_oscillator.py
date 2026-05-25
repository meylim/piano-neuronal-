"""Harmonic oscillator bank for additive synthesis.

Generates sinusoidal partials from frame-rate control signals (f0, amplitudes,
harmonic_distribution). Uses cumulative phase for stability.

NOTE: bf16 safe for this additive (non-recursive) model.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple

from piano_neuronal.s2_baseline.modules.core import remove_above_nyquist, upsample_with_window


class HarmonicOscillator(nn.Module):
    """Bank of harmonic oscillators using phase accumulation.

    Args:
        sample_rate: audio sample rate (16000).
        n_harmonics: number of harmonic partials (96).
        min_frequency: minimum fundamental frequency in Hz (default 20).
    """

    def __init__(self, sample_rate: int = 16000, n_harmonics: int = 96,
                 min_frequency: float = 20.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.min_frequency = min_frequency

    def forward(
        self,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize audio from harmonic parameters.

        Args:
            amplitudes: (B, n_frames, 1) — overall amplitude envelope.
            harmonic_distribution: (B, n_frames, n_harmonics) — per-partial weights.
            f0_hz: (B, n_frames, 1) — fundamental frequency in Hz.

        Returns:
            (B, n_samples) — synthesized audio.
        """
        n_frames = f0_hz.shape[1]
        n_samples = n_frames * (self.sample_rate // 250)  # frame_rate = 250

        # Upsample frame-rate controls to sample rate
        f0_upsampled = upsample_with_window(f0_hz, n_samples)  # (B, n_samples, 1)
        amp_upsampled = upsample_with_window(amplitudes, n_samples)  # (B, n_samples, 1)
        hd_upsampled = upsample_with_window(harmonic_distribution, n_samples)  # (B, n_samples, n_harmonics)

        # Mask frequencies below minimum and clamp inf/nan
        f0_upsampled = torch.where(
            f0_upsampled < self.min_frequency,
            torch.zeros_like(f0_upsampled),
            f0_upsampled
        )
        f0_upsampled = torch.nan_to_num(f0_upsampled, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute harmonic frequencies: f_n = f0 * n
        harmonic_indices = torch.arange(1, self.n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype)
        harmonic_freqs = f0_upsampled * harmonic_indices  # (B, n_samples, n_harmonics)

        # Remove above Nyquist
        harmonic_amps = remove_above_nyquist(harmonic_freqs, hd_upsampled, self.sample_rate)

        # Normalise harmonic distribution (safe: avoid division by near-zero)
        harm_sum = harmonic_amps.sum(dim=-1, keepdim=True)
        harmonic_amps = torch.where(
            harm_sum > 1e-6,
            harmonic_amps / harm_sum,
            torch.zeros_like(harmonic_amps)
        )

        # Apply overall amplitude
        harmonic_amps = harmonic_amps * amp_upsampled  # (B, n_samples, n_harmonics)

        # Generate phase: cumsum of angular frequency
        angular_freq = 2.0 * math.pi * harmonic_freqs / self.sample_rate  # (B, n_samples, n_harmonics)
        phase = torch.cumsum(angular_freq, dim=1)  # (B, n_samples, n_harmonics)

        # Synthesize: sum of cosines
        waveforms = harmonic_amps * torch.cos(phase)  # (B, n_samples, n_harmonics)
        audio = waveforms.sum(dim=-1)  # (B, n_samples)

        return audio