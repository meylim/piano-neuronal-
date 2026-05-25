"""Inharmonic additive synthesis for piano strings.

MultiInharmonic handles 2-3 unison strings per note, each with its own
inharmonicity coefficient B and slight detuning, as in the DDSP-Piano paper.

The inharmonic frequency formula: f_n = f0 * n * sqrt(B * n^2 + 1)
where B is the inharmonicity coefficient from the physics model.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple

from piano_neuronal.s2_baseline.config import (
    N_HARMONICS, N_SUBSTRINGS, SAMPLE_RATE
)
from piano_neuronal.s2_baseline.modules.harmonic_oscillator import HarmonicOscillator
from piano_neuronal.s2_baseline.modules.core import remove_above_nyquist, upsample_with_window


def get_inharmonic_frequencies(
    f0_hz: torch.Tensor,
    inharm_coef: torch.Tensor,
    n_harmonics: int = N_HARMONICS,
) -> torch.Tensor:
    """Compute inharmonic partial frequencies.

    f_n = f0 * n * sqrt(B * n^2 + 1)

    Args:
        f0_hz: (B, n_frames, 1 or n_substrings) — fundamental frequency.
        inharm_coef: (B, n_frames, 1 or n_substrings) — inharmonicity coefficient B.
        n_harmonics: number of partials.

    Returns:
        (B, n_frames, n_harmonics) — inharmonic frequencies in Hz.
    """
    n = torch.arange(1, n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype)
    # f_n = f0 * n * sqrt(B * n^2 + 1)
    freqs = f0_hz * n * torch.sqrt(inharm_coef * n ** 2 + 1.0)
    return freqs


class MultiInharmonic(nn.Module):
    """Multi-substring additive synthesizer with inharmonicity.

    Each note can have n_substrings (2 for unison strings), each with
    its own B coefficient and slight detuning. Outputs are summed.

    Args:
        sample_rate: audio sample rate.
        n_harmonics: partials per string.
        n_substrings: number of strings per note (2 for unison).
        min_frequency: minimum f0 in Hz.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_harmonics: int = N_HARMONICS,
        n_substrings: int = N_SUBSTRINGS,
        min_frequency: float = 20.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.n_substrings = n_substrings
        self.min_frequency = min_frequency

    def forward(
        self,
        f0_hz: torch.Tensor,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        inharm_coef: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize audio from inharmonic harmonic parameters.

        Args:
            f0_hz: (B, n_frames, n_substrings) — per-string fundamentals.
            amplitudes: (B, n_frames, 1) — overall amplitude.
            harmonic_distribution: (B, n_frames, n_harmonics) — per-partial weights.
            inharm_coef: (B, n_frames, n_substrings) — per-string B coefficient.

        Returns:
            (B, n_samples) — summed audio from all substrings.
        """
        n_frames = f0_hz.shape[1]
        n_samples = n_frames * (self.sample_rate // 250)

        # Upsample controls to sample rate
        amp_up = upsample_with_window(amplitudes, n_samples)  # (B, S, 1)
        hd_up = upsample_with_window(harmonic_distribution, n_samples)  # (B, S, H)
        f0_up = upsample_with_window(f0_hz, n_samples)  # (B, S, n_substrings)
        inharm_up = upsample_with_window(inharm_coef, n_samples)  # (B, S, n_substrings)

        # Mask low fundamentals and clamp inf/nan from upsampled values
        f0_up = torch.where(f0_up < self.min_frequency, torch.zeros_like(f0_up), f0_up)
        f0_up = torch.nan_to_num(f0_up, nan=0.0, posinf=0.0, neginf=0.0)

        # Sum across substrings
        audio = torch.zeros(n_samples, device=f0_hz.device, dtype=f0_hz.dtype)
        audio = audio.unsqueeze(0).expand(f0_hz.shape[0], -1)  # (B, S)

        harmonic_indices = torch.arange(
            1, self.n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype
        )

        for s in range(self.n_substrings):
            f0_s = f0_up[:, :, s:s+1]  # (B, S, 1)
            B_s = inharm_up[:, :, s:s+1]  # (B, S, 1)

            # Inharmonic frequencies
            freqs_s = f0_s * harmonic_indices * torch.sqrt(
                B_s * harmonic_indices ** 2 + 1.0
            )  # (B, S, H)

            # Mask above Nyquist
            amps_s = remove_above_nyquist(freqs_s, hd_up, self.sample_rate)

            # Normalise (safe: avoid division by near-zero)
            amps_sum = amps_s.sum(dim=-1, keepdim=True)
            amps_s = torch.where(
                amps_sum > 1e-6,
                amps_s / amps_sum,
                torch.zeros_like(amps_s)
            )
            amps_s = amps_s * amp_up  # (B, S, H)

            # Phase accumulation
            angular_freq = 2.0 * math.pi * freqs_s / self.sample_rate
            phase = torch.cumsum(angular_freq, dim=1)
            waveforms = amps_s * torch.cos(phase)  # (B, S, H)
            audio = audio + waveforms.sum(dim=-1)  # (B, S)

        return audio