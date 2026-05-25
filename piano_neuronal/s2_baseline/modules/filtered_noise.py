"""Filtered noise synthesizer for DDSP-Piano.

Generates white noise filtered by time-varying FIR filters derived from
magnitude response predictions. Uses overlap-add with windowed impulse
responses for efficiency.
"""

import torch
import torch.nn as nn
from typing import Tuple

from piano_neuronal.s2_baseline.config import N_NOISE_BANDS, SAMPLE_RATE
from piano_neuronal.s2_baseline.modules.core import (
    frequency_impulse_response, fft_convolve, scale_function, upsample_with_window
)


class DynamicSizeFilteredNoise(nn.Module):
    """Time-varying filtered noise synthesizer.

    Takes frame-rate magnitude predictions and outputs sample-rate noise
    filtered through FIR filters derived from those magnitudes.

    Args:
        sample_rate: audio sample rate.
        n_bands: number of frequency bands (magnitudes per frame).
        frame_rate: conditioning frame rate (250 Hz).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_bands: int = N_NOISE_BANDS,
        frame_rate: int = 250,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.frame_rate = frame_rate

    def forward(
        self,
        magnitudes: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Generate filtered noise.

        Args:
            magnitudes: (B, n_frames, n_bands) — magnitude response per frame.
            n_samples: number of output audio samples.

        Returns:
            (B, n_samples) — filtered noise audio.
        """
        n_frames = magnitudes.shape[1]

        # Scale magnitudes to positive range
        magnitudes = scale_function(magnitudes)

        # Upsample magnitudes to sample rate
        mag_up = upsample_with_window(magnitudes, n_samples)  # (B, S, n_bands)

        # Generate impulse responses from magnitudes
        window_size = 2 * self.n_bands + 1  # ~129 for 64 bands
        irs = frequency_impulse_response(mag_up, window_size=window_size)  # (B, S, W)

        # Generate white noise
        noise = torch.randn(n_samples, device=magnitudes.device, dtype=magnitudes.dtype)
        noise = noise.unsqueeze(0).expand(magnitudes.shape[0], -1)  # (B, S)

        # Apply time-varying filter via overlap-add
        # Simplified: segment-based convolution
        hop_size = n_samples // n_frames
        window_size = irs.shape[-1]
        output = torch.zeros(magnitudes.shape[0], n_samples, device=magnitudes.device, dtype=magnitudes.dtype)

        for i in range(n_frames):
            start = i * hop_size
            end = min(start + hop_size + window_size, n_samples)
            seg_len = end - start

            noise_seg = noise[:, start:end]
            if noise_seg.shape[-1] < window_size:
                noise_seg = torch.nn.functional.pad(noise_seg, (0, window_size - noise_seg.shape[-1]))

            ir = irs[:, i, :]  # (B, W)
            filtered = fft_convolve(noise_seg, ir)  # (B, S)
            output[:, start:start + seg_len] += filtered[:, :seg_len]

        # Normalise amplitude
        output = output / (output.abs().max(dim=-1, keepdim=True)[0] + 1e-7) * 0.1

        return output