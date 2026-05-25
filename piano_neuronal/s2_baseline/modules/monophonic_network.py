"""Monophonic deep network for DDSP-Piano.

Processes pitch, conditioning, and context for a single voice and outputs
the synthesizer control parameters: amplitudes, harmonic_distribution, and
magnitudes (for filtered noise).

Architecture (from gin config):
  Input: (extended_pitch/128, conditioning/128, context) = 35
  Dense(128) -> LeakyReLU -> GRU(192) -> Dense(192) -> LeakyReLU -> LayerNorm
  -> Dense(192) -> split into [1, 96, 64] = amplitudes + harmonic_dist + magnitudes
"""

import torch
import torch.nn as nn

from piano_neuronal.s2_baseline.config import (
    N_HARMONICS, N_NOISE_BANDS, MONO_GRU_UNITS, MONO_DENSE_UNITS,
    N_SYNTHS, Z_DIM
)


class FcStack(nn.Module):
    """Fully-connected stack: Linear → LeakyReLU → Linear → LeakyReLU."""

    def __init__(self, output_size: int, n_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.LazyLinear(output_size))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MonophonicNetwork(nn.Module):
    """Deep monophonic decoder producing synthesizer controls.

    Output splits: amplitudes (1), harmonic_distribution (n_harmonics),
    magnitudes (n_noise_bands).

    Args:
        n_harmonics: number of harmonic partials (96).
        n_noise_bands: number of noise magnitude bins (64).
        gru_units: GRU hidden size (192).
        dense_units: dense layer size (192).
    """

    def __init__(
        self,
        n_harmonics: int = N_HARMONICS,
        n_noise_bands: int = N_NOISE_BANDS,
        gru_units: int = MONO_GRU_UNITS,
        dense_units: int = MONO_DENSE_UNITS,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.output_dim = 1 + n_harmonics + n_noise_bands  # 161

        # Input stacks (from gin config):
        # pitch: 1 -> Dense(64, 3)
        # conditioning: 2*n_synths -> Dense(64, 3)
        # context: context_dim -> Dense(64, 3)
        self.pitch_stack = FcStack(64, 3)
        self.cond_stack = FcStack(64, 3)
        self.context_stack = FcStack(64, 3)

        # GRU
        self.gru = nn.GRU(
            input_size=64 + 64 + 64,
            hidden_size=gru_units,
            batch_first=True,
        )

        # Output: GRU hidden -> Dense(192) -> LeakyReLU -> LayerNorm -> Dense(output_dim)
        self.output_dense = nn.Linear(gru_units, dense_units)
        self.layer_norm = nn.LayerNorm(dense_units)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.final_dense = nn.Linear(dense_units, self.output_dim)

    def forward(
        self,
        pitch: torch.Tensor,
        conditioning: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple:
        """Produce synthesizer control parameters for a single voice.

        Args:
            pitch: (B, n_frames) — normalised pitch (pitch/128).
            conditioning: (B, n_frames, 2) — [active, velocity] for this voice.
            context: (B, n_frames, context_dim) — FiLM context.

        Returns:
            amplitudes: (B, n_frames, 1) — overall amplitude.
            harmonic_distribution: (B, n_frames, n_harmonics) — per-partial weights.
            magnitudes: (B, n_frames, n_noise_bands) — noise magnitude response.
        """
        # Normalise inputs
        pitch_norm = pitch.unsqueeze(-1) / 128.0  # (B, T, 1)

        # Process through stacks
        pitch_feat = self.pitch_stack(pitch_norm)  # (B, T, 64)
        cond_feat = self.cond_stack(conditioning)  # (B, T, 64)
        ctx_feat = self.context_stack(context)  # (B, T, 64)

        # Concatenate and run GRU
        gru_input = torch.cat([pitch_feat, cond_feat, ctx_feat], dim=-1)  # (B, T, 192)
        gru_out, _ = self.gru(gru_input)  # (B, T, gru_units)

        # Output projection
        hidden = self.leaky_relu(self.layer_norm(self.output_dense(gru_out)))
        raw_output = self.final_dense(hidden)  # (B, T, 1 + n_harmonics + n_noise_bands)

        # Split output
        amplitudes = raw_output[..., 0:1]  # (B, T, 1)
        harmonic_distribution = raw_output[..., 1:1 + self.n_harmonics]  # (B, T, n_harmonics)
        magnitudes = raw_output[..., 1 + self.n_harmonics:]  # (B, T, n_noise_bands)

        return amplitudes, harmonic_distribution, magnitudes