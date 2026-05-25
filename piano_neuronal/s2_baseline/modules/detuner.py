"""Detuner module for piano unison string modeling.

Each note can have 2-3 strings (substrings) that are slightly detuned
relative to each other. The Detuner module predicts per-substring detuning
from the pitch, creating natural unison beating effects.
"""

import torch
import torch.nn as nn

from piano_neuronal.s2_baseline.config import N_SUBSTRINGS


class Detuner(nn.Module):
    """Pitch detuning for unison strings.

    Predicts per-substring frequency offset (in cents) from the normalised
    pitch. Adds the global detuning from the z_encoder.

    Args:
        n_substrings: number of strings per note (2 for unison).
    """

    def __init__(self, n_substrings: int = N_SUBSTRINGS):
        super().__init__()
        self.n_substrings = n_substrings
        # Linear projection: normalised pitch -> per-substring detuning
        self.linear = nn.Linear(1, n_substrings)
        # Initialise with small values (detuning is subtle, ~0-5 cents)
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        pitch: torch.Tensor,
        global_detuning: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-substring detuning ratio.

        Args:
            pitch: (B, T) or (B, T, n_synths) — normalised pitch (pitch/128).
            global_detuning: (B, 1) — global detuning from z_encoder.

        Returns:
            If input is (B, T): (B, T, n_substrings)
            If input is (B, T, n_synths): (B, T, n_synths, n_substrings)
        """
        original_shape = pitch.shape
        if pitch.ndim == 3:
            # (B, T, n_synths) → flatten to (B*T*n_synths, 1)
            B, T, S = pitch.shape
            pitch_flat = pitch.reshape(-1, 1)
            detuning_cents = self.linear(pitch_flat)  # (B*T*S, n_substrings)
            detuning_cents = torch.tanh(detuning_cents)
            # Expand global_detuning from (B, 1) to (B*T*S, 1)
            global_det_expanded = global_detuning.unsqueeze(1).unsqueeze(2).expand(B, T, S, 1).reshape(-1, 1)
            detuning_cents = detuning_cents + global_det_expanded
            detuning_ratio = 2.0 ** (detuning_cents / 1200.0)
            return detuning_ratio.reshape(B, T, S, self.n_substrings)
        else:
            # (B, T) → (B, T, n_substrings)
            detuning_cents = self.linear(pitch.unsqueeze(-1))
            detuning_cents = torch.tanh(detuning_cents)
            detuning_cents = detuning_cents + global_detuning.unsqueeze(1)
            return 2.0 ** (detuning_cents / 1200.0)