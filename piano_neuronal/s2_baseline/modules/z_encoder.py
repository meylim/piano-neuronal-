"""Instrument embedding (Z-encoder) for DDSP-Piano.

For the baseline reproduction with n_instruments=1 (single sfizz piano),
this reduces to a trivial single-vector embedding. The architecture is
kept general for future multi-instrument extension.
"""

import torch
import torch.nn as nn

from piano_neuronal.s2_baseline.config import N_INSTRUMENTS, Z_DIM


class OneHotZEncoder(nn.Module):
    """Instrument embedding producing (z, global_inharm, global_detuning).

    For n_instruments=1, this is a single learnable vector of dimension z_dim.

    Args:
        n_instruments: number of instruments (1 for baseline).
        z_dim: embedding dimension.
    """

    def __init__(
        self,
        n_instruments: int = N_INSTRUMENTS,
        z_dim: int = Z_DIM,
    ):
        super().__init__()
        self.n_instruments = n_instruments
        self.z_dim = z_dim

        # Instrument embedding
        self.z_embedding = nn.Embedding(n_instruments, z_dim)
        # Global inharmonicity modifier (scalar per instrument)
        self.inharm_embedding = nn.Embedding(n_instruments, 1)
        # Global detuning modifier (scalar per instrument)
        self.detune_embedding = nn.Embedding(n_instruments, 1)

    def forward(
        self,
        piano_model: torch.Tensor,
    ) -> tuple:
        """Get instrument embedding.

        Args:
            piano_model: (B,) — instrument index (all zeros for baseline).

        Returns:
            z: (B, z_dim) — instrument embedding.
            global_inharm: (B, 1) — inharmonicity modifier.
            global_detune: (B, 1) — detuning modifier.
        """
        z = self.z_embedding(piano_model)          # (B, z_dim)
        global_inharm = self.inharm_embedding(piano_model)   # (B, 1)
        global_detune = self.detune_embedding(piano_model)   # (B, 1)

        return z, global_inharm, global_detune