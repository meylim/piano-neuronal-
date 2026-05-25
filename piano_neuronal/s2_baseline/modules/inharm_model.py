"""Parametric inharmonicity model for piano strings.

Implements the two-bridge model from Rigaud et al. (DAFx-2011):
  B(pitch) = 10^(alpha_b + beta_b * (pitch - pitch_ref) / 88)
                * 10^(alpha_t + beta_t * (pitch - pitch_ref) / 88)

The model is physics-informed: treble inharmonicity increases with pitch,
bass inharmonicity has a different slope. Learnable parameters adjust
the curve per instrument.
"""

import torch
import torch.nn as nn
import math

from piano_neuronal.s2_baseline.config import (
    N_INSTRUMENTS, INHARM_TREBLE_SLOPE_INIT, INHARM_BASS_SLOPE_INIT
)


class InharmonicityNetwork(nn.Module):
    """Parametric inharmonicity B(pitch) model.

    Computes per-string inharmonicity coefficients using physics-based
    priors with learnable per-instrument modifiers.

    The B coefficient controls how much harmonics deviate from integer
    multiples of the fundamental: f_n = f0 * n * sqrt(B * n^2 + 1).

    Args:
        n_instruments: number of instruments (1 for baseline).
        pitch_ref: reference MIDI note for the slope (default 60 = C4).
    """

    def __init__(
        self,
        n_instruments: int = N_INSTRUMENTS,
        pitch_ref: float = 60.0,
    ):
        super().__init__()
        self.n_instruments = n_instruments
        self.pitch_ref = pitch_ref

        # Learnable per-instrument embeddings (initialised from physics priors)
        # alpha values are log10 of the |B| initial slopes.
        # Bass slope init is negative (B decreases with pitch in bass), so we use abs()
        self.alpha_b = nn.Parameter(torch.tensor(abs(INHARM_BASS_SLOPE_INIT)).log10())
        self.beta_b = nn.Parameter(torch.tensor(0.0))
        # Treble slope: B increases with pitch (positive init)
        self.alpha_t = nn.Parameter(torch.tensor(abs(INHARM_TREBLE_SLOPE_INIT)).log10())
        self.beta_t = nn.Parameter(torch.tensor(0.0))

        # Global modifiers from z_encoder
        self.global_inharm_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        pitch: torch.Tensor,
        global_inharm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute inharmonicity coefficients B for each pitch.

        Args:
            pitch: (B, n_frames) or (B, n_frames, n_substrings) — MIDI pitch.
            global_inharm: (B, 1) — global inharmonicity modifier from z_encoder.

        Returns:
            inharm_coef: same shape as pitch — B coefficient per note.
        """
        # Normalised pitch distance from reference
        pitch_offset = (pitch - self.pitch_ref) / 88.0

        # Two-bridge model: B(p) = 10^(alpha_b + beta_b * offset) + 10^(alpha_t + beta_t * offset)
        bass_term = 10.0 ** (self.alpha_b + self.beta_b * pitch_offset)
        treble_term = 10.0 ** (self.alpha_t + self.beta_t * pitch_offset)

        inharm = bass_term + treble_term  # B(p)

        # Apply global modifier from z_encoder (broadcast over frames and voices)
        inharm = inharm * (1.0 + global_inharm.unsqueeze(-1))

        return inharm