"""FiLM-context network for DDSP-Piano.

Processes conditioning (pitch, velocity), pedal, and instrument embedding
through a GRU + dense network to produce context vectors that modulate
the monophonic decoder via FiLM (Feature-wise Linear Modulation).
"""

import torch
import torch.nn as nn

from piano_neuronal.s2_baseline.config import (
    Z_DIM, N_SYNTHS, CONTEXT_DENSE_UNITS, CONTEXT_GRU_UNITS
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


class ContextNetwork(nn.Module):
    """FiLM-style context network.

    Input: collapsed conditioning (pitch+velocity), pedal, and z embedding.
    Output: context vector for FiLM modulation of the monophonic decoder.

    Architecture (from gin config):
      - Conditioning head: FcStack(32, 2)
      - Pedal head: FcStack(16, 2)
      - Piano ID head: Embedding(n_instruments, 32)
      - GRU: input=32+16+32+32=112, hidden=64, output=64
      - Dense → LayerNorm → LeakyReLU → Dense(32)

    Args:
        z_dim: instrument embedding dimension.
        n_synths: max polyphony.
        context_dim: output context dimension (32).
    """

    def __init__(
        self,
        z_dim: int = Z_DIM,
        n_synths: int = N_SYNTHS,
        context_dim: int = 32,
        n_instruments: int = 1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_synths = n_synths
        self.context_dim = context_dim

        # Input: collapsed conditioning (n_synths * 2) + pedal (4) + z (z_dim)
        cond_dim = n_synths * 2  # pitch + velocity per synth
        pedal_dim = 4
        total_input = cond_dim + pedal_dim + z_dim

        self.conditioning_head = FcStack(CONTEXT_DENSE_UNITS, 2)
        self.pedal_head = FcStack(16, 2)

        self.gru = nn.GRU(
            input_size=CONTEXT_DENSE_UNITS + 16 + z_dim,
            hidden_size=CONTEXT_GRU_UNITS,
            batch_first=True,
        )

        self.output_dense = nn.Linear(CONTEXT_GRU_UNITS, context_dim)
        self.layer_norm = nn.LayerNorm(context_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        conditioning: torch.Tensor,
        pedal: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Process conditioning into context vectors.

        Args:
            conditioning: (B, n_frames, n_synths, 2) — [pitch, velocity].
            pedal: (B, n_frames, 4) — pedal channels.
            z: (B, z_dim) — instrument embedding (broadcast over time).

        Returns:
            context: (B, n_frames, context_dim) — context for FiLM.
        """
        B, T, S, D = conditioning.shape

        # Collapse synth dimension: (B, T, S*2)
        cond_flat = conditioning.reshape(B, T, S * D)

        # Process through heads
        cond_feat = self.conditioning_head(cond_flat)  # (B, T, 32)
        pedal_feat = self.pedal_head(pedal)  # (B, T, 16)

        # Broadcast z over time: (B, z_dim) -> (B, T, z_dim)
        z_expanded = z.unsqueeze(1).expand(-1, T, -1)

        # Concatenate and run through GRU
        gru_input = torch.cat([cond_feat, pedal_feat, z_expanded], dim=-1)
        gru_out, _ = self.gru(gru_input)  # (B, T, 64)

        # Output projection
        context = self.leaky_relu(self.layer_norm(self.output_dense(gru_out)))

        return context