"""Learned impulse response reverb for DDSP-Piano.

MultiInstrumentReverb uses a learned FIR impulse response per instrument,
convolved via FFT with the dry signal. For n_instruments=1, this is a
single learned IR.

The IR is constrained with an exponential decay mask during training to
encourage realistic reverb behaviour.
"""

import torch
import torch.nn as nn
import math

from piano_neuronal.s2_baseline.config import (
    N_INSTRUMENTS, REVERB_IR_LENGTH, SAMPLE_RATE, REVERB_IR_DURATION_S
)
from piano_neuronal.s2_baseline.modules.core import fft_convolve


class MultiInstrumentReverb(nn.Module):
    """Learned impulse response reverb (one IR per instrument).

    For the baseline reproduction with n_instruments=1, this reduces to
    a single learnable IR. The IR is initialised with a small exponential
    decay and constrained during training.

    Args:
        n_instruments: number of instruments (1 for baseline).
        ir_length: length of the impulse response in samples.
        sample_rate: audio sample rate.
    """

    def __init__(
        self,
        n_instruments: int = N_INSTRUMENTS,
        ir_length: int = REVERB_IR_LENGTH,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.n_instruments = n_instruments
        self.ir_length = ir_length
        self.sample_rate = sample_rate

        # Learnable IR: (n_instruments, ir_length)
        # Initialise with small exponential decay
        ir = torch.zeros(n_instruments, ir_length)
        for i in range(n_instruments):
            # Exponential decay starting at sample 1
            decay = torch.exp(-8.0 * torch.linspace(0, 1, ir_length))
            # Small random initial impulse, then decay
            ir[i, 0] = 1.0  # direct sound at t=0 (masked later)
            ir[i, 1:] = decay[1:] * 0.01  # very low reverb level initially

        self.ir = nn.Parameter(ir)

    def forward(
        self,
        dry_signal: torch.Tensor,
        piano_model: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reverb to dry signal.

        Args:
            dry_signal: (B, n_samples) — dry audio.
            piano_model: (B,) — instrument index (all zeros for baseline).

        Returns:
            (B, n_samples) — wet audio (dry + reverb).
        """
        n_samples = dry_signal.shape[-1]

        # Apply exponential decay mask to IR for stability
        decay_mask = torch.exp(
            -4.0 * torch.linspace(0, 1, self.ir_length, device=dry_signal.device, dtype=dry_signal.dtype)
        )
        # Start mask from sample 1 (keep direct sound unmasked)
        mask = torch.ones(self.ir_length, device=dry_signal.device, dtype=dry_signal.dtype)
        mask[1:] = decay_mask[1:] ** 4.0  # stronger decay

        # Get IR for each sample in batch
        ir_masked = self.ir * mask.unsqueeze(0)  # (n_instruments, ir_length)

        # Gather IRs for each sample in batch
        irs = ir_masked[piano_model]  # (B, ir_length)

        # Zero out the first sample to remove direct sound (it's already in dry)
        irs[:, 0] = 0.0

        # Convolve via FFT
        wet_reverb = fft_convolve(dry_signal, irs)  # (B, n_samples)

        # Mix: dry + wet
        output = dry_signal + wet_reverb

        return output

    def get_ir(self, piano_model: int = 0) -> torch.Tensor:
        """Get the impulse response for a given instrument."""
        return self.ir[piano_model].detach()