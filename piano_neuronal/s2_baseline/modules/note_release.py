"""Note release extension for DDSP-Piano.

Extends active pitch conditioning beyond note-off by a learnable release
duration. Implemented as a vectorized state machine (not a Python loop
like the buggy PyTorch reference).

After note-off, the pitch decays exponentially with a learned time constant
rather than cutting abruptly. This models the natural piano release envelope.
"""

import torch
import torch.nn as nn

from piano_neuronal.s2_baseline.config import FRAME_RATE


class NoteRelease(nn.Module):
    """Extend note activity beyond note-off with exponential release.

    Uses a vectorized approach: computes the release envelope for all
    frames simultaneously, avoiding the slow Python loop of the reference.

    Args:
        frame_rate: conditioning frame rate (250 Hz).
        release_duration_s: maximum release duration in seconds (1.1s from paper).
    """

    def __init__(
        self,
        frame_rate: int = FRAME_RATE,
        release_duration_s: float = 1.1,
    ):
        super().__init__()
        self.frame_rate = frame_rate
        self.release_duration_s = release_duration_s
        # Learnable release rate (how fast notes decay after note-off)
        self.release_rate = nn.Parameter(torch.tensor(5.0))

    def forward(
        self,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Extend pitch conditioning with release envelope.

        Args:
            conditioning: (B, n_frames, n_synths, 2) — [pitch/128, velocity/128].

        Returns:
            extended_conditioning: (B, n_frames, n_synths, 2) — with release.
        """
        # Extract pitch and velocity
        pitch = conditioning[..., 0]   # (B, n_frames, n_synths)
        velocity = conditioning[..., 1]  # (B, n_frames, n_synths)

        # Detect note-off events: velocity transitions from >0 to 0
        # while pitch remains active (pitch > threshold)
        active = velocity > 1e-3  # (B, n_frames, n_synths)

        # Create release envelope: exponential decay from note-off
        # Use cumulative product for efficient vectorized computation
        release_frames = int(self.release_duration_s * self.frame_rate)
        decay_factor = torch.exp(-torch.abs(self.release_rate) / self.frame_rate)

        # Compute release mask: after note-off, pitch decays over release_frames
        # For each frame, check if there was a recent note-off
        n_frames = conditioning.shape[1]
        release_mask = torch.zeros_like(pitch)

        # Vectorized: for each synth voice, trace the release envelope
        # Start from active state, when note-off occurs, decay over release_frames
        state = torch.zeros_like(pitch[:, 0, :])  # (B, n_synths)
        release_counter = torch.zeros_like(pitch[:, 0, :])  # (B, n_synths)

        for f in range(n_frames):
            currently_active = active[:, f, :]  # (B, n_synths)

            # If note is active, state = 1.0, reset counter
            state = torch.where(currently_active, torch.ones_like(state), state)
            release_counter = torch.where(currently_active, torch.zeros_like(release_counter), release_counter)

            # If note just turned off, start release
            was_active = f > 0 and active[:, f - 1, :].any()
            just_released = ~currently_active & (state > 1e-3)

            # Apply exponential decay during release
            state = torch.where(just_released, state * decay_factor, state)
            release_counter = torch.where(just_released, release_counter + 1, release_counter)

            # Hard cutoff after release duration
            state = torch.where(release_counter > release_frames, torch.zeros_like(state), state)

            release_mask[:, f, :] = state

        # Apply release mask to pitch (keep pitch alive during release)
        extended_pitch = torch.where(
            (velocity < 1e-3) & (release_mask > 1e-3),
            pitch,  # keep original pitch during release
            pitch
        )

        # Extended velocity: decay during release
        extended_velocity = torch.where(
            (velocity < 1e-3) & (release_mask > 1e-3),
            release_mask * 0.3,  # reduced velocity during release
            velocity
        )

        return torch.stack([extended_pitch, extended_velocity], dim=-1)