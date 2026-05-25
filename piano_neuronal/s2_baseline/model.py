"""PianoModel: Full DDSP-Piano baseline model connecting all modules.

Signal flow:
  z_encoder(piano_model) → z, global_inharm, global_detuning
  context_network(conditioning, pedal, z) → context
  reverb_model(piano_model) → reverb_ir
  parallelize(conditioning, context, global_inharm, global_detuning)
  note_release(conditioning) → extended_pitch
  inharm_model(extended_pitch, global_inharm) → inharm_coef
  detuner(extended_pitch, global_detuning) → f0_hz
  monophonic_network(extended_pitch, conditioning, context) → amps, harm_dist, mags
  unparallelize → per-voice controls
  for each voice: inharm_synth + filtered_noise → voice_audio
  sum all voices → dry_signal
  reverb(dry_signal) → output
"""

import torch
import torch.nn as nn
from typing import Tuple

from piano_neuronal.s2_baseline.config import (
    N_SYNTHS, N_HARMONICS, N_NOISE_BANDS, N_SUBSTRINGS, N_INSTRUMENTS,
    Z_DIM, SAMPLE_RATE, N_FRAMES
)
from piano_neuronal.s2_baseline.modules.z_encoder import OneHotZEncoder
from piano_neuronal.s2_baseline.modules.note_release import NoteRelease
from piano_neuronal.s2_baseline.modules.context_network import ContextNetwork
from piano_neuronal.s2_baseline.modules.monophonic_network import MonophonicNetwork
from piano_neuronal.s2_baseline.modules.inharm_model import InharmonicityNetwork
from piano_neuronal.s2_baseline.modules.detuner import Detuner
from piano_neuronal.s2_baseline.modules.inharm_synth import MultiInharmonic
from piano_neuronal.s2_baseline.modules.filtered_noise import DynamicSizeFilteredNoise
from piano_neuronal.s2_baseline.modules.reverb import MultiInstrumentReverb
from piano_neuronal.s2_baseline.modules.parallelizer import Parallelizer
from piano_neuronal.s2_baseline.modules.core import scale_function, midi_to_hz


class PianoModel(nn.Module):
    """Full DDSP-Piano model with all modules assembled in a DAG.

    Args:
        sample_rate: audio sample rate (16000).
        n_harmonics: harmonic partials per voice (96).
        n_noise_bands: noise magnitude bins (64).
        n_synths: max polyphony (16).
        n_substrings: unison strings per note (2).
        n_instruments: number of pianos (1 for baseline).
        z_dim: instrument embedding dimension (16).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_harmonics: int = N_HARMONICS,
        n_noise_bands: int = N_NOISE_BANDS,
        n_synths: int = N_SYNTHS,
        n_substrings: int = N_SUBSTRINGS,
        n_instruments: int = N_INSTRUMENTS,
        z_dim: int = Z_DIM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.n_synths = n_synths
        self.n_substrings = n_substrings

        # Modules
        self.z_encoder = OneHotZEncoder(n_instruments, z_dim)
        self.note_release = NoteRelease()
        self.context_network = ContextNetwork(z_dim=z_dim, n_synths=n_synths)
        self.monophonic_network = MonophonicNetwork(
            n_harmonics=n_harmonics,
            n_noise_bands=n_noise_bands,
        )
        self.inharm_model = InharmonicityNetwork(n_instruments=n_instruments)
        self.detuner = Detuner(n_substrings=n_substrings)
        self.inharm_synth = MultiInharmonic(
            sample_rate=sample_rate,
            n_harmonics=n_harmonics,
            n_substrings=n_substrings,
        )
        self.filtered_noise = DynamicSizeFilteredNoise(
            sample_rate=sample_rate,
            n_bands=n_noise_bands,
        )
        self.reverb = MultiInstrumentReverb(n_instruments=n_instruments)
        self.parallelizer = Parallelizer()

    def forward(
        self,
        conditioning: torch.Tensor,
        pedal: torch.Tensor,
        piano_model: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through the DDSP-Piano model.

        Args:
            conditioning: (B, n_frames, n_synths, 2) — [pitch/128, velocity/128].
            pedal: (B, n_frames, 4) — pedal channels.
            piano_model: (B,) — instrument index (all zeros for baseline).

        Returns:
            signal: (B, n_samples) — final output audio (dry + reverb).
            reverb_ir: (B, ir_length) — learned reverb impulse response.
            non_ir_signal: (B, n_samples) — dry signal before reverb.
        """
        B = conditioning.shape[0]
        T = conditioning.shape[1]
        n_samples = T * (self.sample_rate // 250)  # frame_rate = 250

        # 1. Instrument embedding
        z, global_inharm, global_detuning = self.z_encoder(piano_model)

        # 2. Context network
        context = self.context_network(conditioning, pedal, z)  # (B, T, context_dim)

        # 3. Note release: extend pitch beyond note-off
        extended_cond = self.note_release(conditioning)  # (B, T, n_synths, 2)

        # 4. Extract per-voice controls from extended conditioning
        ext_pitch = extended_cond[..., 0]   # (B, T, n_synths) — normalised pitch
        ext_velocity = extended_cond[..., 1]  # (B, T, n_synths) — velocity

        # 5. Inharmonicity
        inharm_coef = self.inharm_model(ext_pitch, global_inharm)  # (B, T, n_synths)

        # 6. Detuning
        detuning_ratio = self.detuner(ext_pitch, global_detuning)  # (B, T, n_synths, n_substrings)

        # 7. Compute f0_hz for each voice and substring
        # MIDI pitch from normalised pitch
        midi_pitch = ext_pitch * 128.0  # (B, T, n_synths)
        f0_base = midi_to_hz(midi_pitch)  # (B, T, n_synths)

        # Apply detuning for each substring
        f0_hz = f0_base.unsqueeze(-1) * detuning_ratio  # (B, T, n_synths, n_substrings)

        # Expand inharm_coef for substrings
        inharm_expanded = inharm_coef.unsqueeze(-1).expand(-1, -1, -1, self.n_substrings)

        # 8. Parallelize for monophonic processing
        # Process each (batch, synth) pair as an independent voice
        # Reshape: (B, T, n_synths, ...) -> (B*n_synths, T, ...)
        cond_mono = conditioning.reshape(B * self.n_synths, T, 2)  # per-voice [pitch, vel]
        ext_pitch_mono = ext_pitch.reshape(B * self.n_synths, T)
        context_expanded = context.unsqueeze(2).expand(-1, -1, self.n_synths, -1)
        context_mono = context_expanded.reshape(B * self.n_synths, T, -1)

        # 9. Monophonic decoder
        amps, harm_dist, mags = self.monophonic_network(
            ext_pitch_mono, cond_mono, context_mono
        )  # each (B*n_synths, T, D)

        # 10. Un-parallelize controls
        amps = amps.reshape(B, self.n_synths, T, -1).permute(0, 2, 1, 3)  # (B, T, n_synths, 1)
        harm_dist = harm_dist.reshape(B, self.n_synths, T, -1).permute(0, 2, 1, 3)  # (B, T, n_synths, H)
        mags = mags.reshape(B, self.n_synths, T, -1).permute(0, 2, 1, 3)  # (B, T, n_synths, N)

        # 11. Synthesize each voice (skip silent voices)
        non_ir_signal = torch.zeros(B, n_samples, device=conditioning.device, dtype=conditioning.dtype)

        # Active voice mask: a voice is active if its pitch is nonzero in any frame
        voice_active = (ext_pitch.sum(dim=1) > 0)  # (B, n_synths)

        for v in range(self.n_synths):
            # Skip voices with no active notes (pitch all zeros)
            if not voice_active[:, v].any():
                continue

            # Additive synthesis (inharm)
            voice_audio = self.inharm_synth(
                f0_hz[:, :, v, :],       # (B, T, n_substrings)
                amps[:, :, v, :],         # (B, T, 1)
                harm_dist[:, :, v, :],    # (B, T, H)
                inharm_expanded[:, :, v, :],  # (B, T, n_substrings)
            )

            # Filtered noise
            noise_audio = self.filtered_noise(
                mags[:, :, v, :],  # (B, T, N)
                n_samples=n_samples,
            )

            non_ir_signal = non_ir_signal + voice_audio + noise_audio

        # 12. Reverb
        signal = self.reverb(non_ir_signal, piano_model)
        reverb_ir = self.reverb.get_ir(piano_model[0].item())  # (ir_length,)

        return signal, reverb_ir.unsqueeze(0).expand(B, -1), non_ir_signal