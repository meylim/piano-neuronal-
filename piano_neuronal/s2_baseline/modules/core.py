"""DDSP signal-processing primitives for differentiable synthesis.

PyTorch reimplementation of core DDSP operations, corrected from the
reference TF implementation. All tensors are (B, T) or (B, T, D).
No .cuda() calls — use .to(device) throughout.

NOTE: bf16 mixed precision is safe for this additive (non-recursive) model.
For a future waveguide/recursive model (Axe A), precision must be reevaluated.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def midi_to_hz(notes: torch.Tensor) -> torch.Tensor:
    """Convert MIDI note numbers to frequencies in Hz. A4 = 69 = 440 Hz."""
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def hz_to_midi(freq_hz: torch.Tensor) -> torch.Tensor:
    """Convert frequencies in Hz to MIDI note numbers."""
    return 12.0 * torch.log2(freq_hz / 440.0 + 1e-12) + 69.0


def scale_function(x: torch.Tensor) -> torch.Tensor:
    """Activation that maps unbounded real to positive: 2 * sigmoid(x)^log10(e) + eps.

    Used for amplitudes and magnitudes in the synthesizer networks.
    """
    return 2.0 * torch.sigmoid(x) ** math.log10(math.e) + 1e-7


def remove_above_nyquist(frequencies: torch.Tensor,
                         amplitudes: torch.Tensor,
                         sample_rate: int) -> torch.Tensor:
    """Zero out harmonic amplitudes whose frequency exceeds Nyquist.

    Args:
        frequencies: (B, T, n_harmonics) — frequency of each partial in Hz.
        amplitudes: (B, T, n_harmonics) — amplitude of each partial.
        sample_rate: audio sample rate.

    Returns:
        amplitudes with above-Nyquist entries set to zero.
    """
    nyquist = sample_rate / 2.0
    return torch.where(frequencies > nyquist, torch.zeros_like(amplitudes), amplitudes)


def resample(inputs: torch.Tensor,
             n_frames: int,
             n_samples: int,
             method: str = "linear") -> torch.Tensor:
    """Resample from frame rate to sample rate.

    Args:
        inputs: (B, n_frames, D) tensor at frame rate.
        n_frames: number of input frames.
        n_samples: number of output samples.
        method: 'linear' for linear interpolation, 'nearest' for nearest.

    Returns:
        (B, n_samples, D) tensor at sample rate.
    """
    # Transpose to (B, D, n_frames) for interpolate, then back
    x = inputs.permute(0, 2, 1)  # (B, D, n_frames)
    mode = "linear" if method == "linear" else "nearest"
    # F.interpolate needs 3D input: (B, D, n_frames) -> (B, D, n_samples)
    x = F.interpolate(x, size=n_samples, mode=mode, align_corners=True if mode == "linear" else None)
    return x.permute(0, 2, 1)  # (B, n_samples, D)


def upsample_with_window(frames: torch.Tensor,
                         n_samples: int) -> torch.Tensor:
    """Upsample frame-rate envelope to sample rate using linear interpolation.

    Args:
        frames: (B, T_frames, D) at frame rate.
        n_samples: target number of audio samples.

    Returns:
        (B, n_samples, D) at sample rate.
    """
    return resample(frames, frames.shape[1], n_samples, method="linear")


def fft_convolve(signal: torch.Tensor,
                 impulse_response: torch.Tensor) -> torch.Tensor:
    """Convolve signal and impulse response using FFT (frequency domain).

    Args:
        signal: (B, T_signal) audio signal.
        impulse_response: (B, T_ir) or (B, T_signal) impulse response.
            If shorter, zero-padded to match signal length.

    Returns:
        (B, T_signal) convolved audio, same length as input signal.
    """
    n_signal = signal.shape[-1]
    n_ir = impulse_response.shape[-1]
    n_fft = n_signal + n_ir - 1

    # Optimise FFT size to power of 2 for speed
    n_fft = int(2 ** math.ceil(math.log2(n_fft)))

    signal_fft = torch.fft.rfft(signal, n=n_fft)
    ir_fft = torch.fft.rfft(impulse_response, n=n_fft)
    conv_fft = signal_fft * ir_fft
    conv = torch.fft.irfft(conv_fft, n=n_fft)

    # Trim to original signal length
    return conv[..., :n_signal]


def frequency_impulse_response(magnitudes: torch.Tensor,
                                window_size: int = 257) -> torch.Tensor:
    """Convert frequency magnitudes to windowed impulse responses.

    Args:
        magnitudes: (B, n_frames, n_bands) magnitude response per frame.
        window_size: FIR filter size (odd, default 257).

    Returns:
        (B, n_frames, window_size) impulse responses.
    """
    n_fft = window_size
    # Create linear-phase FIR from magnitude response
    # magnitudes: (B, n_frames, n_bands) -> (B, n_frames, n_fft//2 + 1)
    # Use torch.fft.irfft with hermitian symmetry
    ir = torch.fft.irfft(magnitudes, n=n_fft, dim=-1)

    # Apply Hann window for smooth frequency response
    window = torch.hann_window(window_size, device=magnitudes.device, dtype=magnitudes.dtype)
    return ir * window


def apply_window_to_impulse_response(impulse_responses: torch.Tensor) -> torch.Tensor:
    """Apply Hann window to impulse responses for causal filtering.

    Args:
        impulse_responses: (B, n_frames, window_size) unwindowed IRs.

    Returns:
        (B, n_frames, window_size) windowed IRs.
    """
    window_size = impulse_responses.shape[-1]
    window = torch.hann_window(window_size, device=impulse_responses.device,
                               dtype=impulse_responses.dtype)
    return impulse_responses * window


def frequency_filter(audio: torch.Tensor,
                     impulse_responses: torch.Tensor) -> torch.Tensor:
    """Apply time-varying FIR filter to audio using overlap-add.

    Args:
        audio: (B, n_samples) input audio.
        impulse_responses: (B, n_frames, window_size) FIR filter per frame.

    Returns:
        (B, n_samples) filtered audio.
    """
    n_samples = audio.shape[-1]
    n_frames = impulse_responses.shape[1]
    window_size = impulse_responses.shape[-1]
    hop_size = n_samples // n_frames

    # Pad audio for valid convolution
    pad_length = window_size // 2
    audio_padded = F.pad(audio, (pad_length, pad_length))

    # Frame-based filtering: for each frame, extract window, convolve, overlap-add
    output = torch.zeros_like(audio)
    window = torch.hann_window(window_size, device=audio.device, dtype=audio.dtype)

    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size

        # Extract frame
        frame = audio_padded[..., start:end]
        if frame.shape[-1] < window_size:
            frame = F.pad(frame, (0, window_size - frame.shape[-1]))

        # Convolve with impulse response
        ir = impulse_responses[:, i, :]
        filtered_frame = fft_convolve(frame, ir)

        # Apply window and overlap-add
        out_start = i * hop_size
        out_end = min(out_start + window_size, n_samples)
        frame_len = out_end - out_start
        output[..., out_start:out_end] += filtered_frame[..., :frame_len] * window[:frame_len]

    return output


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Log that clamps input to avoid log(0)."""
    return torch.log(torch.clamp(x, min=eps))


def angular_to_cumulative(angular_vel: torch.Tensor) -> torch.Tensor:
    """Convert angular velocity to cumulative phase via cumulative sum.

    Uses cumulative sum for stable phase tracking in additive synthesis.
    For inference stability, angular_cumsum is preferred (not implemented here
    as it requires special handling of numerical drift).

    Args:
        angular_vel: (B, T) angular velocity in radians/sample.

    Returns:
        (B, T) cumulative phase.
    """
    return torch.cumsum(angular_vel, dim=-1)