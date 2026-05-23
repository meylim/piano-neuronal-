import numpy as np
import soundfile as sf
import librosa
import torchaudio.transforms as T
import torch
from pathlib import Path
from typing import Optional


def load_audio(
    filepath: Path,
    target_sr: Optional[int] = None,
    mono: bool = False,
    align_onset: bool = True,
) -> tuple[np.ndarray, dict]:
    if target_sr is None:
        from piano_neuronal.config import SOURCE_SAMPLE_RATE
        target_sr = SOURCE_SAMPLE_RATE

    # soundfile returns shape (frames, channels) for stereo
    audio, sr = sf.read(str(filepath), dtype="float32")

    # Transpose to (channels, frames) for consistency with PyTorch convention
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # (1, frames) mono
    else:
        audio = audio.T  # (channels, frames) stereo

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        audio_tensor = torch.from_numpy(audio)
        audio_tensor = resampler(audio_tensor)
        audio = audio_tensor.numpy()
        sr = target_sr

    # Convert to mono if requested
    if mono and audio.shape[0] > 1:
        audio_mono = np.mean(audio, axis=0, keepdims=True)  # (1, frames)
    else:
        audio_mono = audio

    metadata = {"sample_rate": sr, "channels": audio.shape[0], "samples": audio.shape[1]}

    # Onset alignment
    onset_idx = 0
    if align_onset:
        # Use left channel (channel 0) for onset detection
        ref_channel = audio[0]
        onset_idx = _find_onset(ref_channel, sr)
        audio = audio[:, onset_idx:]
        audio_mono = audio_mono[:, onset_idx:] if mono else audio
        metadata["onset_sample_idx"] = onset_idx
        metadata["samples"] = audio.shape[1]

    if mono:
        result = audio_mono
    else:
        result = audio

    metadata["samples"] = result.shape[1]
    metadata["channels"] = result.shape[0]

    return result, metadata


def _find_onset(audio: np.ndarray, sr: int) -> int:
    onsets = librosa.onset.onset_detect(
        y=audio, sr=sr, backtrack=True, units="samples"
    )
    if len(onsets) > 0:
        first_onset = onsets[0]
        pre_buffer = int(0.001 * sr)  # 1 ms before onset
        return max(0, first_onset - pre_buffer)

    # Fallback: trim silence
    _, index = librosa.effects.trim(audio, top_db=30)
    return index[0] if len(index) > 0 else 0


if __name__ == "__main__":
    from piano_neuronal.config import SOURCE_SAMPLE_RATE, PIANO_IN_162_SAMPLES

    # Quick smoke test
    test_files = sorted(PIANO_IN_162_SAMPLES.rglob("*.flac"))
    if test_files:
        f = test_files[0]
        print(f"Loading: {f.name}")
        audio, meta = load_audio(f, target_sr=SOURCE_SAMPLE_RATE, mono=False)
        print(f"Shape: {audio.shape}, SR: {meta['sample_rate']}, Onset: {meta.get('onset_sample_idx', 'N/A')}")

        audio_mono, meta_mono = load_audio(f, target_sr=SOURCE_SAMPLE_RATE, mono=True)
        print(f"Mono shape: {audio_mono.shape}, SR: {meta_mono['sample_rate']}")