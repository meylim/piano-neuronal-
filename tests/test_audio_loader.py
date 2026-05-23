import pytest
import numpy as np
from pathlib import Path
from piano_neuronal.config import PIANO_IN_162_SAMPLES, SOURCE_SAMPLE_RATE


# Skip all tests if dataset is not available
pytestmark = pytest.mark.skipif(
    not PIANO_IN_162_SAMPLES.exists(),
    reason="Piano in 162 dataset not found"
)


def _get_first_flac():
    flacs = sorted(PIANO_IN_162_SAMPLES.rglob("*.flac"))
    return flacs[0] if flacs else None


class TestAudioLoader:
    def test_load_stereo(self):
        from piano_neuronal.s1_data.audio_loader import load_audio
        fpath = _get_first_flac()
        if fpath is None:
            pytest.skip("No FLAC files found")
        audio, meta = load_audio(fpath, target_sr=SOURCE_SAMPLE_RATE, mono=False)
        assert audio.ndim == 2
        assert audio.shape[0] == 2  # stereo
        assert meta["sample_rate"] == SOURCE_SAMPLE_RATE
        assert meta["channels"] == 2

    def test_load_mono(self):
        from piano_neuronal.s1_data.audio_loader import load_audio
        fpath = _get_first_flac()
        if fpath is None:
            pytest.skip("No FLAC files found")
        audio, meta = load_audio(fpath, target_sr=SOURCE_SAMPLE_RATE, mono=True)
        assert audio.ndim == 2
        assert audio.shape[0] == 1  # mono
        assert meta["channels"] == 1

    def test_resample_48k(self):
        from piano_neuronal.s1_data.audio_loader import load_audio
        fpath = _get_first_flac()
        if fpath is None:
            pytest.skip("No FLAC files found")
        audio, meta = load_audio(fpath, target_sr=48000, mono=True)
        assert meta["sample_rate"] == 48000

    def test_onset_alignment(self):
        from piano_neuronal.s1_data.audio_loader import load_audio
        fpath = _get_first_flac()
        if fpath is None:
            pytest.skip("No FLAC files found")
        audio, meta = load_audio(fpath, target_sr=SOURCE_SAMPLE_RATE, mono=True, align_onset=True)
        assert "onset_sample_idx" in meta
        assert meta["onset_sample_idx"] >= 0
        # After alignment, the signal should start with the attack
        # First 5ms should have non-trivial energy (the 1ms pre-buffer may be near-silence)
        first_5ms_samples = int(0.005 * SOURCE_SAMPLE_RATE)
        rms_first_5ms = np.sqrt(np.mean(audio[0, :first_5ms_samples] ** 2))
        total_rms = np.sqrt(np.mean(audio[0] ** 2))
        # First 5ms energy should be at least 0.5% of total
        assert rms_first_5ms > total_rms * 0.005, \
            f"Onset alignment may have trimmed too aggressively: first 5ms RMS={rms_first_5ms:.6f}, total RMS={total_rms:.6f}"

    def test_no_onset_alignment(self):
        from piano_neuronal.s1_data.audio_loader import load_audio
        fpath = _get_first_flac()
        if fpath is None:
            pytest.skip("No FLAC files found")
        audio, meta = load_audio(fpath, target_sr=SOURCE_SAMPLE_RATE, mono=True, align_onset=False)
        assert "onset_sample_idx" not in meta