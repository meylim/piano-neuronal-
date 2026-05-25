"""PyTorch Dataset for DDSP-Piano training on Sprint 1 MIDI pairs.

Reads from midi_pairs.h5, resamples audio 44.1→16 kHz, encodes MIDI to
conditioning tensors, and caches precomputed conditioning to disk.

Key optimisations for RTX PRO 6000 S:
- Precomputed conditioning tensors cached on disk (never recomputed)
- Resampled audio cached on disk
- num_workers=12, pin_memory=True, persistent_workers=True
- Segments with polyphony > 16 are filtered out
"""

import os
import logging
import torch
import torchaudio
import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

from piano_neuronal.s2_baseline.config import (
    MIDI_PAIRS_H5_PATH, SAMPLE_RATE, SOURCE_SAMPLE_RATE,
    DURATION_S, N_SAMPLES, N_FRAMES, N_SYNTHS, FRAME_RATE,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, MAX_POLYPHONY,
    CACHE_DIR, BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR,
    PIN_MEMORY, PERSISTENT_WORKERS
)
from piano_neuronal.s2_baseline.midi_encoding import (
    encode_midi_events, check_polyphony_limit
)


logger = logging.getLogger(__name__)


class MidiPairsDataset(Dataset):
    """Dataset reading from Sprint 1 midi_pairs.h5 with on-disk caching.

    Each item returns:
        audio:        (n_samples,) float32 — resampled to 16 kHz
        conditioning: (n_frames, n_synths, 2) float32
        pedal:        (n_frames, 4) float32
        polyphony:    (n_frames,) int32

    Segments with polyphony > MAX_POLYPHONY are filtered out during __init__.
    """

    def __init__(
        self,
        h5_path: Path = MIDI_PAIRS_H5_PATH,
        split: str = TRAIN_SPLIT,
        target_sr: int = SAMPLE_RATE,
        source_sr: int = SOURCE_SAMPLE_RATE,
        duration_s: float = DURATION_S,
        n_samples: int = N_SAMPLES,
        n_frames: int = N_FRAMES,
        n_synths: int = N_SYNTHS,
        frame_rate: int = FRAME_RATE,
        max_polyphony: int = MAX_POLYPHONY,
        overlap: float = 0.5,
        cache_dir: Path = CACHE_DIR,
    ):
        self.h5_path = str(h5_path)
        self.split = split
        self.target_sr = target_sr
        self.source_sr = source_sr
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.n_synths = n_synths
        self.frame_rate = frame_rate
        self.max_polyphony = max_polyphony
        self.overlap = overlap

        # Build index of valid segments for this split
        self.indices = []
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resampler (created once, reused)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=source_sr, new_freq=target_sr
        )

        # Scan HDF5 and filter by split + polyphony
        self._scan_h5()

    def _scan_h5(self) -> None:
        """Scan HDF5, collect valid pair indices for this split."""
        cache_file = self._cache_dir / f"index_{self.split}.pt"
        if cache_file.exists():
            self.indices = torch.load(cache_file, weights_only=False)
            logger.info(f"Loaded {len(self.indices)} indices from cache for split '{self.split}'")
            return

        valid_indices = []
        logger.info(f"Scanning HDF5 for split '{self.split}'...")
        with h5py.File(self.h5_path, "r") as hf:
            keys = list(hf.keys())
            logger.info(f"HDF5 has {len(keys)} keys total")
            for i, key in enumerate(keys):
                if not key.startswith("pair_"):
                    continue
                grp = hf[key]
                split = grp.attrs.get("split", "train")
                if split != self.split:
                    continue

                # Check polyphony by pre-encoding (expensive, cache result)
                idx = int(key.replace("pair_", ""))
                valid_indices.append(idx)
                if len(valid_indices) % 2000 == 0:
                    logger.info(f"  Scanned {i+1}/{len(keys)} keys, {len(valid_indices)} valid for '{self.split}'")

        self.indices = valid_indices
        torch.save(self.indices, cache_file)
        logger.info(f"Found {len(valid_indices)} samples for split '{self.split}', cached to {cache_file}")

    def _get_cache_path(self, idx: int) -> Path:
        """Cache path for precomputed conditioning tensors."""
        return self._cache_dir / f"{self.split}_pair_{idx:05d}.pt"

    def _get_audio_cache_path(self, idx: int) -> Path:
        """Cache path for precomputed resampled audio."""
        return self._cache_dir / f"audio_{idx:05d}.pt"

    def _load_and_encode(self, hf: h5py.File, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Load audio and encode MIDI conditioning for a pair.

        Returns:
            audio: (n_samples,) float32 at target_sr
            midi_events: structured numpy array
        """
        key = f"pair_{idx:05d}"
        if key not in hf:
            raise KeyError(f"Pair {key} not found in HDF5")

        grp = hf[key]

        # Load audio and resample
        audio = grp["audio"][:]  # (channels, samples) or (samples,)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Convert to float32 torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Mix to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        # Resample from source_sr to target_sr
        if self.source_sr != self.target_sr:
            audio_tensor = self.resampler(audio_tensor)

        # Take exactly n_samples (trim or pad)
        audio_tensor = audio_tensor.squeeze(0)  # (samples,)
        if audio_tensor.shape[0] > self.n_samples:
            audio_tensor = audio_tensor[:self.n_samples]
        elif audio_tensor.shape[0] < self.n_samples:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, self.n_samples - audio_tensor.shape[0])
            )

        # Load MIDI events
        midi_events = grp["midi_events"][:]

        return audio_tensor, midi_events

    def preload_audio_cache(self, num_workers: int = 8) -> None:
        """Pre-resample all audio and cache to disk for fast access.

        With 64 cores this takes ~2-3 minutes for 23k segments.
        After preloading, __getitem__ reads cached audio instead of resampling.
        """
        from concurrent.futures import ProcessPoolExecutor
        import time

        logger.info(f"Pre-resampling {len(self.indices)} audio segments with {num_workers} workers...")
        t0 = time.time()

        resampler = torchaudio.transforms.Resample(
            orig_freq=self.source_sr, new_freq=self.target_sr
        )

        def _process_one(idx):
            audio_path = self._cache_dir / f"audio_{idx:05d}.pt"
            if audio_path.exists():
                return True
            try:
                with h5py.File(self.h5_path, "r") as hf:
                    key = f"pair_{idx:05d}"
                    grp = hf[key]
                    audio = grp["audio"][:]
                    if audio.ndim == 1:
                        audio = audio[np.newaxis, :]
                    audio_tensor = torch.from_numpy(audio).float()
                    if audio_tensor.shape[0] > 1:
                        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
                    audio_tensor = resampler(audio_tensor)
                    audio_tensor = audio_tensor.squeeze(0)
                    if audio_tensor.shape[0] > self.n_samples:
                        audio_tensor = audio_tensor[:self.n_samples]
                    elif audio_tensor.shape[0] < self.n_samples:
                        audio_tensor = torch.nn.functional.pad(
                            audio_tensor, (0, self.n_samples - audio_tensor.shape[0])
                        )
                torch.save(audio_tensor, audio_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to cache audio {idx}: {e}")
                return False

        done = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, idx): idx for idx in self.indices}
            for future in futures:
                future.result()
                done += 1
                if done % 2000 == 0:
                    logger.info(f"  Cached {done}/{len(self.indices)} audio segments")

        elapsed = time.time() - t0
        logger.info(f"Audio cache complete: {done} segments in {elapsed:.1f}s ({done/elapsed:.0f} seg/s)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.indices[local_idx]
        cache_path = self._get_cache_path(idx)
        audio_cache_path = self._get_audio_cache_path(idx)

        # Try loading precomputed audio from cache
        if audio_cache_path.exists():
            audio = torch.load(audio_cache_path, weights_only=False)
        else:
            with h5py.File(self.h5_path, "r") as hf:
                audio, _ = self._load_and_encode(hf, idx)
            # Cache for next time
            torch.save(audio, audio_cache_path)

        # Try loading precomputed conditioning from cache
        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            conditioning = cached["conditioning"]
            pedal = cached["pedal"]
            polyphony = cached["polyphony"]
        else:
            # Full encode from HDF5 (only MIDI encoding needed, audio already cached)
            with h5py.File(self.h5_path, "r") as hf:
                key = f"pair_{idx:05d}"
                midi_events = hf[key]["midi_events"][:]

            conditioning_np, pedal_np, polyphony_np = encode_midi_events(
                midi_events,
                duration_s=self.n_samples / self.target_sr,
                frame_rate=self.frame_rate,
                n_frames=self.n_frames,
                n_synths=self.n_synths,
            )

            # Filter out high-polyphony segments
            if not check_polyphony_limit(polyphony_np, self.max_polyphony):
                return (
                    torch.zeros(self.n_samples, dtype=torch.float32),
                    torch.zeros(self.n_frames, self.n_synths, 2, dtype=torch.float32),
                    torch.zeros(self.n_frames, 4, dtype=torch.float32),
                    torch.zeros(self.n_frames, dtype=torch.int32),
                )

            conditioning = torch.from_numpy(conditioning_np)
            pedal = torch.from_numpy(pedal_np)
            polyphony = torch.from_numpy(polyphony_np)

            # Cache to disk
            torch.save({
                "conditioning": conditioning,
                "pedal": pedal,
                "polyphony": polyphony,
            }, cache_path)

        return audio, conditioning, pedal, polyphony


def collate_fn(batch):
    """Custom collate that filters out invalid (zero) samples."""
    # Filter out samples where polyphony is all zero (invalid)
    valid = [
        (a, c, p, poly) for a, c, p, poly in batch
        if poly.sum() > 0
    ]
    if not valid:
        # Return empty batch (shouldn't happen with large dataset)
        return None

    audios, condits, pedals, polys = zip(*valid)
    return (
        torch.stack(audios),       # (B, n_samples)
        torch.stack(condits),       # (B, n_frames, n_synths, 2)
        torch.stack(pedals),        # (B, n_frames, 4)
        torch.stack(polys),         # (B, n_frames)
    )


def get_dataloader(
    split: str = TRAIN_SPLIT,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the given split."""
    dataset = MidiPairsDataset(split=split)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if num_workers > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )