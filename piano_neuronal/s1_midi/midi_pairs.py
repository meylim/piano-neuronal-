"""Create (MIDI, audio) pairs for polyphonic training.

Strategy:
1. Render each MAESTRO v3 MIDI file via sfizz + Piano in 162 SFZ.
2. Apply velocity augmentation (0.7x, 1.0x, 1.3x) — scaling MIDI velocities
   BEFORE rendering (not audio gain) to produce natural timbre variation.
3. Extract 30-second segments from longer pieces for additional pairs.
4. Target: all available pairs (set target_pairs=0 for unlimited).

Memory-safe design:
- Renders in batches of BATCH_SIZE and writes to HDF5 immediately,
  then deletes WAVs before the next batch. This prevents all 3800+ WAVs
  from existing on disk simultaneously (~270 GB).
- maxtasksperchild forces sfizz worker restarts to prevent memory leaks.
- explicit del + gc.collect() after each batch.
"""

import gc
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import h5py
import pretty_midi
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from piano_neuronal.config import (
    MAESTRO_V3_DIR,
    SFZ_CLOSE_PATH,
    MIDI_PAIRS_H5_PATH,
    OUTPUT_DIR,
    SOURCE_SAMPLE_RATE,
)
from piano_neuronal.s1_midi.midi_renderer import render_midi_to_audio, get_renderer_info


VELOCITY_SCALES = [1.0]  # No augmentation — velocity scaling creates false signal (clipping at 127, compression at 0.7). The Neural Hammer (Strate 1) handles continuous velocity.
SEGMENT_DURATION_S = 30
MIN_SEGMENT_DURATION_S = 10
N_WORKERS = 8  # Each sfizz_render loads ~2-4 GB (full SFZ instrument)
BATCH_SIZE = 50  # Render batches — controls peak disk usage


def _load_maestro_splits() -> dict[str, str]:
    """Load official MAESTRO v3 train/val/test splits from the CSV.

    Uses the canonical split from maestro-v3.0.0.csv to prevent data leakage:
    all segments and velocity augmentations of the same piece go into
    the same split. This is the standard split every reviewer expects.
    """
    import pandas as pd
    csv_path = MAESTRO_V3_DIR / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        csv_path = MAESTRO_V3_DIR / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        # Fallback: try top-level
        csv_path = MAESTRO_V3_DIR / "maestro-v3.0.0.csv"

    df = pd.read_csv(csv_path)
    # Map midi_filename (without path) -> split
    splits = {}
    for _, row in df.iterrows():
        # Use stem (filename without extension) as key
        stem = Path(row["midi_filename"]).stem
        splits[stem] = row["split"]  # "train", "test", or "validation"
    return splits


def find_maestro_midi_files() -> list[Path]:
    """Find all MAESTRO v3 MIDI files, sorted by size (shortest first)."""
    midi_dir = MAESTRO_V3_DIR / "maestro-v3.0.0"
    if not midi_dir.exists():
        midi_dir = MAESTRO_V3_DIR

    midi_files = sorted(midi_dir.rglob("*.midi"))
    if not midi_files:
        midi_files = sorted(midi_dir.rglob("*.mid"))

    # Sort by file size (shorter files render faster)
    midi_files.sort(key=lambda p: p.stat().st_size)

    print(f"Found {len(midi_files)} MAESTRO MIDI files in {midi_dir}")
    return midi_files


def _render_one(args: tuple) -> dict:
    """Worker: render a single (MIDI, velocity) pair via sfizz_render."""
    midi_path, vel_scale, render_dir = args
    output_wav = Path(render_dir) / f"{midi_path.stem}_vel{vel_scale:.1f}.wav"

    try:
        render_midi_to_audio(
            midi_path=midi_path,
            sfz_path=SFZ_CLOSE_PATH,
            output_path=output_wav,
            velocity_scale=vel_scale,
        )
        return {"success": True, "midi_path": midi_path, "vel_scale": vel_scale, "wav_path": output_wav}
    except Exception as e:
        return {"success": False, "midi_path": midi_path, "vel_scale": vel_scale, "error": str(e)}


def _process_render_result(result: dict, hf: h5py.File, pair_count: int,
                           target_pairs: int, split: str = "train") -> tuple[int, list]:
    """Load a rendered WAV, write pairs to HDF5, return updated pair_count and errors."""
    errors = []
    midi_path = result["midi_path"]
    vel_scale = result["vel_scale"]
    wav_path = result["wav_path"]

    try:
        # Load rendered audio
        audio, sr = sf.read(str(wav_path), dtype="float32")

        # Load original MIDI for events
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        midi_events = _serialize_midi_events(pm)
        del pm  # Free PrettyMIDI memory
        gc.collect()

        duration_s = audio.shape[0] / sr if audio.ndim == 1 else audio.shape[0] / sr

        if duration_s > SEGMENT_DURATION_S * 1.5:
            # Extract 30-second segments
            n_segments = max(1, int(duration_s / SEGMENT_DURATION_S))
            for seg_idx in range(n_segments):
                if target_pairs > 0 and pair_count >= target_pairs:
                    break
                start_sample = int(seg_idx * SEGMENT_DURATION_S * sr)
                end_sample = min(start_sample + int(SEGMENT_DURATION_S * sr), audio.shape[0])

                if audio.ndim == 1:
                    segment = audio[start_sample:end_sample]
                else:
                    segment = audio[start_sample:end_sample, :]

                if len(segment) < MIN_SEGMENT_DURATION_S * sr:
                    continue

                pair_name = f"pair_{pair_count:05d}"
                grp = hf.create_group(pair_name)
                if segment.ndim == 1:
                    grp.create_dataset("audio", data=segment[np.newaxis, :], compression="lzf")
                else:
                    grp.create_dataset("audio", data=segment.T, compression="lzf")
                grp.create_dataset("midi_events", data=midi_events)
                grp.attrs["source_file"] = midi_path.stem
                grp.attrs["velocity_scale"] = vel_scale
                grp.attrs["segment_idx"] = seg_idx
                grp.attrs["segment_start_s"] = start_sample / sr
                grp.attrs["duration_s"] = len(segment) / sr
                grp.attrs["split"] = split

                pair_count += 1
        else:
            # Use entire piece
            pair_name = f"pair_{pair_count:05d}"
            grp = hf.create_group(pair_name)
            if audio.ndim == 1:
                grp.create_dataset("audio", data=audio[np.newaxis, :], compression="lzf")
            else:
                grp.create_dataset("audio", data=audio.T, compression="lzf")
            grp.create_dataset("midi_events", data=midi_events)
            grp.attrs["source_file"] = midi_path.stem
            grp.attrs["velocity_scale"] = vel_scale
            grp.attrs["segment_idx"] = 0
            grp.attrs["segment_start_s"] = 0.0
            grp.attrs["duration_s"] = duration_s
            grp.attrs["split"] = split

            pair_count += 1

        # Free audio memory immediately
        del audio
        gc.collect()

    except Exception as e:
        errors.append((midi_path.name, vel_scale, str(e)))

    # Delete WAV after processing (saves disk space)
    if wav_path.exists():
        wav_path.unlink()

    return pair_count, errors


def create_midi_pairs_dataset(target_pairs: int = 0) -> None:
    """Create the (MIDI, audio) pair dataset with batched rendering.

    Renders in batches of BATCH_SIZE to control peak disk usage:
    instead of rendering all 3800+ WAVs first (would need ~270 GB),
    we render a batch, write to HDF5, delete WAVs, then proceed.

    Supports resume: if MIDI_PAIRS_H5_PATH exists, counts existing pairs
    and skips already-rendered MIDI files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = find_maestro_midi_files()
    if not midi_files:
        print("ERROR: No MAESTRO MIDI files found. Run download_maestro.py first.")
        return

    renderer_info = get_renderer_info()
    print(f"Renderer: {renderer_info['renderer']} v{renderer_info['version']}")

    # Load official MAESTRO splits (train/validation/test by piece)
    maestro_splits = _load_maestro_splits()
    n_covered = sum(1 for f in midi_files if f.stem in maestro_splits)
    print(f"MAESTRO splits: {n_covered}/{len(midi_files)} files have official split assignments")

    render_dir = OUTPUT_DIR / "rendered_audio"
    render_dir.mkdir(exist_ok=True)

    # Resume: check existing pairs
    existing_pairs = 0
    already_rendered = set()  # Track (midi_stem, vel_scale) already processed
    if MIDI_PAIRS_H5_PATH.exists():
        with h5py.File(MIDI_PAIRS_H5_PATH, "r") as hf:
            existing_pairs = sum(1 for k in hf.keys() if k.startswith("pair_"))
            for k in hf.keys():
                if k.startswith("pair_"):
                    stem = hf[k].attrs.get("source_file", "")
                    vel = hf[k].attrs.get("velocity_scale", 1.0)
                    already_rendered.add((stem, vel))
        print(f"Resuming: {existing_pairs} pairs already exist, {len(already_rendered)} unique renders done")

    # Build work items: (midi_path, vel_scale, render_dir), skip already done
    work_items = []
    skipped = 0
    for midi_path in midi_files:
        for vel_scale in VELOCITY_SCALES:
            if (midi_path.stem, vel_scale) in already_rendered:
                skipped += 1
                continue
            work_items.append((midi_path, vel_scale, str(render_dir)))

    total_jobs = len(work_items)
    print(f"Render jobs: {total_jobs} ({len(midi_files)} files x {len(VELOCITY_SCALES)} velocities, {skipped} skipped)")
    print(f"Processing in batches of {BATCH_SIZE} with {N_WORKERS} workers")

    pair_count = existing_pairs
    total_errors = []

    h5_mode = "a" if existing_pairs > 0 else "w"
    with h5py.File(MIDI_PAIRS_H5_PATH, h5_mode) as hf:
        hf.attrs["renderer"] = renderer_info["renderer"]
        hf.attrs["renderer_version"] = renderer_info["version"]
        hf.attrs["sample_rate"] = renderer_info["sample_rate"]
        hf.attrs["velocity_scales"] = VELOCITY_SCALES
        hf.attrs["target_pairs"] = target_pairs

        # Process in batches to limit peak disk usage
        with Pool(N_WORKERS, maxtasksperchild=20) as pool:
            for batch_start in range(0, total_jobs, BATCH_SIZE):
                batch = work_items[batch_start:batch_start + BATCH_SIZE]
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (total_jobs + BATCH_SIZE - 1) // BATCH_SIZE

                # Render batch
                batch_results = list(tqdm(
                    pool.imap_unordered(_render_one, batch),
                    total=len(batch),
                    desc=f"Rendering batch {batch_num}/{total_batches}",
                    leave=False,
                ))

                # Write batch results to HDF5 and delete WAVs
                for result in batch_results:
                    if not result["success"]:
                        total_errors.append((result.get("midi_path", Path("?")).name,
                                             result["vel_scale"], result["error"]))
                        # Try to clean up failed WAV if it exists
                        if result.get("wav_path") and Path(result["wav_path"]).exists():
                            Path(result["wav_path"]).unlink()
                        continue

                    if target_pairs > 0 and pair_count >= target_pairs:
                        # Clean up remaining WAVs for this result
                        wav_path = result["wav_path"]
                        if wav_path.exists():
                            wav_path.unlink()
                        continue

                    # Assign split from MAESTRO official split (by piece)
                    midi_stem = result["midi_path"].stem
                    split = maestro_splits.get(midi_stem, "train")
                    # Normalize "validation" -> "val"
                    if split == "validation":
                        split = "val"

                    pair_count, errors = _process_render_result(
                        result, hf, pair_count, target_pairs, split=split
                    )
                    total_errors.extend(errors)

                # Flush HDF5 and free memory
                hf.flush()
                del batch_results
                gc.collect()

                print(f"  Progress: {pair_count} pairs written, "
                      f"{batch_start + len(batch)}/{total_jobs} renders done, "
                      f"{len(total_errors)} errors")

                if target_pairs > 0 and pair_count >= target_pairs:
                    print(f"  Target of {target_pairs} pairs reached, stopping.")
                    break

    # Clean up render directory if empty
    try:
        render_dir.rmdir()
    except OSError:
        pass

    print(f"\nCreated {pair_count} MIDI-audio pairs -> {MIDI_PAIRS_H5_PATH}")
    if total_errors:
        print(f"{len(total_errors)} errors:")
        for fname, vel, err in total_errors[:10]:
            print(f"  {fname} (vel={vel}): {err}")
        if len(total_errors) > 10:
            print(f"  ... and {len(total_errors) - 10} more")


def _serialize_midi_events(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """Serialize MIDI events as a structured numpy array for storage."""
    events = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            events.append((
                note.start, note.end, note.pitch, note.velocity, int(instrument.program)
            ))
    if not events:
        return np.array([], dtype="float32")
    return np.array(events, dtype=[
        ("start", "f4"), ("end", "f4"), ("pitch", "i4"),
        ("velocity", "i4"), ("program", "i4")
    ])


if __name__ == "__main__":
    create_midi_pairs_dataset(target_pairs=0)