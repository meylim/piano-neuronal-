"""Create (MIDI, audio) pairs for polyphonic training.

Strategy:
1. Render each MAESTRO v3 MIDI file via sfizz + Piano in 162 SFZ.
2. Apply velocity augmentation (0.7x, 1.0x, 1.3x) — scaling MIDI velocities
   BEFORE rendering (not audio gain) to produce natural timbre variation.
3. Extract 30-second segments from longer pieces for additional pairs.
4. Target: >=5,000 pairs total.

Optimizations:
- Parallel rendering with N_WORKERS sfizz_render processes
- Sort MIDI files by size (shortest first) for faster early progress
- Clean up WAV files after processing
- Stop as soon as target_pairs is reached
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import h5py
import pretty_midi
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from piano_neuronal.config import (
    MAESTRO_V3_DIR,
    SFZ_CLOSE_PATH,
    MIDI_PAIRS_H5_PATH,
    OUTPUT_DIR,
    SOURCE_SAMPLE_RATE,
)
from piano_neuronal.s1_midi.midi_renderer import render_midi_to_audio, get_renderer_info


VELOCITY_SCALES = [0.7, 1.0, 1.3]
SEGMENT_DURATION_S = 30
MIN_SEGMENT_DURATION_S = 10
N_WORKERS = 8  # Each sfizz_render loads ~2-4 GB (full SFZ instrument)


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


def create_midi_pairs_dataset(target_pairs: int = 0) -> None:
    """Create the (MIDI, audio) pair dataset with parallel rendering."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = find_maestro_midi_files()
    if not midi_files:
        print("ERROR: No MAESTRO MIDI files found. Run download_maestro.py first.")
        return

    renderer_info = get_renderer_info()
    print(f"Renderer: {renderer_info['renderer']} v{renderer_info['version']}")

    render_dir = OUTPUT_DIR / "rendered_audio"
    render_dir.mkdir(exist_ok=True)

    # Build work items: (midi_path, vel_scale, render_dir)
    # Render all MIDI files at all 3 velocity scales for full augmentation
    work_items = []
    for midi_path in midi_files:
        for vel_scale in VELOCITY_SCALES:
            work_items.append((midi_path, vel_scale, str(render_dir)))

    print(f"Render jobs: {len(work_items)} ({len(midi_files)} files x {len(VELOCITY_SCALES)} velocities)")
    print(f"Rendering with {N_WORKERS} workers...")

    # Phase 1: Render all MIDI files in parallel
    render_results = []
    with Pool(N_WORKERS) as pool:
        with tqdm(total=len(work_items), desc="Rendering MIDI") as pbar:
            for result in pool.imap_unordered(_render_one, work_items, chunksize=1):
                render_results.append(result)
                pbar.update(1)

    successful_renders = [r for r in render_results if r["success"]]
    failed_renders = [r for r in render_results if not r["success"]]
    print(f"\nRendered: {len(successful_renders)}, Failed: {len(failed_renders)}")

    if failed_renders:
        print("Sample errors:")
        for r in failed_renders[:5]:
            print(f"  {r['midi_path'].name} vel={r['vel_scale']}: {r['error']}")

    # Phase 2: Load rendered audio and write to HDF5
    print(f"\nPhase 2: Creating pairs from {len(successful_renders)} rendered files...")

    pair_count = 0
    errors = []

    with h5py.File(MIDI_PAIRS_H5_PATH, "w") as hf:
        hf.attrs["renderer"] = renderer_info["renderer"]
        hf.attrs["renderer_version"] = renderer_info["version"]
        hf.attrs["sample_rate"] = renderer_info["sample_rate"]
        hf.attrs["velocity_scales"] = VELOCITY_SCALES
        hf.attrs["target_pairs"] = target_pairs

        for result in tqdm(successful_renders, desc="Writing pairs"):
            if target_pairs > 0 and pair_count >= target_pairs:
                break

            midi_path = result["midi_path"]
            vel_scale = result["vel_scale"]
            wav_path = result["wav_path"]

            try:
                # Load rendered audio
                audio, sr = sf.read(str(wav_path), dtype="float32")

                # Load original MIDI for events
                pm = pretty_midi.PrettyMIDI(str(midi_path))
                midi_events = _serialize_midi_events(pm)

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

                    pair_count += 1

                # Clean up WAV after processing
                if wav_path.exists():
                    wav_path.unlink()

            except Exception as e:
                errors.append((midi_path.name, vel_scale, str(e)))
                continue

        # Flush before closing
        hf.flush()

    print(f"\nCreated {pair_count} MIDI-audio pairs -> {MIDI_PAIRS_H5_PATH}")
    if errors:
        print(f"{len(errors)} errors during pair creation:")
        for fname, vel, err in errors[:5]:
            print(f"  {fname} (vel={vel}): {err}")

    # Clean up render directory if empty
    try:
        render_dir.rmdir()
    except OSError:
        pass


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