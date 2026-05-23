"""Create (MIDI, audio) pairs for polyphonic training.

Strategy:
1. Render each MAESTRO v3 MIDI file via sfizz + Piano in 162 SFZ.
2. Apply velocity augmentation (0.7x, 1.0x, 1.3x) — scaling MIDI velocities
   BEFORE rendering (not audio gain) to produce natural timbre variation.
3. Extract 30-second segments from longer pieces for additional pairs.
4. Target: ≥5,000 pairs total.
"""

import numpy as np
import h5py
import pretty_midi
import soundfile as sf
from pathlib import Path
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


def find_maestro_midi_files() -> list[Path]:
    """Find all MAESTRO v3 MIDI files."""
    midi_dir = MAESTRO_V3_DIR / "maestro-v3.0.0"
    if not midi_dir.exists():
        # Try top-level
        midi_dir = MAESTRO_V3_DIR

    midi_files = sorted(midi_dir.rglob("*.midi"))
    if not midi_files:
        midi_files = sorted(midi_dir.rglob("*.mid"))

    print(f"Found {len(midi_files)} MAESTRO MIDI files in {midi_dir}")
    return midi_files


def create_midi_pairs_dataset(target_pairs: int = 5000) -> None:
    """Create the (MIDI, audio) pair dataset.

    For each MAESTRO MIDI file:
    - Render with velocity scales {0.7, 1.0, 1.3}
    - Extract 30-second segments from longer pieces
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = find_maestro_midi_files()
    if not midi_files:
        print("ERROR: No MAESTRO MIDI files found. Run download_maestro.py first.")
        return

    renderer_info = get_renderer_info()
    print(f"Renderer: {renderer_info['renderer']} v{renderer_info['version']}")

    # Temporary output directory for rendered audio
    render_dir = OUTPUT_DIR / "rendered_audio"
    render_dir.mkdir(exist_ok=True)

    pair_count = 0
    pairs_data = []

    with h5py.File(MIDI_PAIRS_H5_PATH, "w") as hf:
        hf.attrs["renderer"] = renderer_info["renderer"]
        hf.attrs["renderer_version"] = renderer_info["version"]
        hf.attrs["sample_rate"] = renderer_info["sample_rate"]
        hf.attrs["velocity_scales"] = VELOCITY_SCALES

        for midi_path in tqdm(midi_files, desc="Rendering MAESTRO files"):
            for vel_scale in VELOCITY_SCALES:
                if pair_count >= target_pairs:
                    break

                try:
                    # Render MIDI → audio via sfizz
                    output_wav = render_dir / f"{midi_path.stem}_vel{vel_scale:.1f}.wav"
                    render_midi_to_audio(
                        midi_path=midi_path,
                        sfz_path=SFZ_CLOSE_PATH,
                        output_path=output_wav,
                        velocity_scale=vel_scale,
                    )

                    # Load rendered audio
                    audio, sr = sf.read(str(output_wav), dtype="float32")

                    # Load original MIDI for events
                    pm = pretty_midi.PrettyMIDI(str(midi_path))
                    midi_events = _serialize_midi_events(pm)

                    # If audio is long enough, extract segments
                    duration_s = len(audio) / sr
                    if duration_s > SEGMENT_DURATION_S * 1.5:
                        # Extract 30-second segments
                        n_segments = max(1, int(duration_s / SEGMENT_DURATION_S))
                        for seg_idx in range(n_segments):
                            if pair_count >= target_pairs:
                                break
                            start_sample = int(seg_idx * SEGMENT_DURATION_S * sr)
                            end_sample = min(start_sample + int(SEGMENT_DURATION_S * sr), len(audio))

                            segment = audio[start_sample:end_sample]
                            if len(segment) < MIN_SEGMENT_DURATION_S * sr:
                                continue

                            pair_name = f"pair_{pair_count:05d}"
                            grp = hf.create_group(pair_name)
                            grp.create_dataset("audio", data=segment.T, compression="gzip")  # (2, N)
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
                        grp.create_dataset("audio", data=audio.T, compression="gzip")
                        grp.create_dataset("midi_events", data=midi_events)
                        grp.attrs["source_file"] = midi_path.stem
                        grp.attrs["velocity_scale"] = vel_scale
                        grp.attrs["segment_idx"] = 0
                        grp.attrs["segment_start_s"] = 0.0
                        grp.attrs["duration_s"] = len(audio) / sr

                        pair_count += 1

                except Exception as e:
                    print(f"Error rendering {midi_path.name} vel_scale={vel_scale}: {e}")
                    continue

            if pair_count >= target_pairs:
                break

    print(f"\nCreated {pair_count} MIDI-audio pairs → {MIDI_PAIRS_H5_PATH}")


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
    create_midi_pairs_dataset(target_pairs=5000)