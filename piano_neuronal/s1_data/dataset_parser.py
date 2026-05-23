import re
from pathlib import Path
from typing import Optional

from piano_neuronal.config import (
    PIANO_IN_162_SAMPLES,
    VELOCITY_LAYERS,
    MIDI_NOTE_MIN,
)

_FILENAME_RE = re.compile(
    r"(\d+)-"                                    # note_index (01-88)
    r"(PedalOn|PedalOff)"                        # pedal state
    r"(Pianissimo|Piano|MezzoPiano|MezzoForte|Forte)"  # velocity layer
    r"(\d)"                                       # round-robin (1 or 2)
    r"(Close|Ambient)"                            # mic position
)

NOTE_INDEX_TO_MIDI = lambda idx: idx + 20  # note_index 01 → MIDI 21 (A0)


def parse_filename(filepath: Path) -> dict:
    filename = filepath.stem

    match = _FILENAME_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    note_index = int(match.group(1))
    pedal = "On" if match.group(2) == "PedalOn" else "Off"
    velocity_name = match.group(3)
    round_robin = int(match.group(4))
    mic = match.group(5)

    midi_note = NOTE_INDEX_TO_MIDI(note_index)
    vel_info = VELOCITY_LAYERS[velocity_name]

    path_str = str(filepath).replace("\\", "/")
    if "PedalOn" not in path_str and pedal == "On":
        pass  # Already extracted from filename

    return {
        "note_index": note_index,
        "midi_note": midi_note,
        "note_name": _midi_to_note_name(midi_note),
        "pedal": pedal,
        "velocity_layer": velocity_name,
        "velocity_lovel": vel_info["lovel"],
        "velocity_hivel": vel_info["hivel"],
        "velocity_midi_center": vel_info["center"],
        "velocity_continuous": vel_info["continuous"],
        "round_robin": round_robin,
        "mic": mic,
        "file_path": str(filepath),
    }


def _midi_to_note_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    name = names[midi_note % 12]
    return f"{name}{octave}"


def discover_all_files() -> list[dict]:
    root = PIANO_IN_162_SAMPLES
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Ensure the Piano in 162 dataset is extracted to the configured path."
        )

    results = []
    flac_files = sorted(root.rglob("*.flac"))
    parse_errors = []

    for fpath in flac_files:
        try:
            metadata = parse_filename(fpath)
            results.append(metadata)
        except ValueError as e:
            parse_errors.append((str(fpath), str(e)))

    if parse_errors:
        print(f"WARNING: {len(parse_errors)} files could not be parsed:")
        for fp, err in parse_errors[:10]:
            print(f"  {fp}: {err}")
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more")

    expected = 88 * 5 * 2 * 2 * 2  # notes × velocities × RRs × mics × pedals
    print(f"Parsed {len(results)}/{expected} files successfully")
    if len(results) != expected:
        print(f"WARNING: Expected {expected} files, got {len(results)}")

    return results


if __name__ == "__main__":
    files = discover_all_files()
    if files:
        print(f"\nFirst file: {files[0]}")
        print(f"Last file:  {files[-1]}")
        # Verify all note indices
        notes = sorted(set(f["midi_note"] for f in files))
        print(f"Note range: MIDI {notes[0]}–{notes[-1]} ({len(notes)} unique notes)")