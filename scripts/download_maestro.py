"""Download MAESTRO v3 MIDI dataset for synthetic pair generation."""

import os
import urllib.request
import zipfile
from pathlib import Path

from piano_neuronal.config import MAESTRO_V3_DIR

MAESTRO_V3_MIDI_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_and_extract_maestro():
    MAESTRO_V3_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MAESTRO_V3_DIR / "maestro-v3.0.0-midi.zip"

    if not zip_path.exists():
        print(f"Downloading MAESTRO v3 MIDI from {MAESTRO_V3_MIDI_URL}...")
        urllib.request.urlretrieve(MAESTRO_V3_MIDI_URL, str(zip_path))
        print("Download complete.")
    else:
        print(f"MAESTRO zip already present: {zip_path}")

    # Check if already extracted
    midi_dir = MAESTRO_V3_DIR / "maestro-v3.0.0"
    if midi_dir.exists() and any(midi_dir.rglob("*.midi")):
        print(f"MAESTRO already extracted: {midi_dir}")
        return

    print("Extracting MAESTRO MIDI files...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(MAESTRO_V3_DIR))
    print(f"MAESTRO extracted to {MAESTRO_V3_DIR}")

    # Verify
    midi_files = list(MAESTRO_V3_DIR.rglob("*.midi"))
    print(f"Found {len(midi_files)} MIDI files")


if __name__ == "__main__":
    download_and_extract_maestro()