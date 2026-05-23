"""Run the full Sprint 1 data pipeline.

Usage: python -m piano_neuronal.s1_data.run_pipeline [--skip-features] [--skip-midi]
"""

import argparse
from pathlib import Path

from piano_neuronal.config import OUTPUT_DIR, FEATURES_H5_PATH, MANIFEST_PATH, MIDI_PAIRS_H5_PATH


def main():
    parser = argparse.ArgumentParser(description="Sprint 1 data pipeline")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction (use existing HDF5)")
    parser.add_argument("--skip-midi", action="store_true", help="Skip MIDI pair generation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_features:
        print("=" * 60)
        print("PHASE 1: Parse dataset + Extract features + Serialize")
        print("=" * 60)
        from piano_neuronal.s1_data.serialize import process_and_serialize
        process_and_serialize()

        print(f"\nFeatures HDF5: {FEATURES_H5_PATH}")
        print(f"Manifest Parquet: {MANIFEST_PATH}")

    if not args.skip_midi:
        print("\n" + "=" * 60)
        print("PHASE 2: Generate MIDI-audio pairs")
        print("=" * 60)
        from piano_neuronal.s1_midi.midi_pairs import create_midi_pairs_dataset
        create_midi_pairs_dataset(target_pairs=5000)

        print(f"\nMIDI pairs HDF5: {MIDI_PAIRS_H5_PATH}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Run validation:")
    print("  python -m piano_neuronal.s1_validate.validation")
    print("=" * 60)


if __name__ == "__main__":
    main()