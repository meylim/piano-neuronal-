"""Run the full Sprint 1 data pipeline.

Usage: python -m piano_neuronal.s1_data.run_pipeline [--skip-features] [--skip-midi] [--workers N] [--target-pairs N]
"""

import argparse
from pathlib import Path

from piano_neuronal.config import OUTPUT_DIR, FEATURES_H5_PATH, MANIFEST_PATH, MIDI_PAIRS_H5_PATH


def main():
    parser = argparse.ArgumentParser(description="Sprint 1 data pipeline")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction (use existing HDF5)")
    parser.add_argument("--skip-midi", action="store_true", help="Skip MIDI pair generation")
    parser.add_argument("--workers", type=int, default=25, help="Number of parallel workers (default: 25)")
    parser.add_argument("--target-pairs", type=int, default=0, help="Target number of MIDI pairs (0 = all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_features:
        print("=" * 60)
        print("PHASE 1: Parse dataset + Extract features + Serialize")
        print("=" * 60)
        import piano_neuronal.s1_data.serialize as ser
        ser.N_WORKERS = args.workers
        ser.process_and_serialize()

        print(f"\nFeatures HDF5: {FEATURES_H5_PATH}")
        print(f"Manifest Parquet: {MANIFEST_PATH}")

    if not args.skip_midi:
        print("\n" + "=" * 60)
        print("PHASE 2: Generate MIDI-audio pairs")
        print("=" * 60)
        import piano_neuronal.s1_midi.midi_pairs as mp
        mp.N_WORKERS = args.workers
        mp.create_midi_pairs_dataset(target_pairs=args.target_pairs)

        print(f"\nMIDI pairs HDF5: {MIDI_PAIRS_H5_PATH}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Run validation:")
    print("  python -m piano_neuronal.s1_validate.validation")
    print("=" * 60)


if __name__ == "__main__":
    main()