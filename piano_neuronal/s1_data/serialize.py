"""Serialize all Piano in 162 samples + features into HDF5 and manifest Parquet.

Split strategy:
  - Train: notes not in val/test, all velocity layers except 'MezzoPiano' (mp)
  - Val: 10% of remaining notes (random, stratified by register)
  - Test: 'MezzoPiano' velocity layer (reserved for velocity interpolation check)
    + held-out notes (every 8th note for cross-pitch generalization)

Each manifest row carries its assignment: train / val / test.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

from piano_neuronal.config import (
    FEATURES_H5_PATH,
    MANIFEST_PATH,
    OUTPUT_DIR,
    SOURCE_SAMPLE_RATE,
    SPLIT_VELOCITY_TEST,
    SPLIT_VAL_RATIO,
    SPLIT_SEED,
    PIANO_IN_162_SAMPLES,
)
from piano_neuronal.s1_data.dataset_parser import discover_all_files, parse_filename
from piano_neuronal.s1_data.audio_loader import load_audio
from piano_neuronal.s1_features.extract_all import extract_all_features


def assign_split(midi_note: int, velocity_layer: str, rng: np.random.Generator) -> str:
    """Assign train/val/test split for a sample.

    Rules:
    - 'MezzoPiano' velocity layer → always test (velocity interpolation)
    - Every 8th note (C notes) → test (cross-pitch generalization)
    - 10% of remaining → val
    - Everything else → train
    """
    # All mp samples go to test
    if velocity_layer == SPLIT_VELOCITY_TEST:
        return "test"

    # Every 8th note: test for cross-pitch generalization
    # MIDI notes 24 (C1), 36 (C2), 48 (C3), 60 (C4), 72 (C5), 84 (C6), 96 (C7)
    if midi_note % 12 == 0 and midi_note in range(21, 109):
        return "test"

    # Remaining: 10% val, 90% train
    if rng.random() < SPLIT_VAL_RATIO:
        return "val"

    return "train"


def process_and_serialize():
    """Main pipeline: discover files, extract features, serialize to HDF5 + manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all FLAC files
    print("Discovering Piano in 162 files...")
    all_files = discover_all_files()
    if not all_files:
        print("ERROR: No files found. Check dataset path in config.py")
        return

    rng = np.random.default_rng(SPLIT_SEED)

    # Assign splits
    for f in all_files:
        f["split"] = assign_split(f["midi_note"], f["velocity_layer"], rng)

    # Verify split distribution
    split_counts = {}
    for f in all_files:
        split_counts[f["split"]] = split_counts.get(f["split"], 0) + 1
    print(f"Split distribution: {split_counts}")

    # Open HDF5 for writing (use rdcc_nbytes=0 to avoid Windows file locking issues)
    with h5py.File(FEATURES_H5_PATH, "w", rdcc_nbytes=0) as hf:
        # Flat group names to avoid HDF5 deep nesting issues
        manifest_records = []

        for i, meta in enumerate(tqdm(all_files, desc="Processing samples")):
            filepath = Path(meta["file_path"])
            group_name = (
                f"note{meta['midi_note']:03d}"
                f"_pedal{meta['pedal']}"
                f"_vel{meta['velocity_layer']}"
                f"_mic{meta['mic']}"
                f"_rr{meta['round_robin']}"
            )

            try:
                # Load audio once in stereo, derive mono from it
                audio_stereo, load_meta = load_audio(
                    filepath, target_sr=SOURCE_SAMPLE_RATE, mono=False, align_onset=True
                )
                # Derive mono from stereo (mean of channels)
                audio_mono = np.mean(audio_stereo, axis=0, keepdims=True)  # (1, N)

                # Extract features (mono for most, close+ambient for IR)
                features, exc_result, ir_result = extract_all_features(
                    audio_mono=audio_mono[0],
                    audio_close_mono=audio_mono[0] if meta["mic"] == "Close" else None,
                    audio_ambient_mono=audio_mono[0] if meta["mic"] == "Ambient" else None,
                    sr=SOURCE_SAMPLE_RATE,
                    midi_note=meta["midi_note"],
                    mic_type=meta["mic"],
                    pedal=meta["pedal"],
                )

                # Write to HDF5
                grp = hf.create_group(group_name)
                grp.create_dataset("audio_stereo", data=audio_stereo, compression="gzip", chunks=True)
                grp.create_dataset("audio_mono", data=audio_mono, compression="gzip", chunks=True)
                grp.attrs["sample_rate"] = load_meta["sample_rate"]
                grp.attrs["midi_note"] = meta["midi_note"]
                grp.attrs["velocity_layer"] = meta["velocity_layer"]
                grp.attrs["velocity_continuous"] = meta["velocity_continuous"]
                grp.attrs["pedal"] = meta["pedal"]
                grp.attrs["mic"] = meta["mic"]
                grp.attrs["round_robin"] = meta["round_robin"]
                grp.attrs["onset_sample_idx"] = load_meta.get("onset_sample_idx", 0)

                # Store scalar features as attributes
                for key, val in features.items():
                    if isinstance(val, (int, float)):
                        grp.attrs[key] = val
                    elif isinstance(val, np.ndarray) and val.ndim == 1:
                        grp.create_dataset(f"feat_{key}", data=val, compression="gzip")

                # Store excitation arrays
                if exc_result is not None:
                    grp.create_dataset("excitation_raw", data=exc_result["excitation_raw"], compression="gzip")
                    grp.create_dataset("excitation_residual", data=exc_result["excitation_residual"], compression="gzip")

                # Store IR if extracted
                if ir_result is not None and len(ir_result["ir"]) > 0:
                    grp.create_dataset("room_ir", data=ir_result["ir"], compression="gzip")

                # Manifest record
                record = {
                    "file_path": str(filepath),
                    "group_path": group_name,
                    "midi_note": meta["midi_note"],
                    "velocity_layer": meta["velocity_layer"],
                    "velocity_continuous": meta["velocity_continuous"],
                    "pedal": meta["pedal"],
                    "mic": meta["mic"],
                    "round_robin": meta["round_robin"],
                    "split": meta["split"],
                    **features,
                }
                # Remove array values from manifest (keep only scalars)
                record = {k: v for k, v in record.items() if isinstance(v, (int, float, str))}
                manifest_records.append(record)

            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
                continue

    # Write manifest as Parquet
    if manifest_records:
        df = pd.DataFrame(manifest_records)
        df.to_parquet(MANIFEST_PATH, engine="pyarrow")
        print(f"\nManifest saved: {len(manifest_records)} entries → {MANIFEST_PATH}")

        # Print split summary
        print("\nSplit summary:")
        for split in ["train", "val", "test"]:
            count = len(df[df["split"] == split])
            print(f"  {split}: {count} samples ({count/len(df)*100:.1f}%)")
    else:
        print("ERROR: No samples processed successfully.")


if __name__ == "__main__":
    process_and_serialize()