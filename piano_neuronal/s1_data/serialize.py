"""Serialize all Piano in 162 samples + features into HDF5 and manifest Parquet.

Multiprocessing version with batched processing to control memory usage.
Extracts features in parallel across CPU cores, writes results to HDF5
in batches of BATCH_SIZE files.

Split strategy:
  - Train: notes not in val/test, all velocity layers except 'MezzoPiano' (mp)
  - Val: 10% of remaining notes (random, stratified by register)
  - Test: 'MezzoPiano' velocity layer (reserved for velocity interpolation check)
    + held-out notes (every 12th note for cross-pitch generalization)

Each manifest row carries its assignment: train / val / test.
"""

import gc
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # Limit BLAS threads per worker

import numpy as np
import pandas as pd
import h5py
import soundfile as sf
import librosa
from pathlib import Path
from multiprocessing import Pool, cpu_count
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
from piano_neuronal.s1_data.dataset_parser import discover_all_files
from piano_neuronal.s1_features.extract_all import extract_all_features
from piano_neuronal.s1_features.room_ir import extract_room_ir
from piano_neuronal.s1_features.inharmonicity import interpolate_B_fallback

BATCH_SIZE = 50  # Files per batch — controls memory usage
N_WORKERS = 10


def _make_group_name(meta: dict) -> str:
    """Build HDF5 group name from file metadata (for resume filtering)."""
    return (
        f"note{meta['midi_note']:03d}"
        f"_pedal{meta['pedal']}"
        f"_vel{meta['velocity_layer']}"
        f"_mic{meta['mic']}"
        f"_rr{meta['round_robin']}"
    )


def assign_split(midi_note: int, velocity_layer: str, rng: np.random.Generator) -> str:
    if velocity_layer == SPLIT_VELOCITY_TEST:
        return "test"
    if midi_note % 12 == 0 and midi_note in range(21, 109):
        return "test"
    if rng.random() < SPLIT_VAL_RATIO:
        return "val"
    return "train"


def _extract_one(meta: dict) -> dict:
    """Worker: load audio, extract features. Returns serializable dict."""
    filepath = Path(meta["file_path"])

    try:
        audio, sr_file = sf.read(str(filepath), dtype="float32")
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T  # (channels, frames)

        sr = sr_file  # 44100 Hz — no resampling needed

        # Onset alignment
        ref_channel = audio[0]
        onsets = librosa.onset.onset_detect(y=ref_channel, sr=sr, backtrack=True, units="samples")
        if len(onsets) > 0:
            pre_buffer = int(0.001 * sr)
            onset_idx = max(0, onsets[0] - pre_buffer)
            audio = audio[:, onset_idx:]
        else:
            onset_idx = 0

        audio_mono = np.mean(audio, axis=0)

        features, exc_result, ir_result, mfcc_mean, mfcc_std = extract_all_features(
            audio_mono=audio_mono,
            audio_close_mono=audio_mono if meta["mic"] == "Close" else None,
            audio_ambient_mono=audio_mono if meta["mic"] == "Ambient" else None,
            sr=sr,
            midi_note=meta["midi_note"],
            mic_type=meta["mic"],
            pedal=meta["pedal"],
        )

        group_name = (
            f"note{meta['midi_note']:03d}"
            f"_pedal{meta['pedal']}"
            f"_vel{meta['velocity_layer']}"
            f"_mic{meta['mic']}"
            f"_rr{meta['round_robin']}"
        )

        return {
            "success": True,
            "meta": meta,
            "group_name": group_name,
            "audio_stereo": audio,
            "audio_mono": audio_mono,
            "sr": sr,
            "onset_idx": onset_idx,
            "features": features,
            "exc_result": exc_result,
            "ir_result": ir_result,
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "filepath": filepath.name}


def process_and_serialize():
    """Main pipeline: extract features in parallel batches, write to HDF5 incrementally.

    Supports resuming: if FEATURES_H5_PATH already exists, skips groups that
    are already present and appends new ones.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering Piano in 162 files...")
    all_files = discover_all_files()
    if not all_files:
        print("ERROR: No files found.")
        return

    rng = np.random.default_rng(SPLIT_SEED)
    for f in all_files:
        f["split"] = assign_split(f["midi_note"], f["velocity_layer"], rng)

    # Resume: check which groups already exist
    existing_groups = set()
    resume_mode = FEATURES_H5_PATH.exists()
    if resume_mode:
        with h5py.File(FEATURES_H5_PATH, "r") as hf:
            existing_groups = set(hf.keys())
        print(f"Resuming: {len(existing_groups)} groups already in HDF5, skipping them")

    split_counts = {}
    for f in all_files:
        split_counts[f["split"]] = split_counts.get(f["split"], 0) + 1
    print(f"Split distribution: {split_counts}")
    print(f"Processing with {N_WORKERS} workers, batch size {BATCH_SIZE}")

    # Filter out already-processed files
    if existing_groups:
        before = len(all_files)
        all_files = [f for f in all_files if f"group_name_for_resume" not in f or True]
        # We'll filter during processing based on group_name
        print(f"  {before - len([f for f in all_files if _make_group_name(f) in existing_groups])} files already processed, {len(all_files) - len([f for f in all_files if _make_group_name(f) in existing_groups])} remaining")
        todo_files = [f for f in all_files if _make_group_name(f) not in existing_groups]
    else:
        todo_files = all_files

    manifest_records = []
    errors = []
    success_count = len(existing_groups)

    # Process in batches to control memory
    # maxtasksperchild forces worker restart every 100 tasks to prevent
    # memory accumulation from librosa/numpy FFT buffers
    h5_mode = "a" if resume_mode else "w"
    # Use libver='latest' for new files to enable HDF5 v2 format with 64-bit addressing.
    # This prevents "addr overflow" errors when the file grows beyond ~17GB.
    h5_kwargs = {"rdcc_nbytes": 0}
    if h5_mode == "w":
        h5_kwargs["libver"] = "latest"
    with h5py.File(FEATURES_H5_PATH, h5_mode, **h5_kwargs) as hf:
        with Pool(N_WORKERS, maxtasksperchild=100) as pool:
            for batch_start in range(0, len(todo_files), BATCH_SIZE):
                batch = todo_files[batch_start:batch_start + BATCH_SIZE]
                batch_results = list(tqdm(
                    pool.imap_unordered(_extract_one, batch),
                    total=len(batch),
                    desc=f"Batch {batch_start // BATCH_SIZE + 1}/{(len(todo_files) + BATCH_SIZE - 1) // BATCH_SIZE}",
                    leave=False,
                ))

                # Write batch results to HDF5 immediately
                for result in batch_results:
                    if not result["success"]:
                        errors.append((result.get("filepath", "?"), result["error"]))
                        continue

                    meta = result["meta"]
                    group_name = result["group_name"]

                    try:
                        audio_stereo = result["audio_stereo"]
                        audio_mono = result["audio_mono"]
                        features = result["features"]
                        exc_result = result["exc_result"]
                        ir_result = result["ir_result"]

                        grp = hf.create_group(group_name)
                        grp.create_dataset("audio_stereo", data=audio_stereo, compression="lzf", chunks=True)
                        grp.create_dataset("audio_mono", data=audio_mono[np.newaxis, :], compression="lzf", chunks=True)
                        grp.attrs["sample_rate"] = result["sr"]
                        grp.attrs["midi_note"] = meta["midi_note"]
                        grp.attrs["velocity_layer"] = meta["velocity_layer"]
                        grp.attrs["velocity_continuous"] = meta["velocity_continuous"]
                        grp.attrs["pedal"] = meta["pedal"]
                        grp.attrs["mic"] = meta["mic"]
                        grp.attrs["round_robin"] = meta["round_robin"]
                        grp.attrs["onset_sample_idx"] = result["onset_idx"]

                        for key, val in features.items():
                            if isinstance(val, (int, float)):
                                grp.attrs[key] = val
                            elif isinstance(val, np.ndarray) and val.ndim == 1:
                                grp.create_dataset(f"feat_{key}", data=val, compression="lzf")

                        if exc_result is not None:
                            grp.create_dataset("excitation_raw", data=exc_result["excitation_raw"], compression="lzf")
                            grp.create_dataset("excitation_residual", data=exc_result["excitation_residual"], compression="lzf")

                        if ir_result is not None and len(ir_result["ir"]) > 0:
                            grp.create_dataset("room_ir", data=ir_result["ir"], compression="lzf")

                        # MFCC arrays
                        if result.get("mfcc_mean") is not None:
                            grp.create_dataset("feat_mfcc_mean", data=result["mfcc_mean"], compression="lzf")
                        if result.get("mfcc_std") is not None:
                            grp.create_dataset("feat_mfcc_std", data=result["mfcc_std"], compression="lzf")

                        # Manifest record (scalars only)
                        record = {
                            "file_path": str(Path(meta["file_path"])),
                            "group_path": group_name,
                            "midi_note": meta["midi_note"],
                            "velocity_layer": meta["velocity_layer"],
                            "velocity_continuous": meta["velocity_continuous"],
                            "pedal": meta["pedal"],
                            "mic": meta["mic"],
                            "round_robin": meta["round_robin"],
                            "split": meta["split"],
                        }
                        for key, val in features.items():
                            if isinstance(val, (int, float)):
                                record[key] = val
                        manifest_records.append(record)
                        success_count += 1

                    except Exception as e:
                        errors.append((meta.get("file_path", "?"), str(e)))

                # Free batch memory and flush HDF5
                del batch_results
                gc.collect()  # Force Python GC to reclaim numpy arrays from workers
                hf.flush()

                print(f"  Progress: {success_count}/{len(all_files)} files "
                      f"({success_count/len(all_files)*100:.1f}%), "
                      f"{len(todo_files) - (batch_start + len(batch))} remaining, "
                      f"{len(errors)} errors")

    # Write manifest
    if manifest_records:
        df = pd.DataFrame(manifest_records)
        df.to_parquet(MANIFEST_PATH, engine="pyarrow")
        print(f"\nManifest saved: {len(manifest_records)} entries -> {MANIFEST_PATH}")
        for split in ["train", "val", "test"]:
            count = len(df[df["split"] == split])
            print(f"  {split}: {count} samples ({count/len(df)*100:.1f}%)")
    else:
        print("ERROR: No samples processed.")

    if errors:
        print(f"\n{len(errors)} errors encountered:")
        for fname, err in errors[:20]:
            print(f"  {fname}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    print(f"\nDone: {success_count}/{len(all_files)} files processed successfully.")

    # --- Post-processing: extract Room IR from Close+Ambient pairs ---
    print("\nExtracting Room IR from Close+Ambient pairs...")
    _extract_room_ir_pairs(all_files)

    # --- Post-processing: interpolate B=0 values ---
    print("\nInterpolating B=0 values...")
    _interpolate_B_values()


def _extract_room_ir_pairs(all_files: list[dict]) -> None:
    """Pair Close and Ambient mic recordings to extract Room IR.

    Room IR requires both mic positions for the same note/velocity/pedal/RR.
    This is a post-processing step because the parallel extraction processes
    each file independently.
    """
    # Group files by (midi_note, velocity_layer, pedal, round_robin)
    from collections import defaultdict
    pairs = defaultdict(dict)
    for f in all_files:
        key = (f["midi_note"], f["velocity_layer"], f["pedal"], f["round_robin"])
        pairs[key][f["mic"]] = f["file_path"]

    # Find groups that have both Close and Ambient
    both_mics = {k: v for k, v in pairs.items() if "Close" in v and "Ambient" in v}
    print(f"  Found {len(both_mics)} Close+Ambient pairs")

    ir_count = 0
    errors_ir = 0
    with h5py.File(FEATURES_H5_PATH, "a") as hf:
        for (note, vel, pedal, rr), mics in tqdm(both_mics.items(), desc="Room IR"):
            try:
                close_group_name = (
                    f"note{note:03d}_pedal{pedal}_vel{vel}_micClose_rr{rr}"
                )
                ambient_group_name = (
                    f"note{note:03d}_pedal{pedal}_vel{vel}_micAmbient_rr{rr}"
                )

                if close_group_name not in hf or ambient_group_name not in hf:
                    continue

                # Only extract IR for PedalOff (room signature, not sustain resonance)
                if pedal != "Off":
                    continue

                close_mono = hf[close_group_name]["audio_mono"][:]
                ambient_mono = hf[ambient_group_name]["audio_mono"][:]

                # Ensure 1D
                if close_mono.ndim > 1:
                    close_mono = close_mono.squeeze()
                if ambient_mono.ndim > 1:
                    ambient_mono = ambient_mono.squeeze()

                sr = int(hf[close_group_name].attrs["sample_rate"])
                ir_result = extract_room_ir(close_mono, ambient_mono, sr, onset_idx=0)

                if len(ir_result["ir"]) > 0:
                    # Write IR to Close group
                    if "room_ir" in hf[close_group_name]:
                        del hf[close_group_name]["room_ir"]
                    hf[close_group_name].create_dataset(
                        "room_ir", data=ir_result["ir"], compression="lzf"
                    )
                    hf[close_group_name].attrs["ir_duration_s"] = ir_result["ir_duration_s"]
                    hf[close_group_name].attrs["ir_t60"] = ir_result["ir_t60"]
                    ir_count += 1

            except Exception as e:
                errors_ir += 1
                if errors_ir <= 5:
                    print(f"  Error extracting IR for note{note} vel={vel} pedal={pedal}: {e}")

    print(f"  Room IR extracted: {ir_count}, errors: {errors_ir}")


def _interpolate_B_values() -> None:
    """Interpolate B=0 values from neighbouring notes in the HDF5.

    Loads all B and r_squared values, groups by (velocity_layer, pedal, mic),
    and linearly interpolates any B=0 / r_squared=0 entries.
    """
    with h5py.File(FEATURES_H5_PATH, "a") as hf:
        groups = list(hf.keys())
        results = []

        # Collect all B values
        for gname in groups:
            grp = hf[gname]
            if "B" in grp.attrs and "midi_note" in grp.attrs:
                results.append({
                    "group_name": gname,
                    "midi_note": int(grp.attrs["midi_note"]),
                    "velocity_layer": grp.attrs["velocity_layer"],
                    "pedal": grp.attrs["pedal"],
                    "mic": grp.attrs["mic"],
                    "B": float(grp.attrs["B"]),
                    "B_fit_r_squared": float(grp.attrs.get("B_fit_r_squared", 0.0)),
                })

        # Interpolate
        interpolated = interpolate_B_fallback(results)

        # Write back interpolated values
        updated = 0
        for r in interpolated:
            if r.get("B_fit_r_squared") == -1.0:  # Interpolated marker
                grp = hf[r["group_name"]]
                grp.attrs["B"] = r["B"]
                grp.attrs["B_fit_r_squared"] = -1.0  # Mark as interpolated
                updated += 1

    print(f"  Interpolated B values: {updated} notes")


if __name__ == "__main__":
    process_and_serialize()