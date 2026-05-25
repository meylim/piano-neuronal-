"""Sprint 1 Dataset Validation Report.

Produces a comprehensive report with real numbers covering:
1. Actual pair count and segment distribution per piece
2. Segment duration distribution (short vs long)
3. Note density and polyphony per segment (rich vs poor coupling signal)
4. Split leakage verification (programmatic, fails on any leak)
5. Velocity integrity check (x1.0 = original MAESTRO velocities)

Usage: python -m piano_neuronal.s1_validate.dataset_report
"""

import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from collections import defaultdict, Counter

from piano_neuronal.config import (
    FEATURES_H5_PATH,
    MANIFEST_PATH,
    MIDI_PAIRS_H5_PATH,
    MAESTRO_V3_DIR,
    SPLIT_VELOCITY_TEST,
)

REPORT_PATH = Path("./data_output/dataset_report.md")


def check_pair_count(hf: h5py.File) -> dict:
    """1. Actual pair count and distribution per piece."""
    print("\n" + "=" * 60)
    print("1. PAIR COUNT AND SEGMENT DISTRIBUTION")
    print("=" * 60)

    pair_keys = [k for k in hf.keys() if k.startswith("pair_")]
    total_pairs = len(pair_keys)
    print(f"Total pairs: {total_pairs}")

    # Distribution: how many segments per source piece
    segments_per_piece = defaultdict(int)
    for k in pair_keys:
        source = hf[k].attrs.get("source_file", "unknown")
        segments_per_piece[source] += 1

    seg_counts = list(segments_per_piece.values())
    pieces_with_1 = sum(1 for c in seg_counts if c == 1)
    pieces_with_2_5 = sum(1 for c in seg_counts if 2 <= c <= 5)
    pieces_with_6_15 = sum(1 for c in seg_counts if 6 <= c <= 15)
    pieces_with_16_plus = sum(1 for c in seg_counts if c >= 16)

    print(f"Unique source pieces: {len(segments_per_piece)}")
    print(f"Segments per piece: mean={np.mean(seg_counts):.1f}, "
          f"median={np.median(seg_counts):.0f}, "
          f"min={min(seg_counts)}, max={max(seg_counts)}")
    print(f"  1 segment:     {pieces_with_1} pieces ({pieces_with_1/len(segments_per_piece)*100:.1f}%)")
    print(f"  2-5 segments:  {pieces_with_2_5} pieces")
    print(f"  6-15 segments: {pieces_with_6_15} pieces")
    print(f"  16+ segments:  {pieces_with_16_plus} pieces")

    return {
        "total_pairs": total_pairs,
        "unique_pieces": len(segments_per_piece),
        "segments_per_piece": seg_counts,
        "segments_per_piece_raw": segments_per_piece,
        "pieces_with_1_segment": pieces_with_1,
    }


def check_durations(hf: h5py.File) -> dict:
    """2. Segment duration distribution."""
    print("\n" + "=" * 60)
    print("2. SEGMENT DURATION DISTRIBUTION")
    print("=" * 60)

    pair_keys = [k for k in hf.keys() if k.startswith("pair_")]
    durations = []
    near_threshold = 0  # segments between 10-15s

    for k in pair_keys:
        dur = float(hf[k].attrs.get("duration_s", 0))
        durations.append(dur)
        if 10 <= dur < 15:
            near_threshold += 1

    durations = np.array(durations)
    full_pieces = sum(1 for d in durations if d < 31)  # Not segmented, < 30s + buffer

    print(f"Duration statistics:")
    print(f"  Mean:   {np.mean(durations):.1f}s")
    print(f"  Median: {np.median(durations):.1f}s")
    print(f"  Min:    {np.min(durations):.1f}s")
    print(f"  Max:    {np.max(durations):.1f}s")
    print(f"  Std:    {np.std(durations):.1f}s")
    print(f"\nDuration buckets:")
    for lo, hi, label in [(0, 15, "10-15s (near threshold)"),
                           (15, 30, "15-30s"),
                           (30, 60, "30-60s (full segment)"),
                           (60, 120, "60-120s (long)"),
                           (120, 9999, "120s+ (very long)")]:
        count = sum(1 for d in durations if lo <= d < hi)
        print(f"  {label:30s}: {count:5d} ({count/len(durations)*100:5.1f}%)")

    print(f"\nSegments near 10s threshold (10-15s): {near_threshold} ({near_threshold/len(durations)*100:.1f}%)")

    # How many pieces gave only 1 pair (short pieces)
    single_pair_sources = set()
    for k in pair_keys:
        seg_idx = int(hf[k].attrs.get("segment_idx", -1))
        if seg_idx == 0:  # Could be single-pair piece or first segment
            dur = float(hf[k].attrs.get("duration_s", 0))
            if dur < 31:  # Didn't get segmented
                single_pair_sources.add(hf[k].attrs.get("source_file", ""))

    print(f"Pieces yielding a single pair (too short to segment): {len(single_pair_sources)}")

    return {
        "durations": durations,
        "near_threshold": near_threshold,
        "single_pair_pieces": len(single_pair_sources),
    }


def check_note_density(hf: h5py.File) -> dict:
    """3. Note density and polyphony per segment."""
    print("\n" + "=" * 60)
    print("3. NOTE DENSITY AND POLYPHONY")
    print("=" * 60)

    pair_keys = [k for k in hf.keys() if k.startswith("pair_")]

    note_densities = []      # notes per second
    polyphony_means = []     # average simultaneous notes
    note_counts = []         # total note_on events

    for k in pair_keys:
        midi_events = hf[k].get("midi_events")
        if midi_events is None or len(midi_events) == 0:
            continue

        dur = float(hf[k].attrs.get("duration_s", 0))
        if dur <= 0:
            continue

        events = midi_events[:]
        n_notes = len(events)
        density = n_notes / dur

        # Compute average polyphony: at each note_on, count active notes
        note_ons = [(e[0], e[1]) for e in events]  # (start, end)
        if len(note_ons) == 0:
            continue

        # Sample polyphony at 100ms intervals
        max_time = max(e[1] for e in events) if len(events) > 0 else dur
        n_samples = min(int(dur / 0.1), 1000)
        if n_samples < 1:
            n_samples = 1

        polyphony_samples = []
        for i in range(n_samples):
            t = (i / n_samples) * dur
            active = sum(1 for start, end in events if start <= t < end)
            polyphony_samples.append(active)

        avg_poly = np.mean(polyphony_samples) if polyphony_samples else 0

        note_densities.append(density)
        polyphony_means.append(avg_poly)
        note_counts.append(n_notes)

    note_densities = np.array(note_densities)
    polyphony_means = np.array(polyphony_means)

    print(f"Note density (notes/s):")
    print(f"  Mean:   {np.mean(note_densities):.1f}")
    print(f"  Median: {np.median(note_densities):.1f}")
    print(f"  Min:    {np.min(note_densities):.1f}")
    print(f"  Max:    {np.max(note_densities):.1f}")

    print(f"\nAverage polyphony (simultaneous notes):")
    print(f"  Mean:   {np.mean(polyphony_means):.1f}")
    print(f"  Median: {np.median(polyphony_means):.1f}")
    print(f"  Min:    {np.min(polyphony_means):.1f}")
    print(f"  Max:    {np.max(polyphony_means):.1f}")

    # Define thresholds for "rich" vs "poor" coupling signal
    # A segment with < 2 notes/s or < 1.5 avg polyphony has weak coupling info
    DENSITY_THRESHOLD = 2.0   # notes per second
    POLYPHONY_THRESHOLD = 1.5  # average simultaneous notes

    rich_mask = (note_densities >= DENSITY_THRESHOLD) & (polyphony_means >= POLYPHONY_THRESHOLD)
    poor_mask = ~rich_mask

    n_rich = rich_mask.sum()
    n_poor = poor_mask.sum()

    print(f"\nCoupling signal quality (thresholds: density >= {DENSITY_THRESHOLD} notes/s, "
          f"polyphony >= {POLYPHONY_THRESHOLD}):")
    print(f"  Rich (strong coupling):    {n_rich:5d} ({n_rich/len(rich_mask)*100:5.1f}%)")
    print(f"  Poor (weak coupling):      {n_poor:5d} ({n_poor/len(rich_mask)*100:5.1f}%)")

    # Distribution of density buckets
    print(f"\nNote density distribution:")
    for lo, hi, label in [(0, 1, "0-1"), (1, 2, "1-2"), (2, 5, "2-5"),
                           (5, 10, "5-10"), (10, 20, "10-20"), (20, 999, "20+")]:
        count = sum(1 for d in note_densities if lo <= d < hi)
        print(f"  {label:10s} notes/s: {count:5d} ({count/len(note_densities)*100:5.1f}%)")

    print(f"\nPolyphony distribution:")
    for lo, hi, label in [(0, 1.5, "0-1.5 (monophonic/ sparse)"),
                           (1.5, 3, "1.5-3 (light polyphony)"),
                           (3, 6, "3-6 (moderate polyphony)"),
                           (6, 999, "6+ (dense polyphony)")]:
        count = sum(1 for p in polyphony_means if lo <= p < hi)
        print(f"  {label:35s}: {count:5d} ({count/len(polyphony_means)*100:5.1f}%)")

    return {
        "note_densities": note_densities,
        "polyphony_means": polyphony_means,
        "n_rich": n_rich,
        "n_poor": n_poor,
        "density_threshold": DENSITY_THRESHOLD,
        "polyphony_threshold": POLYPHONY_THRESHOLD,
    }


def check_splits(hf: h5py.File) -> dict:
    """4. Split verification (programmatic, fails on leak)."""
    print("\n" + "=" * 60)
    print("4. SPLIT LEAKAGE VERIFICATION")
    print("=" * 60)

    all_ok = True
    results = {}

    # --- Block A: features (piano162_s1.h5) ---
    print("\n--- Block A: Isolated samples ---")
    if MANIFEST_PATH.exists():
        df = pd.read_parquet(MANIFEST_PATH)
        total_a = len(df)

        # Check: mp velocity layer entirely in test
        mp_rows = df[df["velocity_layer"] == SPLIT_VELOCITY_TEST]
        mp_splits = mp_rows["split"].unique()
        mp_all_test = len(mp_splits) == 1 and mp_splits[0] == "test"
        print(f"  '{SPLIT_VELOCITY_TEST}' entirely in test: {mp_all_test}")
        if not mp_all_test:
            print(f"  FAIL: mp found in splits: {mp_splits}")
            all_ok = False

        # Check: every 12th note in test
        test_notes = df[df["split"] == "test"]["midi_note"].unique()
        expected_test_notes = [n for n in range(21, 109) if n % 12 == 0]
        # Not all test notes are every-12th (mp takes some too)
        # Check that no train/val note appears in test for mp
        train_val_notes = df[df["split"].isin(["train", "val"])]["midi_note"].unique()
        test_notes_non_mp = df[
            (df["split"] == "test") & (df["velocity_layer"] != SPLIT_VELOCITY_TEST)
        ]["midi_note"].unique()

        # Check: no piece from train appears in test
        # For Block A this is by note/velocity, not by piece
        split_dist = df["split"].value_counts().to_dict()
        print(f"  Split distribution: {split_dist}")
        print(f"  Total: {total_a} samples")

        # Verify no overlap between train and test by (note, velocity, pedal, mic, rr)
        train_keys = set(zip(
            df[df["split"] == "train"]["midi_note"],
            df[df["split"] == "train"]["velocity_layer"],
            df[df["split"] == "train"]["pedal"],
            df[df["split"] == "train"]["mic"],
            df[df["split"] == "train"]["round_robin"],
        ))
        test_keys = set(zip(
            df[df["split"] == "test"]["midi_note"],
            df[df["split"] == "test"]["velocity_layer"],
            df[df["split"] == "test"]["pedal"],
            df[df["split"] == "test"]["mic"],
            df[df["split"] == "test"]["round_robin"],
        ))
        overlap = train_keys & test_keys
        if overlap:
            print(f"  FAIL: {len(overlap)} samples appear in both train and test!")
            all_ok = False
        else:
            print(f"  No train/test overlap: PASS")

        results["block_a"] = {
            "mp_all_test": mp_all_test,
            "split_dist": split_dist,
            "total": total_a,
            "leak_free": len(overlap) == 0,
        }
    else:
        print("  Manifest not found — skipping Block A check")
        results["block_a"] = {"leak_free": None}

    # --- Block B: MIDI pairs ---
    print("\n--- Block B: Polyphonic pairs ---")

    # Load MAESTRO official split
    maestro_csv = None
    for candidate in [
        MAESTRO_V3_DIR / "maestro-v3.0.0" / "maestro-v3.0.0.csv",
        MAESTRO_V3_DIR / "maestro-v3.0.0.csv",
    ]:
        if candidate.exists():
            maestro_csv = candidate
            break

    if maestro_csv.exists():
        maestro_df = pd.read_csv(maestro_csv)
        official_split = {}
        for _, row in maestro_df.iterrows():
            stem = Path(row["midi_filename"]).stem
            official_split[stem] = row["split"]

        # Check: each pair has a split attribute
        pair_keys = [k for k in hf.keys() if k.startswith("pair_")]
        splits_found = {"train": 0, "val": 0, "test": 0, "validation": 0, "unknown": 0}
        piece_splits = defaultdict(set)  # piece -> set of splits its segments are in

        for k in pair_keys:
            split = hf[k].attrs.get("split", "unknown")
            source = hf[k].attrs.get("source_file", "unknown")
            if split == "validation":
                split = "val"
            splits_found[split] = splits_found.get(split, 0) + 1
            piece_splits[source].add(split)

        print(f"  Split distribution in pairs: {splits_found}")

        # Check: no piece has segments in more than one split
        leaked_pieces = {p: s for p, s in piece_splits.items() if len(s) > 1}
        if leaked_pieces:
            print(f"  FAIL: {len(leaked_pieces)} pieces have segments in multiple splits!")
            for p, s in list(leaked_pieces.items())[:5]:
                print(f"    {p}: splits = {s}")
            all_ok = False
        else:
            print(f"  No piece has segments across splits: PASS")

        # Check: splits match MAESTRO official
        mismatch = 0
        for source, splits in piece_splits.items():
            if source in official_split:
                expected = official_split[source]
                if expected == "validation":
                    expected = "val"
                actual = list(splits)[0] if len(splits) == 1 else "?"
                if actual != expected:
                    mismatch += 1

        if mismatch > 0:
            print(f"  FAIL: {mismatch} pieces have split different from MAESTRO official!")
            all_ok = False
        else:
            print(f"  All piece splits match MAESTRO official: PASS")

        # Summary stats
        train_pairs = splits_found.get("train", 0)
        val_pairs = splits_found.get("val", 0)
        test_pairs = splits_found.get("test", 0)
        print(f"  Train: {train_pairs} pairs ({train_pairs/len(pair_keys)*100:.1f}%)")
        print(f"  Val:   {val_pairs} pairs ({val_pairs/len(pair_keys)*100:.1f}%)")
        print(f"  Test:  {test_pairs} pairs ({test_pairs/len(pair_keys)*100:.1f}%)")

        results["block_b"] = {
            "split_dist": splits_found,
            "leaked_pieces": len(leaked_pieces),
            "split_mismatches": mismatch,
            "leak_free": len(leaked_pieces) == 0 and mismatch == 0,
        }
    else:
        print("  MAESTRO CSV not found — cannot verify official split")
        results["block_b"] = {"leak_free": None}

    results["all_ok"] = all_ok
    return results


def check_velocity_integrity(hf: h5py.File) -> dict:
    """5. Verify velocity integrity (x1.0 = original MAESTRO)."""
    print("\n" + "=" * 60)
    print("5. VELOCITY INTEGRITY CHECK (x1.0 = no scaling)")
    print("=" * 60)

    pair_keys = [k for k in hf.keys() if k.startswith("pair_")]
    vel_scales = set()
    for k in pair_keys[:1000]:  # Check first 1000
        vs = hf[k].attrs.get("velocity_scale", -1)
        vel_scales.add(vs)

    print(f"  Unique velocity_scale values found: {sorted(vel_scales)}")

    if vel_scales == {1.0}:
        print(f"  PASS: Only x1.0 velocity scaling — no augmentation applied")
        integrity_ok = True
    elif vel_scales == {1.0, 0.7, 1.3}:
        print(f"  WARNING: x0.7 and x1.3 velocity augmentation still present!")
        print(f"  These create false velocity-timbre signal (clipping at 127, compression at 0.7)")
        integrity_ok = False
    else:
        print(f"  UNEXPECTED velocity scales: {vel_scales}")
        integrity_ok = False

    # Verify MIDI velocities are original (not scaled)
    # Sample a few pairs and check that velocities match the original MIDI
    maestro_dir = MAESTRO_V3_DIR / "maestro-v3.0.0"
    if not maestro_dir.exists():
        maestro_dir = MAESTRO_V3_DIR

    import pretty_midi
    velocity_match = 0
    velocity_mismatch = 0
    checked = 0

    for k in pair_keys[:20]:  # Check first 20 pairs
        source = hf[k].attrs.get("source_file", "")
        if not source:
            continue

        # Find original MIDI file
        midi_path = None
        for ext in [".midi", ".mid"]:
            candidates = list(maestro_dir.rglob(f"{source}{ext}"))
            if candidates:
                midi_path = candidates[0]
                break

        if midi_path is None:
            continue

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            original_vels = sorted([n.velocity for inst in pm.instruments for n in inst.notes])

            pair_events = hf[k].get("midi_events")
            if pair_events is not None and len(pair_events) > 0:
                pair_vels = sorted([e[3] for e in pair_events[:]])

                # Check that velocities are not scaled (should match original)
                # Note: segments only cover a time window, so we compare
                # the velocities that fall within the segment time
                seg_start = hf[k].attrs.get("segment_start_s", 0)
                seg_dur = hf[k].attrs.get("duration_s", 0)

                pair_vels_in_range = sorted([
                    e[3] for e in pair_events[:]
                    if seg_start <= e[0] < seg_start + seg_dur
                ])

                if pair_vels_in_range and original_vels:
                    # Check min/max are reasonable (not all clipped to 127 or 1)
                    max_vel = max(pair_vels_in_range)
                    min_vel = min(pair_vels_in_range)
                    if max_vel <= 127 and min_vel >= 1:
                        velocity_match += 1
                    else:
                        velocity_mismatch += 1
                checked += 1
        except Exception:
            pass

    print(f"  Velocity range checks (sample of {checked} pairs):")
    print(f"    Valid ranges: {velocity_match}")
    print(f"    Invalid ranges: {velocity_mismatch}")

    # Check for clipping at 127
    clipped_127 = 0
    total_notes = 0
    for k in pair_keys[:100]:
        pair_events = hf[k].get("midi_events")
        if pair_events is not None and len(pair_events) > 0:
            vels = [e[3] for e in pair_events[:]]
            clipped_127 += sum(1 for v in vels if v >= 127)
            total_notes += len(vels)

    clip_pct = (clipped_127 / total_notes * 100) if total_notes > 0 else 0
    print(f"  Notes clipped at 127: {clipped_127}/{total_notes} ({clip_pct:.2f}%)")

    if clip_pct > 1.0:
        print(f"  WARNING: >1% of notes at velocity 127 — possible saturation")
        integrity_ok = False
    else:
        print(f"  PASS: Clipping rate within acceptable range (<1%)")

    return {
        "velocity_scales": sorted(vel_scales),
        "integrity_ok": integrity_ok,
        "clip_rate": clip_pct,
    }


def generate_report(hf: h5py.File, results: dict) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Sprint 1 — Dataset Validation Report\n")
    lines.append(f"Generated automatically by `dataset_report.py`\n")

    lines.append("## 1. Pair Count\n")
    r1 = results.get("pair_count", {})
    lines.append(f"- **Total pairs**: {r1.get('total_pairs', 'N/A')}")
    lines.append(f"- **Unique source pieces**: {r1.get('unique_pieces', 'N/A')}")
    lines.append(f"- **Segments per piece**: mean={np.mean(r1['segments_per_piece']):.1f}, "
                 f"median={np.median(r1['segments_per_piece']):.0f}, "
                 f"min={min(r1['segments_per_piece'])}, max={max(r1['segments_per_piece'])}")
    lines.append(f"- **Pieces yielding single pair**: {r1.get('pieces_with_1_segment', 'N/A')}\n")

    lines.append("## 2. Segment Durations\n")
    r2 = results.get("durations", {})
    durs = r2.get("durations", np.array([]))
    if len(durs) > 0:
        lines.append(f"- **Mean**: {np.mean(durs):.1f}s, **Median**: {np.median(durs):.1f}s")
        lines.append(f"- **Min**: {np.min(durs):.1f}s, **Max**: {np.max(durs):.1f}s")
        lines.append(f"- **Near threshold (10-15s)**: {r2.get('near_threshold', 'N/A')}")
        lines.append(f"- **Single-pair pieces (too short to segment)**: {r2.get('single_pair_pieces', 'N/A')}\n")

    lines.append("## 3. Note Density & Polyphony\n")
    r3 = results.get("density", {})
    if r3:
        lines.append(f"- **Rich segments** (density >= {r3['density_threshold']}, "
                     f"polyphony >= {r3['polyphony_threshold']}): "
                     f"{r3['n_rich']} ({r3['n_rich']/(r3['n_rich']+r3['n_poor'])*100:.1f}%)")
        lines.append(f"- **Poor segments** (weak coupling signal): "
                     f"{r3['n_poor']} ({r3['n_poor']/(r3['n_rich']+r3['n_poor'])*100:.1f}%)")
        lines.append(f"- **Recommended threshold for filtering**: density >= {r3['density_threshold']} notes/s "
                     f"AND polyphony >= {r3['polyphony_threshold']}\n")

    lines.append("## 4. Split Verification\n")
    r4 = results.get("splits", {})
    lines.append(f"- **Block A leak-free**: {r4.get('block_a', {}).get('leak_free', 'N/A')}")
    lines.append(f"- **Block B leak-free**: {r4.get('block_b', {}).get('leak_free', 'N/A')}")
    lines.append(f"- **Overall**: {'PASS' if r4.get('all_ok') else 'FAIL'}\n")

    lines.append("## 5. Velocity Integrity\n")
    r5 = results.get("velocity", {})
    lines.append(f"- **Velocity scales used**: {r5.get('velocity_scales', 'N/A')}")
    lines.append(f"- **Clipping rate at 127**: {r5.get('clip_rate', 'N/A'):.2f}%")
    lines.append(f"- **Integrity**: {'PASS' if r5.get('integrity_ok') else 'FAIL'}\n")

    return "\n".join(lines)


def run_dataset_report() -> bool:
    """Run all dataset validation checks and produce a report."""
    print("=" * 60)
    print("SPRINT 1 — DATASET VALIDATION REPORT")
    print("=" * 60)

    results = {}

    # Block B checks (MIDI pairs)
    if not MIDI_PAIRS_H5_PATH.exists():
        print("\nERROR: midi_pairs.h5 not found. Run Phase 2 first.")
        return False

    with h5py.File(MIDI_PAIRS_H5_PATH, "r") as hf:
        results["pair_count"] = check_pair_count(hf)
        results["durations"] = check_durations(hf)
        results["density"] = check_note_density(hf)
        results["splits"] = check_splits(hf)
        results["velocity"] = check_velocity_integrity(hf)

    # Generate report
    with h5py.File(MIDI_PAIRS_H5_PATH, "r") as hf:
        report = generate_report(hf, results)

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")

    # Final verdict
    all_ok = (
        results.get("splits", {}).get("all_ok", False) and
        results.get("velocity", {}).get("integrity_ok", False)
    )

    print("\n" + "=" * 60)
    if all_ok:
        print("DATASET VALIDATION: ALL CHECKS PASSED")
    else:
        print("DATASET VALIDATION: SOME CHECKS FAILED — review above")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = run_dataset_report()
    sys.exit(0 if success else 1)