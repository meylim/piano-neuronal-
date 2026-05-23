from pathlib import Path

# Dataset paths (actual locations on this machine)
PIANO_IN_162_ROOT = Path(r"C:\Users\sajid\Documents\instrument\Ivy Audio - Piano in 162 sfz")
PIANO_IN_162_SAMPLES = PIANO_IN_162_ROOT / "Piano in 162 Samples"
SFZ_CLOSE_PATH = PIANO_IN_162_ROOT / "IvyAudio-PianoIn162-Close.sfz"
SFZ_AMBIENT_PATH = PIANO_IN_162_ROOT / "IvyAudio-PianoIn162-Ambient.sfz"
MAESTRO_V3_DIR = Path(r"C:\Users\sajid\Documents\datasets\maestro_v3")

# Sample rates (CORRECTED: source is 44.1 kHz, NOT 96 kHz)
SOURCE_SAMPLE_RATE = 44100
SAMPLE_RATE_A = 48000   # Axe A: upsampled for web/mobile standard
SAMPLE_RATE_B = 44100   # Axe B: native, no resampling needed

# Note range
MIDI_NOTE_MIN = 21   # A0
MIDI_NOTE_MAX = 108  # C8

# SFZ velocity mapping (from the actual .sfz file)
VELOCITY_LAYERS = {
    "Pianissimo": {"lovel": 1,   "hivel": 33,  "center": 17,  "continuous": 17 / 127},
    "Piano":       {"lovel": 34,  "hivel": 64,  "center": 49,  "continuous": 49 / 127},
    "MezzoPiano":  {"lovel": 65,  "hivel": 80,  "center": 72,  "continuous": 72 / 127},
    "MezzoForte":  {"lovel": 81,  "hivel": 101, "center": 91,  "continuous": 91 / 127},
    "Forte":       {"lovel": 102, "hivel": 127, "center": 114, "continuous": 114 / 127},
}

# Inharmonicity extraction: adaptive n_fft by register
N_FFT_REGIMES = {
    "bass":   {"note_range": (21, 35),  "n_fft": 16384},  # A0–B1: close partials need long window
    "low_mid": {"note_range": (36, 59), "n_fft": 8192},   # C2–B3
    "mid":    {"note_range": (60, 83),  "n_fft": 4096},    # C4–B6
    "high":   {"note_range": (84, 108), "n_fft": 2048},    # C7–C8: few partials, short window
}

# Feature extraction parameters
EXCITATION_DURATION_MS = 50
MFCC_N_MFCC = 13
MFCC_N_FFT = 2048

# Decay fitting
DECAY_SKIP_MS = 50         # Skip attack transient before fitting
DECAY_MIN_DURATION_S = 1.0 # Minimum duration after skip for valid fit

# Room IR extraction (Close → Ambient, NOT soundboard structural IR)
IR_REGULARIZATION = 1e-3
IR_NOTES = list(range(48, 72))  # C3–B4: representative mid-range

# Train/val/test split strategy
SPLIT_VELOCITY_TEST = "MezzoPiano"  # Entire 'mp' layer reserved for test (interpolation check)
SPLIT_VAL_RATIO = 0.1               # 10% of remaining notes for validation
SPLIT_SEED = 42

# Output
OUTPUT_DIR = Path("./data_output")
FEATURES_H5_PATH = OUTPUT_DIR / "features.h5"
MIDI_PAIRS_H5_PATH = OUTPUT_DIR / "midi_pairs.h5"
MANIFEST_PATH = OUTPUT_DIR / "manifest.parquet"


def get_n_fft_for_note(midi_note: int) -> int:
    return next(
        regime["n_fft"]
        for regime in N_FFT_REGIMES.values()
        if regime["note_range"][0] <= midi_note <= regime["note_range"][1]
    )


def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"PIANO_IN_162_SAMPLES = {PIANO_IN_162_SAMPLES}")
    print(f"SOURCE_SAMPLE_RATE = {SOURCE_SAMPLE_RATE}")
    print(f"SAMPLE_RATE_A = {SAMPLE_RATE_A}")
    print(f"SAMPLE_RATE_B = {SAMPLE_RATE_B}")
    print(f"N_FFT for A0 (MIDI 21) = {get_n_fft_for_note(21)}")
    print(f"N_FFT for C4 (MIDI 60) = {get_n_fft_for_note(60)}")
    print(f"N_FFT for C8 (MIDI 108) = {get_n_fft_for_note(108)}")