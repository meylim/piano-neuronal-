"""Sprint 2 configuration — DDSP-Piano baseline reproduction at 16 kHz.

Optimised for RTX 5090 (32 GB VRAM, Blackwell) or RTX 4090 (24 GB VRAM) on Vast.ai.
Batch size and lr are CLI-configurable for quick iteration.
Default batch=32 fits both cards; batch=64 possible on 32 GB VRAM.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Reuse Sprint 1 paths
from piano_neuronal.config import OUTPUT_DIR, MIDI_PAIRS_H5_PATH, MANIFEST_PATH

# ── Audio ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000       # DDSP-Piano baseline (JAES 2023)
FRAME_RATE = 250          # Hz, conditioning frame rate
DURATION_S = 3.0          # seconds per training segment
N_SAMPLES = int(DURATION_S * SAMPLE_RATE)  # 48000
N_FRAMES = int(DURATION_S * FRAME_RATE)     # 750

# ── Architecture (DDSP-Piano v2 / DAFx22) ───────────────────────────────
N_SYNTHS = 16             # max polyphony (voice slots)
N_HARMONICS = 96           # harmonic partials per voice
N_NOISE_BANDS = 64         # filtered noise magnitude bins
N_SUBSTRINGS = 2           # unison strings per note (2-3 strings)
Z_DIM = 16                 # instrument embedding dimension
N_INSTRUMENTS = 1          # single piano (sfizz / Piano in 162)
MIDI_NOTE_MIN = 21         # A0
MIDI_NOTE_MAX = 108        # C8
REVERB_IR_DURATION_S = 1.5
REVERB_IR_LENGTH = int(REVERB_IR_DURATION_S * SAMPLE_RATE)  # 24000

# ── Network sizes (from gin config) ─────────────────────────────────────
CONTEXT_DENSE_UNITS = 32
CONTEXT_GRU_UNITS = 64
MONO_INPUT_DIM = 35       # pitch(1) + conditioning(2*n_synths) + context(z_dim)
MONO_GRU_UNITS = 192
MONO_DENSE_UNITS = 192     # hidden dim for output stack

# ── Inharmonicity (parametric B model) ──────────────────────────────────
# Physics-based initial values from Rigaud et al. DAFx-2011
INHARM_TREBLE_SLOPE_INIT = 9.26e-2
INHARM_BASS_SLOPE_INIT = -8.47e-2

# ── Training defaults (overridable via CLI) ─────────────────────────────
BATCH_SIZE = 32           # fits 48 GB VRAM with bf16; try 64 if memory allows
LEARNING_RATE = 0.003      # scaled up from 0.001 (batch 5× larger)
WARMUP_STEPS = 500        # linear warmup before full lr
EPOCHS = 30               # early stopping target (patience 8)
STEPS_PER_EPOCH = 5000
PATIENCE = 8              # early stopping patience (epochs)
CKPT_EVERY_STEPS = 500   # checkpoint frequency
GRAD_CLIP_NORM = 5.0      # gradient clipping

# ── Loss weights ─────────────────────────────────────────────────────────
MR_STFT_N_FFTS: List[int] = [2048, 1024, 512, 256, 128, 64]
MR_STFT_MAG_WEIGHT = 1.0
MR_STFT_LOGMAG_WEIGHT = 1.0
REVERB_REG_WEIGHT = 0.01
INHARM_LOSS_WEIGHT = 10.0

# ── Data pipeline ───────────────────────────────────────────────────────
SOURCE_SAMPLE_RATE = 44100  # Sprint 1 audio rate
SEGMENT_OVERLAP = 0.5      # 50% overlap for training segments
MAX_POLYPHONY = 16          # filter out segments exceeding this
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
NUM_WORKERS = 12            # DataLoader workers (Vast.ai instance)
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ── Paths ────────────────────────────────────────────────────────────────
S2_OUTPUT_DIR = OUTPUT_DIR / "s2_baseline"
CHECKPOINT_DIR = S2_OUTPUT_DIR / "checkpoints"
LOG_DIR = S2_OUTPUT_DIR / "logs"
CACHE_DIR = S2_OUTPUT_DIR / "cache"  # precomputed conditioning tensors
EVAL_DIR = S2_OUTPUT_DIR / "eval"


@dataclass
class TrainConfig:
    """Mutable training config — override via CLI."""
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    epochs: int = EPOCHS
    steps_per_epoch: int = STEPS_PER_EPOCH
    warmup_steps: int = WARMUP_STEPS
    patience: int = PATIENCE
    ckpt_every_steps: int = CKPT_EVERY_STEPS
    grad_clip_norm: float = GRAD_CLIP_NORM
    resume: str = ""              # path to checkpoint for resumption
    local_debug: bool = False     # 3 steps + 1 val, batch=2
    smoke_test: bool = False      # 5 steps + 1 val + 1 ckpt
    no_compile: bool = False      # disable torch.compile
    device: str = "auto"          # "auto", "cuda", "cpu"