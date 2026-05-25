"""MIDI event encoding for DDSP-Piano conditioning.

Converts structured MIDI event arrays from Sprint 1 HDF5 into the tensor
format expected by the DDSP-Piano model:

  - conditioning: (n_frames, n_synths, 2) — [active_pitch/128, velocity/128]
  - pedal:        (n_frames, 4)             — sustain, sostenuto, soft, hold
  - polyphony:    (n_frames,)               — number of active notes per frame

Also filters out segments where polyphony exceeds n_synths (16 by default).
"""

import numpy as np
import pretty_midi
from typing import Tuple, Optional

from piano_neuronal.s2_baseline.config import (
    FRAME_RATE, N_SYNTHS, DURATION_S, N_FRAMES, MIDI_NOTE_MIN, MIDI_NOTE_MAX
)


def midi_events_to_piano_roll(
    midi_events: np.ndarray,
    duration_s: float = DURATION_S,
    frame_rate: int = FRAME_RATE,
    n_frames: int = N_FRAMES,
    sustain_pedal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert structured MIDI events to piano roll.

    Args:
        midi_events: structured array with fields (start, end, pitch, velocity, program).
                     As stored in midi_pairs.h5.
        duration_s: segment duration in seconds.
        frame_rate: frames per second (250 Hz).
        n_frames: number of output frames.
        sustain_pedal: whether to extend notes through sustain pedal (CC64).
                       For our sfizz-rendered audio, sustain is already baked in.

    Returns:
        piano_roll: (n_frames, 88) — note activity (0 or velocity/127).
        onset_roll: (n_frames, 88) — note onsets (velocity/127 for onset frame, 0 otherwise).
    """
    n_pitches = MIDI_NOTE_MAX - MIDI_NOTE_MIN + 1  # 88
    piano_roll = np.zeros((n_frames, n_pitches), dtype=np.float32)
    onset_roll = np.zeros((n_frames, n_pitches), dtype=np.float32)

    for event in midi_events:
        pitch = int(event["pitch"])
        velocity = int(event["velocity"])
        start_s = float(event["start"])
        end_s = float(event["end"])

        # Skip notes outside our range
        if pitch < MIDI_NOTE_MIN or pitch > MIDI_NOTE_MAX:
            continue

        pitch_idx = pitch - MIDI_NOTE_MIN
        start_frame = int(round(start_s * frame_rate))
        end_frame = min(int(round(end_s * frame_rate)), n_frames)

        # Set onset
        if start_frame < n_frames:
            onset_roll[start_frame, pitch_idx] = velocity / 127.0

        # Set sustained region
        for f in range(max(0, start_frame), end_frame):
            piano_roll[f, pitch_idx] = velocity / 127.0

    return piano_roll, onset_roll


def piano_roll_to_conditioning(
    piano_roll: np.ndarray,
    onset_roll: np.ndarray,
    n_synths: int = N_SYNTHS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert piano roll to DDSP-Piano conditioning format.

    Uses a voice-allocation algorithm that maintains temporal coherence:
    each active note is assigned to a voice slot, and a note stays on the
    same slot across its entire duration.

    Args:
        piano_roll: (n_frames, 88) — note activity (velocity-normalized).
        onset_roll: (n_frames, 88) — note onsets (velocity-normalized).
        n_synths: max polyphony (number of voice slots).

    Returns:
        conditioning: (n_frames, n_synths, 2) — [pitch/128, velocity/128] per voice.
        polyphony: (n_frames,) — number of active voices per frame.
    """
    n_frames = piano_roll.shape[0]
    n_pitches = piano_roll.shape[1]

    conditioning = np.zeros((n_frames, n_synths, 2), dtype=np.float32)
    polyphony = np.zeros(n_frames, dtype=np.int32)

    # Track which voice slot each (pitch) is assigned to
    pitch_to_slot = {}
    # Track which slots are free
    free_slots = list(range(n_synths))

    for f in range(n_frames):
        # Find active pitches this frame
        active = np.where(piano_roll[f] > 0)[0]

        # Find newly onset pitches
        onsets = np.where(onset_roll[f] > 0)[0]

        # Assign new onsets to voice slots
        for pitch_idx in onsets:
            if pitch_idx in pitch_to_slot:
                continue  # already assigned

            if free_slots:
                slot = free_slots.pop(0)
                pitch_to_slot[pitch_idx] = slot
            # If no free slots, the note is dropped (polyphony exceeded)

        # Fill conditioning for active notes
        new_pitch_to_slot = {}
        for pitch_idx in active:
            if pitch_idx in pitch_to_slot:
                slot = pitch_to_slot[pitch_idx]
                # pitch_idx is 0-indexed from MIDI_NOTE_MIN
                midi_pitch = pitch_idx + MIDI_NOTE_MIN
                conditioning[f, slot, 0] = midi_pitch / 128.0  # normalised pitch
                conditioning[f, slot, 1] = piano_roll[f, pitch_idx]  # velocity
                new_pitch_to_slot[pitch_idx] = slot

        # Release slots for notes that are no longer active
        released = set(pitch_to_slot.keys()) - set(active.tolist())
        for pitch_idx in released:
            slot = pitch_to_slot[pitch_idx]
            free_slots.append(slot)
            free_slots.sort()

        pitch_to_slot = new_pitch_to_slot
        polyphony[f] = len(active)

    return conditioning, polyphony


def encode_midi_events(
    midi_events: np.ndarray,
    duration_s: float = DURATION_S,
    frame_rate: int = FRAME_RATE,
    n_frames: int = N_FRAMES,
    n_synths: int = N_SYNTHS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full encoding pipeline: MIDI events → conditioning + polyphony.

    Args:
        midi_events: structured array from midi_pairs.h5.
        duration_s: segment duration.
        frame_rate: conditioning frame rate.
        n_frames: number of output frames.
        n_synths: max polyphony.

    Returns:
        conditioning: (n_frames, n_synths, 2) float32.
        pedal: (n_frames, 4) float32 — zeros (pedal not extracted from sfizz).
        polyphony: (n_frames,) int32 — voice count per frame.
    """
    piano_roll, onset_roll = midi_events_to_piano_roll(
        midi_events, duration_s, frame_rate, n_frames
    )
    conditioning, polyphony = piano_roll_to_conditioning(piano_roll, onset_roll, n_synths)

    # Pedal: not extracted from our sfizz-rendered audio, so all zeros.
    # DDSP-Piano expects 4 pedal channels (sustain, sostenuto, soft, hold).
    pedal = np.zeros((n_frames, 4), dtype=np.float32)

    return conditioning, pedal, polyphony


def check_polyphony_limit(polyphony: np.ndarray, max_polyphony: int = N_SYNTHS) -> bool:
    """Return True if the segment is within polyphony limits."""
    return int(polyphony.max()) <= max_polyphony