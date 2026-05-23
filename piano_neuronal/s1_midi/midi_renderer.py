"""MIDI → audio renderer using sfizz CLI.

Uses sfizz_render as the sole renderer. If sfizz is not available, the pipeline
stops — no fallback to a different renderer to avoid inconsistent targets.
The renderer used is recorded in the manifest.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional

import pretty_midi
import numpy as np
import soundfile as sf

from piano_neuronal.config import SFZ_CLOSE_PATH, SOURCE_SAMPLE_RATE


def check_sfizz_available() -> str:
    """Check that sfizz_render is available. Returns the command path or raises."""
    sfizz_path = shutil.which("sfizz_render")
    if sfizz_path is None:
        raise RuntimeError(
            "sfizz_render not found on PATH.\n"
            "Install sfizz CLI from https://sfz.tools/sfizz/downloads\n"
            "The MIDI rendering pipeline requires sfizz_render as the sole renderer.\n"
            "No fallback — mixing renderers produces inconsistent targets."
        )
    return sfizz_path


def render_midi_to_audio(
    midi_path: Path,
    sfz_path: Path,
    output_path: Path,
    sample_rate: int = SOURCE_SAMPLE_RATE,
    velocity_scale: float = 1.0,
) -> Path:
    """Render a MIDI file to WAV using sfizz CLI.

    Args:
        midi_path: Path to input MIDI file
        sfz_path: Path to SFZ instrument file
        output_path: Path for output WAV file
        sample_rate: Target sample rate (default: 44100)
        velocity_scale: Scale factor applied to MIDI velocities BEFORE rendering.
            1.0 = original, 0.7 = softer, 1.3 = louder.
            This scales the timbre, not just the gain.

    Returns:
        Path to the rendered WAV file.
    """
    sfizz_cmd = check_sfizz_available()

    if velocity_scale != 1.0:
        # Create a temporary MIDI with scaled velocities
        midi_path = _scale_midi_velocities(midi_path, velocity_scale)

    # Build sfizz_render command
    cmd = [
        str(sfizz_cmd),
        "--samplerate", str(sample_rate),
        "--sfz", str(sfz_path),
        "--midi", str(midi_path),
        "--output", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(
            f"sfizz_render failed (exit code {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(f"sfizz_render produced no output: {output_path}")

    return output_path


def _scale_midi_velocities(midi_path: Path, scale: float) -> Path:
    """Create a temporary MIDI file with all note velocities scaled.

    Velocity scaling is applied to MIDI note velocities BEFORE rendering,
    so the timbre changes naturally with velocity (as a real piano responds).
    This is NOT a gain change on the audio output.
    """
    import tempfile

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    for instrument in pm.instruments:
        for note in instrument.notes:
            note.velocity = max(1, min(127, int(note.velocity * scale)))

    # Write to temp file
    tmp_path = Path(tempfile.mktemp(suffix=".midi"))
    pm.write(str(tmp_path))
    return tmp_path


def get_renderer_info() -> dict:
    """Return information about the renderer for manifest recording."""
    try:
        sfizz_path = check_sfizz_available()
        result = subprocess.run([str(sfizz_path), "--version"], capture_output=True, text=True, timeout=10)
        version = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (RuntimeError, subprocess.TimeoutExpired):
        version = "not_available"

    return {
        "renderer": "sfizz_render",
        "version": version,
        "sample_rate": SOURCE_SAMPLE_RATE,
    }


if __name__ == "__main__":
    info = get_renderer_info()
    print(f"Renderer: {info['renderer']} v{info['version']}")
    print(f"Sample rate: {info['sample_rate']}")