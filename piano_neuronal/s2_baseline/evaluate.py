"""Evaluation on test set: MR-STFT loss per resolution + audio generation.

Loads the best model checkpoint and computes MR-STFT loss on the test set.
Saves generated audio samples for perceptual evaluation.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict

from piano_neuronal.s2_baseline.config import (
    S2_OUTPUT_DIR, EVAL_DIR, CHECKPOINT_DIR, SAMPLE_RATE, TEST_SPLIT, BATCH_SIZE
)
from piano_neuronal.s2_baseline.model import PianoModel
from piano_neuronal.s2_baseline.loss import MSSLoss, MR_STFT_N_FFTS
from piano_neuronal.s2_baseline.dataset import get_dataloader
from piano_neuronal.s2_baseline.train import get_device


def evaluate(
    checkpoint_path: Path = None,
    output_dir: Path = EVAL_DIR,
    n_samples: int = 10,
) -> Dict[str, float]:
    """Evaluate model on test set and save audio samples.

    Args:
        checkpoint_path: path to best_model.pt. If None, uses default.
        output_dir: directory for evaluation outputs.
        n_samples: number of audio samples to save for listening.

    Returns:
        Dict of MR-STFT losses per resolution.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PianoModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Data
    test_loader = get_dataloader(split=TEST_SPLIT, batch_size=1, shuffle=False)

    # Loss per resolution
    per_resolution_losses = {f"stft_{n}": [] for n in MR_STFT_N_FFTS}
    total_losses = []

    print(f"Evaluating on test set ({len(test_loader.dataset)} samples)...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                continue

            audio, conditioning, pedal, polyphony = batch
            audio = audio.to(device)
            conditioning = conditioning.to(device)
            pedal = pedal.to(device)
            piano_model = torch.zeros(audio.shape[0], dtype=torch.long, device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                signal, reverb_ir, _ = model(conditioning, pedal, piano_model)

                # Crop
                min_len = min(signal.shape[-1], audio.shape[-1])
                signal = signal[..., :min_len]
                audio = audio[..., :min_len]

                # Per-resolution loss
                for n_fft in MR_STFT_N_FFTS:
                    from piano_neuronal.s2_baseline.loss import SSSLoss
                    ssl = SSSLoss(n_fft=n_fft).to(device)
                    loss = ssl(signal, audio)
                    per_resolution_losses[f"stft_{n_fft}"].append(loss.item())

            # Save audio samples
            if i < n_samples:
                audio_np = signal[0].cpu().numpy()
                target_np = audio[0].cpu().numpy()
                sf.write(str(output_dir / f"sample_{i:02d}_pred.wav"), audio_np, SAMPLE_RATE)
                sf.write(str(output_dir / f"sample_{i:02d}_target.wav"), target_np, SAMPLE_RATE)

    # Aggregate results
    results = {}
    for key, losses in per_resolution_losses.items():
        mean_loss = np.mean(losses) if losses else float("inf")
        results[key] = mean_loss
        print(f"  {key}: {mean_loss:.6f}")

    results["total_mr_stft"] = sum(results.values())
    print(f"  Total MR-STFT: {results['total_mr_stft']:.6f}")

    # Save results
    import json
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'eval_results.json'}")
    print(f"Audio samples saved to {output_dir}/")

    return results