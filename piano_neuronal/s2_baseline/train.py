"""DDSP-Piano baseline training loop.

Optimised for RTX 4090 (24 GB VRAM) on Vast.ai:
- bf16 mixed precision (native on Blackwell/Ada)
- batch_size=32 (configurable to 64)
- lr=0.003 with linear warmup over 500 steps
- torch.compile() with fallback
- Early stopping (patience=8 epochs)
- Checkpoint every 500 steps + auto-resume
- GPU utilisation logging every 50 steps
- --local-debug, --smoke-test modes
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from piano_neuronal.s2_baseline.config import (
    TrainConfig, S2_OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, STEPS_PER_EPOCH,
    WARMUP_STEPS, PATIENCE, CKPT_EVERY_STEPS, GRAD_CLIP_NORM,
    NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT,
    N_FRAMES, N_SYNTHS,
)
from piano_neuronal.s2_baseline.model import PianoModel
from piano_neuronal.s2_baseline.loss import HybridLoss
from piano_neuronal.s2_baseline.dataset import MidiPairsDataset, get_dataloader, collate_fn

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """Get compute device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def log_gpu_utilization(step: int) -> None:
    """Log GPU utilization percentage."""
    if torch.cuda.is_available():
        try:
            util = torch.cuda.utilization()
            if util < 85:
                logger.warning(
                    f"Step {step}: GPU utilization {util:.0f}% < 85% — "
                    f"possible dataloader bottleneck"
                )
            else:
                logger.info(f"Step {step}: GPU utilization {util:.0f}%")
        except Exception:
            pass  # pynvml not available on some systems


def train(config: TrainConfig) -> None:
    """Main training loop.

    Args:
        config: Training configuration (from CLI args or defaults).
    """
    # Setup
    device = get_device(config.device)
    logger.info(f"Training on device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model = PianoModel().to(device)

    # Initialize lazy modules with a dummy forward pass
    dummy_cond = torch.zeros(1, N_FRAMES, N_SYNTHS, 2, device=device)
    dummy_pedal = torch.zeros(1, N_FRAMES, 4, device=device)
    dummy_z = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy_cond, dummy_pedal, dummy_z)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # torch.compile with fallback — Triton can crash on some GPU/driver combos
    compiled = False
    if not config.no_compile:
        try:
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 64
            model = torch.compile(model)
            compiled = True
            logger.info("torch.compile() enabled (suppress_errors=True)")
        except Exception as e:
            logger.warning(f"torch.compile() init failed, continuing without: {e}")
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod

    # Verify compile works at runtime — Triton can crash even if init succeeds
    if compiled:
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                   enabled=device.type == "cuda"):
                _ = model(dummy_cond, dummy_pedal, dummy_z)
            logger.info("torch.compile() runtime OK")
        except Exception as e:
            logger.warning(f"torch.compile() runtime failed ({e}), reverting to eager")
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            compiled = False

    # Loss
    criterion = HybridLoss()

    # Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Learning rate warmup scheduler
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # Data
    logger.info("Loading training dataset...")
    train_dataset = MidiPairsDataset(split=TRAIN_SPLIT)
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info("Pre-resampling audio cache (fast on multi-core)...")
    train_dataset.preload_audio_cache(num_workers=min(16, os.cpu_count() or 4))
    logger.info("Loading validation dataset...")
    val_dataset = MidiPairsDataset(split=VAL_SPLIT)
    val_dataset.preload_audio_cache(num_workers=min(16, os.cpu_count() or 4))
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Use num_workers=0 on local-debug (Windows doesn't support multiprocessing well)
    num_workers = 0 if config.local_debug else NUM_WORKERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(LOG_DIR))

    # Resume from checkpoint
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    if config.resume:
        ckpt_path = Path(config.resume)
        if ckpt_path.exists():
            logger.info(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimiser.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt["global_step"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            start_epoch = ckpt.get("epoch", 0)
            logger.info(f"Resumed at step {global_step}, best_val_loss={best_val_loss:.4f}")

    # Local debug mode
    max_steps = config.steps_per_epoch if not config.local_debug else 3
    max_epochs = 1 if config.local_debug else config.epochs
    if config.smoke_test:
        max_steps = 5
        max_epochs = 1

    # Training
    logger.info(f"Starting training: {max_epochs} epochs, {max_steps} steps/epoch, batch={config.batch_size}")
    logger.info(f"lr={config.lr}, warmup={config.warmup_steps}, patience={config.patience}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        step_times = []

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            if step >= max_steps:
                break

            t0 = time.time()

            audio, conditioning, pedal, polyphony = batch
            audio = audio.to(device)
            conditioning = conditioning.to(device)
            pedal = pedal.to(device)
            # piano_model is always 0 for baseline (n_instruments=1)
            piano_model = torch.zeros(audio.shape[0], dtype=torch.long, device=device)

            # Forward pass with bf16 mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                signal, reverb_ir, non_ir_signal = model(conditioning, pedal, piano_model)

                # Crop to same length
                min_len = min(signal.shape[-1], audio.shape[-1])
                signal = signal[..., :min_len]
                audio = audio[..., :min_len]

                # Get inharm_coef from model for loss computation
                ext_cond = model.note_release(conditioning)
                ext_pitch = ext_cond[..., 0]
                z, global_inharm, global_detuning = model.z_encoder(piano_model)
                inharm_coef = model.inharm_model(ext_pitch, global_inharm)

                total_loss, loss_dict = criterion(signal, audio, reverb_ir, inharm_coef)

            # Backward + clip + step
            optimiser.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimiser.step()
            scheduler.step()

            global_step += 1
            epoch_loss += total_loss.item()
            step_time = time.time() - t0
            step_times.append(step_time)

            # NaN check
            if torch.isnan(total_loss):
                logger.error(f"Step {global_step}: NaN loss detected!")
                continue

            # Logging — every step in debug/smoke, every 50 steps otherwise
            log_interval = 1 if (config.local_debug or config.smoke_test) else 50
            if global_step % log_interval == 0:
                avg_step_time = sum(step_times[-50:]) / len(step_times[-50:])
                steps_per_sec = 1.0 / avg_step_time
                epoch_remaining = (max_steps - step) * avg_step_time / 60.0
                logger.info(
                    f"Epoch {epoch+1} Step {global_step} | "
                    f"loss={total_loss.item():.4f} "
                    f"stft={loss_dict['mr_stft']:.4f} "
                    f"rev={loss_dict['reverb_reg']:.4f} "
                    f"inh={loss_dict['inharm']:.4f} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"epoch ETA: {epoch_remaining:.1f}min"
                )
                writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                writer.add_scalar("train/mr_stft", loss_dict["mr_stft"], global_step)
                writer.add_scalar("train/reverb_reg", loss_dict["reverb_reg"], global_step)
                writer.add_scalar("train/inharm", loss_dict["inharm"], global_step)
                writer.add_scalar("train/lr", optimiser.param_groups[0]["lr"], global_step)

                log_gpu_utilization(global_step)

            # Checkpoint every N steps
            if global_step % config.ckpt_every_steps == 0:
                ckpt_path = CHECKPOINT_DIR / f"ckpt_epoch{epoch+1}_step{global_step}.pt"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                audio, conditioning, pedal, polyphony = batch
                audio = audio.to(device)
                conditioning = conditioning.to(device)
                pedal = pedal.to(device)
                piano_model = torch.zeros(audio.shape[0], dtype=torch.long, device=device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    signal, reverb_ir, non_ir_signal = model(conditioning, pedal, piano_model)
                    min_len = min(signal.shape[-1], audio.shape[-1])
                    signal = signal[..., :min_len]
                    audio = audio[..., :min_len]
                    ext_cond = model.note_release(conditioning)
                    ext_pitch = ext_cond[..., 0]
                    z, global_inharm, global_detuning = model.z_encoder(piano_model)
                    inharm_coef = model.inharm_model(ext_pitch, global_inharm)
                    total_loss, loss_dict = criterion(signal, audio, reverb_ir, inharm_coef)

                val_loss += total_loss.item()
                val_steps += 1

                if config.local_debug or config.smoke_test:
                    break

        avg_val_loss = val_loss / max(val_steps, 1)
        logger.info(f"Epoch {epoch+1} | Val loss: {avg_val_loss:.4f}")
        writer.add_scalar("val/total_loss", avg_val_loss, global_step)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_path)
            logger.info(f"New best model saved: val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")