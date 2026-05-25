"""CLI entry point for Sprint 2 DDSP-Piano baseline training.

Usage:
    # Local debug (3070 Ti, ~1 min):
    python -m piano_neuronal.s2_baseline.run_s2 --local-debug

    # Smoke test (RTX 4090 on Vast.ai, ~2 min):
    python -m piano_neuronal.s2_baseline.run_s2 --smoke-test

    # Full training (RTX 4090, ~10h):
    python -m piano_neuronal.s2_baseline.run_s2 --batch-size 32 --lr 0.003 --epochs 30

    # Resume interrupted training:
    python -m piano_neuronal.s2_baseline.run_s2 --resume checkpoints/ckpt_epoch5_step2500.pt
"""

import argparse
import logging
import sys

from piano_neuronal.s2_baseline.config import TrainConfig, S2_OUTPUT_DIR
from piano_neuronal.s2_baseline.train import train
from piano_neuronal.s2_baseline.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Sprint 2: DDSP-Piano baseline training")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32, try 64 on 48GB)")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate (default: 0.003)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max number of epochs (default: 30)")
    parser.add_argument("--steps-per-epoch", type=int, default=5000,
                        help="Steps per epoch (default: 5000)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear warmup steps (default: 500)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience in epochs (default: 8)")
    parser.add_argument("--ckpt-every-steps", type=int, default=500,
                        help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0,
                        help="Gradient clipping max norm (default: 5.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")

    # Modes
    parser.add_argument("--local-debug", action="store_true",
                        help="Local debug mode: batch=2, 3 steps + 1 val")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Smoke test: 5 steps + 1 val + 1 checkpoint (~2 min on RTX 4090)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile()")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint for resuming training")

    # Evaluation
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Run evaluation only (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for evaluation")

    args = parser.parse_args()

    # Ensure output directory exists before creating log file
    S2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(S2_OUTPUT_DIR / "train.log"), mode="a"),
        ],
    )

    if args.evaluate_only:
        from pathlib import Path
        ckpt_path = Path(args.checkpoint) if args.checkpoint else None
        results = evaluate(checkpoint_path=ckpt_path)
        print(f"\nEvaluation results: {results}")
        return

    # Build config from args
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        warmup_steps=args.warmup_steps,
        patience=args.patience,
        ckpt_every_steps=args.ckpt_every_steps,
        grad_clip_norm=args.grad_clip_norm,
        resume=args.resume,
        local_debug=args.local_debug,
        smoke_test=args.smoke_test,
        no_compile=args.no_compile,
        device=args.device,
    )

    # Adjust for debug/smoke modes
    if args.local_debug:
        config.batch_size = 2
        config.epochs = 1
        config.steps_per_epoch = 3
        config.warmup_steps = 1
        config.patience = 999  # no early stopping
        config.ckpt_every_steps = 2
        logging.info("LOCAL DEBUG MODE: batch=2, 3 steps + 1 val")

    if args.smoke_test:
        config.batch_size = 2
        config.epochs = 1
        config.steps_per_epoch = 5
        config.warmup_steps = 1
        config.patience = 999
        config.ckpt_every_steps = 3
        logging.info("SMOKE TEST MODE: 5 steps + 1 val + 1 ckpt")

    logging.info(f"Config: {config}")

    # Train
    train(config)

    # Evaluate best model
    if not args.local_debug and not args.smoke_test:
        logging.info("Running evaluation on best model...")
        evaluate()


if __name__ == "__main__":
    main()