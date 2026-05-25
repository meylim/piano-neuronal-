#!/bin/bash
# Vast.ai instance setup script for Sprint 2 DDSP-Piano training.
#
# Template: PyTorch (Vast) — CUDA 12.9
# GPU: RTX 4090 (24 GB VRAM, ~$0.82/hr)
#
# Usage:
#   1. Create instance on Vast.ai with PyTorch template
#   2. SSH into instance
#   3. Run: bash scripts/setup_vast.sh
#   4. Run smoke test: python -m piano_neuronal.s2_baseline.run_s2 --smoke-test
#   5. If smoke test passes, run: python -m piano_neuronal.s2_baseline.run_s2

set -e

echo "=== Sprint 2 Vast.ai Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Install dependencies
echo "[1/4] Installing Python dependencies..."
pip install -q torchaudio soundfile h5py pretty_midi tqdm pandas pyarrow tensorboard

# Verify imports
echo "[2/4] Verifying imports..."
python -c "
import torch; print(f'  PyTorch {torch.__version__}')
import torchaudio; print(f'  torchaudio {torchaudio.__version__}')
import h5py; print(f'  h5py {h5py.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  bf16 supported: {torch.cuda.is_bf16_supported()}')
"

# Upload/verify data
echo "[3/4] Checking data..."
echo "  You need to upload:"
echo "    - data_output/midi_pairs.h5 (~125 GB)"
echo "    - data_output/manifest.parquet (~400 KB)"
echo ""
echo "  Option A: rsync from local machine:"
echo "    rsync -avz --progress data_output/midi_pairs.h5 root@<IP>:/workspace/data_output/"
echo "    rsync -avz data_output/manifest.parquet root@<IP>:/workspace/data_output/"
echo ""
echo "  Option B: Download from cloud storage (S3/GCS):"
echo "    # Add your download commands here"
echo ""

# Verify data exists
if [ -f "data_output/midi_pairs.h5" ]; then
    SIZE=$(du -h data_output/midi_pairs.h5 | cut -f1)
    echo "  midi_pairs.h5 found (${SIZE})"
else
    echo "  WARNING: midi_pairs.h5 NOT found — upload required before training"
fi

if [ -f "data_output/manifest.parquet" ]; then
    echo "  manifest.parquet found"
else
    echo "  WARNING: manifest.parquet NOT found — upload required before training"
fi

# Create output directories
echo "[4/4] Creating output directories..."
mkdir -p data_output/s2_baseline/checkpoints
mkdir -p data_output/s2_baseline/logs
mkdir -p data_output/s2_baseline/cache
mkdir -p data_output/s2_baseline/eval

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload midi_pairs.h5 if not done"
echo "  2. Smoke test:    python -m piano_neuronal.s2_baseline.run_s2 --smoke-test"
echo "  3. Full training: python -m piano_neuronal.s2_baseline.run_s2 --batch-size 32 --lr 0.003 --epochs 30"
echo ""
echo "Estimated cost: ~$15 for 30 epochs on RTX 4090 at \$0.82/hr"