#!/bin/bash
# Max-V1 data preparation.
# Prerequisites: activated venv with dependencies installed, HF_TOKEN set.
# Run from project root directory.
set -euo pipefail
set -a; source .env; set +a

# --- Download Qwen3.5-0.8B base model ---
if [ ! -f "pretrained/Qwen3.5-0.8B/config.json" ]; then
    echo "Downloading Qwen3.5-0.8B base model..."
    hf download Qwen/Qwen3.5-0.8B --local-dir pretrained/Qwen3.5-0.8B
else
    echo "Base model ready: pretrained/Qwen3.5-0.8B"
fi

# --- Prepare SFT dataset ---
if [ ! -f "data/sft/max_sft_train.json" ]; then
    echo "Preparing Max SFT training data..."
    python tools/prepare_data.py --data-dir data
else
    echo "Training data ready: data/sft/max_sft_train.json"
fi

echo ""
echo "Setup complete. Run: bash scripts/train.sh"
