#!/bin/bash
# Max-V1 data preparation.
# Prerequisites: activated venv with dependencies installed, HF_TOKEN set.
# Run from project root directory.
set -euo pipefail

# --- Download base model ---
if [ ! -f "${MODEL_DIR}/${MODEL_NAME}/config.json" ]; then
    echo "Downloading base model..."
    hf download ${MODEL_NAME} --local-dir ${MODEL_DIR}/${MODEL_NAME}
else
    echo "Base model ready: ${MODEL_DIR}/${MODEL_NAME}"
fi

# --- Prepare SFT dataset ---
if [ ! -f "data/sft/max_sft_train.json" ]; then
    echo "Preparing Max SFT training data..."
    python tools/prepare_data.py --vqa-dir data/Bench2Drive-VL-base --raw-dir data/Bench2Drive --out data/sft/max_sft_train.json
else
    echo "Training data ready: data/sft/max_sft_train.json"
fi

echo ""
echo "Setup complete. Run: bash scripts/train.sh"
