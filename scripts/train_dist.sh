#!/bin/bash
# Max-V1 SFT training.
# Prerequisites: data prepared (scripts/setup.sh).
# Usage: ./scripts/train.sh [--background]
set -euo pipefail

export NNODES=$WORLD_SIZE
export NODE_RANK=$RANK

TRAIN_CMD="swift sft \
    --external_plugins models/max_v1/register_max.py \
    --model ${MODEL_DIR}/${MODEL_NAME} \
    --model_type max_qwen3_5 \
    --tuner_type full \
    --freeze_vit false \
    --dataset data/sft/max_sft_train.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --report_to swanlab \
    --save_strategy epoch \
    --save_total_limit 10 \
    --logging_steps 100 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 32 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --max_length 65536 \
    --output_dir outputs \
    --add_non_thinking_prefix false \
    --remove_unused_columns false"

if [[ "${1:-}" == "--background" ]]; then
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    LOG_DIR=logs
    mkdir -p ${LOG_DIR}
    nohup $TRAIN_CMD > "${LOG_DIR}/${TIMESTAMP}_run.log" 2>&1 &
    echo "Training launched (PID: $!)"
    echo "Monitor: tail -f ${LOG_DIR}/${TIMESTAMP}_run.log"
else
    $TRAIN_CMD
fi
