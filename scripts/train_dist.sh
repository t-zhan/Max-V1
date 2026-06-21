#!/bin/bash
# Max-V1 SFT training.
# Prerequisites: data prepared (scripts/setup.sh).
# Usage: ./scripts/train.sh [--background]
set -euo pipefail

TRAIN_CMD="swift sft \
    --external_plugins models/max_v1/register_max.py \
    --model ${MODEL_DIR}/${MODEL_NAME} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --model_type max_qwen3_5 \
    --tuner_type full \
    --freeze_vit false \
    --dataset data/sft/max_sft_train.json \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --report_to swanlab \
    --save_strategy epoch \
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
