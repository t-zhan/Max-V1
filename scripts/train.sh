#!/bin/bash
# Max-V1 SFT training.
# Prerequisites: data prepared (scripts/setup.sh).
# Usage: ./scripts/train.sh [--background]
set -euo pipefail
set -a; source .env; set +a

RUN_BG=false
if [[ "${1:-}" == "--background" ]]; then
    RUN_BG=true
fi

TIMESTAMP=$(date +%y%m%d_%H%M%S)
TRAIN_CMD="swift sft \
    --external_plugins models/max_v1/register_max.py \
    --model pretrained/Qwen3.5-0.8B \
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
    --output_dir output \
    --add_non_thinking_prefix false \
    --remove_unused_columns false"

if $RUN_BG; then
    mkdir -p nohup_logs
    nohup $TRAIN_CMD > "nohup_logs/${TIMESTAMP}_run.log" 2>&1 &
    echo "Training launched (PID: $!)"
    echo "Monitor: tail -f nohup_logs/${TIMESTAMP}_run.log"
else
    $TRAIN_CMD
fi
