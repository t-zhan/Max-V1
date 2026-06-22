#!/bin/bash
# Max-V1 SFT training.
# Prerequisites: data prepared (scripts/setup.sh).
# Usage: ./scripts/sft.sh [--background]
set -euo pipefail

TRAIN_CMD="swift sft \
    --model ${MODEL_DIR}/${MODEL_NAME} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --dataset ${DATASET} \
    --report_to ${REPORT_TO} \
    --deepspeed ${DEEPSPEED} \
    --max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --tuner_type ${TUNER_TYPE} \
    --freeze_vit ${FREEZE_VIT} \
    --learning_rate ${LEARNING_RATE} \
    --target_modules ${TARGET_MODULES} \
    --save_strategy ${SAVE_STRATEGY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --dataset_num_proc ${DATASET_NUM_PROC} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --add_non_thinking_prefix ${ADD_NON_THINKING_PREFIX} \
    --remove_unused_columns ${REMOVE_UNUSED_COLUMNS} \
    --external_plugins ${EXTERNAL_PLUGINS}"

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
