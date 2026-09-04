#!/bin/bash
# Max-V1 SFT training.
# Usage: ./scripts/sft.sh [--background]
set -euo pipefail

# Comment out a complete option line below to disable that argument.
TRAIN_CMD=(
    swift sft

    # Model
    --model "${MODEL_DIR}/${MODEL_NAME}"
    --model_type "${MODEL_TYPE}"
    --torch_dtype "${TORCH_DTYPE}"

    # Dataset and preprocessing
    --dataset ${DATASET}
    --max_length "${MAX_LENGTH}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --remove_unused_columns "${REMOVE_UNUSED_COLUMNS}"

    # Model tuning and freezing
    --tuner_type "${TUNER_TYPE}"
    --freeze_vit "${FREEZE_VIT}"
    --freeze_aligner false
    --target_modules ${TARGET_MODULES}
    --add_non_thinking_prefix true

    # Special tokens
    --new_special_tokens "${NEW_SPECIAL_TOKENS}"

    # Optimization
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --deepspeed "${DEEPSPEED}"

    # Learning rate
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    # --lr_scheduler_type reduce_lr_on_plateau
    # --lr_scheduler_kwargs '{"mode":"min","factor":0.8,"patience":20,"threshold":0.02,"threshold_mode":"rel","min_lr":1e-6}'
    # --metric_for_best_model uniad_l2_avg

    # Validation
    --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
    --eval_strategy "${EVAL_STRATEGY}"
    --eval_steps "${EVAL_STEPS}"    

    # Checkpoint saving
    --output_dir "${OUTPUT_DIR}"
    --save_strategy "${SAVE_STRATEGY}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"

    # Logging
    --logging_steps "${LOGGING_STEPS}"
    --report_to ${REPORT_TO}
    --swanlab_project "${SWANLAB_PROJECT_NAME}"

    # Plugins and callbacks
    --external_plugins ${EXTERNAL_PLUGINS}
    --callbacks ${CALLBACKS}
)

if [[ "${1:-}" == "--background" ]]; then
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    LOG_DIR=logs
    mkdir -p "${LOG_DIR}"
    nohup "${TRAIN_CMD[@]}" > "${LOG_DIR}/${TIMESTAMP}_run.log" 2>&1 &
    echo "Training launched (PID: $!)"
    echo "Monitor: tail -f ${LOG_DIR}/${TIMESTAMP}_run.log"
else
    "${TRAIN_CMD[@]}"
fi
