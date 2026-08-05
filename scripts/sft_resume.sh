#!/bin/bash
# Max-V1 SFT scheduled-sampling training from checkpoint weights.
# Required env: SCHEDULED_SAMPLING_RATIO in [0, 1], ROLLOUT_USE_CACHE=true|false.
# Usage: ./scripts/sft_resume.sh [--background]
set -euo pipefail

# Resume the configured SwanLab run.
export SWANLAB_RESUME="must"
export SFT_RESUME_CHECKPOINT=outputs/v53-20260803-183356/checkpoint-666
export SWANLAB_RUN_ID=pxvgxu1n

RUN_OUTPUT_DIR="$(dirname "${SFT_RESUME_CHECKPOINT}")"

# NUM_TRAIN_EPOCHS=20
# LEARNING_RATE=1e-6
# MAX_GRAD_NORM=50.0

# SAVE_STRATEGY=steps
# SAVE_STEPS=100

# WARMUP_STEPS=10
# export WARMUP_SCHEDULED_SAMPLING_STEPS=200
# export WARMUP_SCHEDULED_SAMPLING_RATIO=0.5

# EXTERNAL_PLUGINS="models/max_v1/register_max.py models/max_v1/max_callback.py"
# CALLBACKS="max_loss_log"  # "max_loss_log max_rollout_schedule"

TRAIN_CMD=(
    swift sft

    # Model
    --model "${SFT_RESUME_CHECKPOINT}"
    --resume_from_checkpoint "${SFT_RESUME_CHECKPOINT}"
    --model_type "${MODEL_TYPE}"

    # Dataset and preprocessing
    --dataset ${DATASET}
    --max_length "${MAX_LENGTH}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --remove_unused_columns "${REMOVE_UNUSED_COLUMNS}"

    # Model tuning and freezing
    --tuner_type "${TUNER_TYPE}"
    --freeze_vit "${FREEZE_VIT}"
    --freeze_llm "${FREEZE_LLM}"
    --target_modules ${TARGET_MODULES}
    --add_non_thinking_prefix false

    # Optimization
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    # --max_grad_norm "${MAX_GRAD_NORM}"
    --deepspeed "${DEEPSPEED}"

    # Checkpoint saving
    --output_dir "${RUN_OUTPUT_DIR}"
    --add_version false
    --save_strategy "${SAVE_STRATEGY}"
    # --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"

    # Logging
    --logging_steps "${LOGGING_STEPS}"
    --report_to ${REPORT_TO}
    --swanlab_project "${SWANLAB_PROJECT_NAME}"
    --swanlab_exp_name "${RUN_OUTPUT_DIR}"

    # Plugins and callbacks
    --external_plugins ${EXTERNAL_PLUGINS}
    --callbacks ${CALLBACKS}
)

if [[ "${1:-}" == "--background" ]]; then
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    LOG_DIR=logs
    mkdir -p "${LOG_DIR}"
    nohup "${TRAIN_CMD[@]}" > "${LOG_DIR}/${TIMESTAMP}_sft_ss.log" 2>&1 &
    echo "Training launched (PID: $!)"
    echo "Monitor: tail -f ${LOG_DIR}/${TIMESTAMP}_sft_ss.log"
else
    "${TRAIN_CMD[@]}"
fi
