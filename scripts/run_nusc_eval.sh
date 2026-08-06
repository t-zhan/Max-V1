#!/bin/bash
set -euo pipefail

for model_path in $(find "${NUSC_EVAL_CHECKPOINT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V); do
    torchrun --nproc_per_node="${NPROC_PER_NODE}" tools/nuscenes/test.py \
        --model-path "${model_path}" \
        --traj-file "${NUSC_EVAL_TRAJ_FILE}" \
        --info-pkl "${NUSC_EVAL_INFO_PKL}" \
        --seg-pkl "${NUSC_EVAL_SEG_PKL}" \
        --eval \
        --enable-thinking "${ENABLE_THINKING}" \
        --ego-status "${NUSC_EVAL_EGO_STATUS}" \
        --max-new-tokens "${NUSC_EVAL_MAX_NEW_TOKENS}" \
        --batch-size "${NUSC_EVAL_BS}"
done
