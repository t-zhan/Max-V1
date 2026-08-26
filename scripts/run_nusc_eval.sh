#!/bin/bash
set -euo pipefail

swanlab_project="Max-V1-nusc-eval"

for model_path in $(find "${NUSC_EVAL_CHECKPOINT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -Vr); do
    swanlab_name="$(basename "$(dirname "${model_path}")")/${model_path##*/}-train"

    torchrun --nproc_per_node="${NPROC_PER_NODE}" tools/nuscenes/test.py \
        --model-path "${model_path}" \
        --traj-file "${NUSC_EVAL_TRAJ_FILE}" \
        --info-pkl "${NUSC_EVAL_INFO_PKL}" \
        --seg-pkl "${NUSC_EVAL_SEG_PKL}" \
        --eval \
        --enable-thinking "${ENABLE_THINKING}" \
        --ego-status "${NUSC_EVAL_EGO_STATUS}" \
        --max-new-tokens "${NUSC_EVAL_MAX_NEW_TOKENS}" \
        --batch-size "${NUSC_EVAL_BS}" \
        --num-workers 0 \
        --swanlab-project "${swanlab_project}" \
        --swanlab-name "${swanlab_name}"
done
