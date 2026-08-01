#!/bin/bash
set -euo pipefail

for model_path in $(find "${NUSC_EVAL_CHECKPOINT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V); do
    torchrun --master-port="${NUSC_EVAL_MASTER_PORT}" \
        --nproc_per_node="${NPROC_PER_NODE}" tools/nuscenes/test.py \
        --model-path "${model_path}" \
        --eval \
        --enable-thinking "${NUSC_EVAL_ENABLE_THINKING}" \
        --batch-size "${NUSC_EVAL_BS}"
done
