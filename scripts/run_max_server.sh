#!/bin/bash
set -euo pipefail

i=0
for g in $GPU_RANK_LIST; do
    port=$((MAX_SERVER_BASE_PORT + i))
    echo "Task $i GPU$g → port $port"
    CUDA_VISIBLE_DEVICES=$g python "$PROJECT_ROOT/b2d_bridge/max_server.py" \
        --model-path "$MAX_MODEL_PATH" --port "$port" \
        --enable-thinking "${ENABLE_THINKING}" &
    i=$((i + 1))
done
wait
