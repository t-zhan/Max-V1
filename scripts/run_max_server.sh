#!/bin/bash
set -euo pipefail

for g in $(echo "$GPU_RANK_LIST" | tr ' ' '\n' | sort -nu); do
    port=$((MAX_SERVER_BASE_PORT + g))
    echo "GPU$g → port $port"
    CUDA_VISIBLE_DEVICES=$g python "$PROJECT_ROOT/b2d_bridge/max_server.py" \
        --model-path "$MAX_MODEL_PATH" --port "$port" &
done
wait
