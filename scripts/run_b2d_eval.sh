#!/bin/bash
set -euo pipefail

mkdir -p "$SAVE_PATH"

length=$(echo "$GPU_RANK_LIST" | wc -w)
python "$WORK_DIR/tools/split_xml.py" "${ROUTES_XML%.xml}" "$length" max b2d

i=0
for g in $GPU_RANK_LIST; do
    port=$((CARLA_BASE_PORT + i * 150))
    tm_port=$((CARLA_BASE_TM_PORT + i * 150))
    max_url="http://localhost:$((MAX_SERVER_BASE_PORT + i))"
    routes="${ROUTES_XML%.xml}_${i}_max_b2d.xml"

    echo "Task $i: GPU$g CARLA:$port MAX:$max_url"
    cd "$WORK_DIR"
    GPU_RANK=$g \
    CARLA_BASE_PORT=$port \
    CARLA_BASE_TM_PORT=$tm_port \
    MAX_SERVER_URL=$max_url \
    ROUTES_XML=$routes \
    PYTHONUNBUFFERED=1 \
    python "$WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py" \
        --routes="$routes" \
        --repetitions=1 --track=SENSORS \
        --checkpoint="$SAVE_PATH/checkpoint_${i}.json" \
        --agent="b2d_bridge" --agent-config="." --debug=0 --resume=True \
        --port="$port" --traffic-manager-port="$tm_port" --gpu-rank="$g" \
        --vlm-config="$PROJECT_ROOT/configs/max_config.json" \
        --timeout=100 \
        > "$SAVE_PATH/eval_${i}.log" 2>&1 &
    i=$((i + 1))
done
wait
