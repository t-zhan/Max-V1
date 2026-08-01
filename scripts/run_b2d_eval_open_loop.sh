#!/bin/bash
set -euo pipefail

batch_size=28
n_samples=6000
ratio=0.33
split=val
inference_mode=autoregressive  # teacher_forcing / autoregressive
judge_cot=true

torchrun --nproc_per_node=${NPROC_PER_NODE} tools/b2dvl/eval_open_loop.py \
  --model-path "${B2D_OPEN_LOOP_CHECKPOINT}" \
  --split $split \
  --ratio $ratio \
  --n-samples $n_samples \
  --batch-size $batch_size \
  --inference-mode $inference_mode \
  --enable-thinking "${ENABLE_THINKING}" \
  --judge-cot $judge_cot \
  --deepseek-api-key "$DEEPSEEK_API_KEY" \
  --output ${SAVE_PATH}/$(date +%Y%m%d_%H%M%S)-open_loop-${split}-${inference_mode}.json
