#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/fa/projects/Evaluation_VLM/Evaluation/answer_questions.py"
IDLE_SECONDS_REQUIRED=0
POLL_INTERVAL=1

is_all_gpus_free() {
  local lines
  lines=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  while IFS=',' read -r util mem; do
    util=$(echo "$util" | xargs)
    mem=$(echo "$mem" | xargs)
    if [[ "$util" != "0" ]] || [[ "$mem" != "0" ]]; then
      return 1
    fi
  done <<< "$lines"
  return 0
}

idle_for=0
echo "Waiting for all GPUs to be free for ${IDLE_SECONDS_REQUIRED}s..."

while true; do
  if is_all_gpus_free; then
    idle_for=$((idle_for + POLL_INTERVAL))
    if (( idle_for >= IDLE_SECONDS_REQUIRED )); then
      break
    fi
  else
    idle_for=0
  fi
  sleep "$POLL_INTERVAL"
done

echo "GPUs free for ${IDLE_SECONDS_REQUIRED}s. Launching..."
python "$SCRIPT" \
  --model qwen3.5-FT_40k-27B \
  --task road \
