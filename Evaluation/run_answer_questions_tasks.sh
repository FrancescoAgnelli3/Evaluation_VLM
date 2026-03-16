#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${ROOT_DIR}/answer_questions.py"
IDLE_SECONDS_REQUIRED=200
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

BASE_MODELS=(
  "cosmos2-2B"
  "cosmos2-8B"
  # "qwen-8B"
  # "qwen-2B"
  # "qwen-32B"
)

TASKS=(
  "people"
  "environment"
  "industry"
)

for task in "${TASKS[@]}"; do
  case "$task" in
    (people) lora_suffix="people" ;;
    (environment) lora_suffix="env" ;;
    (industry) lora_suffix="industry" ;;
    (*) echo "Unknown task: ${task}" >&2; exit 1 ;;
  esac

  python3 "$SCRIPT" --task "$task" --model "cosmos2-reason-LoRAFT-${lora_suffix}-2B"
  python3 "$SCRIPT" --task "$task" --model "cosmos2-reason-LoRAFT-${lora_suffix}-8B"
  
  # for model in "${BASE_MODELS[@]}"; do
  #   python3 "$SCRIPT" --task "$task" --model "$model"
  # done
done
