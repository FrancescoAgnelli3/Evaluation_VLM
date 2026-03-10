#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${ROOT_DIR}/answer_questions.py"

BASE_MODELS=(
  "cosmos2-2B"
  "cosmos2-8B"
  # "qwen-8B"
  # "qwen-2B"
  # "qwen-32B"
)

TASKS=(
  "people"
  # "environment"
  "industry"
)

for task in "${TASKS[@]}"; do
  case "$task" in
    (people) lora_suffix="people" ;;
    # (environment) lora_suffix="env" ;;
    (industry) lora_suffix="industry" ;;
    (*) echo "Unknown task: ${task}" >&2; exit 1 ;;
  esac

  # python3 "$SCRIPT" --task "$task" --model "cosmos2-reason-LoRAFT-${lora_suffix}-2B"
  python3 "$SCRIPT" --task "$task" --model "cosmos2-reason-LoRAFT-${lora_suffix}-8B"
  
  for model in "${BASE_MODELS[@]}"; do
    python3 "$SCRIPT" --task "$task" --model "$model"
  done
done
