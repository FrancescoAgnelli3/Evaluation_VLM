#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${ROOT_DIR}/answer_questions.py"
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

task_media_dir() {
  case "$1" in
    road) echo "/opt/dataset/ds/Dataset_Segmented/Originale" ;;
    environment) echo "/opt/dataset/ds_environment/Dataset_Segmented/Originale" ;;
    people) echo "/opt/dataset/ds_people_ripulito/Dataset_Segmented/Originale" ;;
    industry) echo "/opt/dataset/ds_industry_ripulito/Dataset_Segmented/Originale" ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

task_models() {
  case "$1" in
    road)
      printf '%s\n' \
        "cosmos3" \
        "cosmos2-reason-LoRAFT_17k-32B" \
        "cosmos-reason2-32B" \
        "qwen3.5-FT_80k-27B" \
        "qwen3.5-FT_100k-27B" \
        "qwen3.5-FT_57k-27B" \
        "qwen3.5-FT_40k-27B" \
        "qwen3.5-LoRAFT_100k-122B" \
        "qwen3.5-FT_17k-27B" \
        "qwen3.5-LoRAFT_17k-9B" \
        "cosmos2-reason-LoRAFT_17k-8B" \
        "cosmos2-reason-fullFT_80k-8B" \
        "cosmos2-reason-fullFT_40k-8B" \
        "cosmos2-reason-fullFT_57k-8B" \
        "cosmos2-reason-fullFT_100k-8B" \
        "cosmos2-reason-fullFT_17k-8B" \
        "qwen3.5-397B" \
        "qwen3.5-27B" \
        "qwen3.5-122B" \
        "qwen-32B" \
        "qwen-8B" \
        "cosmos-reason2-32B" \
        "cosmos2-8B" \
        "qwen-2B" \
        "cosmos2-2B"
      ;;
    environment)
      printf '%s\n' \
        "qwen3.5-LoRAFT-env-9B" \
        "cosmos2-reason-LoRAFT-env-8B" \
        "qwen-32B" \
        "cosmos-reason2-32B" \
        "cosmos2-reason-LoRAFT-env-2B" \
        "qwen-8B" \
        "cosmos2-8B" \
        "qwen-2B" \
        "cosmos2-2B"
      ;;
    people)
      printf '%s\n' \
        "qwen3.5-LoRAFT-people-9B" \
        "cosmos2-reason-LoRAFT-people-8B" \
        "qwen-32B" \
        "cosmos-reason2-32B" \
        "cosmos2-reason-LoRAFT-people-2B" \
        "qwen-8B" \
        "qwen-2B" \
        "cosmos2-8B" \
        "cosmos2-2B"
      ;;
    industry)
      printf '%s\n' \
        "qwen3.5-LoRAFT-industry-9B" \
        "cosmos2-reason-LoRAFT-industry-2B" \
        "cosmos2-reason-LoRAFT-industry-8B" \
        "qwen-2B" \
        "cosmos3" \
        "cosmos-reason2-32B" \
        "cosmos2-8B" \
        "qwen-8B" \
        "cosmos2-2B"
      ;;
    *)
      echo "Unknown task: $1" >&2
      return 1
      ;;
  esac
}

TASKS=(road environment people industry)

for task in "${TASKS[@]}"; do
  media_dir="$(task_media_dir "$task")"

  mapfile -t models < <(task_models "$task")

  if [[ "${#models[@]}" -eq 0 ]]; then
    echo "No models configured for task '$task'" >&2
    exit 1
  fi

  cmd=(python3 "$SCRIPT" --task "$task" --media-dir "$media_dir")
  for model in "${models[@]}"; do
    cmd+=(--model "$model")
  done

  echo "Running task '$task' with ${#models[@]} model(s)"
  if ! "${cmd[@]}"; then
    echo "Task '$task' failed; continuing with the next task" >&2
  fi
done
