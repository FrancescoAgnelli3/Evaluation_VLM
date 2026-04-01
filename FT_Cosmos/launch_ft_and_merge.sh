#!/usr/bin/env bash
set -euo pipefail

# Launcher for fine-tuning + merging LoRA adapters for Cosmos-Reason2.
# Uses run.sh for training and merge_weights.py for merging.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

run_case() {
  local case_name="$1"
  local model_id="$2"
  local video_dir="$3"
  local json_dir="$4"
  local task="$5"
  local output_root="$6"

  local adapter_dir="${output_root}/adapter"
  local merged_dir="${output_root}/merged"

  echo "==> ${case_name}"
  echo "    model_id:   ${model_id}"
  echo "    video_dir:  ${video_dir}"
  echo "    json_dir:   ${json_dir}"
  echo "    task:       ${task}"
  echo "    adapter:    ${adapter_dir}"
  echo "    merged:     ${merged_dir}"

  "${SCRIPT_DIR}/run.sh" \
    --model_id "${model_id}" \
    --video_dir "${video_dir}" \
    --json_dir "${json_dir}" \
    --task "${task}" \
    --output_dir "${adapter_dir}"

  python "${SCRIPT_DIR}/merge_weights.py" \
    --base_model_id "${model_id}" \
    --adapter_dir "${adapter_dir}" \
    --output_dir "${merged_dir}"

  echo
}

# # Cases
# run_case \
#   "1) environment | 2B" \
#   "nvidia/Cosmos-Reason2-2B" \
#   "/opt/dataset/ds_environment/train_dataset" \
#   "/opt/dataset/ds_environment/train_dataset_json" \
#   "environment" \
#   "/opt/models/Cosmos-Reason2-FT/2B/LoRA/ds_environment"

# run_case \
#   "2) environment | 8B" \
#   "nvidia/Cosmos-Reason2-8B" \
#   "/opt/dataset/ds_environment/train_dataset" \
#   "/opt/dataset/ds_environment/train_dataset_json" \
#   "environment" \
#   "/opt/models/Cosmos-Reason2-FT/8B/LoRA/ds_environment"

run_case \
  "3) industry | 2B" \
  "nvidia/Cosmos-Reason2-2B" \
  "/opt/dataset/ds_industry_ripulito/train_dataset" \
  "/opt/dataset/ds_industry_ripulito/train_dataset_json" \
  "industry" \
  "/opt/models/Cosmos-Reason2-FT/2B/LoRA/ds_industry_ripulito"

run_case \
  "4) industry | 8B (json path per request)" \
  "nvidia/Cosmos-Reason2-8B" \
  "/opt/dataset/ds_industry_ripulito/train_dataset" \
  "/opt/dataset/ds_industry_ripulito/train_dataset_json" \
  "industry" \
  "/opt/models/Cosmos-Reason2-FT/8B/LoRA/ds_industry_ripulito"

run_case \
  "5) people | 2B" \
  "nvidia/Cosmos-Reason2-2B" \
  "/opt/dataset/ds_people_ripulito/train_dataset" \
  "/opt/dataset/ds_people_ripulito/train_dataset_json" \
  "people" \
  "/opt/models/Cosmos-Reason2-FT/2B/LoRA/ds_people_ripulito"

run_case \
  "6) people | 8B" \
  "nvidia/Cosmos-Reason2-8B" \
  "/opt/dataset/ds_people_ripulito/train_dataset" \
  "/opt/dataset/ds_people_ripulito/train_dataset_json" \
  "people" \
  "/opt/models/Cosmos-Reason2-FT/8B/LoRA/ds_people_ripulito"
