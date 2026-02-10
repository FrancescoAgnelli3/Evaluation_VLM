#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_OUT_ROOT="/mnt/ssd1/Qwen3-32B-FT"

run_train_and_merge() {
  local tune="$1"
  local out_dir="$2"
  local merged_dir="$3"

  python3 "${SCRIPT_DIR}/train_qwen3vl_video_json.py" \
    --tune "${tune}" \
    --output_dir "${out_dir}" \
    --use_qlora

  python3 "${SCRIPT_DIR}/merge_weights.py" \
    --adapter_dir "${out_dir}" \
    --output_dir "${merged_dir}"
}

run_train_and_merge \
  both \
  "${BASE_OUT_ROOT}/ft_both_1k/Qwen_FT_adapter" \
  "${BASE_OUT_ROOT}/ft_both_1k/Qwen_FT_merged"

run_train_and_merge \
  llm \
  "${BASE_OUT_ROOT}/ft_llm_1k/Qwen_FT_adapter" \
  "${BASE_OUT_ROOT}/ft_llm_1k/Qwen_FT_merged"

# run_train_and_merge \
#   vision \
#   "${BASE_OUT_ROOT}/ft_vision/Qwen_FT_adapter" \
#   "${BASE_OUT_ROOT}/ft_vision/Qwen_FT_merged"
