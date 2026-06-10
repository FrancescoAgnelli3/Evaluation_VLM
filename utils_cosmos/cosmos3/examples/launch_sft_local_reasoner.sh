#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for local_reasoner_sft (local manifest-based VLM SFT
# via DataPackerDataLoader). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/local_reasoner_sft.toml.
#
# Required env:
#   DATASET_PATH  local manifest dataset root with meta.json + media/ + text/.
#
# Optional env:
#   HF_TOKEN               for gated Qwen3-VL downloads.
#   VLM_SAFETENSORS_PATH   override for the local directory of pre-converted
#                          Qwen3-VL safetensors. Defaults to the repo-local
#                          Cosmos3-Nano-VLM checkpoint below.
#
# Usage:
#   DATASET_PATH=/opt/dataset/ds_pulito/train_dataset_manifest \
#   bash examples/launch_sft_local_reasoner.sh

TOML_FILE="examples/toml/sft_config/local_reasoner_sft.toml"
: "${DATASET_PATH:=/opt/dataset/ds_pulito/train_dataset_manifest}"
: "${VLM_SAFETENSORS_PATH:=/opt/models/dataset_clean/Cosmos3-FT/nano/local_reasoner_sft/hf_exports/iter_000000500/}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/meta.json" ]] || { echo "ERROR: missing $DATASET_PATH/meta.json" >&2; exit 1; }'

TAIL_OVERRIDES=(
    "model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH"
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
