from __future__ import annotations

import os
from typing import Tuple


# ----------------------------
# Model identifiers (HF repos only)
# ----------------------------

QWEN_32B_REPO = os.environ.get("QWEN_32B_REPO", "Qwen/Qwen3-VL-32B-Thinking")
QWEN_122B_REPO = os.environ.get("QWEN_122B_REPO", "Qwen/Qwen3.5-122B-A10B")
QWEN3_5_397B_REPO = os.environ.get("QWEN3_5_397B_REPO", "http://localhost:14000/v1/models")
QWEN3_5_27B_REPO = os.environ.get("QWEN3_5_27B_REPO", "Qwen/Qwen3.5-27B")
QWEN3_5_FT_27B_100K_REPO = os.environ.get(
    "QWEN3_5_FT_27B_100K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-27B_100k/v4-20260320-020441/checkpoint-1531",
)
QWEN3_5_FT_27B_40K_REPO = os.environ.get(
    "QWEN3_5_FT_27B_40K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-27B_40k/v2-20260327-014257/checkpoint-600",
)
QWEN3_5_FT_27B_17K_REPO = os.environ.get(
    "QWEN3_5_FT_27B_17K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-27B_17k/v1-20260326-195026/checkpoint-256",
)
QWEN3_5_FT_27B_57K_REPO = os.environ.get(
    "QWEN3_5_FT_27B_57K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-27B_57k/v0-20260328-054740/checkpoint-856",
)
QWEN3_5_FT_27B_80K_REPO = os.environ.get(
    "QWEN3_5_FT_27B_80K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-27B_80k/v1-20260329-233212/checkpoint-1200",
)
QWEN_8B_REPO = os.environ.get("QWEN_8B_REPO", "Qwen/Qwen3-VL-8B-Thinking")
QWEN_2B_REPO = os.environ.get("QWEN_2B_REPO", "Qwen/Qwen3-VL-2B-Thinking")
QWEN_8B_FT_VISION_REPO = os.environ.get("QWEN_8B_FT_VISION_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-8B-FT-vision")
QWEN_8B_FT_LLM_REPO = os.environ.get("QWEN_8B_FT_LLM_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-8B-FT-llm")
QWEN_8B_FT_LLM_1K_REPO = os.environ.get("QWEN_8B_FT_LLM_1K_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-8B-FT-llm_1k")
QWEN_8B_FT_BOTH_REPO = os.environ.get("QWEN_8B_FT_BOTH_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-8B-FT-both")
QWEN_8B_FT_BOTH_1K_REPO = os.environ.get("QWEN_8B_FT_BOTH_1K_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-8B-FT-both_1k")
QWEN_32B_FT_LLM_REPO = os.environ.get("QWEN_32B_FT_LLM_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-32B_llm")
QWEN_32B_FT_BOTH_REPO = os.environ.get("QWEN_32B_FT_BOTH_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-32B_both")
QWEN_32B_FT_LLM_1K_REPO = os.environ.get("QWEN_32B_FT_LLM_1K_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-32B_llm_1k")
QWEN_32B_FT_BOTH_1K_REPO = os.environ.get("QWEN_32B_FT_BOTH_1K_REPO", "/mnt/Repo/VLM_ft/models/Qwen3-32B_both_1k")
QWEN3_5_FT_122B_100K_REPO = os.environ.get(
    "QWEN3_5_FT_122B_100K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-122B-A10B_ft_100k_v2/v0-20260323-173143/checkpoint-3093-merged",
)
QWEN3_5_LORAFT_9B_17K_REPO = os.environ.get(
    "QWEN3_5_LORAFT_9B_17K_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-9B_17k_merged",
)
QWEN3_5_LORAFT_9B_PEOPLE_REPO = os.environ.get(
    "QWEN3_5_LORAFT_9B_PEOPLE_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-9B_people",
)
QWEN3_5_LORAFT_9B_INDUSTRY_REPO = os.environ.get(
    "QWEN3_5_LORAFT_9B_INDUSTRY_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-9B_industry",
)
QWEN3_5_LORAFT_9B_ENVIRONMENT_REPO = os.environ.get(
    "QWEN3_5_LORAFT_9B_ENVIRONMENT_REPO",
    "/opt/models/Qwen3.5_FT/Qwen3.5-9B_environment",
)

COSMOS_REASON1_REPO = os.environ.get("COSMOS_REASON1_REPO", "nvidia/Cosmos-Reason1-7B")
COSMOS_REASON2_2B_REPO = os.environ.get("COSMOS_REASON2_2B_REPO", "nvidia/Cosmos-Reason2-2B")
COSMOS_REASON2_8B_REPO = os.environ.get("COSMOS_REASON2_8B_REPO", "nvidia/Cosmos-Reason2-8B")
COSMOS_REASON2_LORAFT_2B_13K_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_2B_13K_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/dataset_13k/LoRA/merged",
)
COSMOS_REASON2_LORAFT_8B_13K_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_8B_13K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/dataset_13k/LoRA/merged",
)
COSMOS_REASON2_FULLFT_2B_13K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_2B_13K_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/full_FT/dataset_13k/safetensors/step_780/",
)
COSMOS_REASON2_FULLFT_8B_13K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_13K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_13k/safetensors/step_780/",
)
COSMOS_REASON2_FULLFT_2B_10K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_2B_10K_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/full_FT/dataset_10k/safetensors/step_745",
)
COSMOS_REASON2_FULLFT_2B_5K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_2B_5K_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/full_FT/dataset_5k/safetensors/step_390",
)
COSMOS_REASON2_FULLFT_2B_2K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_2B_2K_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/full_FT/dataset_2k/safetensors/step_155",
)
COSMOS_REASON2_FULLFT_8B_17K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_17K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_17k/safetensors/step_975",
)
COSMOS_REASON2_FULLFT_8B_29K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_29K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_29k/20260301110005/safetensors/step_1359/",
)
COSMOS_REASON2_FULLFT_8B_40K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_40K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_40k/20260307095420/safetensors/step_1875",
)
COSMOS_REASON2_FULLFT_8B_57K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_57K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_57k/20260306124146/safetensors/step_2670",
)
COSMOS_REASON2_FULLFT_8B_80K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_80K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_80k/20260309090732/safetensors/step_3750",
)
COSMOS_REASON2_FULLFT_8B_100K_REPO = os.environ.get(
    "COSMOS_REASON2_FULLFT_8B_100K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/full_FT/dataset_100k/20260310101650/safetensors/step_4686",
)
COSMOS_REASON2_LORAFT_2B_17K_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_2B_17K_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/LoRA/dataset_17k/merged",
)
COSMOS_REASON2_LORAFT_8B_17K_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_8B_17K_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/LoRA/dataset_17k/merged",
)
COSMOS_REASON2_LORAFT_ENV_2B_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_ENV_2B_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/LoRA/ds_environment/merged",
)
COSMOS_REASON2_LORAFT_ENV_8B_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_ENV_8B_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/LoRA/ds_environment/merged",
)
COSMOS_REASON2_LORAFT_PEOPLE_2B_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_PEOPLE_2B_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/LoRA/ds_people_ripulito/merged",
)
COSMOS_REASON2_LORAFT_PEOPLE_8B_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_PEOPLE_8B_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/LoRA/ds_people_ripulito/merged",
)
COSMOS_REASON2_LORAFT_INDUSTRY_2B_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_INDUSTRY_2B_REPO",
    "/storage/models/Cosmos-Reason2-FT/2B/LoRA/ds_industry_ripulito/merged",
)
COSMOS_REASON2_LORAFT_INDUSTRY_8B_REPO = os.environ.get(
    "COSMOS_REASON2_LORAFT_INDUSTRY_8B_REPO",
    "/storage/models/Cosmos-Reason2-FT/8B/LoRA/ds_industry_ripulito/merged",
)

MODEL_CHOICES: Tuple[str, ...] = (
    "qwen-8B",
    # "qwen-32B-FT-llm",
    # "qwen-32B-FT-both",
    "qwen-32B",
    "qwen3.5-122B",
    "qwen3.5-397B",
    "qwen3.5-27B",
    "qwen3.5-FT_100k-27B",
    "qwen3.5-FT_40k-27B",
    "qwen3.5-FT_17k-27B",
    "qwen3.5-FT_57k-27B",
    "qwen3.5-FT_80k-27B",
    "qwen3.5-LoRAFT_100k-122B",
    "qwen3.5-LoRAFT_17k-9B",
    "qwen3.5-LoRAFT-people-9B",
    "qwen3.5-LoRAFT-industry-9B",
    "qwen3.5-LoRAFT-env-9B",
    # "qwen-32B-FT-both-1k",
    # "qwen-32B-FT-llm-1k",
    # "qwen-8B-FT-llm",
    # "qwen-8B-FT-llm-1k",
    # "qwen-8B-FT-both",
    # "qwen-8B-FT-both-1k",
    "cosmos2-2B",
    "cosmos2-8B",
    # "cosmos2-reason-LoRAFT_13k-2B",
    # "cosmos2-reason-LoRAFT_13k-8B",
    "cosmos2-reason-fullFT_13k-2B",
    # "cosmos2-reason-fullFT_13k-8B",
    # "cosmos2-reason-fullFT_10k-2B",
    # "cosmos2-reason-fullFT_5k-2B",
    # "cosmos2-reason-fullFT_2k-2B",
    "cosmos2-reason-fullFT_17k-8B",
    # "cosmos2-reason-fullFT_29k-8B",
    "cosmos2-reason-fullFT_40k-8B",
    "cosmos2-reason-fullFT_57k-8B",
    "cosmos2-reason-fullFT_80k-8B",
    "cosmos2-reason-fullFT_100k-8B",
    # "cosmos2-reason-LoRAFT_17k-2B",
    "cosmos2-reason-LoRAFT_17k-8B",
    "cosmos2-reason-LoRAFT-env-2B",
    "cosmos2-reason-LoRAFT-env-8B",
    "cosmos2-reason-LoRAFT-people-2B",
    "cosmos2-reason-LoRAFT-people-8B",
    "cosmos2-reason-LoRAFT-industry-2B",
    "cosmos2-reason-LoRAFT-industry-8B",
    # "cosmos1",
    "qwen-2B",
    "all",
)
DEFAULT_MODEL_SELECTION = os.environ.get("DEFAULT_MODEL", "cosmos2-2B")


def served_name_for(model_key: str) -> str:
    if model_key == "qwen-32B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_32B", "Qwen3-VL-32B-Thinking")
    if model_key == "qwen3.5-122B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_122B", "Qwen3.5-122B-A10B")
    if model_key == "qwen3.5-397B":
        return os.environ.get("QWEN3_5_397B_NAME", "Qwen3.5-397B-A17B-FP8")
    if model_key == "qwen3.5-27B":
        return os.environ.get("QWEN3_5_27B_NAME", "Qwen3.5-27B")
    if model_key == "qwen3.5-FT_100k-27B":
        return os.environ.get("QWEN3_5_FT_27B_100k_NAME", "Qwen3.5-FT-100k_27B")
    if model_key == "qwen3.5-FT_40k-27B":
        return os.environ.get("QWEN3_5_FT_27B_40k_NAME", "Qwen3.5-FT-40k_27B")
    if model_key == "qwen3.5-FT_17k-27B":
        return os.environ.get("QWEN3_5_FT_27B_17k_NAME", "Qwen3.5-FT-17k_27B")
    if model_key == "qwen3.5-FT_57k-27B":
        return os.environ.get("QWEN3_5_FT_27B_57k_NAME", "Qwen3.5-FT-57k_27B")
    if model_key == "qwen3.5-FT_80k-27B":
        return os.environ.get("QWEN3_5_FT_27B_80k_NAME", "Qwen3.5-FT-80k_27B")
    if model_key == "qwen3.5-LoRAFT_100k-122B":
        return os.environ.get("QWEN3_5_FT_122B_100k_NAME", "Qwen3.5-FT-100k_122B")
    if model_key == "qwen3.5-LoRAFT_17k-9B":
        return os.environ.get("QWEN3_5_LORAFT_9B_17k_NAME", "Qwen3.5-LoRAFT-17k_9B")
    if model_key == "qwen3.5-LoRAFT-people-9B":
        return os.environ.get("QWEN3_5_LORAFT_9B_PEOPLE_NAME", "Qwen3.5-LoRAFT-People-9B")
    if model_key == "qwen3.5-LoRAFT-industry-9B":
        return os.environ.get("QWEN3_5_LORAFT_9B_INDUSTRY_NAME", "Qwen3.5-LoRAFT-Industry-9B")
    if model_key == "qwen3.5-LoRAFT-env-9B":
        return os.environ.get("QWEN3_5_LORAFT_9B_ENVIRONMENT_NAME", "Qwen3.5-LoRAFT-Env-9B")
    if model_key == "qwen-8B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B", "Qwen3-VL-8B-Thinking")
    if model_key == "qwen-2B":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_2B", "Qwen3-VL-2B-Thinking")
    if model_key == "qwen-8B-FT-vision":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B_FT_VISION", "Qwen3-8B-FT-Vision")
    if model_key == "qwen-8B-FT-llm":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B_FT_LLM", "Qwen3-8B-FT-LLM")
    if model_key == "qwen-8B-FT-llm-1k":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B_FT_LLM_1K", "Qwen3-8B-FT-LLM-1k")
    if model_key == "qwen-8B-FT-both":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B_FT_BOTH", "Qwen3-8B-FT-Both")
    if model_key == "qwen-8B-FT-both-1k":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_8B_FT_BOTH_1K", "Qwen3-8B-FT-Both-1k")
    if model_key == "qwen-32B-FT-llm":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_32B_FT_LLM", "Qwen3-32B-FT-LLM")
    if model_key == "qwen-32B-FT-both":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_32B_FT_BOTH", "Qwen3-32B-FT-Both")
    if model_key == "qwen-32B-FT-llm-1k":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_32B_FT_LLM_1K", "Qwen3-32B-FT-LLM-1k")
    if model_key == "qwen-32B-FT-both-1k":
        return os.environ.get("QWEN_VLLM_MODEL_NAME_32B_FT_BOTH_1K", "Qwen3-32B-FT-Both-1k")
    if model_key == "cosmos1":
        return "Cosmos-Reason1"
    if model_key == "cosmos2-2B":
        return "Cosmos-Reason2-2B"
    if model_key == "cosmos2-8B":
        return "Cosmos-Reason2-8B"
    if model_key == "cosmos2-reason-LoRAFT_13k-2B":
        return os.environ.get("COSMOS_REASON2_LORAFT_2B_13k_NAME", "Cosmos-Reason2-LoRAFT-13k_2B")
    if model_key == "cosmos2-reason-LoRAFT_13k-8B":
        return os.environ.get("COSMOS_REASON2_LORAFT_13k_NAME", "Cosmos-Reason2-LoRAFT-13k_8B")
    if model_key == "cosmos2-reason-fullFT_13k-2B":
        return os.environ.get("COSMOS_REASON2_FULLFT_13k_NAME", "Cosmos-Reason2-FullFT-13k_2B")
    if model_key == "cosmos2-reason-fullFT_13k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_13k_NAME", "Cosmos-Reason2-FullFT-13k_8B")
    if model_key == "cosmos2-reason-fullFT_10k-2B":
        return os.environ.get("COSMOS_REASON2_FULLFT_10k_NAME", "Cosmos-Reason2-FullFT-10k_2B")
    if model_key == "cosmos2-reason-fullFT_5k-2B":
        return os.environ.get("COSMOS_REASON2_FULLFT_5K_NAME", "Cosmos-Reason2-FullFT-5k_2B")
    if model_key == "cosmos2-reason-fullFT_2k-2B":
        return os.environ.get("COSMOS_REASON2_FULLFT_2K_NAME", "Cosmos-Reason2-FullFT-2k_2B")
    if model_key == "cosmos2-reason-fullFT_17k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_17k_NAME", "Cosmos-Reason2-FullFT-17k_8B")
    if model_key == "cosmos2-reason-fullFT_29k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_29k_NAME", "Cosmos-Reason2-FullFT-29k_8B")
    if model_key == "cosmos2-reason-fullFT_40k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_40k_NAME", "Cosmos-Reason2-FullFT-40k_8B")
    if model_key == "cosmos2-reason-fullFT_57k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_57k_NAME", "Cosmos-Reason2-FullFT-57k_8B")
    if model_key == "cosmos2-reason-fullFT_80k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_80k_NAME", "Cosmos-Reason2-FullFT-80k_8B")
    if model_key == "cosmos2-reason-fullFT_100k-8B":
        return os.environ.get("COSMOS_REASON2_FULLFT_8B_100k_NAME", "Cosmos-Reason2-FullFT-100k_8B")
    if model_key == "cosmos2-reason-LoRAFT_17k-2B":
        return os.environ.get("COSMOS_REASON2_LORAFT_2B_17k_NAME", "Cosmos-Reason2-LoRAFT-17k_2B")
    if model_key == "cosmos2-reason-LoRAFT_17k-8B":
        return os.environ.get("COSMOS_REASON2_LORAFT_8B_17k_NAME", "Cosmos-Reason2-LoRAFT-17k_8B")
    if model_key == "cosmos2-reason-LoRAFT-env-2B":
        return os.environ.get("COSMOS_REASON2_LORAFT_ENV_2B_NAME", "Cosmos-Reason2-LoRAFT-Env-2B")
    if model_key == "cosmos2-reason-LoRAFT-env-8B":
        return os.environ.get("COSMOS_REASON2_LORAFT_ENV_8B_NAME", "Cosmos-Reason2-LoRAFT-Env-8B")
    if model_key == "cosmos2-reason-LoRAFT-people-2B":
        return os.environ.get("COSMOS_REASON2_LORAFT_PEOPLE_2B_NAME", "Cosmos-Reason2-LoRAFT-People-2B")
    if model_key == "cosmos2-reason-LoRAFT-people-8B":
        return os.environ.get("COSMOS_REASON2_LORAFT_PEOPLE_8B_NAME", "Cosmos-Reason2-LoRAFT-People-8B")
    if model_key == "cosmos2-reason-LoRAFT-industry-2B":
        return os.environ.get("COSMOS_REASON2_LORAFT_INDUSTRY_2B_NAME", "Cosmos-Reason2-LoRAFT-Industry-2B")
    if model_key == "cosmos2-reason-LoRAFT-industry-8B":
        return os.environ.get("COSMOS_REASON2_LORAFT_INDUSTRY_8B_NAME", "Cosmos-Reason2-LoRAFT-Industry-8B")
    raise ValueError(f"Unknown model_key: {model_key}")


def resolve_model_repo(model_key: str) -> str:
    if model_key == "qwen-32B":
        return QWEN_32B_REPO
    if model_key == "qwen3.5-122B":
        return QWEN_122B_REPO
    if model_key == "qwen3.5-397B":
        return QWEN3_5_397B_REPO
    if model_key == "qwen3.5-27B":
        return QWEN3_5_27B_REPO
    if model_key == "qwen3.5-FT_100k-27B":
        return QWEN3_5_FT_27B_100K_REPO
    if model_key == "qwen3.5-FT_40k-27B":
        return QWEN3_5_FT_27B_40K_REPO
    if model_key == "qwen3.5-FT_17k-27B":
        return QWEN3_5_FT_27B_17K_REPO
    if model_key == "qwen3.5-FT_57k-27B":
        return QWEN3_5_FT_27B_57K_REPO
    if model_key == "qwen3.5-FT_80k-27B":
        return QWEN3_5_FT_27B_80K_REPO
    if model_key == "qwen3.5-LoRAFT_100k-122B":
        return QWEN3_5_FT_122B_100K_REPO
    if model_key == "qwen3.5-LoRAFT_17k-9B":
        return QWEN3_5_LORAFT_9B_17K_REPO
    if model_key == "qwen3.5-LoRAFT-people-9B":
        return QWEN3_5_LORAFT_9B_PEOPLE_REPO
    if model_key == "qwen3.5-LoRAFT-industry-9B":
        return QWEN3_5_LORAFT_9B_INDUSTRY_REPO
    if model_key == "qwen3.5-LoRAFT-env-9B":
        return QWEN3_5_LORAFT_9B_ENVIRONMENT_REPO
    if model_key == "qwen-8B":
        return QWEN_8B_REPO
    if model_key == "qwen-2B":
        return QWEN_2B_REPO
    if model_key == "qwen-8B-FT-vision":
        return QWEN_8B_FT_VISION_REPO
    if model_key == "qwen-8B-FT-llm":
        return QWEN_8B_FT_LLM_REPO
    if model_key == "qwen-8B-FT-llm-1k":
        return QWEN_8B_FT_LLM_1K_REPO
    if model_key == "qwen-8B-FT-both":
        return QWEN_8B_FT_BOTH_REPO
    if model_key == "qwen-8B-FT-both-1k":
        return QWEN_8B_FT_BOTH_1K_REPO
    if model_key == "qwen-32B-FT-llm":
        return QWEN_32B_FT_LLM_REPO
    if model_key == "qwen-32B-FT-both":
        return QWEN_32B_FT_BOTH_REPO
    if model_key == "qwen-32B-FT-llm-1k":
        return QWEN_32B_FT_LLM_1K_REPO
    if model_key == "qwen-32B-FT-both-1k":
        return QWEN_32B_FT_BOTH_1K_REPO
    if model_key == "cosmos1":
        return COSMOS_REASON1_REPO
    if model_key == "cosmos2-2B":
        return COSMOS_REASON2_2B_REPO
    if model_key == "cosmos2-8B":
        return COSMOS_REASON2_8B_REPO
    if model_key == "cosmos2-reason-LoRAFT_13k-2B":
        return COSMOS_REASON2_LORAFT_2B_13K_REPO
    if model_key == "cosmos2-reason-LoRAFT_13k-8B":
        return COSMOS_REASON2_LORAFT_8B_13K_REPO
    if model_key == "cosmos2-reason-fullFT_13k-2B":
        return COSMOS_REASON2_FULLFT_2B_13K_REPO
    if model_key == "cosmos2-reason-fullFT_13k-8B":
        return COSMOS_REASON2_FULLFT_8B_13K_REPO
    if model_key == "cosmos2-reason-fullFT_10k-2B":
        return COSMOS_REASON2_FULLFT_2B_10K_REPO
    if model_key == "cosmos2-reason-fullFT_5k-2B":
        return COSMOS_REASON2_FULLFT_2B_5K_REPO
    if model_key == "cosmos2-reason-fullFT_2k-2B":
        return COSMOS_REASON2_FULLFT_2B_2K_REPO
    if model_key == "cosmos2-reason-fullFT_17k-8B":
        return COSMOS_REASON2_FULLFT_8B_17K_REPO
    if model_key == "cosmos2-reason-fullFT_29k-8B":
        return COSMOS_REASON2_FULLFT_8B_29K_REPO
    if model_key == "cosmos2-reason-fullFT_40k-8B":
        return COSMOS_REASON2_FULLFT_8B_40K_REPO
    if model_key == "cosmos2-reason-fullFT_57k-8B":
        return COSMOS_REASON2_FULLFT_8B_57K_REPO
    if model_key == "cosmos2-reason-fullFT_80k-8B":
        return COSMOS_REASON2_FULLFT_8B_80K_REPO
    if model_key == "cosmos2-reason-fullFT_100k-8B":
        return COSMOS_REASON2_FULLFT_8B_100K_REPO
    if model_key == "cosmos2-reason-LoRAFT_17k-2B":
        return COSMOS_REASON2_LORAFT_2B_17K_REPO
    if model_key == "cosmos2-reason-LoRAFT_17k-8B":
        return COSMOS_REASON2_LORAFT_8B_17K_REPO
    if model_key == "cosmos2-reason-LoRAFT-env-2B":
        return COSMOS_REASON2_LORAFT_ENV_2B_REPO
    if model_key == "cosmos2-reason-LoRAFT-env-8B":
        return COSMOS_REASON2_LORAFT_ENV_8B_REPO
    if model_key == "cosmos2-reason-LoRAFT-people-2B":
        return COSMOS_REASON2_LORAFT_PEOPLE_2B_REPO
    if model_key == "cosmos2-reason-LoRAFT-people-8B":
        return COSMOS_REASON2_LORAFT_PEOPLE_8B_REPO
    if model_key == "cosmos2-reason-LoRAFT-industry-2B":
        return COSMOS_REASON2_LORAFT_INDUSTRY_2B_REPO
    if model_key == "cosmos2-reason-LoRAFT-industry-8B":
        return COSMOS_REASON2_LORAFT_INDUSTRY_8B_REPO
    raise ValueError(f"Unknown model_key: {model_key}")
