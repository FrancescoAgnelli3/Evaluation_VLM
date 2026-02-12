#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into the base Qwen3-VL-8B model and save a standalone
fully fine-tuned model (no adapters needed at inference).

Typical usage:
  python merge_lora.py \
    --base_model_id Qwen/Qwen3-VL-8B-Thinking \
    --adapter_dir /mnt/ssd1/Qwen_FT \
    --output_dir /mnt/ssd1/Qwen_FT_merged \
    --dtype bf16
"""

import os

# # Ensure HF/torch caches are redirected before anything that may touch HF.
# os.environ.setdefault("HF_HOME", "/mnt/ssd1/hf")
# os.environ.setdefault("HF_HUB_CACHE", "/mnt/ssd1/hf/hub")
# os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/ssd1/hf/transformers")
# os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/ssd1/hf/datasets")
# os.environ.setdefault("TORCH_HOME", "/mnt/ssd1/torch")

BASE_MODEL_ID = "nvidia/Cosmos-Reason2-8B"
ADAPTER_DIR = "/opt/models/Cosmos-Reason2-FT/adapter"
OUTPUT_DIR = "/opt/models/Cosmos-Reason2-FT/merged"

import argparse
import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, default=BASE_MODEL_ID)
    ap.add_argument("--adapter_dir", type=str, default=ADAPTER_DIR)
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--safe_serialization", action="store_true", help="Save as .safetensors")
    ap.add_argument("--max_shard_size", type=str, default="2GB")
    args = ap.parse_args()

    torch_dtype = parse_dtype(args.dtype)

    # IMPORTANT:
    # If you trained with --use_qlora (4-bit), do NOT load the base in 4-bit here.
    # Load the base in bf16/fp16/fp32, then apply the adapter and merge.
    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(
        base,
        args.adapter_dir,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    # Merge LoRA weights into the base model and drop adapter modules.
    merged = peft_model.merge_and_unload()

    # Save standalone merged model
    merged.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(args.output_dir)

    print(f"Saved merged model to: {args.output_dir}")


if __name__ == "__main__":
    main()
