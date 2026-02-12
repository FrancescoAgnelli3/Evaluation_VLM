# inspect_qwen3vl_linear_layers.py
import argparse
from collections import Counter, defaultdict

import torch
from transformers import Qwen3VLForConditionalGeneration

import os

# Ensure HF/torch caches are redirected before anything that may touch HF.
os.environ.setdefault("HF_HOME", "/mnt/ssd1/hf")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/ssd1/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/ssd1/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/ssd1/hf/datasets")
os.environ.setdefault("TORCH_HOME", "/mnt/ssd1/torch")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--attn_impl", default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    args = ap.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )

    linear_full_names = []
    linear_leaf_names = []
    buckets = defaultdict(list)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_full_names.append(name)
            leaf = name.split(".")[-1]
            linear_leaf_names.append(leaf)

            lname = name.lower()
            if any(k in lname for k in ["vision", "visual", "vit"]):
                buckets["vision"].append(name)
            elif any(k in lname for k in ["projector", "merger", "adapter", "mm"]):
                buckets["projector_or_mm"].append(name)
            else:
                buckets["llm_or_other"].append(name)

    print("\n=== Unique Linear leaf module names (for PEFT target_modules) ===")
    for leaf, c in Counter(linear_leaf_names).most_common():
        print(f"{leaf:30s}  count={c}")

    print("\n=== Sample full paths (first 50) ===")
    for n in linear_full_names[:50]:
        print(n)

    print("\n=== Vision Linear full paths (first 50) ===")
    for n in buckets["vision"][:50]:
        print(n)

    print("\n=== Projector/MM Linear full paths (first 50) ===")
    for n in buckets["projector_or_mm"][:50]:
        print(n)

    print("\n=== LLM/Other Linear full paths (first 50) ===")
    for n in buckets["llm_or_other"][:50]:
        print(n)

    # Suggested target_modules sets (leaf names) you can paste into PEFT
    llm_candidates = {"q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"}
    vision_candidates = {"q_proj","k_proj","v_proj","o_proj","proj","out_proj","fc1","fc2","up_proj","down_proj","gate_proj"}

    llm_targets = sorted({n.split(".")[-1] for n in buckets["llm_or_other"] if n.split(".")[-1] in llm_candidates})
    vision_targets = sorted({n.split(".")[-1] for n in buckets["vision"] if n.split(".")[-1] in vision_candidates})
    mm_targets = sorted({n.split(".")[-1] for n in buckets["projector_or_mm"]})

    print("\n=== Suggested PEFT target_modules ===")
    print("LLM targets:", llm_targets)
    print("Vision targets:", vision_targets)
    print("Projector/MM targets (leaf names):", mm_targets)

if __name__ == "__main__":
    main()
