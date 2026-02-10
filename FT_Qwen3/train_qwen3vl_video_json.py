#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Ensure HF/torch caches are redirected before anything that may touch HF.
os.environ.setdefault("HF_HOME", "/mnt/ssd1/hf")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/ssd1/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/ssd1/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/ssd1/hf/datasets")
os.environ.setdefault("TORCH_HOME", "/mnt/ssd1/torch")

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from qwen_vl_utils import process_vision_info

from transformers import TrainerCallback

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = "/mnt/ssd1/dataset_ft_VLM/dataset_train_subset_1000"
JSON_DIR = "/mnt/ssd1/dataset_ft_VLM/dataset_train_json_subset_1000"
PROMPT_DIR = BASE_DIR / "prompts/prompt_json.txt"
OUTPUT_DIR = "/mnt/ssd1/Qwen3-32B-FT/ft_both/Qwen_FT_adapter/"

# -------------------------
# Dataset
# -------------------------

def _read_prompt(prompt_path: str) -> str:
    """
    Reads a plain text prompt file and returns its full contents.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _collect_examples(video_dir: str, json_dir: str) -> List[Dict[str, str]]:
    """
    Supports either:
      A) metadata.jsonl with {"video": "...", "label": "...json string..."} per line
      B) video files with matching .json label files (same basename)
    """
    vdir = Path(video_dir)
    jdir = Path(json_dir)
    meta = jdir / "metadata.jsonl"
    if meta.exists():
        ex = []
        with open(meta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                video = obj["video"]
                label = obj["label"]
                ex.append({"video": str((vdir / video).resolve()) if not os.path.isabs(video) else video,
                           "label": label})
        if not ex:
            raise RuntimeError("metadata.jsonl exists but is empty.")
        return ex

    # fallback: pair *.mp4 (and other common formats) with *.json in separate dirs
    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    videos = []
    for p in vdir.rglob("*"):
        if p.suffix.lower() in video_exts:
            videos.append(p)

    if not videos:
        raise RuntimeError(f"No videos found under {video_dir}.")

    ex = []
    missing = 0
    for vp in sorted(videos):
        rel = vp.relative_to(vdir)
        jp = (jdir / rel).with_suffix(".json")
        if not jp.exists():
            missing += 1
            continue
        with open(jp, "r", encoding="utf-8") as f:
            label_obj = json.load(f)
        label_str = label_obj if isinstance(label_obj, str) else json.dumps(label_obj, ensure_ascii=False)
        ex.append({"video": str(vp.resolve()), "label": label_str})

    if not ex:
        raise RuntimeError(
            f"Found {len(videos)} videos but no matching .json labels. "
            f"Expected matching .json labels under {json_dir}, or metadata.jsonl in {json_dir}."
        )
    if missing > 0:
        print(f"[warn] {missing} videos had no matching .json label and were skipped.")
    return ex


class VideoJsonDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, str]],
        processor: Any,
        prompt: str,
        num_frames: float,
        max_prompt_tokens: int,
        max_label_tokens: int,
    ):
        self.examples = examples
        self.processor = processor
        self.prompt = prompt
        self.num_frames = num_frames
        self.max_prompt_tokens = max_prompt_tokens
        self.max_label_tokens = max_label_tokens

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        video_path = ex["video"]
        label_json = ex["label"]

        # 1) Build messages with video + fixed prompt.
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{video_path}", "num_frames": self.num_frames},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Prefix for label masking: user-only with generation prompt.
        prefix_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        # Full conversation text with assistant = target JSON
        full_messages = [
            user_messages[0],
            {"role": "assistant", "content": [{"type": "text", "text": label_json}]},
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # 2) Process video into model-ready tensors/kwargs.
        # qwen-vl-utils handles video sampling and returns video tensors + extra kwargs.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [user_messages],
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            # process_vision_info returns list of tuples (video_tensor, metadata)
            # for Qwen3-VL usage patterns in the wild.
            video_tensors, video_metadatas = zip(*video_inputs)
            video_tensors = list(video_tensors)
            video_metadatas = list(video_metadatas)
        else:
            video_tensors, video_metadatas = None, None

        # 3) Tokenize prefix and full text to create labels masked on the prefix.
        prefix_ids = self.processor(
            text=prefix_text,
            images=image_inputs,
            videos=video_tensors,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )["input_ids"][0]

        full = self.processor(
            text=full_text,
            images=image_inputs,
            videos=video_tensors,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]

        # Truncate safely: keep the prefix (which contains video tokens), truncate only the tail.
        max_len = self.max_prompt_tokens + self.max_label_tokens
        prefix_len = prefix_ids.numel()

        if input_ids.numel() > max_len:
            # Ensure we keep at least the whole prefix, otherwise we will drop video tokens.
            if prefix_len >= max_len:
                raise RuntimeError(
                    f"Prefix alone ({prefix_len} tokens) exceeds max_len ({max_len}). "
                    f"Increase --max_prompt_tokens or reduce video tokens (lower num_frames/resolution)."
                )
            keep_prefix = prefix_len
            keep_suffix = max_len - keep_prefix

            input_ids = torch.cat([input_ids[:keep_prefix], input_ids[-keep_suffix:]], dim=0)
            attention_mask = torch.cat([attention_mask[:keep_prefix], attention_mask[-keep_suffix:]], dim=0)

        # Labels: mask the (kept) prefix tokens.
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Add any vision/video fields returned by processor (e.g., pixel values, grids).
        # Keep everything 1D/2D tensors; collator will pad.
        for k, v in full.items():
            if k in batch:
                continue
            # Many Qwen-VL processors return tensors shaped [1, ...]
            if isinstance(v, torch.Tensor):
                batch[k] = v[0]
            else:
                batch[k] = v
        return batch


# -------------------------
# Collator
# -------------------------

@dataclass
class DataCollatorQwenVL:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pad text fields
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        # pad labels manually with -100
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        batch["labels"] = padded_labels

        # For remaining tensor fields: pad by stacking if shapes match; otherwise keep list.
        reserved = {"input_ids", "attention_mask", "labels"}
        other_keys = [k for k in features[0].keys() if k not in reserved]
        for k in other_keys:
            vals = [f[k] for f in features]
            if isinstance(vals[0], torch.Tensor):
                # If shapes match, stack. Else keep list (some VL fields vary).
                same = all(v.shape == vals[0].shape for v in vals)
                batch[k] = torch.stack(vals, dim=0) if same else vals
            else:
                batch[k] = vals
        return batch


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            print(f"step={state.global_step} loss={logs['loss']:.6f}")


# -------------------------
# LoRA targeting utilities
# -------------------------


def _freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _freeze_non_lora(model: torch.nn.Module):
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def _has_lora_params(model: torch.nn.Module) -> bool:
    return any("lora_" in name for name, _ in model.named_parameters())


def _print_trainable(model: torch.nn.Module):
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


def _count_trainable(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-32B-Thinking")
    ap.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    ap.add_argument("--json_dir", type=str, default=JSON_DIR)
    ap.add_argument("--prompt_path", type=str, default=PROMPT_DIR)
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    ap.add_argument("--tune", type=str, choices=["llm", "vision", "both"], default="both")

    ap.add_argument("--use_qlora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048*4)
    ap.add_argument("--max_label_tokens", type=int, default=512)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--bf16", default=True) # if false: use fp8
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--gradient_checkpointing", default=True)
    ap.add_argument("--print_trainable", default=True)
    args = ap.parse_args()

    prompt = _read_prompt(args.prompt_path)
    examples = _collect_examples(args.video_dir, args.json_dir)

    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=args.attn_impl,
        )

        # required for QLoRA training stability + correct grad flow
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    else:
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=args.attn_impl,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        _freeze_all(model)

    model.config.use_cache = False

    llm_targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    vision_targets = ["qkv","proj","linear_fc1","linear_fc2"]

    if args.tune == "llm":
        target_modules = llm_targets
    elif args.tune == "vision":
        target_modules = vision_targets
    else:
        target_modules = sorted(set(llm_targets).union(set(vision_targets)))

    if not target_modules:
        raise RuntimeError("Could not infer target_modules for LoRA. "
                           "Print model.named_modules() and adjust selection heuristics.")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)


    if args.print_trainable:
        _print_trainable(model)
        print("LoRA target_modules (leaf names):", target_modules)

    trainable, total = _count_trainable(model)
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "This usually means LoRA target_modules did not match any submodules. "
            "Inspect model.named_modules() and adjust target_modules."
        )

    train_ds = VideoJsonDataset(
        examples=examples,
        processor=processor,
        prompt=prompt,
        num_frames=args.num_frames,
        max_prompt_tokens=args.max_prompt_tokens,
        max_label_tokens=args.max_label_tokens,
    )

    collator = DataCollatorQwenVL(processor=processor)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        remove_unused_columns=False,  # important for multimodal fields
        dataloader_num_workers=2,
        logging_steps=1,
        logging_strategy="steps",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[PrintLossCallback()],
    )
    trainer.train()

    # Save adapter
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
