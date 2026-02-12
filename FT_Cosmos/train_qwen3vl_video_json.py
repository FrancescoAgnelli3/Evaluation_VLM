#!/usr/bin/env python3
"""
Cosmos-Reason2 video SFT (LoRA / QLoRA) using TRL SFTTrainer.

This version adds a robust internal multi-process launcher (mp.spawn) AND
explicitly initializes torch.distributed (DDP backend) when WORLD_SIZE > 1,
so multi-GPU actually becomes distributed instead of N independent trainings.

Key changes:
- init_process_group(backend="nccl", init_method="env://") when WORLD_SIZE>1
- avoid device_map in multi-process; use model.to(cuda:LOCAL_RANK)
- use torch_dtype=... (not dtype=...)
- save artifacts only on rank 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

# -------------------------
# Paths / defaults
# -------------------------

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = "/opt/dataset/train_dataset"
JSON_DIR = "/opt/dataset/train_dataset_json"
PROMPT_DIR = BASE_DIR / "prompts/prompt_json.txt"
OUTPUT_DIR = "/opt/models/Cosmos-Reason2-FT/adapter/"

# Compat shim: some torch builds expose torch.compiler without is_compiling
if not hasattr(torch, "compiler"):
    class _CompilerShim:
        pass
    torch.compiler = _CompilerShim()

if not hasattr(torch.compiler, "is_compiling"):
    try:
        import torch._dynamo
        torch.compiler.is_compiling = torch._dynamo.is_compiling
    except Exception:
        torch.compiler.is_compiling = lambda: False

# -------------------------
# Dataset utils
# -------------------------

def _read_prompt(prompt_path: str) -> str:
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
        ex: List[Dict[str, str]] = []
        with open(meta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                video = obj["video"]
                if isinstance(video, (list, tuple)):
                    if len(video) != 1:
                        raise RuntimeError(
                            f"metadata.jsonl entry has {len(video)} videos; expected 1: {video}"
                        )
                    video = video[0]
                label = obj["label"]
                ex.append(
                    {
                        "video": str((vdir / video).resolve()) if not os.path.isabs(video) else video,
                        "label": label,
                    }
                )
        if not ex:
            raise RuntimeError("metadata.jsonl exists but is empty.")
        return ex

    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    videos = [p for p in vdir.rglob("*") if p.suffix.lower() in video_exts]
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

# -------------------------
# Multimodal dataset
# -------------------------

def _as_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Normalizes tensors that may be [1, L] or [L] into [L].
    Avoids the common bug where indexing [0] turns [L] into a scalar.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() == 2 and x.size(0) == 1:
        return x.squeeze(0)
    if x.dim() == 1:
        return x
    raise RuntimeError(f"Unexpected {name} shape: {tuple(x.shape)}")


def _squeeze_batch_if_hooking(v: torch.Tensor) -> torch.Tensor:
    """
    Many processor fields are returned as [1, ...]. Remove that batch dim if present.
    """
    if v.dim() >= 1 and v.size(0) == 1:
        return v.squeeze(0)
    return v


class VideoJsonDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, str]],
        processor: Any,
        prompt: str,
        num_frames: int,
        max_prompt_tokens: int,
        max_label_tokens: int,
    ):
        self.examples = examples
        self.processor = processor
        self.prompt = str(prompt)
        self.num_frames = int(num_frames)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_label_tokens = int(max_label_tokens)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        video_path = ex["video"]
        if isinstance(video_path, (list, tuple)):
            if len(video_path) != 1:
                raise RuntimeError(f"Example has {len(video_path)} videos; expected 1: {video_path}")
            video_path = video_path[0]
        label_json = ex["label"]
        if isinstance(label_json, (list, tuple)):
            if len(label_json) != 1:
                raise RuntimeError(f"Example has {len(label_json)} labels; expected 1.")
            label_json = label_json[0]

        # 1) chat messages
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{video_path}", "num_frames": self.num_frames},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        prefix_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        full_messages = [
            user_messages[0],
            {"role": "assistant", "content": [{"type": "text", "text": label_json}]},
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # 2) vision/video preprocessing
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [user_messages],
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            video_tensors, video_metadatas = zip(*video_inputs)
            video_tensors = list(video_tensors)
            video_metadatas = list(video_metadatas)
        else:
            video_tensors, video_metadatas = None, None

        # 3) tokenize prefix to get prefix_len for label masking
        prefix = self.processor(
            text=prefix_text,
            images=image_inputs,
            videos=video_tensors,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )
        prefix_ids = _as_1d(prefix["input_ids"], "prefix.input_ids")
        prefix_len = int(prefix_ids.numel())

        # tokenize full
        full = self.processor(
            text=full_text,
            images=image_inputs,
            videos=video_tensors,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )
        input_ids = _as_1d(full["input_ids"], "full.input_ids")
        attention_mask = _as_1d(full["attention_mask"], "full.attention_mask")

        # 4) truncate: keep entire prefix (incl. video tokens) + last part of suffix
        max_len = self.max_prompt_tokens + self.max_label_tokens
        if input_ids.numel() > max_len:
            if prefix_len >= max_len:
                raise RuntimeError(
                    f"Prefix alone ({prefix_len} tokens) exceeds max_len ({max_len}). "
                    f"Increase max_prompt_tokens or reduce video tokens (e.g., fewer frames)."
                )
            keep_prefix = prefix_len
            keep_suffix = max_len - keep_prefix
            input_ids = torch.cat([input_ids[:keep_prefix], input_ids[-keep_suffix:]], dim=0)
            attention_mask = torch.cat([attention_mask[:keep_prefix], attention_mask[-keep_suffix:]], dim=0)
            prefix_len = keep_prefix

        # 5) labels: mask prefix
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 6) carry over all other processor outputs (pixel values, grids, etc.)
        for k, v in full.items():
            if k in out:
                continue
            if isinstance(v, torch.Tensor):
                out[k] = _squeeze_batch_if_hooking(v)
            else:
                out[k] = v

        return out

# -------------------------
# Collator
# -------------------------

@dataclass
class DataCollatorQwenVL:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _to_tensor(x: Any, dtype: torch.dtype) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x, dtype=dtype)

        input_ids = [_to_tensor(f["input_ids"], torch.long) for f in features]
        attention_mask = [_to_tensor(f["attention_mask"], torch.long) for f in features]
        labels = [_to_tensor(f["labels"], torch.long) for f in features]

        batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        batch["labels"] = padded_labels

        reserved = {"input_ids", "attention_mask", "labels"}
        other_keys = [k for k in features[0].keys() if k not in reserved]
        for k in other_keys:
            vals = [f[k] for f in features]
            if isinstance(vals[0], torch.Tensor):
                same = all(v.shape == vals[0].shape for v in vals)
                batch[k] = torch.stack(vals, dim=0) if same else vals
            else:
                batch[k] = vals
        return batch

# -------------------------
# LoRA helpers
# -------------------------

def _freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _count_trainable(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _print_trainable(model: torch.nn.Module) -> None:
    tr, tot = _count_trainable(model)
    pct = 100.0 * tr / max(tot, 1)
    print(f"Trainable params: {tr:,} / {tot:,} ({pct:.4f}%)")

# -------------------------
# Args
# -------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, default="nvidia/Cosmos-Reason2-8B")

    ap.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    ap.add_argument("--json_dir", type=str, default=JSON_DIR)
    ap.add_argument("--prompt_path", type=str, default=str(PROMPT_DIR))
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    ap.add_argument("--use_qlora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048 * 4)
    ap.add_argument("--max_label_tokens", type=int, default=512)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--print_trainable", action="store_true", default=True)

    return ap

# -------------------------
# Distributed helpers
# -------------------------

def _get_dist_env() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size


def _init_distributed_if_needed() -> Tuple[torch.device, int, int, bool]:
    """
    Initializes torch.distributed if WORLD_SIZE>1 using env:// rendezvous.
    Returns: (device, local_rank, world_size, is_distributed)
    """
    import torch.distributed as dist

    local_rank, world_size = _get_dist_env()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return device, local_rank, world_size, is_distributed


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

# -------------------------
# Main
# -------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = _build_parser()
    args = ap.parse_args(argv)

    device, local_rank, world_size, is_distributed = _init_distributed_if_needed()

    # Optional debug:
    import torch.distributed as dist
    print("RANK", os.environ.get("RANK"), "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
          "WORLD_SIZE", os.environ.get("WORLD_SIZE"), "dist_init", dist.is_initialized())

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
            attn_implementation=args.attn_impl,
        )
        model.to(device)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_impl,
        )
        model.to(device)

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        _freeze_all(model)

    tok = processor.tokenizer

    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    vocab = model.get_input_embeddings().weight.shape[0]
    assert tok.pad_token_id < vocab and tok.eos_token_id < vocab

    model.config.use_cache = False

    target_modules = [
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    if args.print_trainable and _is_rank0():
        _print_trainable(model)
        print("LoRA target_modules (leaf names):", target_modules)

    trainable, _ = _count_trainable(model)
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "LoRA target_modules likely do not match this checkpoint. "
            "Inspect model.named_modules() and adjust target_modules."
        )

    collator = DataCollatorQwenVL(processor=processor)

    training_args = SFTConfig(
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
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        packing=False,
        dataset_text_field="input_ids",
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    train_ds = VideoJsonDataset(
        examples=examples,
        processor=processor,
        prompt=prompt,
        num_frames=args.num_frames,
        max_prompt_tokens=args.max_prompt_tokens,
        max_label_tokens=args.max_label_tokens,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save adapter + processor only on rank 0
    if _is_rank0():
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    # Clean shutdown
    if is_distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

# -------------------------
# Internal spawn launcher
# -------------------------

def _spawn_worker(local_rank: int, world_size: int, argv: List[str]) -> None:
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    main(argv)


def _maybe_internal_spawn(argv: Optional[List[str]] = None) -> bool:
    """
    Internal launcher to avoid torchrun/elastic rendezvous segfaults.
    Controlled by INTERNAL_SPAWN=1 in the environment.
    """
    if os.environ.get("INTERNAL_SPAWN") != "1":
        return False
    if os.environ.get("WORLD_SIZE"):
        return False

    num_gpus = int(os.environ.get("NUM_GPUS", "0")) or torch.cuda.device_count()
    if num_gpus <= 1:
        return False

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.spawn(_spawn_worker, args=(num_gpus, argv or []), nprocs=num_gpus, join=True)
    return True


if __name__ == "__main__":
    if not _maybe_internal_spawn():
        main()
