#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch.distributed as dist

import torch
from torch.utils.data import Dataset

from huggingface_hub import snapshot_download

from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from qwen_vl_utils import process_vision_info


BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = "/opt/dataset/train_dataset_17k"
JSON_DIR = "/opt/dataset/train_dataset_17k_json"
PROMPT_DIR = BASE_DIR / "prompts/prompt_json.txt"
OUTPUT_DIR = "/opt/models/Qwen/dataset_17k/"


# -------------------------
# Dataset helpers
# -------------------------

def _read_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _collect_examples(video_dir: str, json_dir: str) -> List[Dict[str, str]]:
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
        try:
            with open(jp, "r", encoding="utf-8") as f:
                label_obj = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in label file: {jp} ({e})")
            continue
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


def _as_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() == 2 and x.size(0) == 1:
        return x.squeeze(0)
    if x.dim() == 1:
        return x
    raise RuntimeError(f"Unexpected {name} shape: {tuple(x.shape)}")


def _squeeze_batch_if_hooking(v: torch.Tensor) -> torch.Tensor:
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
        self.prompt = prompt
        self.num_frames = num_frames
        self.max_prompt_tokens = max_prompt_tokens
        self.max_label_tokens = max_label_tokens

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
                raise RuntimeError("Example has multiple labels; expected 1.")
            label_json = label_json[0]

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

        prefix = self.processor(
            text=prefix_text,
            images=image_inputs,
            videos=video_tensors,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )
        prefix_ids = _as_1d(prefix["input_ids"], "prefix.input_ids")

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

        max_len = self.max_prompt_tokens + self.max_label_tokens
        prefix_len = prefix_ids.numel()

        if input_ids.numel() > max_len:
            if prefix_len >= max_len:
                raise RuntimeError(
                    f"Prefix alone ({prefix_len} tokens) exceeds max_len ({max_len}). "
                    f"Increase --max_prompt_tokens or reduce video tokens (lower num_frames/resolution)."
                )
            keep_prefix = prefix_len
            keep_suffix = max_len - keep_prefix
            input_ids = torch.cat([input_ids[:keep_prefix], input_ids[-keep_suffix:]], dim=0)
            attention_mask = torch.cat([attention_mask[:keep_prefix], attention_mask[-keep_suffix:]], dim=0)
            prefix_len = keep_prefix

        labels = input_ids.clone()
        labels[:prefix_len] = -100

        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        for k, v in full.items():
            if k in batch:
                continue
            if isinstance(v, torch.Tensor):
                batch[k] = _squeeze_batch_if_hooking(v)
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


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step={state.global_step} loss={logs['loss']:.6f}")


# -------------------------
# LoRA targeting utilities
# -------------------------

def _freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


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


def _dist_info() -> Tuple[int, int, int]:
    """
    torchrun sets these env vars. If not present, defaults to single process.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _init_distributed_from_torchrun() -> Tuple[torch.device, int, int, int, bool]:
    """
    Initializes torch.distributed if WORLD_SIZE>1. Expects torchrun-style env vars.
    Returns: (device, local_rank, rank, world_size, is_distributed)
    """
    import torch.distributed as dist

    local_rank, rank, world_size = _dist_info()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return device, local_rank, rank, world_size, is_distributed


def _resolve_lora_targets(model: torch.nn.Module, tune: str) -> List[str]:
    llm_candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    vision_candidates = ["qkv", "proj", "linear_fc1", "linear_fc2"]

    want = set()
    if tune in ("llm", "both"):
        want |= set(llm_candidates)
    if tune in ("vision", "both"):
        want |= set(vision_candidates)

    present_leaf_names = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in want:
            present_leaf_names.add(leaf)
    if present_leaf_names:
        return sorted(present_leaf_names)

    fallback = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if any(x in leaf for x in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            fallback.append(leaf)
    fallback = sorted(set(fallback))
    if fallback:
        return fallback

    all_linear = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
    return all_linear


def _rank0_download_or_use_local(model_id: str, cache_dir: str) -> str:
    """
    If model_id is a local directory -> return it.
    If it's a HF repo id -> only rank0 downloads to cache_dir; others wait; return cache_dir.
    """
    import torch.distributed as dist

    if os.path.isdir(model_id):
        return model_id

    if _is_rank0():
        snapshot_download(
            repo_id=model_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return cache_dir


# -------------------------
# Main
# -------------------------

def main(argv: Optional[List[str]] = None):
    import torch.distributed as dist
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, default="/opt/models/Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    ap.add_argument("--json_dir", type=str, default=JSON_DIR)
    ap.add_argument("--prompt_path", type=str, default=str(PROMPT_DIR))
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    ap.add_argument("--tune", type=str, choices=["llm", "vision", "both"], default="both")

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
    ap.add_argument("--num_train_epochs", type=float, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.set_defaults(bf16=True)

    ap.add_argument("--attn_impl", type=str, default="eager", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.set_defaults(gradient_checkpointing=True)

    ap.add_argument("--print_trainable", action="store_true")
    ap.add_argument("--no_print_trainable", dest="print_trainable", action="store_false")
    ap.set_defaults(print_trainable=True)

    args = ap.parse_args(argv)

    device, local_rank, rank, world_size, is_distributed = _init_distributed_from_torchrun()

    # -------------------------
    # Rank0-only download (avoids multi-process shard races)
    # -------------------------
    cache_dir = os.path.join(args.output_dir, "hf_cache_model")
    args.model_id = _rank0_download_or_use_local(args.model_id, cache_dir)

    prompt = _read_prompt(args.prompt_path)

    if _is_rank0():
        examples = _collect_examples(args.video_dir, args.json_dir)
    else:
        examples = None

    # Broadcast examples to all ranks (avoid each rank scanning filesystem)
    if is_distributed:
        obj_list = [examples]
        dist.broadcast_object_list(obj_list, src=0)
        examples = obj_list[0]

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=False)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False)

    if getattr(config, "text_config", None) is not None:
        if hasattr(config.text_config, "rope_scaling") and config.text_config.rope_scaling is None:
            config.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}
        if not hasattr(config.text_config, "intermediate_size") and hasattr(config.text_config, "moe_intermediate_size"):
            config.text_config.intermediate_size = config.text_config.moe_intermediate_size

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            quantization_config=bnb_config,
            attn_implementation=args.attn_impl,
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
        # QLoRA still needs a device move for modules that stay in torch (bnb handles quantized weights)
        model.to(device)
    else:
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_impl,
            trust_remote_code=False,
        )
        # IMPORTANT:
        # If you are using DeepSpeed ZeRO-3 (as in TrainingArguments below),
        # do NOT call model.to(device) here; DeepSpeed will place/shard parameters.
        # If you are NOT using DeepSpeed, uncomment the next line (will OOM for 397B):
        # model.to(device)

        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        _freeze_all(model)

    model.config.use_cache = False

    target_modules = _resolve_lora_targets(model, args.tune)
    if not target_modules:
        raise RuntimeError("Could not infer target_modules for LoRA.")

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

    trainable, total = _count_trainable(model)
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "LoRA target_modules did not match any submodules."
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
        deepspeed="ds_zero3.json",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        remove_unused_columns=False,
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

    if _is_rank0():
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if is_distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()