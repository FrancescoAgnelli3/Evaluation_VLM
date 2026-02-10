#!/usr/bin/env python3
"""
Compute average inference time per model over demo videos.

Default behavior:
- Uses videos under ./demos
- Runs all models supported by answer_questions.py (MODEL_CHOICES excluding "all")
- Writes a txt summary of average elapsed seconds per model
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure HF/torch caches are redirected before anything that may touch HF.
os.environ.setdefault("HF_HOME", "/mnt/ssd1/hf")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/ssd1/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/ssd1/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/ssd1/hf/datasets")
os.environ.setdefault("TORCH_HOME", "/mnt/ssd1/torch")

from huggingface_hub import login  # noqa: E402

from vllm_utils import (  # noqa: E402
    ensure_clients,
    shutdown_client,
    shutdown_vllm_server,
)

MODEL_CHOICES = ("qwen-2B", "qwen-8B", "qwen-32B", "cosmos1", "cosmos2-2B", "cosmos2-8B", "all")

login(token=os.environ["HF_TOKEN"])

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_DIR = BASE_DIR / "input_videos"
DEFAULT_OUTPUT_PATH = BASE_DIR / "eval_out" / "average_inference_times.txt"
PROMPT_PERCEPTION_JSON = (BASE_DIR / "prompts" / "perception_prompt.txt").read_text(encoding="utf-8").strip()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average inference time per model on demo videos.")
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        choices=MODEL_CHOICES,
        help="Vision-language model(s) to run. Provide multiple times; default is all models.",
    )
    args = parser.parse_args()
    if not args.model:
        args.model = ["all"]
    if "all" in args.model:
        args.model = [m for m in MODEL_CHOICES if m != "all"]
    return args


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _collect_media(media_dir: Path) -> List[Path]:
    if not media_dir.exists():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")
    return sorted([path for path in media_dir.rglob("*") if path.is_file() and _is_video(path)])


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def process(args: argparse.Namespace) -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    media_paths = _collect_media(args.media_dir)
    if not media_paths:
        raise RuntimeError(f"No video files found under: {args.media_dir}")

    max_samples = args.samples if args.samples is not None else args.limit

    results: Dict[str, Tuple[Optional[float], int, int, Optional[str]]] = {}

    try:
        for model_key in args.model:
            model_clients = ensure_clients([model_key])
            client = model_clients.get(model_key)
            if client is None:
                results[model_key] = (None, 0, len(media_paths), "init_failed")
                continue

            elapsed_values: List[float] = []
            processed = 0
            total_target = min(len(media_paths), max_samples) if max_samples is not None else len(media_paths)

            for media_path in media_paths:
                if max_samples is not None and processed >= max_samples:
                    break

                logging.info("Timing media=%s with model=%s", media_path.name, model_key)
                result = client.run_video_inference_json(media_path, PROMPT_PERCEPTION_JSON)
                if result is not None:
                    elapsed_values.append(result.elapsed_s)
                processed += 1

            avg_s = _avg(elapsed_values)
            results[model_key] = (avg_s, len(elapsed_values), total_target, None)

            shutdown_client(client)
            del client

            # One-server-at-a-time: stop between models to avoid port collisions and ensure correct model loaded.
            shutdown_vllm_server()
    finally:
        shutdown_vllm_server()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"media_dir: {args.media_dir}")
    lines.append("model\tavg_elapsed_s\tsuccess/total\tnotes")
    for model_key in args.model:
        avg_s, success_count, total_count, note = results.get(model_key, (None, 0, 0, "missing"))
        avg_str = f"{avg_s:.4f}" if avg_s is not None else "NA"
        note_str = note or ""
        lines.append(f"{model_key}\t{avg_str}\t{success_count}/{total_count}\t{note_str}")

    args.output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote average inference times to %s", args.output_file)


def main() -> None:
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
