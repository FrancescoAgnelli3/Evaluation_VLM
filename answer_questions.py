#!/usr/bin/env python3
"""
Road-safety perception prompt runner using vLLM (OpenAI-compatible).

Behavior:
- For each video found under --media-dir, send video + PROMPT_PERCEPTION_JSON.
- Save the model's JSON response directly to disk (one JSON file per call).

Notes:
- Keeps Hugging Face authentication exactly as-is.
- vLLM loads models from Hugging Face repo IDs.
- Forces JSON mode for the prompt call.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure HF/torch caches are redirected before anything that may touch HF.
os.environ.setdefault("HF_HOME", "/mnt/ssd1/hf")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/ssd1/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/ssd1/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/ssd1/hf/datasets")
os.environ.setdefault("TORCH_HOME", "/mnt/ssd1/torch")

from huggingface_hub import login  # noqa: E402

from vllm_utils import (
    DEFAULT_MODEL_SELECTION,
    MODEL_CHOICES,
    ensure_clients,
    shutdown_client,
    shutdown_vllm_server,
)

login(token=os.environ["HF_TOKEN"])


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_DIR = "/mnt/ssd1/dataset_ft_VLM/dataset_test"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"

# ----------------------------
# Prompt A: Perception-only JSON
# ----------------------------

PROMPT_PERCEPTION_JSON = (BASE_DIR / "prompts" / "perception_prompt.txt").read_text(encoding="utf-8").strip()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ----------------------------
# CLI / IO helpers
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perception->Deterministic risk pipeline with vLLM-served VLMs (video-only).")
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        choices=MODEL_CHOICES,
        help="Vision-language model(s) to run. Provide multiple times; default is cosmos2-2B.",
    )
    args = parser.parse_args()
    if not args.model:
        args.model = [DEFAULT_MODEL_SELECTION]
    if "all" in args.model:
        args.model = [m for m in MODEL_CHOICES if m != "all"]
    return args


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "result"


def _build_output_basename(question_id: Optional[str], photo_id: str) -> str:
    base_prefix = f"{question_id or 'question'}_{Path(str(photo_id)).stem}"
    return _sanitize_filename(base_prefix)


# ----------------------------
# Main processing loop
# ----------------------------

def process_questions(args: argparse.Namespace) -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    media_dir: Path = args.media_dir
    output_dir: Path = args.output_dir
    if not media_dir.exists():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")
    media_paths = sorted(
        [
            path
            for path in media_dir.rglob("*")
            if path.is_file() and _is_video(path)
        ]
    )

    selected_models: List[str] = []
    for model_key in args.model:
        if model_key not in selected_models:
            selected_models.append(model_key)

    processed_total = 0

    try:
        max_samples = args.samples if args.samples is not None else args.limit

        for model_key in selected_models:
            model_clients = ensure_clients([model_key])
            client = model_clients.get(model_key)
            if client is None:
                continue

            processed = 0
            for media_path in media_paths:
                if max_samples is not None and processed >= max_samples:
                    break

                photo_id = media_path.relative_to(media_dir).as_posix()
                logging.info("Processing media=%s with model=%s", photo_id, model_key)

                # ----------------------------
                # Prompt: Perception JSON
                # ----------------------------
                stage1 = client.run_video_inference_json(media_path, PROMPT_PERCEPTION_JSON)
                output_obj: Dict[str, Any]
                if stage1 is None or not stage1.response_text:
                    output_obj = {"error": "inference_failed", "raw_text": stage1.response_text if stage1 else ""}
                else:
                    try:
                        output_obj = json.loads(stage1.response_text)
                    except Exception as exc:
                        output_obj = {"error": f"json_parse_failed: {exc}", "raw_text": stage1.response_text}

                base_name = _build_output_basename(None, str(photo_id))
                model_name = model_key.replace("-", "_")

                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{base_name}_{model_name}.json"
                out_path.write_text(json.dumps(output_obj, ensure_ascii=False, indent=2), encoding="utf-8")

                processed += 1
                processed_total += 1

            shutdown_client(client)
            del client

            # One-server-at-a-time: stop between models to avoid port collisions and ensure correct model loaded.
            shutdown_vllm_server()

        logging.info("Processed %s entries", processed_total)
    finally:
        shutdown_vllm_server()


def main() -> None:
    args = parse_args()
    process_questions(args)


if __name__ == "__main__":
    main()
