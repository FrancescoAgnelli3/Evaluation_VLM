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
- Optional per-model parallelism via --workers (multiple concurrent requests to the same vLLM server).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure HF/torch caches are redirected before anything that may touch HF.
os.environ.setdefault("HF_HOME", "/mnt/Repo/hf")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/Repo/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/Repo/hf/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/Repo/hf/datasets")
os.environ.setdefault("TORCH_HOME", "/mnt/Repo/torch")

from huggingface_hub import login  # noqa: E402

from vllm_utils import (  # noqa: E402
    DEFAULT_MODEL_SELECTION,
    MODEL_CHOICES,
    ensure_clients,
    shutdown_client,
    shutdown_vllm_server,
)

login(token=os.environ["HF_TOKEN"])


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_DIR = "/mnt/Repo/VLM_ft/dataset_test_out"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"

# ----------------------------
# Prompt A: Perception-only JSON
# ----------------------------

PROMPT_PERCEPTION_JSON = (BASE_DIR / "prompts" / "perception_prompt.txt").read_text(
    encoding="utf-8"
).strip()

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ----------------------------
# CLI / IO helpers
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perception->Deterministic risk pipeline with vLLM-served VLMs (video-only)."
    )
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent in-flight requests per model (same vLLM server). Start small (2-4).",
    )
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
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    return args


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "result"


def _build_output_basename(question_id: Optional[str], photo_id: str) -> str:
    base_prefix = f"{question_id or 'question'}_{Path(str(photo_id)).stem}"
    return _sanitize_filename(base_prefix)


# ----------------------------
# Inference worker
# ----------------------------

def _infer_one(
    client: Any,
    media_path: Path,
    media_dir: Path,
    output_dir: Path,
    model_key: str,
) -> Tuple[Path, Dict[str, Any]]:
    photo_id = media_path.relative_to(media_dir).as_posix()

    stage1 = client.run_video_inference_json(media_path, PROMPT_PERCEPTION_JSON)

    if stage1 is None or not getattr(stage1, "response_text", None):
        output_obj: Dict[str, Any] = {
            "error": "inference_failed",
            "raw_text": stage1.response_text if stage1 else "",
        }
    else:
        try:
            output_obj = json.loads(stage1.response_text)
        except Exception as exc:
            output_obj = {"error": f"json_parse_failed: {exc}", "raw_text": stage1.response_text}

    base_name = _build_output_basename(None, str(photo_id))
    model_name = model_key.replace("-", "_")
    out_path = output_dir / f"{base_name}_{model_name}.json"
    return out_path, output_obj


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
        [p for p in media_dir.rglob("*") if p.is_file() and _is_video(p)]
    )

    selected_models: List[str] = []
    for model_key in args.model:
        if model_key not in selected_models:
            selected_models.append(model_key)

    processed_total = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        max_samples = args.samples if args.samples is not None else args.limit

        for model_key in selected_models:
            model_clients = ensure_clients([model_key])
            client = model_clients.get(model_key)
            if client is None:
                continue

            # Select the subset for this run.
            if max_samples is None:
                this_media = media_paths
            else:
                this_media = media_paths[:max_samples]

            logging.info("Model=%s | videos=%d | workers=%d", model_key, len(this_media), args.workers)

            processed = 0

            if args.workers == 1:
                # Original serial behavior.
                for media_path in this_media:
                    photo_id = media_path.relative_to(media_dir).as_posix()
                    logging.info("Processing media=%s with model=%s", photo_id, model_key)

                    out_path, output_obj = _infer_one(client, media_path, media_dir, output_dir, model_key)
                    out_path.write_text(
                        json.dumps(output_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    processed += 1
                    processed_total += 1
            else:
                # Parallel requests against the same vLLM server.
                # If your client wrapper is NOT thread-safe, the safer pattern is to instantiate a separate
                # lightweight HTTP client per worker. With the current wrapper, try this first; if you see
                # weird errors, rework vllm_utils so each worker has its own client instance.
                with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    future_to_media = {
                        ex.submit(_infer_one, client, media_path, media_dir, output_dir, model_key): media_path
                        for media_path in this_media
                    }

                    for fut in cf.as_completed(future_to_media):
                        media_path = future_to_media[fut]
                        photo_id = media_path.relative_to(media_dir).as_posix()
                        try:
                            out_path, output_obj = fut.result()
                            out_path.write_text(
                                json.dumps(output_obj, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            logging.info("Done media=%s with model=%s", photo_id, model_key)
                            processed += 1
                            processed_total += 1
                        except Exception as exc:
                            logging.info("Failed media=%s with model=%s: %s", photo_id, model_key, exc)
                            # Still emit a JSON error file to keep bookkeeping consistent.
                            base_name = _build_output_basename(None, str(photo_id))
                            model_name = model_key.replace("-", "_")
                            out_path = output_dir / f"{base_name}_{model_name}.json"
                            out_path.write_text(
                                json.dumps({"error": f"exception: {exc}"}, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            processed += 1
                            processed_total += 1

            shutdown_client(client)
            del client

            # One-server-at-a-time: stop between models to avoid port collisions and ensure correct model loaded.
            shutdown_vllm_server()
            logging.info("Completed model=%s | processed=%d", model_key, processed)

        logging.info("Processed %s entries", processed_total)
    finally:
        shutdown_vllm_server()


def main() -> None:
    args = parse_args()
    process_questions(args)


if __name__ == "__main__":
    main()
