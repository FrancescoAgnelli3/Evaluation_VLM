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
from typing import Dict, List, Optional, Tuple

# # Ensure HF/torch caches are redirected before anything that may touch HF.
# os.environ.setdefault("HF_HOME", "/mnt/Repo/hf")
# os.environ.setdefault("HF_HUB_CACHE", "/mnt/Repo/hf/hub")
# os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/Repo/hf/transformers")
# os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/Repo/hf/datasets")
# os.environ.setdefault("TORCH_HOME", "/mnt/Repo/torch")

from huggingface_hub import login  # noqa: E402

from utils.vllm_utils import (  # noqa: E402
    DEFAULT_MODEL_SELECTION,
    MODEL_CHOICES,
    VLLMClientFactory,
    ensure_vllm_server,
    shutdown_vllm_server,
)

login(token=os.environ["HF_TOKEN"])


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_DIR = "/opt/dataset/test_dataset"
TASK_MEDIA_DIRS = {
    "road": "/opt/dataset/test_dataset",
    "people": "/opt/dataset/ds_people/test_dataset",
    "environment": "/opt/dataset/ds_environment/test_dataset",
    "industry": "/opt/dataset/ds_industry/test_dataset",
}
DEFAULT_TASK = "road"
TASK_PROMPTS = {
    "road": BASE_DIR / "prompts" / "prompt_road.txt",
    "people": BASE_DIR / "prompts" / "prompt_people.txt",
    "environment": BASE_DIR / "prompts" / "prompt_environment.txt",
    "industry": BASE_DIR / "prompts" / "prompt_industry.txt",
}

# ----------------------------
# Prompt A: Perception-only JSON
# ----------------------------

PROMPT_TEXTS = {
    task: path.read_text(encoding="utf-8").strip() for task, path in TASK_PROMPTS.items()
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ----------------------------
# CLI / IO helpers
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perception->Deterministic risk pipeline with vLLM-served VLMs (video-only)."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_PROMPTS.keys()),
        default=DEFAULT_TASK,
        help="Selects the prompt and default output folder.",
    )
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
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
        help=(
            "Vision-language model(s) to run. Provide multiple times; default is cosmos2-2B. "
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = BASE_DIR / f"results_{args.task}"
    if args.media_dir == Path(DEFAULT_MEDIA_DIR):
        task_media = TASK_MEDIA_DIRS.get(args.task)
        if task_media:
            args.media_dir = Path(task_media)
    args.prompt_text = PROMPT_TEXTS[args.task]
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

def _output_path(
    media_path: Path,
    media_dir: Path,
    output_dir: Path,
    model_key: str,
) -> Path:
    photo_id = media_path.relative_to(media_dir).as_posix()
    base_name = _build_output_basename(None, str(photo_id))
    model_name = model_key.replace("-", "_")
    return output_dir / f"{base_name}_{model_name}.json"


# ----------------------------
# Inference worker
# ----------------------------

def _infer_one(
    client_factory: VLLMClientFactory,
    media_path: Path,
    media_dir: Path,
    output_dir: Path,
    model_key: str,
    prompt_text: str,
) -> Tuple[Path, bool, Optional[str]]:
    photo_id = media_path.relative_to(media_dir).as_posix()

    client = client_factory.get_client()
    stage1 = client.run_video_inference_json(media_path, prompt_text)

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
    try:
        out_path.write_text(
            json.dumps(output_obj, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return out_path, True, None
    except Exception as exc:
        return out_path, False, f"write_failed: {exc}"


# ----------------------------
# Main processing loop
# ----------------------------

def process_questions(args: argparse.Namespace) -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    media_dir: Path = args.media_dir
    output_dir: Path = args.output_dir
    if not media_dir.exists():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")

    media_paths = [p for p in media_dir.rglob("*") if p.is_file() and _is_video(p)]

    selected_models: List[str] = []
    for model_key in args.model:
        if model_key not in selected_models:
            selected_models.append(model_key)

    processed_total = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        max_samples = args.samples if args.samples is not None else args.limit

        for model_key in selected_models:
            manager = ensure_vllm_server(model_key)
            client_factory = VLLMClientFactory.from_server(manager, model_key)

            # Select the subset for this run.
            if max_samples is None:
                this_media = media_paths
            else:
                this_media = media_paths[:max_samples]

            logging.info("Model=%s | videos=%d | workers=%d", model_key, len(this_media), args.workers)

            processed = 0
            skipped = 0

            if args.workers == 1:
                # Original serial behavior.
                for media_path in this_media:
                    photo_id = media_path.relative_to(media_dir).as_posix()
                    out_path = _output_path(media_path, media_dir, output_dir, model_key)
                    if out_path.exists():
                        logging.info("Skip existing result for media=%s with model=%s", photo_id, model_key)
                        skipped += 1
                        continue
                    logging.info("Processing media=%s with model=%s", photo_id, model_key)

                    out_path, ok, err = _infer_one(
                        client_factory,
                        media_path,
                        media_dir,
                        output_dir,
                        model_key,
                        args.prompt_text,
                    )
                    if not ok:
                        logging.info("Failed write for media=%s with model=%s: %s", photo_id, model_key, err)
                    processed += 1
                    processed_total += 1
            else:
                # Parallel requests against the same vLLM server.
                # Each worker gets a separate HTTP client via the factory (per-thread session).
                pending_media = []
                for media_path in this_media:
                    out_path = _output_path(media_path, media_dir, output_dir, model_key)
                    if out_path.exists():
                        photo_id = media_path.relative_to(media_dir).as_posix()
                        logging.info("Skip existing result for media=%s with model=%s", photo_id, model_key)
                        skipped += 1
                        continue
                    pending_media.append(media_path)

                with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    future_to_media = {}
                    max_in_flight = max(1, args.workers * 2)
                    pending_iter = iter(pending_media)

                    def submit_next() -> None:
                        try:
                            media_path = next(pending_iter)
                        except StopIteration:
                            return
                        fut = ex.submit(
                            _infer_one,
                            client_factory,
                            media_path,
                            media_dir,
                            output_dir,
                            model_key,
                            args.prompt_text,
                        )
                        future_to_media[fut] = media_path

                    for _ in range(min(max_in_flight, len(pending_media))):
                        submit_next()

                    while future_to_media:
                        done, _ = cf.wait(
                            future_to_media,
                            return_when=cf.FIRST_COMPLETED,
                        )
                        for fut in done:
                            media_path = future_to_media.pop(fut)
                            photo_id = media_path.relative_to(media_dir).as_posix()
                            try:
                                out_path, ok, err = fut.result()
                                if ok:
                                    logging.info("Done media=%s with model=%s", photo_id, model_key)
                                else:
                                    logging.info(
                                        "Failed write for media=%s with model=%s: %s",
                                        photo_id,
                                        model_key,
                                        err,
                                    )
                                processed += 1
                                processed_total += 1
                            except Exception as exc:
                                logging.info("Failed media=%s with model=%s: %s", photo_id, model_key, exc)
                                # Still emit a JSON error file to keep bookkeeping consistent.
                                base_name = _build_output_basename(None, str(photo_id))
                                model_name = model_key.replace("-", "_")
                                out_path = output_dir / f"{base_name}_{model_name}.json"
                                out_path.write_text(
                                    json.dumps({"error": f"exception: {exc}"}, ensure_ascii=False, separators=(",", ":")),
                                    encoding="utf-8",
                                )
                                processed += 1
                                processed_total += 1
                            submit_next()

            client_factory.close()
            del client_factory

            # One-server-at-a-time: stop between models to avoid port collisions and ensure correct model loaded.
            shutdown_vllm_server()
            logging.info(
                "Completed model=%s | processed=%d | skipped=%d",
                model_key,
                processed,
                skipped,
            )

        logging.info("Processed %s entries", processed_total)
    finally:
        shutdown_vllm_server()


def main() -> None:
    args = parse_args()
    process_questions(args)


if __name__ == "__main__":
    main()
