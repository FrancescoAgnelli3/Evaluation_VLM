#!/usr/bin/env python3
"""
Build per-video Gemini prompt outputs (perception-only JSON).

Behavior:
- N runs per video (to reduce model variance).
- Each run sends the prompt as-is and parses the JSON response.
- Each run is saved to disk as JSON (one file per call).

Dependencies:
  pip install google-genai

Auth:
  export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY

Usage:
  python build_fixed_targets_gemini_perception.py \
    --videos-dir /path/to/videos \
    --out-dir /path/to/targets \
    --runs 5 \
    --model gemini-flash-latest
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_DIR = BASE_DIR / "demos"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results_gold"


# ----------------------------
# Perception prompt (Task A)
# ----------------------------

PROMPT_PERCEPTION_JSON = (BASE_DIR / "prompts" / "perception_prompt.txt").read_text(encoding="utf-8").strip()

# ----------------------------
# Gemini Files API: wait for ACTIVE
# ----------------------------

def wait_until_file_active(
    client: genai.Client,
    uploaded_file: types.File,
    *,
    timeout_s: int = 600,
    poll_s: float = 2.0,
) -> types.File:
    t0 = time.time()
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "id", None)
    if not file_name:
        raise RuntimeError("Cannot determine uploaded file name/id for polling")

    while True:
        f = client.files.get(name=file_name)
        state = getattr(f, "state", None)
        if state == "ACTIVE":
            return f
        if state == "FAILED":
            err = getattr(f, "error", None)
            raise RuntimeError(f"File processing FAILED: {err}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"File did not become ACTIVE within {timeout_s}s (last state={state})")
        time.sleep(poll_s)




# ----------------------------
# Gemini call (Stage 1)
# ----------------------------

def gemini_generate_perception(
    client: genai.Client,
    model: str,
    video_file: types.File,
    prompt: str,
) -> Dict[str, Any]:
    resp = client.models.generate_content(
        model=model,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    text = getattr(resp, "text", "")
    if not text:
        raise RuntimeError("Empty Gemini response")
    return json.loads(text)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", type=str, default="gemini-flash-latest")
    ap.add_argument("--extensions", type=str, default=".mp4,.mov,.avi,.mkv,.webm")
    ap.add_argument("--file-active-timeout", type=int, default=900)
    ap.add_argument("--poll-seconds", type=float, default=2.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    videos = sorted([p for p in args.videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ["GEMINI_API_KEY"]
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment")
    client = genai.Client(api_key=api_key)

    try:
        for vp in videos:
            # Upload and wait for ACTIVE
            try:
                vf = client.files.upload(file=str(vp))
                vf = wait_until_file_active(
                    client,
                    vf,
                    timeout_s=args.file_active_timeout,
                    poll_s=args.poll_seconds,
                )
            except Exception as e:
                error_path = args.out_dir / f"{vp.stem}.teacher.perception_raw.run_1.json"
                error_path.write_text(
                    json.dumps(
                        {
                            "video": str(vp),
                            "model": args.model,
                            "stage": "upload/activate",
                            "error": f"{type(e).__name__}: {e}",
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                continue

            # Prompt runs: perception JSON only
            for i in range(args.runs):
                raw_path = args.out_dir / f"{vp.stem}.teacher.perception_raw.run_{i + 1}.json"

                t0 = time.time()
                try:
                    d = gemini_generate_perception(client, args.model, vf, PROMPT_PERCEPTION_JSON)
                    raw_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
                except (json.JSONDecodeError, RuntimeError) as e:
                    raw_path.write_text(
                        json.dumps(
                            {
                                "run": i + 1,
                                "stage": "perception_generate_or_parse",
                                "error": f"{type(e).__name__}: {e}",
                                "elapsed_s": time.time() - t0,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

    finally:
        client.close()


if __name__ == "__main__":
    main()
