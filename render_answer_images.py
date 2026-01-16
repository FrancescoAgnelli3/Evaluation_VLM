#!/usr/bin/env python3
"""
Create annotated images from model answer JSON files and summarize JSON outputs.

For every entry in demos/questions.json the script:
1. Loads the associated video frame.
2. Looks up answers from all `_answers.json` files in the results directory.
3. Creates an annotated image that contains the frame and every model's JSON output.
4. Prints all models' raw JSON outputs for comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = BASE_DIR / "frames_annotated"
DEFAULT_QUESTIONS = DEFAULT_FRAMES_DIR / "questions.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
DEFAULT_ANSWERS_DIR = DEFAULT_OUTPUT_DIR

FONT_LOCATIONS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render annotated answer images from JSON outputs."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help="Directory that contains annotated frames.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to questions.json file.",
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=DEFAULT_ANSWERS_DIR,
        help="Directory that contains answer JSON files for all models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotated answers will be stored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many questions to process.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Questions JSON must be a list, found {type(data)}")
    return data


def load_answer_map(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Answer file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Answer JSON must be a list, found {type(data)}")
    answer_map: Dict[str, Dict[str, Optional[str]]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        photo_id = entry.get("photo_id")
        if not photo_id:
            continue
        answer_map[str(photo_id)] = entry
    return answer_map


def discover_answer_files(answer_dir: Path) -> List[Tuple[str, Path]]:
    if not answer_dir.exists():
        raise FileNotFoundError(f"Answers directory not found: {answer_dir}")
    files = []
    for path in sorted(answer_dir.glob("*_answers.json")):
        if not path.is_file():
            continue
        name = path.name
        if not name.endswith("_answers.json"):
            continue
        if name == VLM_FORMAT_ERRORS_FILENAME:
            continue
        model_name = name[: -len("_answers.json")]
        files.append((model_name, path))
    return files


def load_all_answer_maps(answer_dir: Path) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Optional[str]]]]]:
    result: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    ordered_models: List[str] = []
    files = discover_answer_files(answer_dir)
    for model_name, path in files:
        try:
            result[model_name] = load_answer_map(path)
            ordered_models.append(model_name)
        except Exception as exc:
            logging.warning("Skipping %s due to error: %s", path, exc)
    return ordered_models, result


def resolve_image_path(frames_dir: Path, photo_id: str) -> Path:
    candidate = frames_dir / photo_id
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".png")
    if not candidate.exists():
        raise FileNotFoundError(f"Frame not found for photo_id '{photo_id}' -> {candidate}")
    return candidate


def init_font(size: int = 24) -> ImageFont.ImageFont:
    for font_path in FONT_LOCATIONS:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def estimate_char_limit(width_px: int, font: ImageFont.ImageFont) -> int:
    try:
        bbox = font.getbbox("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        avg_char_width = (bbox[2] - bbox[0]) / max(len("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 1)
    except Exception:
        avg_char_width = 8
    approx = width_px / max(avg_char_width, 5)
    return int(max(30, min(140, approx)))


def wrap_text(text: str, width_px: int, font: ImageFont.ImageFont) -> str:
    char_limit = estimate_char_limit(width_px, font)
    wrapper = textwrap.TextWrapper(width=char_limit, replace_whitespace=False)
    lines: List[str] = []
    for block in text.splitlines():
        stripped = block.strip()
        if not stripped:
            lines.append("")
            continue
        wrapped = wrapper.wrap(stripped)
        lines.extend(wrapped if wrapped else [""])
    return "\n".join(lines)


def measure_text_height(width: int, text: str, font: ImageFont.ImageFont) -> int:
    dummy = Image.new("RGB", (width, 10), color="white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((20, 0), text, font=font, spacing=6)
    return (bbox[3] - bbox[1]) + 20


def _format_answer_text(entry: Optional[Dict[str, Optional[str]]]) -> str:
    if not entry:
        return "No answer produced."
    answer = entry.get("answer")
    if answer is None:
        return "null"
    if isinstance(answer, str):
        cleaned = answer.strip()
        return cleaned if cleaned else "No answer produced."
    try:
        return json.dumps(answer, ensure_ascii=False)
    except Exception:
        return str(answer)


def build_text_block(
    question: str,
    photo_id: str,
    models: List[str],
    answer_maps: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> str:
    sections = [f"Question: {question.strip()}"]
    for model in models:
        entry = answer_maps.get(model, {}).get(photo_id)
        sections.append(f"{model}: {_format_answer_text(entry)}")
    return "\n\n".join(sections)


def compose_image(
    frame_path: Path,
    text: str,
    output_path: Path,
    font: ImageFont.ImageFont,
) -> None:
    frame_image = Image.open(frame_path).convert("RGB")
    wrapped_text = wrap_text(text, frame_image.width - 40, font)
    text_height = measure_text_height(frame_image.width, wrapped_text, font)
    combined = Image.new(
        "RGB",
        (frame_image.width, frame_image.height + text_height),
        color="white",
    )
    combined.paste(frame_image, (0, 0))
    draw = ImageDraw.Draw(combined)
    draw.multiline_text(
        (20, frame_image.height + 10),
        wrapped_text,
        font=font,
        fill=(0, 0, 0),
        spacing=6,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def process_questions(args: argparse.Namespace) -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    frames_dir: Path = args.frames_dir
    questions_path: Path = args.questions
    output_dir: Path = args.output_dir
    answers_dir: Path = args.answers_dir

    questions = load_questions(questions_path)
    models, answer_maps = load_all_answer_maps(answers_dir)
    if not models:
        raise FileNotFoundError(f"No answer JSON files were found in {answers_dir}")
    font = init_font()

    processed = 0
    for entry in questions:
        if args.limit is not None and processed >= args.limit:
            break

        photo_id = entry.get("photo_id")
        question = entry.get("question")
        if not photo_id or not question:
            logging.warning("Skipping malformed entry: %s", entry)
            continue

        try:
            image_path = resolve_image_path(frames_dir, photo_id)
        except FileNotFoundError as exc:
            logging.error(exc)
            continue

        text_block = build_text_block(question, str(photo_id), models, answer_maps)
        output_path = output_dir / f"{Path(photo_id).stem}_qa.png"
        compose_image(image_path, text_block, output_path, font)
        processed += 1

    logging.info("Processed %s entries and saved annotated images to %s", processed, output_dir)
    print_json_comparison(questions, models, answer_maps)


def print_json_comparison(
    questions: List[Dict[str, object]],
    models: List[str],
    answer_maps: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> None:
    if not questions:
        return
    print("\n=== Model JSON outputs ===")
    for entry in questions:
        photo_id = str(entry.get("photo_id", ""))
        question_id = entry.get("question_id")
        question = (str(entry.get("question") or "")).strip()
        print(f"\nQuestion {question_id} ({photo_id})")
        if question:
            print(f"Question text: {question}")
        for model in models:
            print(f"\n[{model}]")
            print(_format_answer_text(answer_maps.get(model, {}).get(photo_id)))


def main() -> None:
    args = parse_args()
    process_questions(args)


if __name__ == "__main__":
    main()
