#!/usr/bin/env python3
"""Count results.json files per model under the results directory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re


_TIME_PART_RE = re.compile(r"^[0-9]+m[0-9]+s$")


def _extract_model_name(filename: str) -> str | None:
    """
    Extract model name from a filename stem.
    Expected pattern example:
      question_..._06m07s_qwen_32B.json -> qwen_32B
      question_..._06m07s_qwen_8B.json  -> qwen_8B
      question_..._06m07s_qwen_8B_FT_both_1k.json -> qwen_8B_FT_both_1k
      question_..._06m07s_cosmos1.json  -> cosmos1
    """
    parts = filename.split("_")
    if not parts:
        return None

    # Model name starts after the last time token like 06m07s.
    last_time_idx = None
    for i, part in enumerate(parts):
        if _TIME_PART_RE.match(part):
            last_time_idx = i

    if last_time_idx is not None and last_time_idx + 1 < len(parts):
        return "_".join(parts[last_time_idx + 1 :])

    # Fallback: use the last token.
    return parts[-1]


def count_results(base_dir: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    for path in base_dir.rglob("*.json"):
        if not path.is_file():
            continue
        model = _extract_model_name(path.stem)
        if not model:
            continue
        counter[model] += 1
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count the number of results JSON files per model.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to the results directory (default: ./results)",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir).expanduser().resolve()
    if not base_dir.exists():
        print(f"results directory not found: {base_dir}")
        return 1

    counts = count_results(base_dir)
    if not counts:
        print("No results.json files found.")
        return 0

    total = sum(counts.values())
    print(f"Results directory: {base_dir}")
    print("Counts per model:")
    for model in sorted(counts):
        print(f"  {model}: {counts[model]}")
    print(f"Total results.json files: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
