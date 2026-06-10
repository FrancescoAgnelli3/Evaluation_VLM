#!/usr/bin/env python3
import argparse
import math
import random
import re
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a camera-balanced subsample from test_dataset/test_dataset_json. "
            "Camera is the first number in the filename."
        )
    )
    parser.add_argument(
        "--mp4-dir",
        type=Path,
        default=Path("/opt/dataset/test_dataset"),
        help="Directory containing .mp4 files (default /opt/dataset/test_dataset).",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("/opt/dataset/test_dataset_json"),
        help="Directory containing .json files (default /opt/dataset/test_dataset_json).",
    )
    parser.add_argument(
        "--out-mp4-dir",
        type=Path,
        default=Path("/opt/dataset/test_dataset_subsample"),
        help="Output directory for subsampled .mp4 files.",
    )
    parser.add_argument(
        "--out-json-dir",
        type=Path,
        default=Path("/opt/dataset/test_dataset_subsample_json"),
        help="Output directory for subsampled .json files.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=3000,
        help="Number of pairs to sample (default 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def camera_from_stem(stem: str) -> str:
    match = re.match(r"^(\d+)", stem)
    if match:
        return match.group(1)
    return "unknown"


def allocate_targets(group_sizes, total_target):
    total_available = sum(group_sizes.values())
    if total_target >= total_available:
        return {k: v for k, v in group_sizes.items()}, total_available

    exact_targets = {
        k: (total_target * size / total_available) for k, size in group_sizes.items()
    }
    base_targets = {k: int(math.floor(v)) for k, v in exact_targets.items()}
    remainder = total_target - sum(base_targets.values())

    if remainder > 0:
        fractions = sorted(
            exact_targets.items(), key=lambda kv: (kv[1] - base_targets[kv[0]]), reverse=True
        )
        for k, _ in fractions:
            if remainder == 0:
                break
            if base_targets[k] < group_sizes[k]:
                base_targets[k] += 1
                remainder -= 1

    if remainder > 0:
        for k in sorted(group_sizes.keys()):
            if remainder == 0:
                break
            available = group_sizes[k] - base_targets[k]
            if available <= 0:
                continue
            take = min(available, remainder)
            base_targets[k] += take
            remainder -= take

    return base_targets, total_target


def main():
    args = parse_args()

    mp4_files = {p.stem: p for p in args.mp4_dir.glob("*.mp4")}
    json_files = {p.stem: p for p in args.json_dir.glob("*.json")}
    common_stems = sorted(set(mp4_files.keys()) & set(json_files.keys()))

    if not common_stems:
        raise SystemExit("No matching mp4/json pairs found by filename stem.")

    groups = {}
    for stem in common_stems:
        cam = camera_from_stem(stem)
        groups.setdefault(cam, []).append(stem)

    group_sizes = {k: len(v) for k, v in groups.items()}
    targets, target_total = allocate_targets(group_sizes, args.size)

    random.seed(args.seed)
    selected_stems = []
    for cam, stems in groups.items():
        random.shuffle(stems)
        take = targets.get(cam, 0)
        selected_stems.extend(stems[:take])

    ensure_dir(args.out_mp4_dir)
    ensure_dir(args.out_json_dir)

    op = shutil.move if args.move else shutil.copy2

    for stem in selected_stems:
        op(mp4_files[stem], args.out_mp4_dir / mp4_files[stem].name)
        op(json_files[stem], args.out_json_dir / json_files[stem].name)

    print(f"Total pairs available: {len(common_stems)}")
    print(f"Subsampled pairs: {len(selected_stems)} (requested {args.size})")
    print("Camera distribution (available -> sampled):")
    for cam in sorted(group_sizes.keys(), key=lambda x: (x == "unknown", x)):
        print(f"  {cam}: {group_sizes[cam]} -> {targets.get(cam, 0)}")
    if "unknown" in group_sizes:
        print("Warning: some filenames did not start with a number; grouped under 'unknown'.")
    if args.move:
        print("Mode: move")
    else:
        print("Mode: copy")


if __name__ == "__main__":
    main()
