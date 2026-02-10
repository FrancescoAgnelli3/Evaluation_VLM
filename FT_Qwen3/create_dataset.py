#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path


def collect_pairs(dataset_dir: Path):
    pairs = []
    for json_path in sorted(dataset_dir.glob("*.json")):
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            print(f"Skipping {json_path.name}: missing {mp4_path.name}")
            continue
        pairs.append((json_path, mp4_path))
    return pairs


def split_pairs(pairs, train_ratio, rng):
    rng.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    return pairs[:split_idx], pairs[split_idx:]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs, dest_dir: Path):
    for json_path, mp4_path in pairs:
        shutil.copy2(json_path, dest_dir / json_path.name)
        shutil.copy2(mp4_path, dest_dir / mp4_path.name)


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test folders with paired JSON+MP4 files."
    )
    parser.add_argument(
        "--dataset-dir",
        default="/projects/FT_VLM/dataset",
        help="Source dataset directory containing .json and .mp4 pairs.",
    )
    parser.add_argument(
        "--train-dir",
        default="/projects/FT_VLM/train_dataset",
        help="Output train dataset directory.",
    )
    parser.add_argument(
        "--test-dir",
        default="/projects/FT_VLM/test_dataset",
        help="Output test dataset directory.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of pairs to place in the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be between 0 and 1.")

    pairs = collect_pairs(dataset_dir)
    if not pairs:
        raise SystemExit("No valid JSON+MP4 pairs found.")

    rng = random.Random(args.seed)
    train_pairs, test_pairs = split_pairs(pairs, args.train_ratio, rng)

    ensure_dir(train_dir)
    ensure_dir(test_dir)

    copy_pairs(train_pairs, train_dir)
    copy_pairs(test_pairs, test_dir)

    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)} -> {train_dir}")
    print(f"Test pairs: {len(test_pairs)} -> {test_dir}")


if __name__ == "__main__":
    main()
