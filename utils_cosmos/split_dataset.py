#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split paired mp4/json dataset into train/test folders. "
            "Pairs are matched by filename stem."
        )
    )
    parser.add_argument(
        "--mp4-dir",
        type=Path,
        default=Path("/opt/dataset/ds_people/Dataset_training_v1_batch_1"),
        help="Directory containing .mp4 files.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("/opt/dataset/ds_people/Dataset_training_v1_batch_1_json_final_prompt"),
        help="Directory containing .json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/opt/dataset/ds_people"),
        help="Base output directory for split folders.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train split fraction (default 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default 42).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def place_pair(src: Path, dst: Path, mode: str):
    if dst.is_symlink() or dst.exists():
        if mode == "symlink" and dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src):
            return
        if dst.is_dir():
            raise IsADirectoryError(f"Destination is a directory: {dst}")
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "move":
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    if not (0.0 < args.split < 1.0):
        raise SystemExit("--split must be between 0 and 1 (exclusive).")

    mp4_dir = args.mp4_dir
    json_dir = args.json_dir
    out_dir = args.out_dir

    mp4_files = {p.stem: p for p in mp4_dir.glob("*.mp4")}
    json_files = {p.stem: p for p in json_dir.glob("*.json")}

    common_stems = sorted(set(mp4_files.keys()) & set(json_files.keys()))
    if not common_stems:
        raise SystemExit("No matching mp4/json pairs found by filename stem.")

    random.seed(args.seed)
    random.shuffle(common_stems)

    train_count = int(len(common_stems) * args.split)
    train_stems = common_stems[:train_count]
    test_stems = common_stems[train_count:]

    train_mp4_dir = out_dir / "train_dataset"
    train_json_dir = out_dir / "train_dataset_json"
    test_mp4_dir = out_dir / "test_dataset"
    test_json_dir = out_dir / "test_dataset_json"

    for d in (train_mp4_dir, train_json_dir, test_mp4_dir, test_json_dir):
        ensure_dir(d)

    if args.move and args.copy:
        raise SystemExit("--move and --copy are mutually exclusive.")
    if args.move:
        mode = "move"
    elif args.copy:
        mode = "copy"
    else:
        mode = "symlink"

    def copy_pair(stem, mp4_target_dir, json_target_dir):
        place_pair(mp4_files[stem], mp4_target_dir / mp4_files[stem].name, mode)
        place_pair(json_files[stem], json_target_dir / json_files[stem].name, mode)

    for stem in train_stems:
        copy_pair(stem, train_mp4_dir, train_json_dir)

    for stem in test_stems:
        copy_pair(stem, test_mp4_dir, test_json_dir)

    print(f"Total pairs: {len(common_stems)}")
    print(f"Train pairs: {len(train_stems)}")
    print(f"Test pairs: {len(test_stems)}")
    print(f"Output: {out_dir}")
    if mode == "move":
        print("Mode: move")
    elif mode == "copy":
        print("Mode: copy")
    else:
        print("Mode: symlink")


if __name__ == "__main__":
    main()
