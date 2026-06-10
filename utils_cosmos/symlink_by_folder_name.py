#!/usr/bin/env python3
"""
Create symlinks from one or more source folders into a flat destination folder.

Rules:
- For each immediate subfolder of SRC:
  - If the folder name contains 'json' (case-insensitive), symlink all .json files
    inside that folder and its subfolders.
  - Otherwise, symlink all .mp4 and .mov files inside that folder and its subfolders.
- The destination is flat: all selected files are symlinked directly in DST.

Usage:
  python symlink_by_folder_name.py SRC_DIR [SRC_DIR ...] DST_DIR
  python symlink_by_folder_name.py --all_train
  python symlink_by_folder_name.py --all_test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def iter_files(root: Path, ext: str):
    ext = ext.lower()
    for path in root.rglob(f"*{ext}"):
        if path.is_file():
            yield path


def ensure_symlink(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        # If already a symlink to the right target, keep it.
        if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src):
            return True
        # Otherwise, skip to avoid overwriting.
        return False
    dst.symlink_to(src)
    return True


class SymlinkLimiter:
    def __init__(self, max_files: int | None):
        self.max_files = max_files
        self.json_count = 0
        self.video_count = 0

    def reached_json(self) -> bool:
        return self.max_files is not None and self.json_count >= self.max_files

    def reached_video(self) -> bool:
        return self.max_files is not None and self.video_count >= self.max_files

    def note_json(self, success: bool) -> None:
        if success:
            self.json_count += 1

    def note_video(self, success: bool) -> None:
        if success:
            self.video_count += 1


def make_flat_dst_path(dst_root: Path, src: Path) -> Path:
    """
    Return a destination path directly under dst_root.
    If a filename collision occurs, append __N before the extension.
    """
    candidate = dst_root / src.name
    if not (candidate.exists() or candidate.is_symlink()):
        return candidate
    if candidate.is_symlink() and os.path.realpath(candidate) == os.path.realpath(src):
        return candidate

    stem = src.stem
    suffix = src.suffix
    idx = 1
    while True:
        candidate = dst_root / f"{stem}__{idx}{suffix}"
        if not (candidate.exists() or candidate.is_symlink()):
            return candidate
        if candidate.is_symlink() and os.path.realpath(candidate) == os.path.realpath(src):
            return candidate
        idx += 1


def symlink_split_dataset(
    ds_root: Path,
    keyword: str,
    dst_json_root: Path,
    dst_mp4_root: Path,
    limiter: SymlinkLimiter | None = None,
    skip_substr: str | None = None,
    exact_dir_name: bool = False,
    keyword_case_sensitive: bool = False,
    json_parent_substr: str | None = None,
) -> int:
    if not ds_root.is_dir():
        print(f"DS root is not a directory: {ds_root}", file=sys.stderr)
        return 2

    dst_json_root.mkdir(parents=True, exist_ok=True)
    dst_mp4_root.mkdir(parents=True, exist_ok=True)

    if exact_dir_name:
        matching_dirs = [
            path for path in ds_root.iterdir()
            if path.is_dir() and path.name == keyword
        ]
    else:
        if keyword_case_sensitive:
            matching_dirs = [
                path for path in ds_root.iterdir()
                if path.is_dir() and keyword in path.name
            ]
        else:
            matching_dirs = [
                path for path in ds_root.iterdir()
                if path.is_dir() and keyword in path.name.lower()
            ]
    # If both variants exist, keep only the "_corretto" folder.
    by_base: dict[str, list[Path]] = {}
    for path in matching_dirs:
        name_lower = path.name.lower()
        base = name_lower.replace("_corretto", "")
        by_base.setdefault(base, []).append(path)
    filtered_dirs: list[Path] = []
    for group in by_base.values():
        if len(group) == 1:
            filtered_dirs.append(group[0])
            continue
        corretto = [p for p in group if "_corretto" in p.name.lower()]
        if corretto:
            filtered_dirs.extend(corretto)
        else:
            filtered_dirs.extend(group)

    for matching_dir in sorted(filtered_dirs):
        if skip_substr and skip_substr in matching_dir.name.lower():
            continue
        for subentry in sorted(matching_dir.rglob("*")):
            if not subentry.is_file():
                continue
            suffix = subentry.suffix.lower()
            if suffix == ".json":
                if json_parent_substr and not any(
                    json_parent_substr in str(parent).lower()
                    for parent in subentry.parents
                ):
                    continue
                if limiter and limiter.reached_json():
                    return 0
                dst_file = make_flat_dst_path(dst_json_root, subentry)
                success = ensure_symlink(subentry, dst_file)
                if limiter:
                    limiter.note_json(success)
            elif suffix in {".mp4", ".mov"}:
                if limiter and limiter.reached_video():
                    return 0
                dst_file = make_flat_dst_path(dst_mp4_root, subentry)
                success = ensure_symlink(subentry, dst_file)
                if limiter:
                    limiter.note_video(success)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create symlinks from one or more source folders to a destination "
            "folder using folder-name-based extension rules."
        )
    )
    parser.add_argument(
        "--all_train",
        action="store_true",
        help=(
            "If set, ignore --src_dirs/--dst_dir and instead symlink all folders "
            "in --ds_root containing 'training' to --dst_json_dir and --dst_mp4_dir."
        ),
    )
    parser.add_argument(
        "--all_test",
        action="store_true",
        help=(
            "If set, ignore --src_dirs/--dst_dir and instead symlink all folders "
            "in --ds_root containing 'test' to --test_dst_json_dir and --test_dst_mp4_dir."
        ),
    )
    parser.add_argument(
        "--ds_root",
        default="ds_pulito",
        help="Root directory to scan when using --all_train.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help=(
            "Maximum number of symlinks to create (or reuse) before stopping. "
            "If unset, no limit is applied."
        ),
    )
    parser.add_argument(
        "--dst_json_dir",
        default="ds_pulito/train_dataset_json",
        help="Destination directory for .json files when using --all_train.",
    )
    parser.add_argument(
        "--dst_mp4_dir",
        default="ds_pulito/train_dataset",
        help="Destination directory for .mp4/.mov files when using --all_train.",
    )
    parser.add_argument(
        "--test_dst_json_dir",
        default="ds_pulito/test_dataset_json",
        help="Destination directory for .json files when using --all_test.",
    )
    parser.add_argument(
        "--test_dst_mp4_dir",
        default="ds_pulito/test_dataset",
        help="Destination directory for .mp4/.mov files when using --all_test.",
    )
    parser.add_argument(
        "--src_dirs",
        nargs="+",
        default=[
        "ds/Dataset_training_v1_vlm_batch_1",
        # "ds/Dataset_training_v1_vlm_batch_1_json_final_prompt",
        "ds/Dataset_training_v2_vlm_batch_2",
        # "ds/Dataset_training_v2_vlm_batch_2_json_final_prompt",
        "ds/Dataset_training_v3_vlm_batch_5",
        # "ds/Dataset_training_v3_vlm_batch_5_json_final_prompt",
        "ds/Dataset_training_v4_vlm_batch_3",
        # "ds/Dataset_training_v4_vlm_batch_3_json_final_prompt",
        "ds/Dataset_training_v5_vlm_batch_6",
        # "ds/Dataset_training_v5_vlm_batch_6_json_final_prompt",
        "ds/Dataset_training_v6_vlm_batch_7",
        # "ds/Dataset_training_v6_vlm_batch_7_json_final_prompt",
        "ds/Dataset_training_v7_vlm_batch_8",
        # "ds/Dataset_training_v7_vlm_batch_8_json_final_prompt",
        "ds/Dataset_training_v8_vlm_batch_8",
        # "ds/Dataset_training_v8_vlm_batch_8_json_final_prompt",
        "ds/Dataset_training_v9_vlm_batch_9",
        # "ds/Dataset_training_v9_vlm_batch_9_json_final_prompt",
        "ds/Dataset_training_v10_vlm_batch_4",
        # "ds/Dataset_training_v10_vlm_batch_4_json_final_prompt",
        "ds/Dataset_training_v11_vlm_batch_1",
        # "ds/Dataset_training_v11_vlm_batch_1_json_final_prompt",
        "ds/Dataset_training_v12_vlm_batch_2",
        # "ds/Dataset_training_v12_vlm_batch_2_json_final_prompt",
        "ds/Dataset_training_v13_vlm_batch_3",
        # "ds/Dataset_training_v13_vlm_batch_3_json_final_prompt",
        "ds/Dataset_training_v14_vlm_batch_10",
        # "ds/Dataset_training_v14_vlm_batch_10_json_final_prompt",
        "ds/Dataset_training_v15_vlm_batch_4",
        # "ds/Dataset_training_v15_vlm_batch_4_json_final_prompt",
        "ds/Dataset_training_v16_vlm_batch_11",
        # "ds/Dataset_training_v16_vlm_batch_11_json_final_prompt",
        "ds/Dataset_training_v17_vlm_batch_5",
        # "ds/Dataset_training_v17_vlm_batch_5_json_final_prompt",
        ],
        help="One or more source directories.",
        )
    parser.add_argument(
        "--dst_dir",
        default="train_dataset_17k",
        help="Destination directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    limiter = SymlinkLimiter(args.max_files)

    if args.all_train:
        ds_root = Path(args.ds_root).expanduser().resolve()
        dst_json_root = Path(args.dst_json_dir).expanduser().resolve()
        dst_mp4_root = Path(args.dst_mp4_dir).expanduser().resolve()
        return symlink_split_dataset(
            ds_root,
            "training",
            dst_json_root,
            dst_mp4_root,
            limiter=limiter,
            skip_substr="batch_14",
            json_parent_substr="json_final_prompt",
        )
    elif args.all_test:
        ds_root = Path(args.ds_root).expanduser().resolve()
        dst_json_root = Path(args.test_dst_json_dir).expanduser().resolve()
        dst_mp4_root = Path(args.test_dst_mp4_dir).expanduser().resolve()
        return symlink_split_dataset(
            ds_root,
            "test",
            dst_json_root,
            dst_mp4_root,
            limiter=limiter,
            skip_substr="test_dataset",
            keyword_case_sensitive=True,
            json_parent_substr="json_final_prompt",
        )
    else:
        src_roots = [Path(p).expanduser().resolve() for p in args.src_dirs]
        dst_root = Path(args.dst_dir).expanduser().resolve()

        dst_root.mkdir(parents=True, exist_ok=True)

        for src_root in src_roots:
            if limiter.reached_json() and limiter.reached_video():
                break
            if not src_root.is_dir():
                print(f"Source is not a directory: {src_root}", file=sys.stderr)
                return 2

            for entry in sorted(src_root.iterdir()):
                if limiter.reached_json() and limiter.reached_video():
                    break
                if not entry.is_dir():
                    if limiter.reached_json() and limiter.reached_video():
                        break
                    dst_file = make_flat_dst_path(dst_root, entry)
                    success = ensure_symlink(entry, dst_file)
                    if limiter:
                        if entry.suffix.lower() == ".json":
                            limiter.note_json(success)
                        elif entry.suffix.lower() in {".mp4", ".mov"}:
                            limiter.note_video(success)
                else:
                    for subentry in sorted(entry.rglob("*")):
                        if limiter.reached_json() and limiter.reached_video():
                            break
                        if subentry.is_file():
                            name_lower = subentry.name.lower()
                            ext = ".json" if "json" in name_lower else ".mp4"
                            if subentry.suffix.lower() == ext or (
                                ext == ".mp4" and subentry.suffix.lower() == ".mov"
                            ):
                                if (
                                    ext == ".json"
                                    and limiter
                                    and limiter.reached_json()
                                ):
                                    continue
                                if (
                                    ext == ".mp4"
                                    and limiter
                                    and limiter.reached_video()
                                ):
                                    continue
                                dst_file = make_flat_dst_path(dst_root, subentry)
                                success = ensure_symlink(subentry, dst_file)
                                if limiter:
                                    if ext == ".json":
                                        limiter.note_json(success)
                                    else:
                                        limiter.note_video(success)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
