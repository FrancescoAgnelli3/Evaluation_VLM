"""Shared filesystem layout helpers for evaluation and inference outputs."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def task_root(task: str) -> Path:
    return ARTIFACTS_DIR / task


def task_results_dir(task: str) -> Path:
    return task_root(task) / "results"


def task_eval_dir(task: str) -> Path:
    return task_root(task) / "eval"

