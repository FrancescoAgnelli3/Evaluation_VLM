#!/usr/bin/env python3
"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def get_path(obj: Any, path: Sequence[Any]) -> Any:
    cur = obj
    for p in path:
        if p == "*":
            raise ValueError("Wildcard cannot be resolved with get_path()")
        if isinstance(cur, list):
            if not isinstance(p, int) or p < 0 or p >= len(cur):
                return None
            cur = cur[p]
            continue
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def is_confidence(x: Any) -> bool:
    v = safe_float(x)
    return v is not None and 0.0 <= v <= 1.0


def is_int_or_unknown(x: Any) -> bool:
    if x == "unknown":
        return True
    return isinstance(x, int) and not isinstance(x, bool)


def is_bool_or_unknown(x: Any) -> bool:
    if x == "unknown":
        return True
    return isinstance(x, bool)


def is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(v, str) for v in x)


def is_int_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in x)


def _set_f1(pred: Iterable[Any], ref: Iterable[Any]) -> float:
    ps = set(str(x) for x in (pred or []))
    rs = set(str(x) for x in (ref or []))
    if not ps and not rs:
        return 1.0
    tp = len(ps & rs)
    fp = len(ps - rs)
    fn = len(rs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


# ----------------------------
# Teacher / student discovery
# ----------------------------

def discover_teacher_standards(results_gold: Path) -> List[Tuple[str, Path]]:
    suffixes = [
        ".json",
    ]
    out: List[Tuple[str, Path]] = []
    for p in sorted(results_gold.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        matched_suffix = next((s for s in suffixes if name.endswith(s)), None)
        if matched_suffix is None:
            continue
        video_id = name[: -len(matched_suffix)]
        out.append((video_id, p))
    return out


def discover_teacher_runs(results_gold: Path, video_id: str) -> List[Path]:
    pats = [
        f"{video_id}.teacher.run_*.json",
        f"{video_id}.teacher.perception_raw.run_*.json",
    ]
    out: List[Path] = []
    for pat in pats:
        out.extend(results_gold.glob(pat))
    return sorted(set(out))


def discover_student_files(results: Path, video_id: str) -> List[Tuple[str, Path]]:
    suffixes = ["_json_answer.json", "_integrated.json", ".json"]
    files: List[Tuple[str, Path]] = []
    vid_re = re.compile(rf"(?:^|_){re.escape(video_id)}_(.+)$")

    for p in results.iterdir():
        if not p.is_file():
            continue
        name = p.name
        matched_suffix = next((s for s in suffixes if name.endswith(s)), None)
        if matched_suffix is None:
            continue
        stem = name[: -len(matched_suffix)] if matched_suffix != ".json" else name[: -len(".json")]
        m = vid_re.search(stem)
        if m:
            model = m.group(1) or "unknown"
            files.append((model, p))

    return sorted(files, key=lambda x: x[0])


# ----------------------------
# Consensus weighting
# ----------------------------

def compute_teacher_run_consensus_weight(
    results_gold: Path,
    video_id: str,
    validate_fn: Callable[[Dict[str, Any]], Any],
    pair_score_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
) -> float:
    run_paths = discover_teacher_runs(results_gold, video_id)
    runs: List[Dict[str, Any]] = []
    for p in run_paths:
        obj = read_json(p)
        if obj is None:
            continue
        vrep = validate_fn(obj)
        if not getattr(vrep, "parse_ok", False):
            continue
        runs.append(obj)

    if len(runs) < 2:
        return 1.0

    pair_scores: List[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pair_scores.append(pair_score_fn(runs[i], runs[j]))

    if not pair_scores:
        return 1.0

    w = float(np.mean(pair_scores))
    return min(1.0, max(0.0, w))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if not m.any():
        return float("nan")
    vv = v[m].astype(float)
    ww = w[m].astype(float)
    denom = float(ww.sum())
    if denom <= 0.0:
        return float("nan")
    return float((vv * ww).sum() / denom)
