#!/usr/bin/env python3
"""Utilities for deterministic model ordering in summaries."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import pandas as pd


def _extract_last_number(pattern: str, text: str) -> Optional[float]:
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return float(matches[-1].group(1))


def _is_ft(name: str) -> bool:
    if "loraft" in name or "fullft" in name or "sft" in name:
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
    return any(t == "ft" or t.endswith("ft") for t in tokens)


def format_model_name(model: object) -> str:
    name = str(model or "")
    norm = name.lower()

    # Cosmos-Reason2 / Cosmos3 normalization
    norm = norm.replace("cosmos_reason2", "cosmos-reason2")
    norm = norm.replace("cosmos_reason3", "cosmos3")
    norm = norm.replace("cosmos2_reason", "cosmos-reason2")
    norm = norm.replace("cosmos2", "cosmos-reason2")
    if "cosmos-reason2-32b" in norm:
        norm = norm.replace("cosmos-reason2-32b", "Cosmos-Reason2-32B")
    if "cosmos3" in norm:
        norm = norm.replace("cosmos3", "Cosmos3")
    if "cosmos2-reason-loraft_17k-32b" in norm:
        norm = norm.replace("cosmos2-reason-loraft_17k-32b", "Cosmos-Reason2-LoRAFT-17k_32B")

    # Qwen normalization
    norm = norm.replace("qwen3.5", "Qwen3.5")
    norm = norm.replace("qwen", "Qwen3")

    # Preserve original casing for any remaining parts by applying replacements on the original
    name = name.replace("cosmos_reason2_32b", "Cosmos-Reason2-32B")
    name = name.replace("cosmos_reason3", "Cosmos3")
    name = name.replace("cosmos2_reason_32b", "Cosmos-Reason2-32B")
    name = name.replace("cosmos_reason2", "Cosmos-Reason2")
    name = name.replace("cosmos-reason2", "Cosmos-Reason2")
    name = name.replace("cosmos2_reason", "Cosmos-Reason2")
    name = name.replace("cosmos2", "Cosmos-Reason2")
    name = name.replace("cosmos-reason2-32B", "Cosmos-Reason2-32B")
    name = name.replace("cosmos3", "Cosmos3")
    name = name.replace("cosmos2-reason-LoRAFT_17k-32B", "Cosmos-Reason2-LoRAFT-17k_32B")
    name = name.replace("qwen3.5", "Qwen3.5")
    name = name.replace("qwen", "Qwen3")

    # If original did not contain the normalized tokens, fall back to norm replacements
    if name == str(model or ""):
        name = norm

    # Escape underscores for CSV rendering
    return name.replace("_", "\\_")


def _model_sort_key(model: object) -> Tuple[int, float, float, str]:
    name = str(model or "")
    norm = name.lower().replace("-", "_").replace("\\_", "_")

    is_cosmos = "cosmos" in norm
    is_qwen = "qwen" in norm
    is_ft = _is_ft(norm)

    if is_cosmos and is_ft:
        category = 0
    elif is_qwen and is_ft:
        category = 1
    elif is_cosmos:
        category = 2
    elif is_qwen:
        category = 3
    else:
        category = 4

    b_val = _extract_last_number(r"(\d+(?:\.\d+)?)\s*b", norm)
    b_key = b_val if b_val is not None else float("inf")

    if is_ft:
        k_val = _extract_last_number(r"(\d+(?:\.\d+)?)\s*k", norm)
        k_key = k_val if k_val is not None else float("inf")
    else:
        k_key = 0.0

    return (category, b_key, k_key, norm)


def sort_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df

    if "global_slot_macro_acc_mean" in df.columns:
        out = df.copy()
        out = out.sort_values(
            by=["global_slot_macro_acc_mean", "model"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        )
        return out

    sort_keys = df["model"].apply(_model_sort_key)
    out = df.copy()
    out["_sort_cat"] = [k[0] for k in sort_keys]
    out["_sort_b"] = [k[1] for k in sort_keys]
    out["_sort_k"] = [k[2] for k in sort_keys]
    out["_sort_name"] = [k[3] for k in sort_keys]

    out = out.sort_values(
        by=["_sort_cat", "_sort_b", "_sort_k", "_sort_name"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return out.drop(columns=["_sort_cat", "_sort_b", "_sort_k", "_sort_name"])
