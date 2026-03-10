#!/usr/bin/env python3
"""
Evaluator for the INDUSTRIAL-SAFETY ASSESSMENT JSON schema.

What this evaluator does
------------------------
- Computes slot-level macro accuracy per section + a global macro accuracy.
- Uses teacher-gated scoring:
  * always scores ".visible"
  * scores ".value" / ".confidence" / ".count" only if teacher says visible==True for the sibling indicator
- Scores list-valued fields with set-F1 (order-independent).
- Scores safety_violations with a bag-of-violations metric (presence_f1 + count_acc).
- Keeps best-effort validation metrics: parse_ok / schema_ok / rule_ok.

Dependencies:
  pip install numpy pandas
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd

from utils.eval_common import (
    _set_f1,
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_confidence,
    is_str_list,
    read_json,
)

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "industry"
GT_DIR = f"/opt/dataset/ds_{TASK_NAME}/test_dataset_json"


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], Set[Any]] = {
    # site_characterization
    ("site_characterization", "site_type", "value"): {
        "construction_site",
        "oil_gas_platform",
        "refinery",
        "industrial_plant",
        "unknown",
    },
    ("site_characterization", "activity_level", "value"): {"inactive", "low", "high", "emergency"},
    # personnel_safety
    ("personnel_safety", "workers_present", "value"): {True, False},
    ("personnel_safety", "ppe_compliance", "hard_hats", "value"): {"compliant", "non_compliant", "unknown"},
    ("personnel_safety", "ppe_compliance", "high_visibility_vests", "value"): {"compliant", "non_compliant", "unknown"},
    ("personnel_safety", "ppe_compliance", "specialized_gear", "value"): {"compliant", "non_compliant", "unknown"},
    # hazardous_conditions
    ("hazardous_conditions", "unsecured_heights", "value"): {True, False},
    ("hazardous_conditions", "heavy_machinery_movement", "value"): {True, False},
    ("hazardous_conditions", "visible_leaks_or_fire", "value"): {True, False},
    # safety_violations (wildcards handled separately)
    ("safety_violations", "*", "violation_type"): {
        "no_ppe",
        "restricted_area_entry",
        "unsafe_lifting",
        "fall_risk",
        "other",
    },
    ("safety_violations", "*", "risk_level"): {"low", "medium", "high", "extreme"},
    # overall_site_risk
    ("overall_site_risk", "risk_rating", "value"): {"safe", "cautionary", "hazardous", "critical"},
}


# ----------------------------
# Slot paths (per-section)
# ----------------------------

def _indicator_paths(*path: str) -> List[Tuple[Any, ...]]:
    return [
        (*path, "visible"),
        (*path, "value"),
        (*path, "confidence"),
    ]


SITE_CHARACTERIZATION_SLOTS: List[Tuple[Any, ...]] = (
    _indicator_paths("site_characterization", "site_type")
    + [
        ("site_characterization", "activity_level", "value"),
        ("site_characterization", "activity_level", "confidence"),
    ]
)

PERSONNEL_SAFETY_SLOTS: List[Tuple[Any, ...]] = [
    ("personnel_safety", "workers_present", "visible"),
    ("personnel_safety", "workers_present", "value"),
    ("personnel_safety", "workers_present", "count"),
    ("personnel_safety", "workers_present", "confidence"),
    ("personnel_safety", "ppe_compliance", "hard_hats", "value"),
    ("personnel_safety", "ppe_compliance", "hard_hats", "confidence"),
    ("personnel_safety", "ppe_compliance", "high_visibility_vests", "value"),
    ("personnel_safety", "ppe_compliance", "high_visibility_vests", "confidence"),
    ("personnel_safety", "ppe_compliance", "specialized_gear", "value"),
    ("personnel_safety", "ppe_compliance", "specialized_gear", "confidence"),
]

HAZARDOUS_CONDITIONS_SLOTS: List[Tuple[Any, ...]] = (
    _indicator_paths("hazardous_conditions", "unsecured_heights")
    + _indicator_paths("hazardous_conditions", "heavy_machinery_movement")
    + _indicator_paths("hazardous_conditions", "visible_leaks_or_fire")
)

OVERALL_SITE_RISK_SLOTS: List[Tuple[Any, ...]] = [
    ("overall_site_risk", "risk_rating", "value"),
    ("overall_site_risk", "risk_rating", "confidence"),
]

LIST_VALUE_PATHS = {
    "primary_risk_factors": ("overall_site_risk", "primary_risk_factors"),
}


# ----------------------------
# Teacher gating for slot scoring
# ----------------------------

def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    Gating rules:
      - score all ...visible
      - for indicator objects with visible/value/confidence:
          score ...value / ...confidence only if teacher sibling visible==True
      - for personnel_safety.workers_present:
          score value/count/confidence only if teacher workers_present.visible==True
      - for ppe_compliance:
          score only if teacher says workers_present.visible==True AND workers_present.value==True
      - fields without "visible" (e.g., activity_level, overall_site_risk.risk_rating) are always scored
    """
    if not path:
        return True

    last = path[-1]
    if last == "visible":
        return True

    # Workers gating
    if len(path) >= 3 and path[0] == "personnel_safety" and path[1] == "workers_present":
        w_vis = get_path(teacher, ("personnel_safety", "workers_present", "visible"))
        if w_vis is False:
            return False
        return True

    # PPE gating (only meaningful if workers are present & visible in teacher)
    if len(path) >= 3 and path[0] == "personnel_safety" and path[1] == "ppe_compliance":
        w_vis = get_path(teacher, ("personnel_safety", "workers_present", "visible"))
        w_val = get_path(teacher, ("personnel_safety", "workers_present", "value"))
        if (w_vis is False) or (w_val is False):
            return False
        return True

    # Generic indicator object gating
    if last in {"value", "confidence", "count"}:
        vis = get_path(teacher, (*path[:-1], "visible"))
        if vis is False:
            return False
        return True

    return True


@dataclass
class SlotScoreDetail:
    macro_acc: float
    correct: int
    total: int
    hits: List[float]


def _score_leaf_slots(student: Dict[str, Any], teacher: Dict[str, Any], paths: List[Tuple[Any, ...]]) -> SlotScoreDetail:
    hits: List[float] = []
    correct = 0
    total = 0

    for p in paths:
        if not should_score_slot(p, teacher):
            continue

        sv = get_path(student, p)
        tv = get_path(teacher, p)
        ok = (sv == tv)
        hits.append(1.0 if ok else 0.0)
        correct += 1 if ok else 0
        total += 1

    return SlotScoreDetail(
        macro_acc=float(np.mean(hits)) if hits else 0.0,
        correct=correct,
        total=total,
        hits=hits,
    )


def _tcast_list_of_str(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    if not all(isinstance(v, str) for v in x):
        return []
    return [str(v) for v in x]


def _score_list_f1(student: Dict[str, Any], teacher: Dict[str, Any], path: Tuple[Any, ...]) -> float:
    s_list = _tcast_list_of_str(get_path(student, path))
    t_list = _tcast_list_of_str(get_path(teacher, path))
    return _set_f1(s_list, t_list)


# ----------------------------
# safety_violations scoring (bag-of-violations)
# ----------------------------

@dataclass
class ViolationScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    hits: List[float]


def _violation_key(v: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(v, dict):
        return None
    vt = v.get("violation_type")
    rl = v.get("risk_level")
    if not (isinstance(vt, str) and isinstance(rl, str)):
        return None
    return (vt, rl)


def _violation_counts(arr: Any) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    if not isinstance(arr, list):
        return out
    for v in arr:
        if not isinstance(v, dict):
            continue
        k = _violation_key(v)
        if k is None:
            continue
        c = v.get("count")
        # if count is missing/bad, treat it as 1 for aggregation (best-effort)
        if isinstance(c, int):
            out[k] = out.get(k, 0) + c
        else:
            out[k] = out.get(k, 0) + 1
    return out


def score_safety_violations(student: Dict[str, Any], teacher: Dict[str, Any]) -> ViolationScores:
    s = student.get("safety_violations")
    t = teacher.get("safety_violations")

    s_counts = _violation_counts(s)
    t_counts = _violation_counts(t)

    s_keys = set(s_counts.keys())
    t_keys = set(t_counts.keys())

    if not t_keys:
        return ViolationScores(
            macro_acc=float("nan"),
            presence_f1=float("nan"),
            count_acc=float("nan"),
            hits=[],
        )

    presence_f1 = _set_f1(s_keys, t_keys)

    count_hits: List[float] = []
    for k in t_keys:
        count_hits.append(1.0 if s_counts.get(k, 0) == t_counts.get(k, 0) else 0.0)
    count_acc = float(np.mean(count_hits)) if count_hits else 1.0

    hits = [presence_f1, count_acc]
    macro = float(np.mean(hits)) if hits else 0.0
    return ViolationScores(macro_acc=macro, presence_f1=presence_f1, count_acc=count_acc, hits=hits)


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    site_characterization_macro_acc: float
    personnel_safety_macro_acc: float
    hazardous_conditions_macro_acc: float
    safety_violations_macro_acc: float
    overall_site_risk_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    site = _score_leaf_slots(student, teacher, SITE_CHARACTERIZATION_SLOTS)
    pers = _score_leaf_slots(student, teacher, PERSONNEL_SAFETY_SLOTS)
    haz = _score_leaf_slots(student, teacher, HAZARDOUS_CONDITIONS_SLOTS)
    risk = _score_leaf_slots(student, teacher, OVERALL_SITE_RISK_SLOTS)

    # list-valued fields
    prf_f1 = _score_list_f1(student, teacher, LIST_VALUE_PATHS["primary_risk_factors"])
    risk_hits = risk.hits + [prf_f1]

    violations = score_safety_violations(student, teacher)
    violations_hits = violations.hits

    all_hits = site.hits + pers.hits + haz.hits + violations_hits + risk_hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        site_characterization_macro_acc=site.macro_acc,
        personnel_safety_macro_acc=pers.macro_acc,
        hazardous_conditions_macro_acc=haz.macro_acc,
        safety_violations_macro_acc=violations.macro_acc,
        overall_site_risk_macro_acc=float(np.mean(risk_hits)) if risk_hits else 0.0,
    )

    return sec, {
        "site_characterization": site.hits,
        "personnel_safety": pers.hits,
        "hazardous_conditions": haz.hits,
        "safety_violations": violations_hits,
        "overall_site_risk": risk_hits,
    }


# ----------------------------
# Validation (best-effort)
# ----------------------------

@dataclass
class ValidationReport:
    parse_ok: bool
    schema_ok: bool
    rule_ok: bool
    errors: List[str]


def _check_enum(value: Any, allowed: Set[Any], tag: str, errors: List[str]) -> None:
    if value is None:
        errors.append(f"missing_enum:{tag}")
    elif value not in allowed:
        errors.append(f"bad_enum:{tag}={value}")


def _check_indicator_obj(obj: Any, tag: str, errors: List[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"not_object:{tag}")
        return
    for k in ["visible", "value", "confidence"]:
        if k not in obj:
            errors.append(f"missing_key:{tag}.{k}")
    vis = obj.get("visible")
    if vis is not None and not isinstance(vis, bool):
        errors.append(f"bad_type:{tag}.visible={vis}")


def _check_value_conf_obj(obj: Any, tag: str, errors: List[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"not_object:{tag}")
        return
    if "value" not in obj:
        errors.append(f"missing_key:{tag}.value")
    if "confidence" not in obj:
        errors.append(f"missing_key:{tag}.confidence")


def _check_workers_present_obj(obj: Any, tag: str, errors: List[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"not_object:{tag}")
        return
    for k in ["visible", "value", "count", "confidence"]:
        if k not in obj:
            errors.append(f"missing_key:{tag}.{k}")
    if "visible" in obj and not isinstance(obj.get("visible"), bool):
        errors.append(f"bad_type:{tag}.visible={obj.get('visible')}")
    if "value" in obj and not isinstance(obj.get("value"), bool):
        errors.append(f"bad_type:{tag}.value={obj.get('value')}")
    if "count" in obj and not isinstance(obj.get("count"), int):
        errors.append(f"bad_type:{tag}.count={obj.get('count')}")


def validate_struct(obj: Any) -> ValidationReport:
    if obj is None or not isinstance(obj, dict):
        return ValidationReport(
            parse_ok=False,
            schema_ok=False,
            rule_ok=False,
            errors=["parse_failed_or_not_object"],
        )

    errors: List[str] = []

    required_top = [
        "site_characterization",
        "personnel_safety",
        "hazardous_conditions",
        "safety_violations",
        "overall_site_risk",
    ]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    for sec in ["site_characterization", "personnel_safety", "hazardous_conditions", "overall_site_risk"]:
        if sec in obj and not isinstance(obj.get(sec), dict):
            errors.append(f"not_object:{sec}")

    # site_characterization
    _check_indicator_obj(
        get_path(obj, ("site_characterization", "site_type")),
        "site_characterization.site_type",
        errors,
    )
    _check_value_conf_obj(
        get_path(obj, ("site_characterization", "activity_level")),
        "site_characterization.activity_level",
        errors,
    )

    # personnel_safety
    _check_workers_present_obj(
        get_path(obj, ("personnel_safety", "workers_present")),
        "personnel_safety.workers_present",
        errors,
    )
    ppe = get_path(obj, ("personnel_safety", "ppe_compliance"))
    if not isinstance(ppe, dict):
        errors.append("not_object:personnel_safety.ppe_compliance")
    else:
        for item in ["hard_hats", "high_visibility_vests", "specialized_gear"]:
            _check_value_conf_obj(
                get_path(obj, ("personnel_safety", "ppe_compliance", item)),
                f"personnel_safety.ppe_compliance.{item}",
                errors,
            )

    # hazardous_conditions
    _check_indicator_obj(
        get_path(obj, ("hazardous_conditions", "unsecured_heights")),
        "hazardous_conditions.unsecured_heights",
        errors,
    )
    _check_indicator_obj(
        get_path(obj, ("hazardous_conditions", "heavy_machinery_movement")),
        "hazardous_conditions.heavy_machinery_movement",
        errors,
    )
    _check_indicator_obj(
        get_path(obj, ("hazardous_conditions", "visible_leaks_or_fire")),
        "hazardous_conditions.visible_leaks_or_fire",
        errors,
    )

    # safety_violations
    sv = obj.get("safety_violations")
    if sv is None:
        errors.append("missing_key:safety_violations")
    elif not isinstance(sv, list):
        errors.append("safety_violations_not_list")

    # overall_site_risk
    _check_value_conf_obj(
        get_path(obj, ("overall_site_risk", "risk_rating")),
        "overall_site_risk.risk_rating",
        errors,
    )
    prf = get_path(obj, ("overall_site_risk", "primary_risk_factors"))
    if prf is None:
        errors.append("missing_key:overall_site_risk.primary_risk_factors")
    elif not is_str_list(prf):
        errors.append("bad_type:overall_site_risk.primary_risk_factors")

    # Confidence checks (rule_ok component)
    rule_ok = True
    confidence_paths = [
        ("site_characterization", "site_type", "confidence"),
        ("site_characterization", "activity_level", "confidence"),
        ("personnel_safety", "workers_present", "confidence"),
        ("personnel_safety", "ppe_compliance", "hard_hats", "confidence"),
        ("personnel_safety", "ppe_compliance", "high_visibility_vests", "confidence"),
        ("personnel_safety", "ppe_compliance", "specialized_gear", "confidence"),
        ("hazardous_conditions", "unsecured_heights", "confidence"),
        ("hazardous_conditions", "heavy_machinery_movement", "confidence"),
        ("hazardous_conditions", "visible_leaks_or_fire", "confidence"),
        ("overall_site_risk", "risk_rating", "confidence"),
    ]
    for p in confidence_paths:
        v = get_path(obj, p)
        tag = ".".join(map(str, p))
        if v is None:
            errors.append(f"missing_conf:{tag}")
            rule_ok = False
        elif not is_confidence(v):
            errors.append(f"bad_conf:{tag}={v}")
            rule_ok = False

    # Enum checks (non-wildcard)
    for path, allowed in ENUMS.items():
        if path[0] == "safety_violations" and path[1] == "*":
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    # safety_violations per-item checks + wildcard enums
    if isinstance(sv, list):
        for i, it in enumerate(sv):
            if not isinstance(it, dict):
                errors.append(f"safety_violations[{i}]_not_object")
                continue

            vt = it.get("violation_type")
            rl = it.get("risk_level")
            desc = it.get("description")
            conf = it.get("confidence")
            cnt = it.get("count")

            if not isinstance(vt, str):
                errors.append(f"safety_violations[{i}].violation_type_missing_or_not_str")
            if not isinstance(rl, str):
                errors.append(f"safety_violations[{i}].risk_level_missing_or_not_str")
            if not isinstance(desc, str):
                errors.append(f"safety_violations[{i}].description_missing_or_not_str")
            if conf is None:
                errors.append(f"missing_conf:safety_violations[{i}].confidence")
                rule_ok = False
            elif not is_confidence(conf):
                errors.append(f"bad_conf:safety_violations[{i}].confidence={conf}")
                rule_ok = False
            if not isinstance(cnt, int):
                errors.append(f"safety_violations[{i}].count_missing_or_not_int")

            _check_enum(vt, ENUMS[("safety_violations", "*", "violation_type")], f"safety_violations[{i}].violation_type", errors)
            _check_enum(rl, ENUMS[("safety_violations", "*", "risk_level")], f"safety_violations[{i}].risk_level", errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.startswith("missing_key:")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        or e.startswith("bad_type:")
        or e.endswith("_not_list")
        or e.endswith("not_list")
        or e.endswith("_not_object")
        or "_missing_or_not_" in e
        for e in errors
    )

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# CLI / main
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-gold", type=Path, default=GT_DIR)
    ap.add_argument("--results", type=Path, default=BASE_DIR / "results_environment")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "eval_out_environment")
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--model", action="append", default=None, help="Evaluate only specified model name(s). Can be repeated.")
    ap.add_argument(
        "--skip-missing-students",
        action="store_true",
        help="Skip gold videos with zero matching student outputs.",
    )
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return ap


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    verbose = bool(args.verbose)
    df_agg_out = args.out / "model_summary.csv"

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    existing_summary = None
    existing_models: Set[str] = set()
    if df_agg_out.exists():
        try:
            existing_summary = pd.read_csv(df_agg_out)
            if "model" in existing_summary.columns:
                existing_models = {m.replace("-", "_") for m in existing_summary["model"].dropna().astype(str)}
            else:
                log(f"Warning: existing summary missing 'model' column: {df_agg_out}")
        except Exception as exc:
            log(f"Warning: failed to read existing summary {df_agg_out}: {exc}")

    teacher_standards = discover_teacher_standards(args.results_gold)
    if args.limit_videos is not None:
        teacher_standards = teacher_standards[: args.limit_videos]
    log(f"Found {len(teacher_standards)} teacher standards in {args.results_gold}")

    rows: List[Dict[str, Any]] = []
    per_video_details: List[Dict[str, Any]] = []

    total_videos = len(teacher_standards)
    for vid_idx, (video_id, teacher_path) in enumerate(teacher_standards, start=1):
        if total_videos:
            log(f"Progress: {vid_idx}/{total_videos} videos ({(vid_idx/total_videos)*100.0:.1f}%)")

        teacher = read_json(teacher_path)
        if teacher is None or not isinstance(teacher, dict):
            log(f"Skipping teacher (parse failed or not object): {teacher_path}")
            continue

        student_files = discover_student_files(args.results, video_id)
        if args.model:
            allowed = {m.replace("-", "_") for m in args.model}
            if "all" not in allowed:
                student_files = [(m, p) for (m, p) in student_files if m.replace("-", "_") in allowed]
        if existing_models:
            student_files = [(m, p) for (m, p) in student_files if m.replace("-", "_") not in existing_models]

        if args.skip_missing_students and not student_files:
            continue

        log(f"Video={video_id}: found {len(student_files)} student outputs")

        for model_name, student_path in student_files:
            student = read_json(student_path)
            vrep = validate_struct(student)

            global_slot = None
            site_slot = None
            personnel_slot = None
            hazards_slot = None
            violations_slot = None
            risk_slot = None
            violations_presence_f1 = None
            violations_count_acc = None

            if vrep.parse_ok and isinstance(student, dict) and "error" not in student:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                site_slot = sec.site_characterization_macro_acc
                personnel_slot = sec.personnel_safety_macro_acc
                hazards_slot = sec.hazardous_conditions_macro_acc
                violations_slot = sec.safety_violations_macro_acc
                risk_slot = sec.overall_site_risk_macro_acc

                vs = score_safety_violations(student, teacher)
                violations_presence_f1 = vs.presence_f1
                violations_count_acc = vs.count_acc

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": ("error" in student) if isinstance(student, dict) else False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "site_characterization_macro_acc": site_slot,
                    "personnel_safety_macro_acc": personnel_slot,
                    "hazardous_conditions_macro_acc": hazards_slot,
                    "safety_violations_macro_acc": violations_slot,
                    "overall_site_risk_macro_acc": risk_slot,
                    "violations_presence_f1": violations_presence_f1,
                    "violations_count_acc": violations_count_acc,
                    "student_path": str(student_path),
                    "teacher_path": str(teacher_path),
                    "errors": "|".join(vrep.errors[:30]),
                }
            )

            per_video_details.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "validation": vrep.__dict__,
                    "student_path": str(student_path),
                    "teacher_path": str(teacher_path),
                }
            )

    df = pd.DataFrame(rows)
    df_out = args.out / "per_video_scores.csv"
    if df_out.exists():
        try:
            df_existing = pd.read_csv(df_out)
            if not df_existing.empty:
                if df.empty:
                    df = df_existing
                else:
                    df = pd.concat([df_existing, df], ignore_index=True)
        except Exception as exc:
            log(f"Warning: failed to read existing per-video scores {df_out}: {exc}")
    df.to_csv(df_out, index=False)

    agg_rows: List[Dict[str, Any]] = []
    if not df.empty:
        for model_name, g in df.groupby("model"):
            g2 = g.copy()

            def mean_or_nan(col: str) -> float:
                x = pd.to_numeric(g2[col], errors="coerce")
                return float(x.mean()) if x.notna().any() else float("nan")

            parse_rate = float((g2["parse_ok"] & ~g2["has_error"]).mean())
            schema_rate = float(g2["schema_ok"].mean())
            rule_rate = float(g2["rule_ok"].mean())

            global_mean = mean_or_nan("global_slot_macro_acc")
            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")

            agg_rows.append(
                {
                    "model": model_name,
                    "n_videos": int(len(g2)),
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "site_characterization_macro_acc_mean": mean_or_nan("site_characterization_macro_acc"),
                    "personnel_safety_macro_acc_mean": mean_or_nan("personnel_safety_macro_acc"),
                    "hazardous_conditions_macro_acc_mean": mean_or_nan("hazardous_conditions_macro_acc"),
                    "safety_violations_macro_acc_mean": mean_or_nan("safety_violations_macro_acc"),
                    "overall_site_risk_macro_acc_mean": mean_or_nan("overall_site_risk_macro_acc"),
                    "violations_presence_f1_mean": mean_or_nan("violations_presence_f1"),
                    "violations_count_acc_mean": mean_or_nan("violations_count_acc"),
                }
            )

    df_agg = pd.DataFrame(agg_rows)
    if not df_agg.empty:
        df_agg = df_agg.sort_values(
            by=["rule_rate", "global_slot_macro_acc_mean", "safety_violations_macro_acc_mean"],
            ascending=False,
        )

    if existing_summary is not None and not existing_summary.empty:
        if df_agg.empty:
            df_agg = existing_summary
        else:
            df_agg = pd.concat([existing_summary, df_agg], ignore_index=True)
    df_agg.to_csv(df_agg_out, index=False)

    details_path = args.out / "details.json"
    if details_path.exists():
        try:
            existing_details = json.loads(details_path.read_text(encoding="utf-8"))
            if isinstance(existing_details, list) and existing_details:
                per_video_details = existing_details + per_video_details
        except Exception as exc:
            log(f"Warning: failed to read existing details {details_path}: {exc}")
    details_path.write_text(json.dumps(per_video_details, indent=2), encoding="utf-8")

    print(f"Wrote: {df_out}")
    print(f"Wrote: {df_agg_out}")
    print(f"Wrote: {details_path}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
