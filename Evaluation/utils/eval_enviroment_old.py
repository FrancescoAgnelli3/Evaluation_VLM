#!/usr/bin/env python3
"""
Evaluator for the INFRASTRUCTURE + CLEANLINESS MAINTENANCE REPORT JSON schema.

What this evaluator does
------------------------
- Computes slot-level macro accuracy per section + a global macro accuracy.
- Uses teacher-gated scoring:
  * always scores ".visible"
  * scores ".value" only if teacher says visible==True
  * scores evidence_frames only for "positive observations" according to teacher
    (visible==True and value is not "unknown" and not a negative/absence value like "none"/False)
- For evidence_frames, uses set-F1 (order-independent).
- Keeps best-effort validation metrics: parse_ok / schema_ok / rule_ok.

Dependencies:
  pip install numpy pandas
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.eval_common import (
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_bool_or_unknown,
    is_confidence,
    is_int_list,
    is_str_list,
    read_json,
    safe_float,
)
from utils.latex_table import write_latex_table
from utils.run_paths import task_eval_dir, task_results_dir

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "environment"
GT_DIR = f"/opt/dataset/ds_{TASK_NAME}/test_dataset_json"
DEFAULT_RESULTS_DIR = task_results_dir(TASK_NAME)
DEFAULT_OUT_DIR = task_eval_dir(TASK_NAME)


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], set] = {
    # cleanliness_and_waste
    ("cleanliness_and_waste", "loose_litter", "value"): {"none", "low", "moderate", "high", "unknown"},
    ("cleanliness_and_waste", "bulky_items_present", "value"): {True, False, "unknown"},
    ("cleanliness_and_waste", "waste_location", "value"): {"sidewalk", "green_area", "roadside_layby", "curbside", "unknown"},
    ("cleanliness_and_waste", "potentially_hazardous_waste", "value"): {True, False, "unknown"},
    # infrastructure_condition
    ("infrastructure_condition", "street_lighting_condition", "value"): {"intact", "possibly_damaged", "unknown"},
    ("infrastructure_condition", "signage_and_markings", "value"): {"intact", "faded", "damaged", "unknown"},
    ("infrastructure_condition", "walkway_and_road_surface", "value"): {"intact", "uneven", "potholes", "degraded", "unknown"},
    ("infrastructure_condition", "public_furniture_condition", "value"): {"intact", "deteriorated", "unknown"},
    ("infrastructure_condition", "spray_paint_markings", "value"): {"none", "markings_present", "unknown"},
    ("infrastructure_condition", "spray_paint_markings", "extent"): {"small", "medium", "large", "unknown"},
    # environmental_indicators
    ("environmental_indicators", "standing_water_or_leaks", "value"): {True, False, "unknown"},
    ("environmental_indicators", "visible_smoke_or_haze", "value"): {True, False, "unknown"},
    ("environmental_indicators", "vegetation_overgrowth", "value"): {"none", "low", "moderate", "high", "unknown"},
    # observed_activities
    ("observed_activities", "spray_paint_marking_in_progress", "value"): {True, False, "unknown"},
    ("observed_activities", "spray_paint_marking_in_progress", "surface_target"): {
        "wall",
        "public_building",
        "street_furniture",
        "sidewalk",
        "underpass",
        "unknown",
    },
    ("observed_activities", "waste_disposal_activity_in_progress", "value"): {True, False, "unknown"},
}

# Fields we consider as "absence/negative" for the evidence_frames gating.
NEGATIVE_STRING_VALUES = {"none"}


# ----------------------------
# Slot paths (per-section)
# ----------------------------

def _std_indicator_paths(section: str, field: str) -> List[Tuple[Any, ...]]:
    # Standard indicator: visible, value, confidence, evidence_frames
    return [
        (section, field, "visible"),
        (section, field, "value"),
        (section, field, "confidence"),
        (section, field, "evidence_frames"),
    ]


CLEANLINESS_SLOTS: List[Tuple[Any, ...]] = (
    _std_indicator_paths("cleanliness_and_waste", "loose_litter")
    + _std_indicator_paths("cleanliness_and_waste", "bulky_items_present")
    + _std_indicator_paths("cleanliness_and_waste", "waste_location")
    + _std_indicator_paths("cleanliness_and_waste", "potentially_hazardous_waste")
)

INFRA_SLOTS: List[Tuple[Any, ...]] = (
    _std_indicator_paths("infrastructure_condition", "street_lighting_condition")
    + _std_indicator_paths("infrastructure_condition", "signage_and_markings")
    + _std_indicator_paths("infrastructure_condition", "walkway_and_road_surface")
    + _std_indicator_paths("infrastructure_condition", "public_furniture_condition")
    + _std_indicator_paths("infrastructure_condition", "spray_paint_markings")
    + [
        ("infrastructure_condition", "spray_paint_markings", "extent"),
    ]
)

ENV_SLOTS: List[Tuple[Any, ...]] = (
    _std_indicator_paths("environmental_indicators", "standing_water_or_leaks")
    + _std_indicator_paths("environmental_indicators", "visible_smoke_or_haze")
    + _std_indicator_paths("environmental_indicators", "vegetation_overgrowth")
)

ACTIVITY_SLOTS: List[Tuple[Any, ...]] = (
    _std_indicator_paths("observed_activities", "spray_paint_marking_in_progress")
    + [
        ("observed_activities", "spray_paint_marking_in_progress", "surface_target"),
    ]
    + _std_indicator_paths("observed_activities", "waste_disposal_activity_in_progress")
)

# Limitations are free-form: not scored by default (too easy to “game”, too hard to normalize).
# If you want presence scoring, add:
# LIMITATIONS_SLOTS = [("limitations", "visibility_constraints"), ("limitations", "notes")]


def _set_f1(pred: Iterable[Any], ref: Iterable[Any]) -> float:
    ps = set(int(x) for x in (pred or []) if isinstance(x, int) and not isinstance(x, bool))
    rs = set(int(x) for x in (ref or []) if isinstance(x, int) and not isinstance(x, bool))
    if not ps and not rs:
        return 1.0
    tp = len(ps & rs)
    fp = len(ps - rs)
    fn = len(rs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def _is_positive_value(v: Any) -> bool:
    # Gate evidence_frames on teacher-positive observations.
    # - unknown => not positive
    # - boolean => positive only if True
    # - string => positive if not "unknown" and not a negative string (e.g., "none")
    if v is None:
        return False
    if v == "unknown":
        return False
    if isinstance(v, bool):
        return v is True
    if isinstance(v, str):
        return v not in NEGATIVE_STRING_VALUES
    return False


# ----------------------------
# Teacher gating for slot scoring
# ----------------------------

def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    Gating rules:
      - score all .visible
      - score .value / .confidence only if teacher visible==True
      - score .extent / .surface_target only if teacher visible==True
      - score evidence_frames only if teacher visible==True and teacher value is "positive"
    """
    if not path:
        return True

    last = path[-1]
    if last == "visible":
        return True

    # Only gate on standard indicator structure: (section, field, key)
    if len(path) < 3:
        return True

    section, field = path[0], path[1]
    t_vis = get_path(teacher, (section, field, "visible"))

    if last in {"value", "confidence", "extent", "surface_target"}:
        return t_vis is True

    if last == "evidence_frames":
        if t_vis is not True:
            return False
        t_val = get_path(teacher, (section, field, "value"))
        return _is_positive_value(t_val)

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

        key = p[-1]

        # evidence_frames gets set-F1, others are exact match
        if key == "evidence_frames":
            sv = get_path(student, p) or []
            tv = get_path(teacher, p) or []
            ok_score = _set_f1(sv, tv)
            hits.append(float(ok_score))
            # For "correct/total" accounting, treat perfect F1 as correct.
            correct += 1 if ok_score >= 0.999999 else 0
            total += 1
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


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    cleanliness_macro_acc: float
    infrastructure_macro_acc: float
    environmental_macro_acc: float
    activities_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    cl = _score_leaf_slots(student, teacher, CLEANLINESS_SLOTS)
    inf = _score_leaf_slots(student, teacher, INFRA_SLOTS)
    env = _score_leaf_slots(student, teacher, ENV_SLOTS)
    act = _score_leaf_slots(student, teacher, ACTIVITY_SLOTS)

    all_hits = cl.hits + inf.hits + env.hits + act.hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        cleanliness_macro_acc=cl.macro_acc,
        infrastructure_macro_acc=inf.macro_acc,
        environmental_macro_acc=env.macro_acc,
        activities_macro_acc=act.macro_acc,
    )
    return sec, {
        "cleanliness_and_waste": cl.hits,
        "infrastructure_condition": inf.hits,
        "environmental_indicators": env.hits,
        "observed_activities": act.hits,
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


def _check_enum(value: Any, allowed: set, tag: str, errors: List[str]) -> None:
    if value is None:
        errors.append(f"missing_enum:{tag}")
    elif value not in allowed:
        errors.append(f"bad_enum:{tag}={value}")


def _check_indicator_obj(obj: Any, tag: str, errors: List[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"not_object:{tag}")
        return
    for k in ["visible", "value", "confidence", "evidence_frames"]:
        if k not in obj:
            errors.append(f"missing_key:{tag}.{k}")


def validate_struct(obj: Optional[Dict[str, Any]]) -> ValidationReport:
    if obj is None:
        return ValidationReport(parse_ok=False, schema_ok=False, rule_ok=False, errors=["parse_failed_or_not_object"])

    errors: List[str] = []

    required_top = [
        "cleanliness_and_waste",
        "infrastructure_condition",
        "environmental_indicators",
        "observed_activities",
        "limitations",
    ]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    # Type checks for section objects
    for sec in ["cleanliness_and_waste", "infrastructure_condition", "environmental_indicators", "observed_activities", "limitations"]:
        if sec in obj and not isinstance(obj.get(sec), dict):
            errors.append(f"not_object:{sec}")

    # Indicator object presence checks (best-effort; does not enforce extra keys strictly)
    # cleanliness_and_waste
    cw = obj.get("cleanliness_and_waste", {})
    _check_indicator_obj(get_path(obj, ("cleanliness_and_waste", "loose_litter")), "cleanliness_and_waste.loose_litter", errors)
    _check_indicator_obj(get_path(obj, ("cleanliness_and_waste", "bulky_items_present")), "cleanliness_and_waste.bulky_items_present", errors)
    _check_indicator_obj(get_path(obj, ("cleanliness_and_waste", "waste_location")), "cleanliness_and_waste.waste_location", errors)
    _check_indicator_obj(get_path(obj, ("cleanliness_and_waste", "potentially_hazardous_waste")), "cleanliness_and_waste.potentially_hazardous_waste", errors)

    # infrastructure_condition
    _check_indicator_obj(get_path(obj, ("infrastructure_condition", "street_lighting_condition")), "infrastructure_condition.street_lighting_condition", errors)
    _check_indicator_obj(get_path(obj, ("infrastructure_condition", "signage_and_markings")), "infrastructure_condition.signage_and_markings", errors)
    _check_indicator_obj(get_path(obj, ("infrastructure_condition", "walkway_and_road_surface")), "infrastructure_condition.walkway_and_road_surface", errors)
    _check_indicator_obj(get_path(obj, ("infrastructure_condition", "public_furniture_condition")), "infrastructure_condition.public_furniture_condition", errors)
    _check_indicator_obj(get_path(obj, ("infrastructure_condition", "spray_paint_markings")), "infrastructure_condition.spray_paint_markings", errors)
    spm = get_path(obj, ("infrastructure_condition", "spray_paint_markings"))
    if isinstance(spm, dict) and "extent" not in spm:
        errors.append("missing_key:infrastructure_condition.spray_paint_markings.extent")

    # environmental_indicators
    _check_indicator_obj(get_path(obj, ("environmental_indicators", "standing_water_or_leaks")), "environmental_indicators.standing_water_or_leaks", errors)
    _check_indicator_obj(get_path(obj, ("environmental_indicators", "visible_smoke_or_haze")), "environmental_indicators.visible_smoke_or_haze", errors)
    _check_indicator_obj(get_path(obj, ("environmental_indicators", "vegetation_overgrowth")), "environmental_indicators.vegetation_overgrowth", errors)

    # observed_activities
    _check_indicator_obj(get_path(obj, ("observed_activities", "spray_paint_marking_in_progress")), "observed_activities.spray_paint_marking_in_progress", errors)
    spip = get_path(obj, ("observed_activities", "spray_paint_marking_in_progress"))
    if isinstance(spip, dict) and "surface_target" not in spip:
        errors.append("missing_key:observed_activities.spray_paint_marking_in_progress.surface_target")
    _check_indicator_obj(get_path(obj, ("observed_activities", "waste_disposal_activity_in_progress")), "observed_activities.waste_disposal_activity_in_progress", errors)

    # limitations (free-form)
    lim = obj.get("limitations", {})
    if isinstance(lim, dict):
        if "visibility_constraints" not in lim:
            errors.append("missing_key:limitations.visibility_constraints")
        if "notes" not in lim:
            errors.append("missing_key:limitations.notes")
        vc = lim.get("visibility_constraints")
        if vc is not None and not is_str_list(vc):
            errors.append("bad_type:limitations.visibility_constraints")
        nt = lim.get("notes")
        if nt is not None and not isinstance(nt, str):
            errors.append("bad_type:limitations.notes")

    # Confidence checks (rule_ok component)
    rule_ok = True
    # Scan all standard confidence fields present in slot lists
    for p in (CLEANLINESS_SLOTS + INFRA_SLOTS + ENV_SLOTS + ACTIVITY_SLOTS):
        if p[-1] != "confidence":
            continue
        v = get_path(obj, p)
        tag = ".".join(map(str, p))
        if v is None:
            errors.append(f"missing_conf:{tag}")
            rule_ok = False
        elif not is_confidence(v):
            errors.append(f"bad_conf:{tag}={v}")
            rule_ok = False

    # evidence_frames type checks
    for p in (CLEANLINESS_SLOTS + INFRA_SLOTS + ENV_SLOTS + ACTIVITY_SLOTS):
        if p[-1] != "evidence_frames":
            continue
        v = get_path(obj, p)
        tag = ".".join(map(str, p))
        if v is None:
            errors.append(f"missing_evidence_frames:{tag}")
            # not necessarily rule-breaking; keep schema signal only
        elif not is_int_list(v):
            errors.append(f"bad_type:{tag} (expected int list)")

    # Enum checks (all defined ENUMS)
    for path, allowed in ENUMS.items():
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.startswith("missing_key:")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        or e.startswith("bad_type:")
        for e in errors
    )

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# CLI / main
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-gold", type=Path, default=GT_DIR)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
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

    def log(msg: str) -> None:
        if verbose:
            print(msg)

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
        if teacher is None:
            log(f"Skipping teacher (parse failed): {teacher_path}")
            continue

        student_files = discover_student_files(args.results, video_id)
        if args.model:
            allowed = {m.replace("-", "_") for m in args.model}
            if "all" not in allowed:
                student_files = [(m, p) for (m, p) in student_files if m.replace("-", "_") in allowed]

        if args.skip_missing_students and not student_files:
            continue

        log(f"Video={video_id}: found {len(student_files)} student outputs")

        for model_name, student_path in student_files:
            student = read_json(student_path)
            vrep = validate_struct(student)

            global_slot = None
            cleanliness_slot = None
            infra_slot = None
            env_slot = None
            act_slot = None

            if vrep.parse_ok and "error" not in student:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                cleanliness_slot = sec.cleanliness_macro_acc
                infra_slot = sec.infrastructure_macro_acc
                env_slot = sec.environmental_macro_acc
                act_slot = sec.activities_macro_acc

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": ("error" in student) if isinstance(student, dict) else False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "cleanliness_and_waste_macro_acc": cleanliness_slot,
                    "infrastructure_condition_macro_acc": infra_slot,
                    "environmental_indicators_macro_acc": env_slot,
                    "observed_activities_macro_acc": act_slot,
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
    df.to_csv(df_out, index=False)

    # Aggregate per model
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
                    "cleanliness_and_waste_macro_acc_mean": mean_or_nan("cleanliness_and_waste_macro_acc"),
                    "infrastructure_condition_macro_acc_mean": mean_or_nan("infrastructure_condition_macro_acc"),
                    "environmental_indicators_macro_acc_mean": mean_or_nan("environmental_indicators_macro_acc"),
                    "observed_activities_macro_acc_mean": mean_or_nan("observed_activities_macro_acc"),
                }
            )

    df_agg = pd.DataFrame(agg_rows)
    if not df_agg.empty:
        df_agg = df_agg.sort_values(
            by=["rule_rate", "global_slot_macro_acc_mean"],
            ascending=False,
        )

    df_agg_out = args.out / "model_summary.csv"
    df_agg.to_csv(df_agg_out, index=False)
    latex_out = args.out / "model_summary_latex.txt"
    write_latex_table(df_agg, latex_out)

    (args.out / "details.json").write_text(json.dumps(per_video_details, indent=2), encoding="utf-8")

    print(f"Wrote: {df_out}")
    print(f"Wrote: {df_agg_out}")
    print(f"Wrote: {args.out / 'details.json'}")
    print(f"Wrote: {latex_out}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
