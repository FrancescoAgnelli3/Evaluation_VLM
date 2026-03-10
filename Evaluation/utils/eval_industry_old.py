#!/usr/bin/env python3
"""
Evaluator for the SITE-SAFETY HAZARD ASSESSMENT JSON schema.

This adapts the previous road-safety evaluator to the new prompt/schema:

Top-level keys:
  - context
  - hazard_conditions
  - safety_events
  - risk_summary

Scoring:
  - "global_slot_macro_acc" is a true macro over *all scorable leaves* across:
      context + hazard_conditions + safety_events
    (risk_summary has separate metrics and is kept separate, like Risk Assessment before.)

  - Per-section macro metrics:
      context_slot_macro_acc
      hazard_conditions_slot_macro_acc
      safety_events_macro_acc
      global_slot_macro_acc

  - risk_summary metrics:
      risk_summary_macro_acc (visible/value not used here; this section has no "visible" fields)
      overall_risk_rating_acc
      primary_hazard_factors_f1
      recommended_controls_f1

Validation:
  - best-effort parse_ok / schema_ok / rule_ok
  - light rules:
      * if a field has visible==False, we do not require value to be any particular token
        (because the schema does not constrain it beyond being present).
      * if visible==True, confidence must be in [0,1] (best-effort).
      * hazard_conditions.*.value must be in {present, absent, uncertain} (when present).
      * safety_events.* enums validated.

Teacher-run consensus weighting (optional):
  - consensus weight computed from teacher runs only:
      sym_global_slot_acc (context+hazards+events)
      sym_risk_summary_acc
    and then averaged 50/50

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
    _set_f1,
    compute_teacher_run_consensus_weight,
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_confidence,
    read_json,
    safe_float,
    weighted_mean,
)

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "industry"
GT_DIR = f"/opt/dataset/ds_{TASK_NAME}/test_dataset_json"


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], set] = {
    # context
    ("context", "environment_type", "value"): {"construction", "industrial_plant", "warehouse", "roadway_work", "unknown"},
    ("context", "camera_view", "value"): {"wide_area", "mid_range", "close_up", "unknown"},
    # hazard_conditions values
    ("hazard_conditions", "*", "value"): {"present", "absent", "uncertain"},
    # safety_events
    ("safety_events", "*", "event_type"): {"fall_hazard", "equipment_motion", "fire_or_smoke", "obstruction", "electrical", "storage", "other"},
    ("safety_events", "*", "severity"): {"low", "medium", "high", "extreme"},
    # risk_summary
    ("risk_summary", "overall_risk_rating", "value"): {"low", "moderate", "high", "critical"},
}

HAZARD_KEYS = [
    "unguarded_edges_or_openings",
    "moving_heavy_equipment",
    "vehicle_pedestrian_proximity_risk",
    "poor_housekeeping_or_obstructions",
    "unsafe_material_storage",
    "electrical_hazard_indicators",
    "fire_smoke_or_leak_indicators",
    "weather_or_visibility_impairment",
]


# ----------------------------
# Slot paths (per-section)
# ----------------------------

CONTEXT_SLOTS: List[Tuple[Any, ...]] = [
    ("context", "environment_type", "visible"),
    ("context", "environment_type", "value"),
    ("context", "environment_type", "confidence"),
    ("context", "camera_view", "visible"),
    ("context", "camera_view", "value"),
    ("context", "camera_view", "confidence"),
    ("context", "people_visible", "visible"),
    ("context", "people_visible", "value"),
    ("context", "people_visible", "confidence"),
]

HAZARD_SLOTS: List[Tuple[Any, ...]] = []
for hk in HAZARD_KEYS:
    HAZARD_SLOTS.extend(
        [
            ("hazard_conditions", hk, "visible"),
            ("hazard_conditions", hk, "value"),
            ("hazard_conditions", hk, "confidence"),
            ("hazard_conditions", hk, "intermittent"),
        ]
    )


# ----------------------------
# Teacher gating for slot scoring
# ----------------------------

def _visible_path_for(path: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
    """
    Maps any leaf within a "visible/value/confidence/intermittent" group to its "visible" path.
    """
    if len(path) >= 3 and path[-1] in {"value", "confidence", "intermittent"}:
        return tuple(path[:-1]) + ("visible",)
    return None


def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    Generic gating:
      - always score *.visible leaves
      - score other leaves only if teacher says the corresponding visible==True
    """
    if not path:
        return True
    if path[-1] == "visible":
        return True
    vpath = _visible_path_for(path)
    if vpath is None:
        return True
    tvis = get_path(teacher, vpath)
    return tvis is True


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


# ----------------------------
# safety_events scoring (bag-of-events)
# ----------------------------

@dataclass
class EventScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    intermittent_acc: float
    hits: List[float]


def _event_key(ev: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Use (event_type, severity) as the identity of an event.
    We deliberately ignore 'description' because it is free text and not reliably comparable.
    """
    if not isinstance(ev, dict):
        return None
    et = ev.get("event_type")
    sv = ev.get("severity")
    if not (isinstance(et, str) and isinstance(sv, str)):
        return None
    return (et, sv)


def _event_counts(arr: Any) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    if not isinstance(arr, list):
        return out
    for ev in arr:
        if not isinstance(ev, dict):
            continue
        k = _event_key(ev)
        if k is None:
            continue
        out[k] = out.get(k, 0) + 1
    return out


def _event_intermittent_map(arr: Any) -> Dict[Tuple[str, str], bool]:
    """
    For each key, set intermittent=True if *any* instance has intermittent=True.
    """
    out: Dict[Tuple[str, str], bool] = {}
    if not isinstance(arr, list):
        return out
    for ev in arr:
        if not isinstance(ev, dict):
            continue
        k = _event_key(ev)
        if k is None:
            continue
        inter = ev.get("intermittent")
        out[k] = bool(out.get(k, False) or (inter is True))
    return out


def score_safety_events(student: Dict[str, Any], teacher: Dict[str, Any]) -> EventScores:
    s = student.get("safety_events")
    t = teacher.get("safety_events")

    s_counts = _event_counts(s)
    t_counts = _event_counts(t)

    s_keys = set(s_counts.keys())
    t_keys = set(t_counts.keys())

    if not t_keys:
        return EventScores(
            macro_acc=float("nan"),
            presence_f1=float("nan"),
            count_acc=float("nan"),
            intermittent_acc=float("nan"),
            hits=[],
        )

    presence_f1 = _set_f1(s_keys, t_keys)

    # count accuracy: exact match per teacher key (missing => 0)
    count_hits: List[float] = []
    for k in t_keys:
        count_hits.append(1.0 if s_counts.get(k, 0) == t_counts.get(k, 0) else 0.0)
    count_acc = float(np.mean(count_hits)) if count_hits else 1.0

    # intermittent accuracy: only for teacher-present keys (if missing => False)
    s_inter = _event_intermittent_map(s)
    t_inter = _event_intermittent_map(t)
    inter_hits: List[float] = []
    for k in t_keys:
        inter_hits.append(1.0 if bool(s_inter.get(k, False)) == bool(t_inter.get(k, False)) else 0.0)
    intermittent_acc = float(np.mean(inter_hits)) if inter_hits else 1.0

    hits = [presence_f1, count_acc, intermittent_acc]
    macro = float(np.mean(hits)) if hits else 0.0
    return EventScores(macro_acc=macro, presence_f1=presence_f1, count_acc=count_acc, intermittent_acc=intermittent_acc, hits=hits)


# ----------------------------
# risk_summary scoring (separate)
# ----------------------------

@dataclass
class RiskSummaryScores:
    macro_acc: float
    overall_risk_rating_acc: float
    primary_hazard_factors_f1: float
    recommended_controls_f1: float


def score_risk_summary(student: Dict[str, Any], teacher: Dict[str, Any]) -> RiskSummaryScores:
    s = student.get("risk_summary")
    t = teacher.get("risk_summary")
    if not isinstance(s, dict) or not isinstance(t, dict):
        return RiskSummaryScores(macro_acc=0.0, overall_risk_rating_acc=0.0, primary_hazard_factors_f1=0.0, recommended_controls_f1=0.0)

    s_rr = get_path(s, ("overall_risk_rating", "value"))
    t_rr = get_path(t, ("overall_risk_rating", "value"))
    overall_acc = 1.0 if s_rr == t_rr else 0.0

    s_phf = tcast_list_of_str(s.get("primary_hazard_factors"))
    t_phf = tcast_list_of_str(t.get("primary_hazard_factors"))
    phf_f1 = _set_f1(s_phf, t_phf)

    s_rc = tcast_list_of_str(s.get("recommended_controls"))
    t_rc = tcast_list_of_str(t.get("recommended_controls"))
    rc_f1 = _set_f1(s_rc, t_rc)

    per = [overall_acc, phf_f1, rc_f1]
    macro = float(np.mean(per)) if per else 0.0
    return RiskSummaryScores(macro_acc=macro, overall_risk_rating_acc=overall_acc, primary_hazard_factors_f1=phf_f1, recommended_controls_f1=rc_f1)


def tcast_list_of_str(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x:
        if isinstance(it, str):
            out.append(it)
        else:
            out.append(str(it))
    return out


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    context_macro_acc: float
    hazard_conditions_macro_acc: float
    safety_events_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    """
    Returns:
      - section/global macro scores
      - raw hit lists per section (used to compute global as a true macro over leaves)
    """
    ctx = _score_leaf_slots(student, teacher, CONTEXT_SLOTS)
    haz = _score_leaf_slots(student, teacher, HAZARD_SLOTS)
    sev = score_safety_events(student, teacher)

    all_hits = ctx.hits + haz.hits + sev.hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        context_macro_acc=ctx.macro_acc,
        hazard_conditions_macro_acc=haz.macro_acc,
        safety_events_macro_acc=sev.macro_acc,
    )
    return sec, {"context": ctx.hits, "hazard_conditions": haz.hits, "safety_events": sev.hits}


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


def validate_struct(obj: Optional[Dict[str, Any]]) -> ValidationReport:
    if obj is None:
        return ValidationReport(parse_ok=False, schema_ok=False, rule_ok=False, errors=["parse_failed_or_not_object"])

    errors: List[str] = []

    required_top = ["context", "hazard_conditions", "safety_events", "risk_summary"]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    ctx = obj.get("context")
    if not isinstance(ctx, dict):
        errors.append("not_object:context")

    haz = obj.get("hazard_conditions")
    if not isinstance(haz, dict):
        errors.append("not_object:hazard_conditions")

    sev = obj.get("safety_events")
    if not isinstance(sev, list):
        errors.append("safety_events_not_list")

    rs = obj.get("risk_summary")
    if not isinstance(rs, dict):
        errors.append("not_object:risk_summary")

    rule_ok = True

    # confidence checks for context + hazards + events (best-effort)
    def check_conf(path: Tuple[Any, ...], tag: str) -> None:
        v = get_path(obj, path)
        if v is None:
            errors.append(f"missing_conf:{tag}")
            return
        if not is_confidence(v):
            errors.append(f"bad_conf:{tag}={v}")

    # context confidence when visible true
    for k in ["environment_type", "camera_view", "people_visible"]:
        vis = get_path(obj, ("context", k, "visible"))
        if vis is True:
            check_conf(("context", k, "confidence"), f"context.{k}.confidence")
        elif vis not in (True, False):
            errors.append(f"missing_or_bad_visible:context.{k}.visible")
            rule_ok = False

    # hazards: visible present + if visible true -> confidence + intermittent should be bool
    for hk in HAZARD_KEYS:
        vis = get_path(obj, ("hazard_conditions", hk, "visible"))
        if vis not in (True, False):
            errors.append(f"missing_or_bad_visible:hazard_conditions.{hk}.visible")
            rule_ok = False
            continue
        if vis is True:
            check_conf(("hazard_conditions", hk, "confidence"), f"hazard_conditions.{hk}.confidence")
            inter = get_path(obj, ("hazard_conditions", hk, "intermittent"))
            if inter is None:
                errors.append(f"missing_field:hazard_conditions.{hk}.intermittent")
                rule_ok = False
            elif not isinstance(inter, bool):
                errors.append(f"bad_type:hazard_conditions.{hk}.intermittent={inter}")
                rule_ok = False

    # safety_events
    if isinstance(sev, list):
        for i, ev in enumerate(sev):
            if not isinstance(ev, dict):
                errors.append(f"safety_events[{i}]_not_object")
                rule_ok = False
                continue
            c = ev.get("confidence")
            if c is None:
                errors.append(f"missing_conf:safety_events[{i}].confidence")
                rule_ok = False
            elif not is_confidence(c):
                errors.append(f"bad_conf:safety_events[{i}].confidence={c}")
                rule_ok = False
            inter = ev.get("intermittent")
            if inter is None:
                errors.append(f"missing_field:safety_events[{i}].intermittent")
                rule_ok = False
            elif not isinstance(inter, bool):
                errors.append(f"bad_type:safety_events[{i}].intermittent={inter}")
                rule_ok = False

    # Enum checks (non-wildcard)
    for path, allowed in ENUMS.items():
        if "*" in path:
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    # Enum checks (wildcard): hazards
    if isinstance(haz, dict):
        allowed = ENUMS[("hazard_conditions", "*", "value")]
        for hk in HAZARD_KEYS:
            v = get_path(obj, ("hazard_conditions", hk, "value"))
            _check_enum(v, allowed, f"hazard_conditions.{hk}.value", errors)

    # Enum checks (wildcard): safety_events
    if isinstance(sev, list):
        for i, ev in enumerate(sev):
            if not isinstance(ev, dict):
                continue
            _check_enum(ev.get("event_type"), ENUMS[("safety_events", "*", "event_type")], f"safety_events[{i}].event_type", errors)
            _check_enum(ev.get("severity"), ENUMS[("safety_events", "*", "severity")], f"safety_events[{i}].severity", errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.endswith("_not_list")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        for e in errors
    )

    if any(e.startswith("bad_conf:") or e.startswith("missing_or_bad_visible:") or e.startswith("bad_type:") for e in errors):
        rule_ok = False

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# Teacher-run consensus weighting
# ----------------------------

def _sym_global_slot_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa, _ = score_global_and_sections(a, b)
    sb, _ = score_global_and_sections(b, a)
    return 0.5 * (sa.global_macro_acc + sb.global_macro_acc)


def _sym_risk_summary_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return 0.5 * (score_risk_summary(a, b).macro_acc + score_risk_summary(b, a).macro_acc)


def _consensus_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gpair = _sym_global_slot_acc(a, b)
    rpair = _sym_risk_summary_acc(a, b)
    vals = [v for v in (gpair, rpair) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------
# CLI / main
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-gold", type=Path, default=GT_DIR)
    ap.add_argument("--results", type=Path, default=BASE_DIR / "results_industry")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "eval_out_industry")
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--model", action="append", default=None, help="Evaluate only specified model name(s). Can be repeated.")
    ap.add_argument(
        "--include-teacher-runs",
        action="store_true",
        help="Compute per-video consensus weights from teacher runs and report weighted means.",
    )
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

        consensus_weight = 1.0
        if args.include_teacher_runs:
            consensus_weight = compute_teacher_run_consensus_weight(
                args.results_gold,
                video_id,
                validate_struct,
                _consensus_pair_score,
            )

        student_files = discover_student_files(args.results, video_id)
        if args.model:
            allowed = {m.replace("-", "_") for m in args.model}
            if "all" not in allowed:
                student_files = [(m, p) for (m, p) in student_files if m.replace("-", "_") in allowed]

        if args.skip_missing_students and not student_files:
            continue

        log(f"Video={video_id}: found {len(student_files)} student outputs (consensus_weight={consensus_weight:.3f})")

        for model_name, student_path in student_files:
            student = read_json(student_path)
            vrep = validate_struct(student)

            global_slot = None
            context_slot = None
            hazard_slot = None
            safety_events_slot = None

            risk_summary_macro = None
            overall_risk_rating_acc = None
            primary_hazard_factors_f1 = None
            recommended_controls_f1 = None

            if vrep.parse_ok and "error" not in student:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                context_slot = sec.context_macro_acc
                hazard_slot = sec.hazard_conditions_macro_acc
                safety_events_slot = sec.safety_events_macro_acc

                rs = score_risk_summary(student, teacher)
                risk_summary_macro = rs.macro_acc
                overall_risk_rating_acc = rs.overall_risk_rating_acc
                primary_hazard_factors_f1 = rs.primary_hazard_factors_f1
                recommended_controls_f1 = rs.recommended_controls_f1

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": ("error" in student) if isinstance(student, dict) else False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "context_slot_macro_acc": context_slot,
                    "hazard_conditions_slot_macro_acc": hazard_slot,
                    "safety_events_macro_acc": safety_events_slot,
                    "risk_summary_macro_acc": risk_summary_macro,
                    "overall_risk_rating_acc": overall_risk_rating_acc,
                    "primary_hazard_factors_f1": primary_hazard_factors_f1,
                    "recommended_controls_f1": recommended_controls_f1,
                    "teacher_consensus_weight": consensus_weight,
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
                    "teacher_consensus_weight": consensus_weight,
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
            risk_summary_mean = mean_or_nan("risk_summary_macro_acc")
            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")

            weighted_global = float("nan")
            weighted_risk_summary = float("nan")
            if args.include_teacher_runs:
                weighted_global = weighted_mean(g2["global_slot_macro_acc"], g2["teacher_consensus_weight"])
                weighted_risk_summary = weighted_mean(g2["risk_summary_macro_acc"], g2["teacher_consensus_weight"])

            agg_rows.append(
                {
                    "model": model_name,
                    "n_videos": int(len(g2)),
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "context_slot_macro_acc_mean": mean_or_nan("context_slot_macro_acc"),
                    "hazard_conditions_slot_macro_acc_mean": mean_or_nan("hazard_conditions_slot_macro_acc"),
                    "safety_events_macro_acc_mean": mean_or_nan("safety_events_macro_acc"),
                    "risk_summary_macro_acc_mean": risk_summary_mean,
                    "overall_risk_rating_acc_mean": mean_or_nan("overall_risk_rating_acc"),
                    "primary_hazard_factors_f1_mean": mean_or_nan("primary_hazard_factors_f1"),
                    "recommended_controls_f1_mean": mean_or_nan("recommended_controls_f1"),
                    "weight_global_slot_macro_acc_mean": weighted_global,
                    "weight_risk_summary_macro_acc_mean": weighted_risk_summary,
                }
            )

    df_agg = pd.DataFrame(agg_rows)
    if not df_agg.empty:
        df_agg = df_agg.sort_values(
            by=["rule_rate", "global_slot_macro_acc_mean", "risk_summary_macro_acc_mean"],
            ascending=False,
        )

    df_agg_out = args.out / "model_summary.csv"
    df_agg.to_csv(df_agg_out, index=False)

    (args.out / "details.json").write_text(json.dumps(per_video_details, indent=2), encoding="utf-8")

    print(f"Wrote: {df_out}")
    print(f"Wrote: {df_agg_out}")
    print(f"Wrote: {args.out / 'details.json'}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
