#!/usr/bin/env python3
"""
Evaluator for the INDUSTRIAL-SAFETY ASSESSMENT JSON schema.

Schema (prompt-driven):
Top-level keys:
  - site_characterization
  - personnel_safety
  - hazardous_conditions
  - safety_violations
  - overall_site_risk

Scoring:
  - global_slot_macro_acc is a true macro over *all scorable leaves* across:
      site_characterization + personnel_safety + hazardous_conditions + safety_violations
    (overall_site_risk is scored separately, like previous "risk_summary".)

  - Per-section macro metrics:
      site_characterization_slot_macro_acc
      personnel_safety_slot_macro_acc
      hazardous_conditions_slot_macro_acc
      safety_violations_macro_acc
      global_slot_macro_acc

  - overall_site_risk metrics (separate):
      overall_site_risk_macro_acc
      risk_rating_acc
      primary_risk_factors_f1

Validation (best-effort):
  - parse_ok / schema_ok / rule_ok
  - light rules:
      * visible must be boolean where present
      * if visible==True, confidence must be in [0,1] (best-effort)
      * enum validation for declared enum fields
      * type checks for booleans/integers where specified
      * PPE compliance values must be in {compliant, non_compliant, unknown}

Teacher-run consensus weighting (optional):
  - consensus weight computed from teacher runs only:
      sym_global_slot_acc
      sym_overall_site_risk_acc
    and then averaged 50/50

Dependencies:
  pip install numpy pandas
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from utils.eval_common import (
    _set_f1,
    compute_teacher_run_consensus_weight,
    build_student_index,
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_confidence,
    read_json,
)
from utils.latex_table import write_latex_table
from utils.model_sort import format_model_name, sort_model_summary
from utils.run_paths import task_eval_dir, task_results_dir

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "industry"
GT_DIR = f"/opt/dataset/ds_{TASK_NAME}_ripulito/test_dataset_json"
DEFAULT_RESULTS_DIR = task_results_dir(TASK_NAME)
DEFAULT_OUT_DIR = task_eval_dir(TASK_NAME)


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], set] = {
    # site_characterization
    ("site_characterization", "site_type", "value"): {
        "construction_site",
        "oil_gas_platform",
        "refinery",
        "industrial_plant",
        "unknown",
    },
    ("site_characterization", "activity_level", "value"): {"inactive", "low", "high", "emergency"},
    # personnel_safety.ppe_compliance
    ("personnel_safety", "ppe_compliance", "*", "value"): {"compliant", "non_compliant", "unknown"},
    # safety_violations array
    ("safety_violations", "*", "violation_type"): {"no_ppe", "restricted_area_entry", "unsafe_lifting", "fall_risk", "other"},
    ("safety_violations", "*", "risk_level"): {"low", "medium", "high", "extreme"},
    # overall_site_risk
    ("overall_site_risk", "risk_rating", "value"): {"safe", "cautionary", "hazardous", "critical"},
}

HAZARD_KEYS = [
    "unsecured_heights",
    "heavy_machinery_movement",
    "visible_leaks_or_fire",
]

PPE_KEYS = [
    "hard_hats",
    "high_visibility_vests",
    "specialized_gear",
]


# ----------------------------
# Slot paths (per-section)
# ----------------------------

SITE_CHARACTERIZATION_SLOTS: List[Tuple[Any, ...]] = [
    ("site_characterization", "site_type", "visible"),
    ("site_characterization", "site_type", "value"),
    ("site_characterization", "site_type", "confidence"),
    ("site_characterization", "activity_level", "value"),
    ("site_characterization", "activity_level", "confidence"),
]

PERSONNEL_SAFETY_SLOTS: List[Tuple[Any, ...]] = [
    ("personnel_safety", "workers_present", "visible"),
    ("personnel_safety", "workers_present", "value"),
    ("personnel_safety", "workers_present", "count"),
    ("personnel_safety", "workers_present", "confidence"),
]
for pk in PPE_KEYS:
    PERSONNEL_SAFETY_SLOTS.extend(
        [
            ("personnel_safety", "ppe_compliance", pk, "value"),
            ("personnel_safety", "ppe_compliance", pk, "confidence"),
        ]
    )

HAZARDOUS_CONDITIONS_SLOTS: List[Tuple[Any, ...]] = []
for hk in HAZARD_KEYS:
    HAZARDOUS_CONDITIONS_SLOTS.extend(
        [
            ("hazardous_conditions", hk, "visible"),
            ("hazardous_conditions", hk, "value"),
            ("hazardous_conditions", hk, "confidence"),
        ]
    )


# ----------------------------
# Teacher gating for slot scoring
# ----------------------------

def _visible_path_for(path: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
    """
    Maps any leaf within a "visible/value/confidence/count" group to its "visible" path.
    """
    if len(path) >= 3 and path[-1] in {"value", "confidence", "count"}:
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
# safety_violations scoring (bag-of-violations)
# ----------------------------

@dataclass
class ViolationScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    hits: List[float]


def _violation_key(v: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Use (violation_type, risk_level) as the identity of a violation.
    Ignore 'description' (free text) and do not score confidence directly.
    """
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

    # count accuracy: exact match per teacher key (missing => 0)
    count_hits: List[float] = []
    for k in t_keys:
        count_hits.append(1.0 if s_counts.get(k, 0) == t_counts.get(k, 0) else 0.0)
    count_acc = float(np.mean(count_hits)) if count_hits else 1.0

    hits = [presence_f1, count_acc]
    macro = float(np.mean(hits)) if hits else 0.0
    return ViolationScores(macro_acc=macro, presence_f1=presence_f1, count_acc=count_acc, hits=hits)


# ----------------------------
# overall_site_risk scoring (separate)
# ----------------------------

@dataclass
class OverallSiteRiskScores:
    macro_acc: float
    risk_rating_acc: float
    primary_risk_factors_f1: float


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


def score_overall_site_risk(student: Dict[str, Any], teacher: Dict[str, Any]) -> OverallSiteRiskScores:
    s = student.get("overall_site_risk")
    t = teacher.get("overall_site_risk")
    if not isinstance(s, dict) or not isinstance(t, dict):
        return OverallSiteRiskScores(macro_acc=0.0, risk_rating_acc=0.0, primary_risk_factors_f1=0.0)

    s_rr = get_path(s, ("risk_rating", "value"))
    t_rr = get_path(t, ("risk_rating", "value"))
    risk_rating_acc = 1.0 if s_rr == t_rr else 0.0

    s_prf = tcast_list_of_str(s.get("primary_risk_factors"))
    t_prf = tcast_list_of_str(t.get("primary_risk_factors"))
    prf_f1 = _set_f1(s_prf, t_prf)

    vals = [risk_rating_acc, prf_f1]
    macro = float(np.mean(vals)) if vals else 0.0
    return OverallSiteRiskScores(macro_acc=macro, risk_rating_acc=risk_rating_acc, primary_risk_factors_f1=prf_f1)


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


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    """
    Returns:
      - section/global macro scores
      - raw hit lists per section (used to compute global as a true macro over leaves)
    """
    sc = _score_leaf_slots(student, teacher, SITE_CHARACTERIZATION_SLOTS)
    ps = _score_leaf_slots(student, teacher, PERSONNEL_SAFETY_SLOTS)
    hz = _score_leaf_slots(student, teacher, HAZARDOUS_CONDITIONS_SLOTS)
    sv = score_safety_violations(student, teacher)

    all_hits = sc.hits + ps.hits + hz.hits + sv.hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        site_characterization_macro_acc=sc.macro_acc,
        personnel_safety_macro_acc=ps.macro_acc,
        hazardous_conditions_macro_acc=hz.macro_acc,
        safety_violations_macro_acc=sv.macro_acc,
    )
    return sec, {
        "site_characterization": sc.hits,
        "personnel_safety": ps.hits,
        "hazardous_conditions": hz.hits,
        "safety_violations": sv.hits,
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


def _check_bool(value: Any, tag: str, errors: List[str]) -> None:
    if value is None:
        errors.append(f"missing_bool:{tag}")
    elif not isinstance(value, bool):
        errors.append(f"bad_bool:{tag}={value}")


def _check_int(value: Any, tag: str, errors: List[str]) -> None:
    if value is None:
        errors.append(f"missing_int:{tag}")
    elif not isinstance(value, int):
        errors.append(f"bad_int:{tag}={value}")


def validate_struct(obj: Optional[Dict[str, Any]]) -> ValidationReport:
    if obj is None or not isinstance(obj, dict):
        return ValidationReport(parse_ok=False, schema_ok=False, rule_ok=False, errors=["parse_failed_or_not_object"])

    errors: List[str] = []

    required_top = ["site_characterization", "personnel_safety", "hazardous_conditions", "safety_violations", "overall_site_risk"]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    sc = obj.get("site_characterization")
    if not isinstance(sc, dict):
        errors.append("not_object:site_characterization")

    ps = obj.get("personnel_safety")
    if not isinstance(ps, dict):
        errors.append("not_object:personnel_safety")

    hz = obj.get("hazardous_conditions")
    if not isinstance(hz, dict):
        errors.append("not_object:hazardous_conditions")

    sv = obj.get("safety_violations")
    if not isinstance(sv, list):
        errors.append("safety_violations_not_list")

    osr = obj.get("overall_site_risk")
    if not isinstance(osr, dict):
        errors.append("not_object:overall_site_risk")

    rule_ok = True

    def check_conf(path: Tuple[Any, ...], tag: str) -> None:
        v = get_path(obj, path)
        if v is None:
            errors.append(f"missing_conf:{tag}")
            return
        if not is_confidence(v):
            errors.append(f"bad_conf:{tag}={v}")

    # site_characterization.site_type (visible-gated)
    vis = get_path(obj, ("site_characterization", "site_type", "visible"))
    if vis not in (True, False):
        errors.append("missing_or_bad_visible:site_characterization.site_type.visible")
        rule_ok = False
    elif vis is True:
        check_conf(("site_characterization", "site_type", "confidence"), "site_characterization.site_type.confidence")

    # activity_level has no visible; still validate confidence best-effort
    check_conf(("site_characterization", "activity_level", "confidence"), "site_characterization.activity_level.confidence")

    # personnel_safety.workers_present (visible-gated)
    wvis = get_path(obj, ("personnel_safety", "workers_present", "visible"))
    if wvis not in (True, False):
        errors.append("missing_or_bad_visible:personnel_safety.workers_present.visible")
        rule_ok = False
    elif wvis is True:
        check_conf(("personnel_safety", "workers_present", "confidence"), "personnel_safety.workers_present.confidence")
        _check_bool(get_path(obj, ("personnel_safety", "workers_present", "value")), "personnel_safety.workers_present.value", errors)
        _check_int(get_path(obj, ("personnel_safety", "workers_present", "count")), "personnel_safety.workers_present.count", errors)

    # PPE compliance (always present in schema; validate enums + confidence)
    for pk in PPE_KEYS:
        pv = get_path(obj, ("personnel_safety", "ppe_compliance", pk, "value"))
        _check_enum(pv, ENUMS[("personnel_safety", "ppe_compliance", "*", "value")], f"personnel_safety.ppe_compliance.{pk}.value", errors)
        check_conf(("personnel_safety", "ppe_compliance", pk, "confidence"), f"personnel_safety.ppe_compliance.{pk}.confidence")

    # hazardous_conditions (visible-gated; value must be boolean if visible True)
    for hk in HAZARD_KEYS:
        hvis = get_path(obj, ("hazardous_conditions", hk, "visible"))
        if hvis not in (True, False):
            errors.append(f"missing_or_bad_visible:hazardous_conditions.{hk}.visible")
            rule_ok = False
            continue
        if hvis is True:
            check_conf(("hazardous_conditions", hk, "confidence"), f"hazardous_conditions.{hk}.confidence")
            _check_bool(get_path(obj, ("hazardous_conditions", hk, "value")), f"hazardous_conditions.{hk}.value", errors)

    # safety_violations list elements: enums + confidence + count
    if isinstance(sv, list):
        for i, v in enumerate(sv):
            if not isinstance(v, dict):
                errors.append(f"safety_violations[{i}]_not_object")
                rule_ok = False
                continue
            _check_enum(v.get("violation_type"), ENUMS[("safety_violations", "*", "violation_type")], f"safety_violations[{i}].violation_type", errors)
            _check_enum(v.get("risk_level"), ENUMS[("safety_violations", "*", "risk_level")], f"safety_violations[{i}].risk_level", errors)
            c = v.get("confidence")
            if c is None:
                errors.append(f"missing_conf:safety_violations[{i}].confidence")
                rule_ok = False
            elif not is_confidence(c):
                errors.append(f"bad_conf:safety_violations[{i}].confidence={c}")
                rule_ok = False
            _check_int(v.get("count"), f"safety_violations[{i}].count", errors)

    # overall_site_risk: validate risk_rating enum + confidence; primary_risk_factors list presence
    rr_val = get_path(obj, ("overall_site_risk", "risk_rating", "value"))
    _check_enum(rr_val, ENUMS[("overall_site_risk", "risk_rating", "value")], "overall_site_risk.risk_rating.value", errors)
    check_conf(("overall_site_risk", "risk_rating", "confidence"), "overall_site_risk.risk_rating.confidence")
    prf = get_path(obj, ("overall_site_risk", "primary_risk_factors"))
    if prf is None:
        errors.append("missing_field:overall_site_risk.primary_risk_factors")
        rule_ok = False
    elif not isinstance(prf, list):
        errors.append("bad_type:overall_site_risk.primary_risk_factors")
        rule_ok = False

    # Enum checks (non-wildcard)
    for path, allowed in ENUMS.items():
        if "*" in path:
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.endswith("_not_list")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        or e.startswith("missing_bool:")
        or e.startswith("missing_int:")
        for e in errors
    )

    if any(
        e.startswith("bad_conf:")
        or e.startswith("missing_conf:")
        or e.startswith("missing_or_bad_visible:")
        or e.startswith("bad_type:")
        or e.startswith("bad_bool:")
        or e.startswith("bad_int:")
        for e in errors
    ):
        rule_ok = False

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# Teacher-run consensus weighting
# ----------------------------

def _sym_global_slot_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa, _ = score_global_and_sections(a, b)
    sb, _ = score_global_and_sections(b, a)
    return 0.5 * (sa.global_macro_acc + sb.global_macro_acc)


def _sym_overall_site_risk_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return 0.5 * (score_overall_site_risk(a, b).macro_acc + score_overall_site_risk(b, a).macro_acc)


def _consensus_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gpair = _sym_global_slot_acc(a, b)
    rpair = _sym_overall_site_risk_acc(a, b)
    vals = [v for v in (gpair, rpair) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


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
    if args.results == DEFAULT_RESULTS_DIR and not args.results.exists():
        legacy_results = BASE_DIR / "results_industry"
        if legacy_results.exists():
            args.results = legacy_results
    args.out.mkdir(parents=True, exist_ok=True)
    verbose = bool(args.verbose)
    df_agg_out = args.out / "model_summary.csv"

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    if df_agg_out.exists():
        log(f"Overwriting existing summary: {df_agg_out}")

    teacher_standards = discover_teacher_standards(args.results_gold)
    if args.limit_videos is not None:
        teacher_standards = teacher_standards[: args.limit_videos]
    log(f"Found {len(teacher_standards)} teacher standards in {args.results_gold}")

    student_index = build_student_index(args.results, video_ids={vid for (vid, _p) in teacher_standards})

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

        student_files = discover_student_files(args.results, video_id, index=student_index)
        if args.model:
            allowed = {m.replace("-", "_") for m in args.model}
            if "all" not in allowed:
                student_files = [(m, p) for (m, p) in student_files if m.replace("-", "_") in allowed]
        if args.skip_missing_students and not student_files:
            continue
        log(f"Video={video_id}: found {len(student_files)} student outputs (consensus_weight={consensus_weight:.3f})")

        for model_name, student_path in student_files:
            student = read_json(student_path)
            if student is None or not isinstance(student, dict):
                log(f"Skipping student (parse failed or not object): {student_path}")
                continue
            if "error" in student:
                log(f"Skipping student (error present): {student_path}")
                continue
            vrep = validate_struct(student)

            global_slot = None
            site_characterization_slot = None
            personnel_safety_slot = None
            hazardous_conditions_slot = None
            safety_violations_slot = None

            overall_site_risk_macro = None
            risk_rating_acc = None
            primary_risk_factors_f1 = None

            if vrep.parse_ok:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                site_characterization_slot = sec.site_characterization_macro_acc
                personnel_safety_slot = sec.personnel_safety_macro_acc
                hazardous_conditions_slot = sec.hazardous_conditions_macro_acc
                safety_violations_slot = sec.safety_violations_macro_acc

                osr = score_overall_site_risk(student, teacher)
                overall_site_risk_macro = osr.macro_acc
                risk_rating_acc = osr.risk_rating_acc
                primary_risk_factors_f1 = osr.primary_risk_factors_f1

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "site_characterization_slot_macro_acc": site_characterization_slot,
                    "personnel_safety_slot_macro_acc": personnel_safety_slot,
                    "hazardous_conditions_slot_macro_acc": hazardous_conditions_slot,
                    "safety_violations_macro_acc": safety_violations_slot,
                    "overall_site_risk_macro_acc": overall_site_risk_macro,
                    "risk_rating_acc": risk_rating_acc,
                    "primary_risk_factors_f1": primary_risk_factors_f1,
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
            osr_mean = mean_or_nan("overall_site_risk_macro_acc")
            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")

            agg_rows.append(
                {
                    "model": model_name,
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "site_characterization_slot_macro_acc_mean": mean_or_nan("site_characterization_slot_macro_acc"),
                    "personnel_safety_slot_macro_acc_mean": mean_or_nan("personnel_safety_slot_macro_acc"),
                    "hazardous_conditions_slot_macro_acc_mean": mean_or_nan("hazardous_conditions_slot_macro_acc"),
                    "safety_violations_macro_acc_mean": mean_or_nan("safety_violations_macro_acc"),
                    "overall_site_risk_macro_acc_mean": osr_mean,
                    "risk_rating_acc_mean": mean_or_nan("risk_rating_acc"),
                    "primary_risk_factors_f1_mean": mean_or_nan("primary_risk_factors_f1"),
                }
            )

    df_agg = pd.DataFrame(agg_rows)
    if not df_agg.empty:
        df_agg["model"] = df_agg["model"].map(format_model_name)

    df_agg = sort_model_summary(df_agg)
    df_agg.to_csv(df_agg_out, index=False)
    latex_out = args.out / "model_summary_latex.txt"
    write_latex_table(df_agg, latex_out)

    details_path = args.out / "details.json"
    details_path.write_text(json.dumps(per_video_details, indent=2), encoding="utf-8")

    print(f"Wrote: {df_out}")
    print(f"Wrote: {df_agg_out}")
    print(f"Wrote: {details_path}")
    print(f"Wrote: {latex_out}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
