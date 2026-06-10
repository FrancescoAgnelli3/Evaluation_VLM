#!/usr/bin/env python3
"""
Evaluator for the PUBLIC-SAFETY (urban surveillance) JSON schema.

Scoring philosophy
------------------
1) "Slot accuracy" is a GLOBAL metric computed over *all scorable leaves* across:
   - scene_context
   - crowd_analysis
   - violence_and_aggression
   plus a dedicated section metric for specific_events.
   risk_assessment is kept separate (dedicated metrics), but can be included in teacher-run consensus weighting.

2) Per-section macro metrics are also reported:
   - scene_context_slot_macro_acc
   - crowd_analysis_slot_macro_acc
   - violence_and_aggression_slot_macro_acc
   - specific_events_macro_acc
   - global_slot_macro_acc  (true macro over all scored leaves across non-risk sections + event metrics)

3) Epistemic gating:
   - score all ".visible"
   - score ".value" only if teacher says the corresponding element is visible==true (when that field has a visible flag)
   - fields without a visible flag (e.g., crowd_analysis.estimated_count, crowd_analysis.crowd_dynamic, risk_assessment.overall_threat_level)
     are always scored on value (no gating possible by schema).

Specific events scoring (bag-of-events; counts ignored for presence F1)
----------------------------------------------------------------------
- presence_f1 over event_type (set-based)
- count_acc over event_type counts (exact)
- persons_count_acc over event_type -> involved_persons_count (exact multiset per type)
- severity_acc over event_type -> severity (exact multiset per type)

Text fields:
- specific_events[].description is NOT scored (too brittle)
- stalking_behavior_observed.description is NOT scored (too brittle)
- weapons_detected.types is scored as set-F1 when teacher indicates weapons_detected is visible==true and value==true.
- risk_assessment.main_threat_factors scored as set-F1.

Teacher-run consensus (optional)
--------------------------------
If --include-teacher-runs is enabled, computes per-video consensus weights from teacher runs only,
and reports weighted means for:
  - weight_global_slot_macro_acc_mean
  - weight_specific_events_macro_acc_mean
  - weight_risk_assessment_macro_acc_mean

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
    discover_student_files,
    discover_teacher_standards,
    get_path,
    is_confidence,
    is_int_or_unknown,
    read_json,
)
from utils.latex_table import write_latex_table
from utils.model_sort import format_model_name, sort_model_summary
from utils.run_paths import task_eval_dir, task_results_dir

BASE_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = "people"
GT_DIR = "/opt/dataset/ds_people/test_dataset_json"#f"/opt/dataset/ds_{TASK_NAME}_ripulito/test_dataset_json"
DEFAULT_RESULTS_DIR = task_results_dir(TASK_NAME)
DEFAULT_OUT_DIR = task_eval_dir(TASK_NAME)


# ----------------------------
# Enums
# ----------------------------

ENUMS: Dict[Tuple[Any, ...], set] = {
    ("scene_context", "location_type", "value"): {"street", "square", "park", "indoor_public", "unknown"},
    ("scene_context", "lighting", "value"): {"daylight", "low_light", "night", "unknown"},
    ("crowd_analysis", "crowd_dynamic", "value"): {"static", "peaceful_moving", "agitated", "violent_disorder", "unknown"},
    ("specific_events", "*", "event_type"): {"assault", "brawl", "riot", "armed_threat", "harassment", "crowd_surge"},
    ("specific_events", "*", "severity"): {"low", "medium", "high", "critical"},
    ("risk_assessment", "overall_threat_level", "value"): {"low", "medium", "high", "critical"},
}


# ----------------------------
# Slot paths (per-section)
# ----------------------------

SCENE_CONTEXT_SLOTS: List[Tuple[Any, ...]] = [
    ("scene_context", "location_type", "visible"),
    ("scene_context", "location_type", "value"),
    ("scene_context", "lighting", "visible"),
    ("scene_context", "lighting", "value"),
]

# Note: estimated_count and crowd_dynamic have no "visible" in the schema, so only score their values.
CROWD_ANALYSIS_SLOTS: List[Tuple[Any, ...]] = [
    ("crowd_analysis", "gathering_present", "visible"),
    ("crowd_analysis", "gathering_present", "value"),
    ("crowd_analysis", "estimated_count", "value"),
    ("crowd_analysis", "crowd_dynamic", "value"),
]

VIOLENCE_AGGR_SLOTS: List[Tuple[Any, ...]] = [
    ("violence_and_aggression", "physical_altercation", "visible"),
    ("violence_and_aggression", "physical_altercation", "value"),
    ("violence_and_aggression", "weapons_detected", "visible"),
    ("violence_and_aggression", "weapons_detected", "value"),
    ("violence_and_aggression", "stalking_behavior_observed", "visible"),
    ("violence_and_aggression", "stalking_behavior_observed", "value"),
]

# Dedicated list-valued paths
LIST_VALUE_PATHS = {
    "weapons_detected_types": ("violence_and_aggression", "weapons_detected", "types"),  # list[str]
    "risk_main_threat_factors": ("risk_assessment", "main_threat_factors"),  # list[str]
}


# ----------------------------
# Teacher gating for leaf slot scoring
# ----------------------------

def should_score_slot(path: Tuple[Any, ...], teacher: Dict[str, Any]) -> bool:
    """
    General rule:
      - score all ...visible
      - score ...value only if corresponding ...visible is True in teacher, when pattern is (<sec>, <field>, "value")
      - score other paths as-is (e.g., fields without visible)
    """
    if not path:
        return True
    if path[-1] == "visible":
        return True

    if len(path) == 3 and path[-1] == "value":
        sec, field, _ = path
        vis = get_path(teacher, (sec, field, "visible"))
        if isinstance(vis, bool):
            return vis is True
        return True  # no visible in schema => no gating

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


# ----------------------------
# List-valued scoring (set-based)
# ----------------------------

@dataclass
class ListValueScores:
    macro_acc: float
    f1: float
    hits: List[float]


def _as_list_of_str(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x:
        if isinstance(it, str):
            out.append(it)
    return out


def score_weapons_types(student: Dict[str, Any], teacher: Dict[str, Any]) -> ListValueScores:
    """
    Score weapons_detected.types as set-F1, only when teacher says weapons_detected is visible==True AND value==True.
    """
    vis_t = get_path(teacher, ("violence_and_aggression", "weapons_detected", "visible"))
    val_t = get_path(teacher, ("violence_and_aggression", "weapons_detected", "value"))
    if vis_t is not True or val_t is not True:
        return ListValueScores(macro_acc=0.0, f1=0.0, hits=[])

    s_val = get_path(student, LIST_VALUE_PATHS["weapons_detected_types"])
    t_val = get_path(teacher, LIST_VALUE_PATHS["weapons_detected_types"])
    s_list = _as_list_of_str(s_val)
    t_list = _as_list_of_str(t_val)
    f1 = _set_f1(s_list, t_list)
    return ListValueScores(macro_acc=f1, f1=f1, hits=[f1])


def score_list_of_strings(student: Dict[str, Any], teacher: Dict[str, Any], base_path: Tuple[Any, ...]) -> ListValueScores:
    """
    Score list[str] as set-F1 (no gating by visible unless caller enforces it).
    """
    s_val = get_path(student, base_path)
    t_val = get_path(teacher, base_path)
    s_list = _as_list_of_str(s_val)
    t_list = _as_list_of_str(t_val)
    f1 = _set_f1(s_list, t_list)
    return ListValueScores(macro_acc=f1, f1=f1, hits=[f1])


# ----------------------------
# Specific events scoring (bag-of-events)
# ----------------------------

@dataclass
class EventScores:
    macro_acc: float
    presence_f1: float
    count_acc: float
    persons_count_acc: float
    severity_acc: float
    hits: List[float]


def _event_type(ev: Dict[str, Any]) -> Optional[str]:
    et = ev.get("event_type")
    return et if isinstance(et, str) else None


def _event_persons(ev: Dict[str, Any]) -> Any:
    return ev.get("involved_persons_count")


def _event_severity(ev: Dict[str, Any]) -> Any:
    return ev.get("severity")


def _events_by_type(arr: Any) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(arr, list):
        return out
    for ev in arr:
        if not isinstance(ev, dict):
            continue
        k = _event_type(ev)
        if k is None:
            continue
        out.setdefault(k, []).append(ev)
    return out


def score_specific_events(student: Dict[str, Any], teacher: Dict[str, Any]) -> EventScores:
    s = student.get("specific_events")
    t = teacher.get("specific_events")

    s_map = _events_by_type(s)
    t_map = _events_by_type(t)

    s_types = set(s_map.keys())
    t_types = set(t_map.keys())

    if not t_types:
        return EventScores(
            macro_acc=float("nan"),
            presence_f1=float("nan"),
            count_acc=float("nan"),
            persons_count_acc=float("nan"),
            severity_acc=float("nan"),
            hits=[],
        )

    presence_f1 = _set_f1(s_types, t_types)

    # count accuracy per type (exact)
    count_hits: List[float] = []
    for et in t_types:
        count_hits.append(1.0 if len(s_map.get(et, [])) == len(t_map.get(et, [])) else 0.0)
    count_acc = float(np.mean(count_hits)) if count_hits else 1.0

    # persons_count_acc: compare multiset of involved_persons_count per type (exact as strings)
    pc_hits: List[float] = []
    for et in t_types:
        s_counts = [str(_event_persons(ev)) for ev in s_map.get(et, [])]
        t_counts = [str(_event_persons(ev)) for ev in t_map.get(et, [])]
        pc_hits.append(1.0 if sorted(s_counts) == sorted(t_counts) else 0.0)
    persons_count_acc = float(np.mean(pc_hits)) if pc_hits else 1.0

    # severity_acc: compare multiset of severity per type (exact as strings)
    sev_hits: List[float] = []
    for et in t_types:
        s_sev = [str(_event_severity(ev)) for ev in s_map.get(et, [])]
        t_sev = [str(_event_severity(ev)) for ev in t_map.get(et, [])]
        sev_hits.append(1.0 if sorted(s_sev) == sorted(t_sev) else 0.0)
    severity_acc = float(np.mean(sev_hits)) if sev_hits else 1.0

    hits = [presence_f1, count_acc, persons_count_acc, severity_acc]
    macro = float(np.mean(hits)) if hits else 0.0
    return EventScores(
        macro_acc=macro,
        presence_f1=presence_f1,
        count_acc=count_acc,
        persons_count_acc=persons_count_acc,
        severity_acc=severity_acc,
        hits=hits,
    )


# ----------------------------
# Risk assessment scoring (separate)
# ----------------------------

@dataclass
class RiskScores:
    macro_acc: float
    overall_threat_level_acc: float
    main_threat_factors_f1: float
    immediate_intervention_required_acc: float
    hits: List[float]


def score_risk_assessment(student: Dict[str, Any], teacher: Dict[str, Any]) -> RiskScores:
    # overall_threat_level: dict {value, confidence}
    s_lvl = get_path(student, ("risk_assessment", "overall_threat_level", "value"))
    t_lvl = get_path(teacher, ("risk_assessment", "overall_threat_level", "value"))
    lvl_acc = 1.0 if s_lvl == t_lvl else 0.0

    # main_threat_factors: list[str] set-F1
    mtf = score_list_of_strings(student, teacher, LIST_VALUE_PATHS["risk_main_threat_factors"])
    mtf_f1 = mtf.f1

    # immediate_intervention_required: bool
    s_iir = get_path(student, ("risk_assessment", "immediate_intervention_required"))
    t_iir = get_path(teacher, ("risk_assessment", "immediate_intervention_required"))
    iir_acc = 1.0 if s_iir == t_iir else 0.0

    hits = [lvl_acc, mtf_f1, iir_acc]
    return RiskScores(
        macro_acc=float(np.mean(hits)) if hits else 0.0,
        overall_threat_level_acc=lvl_acc,
        main_threat_factors_f1=mtf_f1,
        immediate_intervention_required_acc=iir_acc,
        hits=hits,
    )


# ----------------------------
# Global slot metric aggregation
# ----------------------------

@dataclass
class GlobalSlotScores:
    global_macro_acc: float
    scene_context_macro_acc: float
    crowd_analysis_macro_acc: float
    violence_and_aggression_macro_acc: float
    specific_events_macro_acc: float
    risk_assessment_macro_acc: float


def score_global_and_sections(student: Dict[str, Any], teacher: Dict[str, Any]) -> Tuple[GlobalSlotScores, Dict[str, List[float]]]:
    """
    Returns:
      - section/global macro scores
      - raw hit lists per section (used to compute global as a true macro over leaves)
    """
    sc = _score_leaf_slots(student, teacher, SCENE_CONTEXT_SLOTS)
    ca = _score_leaf_slots(student, teacher, CROWD_ANALYSIS_SLOTS)
    va = _score_leaf_slots(student, teacher, VIOLENCE_AGGR_SLOTS)

    # List-valued extras (each contributes one hit)
    wtypes = score_weapons_types(student, teacher)

    ev = score_specific_events(student, teacher)
    risk = score_risk_assessment(student, teacher)

    scene_hits = sc.hits
    crowd_hits = ca.hits
    violence_hits = va.hits + wtypes.hits
    events_hits = ev.hits

    all_hits = scene_hits + crowd_hits + violence_hits + events_hits
    global_macro = float(np.mean(all_hits)) if all_hits else 0.0

    sec = GlobalSlotScores(
        global_macro_acc=global_macro,
        scene_context_macro_acc=sc.macro_acc,
        crowd_analysis_macro_acc=ca.macro_acc,
        violence_and_aggression_macro_acc=float(np.mean(violence_hits)) if violence_hits else 0.0,
        specific_events_macro_acc=ev.macro_acc,
        risk_assessment_macro_acc=risk.macro_acc,
    )
    return sec, {
        "scene_context": scene_hits,
        "crowd_analysis": crowd_hits,
        "violence_and_aggression": violence_hits,
        "specific_events": events_hits,
        "risk_assessment": risk.hits,
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


def validate_struct(obj: Optional[Dict[str, Any]]) -> ValidationReport:
    if obj is None:
        return ValidationReport(parse_ok=False, schema_ok=False, rule_ok=False, errors=["parse_failed_or_not_object"])

    errors: List[str] = []
    rule_ok = True

    required_top = [
        "scene_context",
        "crowd_analysis",
        "violence_and_aggression",
        "specific_events",
        "risk_assessment",
    ]
    for k in required_top:
        if k not in obj:
            errors.append(f"missing_top_level:{k}")

    # type checks
    if not isinstance(obj.get("scene_context"), dict):
        errors.append("not_object:scene_context")
    if not isinstance(obj.get("crowd_analysis"), dict):
        errors.append("not_object:crowd_analysis")
    if not isinstance(obj.get("violence_and_aggression"), dict):
        errors.append("not_object:violence_and_aggression")
    if not isinstance(obj.get("risk_assessment"), dict):
        errors.append("not_object:risk_assessment")

    sev = obj.get("specific_events")
    if not isinstance(sev, list):
        errors.append("specific_events_not_list")

    # helpers
    def check_triplet(sec: str, field: str) -> None:
        base = get_path(obj, (sec, field))
        if not isinstance(base, dict):
            errors.append(f"not_object:{sec}.{field}")
            return
        v = base.get("visible")
        if not isinstance(v, bool):
            errors.append(f"bad_visible:{sec}.{field}.visible={v}")
        c = base.get("confidence")
        if c is None:
            errors.append(f"missing_conf:{sec}.{field}.confidence")
        elif not is_confidence(c):
            errors.append(f"bad_conf:{sec}.{field}.confidence={c}")

    def check_value_conf(sec: str, field: str) -> None:
        base = get_path(obj, (sec, field))
        if not isinstance(base, dict):
            errors.append(f"not_object:{sec}.{field}")
            return
        if "value" not in base:
            errors.append(f"missing_value:{sec}.{field}.value")
        c = base.get("confidence")
        if c is None:
            errors.append(f"missing_conf:{sec}.{field}.confidence")
        elif not is_confidence(c):
            errors.append(f"bad_conf:{sec}.{field}.confidence={c}")

    # scene_context
    for fld in ["location_type", "lighting"]:
        check_triplet("scene_context", fld)

    # crowd_analysis
    check_triplet("crowd_analysis", "gathering_present")
    check_value_conf("crowd_analysis", "estimated_count")
    check_value_conf("crowd_analysis", "crowd_dynamic")

    est_count = get_path(obj, ("crowd_analysis", "estimated_count", "value"))
    if not is_int_or_unknown(est_count):
        errors.append(f"bad_value:crowd_analysis.estimated_count.value={est_count}")
        rule_ok = False

    # violence_and_aggression
    for fld in ["physical_altercation", "weapons_detected", "stalking_behavior_observed"]:
        check_triplet("violence_and_aggression", fld)

    # weapons_detected.types and description requirements
    wd = get_path(obj, ("violence_and_aggression", "weapons_detected"))
    if isinstance(wd, dict):
        types = wd.get("types")
        if types is None:
            errors.append("missing_field:violence_and_aggression.weapons_detected.types")
            rule_ok = False
        elif not isinstance(types, list):
            errors.append("bad_type:violence_and_aggression.weapons_detected.types_not_list")
            rule_ok = False
        else:
            for i, it in enumerate(types):
                if not isinstance(it, str):
                    errors.append(f"bad_type:weapons_detected.types[{i}]_not_str")
                    rule_ok = False

    sb = get_path(obj, ("violence_and_aggression", "stalking_behavior_observed"))
    if isinstance(sb, dict):
        desc = sb.get("description")
        if desc is None or not isinstance(desc, str):
            errors.append("missing_or_bad_type:violence_and_aggression.stalking_behavior_observed.description")
            rule_ok = False

    # specific_events checks
    if isinstance(sev, list):
        for i, e in enumerate(sev):
            if not isinstance(e, dict):
                errors.append(f"specific_events[{i}]_not_object")
                rule_ok = False
                continue
            et = e.get("event_type")
            if not isinstance(et, str):
                errors.append(f"specific_events[{i}].event_type_missing_or_not_str")
                rule_ok = False
            pc = e.get("involved_persons_count")
            if not (isinstance(pc, int) and not isinstance(pc, bool)):
                errors.append(f"specific_events[{i}].involved_persons_count_bad={pc}")
                rule_ok = False
            sev_v = e.get("severity")
            if not isinstance(sev_v, str):
                errors.append(f"specific_events[{i}].severity_missing_or_not_str")
                rule_ok = False
            c = e.get("confidence")
            if c is None:
                errors.append(f"missing_conf:specific_events[{i}].confidence")
                rule_ok = False
            elif not is_confidence(c):
                errors.append(f"bad_conf:specific_events[{i}].confidence={c}")
                rule_ok = False
            desc = e.get("description")
            if desc is None or not isinstance(desc, str):
                errors.append(f"specific_events[{i}].description_missing_or_not_str")
                rule_ok = False

    # risk_assessment checks
    check_value_conf("risk_assessment", "overall_threat_level")

    mtf = get_path(obj, ("risk_assessment", "main_threat_factors"))
    if mtf is None:
        errors.append("missing_field:risk_assessment.main_threat_factors")
        rule_ok = False
    elif not isinstance(mtf, list):
        errors.append("bad_type:risk_assessment.main_threat_factors_not_list")
        rule_ok = False
    else:
        for i, it in enumerate(mtf):
            if not isinstance(it, str):
                errors.append(f"bad_type:risk_assessment.main_threat_factors[{i}]_not_str")
                rule_ok = False

    iir = get_path(obj, ("risk_assessment", "immediate_intervention_required"))
    if not isinstance(iir, bool):
        errors.append(f"bad_type:risk_assessment.immediate_intervention_required={iir}")
        rule_ok = False

    # Enum checks (non-wildcard)
    for path, allowed in ENUMS.items():
        if "*" in path:
            continue
        v = get_path(obj, path)
        _check_enum(v, allowed, ".".join(map(str, path)), errors)

    # Enum checks (wildcard) for specific_events
    if isinstance(sev, list):
        for i, e in enumerate(sev):
            if not isinstance(e, dict):
                continue
            _check_enum(e.get("event_type"), ENUMS[("specific_events", "*", "event_type")], f"specific_events[{i}].event_type", errors)
            _check_enum(e.get("severity"), ENUMS[("specific_events", "*", "severity")], f"specific_events[{i}].severity", errors)

    schema_ok = not any(
        e.startswith("missing_top_level:")
        or e.startswith("not_object:")
        or e.endswith("_not_list")
        or e.startswith("missing_enum:")
        or e.startswith("bad_enum:")
        or e.startswith("missing_value:")
        for e in errors
    )

    if any(e.startswith("bad_conf:") or e.startswith("bad_visible:") for e in errors):
        rule_ok = False

    return ValidationReport(parse_ok=True, schema_ok=schema_ok, rule_ok=rule_ok, errors=errors)


# ----------------------------
# Teacher-run consensus weighting
# ----------------------------

def _sym_global_slot_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa, _ = score_global_and_sections(a, b)
    sb, _ = score_global_and_sections(b, a)
    return 0.5 * (sa.global_macro_acc + sb.global_macro_acc)


def _sym_events_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return 0.5 * (score_specific_events(a, b).macro_acc + score_specific_events(b, a).macro_acc)


def _sym_risk_acc(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return 0.5 * (score_risk_assessment(a, b).macro_acc + score_risk_assessment(b, a).macro_acc)


def _consensus_pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    gpair = _sym_global_slot_acc(a, b)
    epair = _sym_events_acc(a, b)
    rpair = _sym_risk_acc(a, b)
    vals = [v for v in (gpair, epair, rpair) if not np.isnan(v)]
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
        legacy_results = BASE_DIR / "results_people"
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
            if student is None or not isinstance(student, dict):
                log(f"Skipping student (parse failed or not object): {student_path}")
                continue
            if "error" in student:
                log(f"Skipping student (error present): {student_path}")
                continue
            vrep = validate_struct(student)

            global_slot = None
            scene_slot = None
            crowd_slot = None
            violence_slot = None
            events_macro = None
            risk_macro = None

            events_presence_f1 = None
            events_count_acc = None
            events_persons_count_acc = None
            events_severity_acc = None

            risk_overall_threat_level_acc = None
            risk_main_threat_factors_f1 = None
            risk_immediate_intervention_required_acc = None

            if vrep.parse_ok:
                sec, _hits = score_global_and_sections(student, teacher)
                global_slot = sec.global_macro_acc
                scene_slot = sec.scene_context_macro_acc
                crowd_slot = sec.crowd_analysis_macro_acc
                violence_slot = sec.violence_and_aggression_macro_acc
                events_macro = sec.specific_events_macro_acc
                risk_macro = sec.risk_assessment_macro_acc

                evs = score_specific_events(student, teacher)
                events_presence_f1 = evs.presence_f1
                events_count_acc = evs.count_acc
                events_persons_count_acc = evs.persons_count_acc
                events_severity_acc = evs.severity_acc

                rsk = score_risk_assessment(student, teacher)
                risk_overall_threat_level_acc = rsk.overall_threat_level_acc
                risk_main_threat_factors_f1 = rsk.main_threat_factors_f1
                risk_immediate_intervention_required_acc = rsk.immediate_intervention_required_acc

            rows.append(
                {
                    "video": video_id,
                    "model": model_name,
                    "parse_ok": vrep.parse_ok,
                    "has_error": False,
                    "schema_ok": vrep.schema_ok,
                    "rule_ok": vrep.rule_ok,
                    "global_slot_macro_acc": global_slot,
                    "scene_context_slot_macro_acc": scene_slot,
                    "crowd_analysis_slot_macro_acc": crowd_slot,
                    "violence_and_aggression_slot_macro_acc": violence_slot,
                    "specific_events_macro_acc": events_macro,
                    "events_presence_f1": events_presence_f1,
                    "events_count_acc": events_count_acc,
                    "events_persons_count_acc": events_persons_count_acc,
                    "events_severity_acc": events_severity_acc,
                    "risk_assessment_macro_acc": risk_macro,
                    "overall_threat_level_acc": risk_overall_threat_level_acc,
                    "main_threat_factors_f1": risk_main_threat_factors_f1,
                    "immediate_intervention_required_acc": risk_immediate_intervention_required_acc,
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
            events_mean = mean_or_nan("specific_events_macro_acc")
            risk_mean = mean_or_nan("risk_assessment_macro_acc")

            effective_global = float(global_mean * schema_rate) if not np.isnan(global_mean) else float("nan")
            effective_risk = float(risk_mean * schema_rate) if not np.isnan(risk_mean) else float("nan")

            agg_rows.append(
                {
                    "model": model_name,
                    "parse_rate": parse_rate,
                    "schema_rate": schema_rate,
                    "rule_rate": rule_rate,
                    "global_slot_macro_acc_mean": global_mean,
                    "effective_global_slot_macro_acc_mean": effective_global,
                    "scene_context_slot_macro_acc_mean": mean_or_nan("scene_context_slot_macro_acc"),
                    "crowd_analysis_slot_macro_acc_mean": mean_or_nan("crowd_analysis_slot_macro_acc"),
                    "violence_and_aggression_slot_macro_acc_mean": mean_or_nan("violence_and_aggression_slot_macro_acc"),
                    "specific_events_macro_acc_mean": events_mean,
                    "events_presence_f1_mean": mean_or_nan("events_presence_f1"),
                    "events_count_acc_mean": mean_or_nan("events_count_acc"),
                    "events_persons_count_acc_mean": mean_or_nan("events_persons_count_acc"),
                    "events_severity_acc_mean": mean_or_nan("events_severity_acc"),
                    "risk_assessment_macro_acc_mean": risk_mean,
                    "effective_risk_assessment_macro_acc_mean": effective_risk,
                    "overall_threat_level_acc_mean": mean_or_nan("overall_threat_level_acc"),
                    "main_threat_factors_f1_mean": mean_or_nan("main_threat_factors_f1"),
                    "immediate_intervention_required_acc_mean": mean_or_nan("immediate_intervention_required_acc"),
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
